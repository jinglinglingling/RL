#!/usr/bin/env python3
"""Prepare OSWorld JSONL rows for NeMo-Gym online GRPO."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are an autonomous desktop agent for OSWorld tasks.
You must interact with the environment by calling tools only.

Rules:
1) Call `osworld_get_observation` when you need a fresh state view.
2) Call `osworld_execute_action` with exactly one action at a time.
   - Use pyautogui-style python commands for normal interactions.
   - Use WAIT when you need to wait.
   - Use DONE when the task is complete.
   - Use FAIL if the task is impossible.
3) Call `osworld_finish` after you are done.
4) Avoid free-form assistant answers; prefer tool calls.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "osworld_get_observation",
        "description": "Fetch the latest desktop observation (screenshot path, instruction, a11y tree, terminal).",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "osworld_execute_action",
        "description": "Execute one desktop action. The action can be a pyautogui command string or WAIT/FAIL/DONE.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Single action to execute, e.g. pyautogui command or WAIT/FAIL/DONE.",
                },
                "pause_seconds": {
                    "type": "number",
                    "description": "Optional wait time after action execution.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "osworld_finish",
        "description": "Mark rollout finished and ask environment to evaluate current state.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional short reason for finishing.",
                }
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_examples_root = (
        project_root / "third_party" / "OSWorld" / "evaluation_examples"
    )
    parser.add_argument(
        "--test-all-meta-path",
        type=Path,
        default=project_root / "configs" / "osworld_test_all_smoke.json",
        help="Path to domain->example_id mapping JSON.",
    )
    parser.add_argument(
        "--test-config-base-dir",
        type=Path,
        default=default_examples_root,
        help="OSWorld evaluation_examples directory.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=project_root / "data" / "nemogym" / "osworld_online_train.jsonl",
        help="Output train JSONL path.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=project_root / "data" / "nemogym" / "osworld_online_val.jsonl",
        help="Output validation JSONL path.",
    )
    parser.add_argument(
        "--agent-ref-name",
        type=str,
        default="osworld_vlm_agent",
        help="NeMo-Gym agent_ref.name used by rows.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Per-row max step budget passed to resources server.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for train/val split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total rows; 0 means no cap.",
    )
    return parser.parse_args()


def _load_task_ids(meta_path: Path) -> list[tuple[str, str]]:
    test_all_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    output: list[tuple[str, str]] = []
    for domain, example_ids in test_all_meta.items():
        if not isinstance(example_ids, list):
            continue
        for example_id in example_ids:
            output.append((str(domain), str(example_id)))
    return output


def _build_row(
    *,
    domain: str,
    example_id: str,
    task_path: Path,
    task_config: dict[str, Any],
    agent_ref_name: str,
    max_steps: int,
) -> dict[str, Any]:
    instruction = str(task_config.get("instruction", "")).strip()
    user_prompt = (
        f"{instruction}\n\n"
        "Start by calling `osworld_get_observation`, then interact via tools until complete."
    )
    return {
        "domain": domain,
        "example_id": example_id,
        "task_config_path": str(task_path),
        "instruction": instruction,
        "max_steps": max_steps,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "tools": TOOLS,
            "parallel_tool_calls": False,
        },
        "agent_ref": {"type": "responses_api_agents", "name": agent_ref_name},
    }


def _split_rows(
    rows: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []

    rows = rows.copy()
    random.Random(seed).shuffle(rows)

    if len(rows) == 1:
        # Keep both splits non-empty for tiny smoke sets.
        return rows, rows.copy()

    val_size = int(round(len(rows) * val_ratio))
    val_size = max(1, val_size)
    val_size = min(val_size, len(rows) - 1)

    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    return train_rows, val_rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    args = parse_args()

    task_pairs = _load_task_ids(args.test_all_meta_path)
    rows: list[dict[str, Any]] = []
    for domain, example_id in task_pairs:
        task_path = (
            args.test_config_base_dir / "examples" / domain / f"{example_id}.json"
        ).resolve()
        if not task_path.exists():
            print(f"[WARN] Missing task config: {task_path}")
            continue
        task_config = json.loads(task_path.read_text(encoding="utf-8"))
        rows.append(
            _build_row(
                domain=domain,
                example_id=example_id,
                task_path=task_path,
                task_config=task_config,
                agent_ref_name=args.agent_ref_name,
                max_steps=args.max_steps,
            )
        )

    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    train_rows, val_rows = _split_rows(rows, args.val_ratio, args.seed)
    _write_jsonl(train_rows, args.train_output)
    _write_jsonl(val_rows, args.val_output)

    print(
        "[OK] Prepared OSWorld Nemo-Gym dataset "
        f"(total={len(rows)}, train={len(train_rows)}, val={len(val_rows)})"
    )
    print(f"  train: {args.train_output}")
    print(f"  val:   {args.val_output}")


if __name__ == "__main__":
    main()
