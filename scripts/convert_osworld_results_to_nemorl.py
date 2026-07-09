#!/usr/bin/env python3
"""Convert OSWorld trajectories into NeMo-RL VLM JSONL samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_load_score(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _normalize_action(action: Any) -> str:
    if isinstance(action, str):
        return action.strip()
    if action is None:
        return ""
    return json.dumps(action, ensure_ascii=False)


def _build_prompt_text(
    instruction: str,
    domain: str,
    example_id: str,
    step_num: int,
) -> str:
    return (
        "You are an OSWorld desktop agent.\n"
        "Given the task instruction and current screenshot, output exactly one "
        "next pyautogui action.\n\n"
        f"Task instruction:\n{instruction}\n\n"
        f"Domain: {domain}\n"
        f"Example ID: {example_id}\n"
        f"Step: {step_num}\n\n"
        "Return only executable action text."
    )


def _collect_episode_samples(
    episode_dir: Path,
    examples_root: Path | None,
    min_episode_score: float,
    include_failed: bool,
    require_image: bool,
) -> tuple[str, list[dict[str, Any]]]:
    domain = episode_dir.parent.name
    example_id = episode_dir.name
    task_id = f"{domain}/{example_id}"

    score_path = episode_dir / "result.txt"
    if not score_path.exists():
        return task_id, []
    score = _safe_load_score(score_path)
    if score is None:
        return task_id, []
    if (not include_failed) and score < min_episode_score:
        return task_id, []

    instruction = ""
    if examples_root is not None:
        example_json = examples_root / domain / f"{example_id}.json"
        if example_json.exists():
            example = _safe_load_json(example_json)
            if isinstance(example, dict):
                instruction = str(example.get("instruction", ""))

    traj_path = episode_dir / "traj.jsonl"
    if not traj_path.exists():
        return task_id, []

    samples: list[dict[str, Any]] = []
    for idx, raw in enumerate(traj_path.read_text(encoding="utf-8").splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue

        action_text = _normalize_action(row.get("action", ""))
        if not action_text or action_text.upper() in {"FAIL", "DONE"}:
            continue

        step_num = int(row.get("step_num", idx + 1))
        screenshot_file = row.get("screenshot_file")
        screenshot_path: Path | None = None
        if isinstance(screenshot_file, str) and screenshot_file:
            candidate = (episode_dir / screenshot_file).resolve()
            if candidate.exists():
                screenshot_path = candidate

        if require_image and screenshot_path is None:
            continue

        prompt_text = _build_prompt_text(
            instruction=instruction,
            domain=domain,
            example_id=example_id,
            step_num=step_num,
        )

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if screenshot_path is not None:
            user_content.append({"type": "image", "image": str(screenshot_path)})

        samples.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": action_text}],
                    },
                ],
                "meta": {
                    "task_id": task_id,
                    "domain": domain,
                    "example_id": example_id,
                    "step_num": step_num,
                    "episode_score": score,
                    "screenshot_file": str(screenshot_path) if screenshot_path else None,
                    "raw_response": row.get("response"),
                    "raw_reward": row.get("reward"),
                    "raw_done": row.get("done"),
                },
            }
        )

    return task_id, samples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=None,
        help="Path like OSWorld/evaluation_examples/examples (optional).",
    )
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-episode-score", type=float, default=1.0)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include episodes whose score < min-episode-score.",
    )
    parser.add_argument(
        "--allow-missing-image",
        action="store_true",
        help="If set, keep samples even when screenshot file is missing.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results_root.exists():
        raise FileNotFoundError(f"results root not found: {args.results_root}")

    require_image = not args.allow_missing_image

    samples_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result_txt in args.results_root.rglob("result.txt"):
        episode_dir = result_txt.parent
        task_id, samples = _collect_episode_samples(
            episode_dir=episode_dir,
            examples_root=args.examples_root,
            min_episode_score=args.min_episode_score,
            include_failed=args.include_failed,
            require_image=require_image,
        )
        if samples:
            samples_by_task[task_id].extend(samples)

    task_ids = sorted(samples_by_task.keys())
    if not task_ids:
        raise RuntimeError("No usable samples found in results root.")

    rng = random.Random(args.seed)
    rng.shuffle(task_ids)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    val_tasks: set[str] = set()
    if len(task_ids) == 1:
        # Single-task edge case: split by samples instead of task.
        only_rows = samples_by_task[task_ids[0]]
        if len(only_rows) <= 1:
            # Keep pipeline usable for smoke tests / tiny datasets.
            train_rows = list(only_rows)
            val_rows = list(only_rows)
        else:
            split_idx = max(1, int(len(only_rows) * (1.0 - args.val_ratio)))
            if split_idx >= len(only_rows):
                split_idx = len(only_rows) - 1
            train_rows = only_rows[:split_idx]
            val_rows = only_rows[split_idx:]
    else:
        val_count = max(1, int(len(task_ids) * args.val_ratio))
        if val_count >= len(task_ids):
            val_count = max(1, len(task_ids) - 1)

        val_tasks = set(task_ids[:val_count])
        for task_id, rows in samples_by_task.items():
            if task_id in val_tasks:
                val_rows.extend(rows)
            else:
                train_rows.extend(rows)

    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: args.max_samples]

    if not train_rows:
        raise RuntimeError("Train split is empty after filtering/splitting.")
    if not val_rows:
        raise RuntimeError("Validation split is empty after filtering/splitting.")

    _write_jsonl(args.output_train, train_rows)
    _write_jsonl(args.output_val, val_rows)

    print("=== Conversion done ===")
    print(f"Results root:        {args.results_root}")
    print(f"Examples root:       {args.examples_root}")
    print(f"Train tasks:         {len(task_ids) - len(val_tasks)}")
    print(f"Validation tasks:    {len(val_tasks)}")
    print(f"Train samples:       {len(train_rows)}")
    print(f"Validation samples:  {len(val_rows)}")
    print(f"Train output:        {args.output_train}")
    print(f"Validation output:   {args.output_val}")


if __name__ == "__main__":
    main()
