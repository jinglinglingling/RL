#!/usr/bin/env python3
"""Build a reproducible OSWorld tiny-overfit set from prior GRPO rollouts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_ID_RE = re.compile(r'"osworld_task_id"\s*:\s*\["([^"]+)"\]')
FILTERED_REWARD_RE = re.compile(
    r'"filtered_rewards"\s*:\s*\[([-+0-9.eE]+)\]'
)
LOSS_MASK_RE = re.compile(r'"sample_loss_mask"\s*:\s*\[([-+0-9.eE]+)\]')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=32)
    parser.add_argument("--min-reward", type=float, default=0.125)
    parser.add_argument("--max-reward", type=float, default=0.875)
    parser.add_argument("--train-repeats", type=int, default=8)
    parser.add_argument("--validation-repeats", type=int, default=4)
    return parser.parse_args()


def load_training_rows(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open() as dataset:
        for line in dataset:
            row = json.loads(line)
            task_id = row["verifier_metadata"]["id"]
            if task_id in rows:
                raise ValueError(f"duplicate task ID in training data: {task_id}")
            rows[task_id] = row
            order.append(task_id)
    return rows, order


def collect_group_rewards(log_roots: list[Path]) -> dict[str, list[float]]:
    task_rewards: dict[str, list[float]] = defaultdict(list)
    for log_root in log_roots:
        for path in sorted(log_root.glob("exp_*/train_data_step*.jsonl")):
            weighted_rewards: dict[str, float] = defaultdict(float)
            weights: dict[str, float] = defaultdict(float)
            with path.open(errors="replace") as log:
                for line in log:
                    task_match = TASK_ID_RE.search(line)
                    # Under dynamic sampling, `rewards` is the pre-filter batch
                    # retained only for diagnostics. `filtered_rewards` is the
                    # trajectory reward aligned with this actual training row.
                    reward_match = FILTERED_REWARD_RE.search(line)
                    mask_match = LOSS_MASK_RE.search(line)
                    if not (task_match and reward_match and mask_match):
                        continue
                    weight = float(mask_match.group(1))
                    if weight <= 0:
                        continue
                    task_id = task_match.group(1)
                    weighted_rewards[task_id] += float(reward_match.group(1)) * weight
                    weights[task_id] += weight
            for task_id, weight in weights.items():
                task_rewards[task_id].append(weighted_rewards[task_id] / weight)
    return task_rewards


def evaluator_function(row: dict[str, Any]) -> Any:
    evaluator = row["verifier_metadata"].get("evaluator", {})
    return evaluator.get("func") if isinstance(evaluator, dict) else None


def select_tasks(
    rows: dict[str, dict[str, Any]],
    order: list[str],
    task_rewards: dict[str, list[float]],
    *,
    num_tasks: int,
    min_reward: float,
    max_reward: float,
) -> list[dict[str, Any]]:
    order_index = {task_id: index for index, task_id in enumerate(order)}
    candidates: list[dict[str, Any]] = []
    for task_id, rewards in task_rewards.items():
        row = rows.get(task_id)
        if row is None:
            continue
        metadata = row["verifier_metadata"]
        if metadata.get("proxy", False):
            continue
        if metadata.get("possibility_of_env_change") != "low":
            continue
        if evaluator_function(row) == "infeasible":
            continue
        if not rewards or any(not (0.0 < reward < 1.0) for reward in rewards):
            continue
        mean_reward = sum(rewards) / len(rewards)
        reward_tolerance = 1.0e-6
        if not (
            min_reward - reward_tolerance
            <= mean_reward
            <= max_reward + reward_tolerance
        ):
            continue
        candidates.append(
            {
                "task_id": task_id,
                "snapshot": metadata["snapshot"],
                "instruction": metadata["instruction"],
                "observed_groups": len(rewards),
                "mean_group_reward": mean_reward,
                "min_group_reward": min(rewards),
                "max_group_reward": max(rewards),
                "proxy": False,
                "possibility_of_env_change": "low",
                "evaluator_function": evaluator_function(row),
                "source_order": order_index[task_id],
            }
        )

    candidates.sort(
        key=lambda task: (
            -task["observed_groups"],
            abs(task["mean_group_reward"] - 0.5),
            task["source_order"],
        )
    )
    if len(candidates) < num_tasks:
        raise ValueError(
            f"only {len(candidates)} tasks satisfy the tiny-overfit criteria; "
            f"requested {num_tasks}"
        )
    return candidates[:num_tasks]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    rows_by_id, source_order = load_training_rows(args.train_data)
    task_rewards = collect_group_rewards(args.log_root)
    selected = select_tasks(
        rows_by_id,
        source_order,
        task_rewards,
        num_tasks=args.num_tasks,
        min_reward=args.min_reward,
        max_reward=args.max_reward,
    )
    selected_rows = [rows_by_id[task["task_id"]] for task in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "tasks.jsonl", selected_rows)
    write_jsonl(
        args.output_dir / f"train-{args.train_repeats}x.jsonl",
        selected_rows * args.train_repeats,
    )
    write_jsonl(
        args.output_dir / f"validation-{args.validation_repeats}x.jsonl",
        selected_rows * args.validation_repeats,
    )

    manifest = {
        "purpose": "tiny-overfit pipeline validation; not benchmark evaluation",
        "source_training_data": str(args.train_data.resolve()),
        "source_log_roots": [str(path.resolve()) for path in args.log_root],
        "criteria": {
            "num_tasks": args.num_tasks,
            "all_observed_groups_have_reward_variance": True,
            "mean_group_reward_range": [args.min_reward, args.max_reward],
            "proxy": False,
            "possibility_of_env_change": "low",
            "exclude_evaluator_function": "infeasible",
        },
        "train_repeats": args.train_repeats,
        "validation_repeats": args.validation_repeats,
        "snapshot_counts": dict(
            sorted(Counter(task["snapshot"] for task in selected).items())
        ),
        "tasks": selected,
    }
    with (args.output_dir / "selection-manifest.json").open("w") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(
        f"selected {len(selected)} tasks from {len(task_rewards)} historically "
        f"observed tasks"
    )
    print(f"snapshots: {manifest['snapshot_counts']}")
    print(f"output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
