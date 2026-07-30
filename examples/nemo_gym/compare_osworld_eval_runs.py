#!/usr/bin/env python3
"""Compare paired OSWorld evaluation JSONL results by task id."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = (row.get("verifier_metadata") or {}).get("id")
            if not task_id:
                raise ValueError(f"{path}:{line_number} has no verifier_metadata.id")
            if task_id in results:
                raise ValueError(f"{path}:{line_number} repeats task id {task_id}")
            results[task_id] = row
    return results


def reward(row: dict[str, Any]) -> float:
    return float(row.get("reward", 0.0))


def main() -> None:
    args = parse_args()
    local = load_results(args.local)
    api = load_results(args.api)
    common_ids = sorted(local.keys() & api.keys())

    comparisons = []
    for task_id in common_ids:
        local_reward = reward(local[task_id])
        api_reward = reward(api[task_id])
        if local_reward == api_reward:
            continue
        comparisons.append(
            {
                "task_id": task_id,
                "local_reward": local_reward,
                "api_reward": api_reward,
                "local_verify_error": local[task_id].get("verify_error"),
                "api_verify_error": api[task_id].get("verify_error"),
            }
        )

    summary = {
        "local_rows": len(local),
        "api_rows": len(api),
        "paired_rows": len(common_ids),
        "local_reward_sum": sum(reward(local[task_id]) for task_id in common_ids),
        "api_reward_sum": sum(reward(api[task_id]) for task_id in common_ids),
        "local_only_task_ids": sorted(local.keys() - api.keys()),
        "api_only_task_ids": sorted(api.keys() - local.keys()),
        "divergent_tasks": comparisons,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote paired comparison to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
