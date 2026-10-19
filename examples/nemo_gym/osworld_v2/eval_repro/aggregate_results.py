#!/usr/bin/env python3
"""Aggregate four OSWorld parity-eval shards into Pass@1 and Pass@4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def discover(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.glob("**/val_data_step0.jsonl"))
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(file.resolve() for file in files))
    if len(unique) != 4:
        raise ValueError(f"Expected four val_data_step0.jsonl files, found {len(unique)}")
    return unique


def reward(row: dict[str, Any]) -> float:
    rewards = row.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 1:
        raise ValueError(f"Unexpected rewards field: {rewards!r}")
    return float(rewards[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    files = discover(args.inputs)
    task_groups: list[list[float]] = []
    shard_rows: list[int] = []
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) % 4:
            raise ValueError(f"{path} has {len(rows)} rows, not divisible by four")
        rewards = [reward(row) for row in rows]
        task_groups.extend(
            rewards[offset : offset + 4] for offset in range(0, len(rewards), 4)
        )
        shard_rows.append(len(rows))

    samples = [value for group in task_groups for value in group]
    if len(task_groups) != 361 or len(samples) != 1444:
        raise ValueError(
            f"Expected 361 tasks/1444 samples, found {len(task_groups)}/{len(samples)}"
        )
    successes = sum(value > 0 for value in samples)
    solved = sum(any(value > 0 for value in group) for group in task_groups)
    result = {
        "files": [str(path) for path in files],
        "shard_rows": shard_rows,
        "tasks": len(task_groups),
        "samples": len(samples),
        "successes": successes,
        "solved_tasks": solved,
        "pass_at_1": successes / len(samples),
        "pass_at_4": solved / len(task_groups),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
