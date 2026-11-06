#!/usr/bin/env python3
"""Atomically remove evaluator-incompatible task IDs from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--exclude", action="append", required=True)
    args = parser.parse_args()

    excluded = set(args.exclude)
    removed: Counter[str] = Counter()
    kept = 0
    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")

    with args.source.open() as source, temporary.open("w") as destination:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            task_id = record.get("verifier_metadata", {}).get("id")
            if task_id in excluded:
                removed[task_id] += 1
                continue
            destination.write(line)
            kept += 1

    missing = excluded - removed.keys()
    if missing:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Excluded task IDs absent from manifest: {sorted(missing)}")

    os.replace(temporary, args.destination)
    print(json.dumps({"kept": kept, "removed": removed}, sort_keys=True))


if __name__ == "__main__":
    main()
