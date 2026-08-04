#!/usr/bin/env python3
"""Convert OSWorld task JSONL into NeMo-Gym rollout rows for GRPO."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Include only this verifier task ID; may be passed more than once.",
    )
    parser.add_argument("--agent-name", default="nemotron_osworld")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_repeats < 1:
        raise ValueError("--num-repeats must be at least 1")

    rows = []
    with args.input.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "responses_create_params" not in row or "verifier_metadata" not in row:
                raise ValueError(
                    f"{args.input}:{line_number} is not an OSWorld Gym task row"
                )
            if (
                args.task_id
                and row["verifier_metadata"].get("id") not in args.task_id
            ):
                continue
            row["agent_ref"] = {
                "type": "responses_api_agents",
                "name": args.agent_name,
            }
            rows.append(row)
            if args.limit is not None and len(rows) >= args.limit:
                break

    if args.task_id:
        found_task_ids = {row["verifier_metadata"]["id"] for row in rows}
        missing_task_ids = set(args.task_id) - found_task_ids
        if missing_task_ids:
            raise ValueError(
                "Requested task IDs were not found: "
                + ", ".join(sorted(missing_task_ids))
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for _ in range(args.num_repeats):
            for row in rows:
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(
        f"Wrote {len(rows) * args.num_repeats} rows to {args.output} "
        f"({len(rows)} unique tasks x {args.num_repeats} repeats)."
    )


if __name__ == "__main__":
    main()
