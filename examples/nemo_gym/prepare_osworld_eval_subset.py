#!/usr/bin/env python3
"""Build a stable domain-balanced OSWorld evaluation subset."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

DOMAIN_ALIASES = {
    "calc": {"calc", "libreoffice_calc"},
    "chrome": {"chrome"},
    "vlc": {"vlc"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=["calc", "chrome", "vlc"])
    parser.add_argument("--per-domain", type=int, default=4)
    args = parser.parse_args()
    if args.per_domain < 1:
        parser.error("--per-domain must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    requested = list(dict.fromkeys(args.domains))
    selected: dict[str, list[dict]] = defaultdict(list)

    with args.input.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("verifier_metadata") or {}
            if not metadata.get("id"):
                raise ValueError(f"{args.input}:{line_number} has no OSWorld task id")
            task_apps = {
                metadata.get("snapshot"),
                *(metadata.get("related_apps") or []),
            }
            for domain in requested:
                aliases = DOMAIN_ALIASES.get(domain, {domain})
                if (
                    len(selected[domain]) < args.per_domain
                    and task_apps.intersection(aliases)
                ):
                    selected[domain].append(row)
                    break

    missing = [
        domain
        for domain in requested
        if len(selected[domain]) < args.per_domain
    ]
    if missing:
        counts = {domain: len(selected[domain]) for domain in requested}
        raise ValueError(f"Not enough rows for domains {missing}; selected counts={counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for domain in requested:
            for row in selected[domain]:
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    task_ids = {
        domain: [row["verifier_metadata"]["id"] for row in selected[domain]]
        for domain in requested
    }
    print(
        f"Wrote {sum(map(len, selected.values()))} rows to {args.output}: "
        f"{json.dumps(task_ids, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
