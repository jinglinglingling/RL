#!/usr/bin/env python3
"""Create a deterministic domain-stratified OSWorld GRPO train/eval split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--agent-name", default="nemotron_osworld")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, agent_name: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            metadata = row.get("verifier_metadata", {})
            if not metadata.get("id") or not metadata.get("snapshot"):
                raise ValueError(
                    f"{path}:{line_number}: missing verifier id or snapshot"
                )
            row["agent_ref"] = {
                "type": "responses_api_agents",
                "name": agent_name,
            }
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")

    rows = load_rows(args.input, args.agent_name)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["verifier_metadata"]["snapshot"]].append(row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    domain_counts: dict[str, dict[str, int]] = {}
    target_train_count = round(len(rows) * args.train_fraction)
    quotas = {
        domain: len(domain_rows) * args.train_fraction
        for domain, domain_rows in by_domain.items()
    }
    train_counts = {domain: math.floor(quota) for domain, quota in quotas.items()}
    remaining = target_train_count - sum(train_counts.values())
    largest_remainders = sorted(
        quotas, key=lambda domain: (-(quotas[domain] % 1), domain)
    )
    for domain in largest_remainders[:remaining]:
        train_counts[domain] += 1

    rng = random.Random(args.seed)
    for domain in sorted(by_domain):
        domain_rows = sorted(
            by_domain[domain], key=lambda row: row["verifier_metadata"]["id"]
        )
        rng.shuffle(domain_rows)
        train_count = train_counts[domain]
        train_rows.extend(domain_rows[:train_count])
        eval_rows.extend(domain_rows[train_count:])
        domain_counts[domain] = {
            "total": len(domain_rows),
            "train": train_count,
            "eval": len(domain_rows) - train_count,
        }

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "heldout.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    manifest = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": args.agent_name,
        },
        "counts": {
            "total": len(rows),
            "train": len(train_rows),
            "eval": len(eval_rows),
        },
        "domains": domain_counts,
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256(train_path)},
            "eval": {"path": str(eval_path.resolve()), "sha256": sha256(eval_path)},
        },
    }
    manifest_path = args.output_dir / "split-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
