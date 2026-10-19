#!/usr/bin/env python3
"""Prepare the exact four contiguous OSWorld shards used by the parity eval."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 361:
        raise ValueError(f"Expected 361 tasks, found {len(rows)} in {path}")
    task_ids = [row["verifier_metadata"]["id"] for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Source contains duplicate verifier task IDs")
    return rows


def prepare(row: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(row)
    task_id = prepared["verifier_metadata"]["id"]
    prepared["responses_create_params"].update(
        {"max_output_tokens": 4096, "temperature": 1.0, "top_p": 1.0}
    )
    prepared["agent_ref"] = {
        "type": "responses_api_agents",
        "name": "nemotron_osworld",
    }
    prepared["context_compaction_contract_version"] = 2
    prepared["context_compaction_group_id"] = f"osworld:{task_id}:repeat:0"
    prepared["context_compaction_task_id"] = task_id
    prepared["context_compaction_rollout_index"] = 0
    prepared["context_compaction_attempt_index"] = 0
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [prepare(row) for row in read_rows(args.input)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sizes = (91, 90, 90, 90)
    offset = 0
    shards = []
    for shard_index, size in enumerate(sizes, start=1):
        shard_rows = rows[offset : offset + size]
        output = args.output_dir / f"validation_nogdrive_361_q{shard_index:02d}.jsonl"
        with output.open("w", encoding="utf-8") as destination:
            for row in shard_rows:
                destination.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        shards.append(
            {
                "name": output.name,
                "rows": len(shard_rows),
                "sha256": sha256(output),
                "first_task_id": shard_rows[0]["verifier_metadata"]["id"],
                "last_task_id": shard_rows[-1]["verifier_metadata"]["id"],
            }
        )
        offset += size

    manifest = {
        "protocol": "osworld-v1-parity-361x4",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "tasks": len(rows),
        "num_generations_per_task": 4,
        "shards": shards,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
