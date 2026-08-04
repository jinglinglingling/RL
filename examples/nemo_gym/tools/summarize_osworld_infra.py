#!/usr/bin/env python3
"""Summarize OSWorld infrastructure failures from NeMo-RL driver logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


COUNT_PATTERNS = {
    "sandbox_create_retries": r"Retrying OpenSandbox sandbox create",
    "pod_ready_timeouts": r"POD_READY_TIMEOUT",
    "hf_429_mentions": r"(?:error: 429|429 Client Error|HTTP error\. Will retry)",
    "setup_failures": r"eval_task setup failed",
    "evaluation_failures": r"eval_task evaluate failed",
    "fresh_sandbox_retries": r"seed_session attempt [12]/3 failed",
    "masked_rollouts": r"(?:mask_sample|emitting marked zero-reward row)",
    "llm_finish_retries": r"LLM did not finish properly, retrying",
    "missing_responses": r"No response found in the response",
    "disk_quota_errors": r"Disk quota exceeded",
}

TASK_FAILURE_RE = re.compile(
    r"eval_task (?P<phase>setup|evaluate) failed "
    r"\(task=(?P<task_id>[0-9a-f-]+),"
)
STEP_RE = re.compile(r"=+ Step (?P<step>\d+)/(?P<max_step>\d+) =+")
CHECKPOINT_RE = re.compile(r"Saving checkpoint for step (?P<step>\d+)")
METRIC_RE = re.compile(
    r"• (?P<name>Loss|Generation KL Error|Avg Total Reward): (?P<value>-?[0-9.]+)"
)


def parse_key_value(value: str, *, value_name: str) -> tuple[str, str]:
    try:
        key, parsed_value = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL={value_name}, got {value!r}"
        ) from error
    if not key or not parsed_value:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL={value_name}, got {value!r}"
        )
    return key, parsed_value


def summarize(path: Path, generations: int | None) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    step_occurrences = [int(match.group("step")) for match in STEP_RE.finditer(text)]
    checkpoints = [int(match.group("step")) for match in CHECKPOINT_RE.finditer(text)]
    task_failures: dict[str, set[str]] = {"setup": set(), "evaluate": set()}
    for match in TASK_FAILURE_RE.finditer(text):
        task_failures[match.group("phase")].add(match.group("task_id"))

    metrics: dict[int, dict[str, float]] = {}
    current_checkpoint: int | None = None
    for line in text.splitlines():
        checkpoint_match = CHECKPOINT_RE.search(line)
        if checkpoint_match:
            current_checkpoint = int(checkpoint_match.group("step"))
            metrics.setdefault(current_checkpoint, {})
            continue
        metric_match = METRIC_RE.search(line)
        if metric_match and current_checkpoint is not None:
            metrics[current_checkpoint][metric_match.group("name")] = float(
                metric_match.group("value")
            )

    result: dict[str, object] = {
        "log": str(path.resolve()),
        "counts": {
            name: len(re.findall(pattern, text))
            for name, pattern in COUNT_PATTERNS.items()
        },
        "dynamic_batch_markers_by_step": dict(
            sorted(Counter(step_occurrences).items())
        ),
        "completed_checkpoint_steps": checkpoints,
        "latest_completed_checkpoint_step": max(checkpoints, default=None),
        "failed_task_ids": {
            phase: sorted(task_ids) for phase, task_ids in task_failures.items()
        },
        "checkpoint_metrics": metrics,
    }
    if generations is not None:
        result["generations_per_batch"] = generations
        result["estimated_base_rollouts"] = len(step_occurrences) * generations
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=LOG",
        help="Label and ray-driver.log path; may be passed more than once.",
    )
    parser.add_argument(
        "--generations",
        action="append",
        default=[],
        metavar="LABEL=N",
        help="Optional rollout-group size for an estimated base-rollout count.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = dict(
        parse_key_value(value, value_name="LOG") for value in args.run
    )
    generations = {
        key: int(value)
        for key, value in (
            parse_key_value(item, value_name="N") for item in args.generations
        )
    }
    unknown_labels = generations.keys() - runs.keys()
    if unknown_labels:
        parser.error(
            "--generations labels must also be supplied via --run: "
            + ", ".join(sorted(unknown_labels))
        )
    if any(value < 1 for value in generations.values()):
        parser.error("--generations values must be positive")

    report = {
        "runs": {
            label: summarize(Path(path), generations.get(label))
            for label, path in runs.items()
        },
        "notes": [
            "HF 429 mentions are diagnostic-text counts, not HTTP request counts; stderr tails can repeat curl output.",
            "Estimated base rollouts multiply step-header occurrences by group size and include dynamic resampling.",
            "Current concurrent runs are a pressure test, not an uncontaminated R4/R8 comparison.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
