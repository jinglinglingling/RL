#!/usr/bin/env python3
"""Invoke one constrained Cursor SDK repair run for the OSWorld watchdog."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from cursor_sdk import Agent, AgentOptions, LocalAgentOptions


def atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    file_descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    path.chmod(0o600)


def read_api_key() -> str:
    key_file_value = os.environ.get("CURSOR_API_KEY_FILE")
    if not key_file_value:
        raise RuntimeError("CURSOR_API_KEY_FILE is required")
    key_file = Path(key_file_value).expanduser()
    if not key_file.is_file():
        raise RuntimeError(f"Cursor API key file does not exist: {key_file}")
    if key_file.stat().st_mode & 0o077:
        raise RuntimeError(f"Cursor API key file must have mode 0600: {key_file}")
    api_key = key_file.read_text().strip()
    if not api_key:
        raise RuntimeError("Cursor API key file is empty")
    return api_key


def resolve_allowed_paths(repo: Path, values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repo / path
        path = path.resolve()
        try:
            path.relative_to(repo)
        except ValueError as error:
            raise ValueError(f"Allowed path escapes repository: {path}") from error
        paths.append(path)
    if not paths:
        raise ValueError("At least one allowed path is required")
    return paths


def build_prompt(
    *,
    repo: Path,
    experiment: str,
    failure_fingerprint: str,
    log_path: Path | None,
    allowed_paths: list[Path],
) -> str:
    relative_allowed = [str(path.relative_to(repo)) for path in allowed_paths]
    log_instruction = (
        f"Inspect the failure log at {log_path}."
        if log_path is not None
        else "No driver log was found; inspect the experiment result and Slurm logs."
    )
    return f"""You are an unattended repair agent for one OSWorld GRPO experiment.

Repository: {repo}
Experiment: {experiment}
Failure fingerprint: {failure_fingerprint}
{log_instruction}

Your task:
1. Diagnose the concrete root cause from repository and runtime evidence.
2. If a minimal, safe fix is clear, edit only these allowlisted paths:
   {json.dumps(relative_allowed)}
3. Run focused syntax/tests for files you changed.
4. Return a concise root-cause, changes, and validation summary.

Hard safety rules:
- Do not submit, cancel, hold, release, or modify any Slurm job.
- Do not commit, amend, push, reset, checkout, clean, or change git configuration.
- Do not read, print, copy, or modify credentials, API keys, netrc, or private env files.
- Do not modify files outside the allowlist.
- Preserve pre-existing uncommitted changes.
- Do not broaden scope into unrelated cleanup.
- If evidence is insufficient or the safe fix is unclear, make no edits and say so.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--failure-fingerprint", required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise RuntimeError(f"Repository is not a git checkout: {repo}")
    allowed_paths = resolve_allowed_paths(repo, args.allowed_path)
    log_path = args.log.resolve() if args.log else None
    if log_path is not None and not log_path.is_file():
        log_path = None
    api_key = read_api_key()
    prompt = build_prompt(
        repo=repo,
        experiment=args.experiment,
        failure_fingerprint=args.failure_fingerprint,
        log_path=log_path,
        allowed_paths=allowed_paths,
    )
    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=api_key,
                model="auto",
                name=f"osworld-night-{args.experiment}",
                local=LocalAgentOptions(cwd=str(repo), setting_sources=[]),
            ),
        )
        status = str(getattr(result.status, "value", result.status)).lower()
        payload = {
            "status": status,
            "run_id": result.id,
            "agent_id": result.agent_id,
            "duration_ms": result.duration_ms,
            "result": result.result,
        }
        atomic_private_json(args.result, payload)
        print(
            json.dumps(
                {
                    "status": status,
                    "run_id": result.id,
                    "agent_id": result.agent_id,
                },
                sort_keys=True,
            )
        )
        return 0 if status == "finished" else 2
    except Exception as error:
        atomic_private_json(
            args.result,
            {
                "status": "startup_error",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
