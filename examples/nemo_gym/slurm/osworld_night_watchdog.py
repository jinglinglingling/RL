#!/usr/bin/env python3
"""Server-side OSWorld experiment watchdog.

The watchdog is intentionally independent of Cursor's IDE terminal bridge. It
runs on a Slurm CPU node, observes allowlisted experiments, retries known
operational failures, and invokes a separate Cursor SDK repair process only for
unknown/code failures.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOG = logging.getLogger("osworld_night_watchdog")
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}
TRANSIENT_PATTERNS = (
    r"ClientConnectorError",
    r"Too Many Requests",
    r"\b429\b",
    r"SetupError",
    r"container start failed",
    r"NODE_FAIL",
    r"PREEMPTED",
    r"\bTIMEOUT\b",
    r"DUE TO TIME LIMIT",
    r"CANCELLED AT",
    r"Connection reset by peer",
    r"temporarily unavailable",
)
DOMINANT_TRANSIENT_PATTERNS = (
    r"Failed to acquire lock on the distribution cache",
    r"Timeout \(\d+s\) when waiting for lock",
    r"Failed to download distribution due to network timeout",
    r"UV_HTTP_TIMEOUT",
    r"ray\.exceptions\.OutOfMemoryError:.*node running low on memory",
)
CODE_FAILURE_PATTERNS = (
    r"ModuleNotFoundError",
    r"ImportError",
    r"AttributeError",
    r"AssertionError",
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"torch\.hub",
    r"flashinfer_cubin",
    r"Could not infer dtype of tokenizers\.Encoding",
    r"FileNotFoundError:.*training_info\.json",
    r"transport endpoint",
)
MAX_LOG_BYTES = 4 * 1024 * 1024
SAFE_SUBMIT_ENV_KEYS = {
    "NRL_FORCE_REBUILD_VENVS",
    "UV_CACHE_DIR_OVERRIDE",
    "WANDB_ENABLED",
    "SBATCH_DEPENDENCY_TYPE",
    "SBATCH_ACCOUNT",
    "SBATCH_PARTITION",
    "SBATCH_TIME",
    "SBATCH_MEM",
    "RAY_memory_usage_threshold",
    "NUM_NODES",
    "GRPO_MAX_NUM_STEPS",
    "OSWORLD_MAX_STEPS",
    "OSWORLD_NUM_PROMPTS_PER_STEP",
    "OSWORLD_NUM_GENERATIONS",
    "OSWORLD_TRAIN_GLOBAL_BATCH_SIZE",
    "OSWORLD_NEMO_GYM_NUM_WORKERS",
    "OSWORLD_MAX_PARALLEL_ROLLOUTS",
    "OSWORLD_GENERATION_BATCH_SIZE",
    "OSWORLD_LEARNING_RATE",
    "OSWORLD_MAX_MODEL_LEN",
    "OSWORLD_MAX_IMAGE_HISTORY_LENGTH",
    "OSWORLD_CONTEXT_PARALLEL_SIZE",
    "OSWORLD_SEQUENCE_LENGTH_DIVISOR",
    "OSWORLD_MONOTONIC_SEGMENTS",
    "OSWORLD_HISTORY_MODE",
    "OSWORLD_INDEPENDENT_TURN_TRAINING",
    "OSWORLD_INDEPENDENT_TURN_SAMPLING",
    "OSWORLD_USE_DYNAMIC_SAMPLING",
    "OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES",
    "OSWORLD_VAL_BATCH_SIZE",
    "OSWORLD_MAX_VAL_SAMPLES",
    "OSWORLD_VAL_PERIOD",
    "RUN_NAME",
    "WANDB_RUN_NAME",
    "WANDB_RUN_ID",
}


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    name: str
    state: str
    reason: str


@dataclass(frozen=True)
class Failure:
    category: str
    fingerprint: str
    event_id: str
    log_path: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_checkpoint_step(path: Path) -> int:
    try:
        payload = json.loads(path.read_text())
        return int(payload.get("last_checkpoint_step", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def read_validated_checkpoint_step(status_path: Path) -> int:
    """Return the highest atomically completed checkpoint, not the status hint."""
    checkpoint_dir = status_path.parent
    completed_steps: list[int] = []
    try:
        candidates = list(checkpoint_dir.glob("step_*"))
    except OSError:
        return -1
    for candidate in candidates:
        match = re.fullmatch(r"step_(\d+)", candidate.name)
        if not match or not candidate.is_dir():
            continue
        step = int(match.group(1))
        if (checkpoint_dir / f"tmp_step_{step}").exists():
            continue
        try:
            training_info = json.loads(
                (candidate / "training_info.json").read_text()
            )
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(training_info, dict):
            completed_steps.append(step)
    return max(completed_steps, default=-1)


def parse_squeue(output: str) -> list[SlurmJob]:
    jobs: list[SlurmJob] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            LOG.warning("Ignoring malformed squeue line: %r", line)
            continue
        jobs.append(
            SlurmJob(
                job_id=fields[0].strip(),
                name=fields[1].strip(),
                state=fields[2].strip().upper(),
                reason=fields[3].strip(),
            )
        )
    return jobs


def classify_log(text: str, *, event_seed: str) -> Failure:
    dominant_transient_matches = [
        pattern
        for pattern in DOMINANT_TRANSIENT_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    code_matches = [
        pattern
        for pattern in CODE_FAILURE_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    transient_matches = [
        pattern
        for pattern in TRANSIENT_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if dominant_transient_matches:
        category = "transient"
        signatures = dominant_transient_matches
    elif code_matches:
        category = "code"
        signatures = code_matches
    elif transient_matches:
        category = "transient"
        signatures = transient_matches
    else:
        category = "unknown"
        signatures = ["no-known-signature"]
    signature_material = "\n".join(sorted(signatures))
    fingerprint = hashlib.sha256(signature_material.encode()).hexdigest()[:20]
    event_id = hashlib.sha256(
        f"{event_seed}\n{fingerprint}\n{hashlib.sha256(text.encode()).hexdigest()}".encode()
    ).hexdigest()[:20]
    return Failure(
        category=category,
        fingerprint=fingerprint,
        event_id=event_id,
        log_path=None,
    )


def compute_backoff(
    consecutive_failures: int, base_seconds: int, maximum_seconds: int
) -> int:
    exponent = max(0, consecutive_failures - 1)
    return min(maximum_seconds, base_seconds * (2**exponent))


def evaluate_metric_thresholds(
    summary: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, float]]
) -> tuple[bool, dict[str, float], list[str]]:
    observed: dict[str, float] = {}
    failures: list[str] = []
    for metric, limits in thresholds.items():
        try:
            value = float(summary[metric])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{metric}:missing_or_non_numeric")
            continue
        if not math.isfinite(value):
            failures.append(f"{metric}:not_finite")
            continue
        observed[metric] = value
        if "min" in limits and value < float(limits["min"]):
            failures.append(f"{metric}:below_min")
        if "max" in limits and value > float(limits["max"]):
            failures.append(f"{metric}:above_max")
    return not failures, observed, failures


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class Watchdog:
    def __init__(
        self,
        config_path: Path,
        *,
        dry_run: bool = False,
        command_timeout: int = 30,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text())
        if self.config.get("version") != 1:
            raise ValueError("Unsupported watchdog config version")
        default_root = Path(__file__).resolve().parents[3]
        self.repo_root = resolve_repo_path(
            default_root, self.config.get("repo_root", str(default_root))
        )
        self.state_dir = resolve_repo_path(
            self.repo_root, self.config["state_dir"]
        )
        self.state_path = self.state_dir / "state.json"
        self.audit_path = self.state_dir / "audit.jsonl"
        self.status_path = self.state_dir / "status.json"
        self.dry_run = dry_run
        self.command_timeout = command_timeout
        self.state = self._load_state()
        self._validate_config()

    def _load_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text())
            if isinstance(state, dict):
                return state
        except (OSError, json.JSONDecodeError):
            pass
        return {"version": 1, "experiments": {}}

    def _save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def _audit(self, event: str, **details: Any) -> None:
        payload = {"timestamp": utc_now(), "event": event, **details}
        append_jsonl(self.audit_path, payload)
        LOG.info("%s %s", event, json.dumps(details, sort_keys=True))

    def _validate_config(self) -> None:
        experiments = self.config.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("Config must contain at least one experiment")
        names: set[str] = set()
        for experiment in experiments:
            name = experiment["name"]
            if name in names:
                raise ValueError(f"Duplicate experiment name: {name}")
            names.add(name)
            script = resolve_repo_path(self.repo_root, experiment["submit_script"])
            if not path_is_within(script, self.repo_root) or not script.is_file():
                raise ValueError(f"Submit script is not allowlisted in repo: {script}")
            prefixes = experiment.get("job_name_prefixes", [])
            if not prefixes or any(not prefix for prefix in prefixes):
                raise ValueError(f"{name}: job_name_prefixes must be non-empty")
            submit_prefix = experiment["submit_job_name_prefix"]
            if submit_prefix not in prefixes:
                raise ValueError(
                    f"{name}: submit prefix must appear in job_name_prefixes"
                )
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", submit_prefix):
                raise ValueError(f"{name}: unsafe submit job-name prefix")
            submit_env = set(experiment.get("submit_env", {}))
            unsafe_env = submit_env - SAFE_SUBMIT_ENV_KEYS
            if unsafe_env:
                raise ValueError(
                    f"{name}: unsupported submit env keys: {sorted(unsafe_env)}"
                )
            extension = experiment.get("extension")
            if extension:
                base_target = int(experiment["target_step"])
                if int(extension["from_step"]) != base_target:
                    raise ValueError(f"{name}: extension must start at target_step")
                if int(extension["target_step"]) <= base_target:
                    raise ValueError(f"{name}: extension target must increase")
                extension_env = set(extension.get("submit_env", {}))
                unsafe_extension_env = extension_env - SAFE_SUBMIT_ENV_KEYS
                if unsafe_extension_env:
                    raise ValueError(
                        f"{name}: unsupported extension env keys: "
                        f"{sorted(unsafe_extension_env)}"
                    )
                if not extension.get("wandb", {}).get("thresholds"):
                    raise ValueError(f"{name}: extension requires metric thresholds")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        LOG.debug("Running command: %s", command)
        return subprocess.run(
            list(command),
            cwd=self.repo_root,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout or self.command_timeout,
            check=False,
        )

    def _list_jobs(self) -> list[SlurmJob]:
        command = [
            "squeue",
            "-h",
            "-u",
            os.environ.get("USER", ""),
            "-o",
            "%i|%j|%T|%R",
        ]
        result = self._run(command)
        if result.returncode != 0:
            raise RuntimeError(f"squeue failed: {result.stdout[-1000:]}")
        return parse_squeue(result.stdout)

    def _experiment_state(self, name: str) -> dict[str, Any]:
        experiments = self.state.setdefault("experiments", {})
        state = experiments.setdefault(
            name,
            {
                "consecutive_failures": 0,
                "submission_history": [],
                "agent_attempts": {},
                "last_checkpoint_step": -1,
            },
        )
        return state

    def _active_jobs(
        self, experiment: Mapping[str, Any], jobs: Iterable[SlurmJob]
    ) -> list[SlurmJob]:
        prefixes = tuple(experiment["job_name_prefixes"])
        return [
            job
            for job in jobs
            if job.state in ACTIVE_SLURM_STATES
            and job.name.startswith(prefixes)
            and "DependencyNeverSatisfied" not in job.reason
        ]

    def _latest_driver_log(self, experiment: Mapping[str, Any]) -> Path | None:
        results_dir = resolve_repo_path(self.repo_root, experiment["results_dir"])
        candidates = list(results_dir.glob("slurm/*-logs/ray-driver.log"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def _read_log_tail(self, path: Path) -> str:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - MAX_LOG_BYTES))
            return stream.read().decode("utf-8", errors="replace")

    def _failure(self, experiment: Mapping[str, Any]) -> Failure:
        path = self._latest_driver_log(experiment)
        if path is None:
            failure = classify_log("", event_seed=f"{experiment['name']}:no-log")
            return failure
        stat = path.stat()
        text = self._read_log_tail(path)
        job_id = path.parent.name.removesuffix("-logs")
        state_result = self._run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=State",
                "-X",
                "-n",
                "-P",
            ],
            timeout=min(self.command_timeout, 30),
        )
        slurm_state = (
            state_result.stdout.strip().splitlines()[0].strip()
            if state_result.returncode == 0 and state_result.stdout.strip()
            else ""
        )
        classification_text = f"SLURM_STATE={slurm_state}\n{text}"
        failure = classify_log(
            classification_text,
            event_seed=f"{path}:{stat.st_mtime_ns}:{stat.st_size}:{slurm_state}",
        )
        return Failure(
            category=failure.category,
            fingerprint=failure.fingerprint,
            event_id=failure.event_id,
            log_path=str(path),
        )

    def _evaluate_extension(
        self,
        experiment: Mapping[str, Any],
        exp_state: dict[str, Any],
        checkpoint_step: int,
    ) -> dict[str, Any] | None:
        extension = experiment.get("extension")
        if not extension or checkpoint_step < int(extension["from_step"]):
            return None
        cached = exp_state.get("extension_decision")
        if isinstance(cached, dict) and cached.get("ready"):
            return cached
        wandb_config = extension["wandb"]
        run_path = (
            f"{wandb_config['entity']}/{wandb_config['project']}/"
            f"{wandb_config['run_id']}"
        )
        try:
            import wandb

            run = wandb.Api(
                timeout=int(wandb_config.get("timeout_seconds", 30))
            ).run(run_path)
            summary = dict(run.summary)
            metric_step = int(summary.get("_step", -1))
        except Exception as error:
            self._audit(
                "extension_gate_unavailable",
                experiment=experiment["name"],
                run_path=run_path,
                error_type=type(error).__name__,
                error=str(error)[:500],
            )
            return {"ready": False, "reason": "wandb_unavailable"}
        required_step = int(extension["from_step"])
        if metric_step < required_step:
            self._audit(
                "extension_gate_waiting",
                experiment=experiment["name"],
                metric_step=metric_step,
                required_step=required_step,
            )
            return {
                "ready": False,
                "reason": "metrics_not_fresh",
                "metric_step": metric_step,
            }
        passed, observed, failures = evaluate_metric_thresholds(
            summary, wandb_config["thresholds"]
        )
        decision = {
            "ready": True,
            "passed": passed,
            "metric_step": metric_step,
            "observed": observed,
            "failures": failures,
            "decided_at": utc_now(),
        }
        exp_state["extension_decision"] = decision
        self._audit(
            "extension_gate_passed" if passed else "extension_gate_rejected",
            experiment=experiment["name"],
            metric_step=metric_step,
            observed=observed,
            failures=failures,
            target_step=int(extension["target_step"]),
        )
        self._save_state()
        return decision

    def _prune_submission_history(self, exp_state: dict[str, Any], now: float) -> None:
        one_hour_ago = now - 3600
        exp_state["submission_history"] = [
            entry
            for entry in exp_state.get("submission_history", [])
            if float(entry.get("timestamp_epoch", 0)) >= one_hour_ago
        ]

    def _agent_available(self) -> tuple[bool, str]:
        agent_config = self.config.get("agent", {})
        if not agent_config.get("enabled", False):
            return False, "disabled"
        key_file_value = os.environ.get("CURSOR_API_KEY_FILE") or agent_config.get(
            "api_key_file"
        )
        if not key_file_value:
            return False, "CURSOR_API_KEY_FILE is not configured"
        key_file = Path(key_file_value).expanduser()
        if not key_file.is_file():
            return False, f"API key file does not exist: {key_file}"
        if key_file.stat().st_mode & 0o077:
            return False, f"API key file must have mode 0600: {key_file}"
        sdk_python = resolve_repo_path(self.repo_root, agent_config["python"])
        runner = resolve_repo_path(self.repo_root, agent_config["runner"])
        if not sdk_python.is_file() or not runner.is_file():
            return False, "SDK Python or agent runner is missing"
        return True, ""

    def _git_snapshot(self) -> tuple[str, dict[str, str]]:
        head_result = self._run(["git", "rev-parse", "HEAD"])
        if head_result.returncode != 0:
            raise RuntimeError("Could not read git HEAD before agent run")
        status_result = self._run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"]
        )
        if status_result.returncode != 0:
            raise RuntimeError("Could not snapshot working tree before agent run")
        snapshot: dict[str, str] = {}
        for line in status_result.stdout.splitlines():
            if len(line) < 4:
                continue
            raw_path = line[3:]
            if " -> " in raw_path:
                raw_path = raw_path.rsplit(" -> ", 1)[1]
            raw_path = raw_path.strip('"')
            path = self.repo_root / raw_path
            if path.is_symlink():
                content_hash = "symlink:" + os.readlink(path)
            elif path.is_file():
                content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                content_hash = "<missing>"
            snapshot[raw_path] = content_hash
        return head_result.stdout.strip(), snapshot

    def _agent_changes_are_allowed(
        self,
        *,
        experiment: Mapping[str, Any],
        before_head: str,
        before_snapshot: Mapping[str, str],
    ) -> bool:
        after_head, after_snapshot = self._git_snapshot()
        if after_head != before_head:
            self._audit(
                "agent_changed_head",
                experiment=experiment["name"],
                before_head=before_head,
                after_head=after_head,
            )
            return False
        changed_paths = sorted(
            path
            for path in set(before_snapshot) | set(after_snapshot)
            if before_snapshot.get(path) != after_snapshot.get(path)
        )
        allowed_roots = [
            resolve_repo_path(self.repo_root, value)
            for value in self.config["agent"].get("allowed_paths", [])
        ]
        violations: list[str] = []
        for value in changed_paths:
            path = (self.repo_root / value).resolve()
            if not any(
                path == allowed or path_is_within(path, allowed)
                for allowed in allowed_roots
            ):
                violations.append(value)
        if violations:
            self._audit(
                "agent_path_violation",
                experiment=experiment["name"],
                paths=violations,
            )
            return False
        self._audit(
            "agent_changes_checked",
            experiment=experiment["name"],
            changed_paths=changed_paths,
        )
        return True

    def _run_agent(
        self,
        experiment: Mapping[str, Any],
        failure: Failure,
        exp_state: dict[str, Any],
    ) -> bool:
        agent_config = self.config["agent"]
        available, reason = self._agent_available()
        if not available:
            self._audit(
                "agent_unavailable",
                experiment=experiment["name"],
                reason=reason,
                failure_fingerprint=failure.fingerprint,
            )
            return False
        before_head, before_snapshot = self._git_snapshot()
        attempts = exp_state.setdefault("agent_attempts", {})
        current_attempts = int(attempts.get(failure.fingerprint, 0))
        maximum = int(agent_config.get("max_attempts_per_fingerprint", 1))
        if current_attempts >= maximum:
            self._audit(
                "agent_attempt_limit",
                experiment=experiment["name"],
                failure_fingerprint=failure.fingerprint,
                attempts=current_attempts,
            )
            return False
        attempts[failure.fingerprint] = current_attempts + 1
        self._save_state()
        result_dir = self.state_dir / "agent-results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / (
            f"{experiment['name']}-{failure.fingerprint}-{int(time.time())}.json"
        )
        command = [
            str(resolve_repo_path(self.repo_root, agent_config["python"])),
            str(resolve_repo_path(self.repo_root, agent_config["runner"])),
            "--repo",
            str(self.repo_root),
            "--experiment",
            experiment["name"],
            "--failure-fingerprint",
            failure.fingerprint,
            "--result",
            str(result_path),
        ]
        if failure.log_path:
            command.extend(["--log", failure.log_path])
        for allowed_path in agent_config.get("allowed_paths", []):
            command.extend(["--allowed-path", allowed_path])
        env = os.environ.copy()
        key_file = os.environ.get("CURSOR_API_KEY_FILE") or agent_config.get(
            "api_key_file", ""
        )
        env["CURSOR_API_KEY_FILE"] = str(Path(key_file).expanduser())
        self._audit(
            "agent_started",
            experiment=experiment["name"],
            failure_fingerprint=failure.fingerprint,
            result_path=str(result_path),
        )
        result = self._run(
            command,
            timeout=int(agent_config.get("timeout_seconds", 2700)),
            env=env,
        )
        if result.returncode != 0:
            self._audit(
                "agent_failed",
                experiment=experiment["name"],
                failure_fingerprint=failure.fingerprint,
                returncode=result.returncode,
                output_tail=result.stdout[-2000:],
            )
            return False
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            self._audit(
                "agent_invalid_result",
                experiment=experiment["name"],
                failure_fingerprint=failure.fingerprint,
            )
            return False
        if payload.get("status") != "finished":
            self._audit(
                "agent_unsuccessful",
                experiment=experiment["name"],
                failure_fingerprint=failure.fingerprint,
                status=payload.get("status"),
                run_id=payload.get("run_id"),
            )
            return False
        if not self._agent_changes_are_allowed(
            experiment=experiment,
            before_head=before_head,
            before_snapshot=before_snapshot,
        ):
            return False
        return self._run_validations(experiment)

    def _run_validations(self, experiment: Mapping[str, Any]) -> bool:
        commands = self.config.get("agent", {}).get("validation_commands", [])
        for raw_command in commands:
            if not isinstance(raw_command, list) or not raw_command:
                raise ValueError("validation_commands entries must be argv arrays")
            command = [str(part) for part in raw_command]
            result = self._run(
                command,
                timeout=int(
                    self.config.get("agent", {}).get(
                        "validation_timeout_seconds", 900
                    )
                ),
            )
            if result.returncode != 0:
                self._audit(
                    "validation_failed",
                    experiment=experiment["name"],
                    command=command,
                    returncode=result.returncode,
                    output_tail=result.stdout[-2000:],
                )
                return False
        self._audit("validation_passed", experiment=experiment["name"])
        return True

    def _submit(
        self,
        experiment: Mapping[str, Any],
        exp_state: dict[str, Any],
        failure: Failure,
        now: float,
    ) -> str | None:
        history = exp_state.setdefault("submission_history", [])
        max_per_hour = int(self.config["submission"]["max_per_hour"])
        if len(history) >= max_per_hour:
            self._audit(
                "submission_rate_limited",
                experiment=experiment["name"],
                submissions_last_hour=len(history),
            )
            return None
        sequence = int(exp_state.get("submission_sequence", 0)) + 1
        job_name = f"{experiment['submit_job_name_prefix']}{sequence}"
        script = resolve_repo_path(self.repo_root, experiment["submit_script"])
        if self.dry_run:
            self._audit(
                "submission_dry_run",
                experiment=experiment["name"],
                job_name=job_name,
                script=str(script),
            )
            return "DRY_RUN"
        env = os.environ.copy()
        for key, value in experiment.get("submit_env", {}).items():
            if key not in SAFE_SUBMIT_ENV_KEYS:
                raise ValueError(f"Unsupported submit env key: {key}")
            env[str(key)] = str(value)
        env["JOB_NAME"] = job_name
        env.pop("SBATCH_DEPENDENCY", None)
        result = self._run(
            ["bash", str(script)],
            timeout=int(self.config["submission"].get("timeout_seconds", 120)),
            env=env,
        )
        if result.returncode != 0:
            self._audit(
                "submission_failed",
                experiment=experiment["name"],
                job_name=job_name,
                returncode=result.returncode,
                output_tail=result.stdout[-2000:],
            )
            return None
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            self._audit(
                "submission_unparseable",
                experiment=experiment["name"],
                job_name=job_name,
                output_tail=result.stdout[-1000:],
            )
            return None
        job_id = match.group(1)
        exp_state["submission_sequence"] = sequence
        exp_state["last_submitted_event_id"] = failure.event_id
        history.append(
            {
                "timestamp": utc_now(),
                "timestamp_epoch": now,
                "job_id": job_id,
                "job_name": job_name,
            }
        )
        self._audit(
            "submitted",
            experiment=experiment["name"],
            job_id=job_id,
            job_name=job_name,
            failure_category=failure.category,
            failure_fingerprint=failure.fingerprint,
        )
        return job_id

    def _handle_experiment(
        self,
        experiment: Mapping[str, Any],
        jobs: Sequence[SlurmJob],
        now: float,
    ) -> dict[str, Any]:
        name = experiment["name"]
        exp_state = self._experiment_state(name)
        checkpoint_path = resolve_repo_path(
            self.repo_root, experiment["checkpoint_status"]
        )
        checkpoint_hint = read_checkpoint_step(checkpoint_path)
        checkpoint_step = read_validated_checkpoint_step(checkpoint_path)
        if checkpoint_hint > checkpoint_step:
            self._audit(
                "checkpoint_hint_not_finalized",
                experiment=name,
                checkpoint_hint=checkpoint_hint,
                validated_checkpoint_step=checkpoint_step,
            )
        previous_step = int(exp_state.get("last_checkpoint_step", -1))
        if checkpoint_step > previous_step:
            exp_state["last_checkpoint_step"] = checkpoint_step
            exp_state["consecutive_failures"] = 0
            exp_state.pop("next_action_at_epoch", None)
            self._audit(
                "checkpoint_advanced",
                experiment=name,
                previous_step=previous_step,
                checkpoint_step=checkpoint_step,
            )
        target_step = int(experiment["target_step"])
        extension_active = False
        extension = experiment.get("extension")
        if extension and checkpoint_step >= int(extension["from_step"]):
            decision = self._evaluate_extension(
                experiment, exp_state, checkpoint_step
            )
            if decision is None or not decision.get("ready"):
                exp_state["status"] = "extension_gate_waiting"
                return {
                    "name": name,
                    "status": "extension_gate_waiting",
                    "checkpoint_step": checkpoint_step,
                    "target_step": target_step,
                    "reason": (decision or {}).get("reason", "not_ready"),
                }
            if not decision.get("passed"):
                exp_state["status"] = "extension_rejected"
                return {
                    "name": name,
                    "status": "extension_rejected",
                    "checkpoint_step": checkpoint_step,
                    "target_step": target_step,
                    "metrics": decision.get("observed", {}),
                    "failures": decision.get("failures", []),
                }
            extension_active = True
            target_step = int(extension["target_step"])
        if checkpoint_step >= target_step:
            exp_state["status"] = "complete"
            return {
                "name": name,
                "status": "complete",
                "checkpoint_step": checkpoint_step,
                "target_step": target_step,
            }

        active_jobs = self._active_jobs(experiment, jobs)
        if active_jobs:
            exp_state["status"] = "active"
            return {
                "name": name,
                "status": "active",
                "checkpoint_step": checkpoint_step,
                "target_step": target_step,
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "state": job.state,
                        "reason": job.reason,
                    }
                    for job in active_jobs
                ],
            }

        planned_extension = extension_active and not exp_state.get(
            "extension_started", False
        )
        if planned_extension:
            extension_fingerprint = hashlib.sha256(
                f"{name}:{checkpoint_step}:{target_step}".encode()
            ).hexdigest()[:20]
            failure = Failure(
                category="extension",
                fingerprint=extension_fingerprint,
                event_id=extension_fingerprint,
                log_path=None,
            )
        else:
            failure = self._failure(experiment)
        is_new_event = exp_state.get("last_failure_event_id") != failure.event_id
        if is_new_event:
            exp_state["last_failure_event_id"] = failure.event_id
            exp_state["consecutive_failures"] = (
                int(exp_state.get("consecutive_failures", 0)) + 1
            )
            backoff = compute_backoff(
                int(exp_state["consecutive_failures"]),
                int(self.config["submission"]["base_backoff_seconds"]),
                int(self.config["submission"]["max_backoff_seconds"]),
            )
            exp_state["next_action_at_epoch"] = now + backoff
            self._audit(
                "failure_detected",
                experiment=name,
                category=failure.category,
                failure_fingerprint=failure.fingerprint,
                event_id=failure.event_id,
                backoff_seconds=backoff,
                log_path=failure.log_path,
            )

        next_action_at = float(exp_state.get("next_action_at_epoch", now))
        if now < next_action_at:
            exp_state["status"] = "backoff"
            return {
                "name": name,
                "status": "backoff",
                "checkpoint_step": checkpoint_step,
                "target_step": target_step,
                "failure_category": failure.category,
                "next_action_at_epoch": next_action_at,
            }

        if exp_state.get("last_submitted_event_id") == failure.event_id:
            exp_state["status"] = "awaiting_new_event"
            return {
                "name": name,
                "status": "awaiting_new_event",
                "checkpoint_step": checkpoint_step,
                "target_step": target_step,
            }

        if failure.category in {"code", "unknown"}:
            if not self._run_agent(experiment, failure, exp_state):
                exp_state["status"] = "needs_agent"
                return {
                    "name": name,
                    "status": "needs_agent",
                    "checkpoint_step": checkpoint_step,
                    "target_step": target_step,
                    "failure_category": failure.category,
                    "failure_fingerprint": failure.fingerprint,
                }

        self._prune_submission_history(exp_state, now)
        submit_experiment = dict(experiment)
        if extension_active:
            submit_experiment["submit_env"] = {
                **experiment.get("submit_env", {}),
                **extension.get("submit_env", {}),
            }
        job_id = self._submit(submit_experiment, exp_state, failure, now)
        if job_id and extension_active and not self.dry_run:
            exp_state["extension_started"] = True
        exp_state["status"] = "submitted" if job_id else "submission_blocked"
        return {
            "name": name,
            "status": exp_state["status"],
            "checkpoint_step": checkpoint_step,
            "target_step": target_step,
            "job_id": job_id,
        }

    def run_once(self) -> dict[str, Any]:
        now = time.time()
        jobs = self._list_jobs()
        agent_available, agent_reason = self._agent_available()
        statuses = [
            self._handle_experiment(experiment, jobs, now)
            for experiment in self.config["experiments"]
        ]
        payload = {
            "timestamp": utc_now(),
            "dry_run": self.dry_run,
            "agent": {
                "available": agent_available,
                "reason": "" if agent_available else agent_reason,
            },
            "experiments": statuses,
        }
        atomic_write_json(self.status_path, payload)
        self._save_state()
        return payload

    def run_forever(self) -> None:
        poll_seconds = int(self.config.get("poll_seconds", 300))
        while True:
            try:
                status = self.run_once()
                LOG.info("heartbeat %s", json.dumps(status, sort_keys=True))
            except Exception:
                LOG.exception("watchdog iteration failed")
                self._audit("watchdog_iteration_failed")
            time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    watchdog = Watchdog(args.config, dry_run=args.dry_run)
    lock_path = watchdog.state_dir / "watchdog.lock"
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.error("Another watchdog process holds %s", lock_path)
            return 3
        watchdog._audit(
            "watchdog_started",
            pid=os.getpid(),
            config=str(args.config.resolve()),
            once=args.once,
            dry_run=args.dry_run,
        )
        if args.once:
            print(json.dumps(watchdog.run_once(), indent=2, sort_keys=True))
        else:
            watchdog.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
