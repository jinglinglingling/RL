# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHDOG_PATH = (
    REPO_ROOT / "examples/nemo_gym/slurm/osworld_night_watchdog.py"
)
SPEC = importlib.util.spec_from_file_location("osworld_night_watchdog", WATCHDOG_PATH)
assert SPEC is not None and SPEC.loader is not None
watchdog_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog_module
SPEC.loader.exec_module(watchdog_module)


def _make_config(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    submit_script = repo / "submit.sh"
    submit_script.write_text("#!/usr/bin/env bash\nexit 0\n")
    checkpoint = repo / "results/checkpoint.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text('{"last_checkpoint_step": 3}\n')
    step_3 = checkpoint.parent / "step_3"
    step_3.mkdir()
    (step_3 / "training_info.json").write_text('{"current_step": 3}\n')
    config = {
        "version": 1,
        "repo_root": str(repo),
        "state_dir": "results/watchdog",
        "poll_seconds": 300,
        "submission": {
            "max_per_hour": 2,
            "base_backoff_seconds": 60,
            "max_backoff_seconds": 600,
        },
        "agent": {"enabled": False},
        "experiments": [
            {
                "name": "tiny",
                "results_dir": "results/run",
                "checkpoint_status": "results/checkpoint.json",
                "target_step": 20,
                "submit_script": "submit.sh",
                "job_name_prefixes": ["tiny-local-", "tiny-night-"],
                "submit_job_name_prefix": "tiny-night-",
                "submit_env": {"SBATCH_PARTITION": "batch"},
            }
        ],
    }
    config_path = tmp_path / "watchdog.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_parse_squeue() -> None:
    jobs = watchdog_module.parse_squeue(
        "123|tiny-local-1|RUNNING|node01\n"
        "124|tiny-local-2|PENDING|Dependency\n"
    )
    assert [(job.job_id, job.state) for job in jobs] == [
        ("123", "RUNNING"),
        ("124", "PENDING"),
    ]


def test_code_failure_takes_precedence_over_cancellation() -> None:
    failure = watchdog_module.classify_log(
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: torch.hub\n"
        "JOB CANCELLED AT 00:00\n",
        event_seed="job-1",
    )
    assert failure.category == "code"


def test_transient_failure_classification() -> None:
    failure = watchdog_module.classify_log(
        "ClientConnectorError: temporarily unavailable",
        event_seed="job-2",
    )
    assert failure.category == "transient"


def test_uv_cache_lock_is_transient_even_after_partial_preflight_failure() -> None:
    failure = watchdog_module.classify_log(
        "Failed to acquire lock on the distribution cache\n"
        "ModuleNotFoundError: No module named 'transformer_engine'\n",
        event_seed="job-3",
    )
    assert failure.category == "transient"


def test_exponential_backoff_is_capped() -> None:
    assert watchdog_module.compute_backoff(1, 60, 600) == 60
    assert watchdog_module.compute_backoff(3, 60, 600) == 240
    assert watchdog_module.compute_backoff(10, 60, 600) == 600


def test_metric_thresholds_require_all_metrics_to_pass() -> None:
    passed, observed, failures = watchdog_module.evaluate_metric_thresholds(
        {
            "validation/accuracy": 0.6,
            "validation/natural_termination_rate": 1.0,
            "validation/truncation_rate": 0.0,
        },
        {
            "validation/accuracy": {"min": 0.4},
            "validation/natural_termination_rate": {"min": 0.95},
            "validation/truncation_rate": {"max": 0.1},
        },
    )
    assert passed
    assert observed["validation/accuracy"] == 0.6
    assert failures == []


def test_metric_thresholds_reject_missing_and_out_of_range_values() -> None:
    passed, _, failures = watchdog_module.evaluate_metric_thresholds(
        {"validation/accuracy": 0.2},
        {
            "validation/accuracy": {"min": 0.4},
            "validation/truncation_rate": {"max": 0.1},
        },
    )
    assert not passed
    assert "validation/accuracy:below_min" in failures
    assert "validation/truncation_rate:missing_or_non_numeric" in failures


def test_checkpoint_reader_handles_invalid_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"last_checkpoint_step": 10}\n')
    assert watchdog_module.read_checkpoint_step(checkpoint) == 10
    checkpoint.write_text("not-json")
    assert watchdog_module.read_checkpoint_step(checkpoint) == -1


def test_validated_checkpoint_reader_ignores_incomplete_saves(tmp_path: Path) -> None:
    status = tmp_path / "latest_checkpoint_status.json"
    status.write_text('{"last_checkpoint_step": 15}\n')
    step_5 = tmp_path / "step_5"
    step_5.mkdir()
    (step_5 / "training_info.json").write_text('{"current_step": 5}\n')
    step_10 = tmp_path / "step_10"
    step_10.mkdir()
    (step_10 / "training_info.json").write_text('{"current_step": 10}\n')
    (tmp_path / "tmp_step_10").mkdir()
    step_15 = tmp_path / "step_15"
    step_15.mkdir()
    (step_15 / "training_info.json").write_text("not-json")

    assert watchdog_module.read_validated_checkpoint_step(status) == 5


def test_dependency_never_satisfied_is_not_active(tmp_path: Path) -> None:
    instance = watchdog_module.Watchdog(_make_config(tmp_path), dry_run=True)
    experiment = instance.config["experiments"][0]
    jobs = [
        watchdog_module.SlurmJob(
            job_id="1",
            name="tiny-local-1",
            state="PENDING",
            reason="DependencyNeverSatisfied",
        ),
        watchdog_module.SlurmJob(
            job_id="2",
            name="tiny-local-2",
            state="RUNNING",
            reason="node01",
        ),
    ]
    active = instance._active_jobs(experiment, jobs)
    assert [job.job_id for job in active] == ["2"]


def test_run_once_reports_active_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = watchdog_module.Watchdog(_make_config(tmp_path), dry_run=True)
    monkeypatch.setattr(
        instance,
        "_list_jobs",
        lambda: [
            watchdog_module.SlurmJob(
                job_id="42",
                name="tiny-local-1",
                state="RUNNING",
                reason="node01",
            )
        ],
    )
    status = instance.run_once()
    experiment = status["experiments"][0]
    assert experiment["status"] == "active"
    assert experiment["checkpoint_step"] == 3
    assert experiment["jobs"][0]["job_id"] == "42"


def test_extension_gate_uses_fresh_wandb_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _make_config(tmp_path)
    config = json.loads(config_path.read_text())
    config["experiments"][0]["extension"] = {
        "from_step": 20,
        "target_step": 40,
        "submit_env": {"GRPO_MAX_NUM_STEPS": "40"},
        "wandb": {
            "entity": "nvidia",
            "project": "osworld-grpo",
            "run_id": "tiny",
            "thresholds": {"validation/accuracy": {"min": 0.4}},
        },
    }
    config_path.write_text(json.dumps(config))
    fake_wandb = SimpleNamespace(
        Api=lambda timeout: SimpleNamespace(
            run=lambda path: SimpleNamespace(
                summary={"_step": 20, "validation/accuracy": 0.7}
            )
        )
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    instance = watchdog_module.Watchdog(config_path, dry_run=True)
    decision = instance._evaluate_extension(
        instance.config["experiments"][0],
        instance._experiment_state("tiny"),
        checkpoint_step=20,
    )
    assert decision is not None
    assert decision["ready"]
    assert decision["passed"]
    assert decision["observed"]["validation/accuracy"] == 0.7


def test_config_rejects_secret_submit_env(tmp_path: Path) -> None:
    config_path = _make_config(tmp_path)
    config = json.loads(config_path.read_text())
    config["experiments"][0]["submit_env"]["WANDB_API_KEY"] = "secret"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported submit env"):
        watchdog_module.Watchdog(config_path, dry_run=True)
