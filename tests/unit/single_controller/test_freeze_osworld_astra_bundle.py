# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from examples.nemo_gym.freeze_osworld_astra_bundle import freeze_astra_bundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path, *, missing_asset: bool = False) -> Path:
    source = root / "accepted"
    task_id = "task_001"
    task_root = source / "tasks" / task_id
    task_path = task_root / "frozen_read" / "task" / "task.json"
    task_path.parent.mkdir(parents=True)

    asset = task_root / "frozen_read" / "reference" / "uploads" / "input.txt"
    if not missing_asset:
        asset.parent.mkdir(parents=True)
        asset.write_text("portable input\n")

    task = {
        "id": task_id,
        "snapshot": "os",
        "instruction": "Edit the uploaded file.",
        "config": [
            {
                "type": "upload_file",
                "parameters": {
                    "files": [
                        {
                            "local_path": "/obsolete/source/input.txt",
                            "path": "/home/user/input.txt",
                        }
                    ]
                },
            }
        ],
        "evaluator": {
            "func": "check_custom",
            "result": {"type": "custom"},
            "expected": {"type": "rule", "rules": {}},
        },
        "metadata": {
            "new_functions_paths": ["/obsolete/getter.py"],
            "upload_path_fixes": [],
        },
    }
    task_path.write_text(json.dumps(task))

    getter = task_root / "work" / "getters" / "custom.py"
    getter.parent.mkdir(parents=True)
    getter.write_text("def get_custom(env=None):\n    return {'ok': True}\n")
    rule = task_root / "frozen_read" / "evaluators" / "getters" / "misc.py"
    rule.parent.mkdir(parents=True)
    rule.write_text("def get_rule(value=None):\n    return value\n")
    metric = task_root / "work" / "metrics" / "custom.py"
    metric.parent.mkdir(parents=True)
    metric.write_text("def check_custom(actual, expected):\n    return 1.0\n")

    routes = {
        "task_id": task_id,
        "task_config": "frozen_read/task/task.json",
        "routes": {
            "getters:get_custom": {
                "file": "work/getters/custom.py",
                "function": "get_custom",
                "source": "accepted_candidate",
                "sha256": _sha256(getter),
            },
            "getters:get_rule": {
                "file": "frozen_read/evaluators/getters/misc.py",
                "function": "get_rule",
                "source": "frozen_original",
                "sha256": _sha256(rule),
            },
            "metrics:check_custom": {
                "file": "work/metrics/custom.py",
                "function": "check_custom",
                "source": "accepted_candidate",
                "sha256": _sha256(metric),
            },
        },
        "issues": [],
        "binary_final_reward": True,
    }
    (task_root / "import_routes_private.json").write_text(json.dumps(routes))
    (task_root / "deployment_private.json").write_text(
        json.dumps({"model_outcome": "revised_tested", "candidate_files": []})
    )

    source.mkdir(exist_ok=True)
    reward_module = source / "binary_reward.py"
    reward_module.write_text("def binary_reward(score):\n    return int(score == 1)\n")
    (source / "ACCEPTANCE_POLICY.json").write_text(
        json.dumps({"reward_policy": "binary_exact_full_success"})
    )
    audit_path = source / "audit_manifest.json"
    audit_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": task_id,
                        "round": 1,
                        "category": "os",
                        "in_training": True,
                        "calibration_successes": 3,
                        "calibration_reps": 8,
                    }
                ]
            }
        )
    )
    scope_sha = _sha256(audit_path)
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-10-15T00:00:00Z",
                "scope_size": 1,
                "counts": {"tasks": 1},
                "task_manifest_sha256": scope_sha,
                "binary_reward_module_sha256": _sha256(reward_module),
                "reward_policy": {
                    "allowed": [0, 1],
                    "full_success_only": True,
                    "errors_propagate": True,
                },
                "packaging_complete": True,
                "tasks_with_packaging_issues": [],
                "task_bindings_complete": True,
                "deployment_integration_complete": True,
            }
        )
    )
    (source / "task_index.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": task_id,
                        "routes": 3,
                        "routing_issues": 0,
                    }
                ],
                "scope_sha256": scope_sha,
            }
        )
    )
    return source


def test_freeze_astra_bundle_builds_portable_binary_gym_snapshot(tmp_path):
    source = _write_bundle(tmp_path)
    output = tmp_path / "snapshot"

    freeze_astra_bundle(source, output, num_repeats=2)

    snapshot = json.loads((output / "SNAPSHOT.json").read_text())
    assert snapshot["unique_tasks"] == 1
    assert snapshot["gym_rows"] == 2
    assert snapshot["selected_route_records"] == 3
    assert snapshot["unique_upload_assets"] == 1
    assert (output / "SHA256SUMS").is_file()
    assert (output / "source_bundle_metadata" / "ACCEPTANCE_POLICY.json").is_file()

    task = json.loads((output / "tasks" / "task_001.json").read_text())
    assert task["metadata"]["binary_final_reward"] is True
    assert task["metadata"]["calib_pass_rate"] == 3 / 8
    assert task["metadata"]["calibration_reps"] == 8
    paths = [Path(path) for path in task["metadata"]["new_functions_paths"]]
    assert len(paths) == 3
    assert all(path.is_relative_to(output / "tmp_funcs" / "task_001") for path in paths)
    assert all(path.is_file() for path in paths)
    frozen_asset = Path(task["config"][0]["parameters"]["files"][0]["local_path"])
    assert frozen_asset.is_relative_to(output / "assets")
    assert frozen_asset.read_text() == "portable input\n"

    rows = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["verifier_metadata"]["id"] == "task_001"
    assert rows[1]["context_compaction_group_id"].endswith("repeat:1")


def test_freeze_astra_bundle_rejects_missing_upload_asset(tmp_path):
    source = _write_bundle(tmp_path, missing_asset=True)
    report = tmp_path / "audit.json"

    with pytest.raises(FileNotFoundError, match="1 upload asset references"):
        freeze_astra_bundle(
            source,
            tmp_path / "snapshot",
            num_repeats=1,
            audit_report=report,
        )
    audit = json.loads(report.read_text())
    assert audit["missing_references"] == 1
    assert audit["affected_tasks"] == 1


def test_freeze_astra_bundle_can_quarantine_missing_upload_asset(tmp_path):
    source = _write_bundle(tmp_path, missing_asset=True)
    output = tmp_path / "snapshot"

    freeze_astra_bundle(
        source,
        output,
        num_repeats=1,
        quarantine_missing_assets=True,
    )

    snapshot = json.loads((output / "SNAPSHOT.json").read_text())
    assert snapshot["source_tasks"] == 1
    assert snapshot["unique_tasks"] == 0
    assert snapshot["quarantined_tasks"] == 1
    quarantine = [
        json.loads(line)
        for line in (output / "quarantine.jsonl").read_text().splitlines()
    ]
    assert quarantine[0]["task_id"] == "task_001"
    assert not (output / "tasks" / "task_001.json").exists()


def test_freeze_astra_bundle_rejects_route_hash_mismatch(tmp_path):
    source = _write_bundle(tmp_path)
    route_path = (
        source / "tasks" / "task_001" / "import_routes_private.json"
    )
    routes = json.loads(route_path.read_text())
    routes["routes"]["metrics:check_custom"]["sha256"] = "0" * 64
    route_path.write_text(json.dumps(routes))

    with pytest.raises(ValueError, match="routed file hash mismatch"):
        freeze_astra_bundle(source, tmp_path / "snapshot", num_repeats=1)


def test_freeze_astra_bundle_rejects_incomplete_deployment_integration(tmp_path):
    source = _write_bundle(tmp_path)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["deployment_integration_complete"] = False
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="deployment_integration_complete"):
        freeze_astra_bundle(source, tmp_path / "snapshot", num_repeats=1)


def test_freeze_astra_bundle_uses_hash_bound_accepted_task_config(tmp_path):
    source = _write_bundle(tmp_path)
    task_root = source / "tasks" / "task_001"
    original = task_root / "frozen_read" / "task" / "task.json"
    accepted = task_root / "work" / "task.json"
    accepted.parent.mkdir(exist_ok=True)
    task = json.loads(original.read_text())
    task["instruction"] = "Accepted corrected instruction."
    accepted.write_text(json.dumps(task))

    routes_path = task_root / "import_routes_private.json"
    routes = json.loads(routes_path.read_text())
    routes["task_config"] = "work/task.json"
    routes_path.write_text(json.dumps(routes))
    (task_root / "deployment_private.json").write_text(
        json.dumps(
            {
                "model_outcome": "revised_tested",
                "candidate_files": [
                    {
                        "packaged_file": "work/task.json",
                        "sha256": _sha256(accepted),
                    }
                ]
            }
        )
    )

    output = tmp_path / "snapshot"
    freeze_astra_bundle(source, output, num_repeats=1)
    frozen = json.loads((output / "tasks" / "task_001.json").read_text())
    assert frozen["instruction"] == "Accepted corrected instruction."


def test_freeze_astra_bundle_copies_candidate_support_closure(tmp_path):
    source = _write_bundle(tmp_path)
    task_root = source / "tasks" / "task_001"
    metric = task_root / "work" / "metrics" / "custom.py"
    metric.write_text(
        "from pathlib import Path\n"
        "from .preservation import preserve\n"
        "def check_custom(actual, expected):\n"
        "    Path(__file__).with_name('baseline.json').read_text()\n"
        "    return preserve(1.0)\n"
    )
    (metric.parent / "preservation.py").write_text(
        "def preserve(value):\n    return value\n"
    )
    (metric.parent / "baseline.json").write_text("{}\n")
    routes_path = task_root / "import_routes_private.json"
    routes = json.loads(routes_path.read_text())
    routes["routes"]["metrics:check_custom"]["sha256"] = _sha256(metric)
    routes_path.write_text(json.dumps(routes))

    output = tmp_path / "snapshot"
    freeze_astra_bundle(source, output, num_repeats=1)

    role_root = output / "tmp_funcs" / "task_001" / "metrics"
    assert (role_root / "preservation.py").is_file()
    assert (role_root / "baseline.json").is_file()


def test_freeze_astra_bundle_copies_task_root_reference_directory(tmp_path):
    source = _write_bundle(tmp_path)
    task_root = source / "tasks" / "task_001"
    getter = task_root / "work" / "getters" / "custom.py"
    getter.write_text(
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parent.parent\n"
        "def get_custom(env=None):\n"
        "    return (ROOT / 'baseline' / 'answer.pdf').read_bytes()\n"
    )
    baseline = task_root / "work" / "baseline" / "answer.pdf"
    baseline.parent.mkdir()
    baseline.write_bytes(b"%PDF-reference\n")
    routes_path = task_root / "import_routes_private.json"
    routes = json.loads(routes_path.read_text())
    routes["routes"]["getters:get_custom"]["sha256"] = _sha256(getter)
    routes_path.write_text(json.dumps(routes))

    output = tmp_path / "snapshot"
    freeze_astra_bundle(source, output, num_repeats=1)

    frozen = output / "tmp_funcs" / "task_001" / "baseline" / "answer.pdf"
    assert frozen.read_bytes() == b"%PDF-reference\n"


def test_freeze_astra_bundle_quarantines_unresolved_evaluator(tmp_path):
    source = _write_bundle(tmp_path)
    deployment_path = (
        source / "tasks" / "task_001" / "deployment_private.json"
    )
    deployment = json.loads(deployment_path.read_text())
    deployment["model_outcome"] = "unresolved"
    deployment_path.write_text(json.dumps(deployment))

    output = tmp_path / "snapshot"
    freeze_astra_bundle(source, output, num_repeats=1)

    snapshot = json.loads((output / "SNAPSHOT.json").read_text())
    assert snapshot["unique_tasks"] == 0
    quarantine = [
        json.loads(line)
        for line in (output / "quarantine.jsonl").read_text().splitlines()
    ]
    assert quarantine == [
        {
            "task_id": "task_001",
            "reason": "evaluator_not_fully_resolved",
            "model_outcome": "unresolved",
        }
    ]


def test_freeze_astra_bundle_rejects_missing_task_root_reference_directory(
    tmp_path,
):
    source = _write_bundle(tmp_path)
    task_root = source / "tasks" / "task_001"
    getter = task_root / "work" / "getters" / "custom.py"
    getter.write_text(
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parent.parent\n"
        "def get_custom(env=None):\n"
        "    return (ROOT / 'sources' / 'input.pptx').read_bytes()\n"
    )
    routes_path = task_root / "import_routes_private.json"
    routes = json.loads(routes_path.read_text())
    routes["routes"]["getters:get_custom"]["sha256"] = _sha256(getter)
    routes_path.write_text(json.dumps(routes))

    report = tmp_path / "audit.json"
    with pytest.raises(FileNotFoundError, match="1 evaluator dependencies"):
        freeze_astra_bundle(
            source,
            tmp_path / "snapshot",
            num_repeats=1,
            audit_report=report,
        )

    audit = json.loads(report.read_text())
    assert audit["affected_tasks"] == 1
    assert audit["missing_evaluator_dependencies"][0]["missing_path"].endswith(
        "/work/sources"
    )
