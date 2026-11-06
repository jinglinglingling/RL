# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an accepted Astra OSWorld verifier bundle into a Gym snapshot.

The accepted bundle is an audit/delivery format rather than a runnable Gym
dataset.  This command follows each task's explicit evaluator import routes,
copies only the selected verifier modules and referenced upload assets, applies
the exact-full-success binary reward contract, and atomically emits the
``tasks/``, ``tmp_funcs/``, ``train.jsonl`` and checksum files consumed by the
OSWorld NeMo Gym integration.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nemo_gym.freeze_osworld_rlvr_band import (
    _gym_row,
    _repair_runtime_commands,
    _upload_entries,
    _write_checksums,
    _write_jsonl,
)


_ALLOWED_ROUTE_ROLES = frozenset({"getters", "metrics"})
_ALLOWED_ROUTE_SOURCES = frozenset({"accepted_candidate", "frozen_original"})
_TASK_ROOT_REFERENCE_DIRS = frozenset(
    {
        "assets",
        "baseline",
        "baseline_alt",
        "fixtures",
        "reference",
        "references",
        "resources",
        "sources",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validated_task_ids(
    source: Path,
) -> tuple[list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    metadata = _load_object(source / "metadata.json")
    index = _load_object(source / "task_index.json")
    audit_path = source / "audit_manifest.json"
    audit = _load_object(audit_path)
    rows = index.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("task_index.json has no tasks list")

    task_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("task_index.json contains a non-object task")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"invalid task_id in task_index.json: {task_id!r}")
        if task_id in seen:
            raise ValueError(f"duplicate task_id in task_index.json: {task_id}")
        if row.get("routing_issues") != 0:
            raise ValueError(f"{task_id}: task index reports routing issues")
        seen.add(task_id)
        task_ids.append(task_id)

    expected = metadata.get("scope_size")
    if expected != len(task_ids):
        raise ValueError(
            f"metadata/index task count mismatch: scope_size={expected!r}, "
            f"index={len(task_ids)}"
        )
    if metadata.get("counts", {}).get("tasks") != len(task_ids):
        raise ValueError("metadata counts.tasks does not match task index")
    if metadata.get("packaging_complete") is not True:
        raise ValueError("bundle metadata does not declare packaging_complete")
    if metadata.get("task_bindings_complete") is not True:
        raise ValueError("bundle metadata does not declare task_bindings_complete")
    if metadata.get("tasks_with_packaging_issues"):
        raise ValueError("bundle metadata reports packaging issues")
    if metadata.get("deployment_integration_complete") is not True:
        raise ValueError(
            "bundle metadata does not declare deployment_integration_complete"
        )
    scope_sha = metadata.get("task_manifest_sha256")
    if not isinstance(scope_sha, str) or _sha256(audit_path) != scope_sha:
        raise ValueError("audit manifest hash does not match bundle metadata")
    if index.get("scope_sha256") != scope_sha:
        raise ValueError("task index scope hash does not match audit manifest")

    audit_rows = audit.get("tasks")
    if not isinstance(audit_rows, list):
        raise ValueError("audit_manifest.json has no tasks list")
    calibration: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        if not isinstance(row, dict):
            raise ValueError("audit manifest contains a non-object task")
        task_id = row.get("task_id")
        successes = row.get("calibration_successes")
        reps = row.get("calibration_reps")
        if not isinstance(task_id, str) or task_id in calibration:
            raise ValueError(f"invalid/duplicate audit task_id: {task_id!r}")
        if (
            row.get("in_training") is not True
            or reps != 8
            or not isinstance(successes, int)
            or not 0 < successes < reps
        ):
            raise ValueError(f"{task_id}: task is outside the Offline-DAPO band")
        calibration[task_id] = {
            "calibration_successes": successes,
            "calibration_reps": reps,
            "calib_pass_rate": successes / reps,
            "round": row.get("round"),
            "category": row.get("category"),
        }
    if set(calibration) != set(task_ids):
        raise ValueError("audit manifest and task index task sets differ")
    reward_policy = metadata.get("reward_policy")
    if not isinstance(reward_policy, dict):
        raise ValueError("bundle metadata has no reward policy")
    if reward_policy.get("allowed") != [0, 1]:
        raise ValueError("bundle reward policy is not binary")
    if reward_policy.get("full_success_only") is not True:
        raise ValueError("bundle reward policy permits partial success")
    if reward_policy.get("errors_propagate") is not True:
        raise ValueError("bundle reward policy does not propagate errors")
    reward_module = source / "binary_reward.py"
    expected_reward_sha = metadata.get("binary_reward_module_sha256")
    if (
        not isinstance(expected_reward_sha, str)
        or _sha256(reward_module) != expected_reward_sha
    ):
        raise ValueError("binary reward module hash does not match metadata")
    return task_ids, metadata, calibration


def _load_authoritative_task(task_root: Path, task_id: str) -> dict[str, Any]:
    route_manifest = _load_object(task_root / "import_routes_private.json")
    relative_value = route_manifest.get("task_config")
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{task_id}: route manifest has no authoritative task config")
    task_path = (task_root / relative_value).resolve()
    if not task_path.is_relative_to(task_root) or task_path.suffix != ".json":
        raise ValueError(f"{task_id}: authoritative task config escapes task root")
    if not task_path.is_file():
        raise FileNotFoundError(
            f"{task_id}: authoritative task config missing: {relative_value}"
        )

    if relative_value != "frozen_read/task/task.json":
        deployment = _load_object(task_root / "deployment_private.json")
        candidates = deployment.get("candidate_files")
        if not isinstance(candidates, list):
            raise ValueError(f"{task_id}: deployment manifest has no candidate files")
        match = next(
            (
                row
                for row in candidates
                if isinstance(row, dict) and row.get("packaged_file") == relative_value
            ),
            None,
        )
        if match is None or match.get("sha256") != _sha256(task_path):
            raise ValueError(
                f"{task_id}: accepted task config is not hash-bound by deployment manifest"
            )
    return _load_object(task_path)


def _module_path_depth(
    node: ast.AST,
    aliases: dict[str, int],
) -> int | None:
    """Return the number of parents from ``__file__`` represented by *node*."""
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return 0
        return aliases.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
    ):
        return _module_path_depth(node.args[0], aliases)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "resolve" and not node.args and not node.keywords:
            return _module_path_depth(node.func.value, aliases)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        depth = _module_path_depth(node.value, aliases)
        return None if depth is None else depth + 1
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and node.slice.value >= 0
    ):
        depth = _module_path_depth(node.value.value, aliases)
        return None if depth is None else depth + node.slice.value + 1
    return None


def _task_root_aliases(tree: ast.AST) -> set[str]:
    aliases: dict[str, int] = {}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(len(assignments) + 1):
        changed = False
        for assignment in assignments:
            value = assignment.value
            if value is None:
                continue
            depth = _module_path_depth(value, aliases)
            if depth is None:
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != depth:
                    aliases[target.id] = depth
                    changed = True
        if not changed:
            break
    return {name for name, depth in aliases.items() if depth == 2}


def _task_root_reference_names(tree: ast.AST) -> set[str]:
    """Find static top-level paths below ``Path(__file__).parent.parent``."""
    aliases = _task_root_aliases(tree)
    references: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id in aliases
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
        ):
            name = node.right.value
            path = Path(name)
            if (
                name
                and not path.is_absolute()
                and len(path.parts) == 1
                and path.parts[0] not in {".", ".."}
            ):
                references.add(name)
    return references


def _copy_task_root_references(
    *,
    tree: ast.AST,
    route_path: Path,
    destination_task_root: Path,
    task_id: str,
    route_name: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_task_root = route_path.parent.parent
    support: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for name in sorted(_task_root_reference_names(tree)):
        source = source_task_root / name
        if not source.exists():
            if name in _TASK_ROOT_REFERENCE_DIRS or Path(name).suffix:
                missing.append(
                    {
                        "task_id": task_id,
                        "route": route_name,
                        "source_file": str(route_path),
                        "missing_path": str(source),
                    }
                )
            continue
        files = [source] if source.is_file() else sorted(source.rglob("*"))
        for dependency in files:
            if not dependency.is_file():
                continue
            relative = dependency.relative_to(source_task_root)
            target = destination_task_root / relative
            digest = _sha256(dependency)
            if target.exists() and _sha256(target) != digest:
                raise ValueError(
                    f"{task_id}: conflicting evaluator support file {relative}"
                )
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dependency, target)
            support.append(
                {
                    "file": str(dependency),
                    "sha256": digest,
                    "snapshot_file": str(relative),
                }
            )
    return support, missing


def _copy_candidate_support_closure(
    *,
    route_path: Path,
    destination_root: Path,
    task_id: str,
    route_name: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Copy task-local Python/data dependencies used by one accepted route."""
    pending = [route_path]
    visited: set[Path] = set()
    support: dict[Path, dict[str, str]] = {}
    task_root_support: dict[Path, dict[str, str]] = {}
    missing: dict[str, dict[str, str]] = {}

    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        task_support, task_missing = _copy_task_root_references(
            tree=tree,
            route_path=current,
            destination_task_root=destination_root.parent,
            task_id=task_id,
            route_name=route_name,
        )
        for record in task_support:
            task_root_support[Path(record["file"])] = record
        for record in task_missing:
            missing[f"{record['source_file']}:{record['missing_path']}"] = record
        local_imports: set[Path] = set()
        data_files: set[Path] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_path = Path(*node.module.split("."))
                candidate = current.parent / f"{module_path}.py"
                if candidate.is_file():
                    local_imports.add(candidate.resolve())
                elif node.level:
                    missing[f"{current}:{node.lineno}:{node.module}"] = {
                        "task_id": task_id,
                        "route": route_name,
                        "source_file": str(current),
                        "missing_module": node.module,
                    }
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = current.parent / f"{alias.name}.py"
                    if candidate.is_file():
                        local_imports.add(candidate.resolve())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if "/" not in value and Path(value).suffix in {
                    ".json",
                    ".txt",
                    ".bin",
                    ".npy",
                    ".npz",
                }:
                    candidate = current.parent / value
                    if candidate.is_file():
                        data_files.add(candidate.resolve())

        for dependency in sorted(local_imports):
            if dependency == route_path:
                continue
            record = {
                "file": str(dependency),
                "sha256": _sha256(dependency),
            }
            support[dependency] = record
            pending.append(dependency)
        for dependency in sorted(data_files):
            support[dependency] = {
                "file": str(dependency),
                "sha256": _sha256(dependency),
            }

    for dependency, record in sorted(support.items(), key=lambda item: str(item[0])):
        target = destination_root / dependency.name
        if target.exists() and _sha256(target) != record["sha256"]:
            raise ValueError(
                f"{task_id}: conflicting evaluator support file {target.name}"
            )
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dependency, target)
        record["snapshot_file"] = target.name
    return [
        *support.values(),
        *task_root_support.values(),
    ], list(missing.values())


def _copy_selected_routes(
    *,
    source: Path,
    stage: Path,
    final: Path,
    task_id: str,
    task: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    task_root = (source / "tasks" / task_id).resolve()
    route_manifest = _load_object(task_root / "import_routes_private.json")
    if route_manifest.get("task_id") != task_id:
        raise ValueError(f"{task_id}: route manifest task_id mismatch")
    if route_manifest.get("issues") != []:
        raise ValueError(f"{task_id}: route manifest reports issues")
    if route_manifest.get("binary_final_reward") is not True:
        raise ValueError(f"{task_id}: route manifest is not binary-final-reward")

    routes = route_manifest.get("routes")
    if not isinstance(routes, dict) or not routes:
        raise ValueError(f"{task_id}: no evaluator routes")

    copied: dict[tuple[str, str], str] = {}
    route_records: list[dict[str, Any]] = []
    support_records: list[dict[str, str]] = []
    missing_dependencies: list[dict[str, str]] = []
    provenance_paths: list[str] = []
    for route_name, route in sorted(routes.items()):
        if not isinstance(route_name, str) or ":" not in route_name:
            raise ValueError(f"{task_id}: invalid route name {route_name!r}")
        role, routed_symbol = route_name.split(":", 1)
        if role not in _ALLOWED_ROUTE_ROLES:
            raise ValueError(f"{task_id}: unsupported evaluator role {role!r}")
        if not isinstance(route, dict):
            raise ValueError(f"{task_id}: route {route_name} is not an object")
        if route.get("function") != routed_symbol:
            raise ValueError(f"{task_id}: route/function mismatch for {route_name}")
        if route.get("source") not in _ALLOWED_ROUTE_SOURCES:
            raise ValueError(f"{task_id}: unsupported route source for {route_name}")

        relative_value = route.get("file")
        expected_sha = route.get("sha256")
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError(f"{task_id}: route {route_name} has no file")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"{task_id}: route {route_name} has invalid sha256")
        route_path = (task_root / relative_value).resolve()
        if not route_path.is_relative_to(task_root):
            raise ValueError(f"{task_id}: route escapes task root: {relative_value}")
        if not route_path.is_file():
            raise FileNotFoundError(f"{task_id}: routed file missing: {relative_value}")
        actual_sha = _sha256(route_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{task_id}: routed file hash mismatch for {relative_value}: "
                f"{actual_sha} != {expected_sha}"
            )

        key = (role, route_path.name)
        previous_sha = copied.get(key)
        if previous_sha is not None and previous_sha != actual_sha:
            raise ValueError(
                f"{task_id}: conflicting routed modules named {role}/{route_path.name}"
            )
        destination = Path("tmp_funcs") / task_id / role / route_path.name
        if previous_sha is None:
            target = stage / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(route_path, target)
            copied[key] = actual_sha
            provenance_paths.append(str(final / destination))
            if route.get("source") == "accepted_candidate":
                support, missing = _copy_candidate_support_closure(
                    route_path=route_path,
                    destination_root=target.parent,
                    task_id=task_id,
                    route_name=route_name,
                )
                support_records.extend(support)
                missing_dependencies.extend(missing)
        route_records.append(
            {
                "route": route_name,
                "file": relative_value,
                "source": route["source"],
                "sha256": actual_sha,
            }
        )

    metadata = task.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{task_id}: task metadata must be an object")
    metadata["new_functions_paths"] = sorted(provenance_paths)
    metadata["evaluator_routes"] = route_records
    metadata["evaluator_support_files"] = support_records
    metadata["binary_final_reward"] = True
    metadata["binary_reward_rule"] = "int(score == 1.0); evaluator errors are masked"
    return route_records, support_records, missing_dependencies


def _configured_asset_fixes(task: dict[str, Any]) -> dict[str, Path]:
    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    fixes = metadata.get("upload_path_fixes")
    if not isinstance(fixes, list):
        return {}
    result: dict[str, Path] = {}
    for fix in fixes:
        if not isinstance(fix, dict):
            continue
        source = fix.get("original") or fix.get("from") or fix.get("old")
        target = fix.get("replacement") or fix.get("to") or fix.get("new")
        if isinstance(source, str) and isinstance(target, str):
            result[source] = Path(target).expanduser()
    return result


def _resolve_upload_asset(
    *,
    task_root: Path,
    configured_path: str,
    configured_fixes: dict[str, Path],
) -> Path:
    original = Path(configured_path).expanduser()
    candidates = [configured_fixes.get(configured_path), original]
    for relative_root in (
        ("frozen_read", "reference", "uploads"),
        ("frozen_read", "task", "uploads"),
        ("frozen_read", "assets"),
        ("frozen_read", "evaluators", "uploads"),
        ("work", "uploads"),
        ("uploads",),
    ):
        candidates.append(task_root.joinpath(*relative_root, original.name))

    attempted: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        attempted.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"upload asset not found for {configured_path!r}; tried: "
        + ", ".join(attempted)
    )


def _copy_upload_assets(
    *,
    task: dict[str, Any],
    task_id: str,
    task_root: Path,
    stage: Path,
    final: Path,
    global_assets: dict[str, Path],
    missing_assets: list[dict[str, str]],
) -> int:
    copied_for_task: set[str] = set()
    fixes = _configured_asset_fixes(task)
    for entry in _upload_entries(task):
        configured_path = entry.get("local_path")
        if not isinstance(configured_path, str) or not configured_path:
            raise ValueError(f"{task_id}: invalid upload_file.local_path")
        try:
            source = _resolve_upload_asset(
                task_root=task_root,
                configured_path=configured_path,
                configured_fixes=fixes,
            )
        except FileNotFoundError as error:
            missing_assets.append(
                {
                    "task_id": task_id,
                    "configured_path": configured_path,
                    "error": str(error),
                }
            )
            continue
        digest = _sha256(source)
        destination = global_assets.get(digest)
        if destination is None:
            destination = Path("assets") / digest[:2] / f"{digest}-{source.name}"
            target = stage / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            global_assets[digest] = destination
        entry["local_path"] = str(final / destination)
        copied_for_task.add(digest)
    return len(copied_for_task)


def freeze_astra_bundle(
    source: Path,
    output: Path,
    *,
    num_repeats: int,
    audit_report: Path | None = None,
    quarantine_missing_assets: bool = False,
) -> None:
    source = source.resolve()
    output = output.resolve()
    if num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    task_ids, bundle_metadata, calibration = _validated_task_ids(source)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (stage / "tasks").mkdir()
        (stage / "tmp_funcs").mkdir()
        (stage / "assets").mkdir()
        provenance = stage / "source_bundle_metadata"
        provenance.mkdir()
        for name in (
            "metadata.json",
            "audit_manifest.json",
            "task_index.json",
            "ACCEPTANCE_POLICY.json",
            "binary_reward.py",
        ):
            shutil.copy2(source / name, provenance / name)
        tasks: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        global_assets: dict[str, Path] = {}
        missing_assets: list[dict[str, str]] = []
        missing_dependencies: list[dict[str, str]] = []
        task_asset_references = 0
        runtime_command_repairs = 0
        quarantined_rows: list[dict[str, Any]] = []
        support_file_records = 0

        for task_id in task_ids:
            task_root = (source / "tasks" / task_id).resolve()
            deployment = _load_object(task_root / "deployment_private.json")
            model_outcome = deployment.get("model_outcome")
            if model_outcome != "revised_tested":
                quarantined_rows.append(
                    {
                        "task_id": task_id,
                        "reason": "evaluator_not_fully_resolved",
                        "model_outcome": model_outcome,
                    }
                )
                continue
            task = _load_authoritative_task(task_root, task_id)
            if task.get("id") != task_id:
                raise ValueError(f"{task_id}: authoritative task JSON id mismatch")
            task_metadata = task.setdefault("metadata", {})
            if not isinstance(task_metadata, dict):
                raise ValueError(f"{task_id}: task metadata must be an object")
            task_metadata.update(calibration[task_id])
            runtime_command_repairs += _repair_runtime_commands(task)
            route_records, support_records, task_missing_dependencies = (
                _copy_selected_routes(
                    source=source,
                    stage=stage,
                    final=output,
                    task_id=task_id,
                    task=task,
                )
            )
            support_file_records += len(support_records)
            missing_dependencies.extend(task_missing_dependencies)
            missing_before = len(missing_assets)
            task_asset_references += _copy_upload_assets(
                task=task,
                task_id=task_id,
                task_root=task_root,
                stage=stage,
                final=output,
                global_assets=global_assets,
                missing_assets=missing_assets,
            )
            task_missing_assets = missing_assets[missing_before:]
            if (
                task_missing_assets or task_missing_dependencies
            ) and quarantine_missing_assets:
                shutil.rmtree(
                    stage / "tmp_funcs" / task_id,
                    ignore_errors=True,
                )
                quarantined_rows.append(
                    {
                        "task_id": task_id,
                        "reason": "missing_runtime_dependencies",
                        "missing_references": [
                            row["configured_path"] for row in task_missing_assets
                        ],
                        "missing_modules": [
                            row["missing_module"]
                            for row in task_missing_dependencies
                            if "missing_module" in row
                        ],
                        "missing_paths": [
                            row["missing_path"]
                            for row in task_missing_dependencies
                            if "missing_path" in row
                        ],
                    }
                )
                continue
            (stage / "tasks" / f"{task_id}.json").write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tasks.append(task)
            manifest_rows.append(
                {
                    "task_id": task_id,
                    **calibration[task_id],
                    "binary_final_reward": True,
                    "routes": route_records,
                }
            )

        if (missing_assets or missing_dependencies) and audit_report is not None:
            audit_report.parent.mkdir(parents=True, exist_ok=True)
            audit_report.write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "missing_upload_assets": missing_assets,
                        "missing_references": len(missing_assets),
                        "affected_tasks": len(
                            {
                                row["task_id"]
                                for row in missing_assets + missing_dependencies
                            }
                        ),
                        "missing_evaluator_dependencies": missing_dependencies,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if (missing_assets or missing_dependencies) and not quarantine_missing_assets:
            raise FileNotFoundError(
                f"{len(missing_assets)} upload asset references and "
                f"{len(missing_dependencies)} evaluator dependencies are missing across "
                f"{len({row['task_id'] for row in missing_assets + missing_dependencies})} tasks"
                + (
                    f"; report: {audit_report.resolve()}"
                    if audit_report is not None
                    else ""
                )
            )

        _write_jsonl(stage / "manifest.jsonl", manifest_rows)
        if quarantined_rows:
            _write_jsonl(stage / "quarantine.jsonl", quarantined_rows)
        row_count = _write_jsonl(
            stage / "train.jsonl",
            (
                _gym_row(task, repeat_index)
                for repeat_index in range(num_repeats)
                for task in tasks
            ),
        )
        snapshot_metadata = {
            "schema_version": 1,
            "source": str(source),
            "source_task_manifest_sha256": bundle_metadata[
                "task_manifest_sha256"
            ],
            "source_bundle_created_at": bundle_metadata.get("created_at"),
            "filter": "Offline DAPO: calibration_reps == 8 and 0 < successes < 8",
            "reward": "binary exact full success: score == 1.0",
            "infrastructure_errors": "masked from optimization via verify_error",
            "unique_tasks": len(tasks),
            "source_tasks": len(task_ids),
            "quarantined_tasks": len(quarantined_rows),
            "num_repeats": num_repeats,
            "gym_rows": row_count,
            "selected_route_records": sum(
                len(row["routes"]) for row in manifest_rows
            ),
            "evaluator_support_file_records": support_file_records,
            "unique_upload_assets": len(global_assets),
            "task_upload_asset_references": task_asset_references,
            "runtime_command_repairs": runtime_command_repairs,
        }
        (stage / "SNAPSHOT.json").write_text(
            json.dumps(snapshot_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_checksums(stage)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        f"Frozen {len(tasks)} accepted tasks / {row_count} Gym rows / "
        f"{len(global_assets)} unique upload assets at {output}; "
        f"quarantined {len(quarantined_rows)} tasks"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Write all missing upload-asset references before failing.",
    )
    parser.add_argument(
        "--quarantine-missing-assets",
        action="store_true",
        help="Exclude tasks with missing uploads and record them in quarantine.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_astra_bundle(
        args.source,
        args.output,
        num_repeats=args.num_repeats,
        audit_report=args.audit_report,
        quarantine_missing_assets=args.quarantine_missing_assets,
    )


if __name__ == "__main__":
    main()
