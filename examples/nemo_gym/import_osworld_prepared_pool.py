# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import a materialized OSWorld RLVR pool without duplicating its assets.

The prepared pool must contain ``rl_band2.jsonl``, ``task_base`` and
``tmp_funcs`` as produced by RPGA's ``build_rlvr_pool2.py``. Every task,
absolute setup/evaluator dependency, custom evaluator route and calibration
record is validated before an immutable inlined Gym dataset is published.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from freeze_osworld_rlvr_band import _gym_row, _write_checksums, _write_jsonl


def _absolute_dependencies(value: Any, *, active: bool = False):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _absolute_dependencies(
                child,
                active=active or key in {"config", "evaluator"},
            )
    elif isinstance(value, list):
        for child in value:
            yield from _absolute_dependencies(child, active=active)
    elif active and isinstance(value, str) and value.startswith("/lustre/"):
        yield Path(value)


def _defined_functions(root: Path, role: str) -> set[str]:
    names: set[str] = set()
    role_root = root / role
    if not role_root.is_dir():
        return names
    for path in sorted(role_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return names


def _required_symbols(task: dict[str, Any]) -> dict[str, set[str]]:
    evaluator = task.get("evaluator", {})
    functions = evaluator.get("func", [])
    if not isinstance(functions, list):
        functions = [functions]
    metrics = {name for name in functions if isinstance(name, str)}
    getters: set[str] = set()
    for key in ("result", "expected"):
        values = evaluator.get(key, [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            result_type = value.get("type") if isinstance(value, dict) else None
            if isinstance(result_type, str):
                getters.add(f"get_{result_type}")
    return {"getters": getters, "metrics": metrics}


def import_pool(
    source: Path,
    output: Path,
    *,
    num_repeats: int,
    rows_path: Path | None = None,
    task_base: Path | None = None,
    functions_roots: list[Path] | None = None,
    copy_functions: bool = False,
) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing snapshot: {output}")
    if num_repeats < 1:
        raise ValueError("num_repeats must be positive")

    rows_path = (rows_path or source / "rl_band2.jsonl").resolve()
    task_base = (task_base or source / "task_base").resolve()
    functions_roots = [
        path.resolve() for path in (functions_roots or [source / "tmp_funcs"])
    ]
    rows_bytes = rows_path.read_bytes()
    source_rows = [
        json.loads(line)
        for line in rows_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not source_rows:
        raise ValueError(f"empty prepared pool: {rows_path}")
    if not task_base.is_dir() or not all(root.is_dir() for root in functions_roots):
        raise FileNotFoundError(
            "prepared pool must contain task_base and all evaluator function roots"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (stage / "tasks").mkdir()
        if copy_functions:
            (stage / "tmp_funcs").mkdir()
        seen: set[str] = set()
        tasks: list[dict[str, Any]] = []
        dependency_count = 0
        custom_route_count = 0
        for row in source_rows:
            metadata = row.get("verifier_metadata")
            if not isinstance(metadata, dict):
                raise ValueError("prepared row has no verifier_metadata")
            task_id = metadata.get("example_id")
            relative_path = metadata.get("task_config_path")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("prepared row has no example_id")
            if task_id in seen:
                raise ValueError(f"duplicate prepared task: {task_id}")
            seen.add(task_id)
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{task_id}: no task_config_path")
            task_path = (task_base / relative_path).resolve()
            if not task_path.is_relative_to(task_base.resolve()) or not task_path.is_file():
                raise FileNotFoundError(f"{task_id}: task config not found: {task_path}")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("id") != task_id:
                raise ValueError(f"{task_id}: task JSON id mismatch")

            calibration = row.get("calibration", {})
            reps = calibration.get("k")
            rate = calibration.get("pass_rate")
            if reps != 8 or not isinstance(rate, (int, float)):
                raise ValueError(f"{task_id}: expected pass@8 calibration")
            passes = float(rate) * reps
            if not (0.0 < passes < reps):
                raise ValueError(f"{task_id}: outside the DAPO trainable band")

            missing = sorted(
                str(path)
                for path in set(_absolute_dependencies(task))
                if not path.exists()
            )
            if missing:
                raise FileNotFoundError(
                    f"{task_id}: missing setup/evaluator dependencies: {missing[:3]}"
                )
            dependency_count += len(set(_absolute_dependencies(task)))

            candidates = [
                root / task_id
                for root in functions_roots
                if (root / task_id).is_dir()
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"{task_id}: evaluator functions exist in multiple roots"
                )
            task_functions = candidates[0] if candidates else functions_roots[0] / task_id
            defined = {
                role: _defined_functions(task_functions, role)
                for role in ("getters", "metrics")
            }
            for role, required in _required_symbols(task).items():
                missing_custom = sorted(
                    name
                    for name in required
                    if "__" in name and name not in defined[role]
                )
                if missing_custom:
                    raise ImportError(
                        f"{task_id}: custom {role} are missing: {missing_custom}"
                    )
                custom_route_count += sum(name in defined[role] for name in required)
            if copy_functions and candidates:
                shutil.copytree(
                    task_functions,
                    stage / "tmp_funcs" / task_id,
                    copy_function=shutil.copy2,
                )

            (stage / "tasks" / f"{task_id}.json").write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tasks.append(task)

        if rows_path.read_bytes() != rows_bytes:
            raise RuntimeError("prepared pool changed during import; rerun")
        _write_jsonl(stage / "source_rows.jsonl", source_rows)
        gym_rows = _write_jsonl(
            stage / "train.jsonl",
            (
                _gym_row(task, repeat_index)
                for repeat_index in range(num_repeats)
                for task in tasks
            ),
        )
        snapshot = {
            "schema_version": 1,
            "source": str(source),
            "source_rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
            "unique_tasks": len(tasks),
            "num_repeats": num_repeats,
            "gym_rows": gym_rows,
            "absolute_dependencies": dependency_count,
            "custom_evaluator_routes": custom_route_count,
            "external_functions_roots": [str(root) for root in functions_roots],
            "functions_materialized": copy_functions,
            "external_asset_root": str(source),
        }
        (stage / "SNAPSHOT.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_checksums(stage)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        f"Imported {len(tasks)} tasks / {gym_rows} Gym rows / "
        f"{dependency_count} dependency references at {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--task-base", type=Path)
    parser.add_argument("--functions-root", action="append", type=Path)
    parser.add_argument("--copy-functions", action="store_true")
    args = parser.parse_args()
    import_pool(
        args.source,
        args.output,
        num_repeats=args.num_repeats,
        rows_path=args.rows,
        task_base=args.task_base,
        functions_roots=args.functions_root,
        copy_functions=args.copy_functions,
    )


if __name__ == "__main__":
    main()
