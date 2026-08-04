# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import fcntl
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import ray
from ray.util import placement_group

dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))
DEFAULT_VENV_DIR = os.path.join(git_root, "venvs")

logger = logging.getLogger(__name__)
VENV_READY_FILENAME = ".nemo_rl_venv_ready"
DEFAULT_UV_HTTP_TIMEOUT_SECONDS = "300"
DEFAULT_UV_LOCK_TIMEOUT_SECONDS = "1800"


def _expected_venv_marker(force_rebuild: bool) -> str:
    fingerprint = os.environ.get("NRL_VENV_FINGERPRINT", "unversioned")
    if not force_rebuild:
        return f"{fingerprint}:reusable"
    rebuild_token = (
        os.environ.get("NRL_VENV_REBUILD_TOKEN")
        or os.environ.get("SLURM_JOB_ID")
        or "force"
    )
    return f"{fingerprint}:force:{rebuild_token}"


def _is_venv_ready(venv_path: Path, expected_marker: str) -> bool:
    python_path = venv_path / "bin" / "python"
    ready_file = venv_path / VENV_READY_FILENAME
    if not python_path.exists() or not os.access(python_path, os.X_OK):
        return False
    try:
        return ready_file.read_text().strip() == expected_marker
    except OSError:
        return False


def _sync_failure_preflight_modules(py_executable: str) -> tuple[str, ...]:
    """Return the minimum imports that prove a partially synced actor env is usable."""
    if "--extra vllm" in py_executable:
        return ("vllm", "torch", "ray", "flashinfer", "flashinfer_cubin")
    if "--extra mcore" in py_executable:
        return (
            "megatron.core",
            "megatron.bridge",
            "torch",
            "ray",
            "antlr4.StdinStream",
        )
    if "--extra nemo_gym" in py_executable:
        return ("nemo_gym", "omegaconf", "ray", "torch.hub", "torchvision")
    return ()


def _run_venv_preflight(py_executable: str, python_path: Path) -> bool:
    """Verify imports that previously failed only after expensive actor startup."""
    modules = _sync_failure_preflight_modules(py_executable)
    if not modules:
        return True
    module_list = repr(modules)
    checks = [
        "import importlib",
        f"modules = {module_list}",
        "[importlib.import_module(module) for module in modules]",
    ]
    if "--extra vllm" in py_executable:
        checks.extend(
            [
                "flashinfer_cubin = importlib.import_module('flashinfer_cubin')",
                "assert getattr(flashinfer_cubin, '__version__', None), "
                "'flashinfer_cubin has no __version__'",
            ]
        )
    preflight = subprocess.run(
        [str(python_path), "-c", "; ".join(checks)],
        check=False,
        capture_output=True,
        text=True,
    )
    if preflight.returncode != 0:
        logger.error(
            "Actor environment preflight failed for %s:\n%s%s",
            python_path,
            preflight.stdout,
            preflight.stderr,
        )
        return False
    return True


def _uv_subprocess_env(venv_path: str) -> dict[str, str]:
    """Build a robust uv environment for concurrent multi-node actor installs."""
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = venv_path
    is_osworld_env = bool(
        os.environ.get("GRPO_ENV_FINGERPRINT")
        and os.environ.get("OSWORLD_POOL_REF")
    )
    if is_osworld_env:
        # Match the previously stable full-training path: actor and Gym service
        # venvs install from the container's pre-populated /root/.cache/uv.
        # The new shared fingerprint cache is retained for the single driver
        # venv only; using it from every node caused lock contention and cold
        # Transformer Engine source builds.
        actor_cache_dir = os.environ.get("NRL_ACTOR_UV_CACHE_DIR", "")
        if actor_cache_dir:
            env["UV_CACHE_DIR"] = actor_cache_dir
        else:
            env.pop("UV_CACHE_DIR", None)
    # OSWorld launches one installer per node and may launch two experiments at
    # once against a shared download/build cache. Source builds can hold a uv
    # cache lock for longer than the 300s default; waiting is correct because
    # the completed artifact is reused by all remaining node-local installs.
    env.setdefault("UV_LOCK_TIMEOUT", DEFAULT_UV_LOCK_TIMEOUT_SECONDS)
    env.setdefault("UV_HTTP_TIMEOUT", DEFAULT_UV_HTTP_TIMEOUT_SECONDS)
    # The shared cache is on Lustre while actor venvs are node-local /tmp.
    # Explicit copies avoid cross-filesystem link behavior.
    env.setdefault("UV_LINK_MODE", "copy")
    # The OSWorld production pool is H100-only. CUDA 13 otherwise makes
    # Transformer Engine compile six architectures, which exceeds the
    # cluster's GPU-idle grace period before policy startup.
    if is_osworld_env:
        env.setdefault("NVTE_CUDA_ARCHS", "90")
        env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    return env


@lru_cache(maxsize=None)
def create_local_venv(
    py_executable: str, venv_name: str, force_rebuild: bool = False
) -> str:
    """Create a virtual environment using uv and execute a command within it.

    The output can be used as a py_executable for a Ray worker assuming the worker
    nodes also have access to the same file system as the head node.

    This function is cached to avoid multiple calls to uv to create the same venv,
    which avoids duplicate logging.

    Args:
        py_executable (str): Command to run with the virtual environment (e.g., "uv.sh run --locked")
        venv_name (str): Name of the virtual environment (e.g., "foobar.Worker")
        force_rebuild (bool): If True, force rebuild the venv even if it already exists

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # This directory is where virtual environments will be installed
    # It is local to the driver process but should be visible to all worker nodes
    # If this directory is not accessible from worker nodes (e.g., on a distributed
    # cluster with non-shared filesystems), you may encounter errors when workers
    # try to access the virtual environments
    #
    # You can override this location by setting the NEMO_RL_VENV_DIR environment variable

    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    logger.info(f"NEMO_RL_VENV_DIR is set to {NEMO_RL_VENV_DIR}.")

    # Create the venv directory if it doesn't exist
    os.makedirs(NEMO_RL_VENV_DIR, exist_ok=True)

    # Full path to the virtual environment
    venv_path = os.path.join(NEMO_RL_VENV_DIR, venv_name)
    ready_file = Path(venv_path) / VENV_READY_FILENAME

    # Force rebuild if requested
    if force_rebuild and os.path.exists(venv_path):
        logger.info(f"Force rebuilding venv at {venv_path}")
        shutil.rmtree(venv_path)
    else:
        ready_file.unlink(missing_ok=True)

    logger.info(f"Creating new venv at {venv_path}")

    # Create the virtual environment
    uv_venv_cmd = ["uv", "venv", "--allow-existing", venv_path]
    subprocess.run(uv_venv_cmd, check=True)

    # Execute the command with the virtual environment
    # NOTE: UV_PROJECT_ENVIRONMENT is appropriate here only b/c there should only be
    #  one call to this in the driver. It is not safe to use this in a multi-process
    #  context.
    #  https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
    env = _uv_subprocess_env(venv_path)

    # Split the py_executable into command and arguments
    exec_cmd = shlex.split(py_executable)
    # Command doesn't matter, since `uv` syncs the environment no matter the command.
    exec_cmd.extend(["echo", f"Finished creating venv {venv_path}"])

    # Always run uv sync first to ensure the build requirements are set (for --no-build-isolation packages)
    subprocess.run(["uv", "sync", "--directory", git_root], env=env, check=True)
    try:
        subprocess.run(exec_cmd, env=env, check=True)
    except subprocess.CalledProcessError as sync_error:
        is_fingerprinted_osworld_env = "osworld-grpo-envs" in os.path.normpath(
            os.environ.get("NEMO_RL_VENV_DIR", "")
        )
        allow_preflight_fallback = (
            os.environ.get(
                "NRL_ALLOW_PARTIAL_VENV_ON_SYNC_FAILURE",
                str(is_fingerprinted_osworld_env),
            ).lower()
            == "true"
        )
        python_path = Path(venv_path) / "bin" / "python"
        if not allow_preflight_fallback or not _run_venv_preflight(
            py_executable, python_path
        ):
            raise sync_error
        logger.warning(
            "Actor environment sync failed, but required imports passed; "
            "continuing without optional packages for %s",
            venv_name,
        )

    # Return the path to the python executable in the virtual environment
    python_path = os.path.join(venv_path, "bin", "python")
    if not _run_venv_preflight(py_executable, Path(python_path)):
        raise RuntimeError(f"Actor environment failed preflight: {venv_path}")
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    marker_tmp = ready_file.with_name(f"{ready_file.name}.{os.getpid()}.tmp")
    marker_tmp.write_text(_expected_venv_marker(force_rebuild))
    os.replace(marker_tmp, ready_file)
    return python_path


# Ray-based helper to create a virtual environment on each Ray node
@ray.remote(num_cpus=1)  # pragma: no cover
def _env_builder(
    py_executable: str, venv_name: str, node_idx: int, force_rebuild: bool = False
):
    NEMO_RL_VENV_DIR = os.path.normpath(
        os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    )
    venv_root = Path(NEMO_RL_VENV_DIR)
    venv_path = Path(NEMO_RL_VENV_DIR) / venv_name
    expected_marker = _expected_venv_marker(force_rebuild)

    # A shared filesystem lock is released by the kernel even if a Slurm segment
    # is killed. This avoids stale STARTED files and prevents multiple Ray nodes
    # or concurrent jobs from mutating the same persistent environment.
    lock_dir = venv_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(venv_name.encode()).hexdigest()
    lock_path = lock_dir / f"{lock_name}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if _is_venv_ready(venv_path, expected_marker):
            logger.info(f"Node {node_idx}: Using validated venv at {venv_path}")
            return str(venv_path / "bin" / "python")

        logger.info(f"Node {node_idx}: Building or repairing venv at {venv_path}")
        return create_local_venv(
            py_executable, venv_name, force_rebuild=force_rebuild
        )


def create_local_venv_on_each_node(py_executable: str, venv_name: str):
    """Create a virtual environment on each Ray node.

    Args:
        py_executable (str): Command to run with the virtual environment
        venv_name (str): Name of the virtual environment

    Returns:
        str: Path to the python executable in the created virtual environment
    """
    # Skip nodes with 0 CPUs (e.g. unschedulable head nodes) — including them
    # makes the STRICT_SPREAD placement group infeasible.
    nodes = [
        n
        for n in ray.nodes()
        if n.get("Alive", False) and n.get("Resources", {}).get("CPU", 0) > 0
    ]
    num_nodes = len(nodes)
    # Reserve one CPU on each node using a STRICT_SPREAD placement group
    bundles = [{"CPU": 1} for _ in range(num_nodes)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    force_rebuild = os.environ.get("NRL_FORCE_REBUILD_VENVS", "false").lower() == "true"
    # Launch one actor per node
    actors = [
        _env_builder.options(placement_group=pg).remote(
            py_executable, venv_name, i, force_rebuild
        )
        for i, _ in enumerate(nodes)
    ]
    # ensure setup runs on each node
    paths = ray.get([actor for actor in actors])
    # Normalize paths to handle double slashes and other path inconsistencies
    normalized_paths = [os.path.normpath(p) for p in paths]
    assert len(set(normalized_paths)) == 1, (
        f"All nodes should have the same venv, but got: {set(normalized_paths)}"
    )

    # Clean up the placement group
    ray.util.remove_placement_group(pg)
    # Return mapping from node IP to venv python path
    return paths[0]


def make_actor_runtime_env(actor_class_fqn: str) -> dict:
    """Build a Ray ``runtime_env`` for one of our registered actors.

    Resolves the actor's tier-specific py_executable via the registry,
    materializes a per-node venv when uv-managed, and packages it with
    ``VIRTUAL_ENV`` / ``UV_PROJECT_ENVIRONMENT`` env vars so workers see
    the same interpreter as the driver.

    Used by ReplayBuffer, AsyncTrajectoryCollector, and SyncRolloutActor
    — three actors that need the VLLM tier's venv on every node. Also
    used by the SGLang router and SGLang generation engines (SGLANG tier).
    """
    # Local import — venvs.py is dep-light; the registry imports
    # PY_EXECUTABLES which transitively pulls heavier deps.
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )

    py_exec = get_actor_python_env(actor_class_fqn)
    if py_exec.startswith("uv"):
        py_exec = create_local_venv_on_each_node(py_exec, actor_class_fqn)
    venv = os.path.dirname(os.path.dirname(py_exec))  # strip bin/python
    return {
        "py_executable": py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": venv,
            "UV_PROJECT_ENVIRONMENT": venv,
        },
    }
