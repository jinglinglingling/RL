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
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nemo_rl.utils.venvs import (
    DEFAULT_UV_HTTP_TIMEOUT_SECONDS,
    DEFAULT_UV_LOCK_TIMEOUT_SECONDS,
    VENV_READY_FILENAME,
    _expected_venv_marker,
    _is_venv_ready,
    _uv_subprocess_env,
    create_local_venv,
)
from tests.unit.conftest import TEST_ASSETS_DIR


def test_create_local_venv():
    # The temporary directory is created within the project.
    # For some reason, creating a virtual environment outside of the project
    # doesn't work reliably.
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        # Mock os.environ to set NEMO_RL_VENV_DIR for this test
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_python = create_local_venv(
                py_executable="uv run --group docs", venv_name="test_venv"
            )
            assert os.path.exists(venv_python)
            assert venv_python == f"{tempdir}/test_venv/bin/python"
            # Check if sphinx package is installed in the created venv

            # Run a Python command to check if sphinx can be imported
            result = subprocess.run(
                [
                    venv_python,
                    "-c",
                    "import sphinx; print('Sphinx package is installed')",
                ],
                capture_output=True,
                text=True,
            )

            # Verify the command executed successfully (return code 0)
            assert result.returncode == 0, f"Failed to import sphinx: {result.stderr}"
            assert "Sphinx package is installed" in result.stdout
            assert (
                Path(tempdir, "test_venv", VENV_READY_FILENAME).read_text()
                == "unversioned:reusable"
            )


def test_venv_readiness_requires_matching_completion_marker(tmp_path):
    venv_path = tmp_path / "actor"
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(0o755)

    with patch.dict(
        os.environ, {"NRL_VENV_FINGERPRINT": "lock-123"}, clear=False
    ):
        marker = _expected_venv_marker(force_rebuild=False)
        assert not _is_venv_ready(venv_path, marker)

        (venv_path / VENV_READY_FILENAME).write_text("stale:reusable")
        assert not _is_venv_ready(venv_path, marker)

        (venv_path / VENV_READY_FILENAME).write_text(marker)
        assert _is_venv_ready(venv_path, marker)


def test_forced_rebuild_marker_is_shared_within_slurm_job():
    with patch.dict(
        os.environ,
        {
            "NRL_VENV_FINGERPRINT": "lock-123",
            "SLURM_JOB_ID": "456",
        },
        clear=False,
    ):
        assert _expected_venv_marker(force_rebuild=True) == "lock-123:force:456"


def test_uv_subprocess_env_uses_multi_node_safe_defaults():
    with patch.dict(os.environ, {}, clear=True):
        env = _uv_subprocess_env("/tmp/actor")

    assert env["UV_PROJECT_ENVIRONMENT"] == "/tmp/actor"
    assert env["UV_LOCK_TIMEOUT"] == DEFAULT_UV_LOCK_TIMEOUT_SECONDS
    assert env["UV_HTTP_TIMEOUT"] == DEFAULT_UV_HTTP_TIMEOUT_SECONDS
    assert env["UV_LINK_MODE"] == "copy"


def test_uv_subprocess_env_preserves_explicit_overrides():
    with patch.dict(
        os.environ,
        {
            "UV_LOCK_TIMEOUT": "42",
            "UV_HTTP_TIMEOUT": "43",
            "UV_LINK_MODE": "clone",
        },
        clear=True,
    ):
        env = _uv_subprocess_env("/tmp/actor")

    assert env["UV_LOCK_TIMEOUT"] == "42"
    assert env["UV_HTTP_TIMEOUT"] == "43"
    assert env["UV_LINK_MODE"] == "clone"


def test_uv_subprocess_env_limits_osworld_builds_to_h100():
    with patch.dict(
        os.environ,
        {
            "GRPO_ENV_FINGERPRINT": "fingerprint",
            "OSWORLD_POOL_REF": "osworld-kvm",
            "UV_CACHE_DIR": "/shared/cold-cache",
        },
        clear=True,
    ):
        env = _uv_subprocess_env("/tmp/actor")

    assert env["NVTE_CUDA_ARCHS"] == "90"
    assert env["TORCH_CUDA_ARCH_LIST"] == "9.0"
    assert "UV_CACHE_DIR" not in env


def test_uv_subprocess_env_allows_explicit_osworld_actor_cache():
    with patch.dict(
        os.environ,
        {
            "GRPO_ENV_FINGERPRINT": "fingerprint",
            "OSWORLD_POOL_REF": "osworld-kvm",
            "NRL_ACTOR_UV_CACHE_DIR": "/tmp/actor-cache",
            "UV_CACHE_DIR": "/shared/driver-cache",
        },
        clear=True,
    ):
        env = _uv_subprocess_env("/tmp/actor")

    assert env["UV_CACHE_DIR"] == "/tmp/actor-cache"
