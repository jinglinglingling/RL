#!/usr/bin/env bash
# Materialize one immutable, fingerprinted driver environment on shared storage.
set -euo pipefail

: "${ROOT:?Set ROOT to the NeMo-RL checkout}"
: "${GRPO_ROOT_VENV:?Set GRPO_ROOT_VENV}"
: "${GRPO_ENV_FINGERPRINT:?Set GRPO_ENV_FINGERPRINT}"
: "${UV_RUNTIME_CACHE_DIR:?Set UV_RUNTIME_CACHE_DIR}"

READY_FILE="${GRPO_ROOT_VENV}/.nemo_rl_env_ready"
LOCK_FILE="${GRPO_ROOT_VENV}.lock"
EXPECTED_MARKER="${GRPO_ENV_FINGERPRINT}"

mkdir -p "$(dirname "${GRPO_ROOT_VENV}")" "${UV_RUNTIME_CACHE_DIR}"

(
  flock -x 9

  if [[ -x "${GRPO_ROOT_VENV}/bin/python" ]] \
    && [[ -f "${READY_FILE}" ]] \
    && [[ "$(<"${READY_FILE}")" == "${EXPECTED_MARKER}" ]] \
    && "${GRPO_ROOT_VENV}/bin/python" -c \
      'from transformers import AutoProcessor; import fastapi, ray, starlette, transformers, wandb' \
      >/dev/null 2>&1; then
    echo "Reusing validated GRPO root environment: ${GRPO_ROOT_VENV}"
    exit 0
  fi

  echo "Building persistent GRPO root environment: ${GRPO_ROOT_VENV}"
  rm -f "${READY_FILE}"
  export UV_PROJECT_ENVIRONMENT="${GRPO_ROOT_VENV}"
  export VIRTUAL_ENV="${GRPO_ROOT_VENV}"
  export UV_CACHE_DIR="${UV_RUNTIME_CACHE_DIR}"
  export PATH="${GRPO_ROOT_VENV}/bin:${PATH}"

  cd "${ROOT}"
  uv sync --locked
  uv pip install --python "${GRPO_ROOT_VENV}/bin/python" \
    --reinstall 'ray[default]==2.55.1'
  if [[ "${WANDB_ENABLED:-false}" == "true" ]]; then
    env -u UV_OVERRIDE uv pip install --python "${GRPO_ROOT_VENV}/bin/python" \
      --reinstall 'wandb==0.21.0' 'protobuf==6.33.5'
  fi

  "${GRPO_ROOT_VENV}/bin/python" -c \
    'from transformers import AutoProcessor; import fastapi, ray, ray._private.node, starlette, transformers, wandb; print("Dependency preflight:", transformers.__version__, fastapi.__version__, starlette.__version__, ray.__version__, wandb.__version__)'

  printf '%s\n' "${EXPECTED_MARKER}" >"${READY_FILE}.tmp"
  mv "${READY_FILE}.tmp" "${READY_FILE}"
) 9>"${LOCK_FILE}"
