#!/bin/bash
set -euo pipefail

: "${OSWORLD_EVAL_RUN_ROOT:?}"
: "${OSWORLD_EVAL_CACHE_ROOT:?}"
: "${OSWORLD_RUNTIME_LIB_DIR:?}"
: "${OSWORLD_GRPO_VAL_DATA:?}"
: "${NANO_OMNI_MODEL_NAME:?}"
: "${NANO_OMNI_CHAT_TEMPLATE:?}"
: "${OPENSANDBOX_DOMAIN:?}"

export LIVE_RL_ROOT=/opt/nemo-rl
export LIVE_GYM_ROOT="$LIVE_RL_ROOT/3rdparty/Gym-workspace/Gym"
LIVE_BRIDGE_ROOT="$LIVE_RL_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"

export LIVE_DRIVER_VENV="${OSWORLD_DRIVER_VENV:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker}"
export LIVE_DRIVER_PYTHON="$LIVE_DRIVER_VENV/bin/python"
if [[ -z "${LIVE_COMPONENT_PYTHON:-}" ]]; then
  for candidate in \
    /root/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3 \
    /opt/nemo_rl_venv/bin/python \
    "$LIVE_DRIVER_PYTHON"; do
    if [[ -x "$candidate" ]]; then
      export LIVE_COMPONENT_PYTHON="$candidate"
      break
    fi
  done
fi
: "${LIVE_COMPONENT_PYTHON:?No container-owned Python runtime found}"

export PATH="$LIVE_DRIVER_VENV/bin:$PATH"
export PYTHONPATH="$LIVE_RL_ROOT:$LIVE_GYM_ROOT:$LIVE_BRIDGE_ROOT/src:$LIVE_BRIDGE_ROOT/3rdparty/Megatron-LM"
export NEMO_GYM_PYTHON="$LIVE_COMPONENT_PYTHON"
unset UV_PYTHON
if [[ -x /root/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3 ]]; then
  export UV_PYTHON_INSTALL_DIR=/root/.local/share/uv/python
  export UV_PYTHON_PREFERENCE=only-managed
else
  unset UV_PYTHON_INSTALL_DIR
  export UV_PYTHON_PREFERENCE=only-system
  export UV_PYTHON_DOWNLOADS=never
fi

for library in libopenblas.so.0 libgfortran.so.5 libquadmath.so.0 libgcc_s.so.1; do
  test -r "$OSWORLD_RUNTIME_LIB_DIR/$library"
done
export LD_LIBRARY_PATH="$OSWORLD_RUNTIME_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR="${OSWORLD_GYM_VENV_DIR:-$OSWORLD_EVAL_CACHE_ROOT/gym-venvs}"
export NEMO_GYM_EXTRA_ROOTS="$LIVE_GYM_ROOT:$LIVE_RL_ROOT"
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export NEMO_LENS_ENABLED=0
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.005
export NRL_REFIT_NUM_BUFFERS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export OSWORLD_GRPO_TRAIN_DATA="$OSWORLD_GRPO_VAL_DATA"
export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-osworld-kvm}"
export OSWORLD_MAX_STEPS=100
export OSWORLD_MAX_PARALLEL_ROLLOUTS=32
export OSWORLD_SLEEP_AFTER_EXECUTION="${OSWORLD_SLEEP_AFTER_EXECUTION:-3}"
export OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"

export WANDB_MODE=offline
export SWANLAB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${OSWORLD_HF_HOME:-$OSWORLD_EVAL_CACHE_ROOT/huggingface}"
export XDG_CACHE_HOME="$OSWORLD_EVAL_CACHE_ROOT/xdg"
export WANDB_DIR="$OSWORLD_EVAL_RUN_ROOT/wandb"
export WANDB_CACHE_DIR="$OSWORLD_EVAL_CACHE_ROOT/wandb"
export WANDB_CONFIG_DIR="$OSWORLD_EVAL_CACHE_ROOT/wandb-config"
export WANDB_DATA_DIR="$OSWORLD_EVAL_CACHE_ROOT/wandb-data"
export SWANLAB_SAVE_DIR="$OSWORLD_EVAL_CACHE_ROOT/swanlab"
export TORCHINDUCTOR_CACHE_DIR="$OSWORLD_EVAL_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$OSWORLD_EVAL_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$OSWORLD_EVAL_CACHE_ROOT/cuda"
export TORCH_EXTENSIONS_DIR="$OSWORLD_EVAL_CACHE_ROOT/torch-extensions"
export TORCH_CUDA_ARCH_LIST='9.0 10.0'

mkdir -p \
  "$OSWORLD_EVAL_RUN_ROOT/exp_001" \
  "$NEMO_GYM_VENV_DIR" \
  "$HF_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$SWANLAB_SAVE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TORCH_EXTENSIONS_DIR"

test -x "$LIVE_DRIVER_PYTHON"
test -x "$LIVE_COMPONENT_PYTHON"
test -r "$NANO_OMNI_MODEL_NAME/config.json"
test -r "$NANO_OMNI_CHAT_TEMPLATE"
test -s "$OSWORLD_GRPO_VAL_DATA"
