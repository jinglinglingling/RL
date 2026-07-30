#!/usr/bin/env bash
# Submit Nemotron Omni OSWorld GRPO through the repository's ray.sub.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

: "${OPENSANDBOX_DOMAIN:?Set the cell-2 OpenSandbox host}"
: "${OPENSANDBOX_API_KEY:?Set the cell-2 OpenSandbox API key}"
: "${OSWORLD_GRPO_TRAIN_DATA:?Set an absolute path to the OSWorld GRPO JSONL}"

if [[ ! -f "${OSWORLD_GRPO_TRAIN_DATA}" ]]; then
  echo "OSWorld GRPO dataset not found: ${OSWORLD_GRPO_TRAIN_DATA}" >&2
  exit 2
fi

export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/users/aroshanghias/containers/nemo-rl-26effe2-56454576.sqsh}"
if [[ ! -f "${CONTAINER}" ]]; then
  echo "NeMo-RL container not found: ${CONTAINER}" >&2
  exit 2
fi

export MOUNTS="${MOUNTS:-/lustre:/lustre}"
export NUM_NODES="${NUM_NODES:-1}"
export GPUS_PER_NODE=8
export OPENSANDBOX_DOMAIN
export OPENSANDBOX_API_KEY
export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-osworld-kvm}"
export OSWORLD_GRPO_TRAIN_DATA
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-5}"
export OSWORLD_DEBUG_TRAJ_DIR="${OSWORLD_DEBUG_TRAJ_DIR:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-${ROOT}/.cache/uv-grpo}"
export UV_OVERRIDE="${UV_OVERRIDE:-${ROOT}/examples/nemo_gym/osworld-uv-overrides.txt}"
export GRPO_ROOT_VENV="${GRPO_ROOT_VENV:-/tmp/nemo_rl_grpo_venv}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/tmp/nemo_rl_actor_venvs}"
export UV_PROJECT_ENVIRONMENT="${GRPO_ROOT_VENV}"
export VIRTUAL_ENV="${GRPO_ROOT_VENV}"
export PATH="${GRPO_ROOT_VENV}/bin:${PATH}"
export NRL_MAMBA_PREFILL_DECODE_SYNC="${NRL_MAMBA_PREFILL_DECODE_SYNC:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
# The available image predates PR #3290's exact MBridge and lockfile. Rebuild
# worker environments from this checkout and acknowledge that known mismatch.
export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONUNBUFFERED=1

GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-1}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/osworld-grpo-smoke}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-8}"
export OSWORLD_TRAIN_GLOBAL_BATCH_SIZE="${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE:-${OSWORLD_NUM_GENERATIONS}}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
WANDB_PROJECT="${WANDB_PROJECT:-osworld-grpo}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-nemotron-omni-osworld}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_RUN_NAME}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_MODE="${WANDB_MODE:-online}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"
CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-10}"
CONFIG_PATH="examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-1n8g-megatron.v1.yaml"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR_OVERRIDE}" "${RESULTS_DIR}" "${CHECKPOINT_DIR}"
if [[ -n "${OSWORLD_DEBUG_TRAJ_DIR}" ]]; then
  mkdir -p "${OSWORLD_DEBUG_TRAJ_DIR}"
fi
export BASE_LOG_DIR="${RESULTS_DIR}/slurm"

# Cursor's command sandbox state is meaningful only on the login host.
unset BASH_ENV ENV
while IFS= read -r cursor_var; do
  unset "${cursor_var}"
done < <(compgen -e "__CURSOR_SANDBOX_" || true)
while IFS= read -r env_var; do
  if [[ "${!env_var-}" == /tmp/cursor-sandbox-cache/* ]]; then
    unset "${env_var}"
  fi
done < <(compgen -e)

export SETUP_COMMAND="rm -rf '${GRPO_ROOT_VENV}' '${NEMO_RL_VENV_DIR}' && \
export UV_PROJECT_ENVIRONMENT='${GRPO_ROOT_VENV}' VIRTUAL_ENV='${GRPO_ROOT_VENV}' NEMO_RL_VENV_DIR='${NEMO_RL_VENV_DIR}' && \
export PATH='${GRPO_ROOT_VENV}/bin':\"\${PATH}\" && \
cd '${ROOT}' && \
uv sync --locked --reinstall && \
uv pip install --python '${GRPO_ROOT_VENV}/bin/python' --no-cache --reinstall 'ray[default]==2.55.1' && \
if [[ '${WANDB_ENABLED}' == 'true' ]]; then env -u UV_OVERRIDE uv pip install --python '${GRPO_ROOT_VENV}/bin/python' --no-cache --reinstall 'wandb==0.21.0' 'protobuf==6.33.5'; fi && \
uv run --locked --no-sync python -c 'from transformers import AutoProcessor; import fastapi, ray, ray._private.node, starlette, transformers, wandb; print(\"Dependency preflight:\", transformers.__version__, fastapi.__version__, starlette.__version__, ray.__version__, wandb.__version__)'"

export COMMAND="cd '${ROOT}' && \
export HF_HOME='${HF_HOME}' HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' && \
export UV_OVERRIDE='${UV_OVERRIDE}' && \
export OSWORLD_GRPO_TRAIN_DATA='${OSWORLD_GRPO_TRAIN_DATA}' OSWORLD_MAX_STEPS='${OSWORLD_MAX_STEPS}' OSWORLD_DEBUG_TRAJ_DIR='${OSWORLD_DEBUG_TRAJ_DIR}' && \
export NRL_MAMBA_PREFILL_DECODE_SYNC='${NRL_MAMBA_PREFILL_DECODE_SYNC}' PYTHONUNBUFFERED=1 && \
export NRL_FORCE_REBUILD_VENVS='${NRL_FORCE_REBUILD_VENVS}' NRL_IGNORE_VERSION_MISMATCH=1 && \
uv run --locked --no-sync examples/run_vlm_grpo.py \
  --config '${CONFIG_PATH}' \
  grpo.max_num_steps=${GRPO_MAX_NUM_STEPS} \
  cluster.num_nodes=${NUM_NODES} \
  checkpointing.enabled=${CHECKPOINTING_ENABLED} \
  checkpointing.checkpoint_dir='${CHECKPOINT_DIR}' \
  checkpointing.save_period=${CHECKPOINT_SAVE_PERIOD} \
  +env.nemo_gym.nemotron_osworld.responses_api_agents.nemotron_osworld.debug_trajectory_dir='${OSWORLD_DEBUG_TRAJ_DIR}' \
  logger.wandb_enabled=${WANDB_ENABLED} \
  +logger.wandb.entity='${WANDB_ENTITY}' \
  logger.wandb.project='${WANDB_PROJECT}' \
  logger.wandb.name='${WANDB_RUN_NAME}' \
  ++logger.wandb.id='${WANDB_RUN_ID}' \
  ++logger.wandb.resume='${WANDB_RESUME}' \
  logger.tensorboard_enabled=true \
  logger.log_dir='${RESULTS_DIR}/logs'"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
JOB_NAME="${JOB_NAME:-osworld-grpo-smoke}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"

cat <<EOF
Submitting OSWorld GRPO
  account/partition: ${SBATCH_ACCOUNT}/${SBATCH_PARTITION}
  container:         ${CONTAINER}
  model:             nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
  topology:          ${NUM_NODES} node(s) x 8 GPU
  sandboxes:         ${OPENSANDBOX_DOMAIN}, pool=${OSWORLD_POOL_REF}, concurrency=4
  train data:        ${OSWORLD_GRPO_TRAIN_DATA}
  GRPO steps:        ${GRPO_MAX_NUM_STEPS}
  rollouts/group:    ${OSWORLD_NUM_GENERATIONS}
  train global batch:${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE}
  rollout max steps: ${OSWORLD_MAX_STEPS}
  debug screenshots: ${OSWORLD_DEBUG_TRAJ_DIR:-disabled}
  wandb:             ${WANDB_ENABLED}, mode=${WANDB_MODE}, entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}, run=${WANDB_RUN_NAME}, version=0.21.0
  checkpointing:     ${CHECKPOINTING_ENABLED}, every ${CHECKPOINT_SAVE_PERIOD} steps
  checkpoint dir:    ${CHECKPOINT_DIR}
  results:           ${RESULTS_DIR}
EOF

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --exclusive
  --gres=gpu:8
  --account="${SBATCH_ACCOUNT}"
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --job-name="${JOB_NAME}"
  --export=ALL
)
if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="afterany:${SBATCH_DEPENDENCY}")
fi

exec sbatch "${SBATCH_ARGS[@]}" "${ROOT}/ray.sub"
