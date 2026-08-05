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

if [[ -z "${GRPO_ENV_FINGERPRINT:-}" ]]; then
  GRPO_ENV_FINGERPRINT="$(
    {
      sha256sum \
        "${ROOT}/uv.lock" \
        "${ROOT}/pyproject.toml" \
        "${ROOT}/examples/nemo_gym/osworld-uv-overrides.txt" \
        "${ROOT}/3rdparty/Gym-workspace/Gym/uv.lock" \
        "${ROOT}/3rdparty/Gym-workspace/Gym/pyproject.toml" \
        "${ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/nemotron_osworld/requirements.txt" \
        "${ROOT}/3rdparty/Gym-workspace/Gym/resources_servers/osworld/requirements.txt"
      printf '%s\n' "${CONTAINER}"
    } | sha256sum | cut -c1-16
  )"
fi
export GRPO_ENV_FINGERPRINT
export GRPO_PERSISTENT_ENV_ROOT="${GRPO_PERSISTENT_ENV_ROOT:-${ROOT}/.cache/osworld-grpo-envs/${GRPO_ENV_FINGERPRINT}}"
if [[ -z "${UV_CACHE_DIR_OVERRIDE+x}" ]]; then
  export UV_CACHE_DIR_OVERRIDE=""
fi
export UV_RUNTIME_CACHE_DIR="${UV_RUNTIME_CACHE_DIR:-${GRPO_PERSISTENT_ENV_ROOT}/uv-cache}"
export UV_OVERRIDE="${UV_OVERRIDE:-${ROOT}/examples/nemo_gym/osworld-uv-overrides.txt}"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
# The production OSWorld pool is H100 (SM90). Transformer Engine otherwise
# compiles every CUDA-13 target (75/80/89/90/100/120), adding tens of minutes.
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export GRPO_ROOT_VENV="${GRPO_ROOT_VENV:-${GRPO_PERSISTENT_ENV_ROOT}/root}"
# Ray workers execute actor environments from node-local storage.  Executing
# one shared venv concurrently from several Lustre clients caused intermittent
# missing-module and truncated-package failures.  The expensive downloads and
# source builds remain deduplicated through the fingerprinted shared uv cache.
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/tmp/nemo_rl_actor_venvs-${GRPO_ENV_FINGERPRINT}}"
export OSWORLD_GYM_VENV_DIR="${OSWORLD_GYM_VENV_DIR:-/tmp/osworld_gym_venvs-${GRPO_ENV_FINGERPRINT}}"
export UV_PROJECT_ENVIRONMENT="${GRPO_ROOT_VENV}"
export VIRTUAL_ENV="${GRPO_ROOT_VENV}"
export PATH="${GRPO_ROOT_VENV}/bin:${PATH}"
export NRL_MAMBA_PREFILL_DECODE_SYNC="${NRL_MAMBA_PREFILL_DECODE_SYNC:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NRL_ALLOW_PARTIAL_VENV_ON_SYNC_FAILURE="${NRL_ALLOW_PARTIAL_VENV_ON_SYNC_FAILURE:-true}"
# The image predates this checkout. Fingerprinted shared environments are built
# once, validated, and then reused by every four-hour resume segment.
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_VENV_FINGERPRINT="${GRPO_ENV_FINGERPRINT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-1}"
GRPO_MAX_NUM_EPOCHS="${GRPO_MAX_NUM_EPOCHS:-1}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/osworld-grpo-smoke}"
export OSWORLD_GRPO_VAL_DATA="${OSWORLD_GRPO_VAL_DATA:-}"
export OSWORLD_VAL_BATCH_SIZE="${OSWORLD_VAL_BATCH_SIZE:-4}"
export OSWORLD_MAX_VAL_SAMPLES="${OSWORLD_MAX_VAL_SAMPLES:-20}"
export OSWORLD_EVAL_MAX_STEPS="${OSWORLD_EVAL_MAX_STEPS:-100}"
export OSWORLD_VAL_PERIOD="${OSWORLD_VAL_PERIOD:-0}"
export OSWORLD_VAL_AT_START="${OSWORLD_VAL_AT_START:-false}"
export OSWORLD_VAL_AT_END="${OSWORLD_VAL_AT_END:-false}"
export OSWORLD_NUM_PROMPTS_PER_STEP="${OSWORLD_NUM_PROMPTS_PER_STEP:-1}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-8}"
export OSWORLD_NEMO_GYM_NUM_WORKERS="${OSWORLD_NEMO_GYM_NUM_WORKERS:-4}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-4}"
export OSWORLD_GENERATION_BATCH_SIZE="${OSWORLD_GENERATION_BATCH_SIZE:-32}"
export OSWORLD_LEARNING_RATE="${OSWORLD_LEARNING_RATE:-2e-7}"
export OSWORLD_TRAIN_GLOBAL_BATCH_SIZE="${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE:-$((OSWORLD_NUM_GENERATIONS * OSWORLD_MAX_STEPS))}"
export OSWORLD_MAX_MODEL_LEN="${OSWORLD_MAX_MODEL_LEN:-16384}"
export OSWORLD_MAX_IMAGE_HISTORY_LENGTH="${OSWORLD_MAX_IMAGE_HISTORY_LENGTH:-2}"
export OSWORLD_CONTEXT_PARALLEL_SIZE="${OSWORLD_CONTEXT_PARALLEL_SIZE:-1}"
if [[ -z "${OSWORLD_SEQUENCE_LENGTH_DIVISOR:-}" ]]; then
  if (( OSWORLD_CONTEXT_PARALLEL_SIZE > 1 )); then
    export OSWORLD_SEQUENCE_LENGTH_DIVISOR="$((16 * OSWORLD_CONTEXT_PARALLEL_SIZE))"
  else
    export OSWORLD_SEQUENCE_LENGTH_DIVISOR=8
  fi
fi
export OSWORLD_MONOTONIC_SEGMENTS="${OSWORLD_MONOTONIC_SEGMENTS:-false}"
export OSWORLD_HISTORY_MODE="${OSWORLD_HISTORY_MODE:-reference}"
export OSWORLD_INDEPENDENT_TURN_TRAINING="${OSWORLD_INDEPENDENT_TURN_TRAINING:-all}"
export OSWORLD_INDEPENDENT_TURN_SAMPLING="${OSWORLD_INDEPENDENT_TURN_SAMPLING:-first}"
export OSWORLD_USE_DYNAMIC_SAMPLING="${OSWORLD_USE_DYNAMIC_SAMPLING:-true}"
export OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES="${OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES:-16}"
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
PRETRAINED_CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT_PATH:-}"
PRETRAINED_CHECKPOINT_FORMAT="${PRETRAINED_CHECKPOINT_FORMAT:-megatron_bridge}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-1n8g-megatron.v1.yaml}"
PRETRAINED_OVERRIDES=""
if [[ -n "${PRETRAINED_CHECKPOINT_PATH}" ]]; then
  PRETRAINED_OVERRIDES="++checkpointing.pretrained_checkpoint.path='${PRETRAINED_CHECKPOINT_PATH}' ++checkpointing.pretrained_checkpoint.format='${PRETRAINED_CHECKPOINT_FORMAT}'"
fi
mkdir -p \
  "${HF_HOME}" \
  "${RESULTS_DIR}" \
  "${CHECKPOINT_DIR}" \
  "${GRPO_PERSISTENT_ENV_ROOT}" \
  "${UV_RUNTIME_CACHE_DIR}"
if [[ -n "${UV_CACHE_DIR_OVERRIDE}" ]]; then
  mkdir -p "${UV_CACHE_DIR_OVERRIDE}"
fi
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

export SETUP_COMMAND="export ROOT='${ROOT}' GRPO_ROOT_VENV='${GRPO_ROOT_VENV}' GRPO_ENV_FINGERPRINT='${GRPO_ENV_FINGERPRINT}' && \
export UV_OVERRIDE='${UV_OVERRIDE}' UV_RUNTIME_CACHE_DIR='${UV_RUNTIME_CACHE_DIR}' UV_LOCK_TIMEOUT='${UV_LOCK_TIMEOUT}' UV_HTTP_TIMEOUT='${UV_HTTP_TIMEOUT}' UV_LINK_MODE='${UV_LINK_MODE}' NVTE_CUDA_ARCHS='${NVTE_CUDA_ARCHS}' TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' WANDB_ENABLED='${WANDB_ENABLED}' && \
bash '${ROOT}/examples/nemo_gym/slurm/ensure_osworld_grpo_root_env.sh'"

export COMMAND="cd '${ROOT}' && \
export HF_HOME='${HF_HOME}' HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' && \
unset UV_CACHE_DIR && export UV_OVERRIDE='${UV_OVERRIDE}' UV_LOCK_TIMEOUT='${UV_LOCK_TIMEOUT}' UV_HTTP_TIMEOUT='${UV_HTTP_TIMEOUT}' UV_LINK_MODE='${UV_LINK_MODE}' NVTE_CUDA_ARCHS='${NVTE_CUDA_ARCHS}' TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
export UV_PROJECT_ENVIRONMENT='${GRPO_ROOT_VENV}' VIRTUAL_ENV='${GRPO_ROOT_VENV}' PATH='${GRPO_ROOT_VENV}/bin':\"\${PATH}\" && \
export NEMO_RL_VENV_DIR='${NEMO_RL_VENV_DIR}' NRL_VENV_FINGERPRINT='${NRL_VENV_FINGERPRINT}' OSWORLD_GYM_VENV_DIR='${OSWORLD_GYM_VENV_DIR}' && \
export OSWORLD_GRPO_TRAIN_DATA='${OSWORLD_GRPO_TRAIN_DATA}' OSWORLD_MAX_STEPS='${OSWORLD_MAX_STEPS}' OSWORLD_DEBUG_TRAJ_DIR='${OSWORLD_DEBUG_TRAJ_DIR}' && \
export OSWORLD_GRPO_VAL_DATA='${OSWORLD_GRPO_VAL_DATA}' OSWORLD_VAL_BATCH_SIZE='${OSWORLD_VAL_BATCH_SIZE}' OSWORLD_MAX_VAL_SAMPLES='${OSWORLD_MAX_VAL_SAMPLES}' OSWORLD_EVAL_MAX_STEPS='${OSWORLD_EVAL_MAX_STEPS}' && \
export OSWORLD_NUM_PROMPTS_PER_STEP='${OSWORLD_NUM_PROMPTS_PER_STEP}' OSWORLD_NUM_GENERATIONS='${OSWORLD_NUM_GENERATIONS}' OSWORLD_TRAIN_GLOBAL_BATCH_SIZE='${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE}' && \
export OSWORLD_NEMO_GYM_NUM_WORKERS='${OSWORLD_NEMO_GYM_NUM_WORKERS}' OSWORLD_MAX_PARALLEL_ROLLOUTS='${OSWORLD_MAX_PARALLEL_ROLLOUTS}' OSWORLD_GENERATION_BATCH_SIZE='${OSWORLD_GENERATION_BATCH_SIZE}' && \
export OSWORLD_LEARNING_RATE='${OSWORLD_LEARNING_RATE}' && \
export OSWORLD_MAX_MODEL_LEN='${OSWORLD_MAX_MODEL_LEN}' OSWORLD_MAX_IMAGE_HISTORY_LENGTH='${OSWORLD_MAX_IMAGE_HISTORY_LENGTH}' OSWORLD_CONTEXT_PARALLEL_SIZE='${OSWORLD_CONTEXT_PARALLEL_SIZE}' && \
export OSWORLD_SEQUENCE_LENGTH_DIVISOR='${OSWORLD_SEQUENCE_LENGTH_DIVISOR}' && \
export OSWORLD_MONOTONIC_SEGMENTS='${OSWORLD_MONOTONIC_SEGMENTS}' && \
export OSWORLD_HISTORY_MODE='${OSWORLD_HISTORY_MODE}' && \
export OSWORLD_INDEPENDENT_TURN_TRAINING='${OSWORLD_INDEPENDENT_TURN_TRAINING}' OSWORLD_INDEPENDENT_TURN_SAMPLING='${OSWORLD_INDEPENDENT_TURN_SAMPLING}' && \
export OSWORLD_USE_DYNAMIC_SAMPLING='${OSWORLD_USE_DYNAMIC_SAMPLING}' OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES='${OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES}' && \
export NRL_MAMBA_PREFILL_DECODE_SYNC='${NRL_MAMBA_PREFILL_DECODE_SYNC}' PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 && \
export NRL_FORCE_REBUILD_VENVS='${NRL_FORCE_REBUILD_VENVS}' NRL_ALLOW_PARTIAL_VENV_ON_SYNC_FAILURE='${NRL_ALLOW_PARTIAL_VENV_ON_SYNC_FAILURE}' NRL_IGNORE_VERSION_MISMATCH=1 && \
'${GRPO_ROOT_VENV}/bin/python' examples/run_vlm_grpo.py \
  --config '${CONFIG_PATH}' \
  grpo.max_num_steps=${GRPO_MAX_NUM_STEPS} \
  grpo.max_num_epochs=${GRPO_MAX_NUM_EPOCHS} \
  grpo.val_period=${OSWORLD_VAL_PERIOD} \
  grpo.val_at_start=${OSWORLD_VAL_AT_START} \
  grpo.val_at_end=${OSWORLD_VAL_AT_END} \
  cluster.num_nodes=${NUM_NODES} \
  checkpointing.enabled=${CHECKPOINTING_ENABLED} \
  checkpointing.checkpoint_dir='${CHECKPOINT_DIR}' \
  checkpointing.save_period=${CHECKPOINT_SAVE_PERIOD} \
  ${PRETRAINED_OVERRIDES} \
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
if [[ -z "${SBATCH_MEM:-}" ]]; then
  if (( OSWORLD_MAX_STEPS >= 30 )); then
    SBATCH_MEM="1536G"
  else
    SBATCH_MEM="1024G"
  fi
fi
JOB_NAME="${JOB_NAME:-osworld-grpo-smoke}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
SBATCH_DEPENDENCY_TYPE="${SBATCH_DEPENDENCY_TYPE:-afterany}"

cat <<EOF
Submitting OSWorld GRPO
  account/partition: ${SBATCH_ACCOUNT}/${SBATCH_PARTITION}
  container:         ${CONTAINER}
  model:             nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
  topology:          ${NUM_NODES} node(s) x 8 GPU
  sandboxes:         ${OPENSANDBOX_DOMAIN}, pool=${OSWORLD_POOL_REF}, concurrency=${OSWORLD_MAX_PARALLEL_ROLLOUTS}
  train data:        ${OSWORLD_GRPO_TRAIN_DATA}
  validation data:   ${OSWORLD_GRPO_VAL_DATA:-disabled}
  validation cadence:start=${OSWORLD_VAL_AT_START}, every ${OSWORLD_VAL_PERIOD} steps, end=${OSWORLD_VAL_AT_END}
  pretrained ckpt:  ${PRETRAINED_CHECKPOINT_PATH:-base model}
  GRPO steps/epochs: ${GRPO_MAX_NUM_STEPS}/${GRPO_MAX_NUM_EPOCHS}
  prompts/step:      ${OSWORLD_NUM_PROMPTS_PER_STEP}
  rollouts/group:    ${OSWORLD_NUM_GENERATIONS}
  NeMo-Gym workers:  ${OSWORLD_NEMO_GYM_NUM_WORKERS}
  train global batch:${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE}
  model/image history:${OSWORLD_MAX_MODEL_LEN} tokens / ${OSWORLD_MAX_IMAGE_HISTORY_LENGTH} images, CP=${OSWORLD_CONTEXT_PARALLEL_SIZE}, divisor=${OSWORLD_SEQUENCE_LENGTH_DIVISOR}
  monotonic segments:${OSWORLD_MONOTONIC_SEGMENTS}
  history mode:      ${OSWORLD_HISTORY_MODE}
  turn training:     ${OSWORLD_INDEPENDENT_TURN_TRAINING}, sample=${OSWORLD_INDEPENDENT_TURN_SAMPLING}
  dynamic sampling:  ${OSWORLD_USE_DYNAMIC_SAMPLING}, max batches=${OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES}
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
  --mem="${SBATCH_MEM}"
  --job-name="${JOB_NAME}"
  --export=ALL
)
if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${SBATCH_DEPENDENCY_TYPE}:${SBATCH_DEPENDENCY}")
fi

exec sbatch "${SBATCH_ARGS[@]}" "${ROOT}/ray.sub"
