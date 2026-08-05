#!/usr/bin/env bash
# Fast 32-task overfit run for validating OSWorld all-turn GRPO learning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
RUN_NAME="${RUN_NAME:-osworld-overfit32-b8r8-20step-4n-20260803}"

export CONFIG_PATH="${ROOT}/examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-overfit32-fast-4n.v1.yaml"
export OSWORLD_GRPO_TRAIN_DATA="${ROOT}/results/osworld-data/overfit32-20260803/train-20x.jsonl"
export OSWORLD_GRPO_VAL_DATA="${ROOT}/results/osworld-data/overfit32-20260803/validation-4x.jsonl"

export NUM_NODES="${NUM_NODES:-4}"
export GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-20}"
export GRPO_MAX_NUM_EPOCHS=1
export OSWORLD_NUM_PROMPTS_PER_STEP="${OSWORLD_NUM_PROMPTS_PER_STEP:-8}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-8}"
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-15}"
export OSWORLD_TRAIN_GLOBAL_BATCH_SIZE="${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE:-$((OSWORLD_NUM_PROMPTS_PER_STEP * OSWORLD_NUM_GENERATIONS * OSWORLD_MAX_STEPS))}"
# The model declares a 131K sequence limit and the reference OSWorld agent uses
# three screenshots. Start conservatively at 32K to preserve more text/action
# history without taking the full KV-cache and activation-memory jump.
export OSWORLD_MAX_MODEL_LEN="${OSWORLD_MAX_MODEL_LEN:-32768}"
export OSWORLD_MAX_IMAGE_HISTORY_LENGTH="${OSWORLD_MAX_IMAGE_HISTORY_LENGTH:-3}"

# Avoid pyxis cache bind mounts. The base submitter chooses a fingerprinted,
# persistent cache that is reused safely by every resume segment.
export UV_CACHE_DIR_OVERRIDE=""

export OSWORLD_NEMO_GYM_NUM_WORKERS="${OSWORLD_NEMO_GYM_NUM_WORKERS:-64}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-64}"
export OSWORLD_GENERATION_BATCH_SIZE="${OSWORLD_GENERATION_BATCH_SIZE:-32}"
export OSWORLD_USE_DYNAMIC_SAMPLING="${OSWORLD_USE_DYNAMIC_SAMPLING:-true}"
export OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES="${OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES:-4}"
export OSWORLD_ROLLOUT_MAX_ATTEMPTS="${OSWORLD_ROLLOUT_MAX_ATTEMPTS:-8}"

# Evaluate the exact same 32 tasks four times each. The repeated 128-row set
# reduces sampling noise while remaining small enough for 64-way execution.
export OSWORLD_VAL_BATCH_SIZE="${OSWORLD_VAL_BATCH_SIZE:-64}"
export OSWORLD_MAX_VAL_SAMPLES="${OSWORLD_MAX_VAL_SAMPLES:-128}"
export OSWORLD_EVAL_MAX_STEPS="${OSWORLD_MAX_STEPS}"
export OSWORLD_VAL_PERIOD="${OSWORLD_VAL_PERIOD:-5}"
export OSWORLD_VAL_AT_START=false
export OSWORLD_VAL_AT_END="${OSWORLD_VAL_AT_END:-true}"
export OSWORLD_DEBUG_TRAJ_DIR=""

export RESULTS_DIR="${ROOT}/results/osworld-grpo-formal/${RUN_NAME}"
export CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-true}"
export CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
export CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-2}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
export WANDB_PROJECT="${WANDB_PROJECT:-osworld-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export JOB_NAME="${JOB_NAME:-osw-overfit32-b8r8}"

exec bash "${ROOT}/examples/nemo_gym/slurm/submit_osworld_grpo.sh"
