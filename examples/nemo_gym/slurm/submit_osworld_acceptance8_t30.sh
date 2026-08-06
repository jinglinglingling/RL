#!/usr/bin/env bash
# Run the fixed eight-task T30 GRPO acceptance test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
export OSWORLD_SEED="${OSWORLD_SEED:-42}"
if [[ "${OSWORLD_SEED}" != "42" ]]; then
  echo "Acceptance experiments use the project-wide fixed seed 42; got ${OSWORLD_SEED}" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-osworld-acceptance8-t30-b8r8-lr2e7-seed${OSWORLD_SEED}-20260806}"
export CONFIG_PATH="${ROOT}/examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-overfit32-fast-4n.v1.yaml"
export OSWORLD_GRPO_TRAIN_DATA="${ROOT}/results/osworld-data/acceptance8-20260806/train-40x.jsonl"
export OSWORLD_GRPO_VAL_DATA="${ROOT}/results/osworld-data/acceptance8-20260806/validation-8x.jsonl"

export NUM_NODES="${NUM_NODES:-4}"
export GRPO_MAX_NUM_STEPS="${GRPO_MAX_NUM_STEPS:-30}"
export OSWORLD_NUM_PROMPTS_PER_STEP="${OSWORLD_NUM_PROMPTS_PER_STEP:-8}"
export OSWORLD_NUM_GENERATIONS="${OSWORLD_NUM_GENERATIONS:-8}"
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-30}"
export OSWORLD_TRAIN_GLOBAL_BATCH_SIZE="${OSWORLD_TRAIN_GLOBAL_BATCH_SIZE:-1920}"
export OSWORLD_NEMO_GYM_NUM_WORKERS="${OSWORLD_NEMO_GYM_NUM_WORKERS:-64}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-64}"
export OSWORLD_GENERATION_BATCH_SIZE="${OSWORLD_GENERATION_BATCH_SIZE:-32}"
export OSWORLD_USE_DYNAMIC_SAMPLING=false
export OSWORLD_DYNAMIC_SAMPLING_MAX_GEN_BATCHES=1
export OSWORLD_INDEPENDENT_TURN_TRAINING=all
export OSWORLD_LEARNING_RATE="${OSWORLD_LEARNING_RATE:-2e-7}"
export OSWORLD_LR_WARMUP_ITERS="${OSWORLD_LR_WARMUP_ITERS:-2}"
export OSWORLD_MAX_MODEL_LEN="${OSWORLD_MAX_MODEL_LEN:-16384}"
export OSWORLD_MAX_IMAGE_HISTORY_LENGTH="${OSWORLD_MAX_IMAGE_HISTORY_LENGTH:-2}"
# T30 samples can reach the full 16K cap. CP2 halves per-rank embedding and
# activation memory while retaining two data-parallel replicas on four nodes.
export OSWORLD_CONTEXT_PARALLEL_SIZE="${OSWORLD_CONTEXT_PARALLEL_SIZE:-2}"
export OSWORLD_SEQUENCE_LENGTH_DIVISOR="${OSWORLD_SEQUENCE_LENGTH_DIVISOR:-32}"

export OSWORLD_VAL_AT_START=false
export OSWORLD_VAL_AT_END=true
export OSWORLD_VAL_PERIOD="${OSWORLD_VAL_PERIOD:-5}"
export OSWORLD_VAL_BATCH_SIZE="${OSWORLD_VAL_BATCH_SIZE:-64}"
export OSWORLD_MAX_VAL_SAMPLES="${OSWORLD_MAX_VAL_SAMPLES:-64}"

export RESULTS_DIR="${ROOT}/results/osworld-grpo-acceptance/${RUN_NAME}"
export CHECKPOINTING_ENABLED=true
export CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
export CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-2}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
export WANDB_PROJECT="${WANDB_PROJECT:-osworld-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export SBATCH_MEM="${SBATCH_MEM:-1500G}"
export JOB_NAME="${JOB_NAME:-osw-accept8-s${OSWORLD_SEED}}"

exec bash "${ROOT}/examples/nemo_gym/slurm/submit_osworld_grpo.sh"
