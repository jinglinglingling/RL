#!/usr/bin/env bash
# Evaluate a frozen Megatron-Bridge OSWorld checkpoint through NeMo-Gym.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

: "${OPENSANDBOX_DOMAIN:?Set the cell-2 OpenSandbox host}"
: "${OPENSANDBOX_API_KEY:?Set the cell-2 OpenSandbox API key}"
: "${EVAL_CHECKPOINT_PATH:?Set the policy weights directory containing iter_*}"
: "${OSWORLD_GRPO_VAL_DATA:?Set an absolute held-out OSWorld JSONL path}"

if [[ ! -f "${OSWORLD_GRPO_VAL_DATA}" ]]; then
  echo "Held-out data not found: ${OSWORLD_GRPO_VAL_DATA}" >&2
  exit 2
fi
if [[ ! -d "${EVAL_CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint weights directory not found: ${EVAL_CHECKPOINT_PATH}" >&2
  exit 2
fi

EVAL_NAME="${EVAL_NAME:-$(basename "$(dirname "$(dirname "${EVAL_CHECKPOINT_PATH}")")")}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/osworld-checkpoint-eval/${EVAL_NAME}}"

OSWORLD_GRPO_TRAIN_DATA="${OSWORLD_GRPO_VAL_DATA}" \
CONFIG_PATH="examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-eval.v1.yaml" \
PRETRAINED_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH}" \
PRETRAINED_CHECKPOINT_FORMAT=megatron_bridge \
GRPO_MAX_NUM_STEPS=0 \
OSWORLD_NUM_GENERATIONS=1 \
OSWORLD_TRAIN_GLOBAL_BATCH_SIZE=1 \
OSWORLD_USE_DYNAMIC_SAMPLING=false \
OSWORLD_DEBUG_TRAJ_DIR="${RESULTS_DIR}/debug-trajectories" \
CHECKPOINTING_ENABLED=false \
RESULTS_DIR="${RESULTS_DIR}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EVAL_NAME}-heldout-eval}" \
WANDB_RUN_ID="${WANDB_RUN_ID:-${EVAL_NAME}-heldout-eval}" \
JOB_NAME="${JOB_NAME:-osw-eval-${EVAL_NAME}}" \
  bash "${ROOT}/examples/nemo_gym/slurm/submit_osworld_grpo.sh"
