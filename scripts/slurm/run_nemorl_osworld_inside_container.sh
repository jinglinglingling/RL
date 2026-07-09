#!/usr/bin/env bash
# Run NeMo-RL OSWorld training inside an attached container shell.
#
# Typical flow:
#   1) submit cluster job (your existing sbatch/ray.sub flow)
#   2) attach into head container
#   3) run this script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NEMORL_ROOT="${NEMORL_ROOT:-${PROJECT_ROOT}/../Nemo-RL-main-1/RL-merge-2689}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/nemorl_osworld_grpo_qwen_vl.yaml}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_val.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/osworld_grpo_qwen_vl}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/osworld_grpo_qwen_vl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_nemorl_osworld_grpo.sh"
if [[ ! -x "${RUN_SCRIPT}" ]]; then
  chmod +x "${RUN_SCRIPT}"
fi

echo "=== Run NeMo-RL OSWorld inside container ==="
echo "PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "NEMORL_ROOT:    ${NEMORL_ROOT}"
echo "CONFIG_PATH:    ${CONFIG_PATH}"
echo "TRAIN_DATA:     ${TRAIN_DATA}"
echo "VAL_DATA:       ${VAL_DATA}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "LOG_DIR:        ${LOG_DIR}"
echo

env \
  NEMORL_ROOT="${NEMORL_ROOT}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  TRAIN_DATA="${TRAIN_DATA}" \
  VAL_DATA="${VAL_DATA}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${RUN_SCRIPT}" "$@"
