#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NEMORL_ROOT="${NEMORL_ROOT:-${PROJECT_ROOT}/../Nemo-RL-main-1/RL-merge-2689}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/nemorl_osworld_grpo_qwen_vl.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_val.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/osworld_grpo_qwen_vl}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/osworld_grpo_qwen_vl}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "[FATAL] Train dataset not found: ${TRAIN_DATA}" >&2
  echo "Run conversion first:" >&2
  echo "  python scripts/convert_osworld_results_to_nemorl.py ..." >&2
  exit 1
fi

if [[ ! -f "${VAL_DATA}" ]]; then
  echo "[FATAL] Validation dataset not found: ${VAL_DATA}" >&2
  echo "Run conversion first:" >&2
  echo "  python scripts/convert_osworld_results_to_nemorl.py ..." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

echo "=== NeMo-RL OSWorld GRPO ==="
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "NEMORL_ROOT:  ${NEMORL_ROOT}"
echo "CONFIG_PATH:  ${CONFIG_PATH}"
echo "TRAIN_DATA:   ${TRAIN_DATA}"
echo "VAL_DATA:     ${VAL_DATA}"
echo "CHECKPOINT:   ${CHECKPOINT_DIR}"
echo "LOG_DIR:      ${LOG_DIR}"
echo

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_nemorl_osworld_grpo.py"
  --nemo-rl-root "${NEMORL_ROOT}"
  --config "${CONFIG_PATH}"
  "data.train.data_path=${TRAIN_DATA}"
  "data.validation.data_path=${VAL_DATA}"
  "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
  "logger.log_dir=${LOG_DIR}"
)

# Additional hydra overrides from caller.
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"
