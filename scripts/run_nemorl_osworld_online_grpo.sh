#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NEMORL_ROOT="${NEMORL_ROOT:-${PROJECT_ROOT}/../Nemo-RL-main-1/RL}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/nemorl_osworld_online_grpo_qwen_vl.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemogym/osworld_online_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemogym/osworld_online_val.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/osworld_online_grpo_qwen_vl}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/osworld_online_grpo_qwen_vl}"

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
  echo "[FATAL] Online NeMo-Gym datasets missing." >&2
  echo "Run dataset prep first:" >&2
  echo "  python scripts/prepare_osworld_nemogym_data.py" >&2
  echo "Expected:" >&2
  echo "  ${TRAIN_DATA}" >&2
  echo "  ${VAL_DATA}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

echo "=== NeMo-RL OSWorld Online GRPO (NeMo-Gym) ==="
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "NEMORL_ROOT:  ${NEMORL_ROOT}"
echo "CONFIG_PATH:  ${CONFIG_PATH}"
echo "TRAIN_DATA:   ${TRAIN_DATA}"
echo "VAL_DATA:     ${VAL_DATA}"
echo "CHECKPOINT:   ${CHECKPOINT_DIR}"
echo "LOG_DIR:      ${LOG_DIR}"
echo

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_nemorl_osworld_online_grpo.py"
  --nemo-rl-root "${NEMORL_ROOT}"
  --config "${CONFIG_PATH}"
  "data.train.data_path=${TRAIN_DATA}"
  "data.validation.data_path=${VAL_DATA}"
  "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
  "logger.log_dir=${LOG_DIR}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"
