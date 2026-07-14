#!/usr/bin/env bash
# Prepare NeMo-Gym data and submit OSWorld online GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_ALL_META_PATH="${TEST_ALL_META_PATH:-${PROJECT_ROOT}/configs/osworld_test_all_smoke20.json}"
TEST_CONFIG_BASE_DIR="${TEST_CONFIG_BASE_DIR:-${PROJECT_ROOT}/third_party/OSWorld/evaluation_examples}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemogym/osworld_online_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemogym/osworld_online_val.jsonl}"
VAL_RATIO="${VAL_RATIO:-0.1}"
DATA_SEED="${DATA_SEED:-42}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

echo "=== [1/2] Prepare OSWorld NeMo-Gym data ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_osworld_nemogym_data.py" \
  --test-all-meta-path "${TEST_ALL_META_PATH}" \
  --test-config-base-dir "${TEST_CONFIG_BASE_DIR}" \
  --train-output "${TRAIN_DATA}" \
  --val-output "${VAL_DATA}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${DATA_SEED}" \
  --max-steps "${MAX_STEPS}" \
  --max-samples "${MAX_SAMPLES}"

echo
echo "=== [2/2] Submit online GRPO training ==="
TRAIN_DATA="${TRAIN_DATA}" \
VAL_DATA="${VAL_DATA}" \
bash "${SCRIPT_DIR}/slurm/submit_nemorl_osworld_online_train.sh" "$@"
