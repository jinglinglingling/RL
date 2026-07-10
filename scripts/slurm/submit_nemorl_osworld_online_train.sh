#!/usr/bin/env bash
# Submit NeMo-RL online GRPO training (NeMo-Gym + OSWorld).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/nemorl_osworld_online_grpo_qwen_vl.yaml}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemogym/osworld_online_train.jsonl}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemogym/osworld_online_val.jsonl}"
export RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_ROOT}/scripts/run_nemorl_osworld_online_grpo.sh}"

bash "${SCRIPT_DIR}/submit_nemorl_osworld_train.sh" "$@"
