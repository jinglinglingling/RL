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

# Pick a per-job master port subrange to reduce cross-job TCPStore collisions.
MASTER_PORT_RANGE_LOW="${NRL_MASTER_PORT_RANGE_LOW:-}"
MASTER_PORT_RANGE_HIGH="${NRL_MASTER_PORT_RANGE_HIGH:-}"
if [[ -z "${MASTER_PORT_RANGE_LOW}" || -z "${MASTER_PORT_RANGE_HIGH}" ]]; then
  base="${NRL_MASTER_PORT_BASE:-24000}"
  cap="${NRL_MASTER_PORT_CAP:-32000}"
  block="${NRL_MASTER_PORT_BLOCK_SIZE:-64}"
  seed="${SLURM_JOB_ID:-$RANDOM}"
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    seed="$(date +%s)"
  fi
  if (( block <= 1 )); then
    block=64
  fi
  if (( cap <= base )); then
    cap=$((base + block))
  fi

  span=$((cap - base))
  slots=$((span / block))
  if (( slots <= 0 )); then
    MASTER_PORT_RANGE_LOW="${base}"
    MASTER_PORT_RANGE_HIGH="${cap}"
  else
    idx=$((seed % slots))
    MASTER_PORT_RANGE_LOW=$((base + idx * block))
    MASTER_PORT_RANGE_HIGH=$((MASTER_PORT_RANGE_LOW + block))
    if (( MASTER_PORT_RANGE_HIGH > cap )); then
      MASTER_PORT_RANGE_HIGH="${cap}"
    fi
  fi
fi
if (( MASTER_PORT_RANGE_LOW < 1024 )); then
  MASTER_PORT_RANGE_LOW=1024
fi
if (( MASTER_PORT_RANGE_HIGH <= MASTER_PORT_RANGE_LOW )); then
  MASTER_PORT_RANGE_HIGH=$((MASTER_PORT_RANGE_LOW + 1))
fi
echo "MASTER_PORT_RANGE: [${MASTER_PORT_RANGE_LOW}, ${MASTER_PORT_RANGE_HIGH})"
echo

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_nemorl_osworld_online_grpo.py"
  --nemo-rl-root "${NEMORL_ROOT}"
  --config "${CONFIG_PATH}"
  "data.train.data_path=${TRAIN_DATA}"
  "data.validation.data_path=${VAL_DATA}"
  "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}"
  "logger.log_dir=${LOG_DIR}"
  "cluster.master_port_range_low=${MASTER_PORT_RANGE_LOW}"
  "cluster.master_port_range_high=${MASTER_PORT_RANGE_HIGH}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"
