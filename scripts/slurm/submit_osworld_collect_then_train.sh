#!/usr/bin/env bash
# Submit OSWorld collect/convert first, then chain NeMo-RL train via dependency.
#
# Usage:
#   bash scripts/slurm/submit_osworld_collect_then_train.sh
#   RUN_COLLECT=1 RUN_CONVERT=1 TRAIN_PROFILE=stable-1g-3b-local \
#     bash scripts/slurm/submit_osworld_collect_then_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COLLECT_SUBMIT_SCRIPT="${COLLECT_SUBMIT_SCRIPT:-${SCRIPT_DIR}/submit_osworld_collect_convert.sh}"
TRAIN_SUBMIT_SCRIPT="${TRAIN_SUBMIT_SCRIPT:-${SCRIPT_DIR}/submit_nemorl_osworld_train.sh}"

if [[ ! -x "${COLLECT_SUBMIT_SCRIPT}" ]]; then
  chmod +x "${COLLECT_SUBMIT_SCRIPT}"
fi
if [[ ! -x "${TRAIN_SUBMIT_SCRIPT}" ]]; then
  chmod +x "${TRAIN_SUBMIT_SCRIPT}"
fi

# For smoke/early bring-up, include low-score trajectories to avoid empty conversion.
export INCLUDE_FAILED="${INCLUDE_FAILED:-1}"
export MIN_EPISODE_SCORE="${MIN_EPISODE_SCORE:-0.0}"

echo "=== Stage A: submit collect+convert ==="
collect_output="$("${COLLECT_SUBMIT_SCRIPT}" 2>&1)"
echo "${collect_output}"

collect_job_id="$(awk '/Submitted job/ {jid=$3} END {print jid}' <<< "${collect_output}")"
if [[ -z "${collect_job_id}" ]]; then
  echo "[FATAL] Failed to parse collect job id from submit output." >&2
  exit 1
fi

export SBATCH_DEPENDENCY="afterok:${collect_job_id}"
export ALLOW_MISSING_DATA_ON_SUBMIT="${ALLOW_MISSING_DATA_ON_SUBMIT:-1}"
echo
echo "=== Stage B: submit train (dependency: ${SBATCH_DEPENDENCY}) ==="
train_output="$("${TRAIN_SUBMIT_SCRIPT}" 2>&1)"
echo "${train_output}"

train_job_id="$(awk '/Submitted job/ {jid=$3} END {print jid}' <<< "${train_output}")"
if [[ -z "${train_job_id}" ]]; then
  echo "[FATAL] Failed to parse train job id from submit output." >&2
  exit 1
fi

echo
echo "Pipeline submitted:"
echo "  collect/convert job: ${collect_job_id}"
echo "  train job:           ${train_job_id} (afterok dependency)"
