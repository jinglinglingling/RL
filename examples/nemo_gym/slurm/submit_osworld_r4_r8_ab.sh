#!/usr/bin/env bash
# Submit an uncontaminated R4 -> R8 -> R4 OSWorld infrastructure comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
SUBMIT="${ROOT}/examples/nemo_gym/slurm/submit_osworld_grpo.sh"

: "${OPENSANDBOX_DOMAIN:?Set the cell-2 OpenSandbox host}"
: "${OPENSANDBOX_API_KEY:?Set the cell-2 OpenSandbox API key}"

AB_DATA="${AB_DATA:-${ROOT}/results/osworld-r4-r8-experiment/fixed-hf-task.jsonl}"
AB_ROOT="${AB_ROOT:-${ROOT}/results/osworld-r4-r8-experiment/sequential-ab}"
AB_NUM_NODES="${AB_NUM_NODES:-1}"
AB_MAX_STEPS="${AB_MAX_STEPS:-15}"
AB_DEPENDENCY="${AB_DEPENDENCY:-}"

if [[ ! -f "${AB_DATA}" ]]; then
  echo "Fixed A/B dataset not found: ${AB_DATA}" >&2
  exit 2
fi

submit_arm() {
  local label="$1"
  local generations="$2"
  local dependency="$3"
  local output

  output="$(
    OSWORLD_GRPO_TRAIN_DATA="${AB_DATA}" \
    NUM_NODES="${AB_NUM_NODES}" \
    GRPO_MAX_NUM_STEPS=1 \
    OSWORLD_NUM_GENERATIONS="${generations}" \
    OSWORLD_TRAIN_GLOBAL_BATCH_SIZE="${generations}" \
    OSWORLD_USE_DYNAMIC_SAMPLING=false \
    OSWORLD_MAX_STEPS="${AB_MAX_STEPS}" \
    RESULTS_DIR="${AB_ROOT}/${label}" \
    OSWORLD_DEBUG_TRAJ_DIR="${AB_ROOT}/${label}/debug-trajectories" \
    CHECKPOINTING_ENABLED=false \
    WANDB_ENABLED=false \
    JOB_NAME="osw-ab-${label}" \
    SBATCH_DEPENDENCY="${dependency}" \
      bash "${SUBMIT}"
  )"
  printf '%s\n' "${output}" >&2
  awk '/Submitted batch job/{print $4}' <<<"${output}"
}

mkdir -p "${AB_ROOT}"
first_r4="$(submit_arm r4-a 4 "${AB_DEPENDENCY}")"
r8="$(submit_arm r8 8 "${first_r4}")"
second_r4="$(submit_arm r4-b 4 "${r8}")"

cat <<EOF
Submitted sequential OSWorld infrastructure A/B:
  R4-A: ${first_r4}
  R8:   ${r8} (after ${first_r4})
  R4-B: ${second_r4} (after ${r8})
  data: ${AB_DATA}
EOF
