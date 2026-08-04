#!/usr/bin/env bash
# T30 arm of the fixed 32-task tiny-overfit horizon comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

export RUN_NAME="${RUN_NAME:-osworld-overfit32-b8r8-20step-t30-4n-20260803}"
export JOB_NAME="${JOB_NAME:-osw-overfit32-t30}"
export OSWORLD_MAX_STEPS=30

exec bash "${ROOT}/examples/nemo_gym/slurm/submit_osworld_overfit32_fast.sh"
