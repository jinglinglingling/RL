#!/usr/bin/env bash
# Submit the laptop-independent OSWorld watchdog on a CPU node.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
CONFIG_PATH="${OSWORLD_WATCHDOG_CONFIG:-${ROOT}/examples/nemo_gym/slurm/osworld_night_watchdog.tiny.json}"
STATE_DIR="${OSWORLD_WATCHDOG_STATE_DIR:-${ROOT}/results/osworld-night-watchdog}"
CELL2_ENV_FILE="${OSWORLD_CELL2_ENV_FILE:?Set OSWORLD_CELL2_ENV_FILE to the protected Cell-2 env file}"
CURSOR_API_KEY_FILE="${CURSOR_API_KEY_FILE:-${HOME}/.config/cursor/osworld-night-operator.key}"
SDK_PYTHON="${OSWORLD_WATCHDOG_SDK_PYTHON:-${ROOT}/.cache/osworld-night-operator/sdk-venv/bin/python}"
SDK_REQUIREMENTS="${ROOT}/examples/nemo_gym/slurm/osworld_night_agent.requirements.txt"
WATCHDOG_SCRIPT="${ROOT}/examples/nemo_gym/slurm/osworld_night_watchdog.py"
JOB_NAME="${OSWORLD_WATCHDOG_JOB_NAME:-osw-night-watchdog}"
SBATCH_ACCOUNT="${OSWORLD_WATCHDOG_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${OSWORLD_WATCHDOG_PARTITION:-cpu_long}"
SBATCH_TIME="${OSWORLD_WATCHDOG_TIME:-7-00:00:00}"

for required_file in "${CONFIG_PATH}" "${CELL2_ENV_FILE}" "${SDK_REQUIREMENTS}" "${WATCHDOG_SCRIPT}"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "Required watchdog file not found: ${required_file}" >&2
    exit 2
  fi
done
if [[ "$(stat -c '%a' "${CELL2_ENV_FILE}")" != "600" ]]; then
  echo "Cell-2 env file must have mode 0600: ${CELL2_ENV_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CELL2_ENV_FILE}"
set +a
if [[ -n "${OPENSANDBOX_BASE_URL:-}" ]]; then
  export OPENSANDBOX_DOMAIN="${OPENSANDBOX_BASE_URL#*://}"
  export OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN%/}"
fi
: "${OPENSANDBOX_DOMAIN:?Cell-2 env must set OPENSANDBOX_DOMAIN or OPENSANDBOX_BASE_URL}"
: "${OPENSANDBOX_API_KEY:?Cell-2 env must set OPENSANDBOX_API_KEY}"

sdk_venv="$(dirname "$(dirname "${SDK_PYTHON}")")"
sdk_marker="${sdk_venv}/.requirements.sha256"
sdk_expected_marker="$(sha256sum "${SDK_REQUIREMENTS}" | awk '{print $1}')"
mkdir -p "$(dirname "${sdk_venv}")"
(
  flock -x 9
  sdk_current_marker=""
  if [[ -f "${sdk_marker}" ]]; then
    sdk_current_marker="$(<"${sdk_marker}")"
  fi
  if [[ ! -x "${SDK_PYTHON}" || "${sdk_current_marker}" != "${sdk_expected_marker}" ]]; then
    uv venv --allow-existing --python 3.13 "${sdk_venv}"
    uv pip install --python "${SDK_PYTHON}" --requirement "${SDK_REQUIREMENTS}"
    printf '%s\n' "${sdk_expected_marker}" >"${sdk_marker}"
  fi
) 9>"${sdk_venv}.lock"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="$(
    python -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])'
  )"
fi
export CURSOR_API_KEY_FILE
export PYTHONUNBUFFERED=1

mkdir -p "${STATE_DIR}"
existing_job="$(
  squeue -h -u "${USER}" -n "${JOB_NAME}" -o '%i' | awk 'NF { print; exit }'
)"
if [[ -n "${existing_job}" ]]; then
  echo "Watchdog already active: ${existing_job}"
  exit 0
fi

if [[ ! -f "${CURSOR_API_KEY_FILE}" ]]; then
  echo "Warning: ${CURSOR_API_KEY_FILE} is absent; Slurm recovery will run, but Cursor repair is disabled until the key file exists." >&2
elif [[ "$(stat -c '%a' "${CURSOR_API_KEY_FILE}")" != "600" ]]; then
  echo "Cursor API key file must have mode 0600: ${CURSOR_API_KEY_FILE}" >&2
  exit 2
fi

job_id="$(
  sbatch \
    --parsable \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --time="${SBATCH_TIME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --job-name="${JOB_NAME}" \
    --chdir="${ROOT}" \
    --output="${STATE_DIR}/watchdog-%j.out" \
    --open-mode=append \
    --export=ALL \
    --wrap="exec '${SDK_PYTHON}' '${WATCHDOG_SCRIPT}' --config '${CONFIG_PATH}'"
)"
echo "Submitted OSWorld night watchdog: ${job_id}"
