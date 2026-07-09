#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OSWORLD_SRC="${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/runs/osworld-results}"

# OSWorld routes OpenAI-compatible requests through model names beginning with "gpt".
# Serve Qwen-VL with this alias in your OpenAI-compatible server.
MODEL_ALIAS="${MODEL_ALIAS:-gpt-4o}"

PROVIDER_NAME="${PROVIDER_NAME:-docker}"
ACTION_SPACE="${ACTION_SPACE:-pyautogui}"
OBSERVATION_TYPE="${OBSERVATION_TYPE:-screenshot}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-3}"
SLEEP_AFTER_EXECUTION="${SLEEP_AFTER_EXECUTION:-3}"
DOMAIN="${DOMAIN:-all}"
TEST_ALL_META_PATH="${TEST_ALL_META_PATH:-evaluation_examples/test_all.json}"
CLIENT_PASSWORD="${CLIENT_PASSWORD:-password}"
HEADLESS="${HEADLESS:-1}"

if [[ ! -d "${OSWORLD_SRC}" ]]; then
  echo "[FATAL] OSWORLD_SRC not found: ${OSWORLD_SRC}" >&2
  echo "Clone OSWorld first, for example:" >&2
  echo "  git clone https://github.com/xlang-ai/OSWorld \"${OSWORLD_SRC}\"" >&2
  exit 1
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "[FATAL] OPENAI_BASE_URL is required." >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[FATAL] OPENAI_API_KEY is required." >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}"

cd "${OSWORLD_SRC}"

CMD=(
  python scripts/python/run_multienv.py
  --provider_name "${PROVIDER_NAME}"
  --action_space "${ACTION_SPACE}"
  --observation_type "${OBSERVATION_TYPE}"
  --model "${MODEL_ALIAS}"
  --num_envs "${NUM_ENVS}"
  --max_steps "${MAX_STEPS}"
  --max_trajectory_length "${MAX_TRAJECTORY_LENGTH}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION}"
  --domain "${DOMAIN}"
  --test_all_meta_path "${TEST_ALL_META_PATH}"
  --result_dir "${RESULT_ROOT}"
  --client_password "${CLIENT_PASSWORD}"
)

if [[ "${HEADLESS}" == "1" ]]; then
  CMD+=(--headless)
fi

# Optional CSV list: "local1:guest1,local2:guest2"
if [[ -n "${VM_SECRET_MOUNTS:-}" ]]; then
  IFS=',' read -r -a _mounts <<< "${VM_SECRET_MOUNTS}"
  for _mount in "${_mounts[@]}"; do
    [[ -n "${_mount}" ]] && CMD+=(--vm_secret_mount "${_mount}")
  done
fi

if [[ -n "${PATH_TO_VM:-}" ]]; then
  CMD+=(--path_to_vm "${PATH_TO_VM}")
fi

echo "=== OSWorld eval config ==="
echo "OSWORLD_SRC:        ${OSWORLD_SRC}"
echo "RESULT_ROOT:        ${RESULT_ROOT}"
echo "MODEL_ALIAS:        ${MODEL_ALIAS}"
echo "OPENAI_BASE_URL:    ${OPENAI_BASE_URL}"
echo "PROVIDER_NAME:      ${PROVIDER_NAME}"
echo "OBSERVATION_TYPE:   ${OBSERVATION_TYPE}"
echo "NUM_ENVS:           ${NUM_ENVS}"
echo "MAX_STEPS:          ${MAX_STEPS}"
echo "DOMAIN:             ${DOMAIN}"
echo

"${CMD[@]}"

python show_result.py \
  --action_space "${ACTION_SPACE}" \
  --model "${MODEL_ALIAS}" \
  --observation_type "${OBSERVATION_TYPE}" \
  --result_dir "${RESULT_ROOT}" \
  --detailed
