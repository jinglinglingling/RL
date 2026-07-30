#!/usr/bin/env bash
# Run frozen-model OSWorld evaluation against an OpenSandbox KVM pool.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GYM="${ROOT}/3rdparty/Gym-workspace/Gym"
export UV_OVERRIDE="${UV_OVERRIDE:-${ROOT}/examples/nemo_gym/osworld-uv-overrides.txt}"

: "${OPENSANDBOX_DOMAIN:?Set the cell-2 OpenSandbox host}"
: "${OPENSANDBOX_API_KEY:?Set the cell-2 OpenSandbox API key}"
: "${POLICY_BASE_URL:?Set an OpenAI-compatible /v1 model endpoint}"

INPUT="${OSWORLD_EVAL_INPUT:-resources_servers/osworld/data/smoke.jsonl}"
OUTPUT="${OSWORLD_EVAL_OUTPUT:-${ROOT}/results/osworld-eval/results.jsonl}"
CONCURRENCY="${OSWORLD_EVAL_CONCURRENCY:-4}"
MODEL_NAME="${POLICY_MODEL_NAME:-vllm_local}"
MODEL_API_KEY="${POLICY_API_KEY:-EMPTY}"

if [[ "${INPUT}" = /* ]]; then
  INPUT_ABS="${INPUT}"
else
  INPUT_ABS="${GYM}/${INPUT}"
fi

if [[ ! -f "${INPUT_ABS}" ]]; then
  echo "OSWorld eval input does not exist: ${INPUT_ABS}" >&2
  exit 2
fi

if [[ "${OUTPUT}" = /* ]]; then
  OUTPUT_ABS="${OUTPUT}"
else
  OUTPUT_ABS="${ROOT}/${OUTPUT}"
fi

mkdir -p "$(dirname "${OUTPUT_ABS}")"
OVERLAY="$(mktemp /tmp/osworld-eval-config.XXXXXX.yaml)"
trap 'rm -f "${OVERLAY}"' EXIT

cat >"${OVERLAY}" <<EOF
nemotron_osworld:
  responses_api_agents:
    nemotron_osworld:
      max_steps: ${OSWORLD_EVAL_MAX_STEPS:-100}
      debug_trajectory_dir: "${OSWORLD_DEBUG_TRAJ_DIR:-}"
      datasets:
      - name: osworld_eval
        type: validation
        jsonl_fpath: ${INPUT_ABS}
        license: Apache 2.0
policy_model:
  responses_api_models:
    vllm_model:
      preserve_reasoning_content: true
EOF

RESUME_ARGS=()
if [[ "${OSWORLD_EVAL_RESUME:-0}" == "1" && -s "${OUTPUT_ABS}" ]]; then
  RESUME_ARGS+=(--resume)
fi
LIMIT_ARGS=()
if [[ -n "${OSWORLD_EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS+=("+limit=${OSWORLD_EVAL_LIMIT}")
fi

cd "${GYM}"
exec "${ROOT}/.venv/bin/gym" eval run \
  --config responses_api_agents/nemotron_osworld/configs/nemotron_osworld.yaml \
  --config resources_servers/osworld/configs/osworld.yaml \
  --config resources_servers/osworld/configs/opensandbox_osworld.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --config "${OVERLAY}" \
  --agent nemotron_osworld \
  --model "${MODEL_NAME}" \
  --model-url "${POLICY_BASE_URL}" \
  --model-api-key "${MODEL_API_KEY}" \
  --input "${INPUT_ABS}" \
  --split validation \
  --concurrency "${CONCURRENCY}" \
  --output "${OUTPUT_ABS}" \
  "${RESUME_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
    "+skip_venv_if_present=${NEMO_GYM_SKIP_VENV_IF_PRESENT:-false}"
