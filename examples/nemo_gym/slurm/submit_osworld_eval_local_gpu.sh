#!/usr/bin/env bash
# Submit frozen-model OSWorld evaluation: local TP8 vLLM + remote cell-2 VMs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
SBATCH_SCRIPT="${ROOT}/examples/nemo_gym/slurm/osworld_eval_local_gpu.sbatch"

if [[ "${OSWORLD_SERVE_ONLY:-0}" != "1" ]]; then
  : "${OPENSANDBOX_DOMAIN:?Set the cell-2 OpenSandbox host}"
  : "${OPENSANDBOX_API_KEY:?Set the cell-2 OpenSandbox API key}"
fi

export OSWORLD_REPO_ROOT="${ROOT}"
export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-osworld-kvm}"
export OSWORLD_EVAL_INPUT="${OSWORLD_EVAL_INPUT:-resources_servers/osworld/data/smoke.jsonl}"
export OSWORLD_EVAL_OUTPUT="${OSWORLD_EVAL_OUTPUT:-results/osworld-eval/local-tp8-${RANDOM}.jsonl}"
export OSWORLD_EVAL_CONCURRENCY="${OSWORLD_EVAL_CONCURRENCY:-4}"
export MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vllm_local}"
export TP_SIZE="${TP_SIZE:-8}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_IMAGE="${VLLM_IMAGE:-${ROOT}/results/osworld-eval/vllm-openai-v0.23.0-x86_64.sqsh}"
if [[ ! -f "${VLLM_IMAGE}" && "${VLLM_IMAGE}" == /* ]]; then
  echo "vLLM squashfs image not found: ${VLLM_IMAGE}" >&2
  exit 2
fi
if [[ -z "${CURL_CA_BUNDLE:-}" ]] && command -v curl-config >/dev/null 2>&1; then
  export CURL_CA_BUNDLE
  CURL_CA_BUNDLE="$(curl-config --ca)"
fi
if [[ -n "${CURL_CA_BUNDLE:-}" ]]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-${CURL_CA_BUNDLE}}"
fi
# Cursor's command sandbox exports private restoration state that is only
# meaningful on the login host. Do not propagate it through sbatch --export=ALL.
unset BASH_ENV ENV
while IFS= read -r cursor_var; do
  unset "${cursor_var}"
done < <(compgen -e "__CURSOR_SANDBOX_" || true)
while IFS= read -r env_var; do
  if [[ "${!env_var-}" == /tmp/cursor-sandbox-cache/* ]]; then
    unset "${env_var}"
  fi
done < <(compgen -e)
export UV_CACHE_DIR="${ROOT}/.cache/uv"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
JOB_NAME="${JOB_NAME:-osworld-eval-local-tp8}"
SBATCH_EXTRA_ARGS=()
if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
  SBATCH_EXTRA_ARGS+=(--test-only)
fi
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--dependency="${SBATCH_DEPENDENCY}")
fi

cat <<EOF
Submitting OSWorld frozen-model evaluation
  account/partition: ${SBATCH_ACCOUNT}/${SBATCH_PARTITION}
  model:             ${MODEL_PATH}
  serving:           ${VLLM_IMAGE}, TP=${TP_SIZE}
  mode:              $(if [[ "${OSWORLD_SERVE_ONLY:-0}" == "1" ]]; then echo "serve-only preflight"; else echo "serve + cell-2 eval"; fi)
  sandbox:           ${OPENSANDBOX_DOMAIN:-<not used>}, pool=${OSWORLD_POOL_REF}
  input:             ${OSWORLD_EVAL_INPUT}
  output:            ${OSWORLD_EVAL_OUTPUT}
  concurrency:       ${OSWORLD_EVAL_CONCURRENCY}
EOF

exec sbatch \
  "${SBATCH_EXTRA_ARGS[@]}" \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --job-name="${JOB_NAME}" \
  --export=ALL \
  "${SBATCH_SCRIPT}"
