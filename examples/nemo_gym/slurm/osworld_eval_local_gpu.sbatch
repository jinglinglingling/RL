#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-osworld-eval-%j.out

set -Eeuo pipefail

ROOT="${OSWORLD_REPO_ROOT:?Set OSWORLD_REPO_ROOT in the submit wrapper}"
VLLM_IMAGE="${VLLM_IMAGE:-${ROOT}/results/osworld-eval/vllm-openai-v0.23.0-x86_64.sqsh}"
VLLM_PORT="${VLLM_PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
STARTUP_TIMEOUT_S="${VLLM_STARTUP_TIMEOUT_S:-1800}"
LOG_DIR="${OSWORLD_EVAL_LOG_DIR:-${ROOT}/results/osworld-eval/logs/${SLURM_JOB_ID}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/.cache/uv}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Pyxis otherwise imports the large vLLM image through the 10 GiB NFS home
# cache. Keep all Enroot staging on the compute node's local NVMe.
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/tmp/enroot-cache-${USER}}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot-data-${USER}-${SLURM_JOB_ID}}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-${USER}-${SLURM_JOB_ID}}"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-/tmp/enroot-temp-${USER}-${SLURM_JOB_ID}}"
# The cluster-wide Enroot config routes registry traffic through a container
# cache whose TLS chain currently fails. Direct registry access is healthy.
DOCKER_NO_PROXY="registry-1.docker.io,auth.docker.io"
export no_proxy="${no_proxy:+${no_proxy},}${DOCKER_NO_PROXY}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${DOCKER_NO_PROXY}"

mkdir -p \
  "${LOG_DIR}" \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${ENROOT_CACHE_PATH}" \
  "${ENROOT_DATA_PATH}" \
  "${ENROOT_RUNTIME_PATH}" \
  "${ENROOT_TEMP_PATH}"

GRES_ARG=()
if sinfo -p "${SLURM_JOB_PARTITION}" -h -o "%G" | grep -q "gpu:"; then
  GRES_ARG=(--gres="gpu:${TP_SIZE}")
fi

cleanup() {
  if [[ -n "${VLLM_STEP_PID:-}" ]]; then
    kill "${VLLM_STEP_PID}" 2>/dev/null || true
    wait "${VLLM_STEP_PID}" 2>/dev/null || true
  fi
  rm -rf \
    "${ENROOT_DATA_PATH}" \
    "${ENROOT_RUNTIME_PATH}" \
    "${ENROOT_TEMP_PATH}"
}
trap cleanup EXIT INT TERM

echo "Starting vLLM ${VLLM_IMAGE} with TP=${TP_SIZE}"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
  "${GRES_ARG[@]}" \
  --export="ALL,no_proxy=${no_proxy},NO_PROXY=${NO_PROXY}" \
  --container-image="${VLLM_IMAGE}" \
  --container-mounts="/lustre:/lustre" \
  --container-workdir="${ROOT}" \
  --no-container-mount-home \
  bash examples/nemo_gym/serve_osworld_eval_vllm.sh \
  >"${LOG_DIR}/vllm.log" 2>&1 &
VLLM_STEP_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${VLLM_STEP_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see ${LOG_DIR}/vllm.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM; see ${LOG_DIR}/vllm.log" >&2
    exit 1
  fi
  sleep 10
done

if [[ "${OSWORLD_SERVE_ONLY:-0}" == "1" ]]; then
  echo "vLLM serving preflight passed"
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models"
  echo
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME:-vllm_local}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}],
      \"temperature\": 0,
      \"max_tokens\": 64
    }" \
    | tee "${LOG_DIR}/chat-completion-smoke.json" \
    | jq -e '.choices | length == 1' >/dev/null
  echo "vLLM chat-completion preflight passed"
  exit 0
fi

echo "vLLM is ready; starting frozen-model OSWorld evaluation"
export POLICY_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
export POLICY_MODEL_NAME="${SERVED_MODEL_NAME:-vllm_local}"

cd "${ROOT}"
bash examples/nemo_gym/run_osworld_eval.sh
