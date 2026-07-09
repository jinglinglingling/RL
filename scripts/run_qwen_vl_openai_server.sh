#!/usr/bin/env bash
set -euo pipefail

# Start an OpenAI-compatible vLLM server for Qwen-VL.
# OSWorld uses the OpenAI-compatible path for models starting with "gpt",
# so we expose served-model-name as gpt-4o by default.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-4o}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL_NAME}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${TRUST_REMOTE_CODE:-}" && "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

echo "=== Qwen-VL OpenAI-compatible server ==="
echo "MODEL_NAME:              ${MODEL_NAME}"
echo "SERVED_MODEL_NAME:       ${SERVED_MODEL_NAME}"
echo "HOST:                    ${HOST}"
echo "PORT:                    ${PORT}"
echo "MAX_MODEL_LEN:           ${MAX_MODEL_LEN}"
echo "TENSOR_PARALLEL_SIZE:    ${TENSOR_PARALLEL_SIZE}"
echo "GPU_MEMORY_UTILIZATION:  ${GPU_MEMORY_UTILIZATION}"
echo
echo "Use with:"
echo "  OPENAI_BASE_URL=http://${HOST}:${PORT}/v1"
echo "  OPENAI_API_KEY=EMPTY"
echo

"${CMD[@]}"
