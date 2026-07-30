#!/usr/bin/env bash
# Serve the frozen Nemotron Omni checkpoint for OSWorld evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vllm_local}"
TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-64000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REASONING_PARSER_PLUGIN="${REASONING_PARSER_PLUGIN:-${ROOT}/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed; run this inside vllm/vllm-openai:v0.23.0" >&2
  exit 2
fi

if [[ ! -f "${REASONING_PARSER_PLUGIN}" ]]; then
  echo "Reasoning parser plugin not found: ${REASONING_PARSER_PLUGIN}" >&2
  exit 2
fi

if [[ "${HF_HUB_OFFLINE:-0}" == "1" && "${MODEL_PATH}" == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" ]]; then
  MODEL_CACHE="${HF_HOME:?HF_HOME is required in offline mode}/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
  MODEL_REVISION="$(<"${MODEL_CACHE}/refs/main")"
  MODEL_SNAPSHOT="${MODEL_CACHE}/snapshots/${MODEL_REVISION}"
  # vLLM 0.23 resolves an offline model ID to the symlinked snapshot path.
  # Transformers then resolves the main remote-code file into blobs/ and looks
  # for its relative Python imports beside it. Materialize those small source
  # files under their import names; model weights remain shared via symlinks.
  for source in "${MODEL_SNAPSHOT}"/*.py; do
    cp --dereference "${source}" "${MODEL_CACHE}/blobs/$(basename "${source}")"
  done
fi

exec vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --chat-template-content-format string \
  --reasoning-parser nano_v3 \
  --reasoning-parser-plugin "${REASONING_PARSER_PLUGIN}" \
  --structured-outputs-config.backend xgrammar \
  --mamba-ssm-cache-dtype float32 \
  --limit-mm-per-prompt '{"image":3}'
