#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PIPELINE_ENV_FILE="${PIPELINE_ENV_FILE:-${PROJECT_ROOT}/configs/pipeline.env}"
if [[ -f "${PIPELINE_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${PIPELINE_ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OSWORLD_SRC="${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}"
NEMORL_ROOT="${NEMORL_ROOT:-${PROJECT_ROOT}/../Nemo-RL-main-1/RL-merge-2689}"

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/runs/osworld-results}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-${OSWORLD_SRC}/evaluation_examples/examples}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_val.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/osworld_grpo_qwen_vl}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/osworld_grpo_qwen_vl}"

PIPELINE_STATE_DIR="${PIPELINE_STATE_DIR:-${PROJECT_ROOT}/runs/pipeline_state}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"    # all|setup|collect|convert|train
FROM_STAGE="${FROM_STAGE:-}"               # optional when PIPELINE_STAGE=all
PIPELINE_RESUME="${PIPELINE_RESUME:-1}"    # 1 skip completed stage
PIPELINE_FORCE="${PIPELINE_FORCE:-0}"      # 1 rerun stage even with marker

# Converter settings
VAL_RATIO="${VAL_RATIO:-0.1}"
MIN_EPISODE_SCORE="${MIN_EPISODE_SCORE:-1.0}"
INCLUDE_FAILED="${INCLUDE_FAILED:-0}"
ALLOW_MISSING_IMAGE="${ALLOW_MISSING_IMAGE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# setup stage options
AUTO_CLONE_OSWORLD="${AUTO_CLONE_OSWORLD:-0}"
INSTALL_WORKSPACE_DEPS="${INSTALL_WORKSPACE_DEPS:-1}"
INSTALL_OSWORLD_DEPS="${INSTALL_OSWORLD_DEPS:-1}"
OSWORLD_REPO_URL="${OSWORLD_REPO_URL:-https://github.com/xlang-ai/OSWorld.git}"

# Training overrides appended as a single shell string split by spaces.
NEMORL_OVERRIDES="${NEMORL_OVERRIDES:-}"
TRAIN_ARGS=()

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/e2e_vlm_rl_pipeline.sh [options] [-- <hydra overrides>]

Options:
  --stage <all|setup|collect|convert|train>   Run only one stage or all (default: all)
  --from-stage <setup|collect|convert|train>  With --stage all, start from this stage
  --resume                                     Skip already completed stages (default)
  --no-resume                                  Do not skip completed stages
  --force                                      Force rerun target stage(s)
  -h, --help                                   Show this help

Examples:
  bash scripts/e2e_vlm_rl_pipeline.sh --stage all --from-stage collect
  bash scripts/e2e_vlm_rl_pipeline.sh --stage train -- grpo.max_num_steps=100
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      PIPELINE_STAGE="$2"
      shift 2
      ;;
    --from-stage)
      FROM_STAGE="$2"
      shift 2
      ;;
    --resume)
      PIPELINE_RESUME=1
      shift
      ;;
    --no-resume)
      PIPELINE_RESUME=0
      shift
      ;;
    --force)
      PIPELINE_FORCE=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      echo "[FATAL] Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${NEMORL_OVERRIDES}" ]]; then
  read -r -a _extra_overrides <<< "${NEMORL_OVERRIDES}"
  TRAIN_ARGS+=("${_extra_overrides[@]}")
fi

stage_index() {
  case "$1" in
    setup) echo 0 ;;
    collect) echo 1 ;;
    convert) echo 2 ;;
    train) echo 3 ;;
    *)
      echo "[FATAL] Invalid stage: $1" >&2
      exit 1
      ;;
  esac
}

should_run_stage() {
  local stage="$1"
  if [[ "${PIPELINE_STAGE}" != "all" ]]; then
    [[ "${stage}" == "${PIPELINE_STAGE}" ]]
    return
  fi
  if [[ -z "${FROM_STAGE}" ]]; then
    return 0
  fi
  local stage_i from_i
  stage_i="$(stage_index "${stage}")"
  from_i="$(stage_index "${FROM_STAGE}")"
  [[ "${stage_i}" -ge "${from_i}" ]]
}

run_stage() {
  local stage="$1"
  shift
  local marker="${PIPELINE_STATE_DIR}/${stage}.done"

  if [[ "${PIPELINE_FORCE}" != "1" && "${PIPELINE_RESUME}" == "1" && -f "${marker}" ]]; then
    echo "[SKIP] stage=${stage} marker exists at ${marker}"
    return 0
  fi

  echo
  echo "========== RUN STAGE: ${stage} =========="
  "$@"
  mkdir -p "${PIPELINE_STATE_DIR}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${marker}"
  echo "[DONE] stage=${stage}"
}

check_openai_endpoint() {
  if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[FATAL] OPENAI_BASE_URL and OPENAI_API_KEY are required for collect stage." >&2
    exit 1
  fi

  "${PYTHON_BIN}" - <<'PY'
import os
import sys
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
if base.endswith("/v1"):
    url = base + "/models"
else:
    url = base + "/v1/models"
req = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
)
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        code = r.getcode()
except Exception as e:
    print(f"[FATAL] OpenAI-compatible endpoint check failed: {e}", file=sys.stderr)
    print(f"Checked URL: {url}", file=sys.stderr)
    sys.exit(1)
print(f"[OK] OpenAI-compatible endpoint reachable: {url} (status={code})")
PY
}

echo "=== E2E VLM RL Pipeline ==="
echo "PROJECT_ROOT:        ${PROJECT_ROOT}"
echo "PIPELINE_ENV_FILE:   ${PIPELINE_ENV_FILE}"
echo "PIPELINE_STAGE:      ${PIPELINE_STAGE}"
echo "FROM_STAGE:          ${FROM_STAGE:-<none>}"
echo "PIPELINE_RESUME:     ${PIPELINE_RESUME}"
echo "PIPELINE_FORCE:      ${PIPELINE_FORCE}"
echo "PIPELINE_STATE_DIR:  ${PIPELINE_STATE_DIR}"
echo "OSWORLD_SRC:         ${OSWORLD_SRC}"
echo "NEMORL_ROOT:         ${NEMORL_ROOT}"
echo "RESULT_ROOT:         ${RESULT_ROOT}"
echo "TRAIN_DATA:          ${TRAIN_DATA}"
echo "VAL_DATA:            ${VAL_DATA}"
echo "CHECKPOINT_DIR:      ${CHECKPOINT_DIR}"
echo "LOG_DIR:             ${LOG_DIR}"
echo

if should_run_stage "setup"; then
  run_stage "setup" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      OSWORLD_SRC="${OSWORLD_SRC}" \
      NEMORL_ROOT="${NEMORL_ROOT}" \
      AUTO_CLONE_OSWORLD="${AUTO_CLONE_OSWORLD}" \
      INSTALL_WORKSPACE_DEPS="${INSTALL_WORKSPACE_DEPS}" \
      INSTALL_OSWORLD_DEPS="${INSTALL_OSWORLD_DEPS}" \
      OSWORLD_REPO_URL="${OSWORLD_REPO_URL}" \
      bash "${SCRIPT_DIR}/setup_osworld_nemorl_env.sh"
fi

if should_run_stage "collect"; then
  check_openai_endpoint
  run_stage "collect" \
    env \
      OSWORLD_SRC="${OSWORLD_SRC}" \
      RESULT_ROOT="${RESULT_ROOT}" \
      bash "${SCRIPT_DIR}/run_osworld_eval_qwen_vl.sh"
fi

if should_run_stage "convert"; then
  CONVERT_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_osworld_results_to_nemorl.py"
    --results-root "${RESULT_ROOT}"
    --examples-root "${EXAMPLES_ROOT}"
    --output-train "${TRAIN_DATA}"
    --output-val "${VAL_DATA}"
    --val-ratio "${VAL_RATIO}"
    --min-episode-score "${MIN_EPISODE_SCORE}"
  )
  if [[ "${INCLUDE_FAILED}" == "1" ]]; then
    CONVERT_CMD+=(--include-failed)
  fi
  if [[ "${ALLOW_MISSING_IMAGE}" == "1" ]]; then
    CONVERT_CMD+=(--allow-missing-image)
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    CONVERT_CMD+=(--max-samples "${MAX_SAMPLES}")
  fi
  run_stage "convert" "${CONVERT_CMD[@]}"
fi

if should_run_stage "train"; then
  TRAIN_CMD=(
    env
    PYTHON_BIN="${PYTHON_BIN}"
    NEMORL_ROOT="${NEMORL_ROOT}"
    TRAIN_DATA="${TRAIN_DATA}"
    VAL_DATA="${VAL_DATA}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR}"
    LOG_DIR="${LOG_DIR}"
    bash "${SCRIPT_DIR}/run_nemorl_osworld_grpo.sh"
  )
  if [[ "${#TRAIN_ARGS[@]}" -gt 0 ]]; then
    TRAIN_CMD+=("${TRAIN_ARGS[@]}")
  fi
  run_stage "train" "${TRAIN_CMD[@]}"
fi

echo
echo "✅ Pipeline finished."
echo "State markers: ${PIPELINE_STATE_DIR}"
