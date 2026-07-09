#!/usr/bin/env bash
# Submit OSWorld collect+convert as a Slurm GPU job.
#
# Usage:
#   bash scripts/slurm/submit_osworld_collect_convert.sh
#   SBATCH_PARTITION=interactive GPUS_PER_NODE=1 bash scripts/slurm/submit_osworld_collect_convert.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_osworld_collect_convert_on_node.sh"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  chmod +x "${RUN_SCRIPT}"
fi

NEMORL_LOG_ROOT="${NEMORL_LOG_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/sbatch_histrory}"
mkdir -p "${NEMORL_LOG_ROOT}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-interactive}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_NODES="${SBATCH_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-16}"
SBATCH_MEM="${SBATCH_MEM:-0}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

JOB_NAME_BASE="${JOB_NAME_BASE:-osw-collect}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${RUN_ID}}"

JOB_SCRIPT="${NEMORL_LOG_ROOT}/${JOB_NAME}.sbatch.sh"
SLURM_OUT="${NEMORL_LOG_ROOT}/slurm-%j.out"

write_export() {
  local var_name="$1"
  local value="${2-}"
  printf "export %s=%q\n" "${var_name}" "${value}" >> "${JOB_SCRIPT}"
}

cat > "${JOB_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF

# Core paths
write_export "PROJECT_ROOT" "${PROJECT_ROOT}"
write_export "OSWORLD_SRC" "${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}"
write_export "OSWORLD_VENV" "${OSWORLD_VENV:-${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}/.venv}"
write_export "PYTHON_BIN" "${PYTHON_BIN:-python}"
write_export "RESULT_ROOT" "${RESULT_ROOT:-${PROJECT_ROOT}/runs/osworld-results}"
write_export "EXAMPLES_ROOT" "${EXAMPLES_ROOT:-${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}/evaluation_examples/examples}"
write_export "TRAIN_DATA" "${TRAIN_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_train.jsonl}"
write_export "VAL_DATA" "${VAL_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_val.jsonl}"

# Stage flags
write_export "RUN_COLLECT" "${RUN_COLLECT:-1}"
write_export "RUN_CONVERT" "${RUN_CONVERT:-1}"

# Collect knobs
write_export "MODEL_ALIAS" "${MODEL_ALIAS:-gpt-4o}"
write_export "PROVIDER_NAME" "${PROVIDER_NAME:-docker}"
write_export "ACTION_SPACE" "${ACTION_SPACE:-pyautogui}"
write_export "OBSERVATION_TYPE" "${OBSERVATION_TYPE:-screenshot}"
write_export "NUM_ENVS" "${NUM_ENVS:-1}"
write_export "MAX_STEPS" "${MAX_STEPS:-15}"
write_export "MAX_TRAJECTORY_LENGTH" "${MAX_TRAJECTORY_LENGTH:-3}"
write_export "SLEEP_AFTER_EXECUTION" "${SLEEP_AFTER_EXECUTION:-3}"
write_export "DOMAIN" "${DOMAIN:-all}"
write_export "TEST_ALL_META_PATH" "${TEST_ALL_META_PATH:-evaluation_examples/test_all.json}"
write_export "CLIENT_PASSWORD" "${CLIENT_PASSWORD:-password}"
write_export "HEADLESS" "${HEADLESS:-1}"
write_export "VM_SECRET_MOUNTS" "${VM_SECRET_MOUNTS:-}"
write_export "PATH_TO_VM" "${PATH_TO_VM:-}"
write_export "OSWORLD_APPTAINER_BIN" "${OSWORLD_APPTAINER_BIN:-}"
write_export "OSWORLD_APPTAINER_IMAGE" "${OSWORLD_APPTAINER_IMAGE:-}"
write_export "OSWORLD_APPTAINER_EXTRA_BINDS" "${OSWORLD_APPTAINER_EXTRA_BINDS:-}"
write_export "OSWORLD_APPTAINER_STORAGE_DIR" "${OSWORLD_APPTAINER_STORAGE_DIR:-}"
write_export "OSWORLD_APPTAINER_USE_FAKEROOT" "${OSWORLD_APPTAINER_USE_FAKEROOT:-1}"
write_export "OSWORLD_APPTAINER_LOG_DIR" "${OSWORLD_APPTAINER_LOG_DIR:-}"
write_export "OSWORLD_APPTAINER_PATCH_NGINX_PORT" "${OSWORLD_APPTAINER_PATCH_NGINX_PORT:-1}"
write_export "OSWORLD_APPTAINER_PATCH_NETWORK_PORTS" "${OSWORLD_APPTAINER_PATCH_NETWORK_PORTS:-1}"
write_export "OSWORLD_APPTAINER_PATCH_ROOT_CHECK" "${OSWORLD_APPTAINER_PATCH_ROOT_CHECK:-1}"
write_export "OSWORLD_APPTAINER_SSH_FORWARD_PORT" "${OSWORLD_APPTAINER_SSH_FORWARD_PORT:-}"
write_export "OSWORLD_APPTAINER_RDP_FORWARD_PORT" "${OSWORLD_APPTAINER_RDP_FORWARD_PORT:-}"
write_export "OSWORLD_APPTAINER_MONITOR_PORT" "${OSWORLD_APPTAINER_MONITOR_PORT:-}"
write_export "OSWORLD_APPTAINER_READY_TIMEOUT" "${OSWORLD_APPTAINER_READY_TIMEOUT:-}"
write_export "OSWORLD_APPTAINER_KVM" "${OSWORLD_APPTAINER_KVM:-}"
write_export "OSWORLD_APPTAINER_NETWORK" "${OSWORLD_APPTAINER_NETWORK:-user}"
write_export "OSWORLD_APPTAINER_USER_PORTS" "${OSWORLD_APPTAINER_USER_PORTS:-}"
write_export "OSWORLD_APPTAINER_CPU_CORES" "${OSWORLD_APPTAINER_CPU_CORES:-}"
write_export "OSWORLD_APPTAINER_RAM_SIZE" "${OSWORLD_APPTAINER_RAM_SIZE:-}"
write_export "OSWORLD_APPTAINER_DISK_SIZE" "${OSWORLD_APPTAINER_DISK_SIZE:-}"
write_export "OSWORLD_APPTAINER_BOOT_MODE" "${OSWORLD_APPTAINER_BOOT_MODE:-}"
write_export "OSWORLD_APPTAINER_SSH_DIAGNOSTICS" "${OSWORLD_APPTAINER_SSH_DIAGNOSTICS:-}"
write_export "OSWORLD_APPTAINER_AUTO_START_SERVER" "${OSWORLD_APPTAINER_AUTO_START_SERVER:-}"
write_export "OSWORLD_SERVER_PORT" "${OSWORLD_SERVER_PORT:-5000}"
write_export "OSWORLD_CHROMIUM_PORT" "${OSWORLD_CHROMIUM_PORT:-9222}"
write_export "OSWORLD_VNC_PORT" "${OSWORLD_VNC_PORT:-8006}"
write_export "OSWORLD_VLC_PORT" "${OSWORLD_VLC_PORT:-8080}"

# Endpoint / local vLLM
write_export "START_VLLM_SERVER" "${START_VLLM_SERVER:-1}"
write_export "VLLM_BACKEND" "${VLLM_BACKEND:-}"
write_export "OPENAI_BASE_URL" "${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
write_export "OPENAI_API_KEY" "${OPENAI_API_KEY:-EMPTY}"
write_export "VLLM_HOST" "${VLLM_HOST:-127.0.0.1}"
write_export "VLLM_PORT" "${VLLM_PORT:-8000}"
write_export "VLLM_MODEL_NAME" "${VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
write_export "VLLM_SERVED_MODEL_NAME" "${VLLM_SERVED_MODEL_NAME:-${MODEL_ALIAS:-gpt-4o}}"
write_export "VLLM_MAX_MODEL_LEN" "${VLLM_MAX_MODEL_LEN:-8192}"
write_export "VLLM_TP" "${VLLM_TP:-1}"
write_export "VLLM_GPU_MEMORY_UTILIZATION" "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
write_export "VLLM_LOG_FILE" "${VLLM_LOG_FILE:-${RESULT_ROOT:-${PROJECT_ROOT}/runs/osworld-results}/vllm-server.log}"
write_export "VLLM_WAIT_TIMEOUT_SEC" "${VLLM_WAIT_TIMEOUT_SEC:-180}"
write_export "VLLM_TRUST_REMOTE_CODE" "${VLLM_TRUST_REMOTE_CODE:-0}"
write_export "VLLM_MM_PROCESSOR_CACHE_GB" "${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
write_export "VLLM_MM_PROCESSOR_CACHE_TYPE" "${VLLM_MM_PROCESSOR_CACHE_TYPE:-}"
write_export "VLLM_DISABLE_MM_PREPROCESSOR_CACHE" "${VLLM_DISABLE_MM_PREPROCESSOR_CACHE:-0}"
write_export "VLLM_CONTAINER" "${VLLM_CONTAINER:-}"
write_export "VLLM_CONTAINER_MOUNTS" "${VLLM_CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}"
write_export "VLLM_CONTAINER_AS_ROOT" "${VLLM_CONTAINER_AS_ROOT:-1}"
write_export "VLLM_CONTAINER_RW" "${VLLM_CONTAINER_RW:-0}"
write_export "VLLM_CONTAINER_NAME" "${VLLM_CONTAINER_NAME:-}"
write_export "VLLM_CONTAINER_FORCE_CREATE" "${VLLM_CONTAINER_FORCE_CREATE:-0}"
write_export "VLLM_CONTAINER_REMOVE_ON_EXIT" "${VLLM_CONTAINER_REMOVE_ON_EXIT:-0}"
write_export "VLLM_CONTAINER_PYTHON" "${VLLM_CONTAINER_PYTHON:-/opt/nemo_rl_venv/bin/python}"
write_export "VLLM_PYTHON_SITE_DIR" "${VLLM_PYTHON_SITE_DIR:-/opt/nemo_rl_venv/lib/python3.12/site-packages}"
write_export "VLLM_VLLM_SOURCE_DIR" "${VLLM_VLLM_SOURCE_DIR:-/opt/nemo-rl/3rdparty/vllm}"
write_export "VLLM_PYTHON_INCLUDE_DIRS" "${VLLM_PYTHON_INCLUDE_DIRS:-/opt/nemo_rl_venv/include/python3.12,/usr/local/include/python3.12,/usr/include/python3.12}"
write_export "VLLM_AUTO_INSTALL_PYTHON_DEV" "${VLLM_AUTO_INSTALL_PYTHON_DEV:-1}"
write_export "VLLM_PYTHON_DEV_PACKAGES" "${VLLM_PYTHON_DEV_PACKAGES:-python3.12-dev python3-dev}"
write_export "VLLM_CONTAINER_RUNTIME" "${VLLM_CONTAINER_RUNTIME:-auto}"
write_export "VLLM_APPTAINER_IMAGE" "${VLLM_APPTAINER_IMAGE:-${VLLM_CONTAINER:-}}"

# Convert knobs
write_export "VAL_RATIO" "${VAL_RATIO:-0.1}"
write_export "MIN_EPISODE_SCORE" "${MIN_EPISODE_SCORE:-1.0}"
write_export "INCLUDE_FAILED" "${INCLUDE_FAILED:-0}"
write_export "ALLOW_MISSING_IMAGE" "${ALLOW_MISSING_IMAGE:-0}"
write_export "MAX_SAMPLES" "${MAX_SAMPLES:-}"

printf "bash %q\n" "${RUN_SCRIPT}" >> "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

SBATCH_CMD=(
  sbatch
  --parsable
  --nodes="${SBATCH_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --job-name="${JOB_NAME}"
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --gres="gpu:${GPUS_PER_NODE}"
  --cpus-per-task="${SBATCH_CPUS_PER_TASK}"
  --output="${SLURM_OUT}"
)

if [[ "${SBATCH_MEM}" != "0" ]]; then
  SBATCH_CMD+=(--mem="${SBATCH_MEM}")
fi
if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
  SBATCH_CMD+=(--constraint="${SBATCH_CONSTRAINT}")
fi
SBATCH_CMD+=("${JOB_SCRIPT}")

echo "=== Submit OSWorld collect+convert ==="
echo "  Job name:    ${JOB_NAME}"
echo "  Account:     ${SBATCH_ACCOUNT}"
echo "  Partition:   ${SBATCH_PARTITION}"
echo "  Time:        ${SBATCH_TIME}"
echo "  Nodes/GPU:   ${SBATCH_NODES} x ${GPUS_PER_NODE}"
echo "  Script:      ${JOB_SCRIPT}"
echo

JOB_ID="$("${SBATCH_CMD[@]}")"
echo "Submitted job ${JOB_ID}"
echo "  Slurm log:   ${NEMORL_LOG_ROOT}/slurm-${JOB_ID}.out"
echo "  Job script:  ${JOB_SCRIPT}"
echo "  Monitor:     tail -f ${NEMORL_LOG_ROOT}/slurm-${JOB_ID}.out"
echo "  Stop:        scancel ${JOB_ID}"
