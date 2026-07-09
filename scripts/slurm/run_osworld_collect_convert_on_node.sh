#!/usr/bin/env bash
# Run OSWorld collect + convert on a GPU node (interactive or batch).
#
# Typical use:
#   # on an allocated GPU node:
#   bash scripts/slurm/run_osworld_collect_convert_on_node.sh
#
# Environment knobs:
#   START_VLLM_SERVER=1|0
#   RUN_COLLECT=1|0
#   RUN_CONVERT=1|0
#   MODEL_ALIAS=gpt-4o
#   PROVIDER_NAME=docker
#   DOMAIN=all
#   TEST_ALL_META_PATH=evaluation_examples/test_all.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OSWORLD_SRC="${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}"
OSWORLD_VENV="${OSWORLD_VENV:-${OSWORLD_SRC}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/runs/osworld-results}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-${OSWORLD_SRC}/evaluation_examples/examples}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_val.jsonl}"

RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"

# ---------------------------
# Collection knobs
# ---------------------------
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
VM_SECRET_MOUNTS="${VM_SECRET_MOUNTS:-}"
PATH_TO_VM="${PATH_TO_VM:-}"
OSWORLD_APPTAINER_BIN="${OSWORLD_APPTAINER_BIN:-}"
OSWORLD_APPTAINER_IMAGE="${OSWORLD_APPTAINER_IMAGE:-}"
OSWORLD_APPTAINER_EXTRA_BINDS="${OSWORLD_APPTAINER_EXTRA_BINDS:-}"
OSWORLD_APPTAINER_STORAGE_DIR="${OSWORLD_APPTAINER_STORAGE_DIR:-}"
OSWORLD_APPTAINER_USE_FAKEROOT="${OSWORLD_APPTAINER_USE_FAKEROOT:-1}"
OSWORLD_APPTAINER_LOG_DIR="${OSWORLD_APPTAINER_LOG_DIR:-}"
OSWORLD_APPTAINER_PATCH_NGINX_PORT="${OSWORLD_APPTAINER_PATCH_NGINX_PORT:-1}"
OSWORLD_APPTAINER_PATCH_NETWORK_PORTS="${OSWORLD_APPTAINER_PATCH_NETWORK_PORTS:-1}"
OSWORLD_APPTAINER_PATCH_ROOT_CHECK="${OSWORLD_APPTAINER_PATCH_ROOT_CHECK:-1}"
OSWORLD_APPTAINER_SSH_FORWARD_PORT="${OSWORLD_APPTAINER_SSH_FORWARD_PORT:-}"
OSWORLD_APPTAINER_RDP_FORWARD_PORT="${OSWORLD_APPTAINER_RDP_FORWARD_PORT:-}"
OSWORLD_APPTAINER_MONITOR_PORT="${OSWORLD_APPTAINER_MONITOR_PORT:-}"
OSWORLD_APPTAINER_READY_TIMEOUT="${OSWORLD_APPTAINER_READY_TIMEOUT:-}"
OSWORLD_APPTAINER_KVM="${OSWORLD_APPTAINER_KVM:-}"
OSWORLD_APPTAINER_NETWORK="${OSWORLD_APPTAINER_NETWORK:-user}"
OSWORLD_APPTAINER_USER_PORTS="${OSWORLD_APPTAINER_USER_PORTS:-}"
OSWORLD_APPTAINER_CPU_CORES="${OSWORLD_APPTAINER_CPU_CORES:-}"
OSWORLD_APPTAINER_RAM_SIZE="${OSWORLD_APPTAINER_RAM_SIZE:-}"
OSWORLD_APPTAINER_DISK_SIZE="${OSWORLD_APPTAINER_DISK_SIZE:-}"
OSWORLD_APPTAINER_BOOT_MODE="${OSWORLD_APPTAINER_BOOT_MODE:-}"
OSWORLD_APPTAINER_SSH_DIAGNOSTICS="${OSWORLD_APPTAINER_SSH_DIAGNOSTICS:-}"
OSWORLD_APPTAINER_AUTO_START_SERVER="${OSWORLD_APPTAINER_AUTO_START_SERVER:-}"
OSWORLD_APPTAINER_AUTO_PORTS_ON_CONFLICT="${OSWORLD_APPTAINER_AUTO_PORTS_ON_CONFLICT:-1}"
OSWORLD_APPTAINER_PORT_SCAN_SPAN="${OSWORLD_APPTAINER_PORT_SCAN_SPAN:-3000}"
OSWORLD_SERVER_PORT="${OSWORLD_SERVER_PORT:-5000}"
OSWORLD_CHROMIUM_PORT="${OSWORLD_CHROMIUM_PORT:-9222}"
OSWORLD_VNC_PORT="${OSWORLD_VNC_PORT:-8006}"
OSWORLD_VLC_PORT="${OSWORLD_VLC_PORT:-8080}"

# ---------------------------
# Optional vLLM server backend
# ---------------------------
# Backward-compat:
#   START_VLLM_SERVER=1 -> VLLM_BACKEND=host
#   START_VLLM_SERVER=0 -> VLLM_BACKEND=external
START_VLLM_SERVER="${START_VLLM_SERVER:-1}"
VLLM_BACKEND="${VLLM_BACKEND:-}"  # host|container|external
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL_ALIAS}}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-${RESULT_ROOT}/vllm-server.log}"
VLLM_WAIT_TIMEOUT_SEC="${VLLM_WAIT_TIMEOUT_SEC:-180}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"
VLLM_MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
VLLM_MM_PROCESSOR_CACHE_TYPE="${VLLM_MM_PROCESSOR_CACHE_TYPE:-}"
# Backward-compatible alias for old toggle semantics.
VLLM_DISABLE_MM_PREPROCESSOR_CACHE="${VLLM_DISABLE_MM_PREPROCESSOR_CACHE:-0}"
if [[ -z "${VLLM_MM_PROCESSOR_CACHE_GB}" && "${VLLM_DISABLE_MM_PREPROCESSOR_CACHE}" == "1" ]]; then
  VLLM_MM_PROCESSOR_CACHE_GB="0"
fi

# Container-backed vLLM (for ppo-style workflows)
VLLM_CONTAINER="${VLLM_CONTAINER:-}"
VLLM_CONTAINER_MOUNTS="${VLLM_CONTAINER_MOUNTS:-/lustre:/lustre,/home:/home}"
VLLM_CONTAINER_AS_ROOT="${VLLM_CONTAINER_AS_ROOT:-1}"
VLLM_CONTAINER_RW="${VLLM_CONTAINER_RW:-0}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-}"
VLLM_CONTAINER_FORCE_CREATE="${VLLM_CONTAINER_FORCE_CREATE:-0}"
VLLM_CONTAINER_REMOVE_ON_EXIT="${VLLM_CONTAINER_REMOVE_ON_EXIT:-0}"
VLLM_CONTAINER_PYTHON="${VLLM_CONTAINER_PYTHON:-/opt/nemo_rl_venv/bin/python}"
VLLM_PYTHON_SITE_DIR="${VLLM_PYTHON_SITE_DIR:-/opt/nemo_rl_venv/lib/python3.12/site-packages}"
VLLM_VLLM_SOURCE_DIR="${VLLM_VLLM_SOURCE_DIR:-/opt/nemo-rl/3rdparty/vllm}"
VLLM_PYTHON_INCLUDE_DIRS="${VLLM_PYTHON_INCLUDE_DIRS:-/opt/nemo_rl_venv/include/python3.12,/usr/local/include/python3.12,/usr/include/python3.12}"
VLLM_AUTO_INSTALL_PYTHON_DEV="${VLLM_AUTO_INSTALL_PYTHON_DEV:-1}"
VLLM_PYTHON_DEV_PACKAGES="${VLLM_PYTHON_DEV_PACKAGES:-python3.12-dev python3-dev}"
VLLM_CONTAINER_RUNTIME="${VLLM_CONTAINER_RUNTIME:-auto}" # auto|enroot|apptainer
VLLM_APPTAINER_IMAGE="${VLLM_APPTAINER_IMAGE:-${VLLM_CONTAINER}}"

# ---------------------------
# Conversion knobs
# ---------------------------
VAL_RATIO="${VAL_RATIO:-0.1}"
MIN_EPISODE_SCORE="${MIN_EPISODE_SCORE:-1.0}"
INCLUDE_FAILED="${INCLUDE_FAILED:-0}"
ALLOW_MISSING_IMAGE="${ALLOW_MISSING_IMAGE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ ! -d "${OSWORLD_SRC}" ]]; then
  echo "[FATAL] OSWORLD_SRC not found: ${OSWORLD_SRC}" >&2
  exit 1
fi
if [[ ! -f "${OSWORLD_VENV}/bin/activate" ]]; then
  echo "[FATAL] OSWorld venv not found: ${OSWORLD_VENV}" >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}" "$(dirname "${TRAIN_DATA}")"

if [[ -z "${VLLM_BACKEND}" ]]; then
  if [[ "${START_VLLM_SERVER}" == "1" ]]; then
    VLLM_BACKEND="host"
  else
    VLLM_BACKEND="external"
  fi
fi
if [[ "${VLLM_BACKEND}" != "host" && "${VLLM_BACKEND}" != "container" && "${VLLM_BACKEND}" != "external" ]]; then
  echo "[FATAL] Invalid VLLM_BACKEND=${VLLM_BACKEND} (expected host|container|external)" >&2
  exit 1
fi

if [[ "${PROVIDER_NAME}" == "apptainer" ]]; then
  if [[ -n "${OSWORLD_APPTAINER_BIN}" ]] && [[ ! -x "${OSWORLD_APPTAINER_BIN}" ]]; then
    echo "[FATAL] OSWORLD_APPTAINER_BIN is not executable: ${OSWORLD_APPTAINER_BIN}" >&2
    exit 1
  fi
  if [[ -n "${OSWORLD_APPTAINER_IMAGE}" ]] && [[ "${OSWORLD_APPTAINER_IMAGE}" != docker://* ]] && [[ ! -f "${OSWORLD_APPTAINER_IMAGE}" ]]; then
    echo "[FATAL] OSWORLD_APPTAINER_IMAGE not found: ${OSWORLD_APPTAINER_IMAGE}" >&2
    exit 1
  fi
fi

resolve_apptainer_ports() {
  if [[ "${PROVIDER_NAME}" != "apptainer" ]]; then
    return 0
  fi

  local resolved=""
  if ! resolved="$("${PYTHON_BIN}" - \
    "${OSWORLD_SERVER_PORT}" \
    "${OSWORLD_CHROMIUM_PORT}" \
    "${OSWORLD_VNC_PORT}" \
    "${OSWORLD_VLC_PORT}" \
    "${OSWORLD_APPTAINER_MONITOR_PORT}" \
    "${OSWORLD_APPTAINER_SSH_FORWARD_PORT}" \
    "${OSWORLD_APPTAINER_RDP_FORWARD_PORT}" \
    "${OSWORLD_APPTAINER_AUTO_PORTS_ON_CONFLICT}" \
    "${OSWORLD_APPTAINER_PORT_SCAN_SPAN}" <<'PY'
import socket
import sys


def parse_opt_int(value):
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


def can_bind(port):
    if port <= 0 or port > 65535:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def conflicts_for(mapping):
    keys = ("server", "chromium", "vnc", "vlc", "monitor", "ssh", "rdp")
    seen = {}
    conflicts = []
    for key in keys:
        port = mapping[key]
        if port in seen:
            conflicts.append((key, port, f"duplicates_{seen[port]}"))
            continue
        seen[port] = key
        if not can_bind(port):
            conflicts.append((key, port, "busy"))
    return conflicts


server = int(sys.argv[1])
chromium = int(sys.argv[2])
vnc = int(sys.argv[3])
vlc = int(sys.argv[4])
monitor = parse_opt_int(sys.argv[5])
ssh = parse_opt_int(sys.argv[6])
rdp = parse_opt_int(sys.argv[7])
auto_reassign = sys.argv[8] == "1"
scan_span = max(0, int(sys.argv[9]))

def normalize_offset(raw, default, low, high):
    if raw is None:
        return default
    if low <= raw <= high:
        return raw
    return default


vlc_offset = normalize_offset(vlc - vnc, 74, 1, 3000)
monitor_offset = normalize_offset((monitor - vnc) if monitor is not None else None, 100, 1, 4000)
ssh_offset = normalize_offset((ssh - vnc) if ssh is not None else None, 1000, 1, 20000)
rdp_offset = normalize_offset((rdp - vnc) if rdp is not None else None, 2000, 1, 30000)


def with_vnc_base(base):
    return {
        "server": server,
        "chromium": chromium,
        "vnc": base,
        "vlc": base + vlc_offset,
        "monitor": base + monitor_offset,
        "ssh": base + ssh_offset,
        "rdp": base + rdp_offset,
    }


def with_full_base(base):
    return {
        "server": base,
        "chromium": base + 1,
        "vnc": base + 100,
        "vlc": base + 174,
        "monitor": base + 200,
        "ssh": base + 1100,
        "rdp": base + 2100,
    }


def is_portset_in_range(mapping):
    return all(1 <= port <= 65535 for port in mapping.values())


initial = with_vnc_base(vnc)
initial_conflicts = conflicts_for(initial)
selected = initial
changed = False

if initial_conflicts:
    if not auto_reassign:
        detail = ",".join(f"{k}:{p}:{r}" for k, p, r in initial_conflicts)
        print(f"[FATAL] Port conflict detected and auto-reassign disabled: {detail}", file=sys.stderr)
        sys.exit(1)

    found = False

    # Attempt 1: only shift the VNC-derived family.
    start_base = max(10000, vnc)
    for delta in range(scan_span + 1):
        base = start_base + delta
        candidate = with_vnc_base(base)
        if not is_portset_in_range(candidate):
            break
        if not conflicts_for(candidate):
            selected = candidate
            changed = True
            found = True
            break

    # Attempt 2: re-seat all host ports together.
    if not found:
        start_base = max(20000, min(server, chromium, vnc))
        for delta in range(scan_span + 1):
            base = start_base + delta
            candidate = with_full_base(base)
            if not is_portset_in_range(candidate):
                break
            if not conflicts_for(candidate):
                selected = candidate
                changed = True
                found = True
                break

    if not found:
        detail = ",".join(f"{k}:{p}:{r}" for k, p, r in initial_conflicts)
        print(
            f"[FATAL] Could not find a conflict-free apptainer port set within span={scan_span}. "
            f"Initial conflicts: {detail}",
            file=sys.stderr,
        )
        sys.exit(1)

print(f"server={selected['server']}")
print(f"chromium={selected['chromium']}")
print(f"vnc={selected['vnc']}")
print(f"vlc={selected['vlc']}")
print(f"monitor={selected['monitor']}")
print(f"ssh={selected['ssh']}")
print(f"rdp={selected['rdp']}")
print(f"changed={1 if changed else 0}")
if initial_conflicts:
    print("initial_conflicts=" + ",".join(f"{k}:{p}:{r}" for k, p, r in initial_conflicts))
PY
  )"; then
    echo "[FATAL] Failed to resolve apptainer ports." >&2
    exit 1
  fi

  local changed="0"
  local initial_conflicts=""
  while IFS='=' read -r key value; do
    case "${key}" in
      server) OSWORLD_SERVER_PORT="${value}" ;;
      chromium) OSWORLD_CHROMIUM_PORT="${value}" ;;
      vnc) OSWORLD_VNC_PORT="${value}" ;;
      vlc) OSWORLD_VLC_PORT="${value}" ;;
      monitor) OSWORLD_APPTAINER_MONITOR_PORT="${value}" ;;
      ssh) OSWORLD_APPTAINER_SSH_FORWARD_PORT="${value}" ;;
      rdp) OSWORLD_APPTAINER_RDP_FORWARD_PORT="${value}" ;;
      changed) changed="${value}" ;;
      initial_conflicts) initial_conflicts="${value}" ;;
    esac
  done <<< "${resolved}"

  if [[ -z "${OSWORLD_APPTAINER_USER_PORTS}" || "${changed}" == "1" ]]; then
    OSWORLD_APPTAINER_USER_PORTS="${OSWORLD_SERVER_PORT},${OSWORLD_CHROMIUM_PORT}"
  fi

  if [[ "${changed}" == "1" ]]; then
    echo "[port-select] Initial conflict(s): ${initial_conflicts}"
    echo "[port-select] Reassigned apptainer ports: server=${OSWORLD_SERVER_PORT}, chromium=${OSWORLD_CHROMIUM_PORT}, vnc=${OSWORLD_VNC_PORT}, vlc=${OSWORLD_VLC_PORT}, monitor=${OSWORLD_APPTAINER_MONITOR_PORT}, ssh=${OSWORLD_APPTAINER_SSH_FORWARD_PORT}, rdp=${OSWORLD_APPTAINER_RDP_FORWARD_PORT}"
  fi
}

resolve_apptainer_ports

# If user passes a path under project root, normalize to absolute.
if [[ "${TEST_ALL_META_PATH}" != /* ]] && [[ -f "${PROJECT_ROOT}/${TEST_ALL_META_PATH}" ]]; then
  TEST_ALL_META_PATH="${PROJECT_ROOT}/${TEST_ALL_META_PATH}"
fi

echo "=== OSWorld collect+convert (node runtime) ==="
echo "PROJECT_ROOT:        ${PROJECT_ROOT}"
echo "OSWORLD_SRC:         ${OSWORLD_SRC}"
echo "OSWORLD_VENV:        ${OSWORLD_VENV}"
echo "RESULT_ROOT:         ${RESULT_ROOT}"
echo "TRAIN_DATA:          ${TRAIN_DATA}"
echo "VAL_DATA:            ${VAL_DATA}"
echo "RUN_COLLECT:         ${RUN_COLLECT}"
echo "RUN_CONVERT:         ${RUN_CONVERT}"
echo "VLLM_BACKEND:        ${VLLM_BACKEND}"
echo "OPENAI_BASE_URL:     ${OPENAI_BASE_URL}"
echo "MODEL_ALIAS:         ${MODEL_ALIAS}"
echo "PROVIDER_NAME:       ${PROVIDER_NAME}"
echo "OSW_APPTAINER_BIN:   ${OSWORLD_APPTAINER_BIN:-<auto>}"
echo "OSW_APPTAINER_IMAGE: ${OSWORLD_APPTAINER_IMAGE:-<docker://happysixd/osworld-docker:latest>}"
echo "OSW_APPTAINER_STORE: ${OSWORLD_APPTAINER_STORAGE_DIR:-<vm-parent-dir>}"
echo "OSW_APPTAINER_ROOT:  ${OSWORLD_APPTAINER_USE_FAKEROOT}"
echo "OSW_APPTAINER_WEB:   ${OSWORLD_APPTAINER_PATCH_NGINX_PORT}"
echo "OSW_APPTAINER_NETPT: ${OSWORLD_APPTAINER_PATCH_NETWORK_PORTS} (ssh=${OSWORLD_APPTAINER_SSH_FORWARD_PORT:-<auto-vnc+1000>},rdp=${OSWORLD_APPTAINER_RDP_FORWARD_PORT:-<auto-vnc+2000>})"
echo "OSW_APPTAINER_ROOTP: ${OSWORLD_APPTAINER_PATCH_ROOT_CHECK}"
echo "OSW_APPTAINER_MON:   ${OSWORLD_APPTAINER_MONITOR_PORT:-<vnc+100>}"
echo "OSW_APPTAINER_WAIT:  ${OSWORLD_APPTAINER_READY_TIMEOUT:-<default-300>}"
echo "OSW_APPTAINER_KVM:   ${OSWORLD_APPTAINER_KVM:-<auto>}"
echo "OSW_APPTAINER_NET:   ${OSWORLD_APPTAINER_NETWORK}"
echo "OSW_APPTAINER_RES:   cpu=${OSWORLD_APPTAINER_CPU_CORES:-<default-4>} ram=${OSWORLD_APPTAINER_RAM_SIZE:-<default-4G>} disk=${OSWORLD_APPTAINER_DISK_SIZE:-<default-32G>}"
echo "OSW_APPTAINER_BOOT:  ${OSWORLD_APPTAINER_BOOT_MODE:-<default-uefi>}"
echo "OSW_APPTAINER_DIAG:  ssh=${OSWORLD_APPTAINER_SSH_DIAGNOSTICS:-0} autostart=${OSWORLD_APPTAINER_AUTO_START_SERVER:-0}"
echo "VLLM_CONTAINER:      ${VLLM_CONTAINER:-<none>}"
echo "VLLM_RUNTIME_HINT:   ${VLLM_CONTAINER_RUNTIME}"
echo "VLLM_MM_CACHE:       gb=${VLLM_MM_PROCESSOR_CACHE_GB:-<default>} type=${VLLM_MM_PROCESSOR_CACHE_TYPE:-<default>}"
echo "CONVERT_FILTER:      min_score=${MIN_EPISODE_SCORE} include_failed=${INCLUDE_FAILED} allow_missing_image=${ALLOW_MISSING_IMAGE} max_samples=${MAX_SAMPLES:-<none>}"
echo

# shellcheck disable=SC1091
source "${OSWORLD_VENV}/bin/activate"

VLLM_PID=""
VLLM_SANDBOX_NAME=""
VLLM_SANDBOX_CREATED="0"
VLLM_RUNTIME_EFFECTIVE=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[cleanup] Stopping vLLM server pid=${VLLM_PID}"
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${VLLM_RUNTIME_EFFECTIVE}" == "enroot" && "${VLLM_CONTAINER_REMOVE_ON_EXIT}" == "1" && "${VLLM_SANDBOX_CREATED}" == "1" && -n "${VLLM_SANDBOX_NAME}" ]]; then
    echo "[cleanup] Removing enroot sandbox ${VLLM_SANDBOX_NAME}"
    enroot remove -f "${VLLM_SANDBOX_NAME}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_openai_endpoint() {
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
import time
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
timeout_sec = int(os.environ.get("VLLM_WAIT_TIMEOUT_SEC", "180"))
url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

deadline = time.time() + timeout_sec
last_err = "unknown"
while time.time() < deadline:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if 200 <= resp.getcode() < 500:
                print(f"[wait] OpenAI endpoint ready: {url}")
                sys.exit(0)
    except Exception as e:
        last_err = str(e)
    time.sleep(2)

print(f"[FATAL] Timed out waiting for endpoint: {url}", file=sys.stderr)
print(f"Last error: {last_err}", file=sys.stderr)
sys.exit(1)
PY
}

build_vllm_pythonpath_prefix() {
  local parts=()
  if [[ -n "${VLLM_PYTHON_SITE_DIR}" ]]; then
    parts+=("${VLLM_PYTHON_SITE_DIR}")
  fi
  if [[ -n "${VLLM_VLLM_SOURCE_DIR}" ]]; then
    parts+=("${VLLM_VLLM_SOURCE_DIR}")
  fi
  if (( ${#parts[@]} > 0 )); then
    local joined
    joined="$(IFS=:; echo "${parts[*]}")"
    echo "${joined}"
  fi
}

build_vllm_bootstrap_script() {
  local vllm_module_args_escaped="$1"
  local pythonpath_prefix="$2"
  cat <<EOF
set -euo pipefail
PRIMARY_PY='${VLLM_CONTAINER_PYTHON}'
PYTHONPATH_PREFIX='${pythonpath_prefix}'
PYTHON_INCLUDE_DIRS='${VLLM_PYTHON_INCLUDE_DIRS}'
AUTO_INSTALL_PYTHON_DEV='${VLLM_AUTO_INSTALL_PYTHON_DEV}'
PYTHON_DEV_PACKAGES='${VLLM_PYTHON_DEV_PACKAGES}'
if [[ -n "\${PYTHONPATH_PREFIX}" ]]; then
  export PYTHONPATH="\${PYTHONPATH_PREFIX}:\${PYTHONPATH:-}"
fi

find_python_header_and_export_cpath() {
  IFS=',' read -r -a _python_include_dirs <<< "\${PYTHON_INCLUDE_DIRS}"
  for _inc in "\${_python_include_dirs[@]}"; do
    if [[ -f "\${_inc}/Python.h" ]]; then
      export CPATH="\${_inc}:\${CPATH:-}"
      echo "[vllm-bootstrap] Using Python.h from: \${_inc}" >&2
      return 0
    fi
  done
  return 1
}

if ! find_python_header_and_export_cpath; then
  if [[ "\${AUTO_INSTALL_PYTHON_DEV}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
    echo "[vllm-bootstrap] Python.h missing; installing dev headers via apt-get ..." >&2
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y >/tmp/vllm-apt-update.log 2>&1 || true
    apt-get install -y \${PYTHON_DEV_PACKAGES} >/tmp/vllm-apt-install.log 2>&1 || true
    if ! find_python_header_and_export_cpath; then
      echo "[vllm-bootstrap] Python.h still missing after apt install." >&2
    fi
  fi
fi

if [[ -x "\${PRIMARY_PY}" ]] && "\${PRIMARY_PY}" -c 'import vllm' >/dev/null 2>&1; then
  eval "exec \\"\${PRIMARY_PY}\\" -m vllm.entrypoints.openai.api_server ${vllm_module_args_escaped}"
fi

if [[ -x /usr/bin/python3 ]] && /usr/bin/python3 -c 'import vllm' >/dev/null 2>&1; then
  eval "exec /usr/bin/python3 -m vllm.entrypoints.openai.api_server ${vllm_module_args_escaped}"
fi

echo '[FATAL] Could not launch vllm inside container' >&2
echo "Tried direct python: \${PRIMARY_PY}" >&2
echo "Tried system python3 with PYTHONPATH_PREFIX=\${PYTHONPATH_PREFIX}" >&2
exit 1
EOF
}

resolve_container_runtime() {
  case "${VLLM_CONTAINER_RUNTIME}" in
    enroot|apptainer)
      echo "${VLLM_CONTAINER_RUNTIME}"
      return 0
      ;;
    auto)
      if [[ "${VLLM_CONTAINER}" == *.sif ]] && command -v apptainer >/dev/null 2>&1; then
        echo "apptainer"
        return 0
      fi
      if command -v enroot >/dev/null 2>&1; then
        echo "enroot"
        return 0
      fi
      if command -v apptainer >/dev/null 2>&1; then
        echo "apptainer"
        return 0
      fi
      echo "none"
      return 0
      ;;
    *)
      echo "invalid"
      return 0
      ;;
  esac
}

start_container_vllm_with_enroot() {
  local vllm_module_args_escaped="$1"
  local pythonpath_prefix="$2"

  if ! command -v enroot >/dev/null 2>&1; then
    echo "[FATAL] VLLM runtime selected as enroot but enroot command not found" >&2
    exit 1
  fi

  # If VLLM_CONTAINER is an image path, create/reuse a named sandbox first.
  # This avoids direct image-start dependency on squashfuse.
  local enroot_target="${VLLM_CONTAINER}"
  if [[ -f "${VLLM_CONTAINER}" ]]; then
    if [[ -z "${VLLM_CONTAINER_NAME}" ]]; then
      local base
      base="$(basename "${VLLM_CONTAINER}")"
      base="${base//[^a-zA-Z0-9._-]/-}"
      VLLM_CONTAINER_NAME="osw-vllm-${base}"
    fi
    VLLM_SANDBOX_NAME="${VLLM_CONTAINER_NAME}"

    if [[ "${VLLM_CONTAINER_FORCE_CREATE}" == "1" ]]; then
      echo "Force-recreating enroot sandbox ${VLLM_SANDBOX_NAME} ..."
      enroot remove -f "${VLLM_SANDBOX_NAME}" >/dev/null 2>&1 || true
    fi
    if ! enroot list | awk '{print $1}' | grep -qx "${VLLM_SANDBOX_NAME}"; then
      echo "Creating enroot sandbox ${VLLM_SANDBOX_NAME} from image ${VLLM_CONTAINER} ..."
      enroot create --name "${VLLM_SANDBOX_NAME}" "${VLLM_CONTAINER}"
      VLLM_SANDBOX_CREATED="1"
    fi
    enroot_target="${VLLM_SANDBOX_NAME}"
  fi

  local enroot_cmd=(enroot start)
  if [[ "${VLLM_CONTAINER_AS_ROOT}" == "1" ]]; then
    enroot_cmd+=(--root)
  fi
  if [[ "${VLLM_CONTAINER_RW}" == "1" ]]; then
    enroot_cmd+=(--rw)
  fi
  enroot_cmd+=(--env "PATH=/opt/nemo_rl_venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
  if [[ -n "${pythonpath_prefix}" ]]; then
    enroot_cmd+=(--env "PYTHONPATH=${pythonpath_prefix}")
  fi
  IFS=',' read -r -a _container_mounts <<< "${VLLM_CONTAINER_MOUNTS}"
  for _mount in "${_container_mounts[@]}"; do
    [[ -n "${_mount}" ]] && enroot_cmd+=(--mount "${_mount}")
  done

  local enroot_bootstrap_script
  enroot_bootstrap_script="$(build_vllm_bootstrap_script "${vllm_module_args_escaped}" "${pythonpath_prefix}")"
  enroot_cmd+=("${enroot_target}" bash -lc "${enroot_bootstrap_script}")

  "${enroot_cmd[@]}" > "${VLLM_LOG_FILE}" 2>&1 &
  VLLM_PID=$!
  echo "container(enroot) vLLM pid=${VLLM_PID} log=${VLLM_LOG_FILE}"
}

start_container_vllm_with_apptainer() {
  local vllm_module_args_escaped="$1"
  local pythonpath_prefix="$2"

  if ! command -v apptainer >/dev/null 2>&1; then
    echo "[FATAL] VLLM runtime selected as apptainer but apptainer command not found" >&2
    exit 1
  fi
  if [[ -z "${VLLM_APPTAINER_IMAGE}" ]]; then
    echo "[FATAL] VLLM runtime apptainer requires VLLM_APPTAINER_IMAGE (or VLLM_CONTAINER)" >&2
    exit 1
  fi

  local apptainer_cmd=(apptainer exec --nv --writable-tmpfs --cleanenv)
  IFS=',' read -r -a _container_mounts <<< "${VLLM_CONTAINER_MOUNTS}"
  for _mount in "${_container_mounts[@]}"; do
    [[ -n "${_mount}" ]] && apptainer_cmd+=(--bind "${_mount}")
  done

  local apptainer_bootstrap_script
  apptainer_bootstrap_script="$(build_vllm_bootstrap_script "${vllm_module_args_escaped}" "${pythonpath_prefix}")"
  apptainer_cmd+=("${VLLM_APPTAINER_IMAGE}" bash -lc "${apptainer_bootstrap_script}")

  "${apptainer_cmd[@]}" > "${VLLM_LOG_FILE}" 2>&1 &
  VLLM_PID=$!
  echo "container(apptainer) vLLM pid=${VLLM_PID} log=${VLLM_LOG_FILE}"
}

if [[ "${VLLM_BACKEND}" == "host" ]]; then
  mkdir -p "$(dirname "${VLLM_LOG_FILE}")"
  echo "Starting host vLLM server ..."
  env \
    MODEL_NAME="${VLLM_MODEL_NAME}" \
    SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME}" \
    HOST="${VLLM_HOST}" \
    PORT="${VLLM_PORT}" \
    MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
    TENSOR_PARALLEL_SIZE="${VLLM_TP}" \
    GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
    bash "${PROJECT_ROOT}/scripts/run_qwen_vl_openai_server.sh" \
    > "${VLLM_LOG_FILE}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM pid=${VLLM_PID} log=${VLLM_LOG_FILE}"
elif [[ "${VLLM_BACKEND}" == "container" ]]; then
  if [[ -z "${VLLM_CONTAINER}" ]]; then
    echo "[FATAL] VLLM_BACKEND=container requires VLLM_CONTAINER=/path/to/image.squashfs" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${VLLM_LOG_FILE}")"
  echo "Starting container vLLM server ..."
  VLLM_MODULE_ARGS=(
    --model "${VLLM_MODEL_NAME}"
    --served-model-name "${VLLM_SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --dtype bfloat16
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --tensor-parallel-size "${VLLM_TP}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  )
  if [[ -n "${VLLM_MM_PROCESSOR_CACHE_GB}" ]]; then
    VLLM_MODULE_ARGS+=(--mm-processor-cache-gb "${VLLM_MM_PROCESSOR_CACHE_GB}")
  fi
  if [[ -n "${VLLM_MM_PROCESSOR_CACHE_TYPE}" ]]; then
    VLLM_MODULE_ARGS+=(--mm-processor-cache-type "${VLLM_MM_PROCESSOR_CACHE_TYPE}")
  fi
  if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
    VLLM_MODULE_ARGS+=(--trust-remote-code)
  fi
  VLLM_MODULE_ARGS_ESCAPED="$(printf '%q ' "${VLLM_MODULE_ARGS[@]}")"
  VLLM_PYTHONPATH_PREFIX="$(build_vllm_pythonpath_prefix)"
  VLLM_RUNTIME_EFFECTIVE="$(resolve_container_runtime)"
  echo "VLLM runtime selected: ${VLLM_RUNTIME_EFFECTIVE}"

  case "${VLLM_RUNTIME_EFFECTIVE}" in
    enroot)
      start_container_vllm_with_enroot "${VLLM_MODULE_ARGS_ESCAPED}" "${VLLM_PYTHONPATH_PREFIX}"
      ;;
    apptainer)
      start_container_vllm_with_apptainer "${VLLM_MODULE_ARGS_ESCAPED}" "${VLLM_PYTHONPATH_PREFIX}"
      ;;
    none)
      echo "[FATAL] Neither enroot nor apptainer is available on this node" >&2
      exit 1
      ;;
    invalid)
      echo "[FATAL] Invalid VLLM_CONTAINER_RUNTIME=${VLLM_CONTAINER_RUNTIME} (expected auto|enroot|apptainer)" >&2
      exit 1
      ;;
    *)
      echo "[FATAL] Unknown runtime selection: ${VLLM_RUNTIME_EFFECTIVE}" >&2
      exit 1
      ;;
  esac
fi

if [[ "${RUN_COLLECT}" == "1" ]]; then
  if ! wait_for_openai_endpoint; then
    echo "[FATAL] vLLM endpoint failed to become ready. Log: ${VLLM_LOG_FILE}" >&2
    exit 1
  fi
  env \
    OSWORLD_SRC="${OSWORLD_SRC}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    MODEL_ALIAS="${MODEL_ALIAS}" \
    PROVIDER_NAME="${PROVIDER_NAME}" \
    ACTION_SPACE="${ACTION_SPACE}" \
    OBSERVATION_TYPE="${OBSERVATION_TYPE}" \
    NUM_ENVS="${NUM_ENVS}" \
    MAX_STEPS="${MAX_STEPS}" \
    MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH}" \
    SLEEP_AFTER_EXECUTION="${SLEEP_AFTER_EXECUTION}" \
    DOMAIN="${DOMAIN}" \
    TEST_ALL_META_PATH="${TEST_ALL_META_PATH}" \
    CLIENT_PASSWORD="${CLIENT_PASSWORD}" \
    HEADLESS="${HEADLESS}" \
    VM_SECRET_MOUNTS="${VM_SECRET_MOUNTS}" \
    PATH_TO_VM="${PATH_TO_VM}" \
    OSWORLD_APPTAINER_BIN="${OSWORLD_APPTAINER_BIN}" \
    OSWORLD_APPTAINER_IMAGE="${OSWORLD_APPTAINER_IMAGE}" \
    OSWORLD_APPTAINER_EXTRA_BINDS="${OSWORLD_APPTAINER_EXTRA_BINDS}" \
    OSWORLD_APPTAINER_STORAGE_DIR="${OSWORLD_APPTAINER_STORAGE_DIR}" \
    OSWORLD_APPTAINER_USE_FAKEROOT="${OSWORLD_APPTAINER_USE_FAKEROOT}" \
    OSWORLD_APPTAINER_LOG_DIR="${OSWORLD_APPTAINER_LOG_DIR}" \
    OSWORLD_APPTAINER_PATCH_NGINX_PORT="${OSWORLD_APPTAINER_PATCH_NGINX_PORT}" \
    OSWORLD_APPTAINER_PATCH_NETWORK_PORTS="${OSWORLD_APPTAINER_PATCH_NETWORK_PORTS}" \
    OSWORLD_APPTAINER_PATCH_ROOT_CHECK="${OSWORLD_APPTAINER_PATCH_ROOT_CHECK}" \
    OSWORLD_APPTAINER_SSH_FORWARD_PORT="${OSWORLD_APPTAINER_SSH_FORWARD_PORT}" \
    OSWORLD_APPTAINER_RDP_FORWARD_PORT="${OSWORLD_APPTAINER_RDP_FORWARD_PORT}" \
    OSWORLD_APPTAINER_MONITOR_PORT="${OSWORLD_APPTAINER_MONITOR_PORT}" \
    OSWORLD_APPTAINER_READY_TIMEOUT="${OSWORLD_APPTAINER_READY_TIMEOUT}" \
    OSWORLD_APPTAINER_KVM="${OSWORLD_APPTAINER_KVM}" \
    OSWORLD_APPTAINER_NETWORK="${OSWORLD_APPTAINER_NETWORK}" \
    OSWORLD_APPTAINER_USER_PORTS="${OSWORLD_APPTAINER_USER_PORTS}" \
    OSWORLD_APPTAINER_CPU_CORES="${OSWORLD_APPTAINER_CPU_CORES}" \
    OSWORLD_APPTAINER_RAM_SIZE="${OSWORLD_APPTAINER_RAM_SIZE}" \
    OSWORLD_APPTAINER_DISK_SIZE="${OSWORLD_APPTAINER_DISK_SIZE}" \
    OSWORLD_APPTAINER_BOOT_MODE="${OSWORLD_APPTAINER_BOOT_MODE}" \
    OSWORLD_APPTAINER_SSH_DIAGNOSTICS="${OSWORLD_APPTAINER_SSH_DIAGNOSTICS}" \
    OSWORLD_APPTAINER_AUTO_START_SERVER="${OSWORLD_APPTAINER_AUTO_START_SERVER}" \
    OSWORLD_SERVER_PORT="${OSWORLD_SERVER_PORT}" \
    OSWORLD_CHROMIUM_PORT="${OSWORLD_CHROMIUM_PORT}" \
    OSWORLD_VNC_PORT="${OSWORLD_VNC_PORT}" \
    OSWORLD_VLC_PORT="${OSWORLD_VLC_PORT}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    bash "${PROJECT_ROOT}/scripts/run_osworld_eval_qwen_vl.sh"
fi

if [[ "${RUN_CONVERT}" == "1" ]]; then
  CONVERT_CMD=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/convert_osworld_results_to_nemorl.py"
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
  "${CONVERT_CMD[@]}"
fi

echo
echo "✅ Done."
echo "  RESULT_ROOT: ${RESULT_ROOT}"
echo "  TRAIN_DATA:  ${TRAIN_DATA}"
echo "  VAL_DATA:    ${VAL_DATA}"
