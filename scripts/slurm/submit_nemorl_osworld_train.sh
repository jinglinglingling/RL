#!/usr/bin/env bash
# One-shot NeMo-RL training submit for OSWorld converted data.
# Uses the same ray.sub + COMMAND contract as Nemo-RL-ppo scripts.
#
# Usage:
#   CONTAINER=/path/to/nemo-rl.squashfs bash scripts/slurm/submit_nemorl_osworld_train.sh
#   NUM_NODES=2 GPUS_PER_NODE=8 CONTAINER=... bash scripts/slurm/submit_nemorl_osworld_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORKTREE_REQUESTED="${WORKTREE:-${PROJECT_ROOT}/../Nemo-RL-main-1/RL-merge-2689}"
NEMORL_LOG_ROOT="${NEMORL_LOG_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/sbatch_histrory}"
RAY_SUB_SRC="${RAY_SUB:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/Omni_project/Nemo-RL-Omni-vllm0.2/nemo-rl/ray.sub}"

resolve_nemorl_worktree() {
  local requested="$1"
  local candidates=(
    "$requested"
    "${PROJECT_ROOT}/../Nemo-RL-main-1/RL"
    "${PROJECT_ROOT}/../Nemo-RL-main-1/RL-pr2791-push"
    "${PROJECT_ROOT}/../Nemo-RL-main-1-clean"
    "${PROJECT_ROOT}/../Nemo-RL-main/RL"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}/nemo_rl/distributed/virtual_cluster.py" && -f "${candidate}/examples/run_vlm_grpo.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

prepare_apptainer_runtime_dirs() {
  local apptainer_bin="$1"
  local inferred_usr_root=""
  local runtime_root=""
  local shim_lib_dir=""
  local dep_name=""
  local dep_src=""
  local dep_candidate=""

  if [[ ! -x "${apptainer_bin}" ]]; then
    return 1
  fi

  if inferred_usr_root="$(cd "$(dirname "${apptainer_bin}")/.." && pwd 2>/dev/null)"; then
    mkdir -p "${inferred_usr_root}/var/lib/apptainer/mnt/session" >/dev/null 2>&1 || true
    shim_lib_dir="${inferred_usr_root}/lib/apptainer-shims"
    mkdir -p "${shim_lib_dir}" >/dev/null 2>&1 || true
    # Local extracted apptainer helpers (squashfuse_ll/mksquashfs) may miss
    # shared libraries inside the enroot runtime. Copy a minimal set and
    # prepend this shim directory to LD_LIBRARY_PATH in the worker command.
    for dep_name in \
      liblzo2.so.2 \
      liblz4.so.1 \
      liblzma.so.5 \
      libzstd.so.1 \
      libz.so.1 \
      libfuse3.so.3; do
      dep_src=""
      for dep_candidate in \
        "/lib/x86_64-linux-gnu/${dep_name}" \
        "/usr/lib/x86_64-linux-gnu/${dep_name}"; do
        if [[ -r "${dep_candidate}" ]]; then
          dep_src="${dep_candidate}"
          break
        fi
      done
      if [[ -n "${dep_src}" ]]; then
        cp -fL "${dep_src}" "${shim_lib_dir}/${dep_name}" >/dev/null 2>&1 || true
      fi
    done
  fi

  # Some local extracted apptainer builds look for localstatedir under the
  # project-root mirror path instead of the binary's sibling /usr path.
  for runtime_root in \
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/local/apptainer-test/extract/usr" \
    "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/local/apptainer-test/extract/usr"; do
    mkdir -p "${runtime_root}/var/lib/apptainer/mnt/session" >/dev/null 2>&1 || true
  done

  return 0
}

validate_apptainer_bin() {
  local apptainer_bin="$1"
  if [[ ! -x "${apptainer_bin}" ]]; then
    return 1
  fi
  prepare_apptainer_runtime_dirs "${apptainer_bin}"
  if ! "${apptainer_bin}" --version >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

if ! WORKTREE="$(resolve_nemorl_worktree "${WORKTREE_REQUESTED}")"; then
  echo "[FATAL] No usable NeMo-RL worktree found." >&2
  echo "        Requested: ${WORKTREE_REQUESTED}" >&2
  echo "        Need both files:" >&2
  echo "          nemo_rl/distributed/virtual_cluster.py" >&2
  echo "          examples/run_vlm_grpo.py" >&2
  exit 1
fi
if [[ "${WORKTREE}" != "${WORKTREE_REQUESTED}" ]]; then
  echo "[WARN] Requested WORKTREE is missing required files: ${WORKTREE_REQUESTED}" >&2
  echo "       Falling back to detected worktree: ${WORKTREE}" >&2
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/nemorl_osworld_grpo_qwen_vl.yaml}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/nemorl/osworld_val.jsonl}"

TRAIN_PROFILE="${TRAIN_PROFILE:-stable-1g-3b-local}" # stable-1g-3b-local | custom
USER_NEMORL_OVERRIDES="${NEMORL_OVERRIDES:-}"
NEMORL_OVERRIDES=""
PROFILE_OVERRIDES=""

NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
ALLOW_MISSING_DATA_ON_SUBMIT="${ALLOW_MISSING_DATA_ON_SUBMIT:-0}"

JOB_NAME_BASE="${JOB_NAME_BASE:-osw-train}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S-%3N)}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_BASE}-${NUM_NODES}n-${RUN_ID}}"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/outputs/slurm-train/${JOB_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${CONTAINER:-}" ]]; then
  for _container_candidate in \
    "/lustre/fsw/portfolios/coreai/users/nliang/enroot-images/nemo-rl-nightly-20260626.sqsh" \
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/biguo/enroot-images/nemo-rl/nemo-rl:ba9e677-51258256-051626.squashfs"; do
    if [[ -f "${_container_candidate}" ]]; then
      CONTAINER="${_container_candidate}"
      break
    fi
  done
  unset _container_candidate
fi
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/nliang/enroot-images/nemo-rl-nightly-20260626.sqsh}"
if [[ ! -f "${CONTAINER}" ]]; then
  echo "[FATAL] CONTAINER image not found: ${CONTAINER}" >&2
  exit 1
fi
export CONTAINER_INIT_SCRIPT="${CONTAINER_INIT_SCRIPT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/Nemo-RL-Super/DPO/container_init.sh}"
export NRL_REPO_DIR="${NRL_REPO_DIR:-${WORKTREE}}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"
export NRL_WORKER_RAY_VERSION="${NRL_WORKER_RAY_VERSION:-2.55.1}"
# Some local NeMo-RL worktrees intentionally diverge from uv.lock.
# Set to 1 to force strict lockfile enforcement.
export NRL_UV_RUN_LOCKED="${NRL_UV_RUN_LOCKED:-0}"
# Keep parity with PPO smoke path: this image often misses tensordict for driver import.
# Install only the minimal direct packages to avoid expensive resolver/network churn.
BASE_SETUP_COMMAND="${SETUP_COMMAND:-/opt/nemo_rl_venv/bin/pip install --quiet --no-input --no-deps tensordict pyvers}"
RAY_ALIGN_COMMAND=""
if [[ -n "${NRL_WORKER_RAY_VERSION}" ]]; then
  # Keep the Ray daemon (head/worker) and runtime-env venvs on the same Ray version.
  # ray.sub runs SETUP_COMMAND before ray start on every node.
  # The container can have a partially broken Ray install (missing files), where
  # pip uninstall/force-reinstall fails before ray start. Proactively remove any
  # stale ray package artifacts, then install a clean target version.
  RAY_ALIGN_COMMAND="rm -rf /opt/nemo_rl_venv/lib/python*/site-packages/ray /opt/nemo_rl_venv/lib/python*/site-packages/ray-*.dist-info /opt/nemo_rl_venv/lib/python*/site-packages/ray-*.egg-info && /opt/nemo_rl_venv/bin/pip install --quiet --no-input --upgrade --no-cache-dir ray==${NRL_WORKER_RAY_VERSION}"
fi
if [[ -n "${RAY_ALIGN_COMMAND}" && "${BASE_SETUP_COMMAND}" != *"ray=="* ]]; then
  export SETUP_COMMAND="${BASE_SETUP_COMMAND} && ${RAY_ALIGN_COMMAND}"
else
  export SETUP_COMMAND="${BASE_SETUP_COMMAND}"
fi

export MOUNTS="${MOUNTS:-/lustre:/lustre,/home:/home}"
if [[ -f "${HOME}/.netrc" ]] && [[ "${MOUNTS}" != *"/root/.netrc"* ]]; then
  export MOUNTS="${MOUNTS},${HOME}/.netrc:/root/.netrc:ro"
fi

USER_ROOT="${USER_ROOT:-/lustre/fs1/portfolios/coreai/users/${USER}}"
USER_ROOT_FSW="${USER_ROOT_FSW:-/lustre/fsw/portfolios/coreai/users/${USER}}"
export CACHE_ROOT="${CACHE_ROOT:-${USER_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME}/token}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${RUN_ID}}"
export OSWORLD_VLM_MAX_IMAGE_SIDE="${OSWORLD_VLM_MAX_IMAGE_SIDE:-0}"

if [[ -n "${OSWORLD_APPTAINER_BIN:-}" ]]; then
  if ! validate_apptainer_bin "${OSWORLD_APPTAINER_BIN}"; then
    echo "[FATAL] OSWORLD_APPTAINER_BIN is not executable/usable: ${OSWORLD_APPTAINER_BIN}" >&2
    exit 1
  fi
elif [[ -n "${APPTAINER_BIN:-}" ]]; then
  if validate_apptainer_bin "${APPTAINER_BIN}"; then
    export OSWORLD_APPTAINER_BIN="${APPTAINER_BIN}"
  else
    echo "[WARN] Ignoring unusable APPTAINER_BIN: ${APPTAINER_BIN}" >&2
  fi
fi

if [[ -z "${OSWORLD_APPTAINER_BIN:-}" ]]; then
  for _apptainer_candidate in \
    "${USER_ROOT}/local/apptainer-test/extract/usr/bin/apptainer" \
    "${USER_ROOT_FSW}/local/apptainer-test/extract/usr/bin/apptainer" \
    "${HOME}/local/apptainer-test/extract/usr/bin/apptainer"; do
    if validate_apptainer_bin "${_apptainer_candidate}"; then
      export OSWORLD_APPTAINER_BIN="${_apptainer_candidate}"
      break
    fi
  done
  unset _apptainer_candidate
fi
if [[ -n "${OSWORLD_APPTAINER_BIN:-}" ]]; then
  export APPTAINER_BIN="${OSWORLD_APPTAINER_BIN}"
fi

if [[ -n "${OSWORLD_APPTAINER_IMAGE:-}" ]]; then
  if [[ "${OSWORLD_APPTAINER_IMAGE}" != docker://* && ! -f "${OSWORLD_APPTAINER_IMAGE}" ]]; then
    echo "[FATAL] OSWORLD_APPTAINER_IMAGE not found: ${OSWORLD_APPTAINER_IMAGE}" >&2
    exit 1
  fi
else
  for _apptainer_image_candidate in \
    "${USER_ROOT}/local/apptainer-test/osworld-docker-latest.sif" \
    "${USER_ROOT_FSW}/local/apptainer-test/osworld-docker-latest.sif" \
    "${HOME}/local/apptainer-test/osworld-docker-latest.sif"; do
    if [[ -f "${_apptainer_image_candidate}" ]]; then
      export OSWORLD_APPTAINER_IMAGE="${_apptainer_image_candidate}"
      break
    fi
  done
  unset _apptainer_image_candidate
fi
if [[ -n "${OSWORLD_APPTAINER_IMAGE:-}" && "${OSWORLD_APPTAINER_IMAGE}" != docker://* ]]; then
  echo "[INFO] Using prebuilt OSWorld SIF image: ${OSWORLD_APPTAINER_IMAGE}" >&2
fi

if [[ -n "${OSWORLD_VM_PATH:-}" ]]; then
  if [[ ! -f "${OSWORLD_VM_PATH}" ]]; then
    echo "[FATAL] OSWORLD_VM_PATH not found: ${OSWORLD_VM_PATH}" >&2
    exit 1
  fi
else
  for _osworld_vm_candidate in \
    "${WORKTREE}/3rdparty/Gym-workspace/Gym/resources_servers/osworld_vlm/docker_vm_data/Ubuntu.qcow2" \
    "${PROJECT_ROOT}/third_party/OSWorld/docker_vm_data/Ubuntu.qcow2" \
    "${PROJECT_ROOT}/../Nemo-RL-main-1/RL/3rdparty/Gym-workspace/Gym/resources_servers/osworld_vlm/docker_vm_data/Ubuntu.qcow2"; do
    if [[ -f "${_osworld_vm_candidate}" ]]; then
      export OSWORLD_VM_PATH="${_osworld_vm_candidate}"
      break
    fi
  done
  unset _osworld_vm_candidate
fi
if [[ -n "${OSWORLD_VM_PATH:-}" ]]; then
  if [[ "${OSWORLD_VM_PATH}" != *.qcow2 ]]; then
    echo "[WARN] OSWORLD_VM_PATH does not end with .qcow2: ${OSWORLD_VM_PATH}" >&2
  fi
fi
if [[ -n "${OSWORLD_VM_PATH:-}" && -n "${OSWORLD_APPTAINER_IMAGE:-}" && "${OSWORLD_APPTAINER_IMAGE}" != docker://* && -f "${OSWORLD_APPTAINER_IMAGE}" ]]; then
  _osworld_vm_resolved="$(readlink -f "${OSWORLD_VM_PATH}" 2>/dev/null || true)"
  _osworld_apptainer_image_resolved="$(readlink -f "${OSWORLD_APPTAINER_IMAGE}" 2>/dev/null || true)"
  if [[ -n "${_osworld_vm_resolved}" && -n "${_osworld_apptainer_image_resolved}" && "${_osworld_vm_resolved}" == "${_osworld_apptainer_image_resolved}" ]]; then
    echo "[FATAL] OSWORLD_VM_PATH points to the apptainer image. Set OSWORLD_VM_PATH to a qcow2 VM disk." >&2
    echo "        OSWORLD_VM_PATH=${OSWORLD_VM_PATH}" >&2
    echo "        OSWORLD_APPTAINER_IMAGE=${OSWORLD_APPTAINER_IMAGE}" >&2
    exit 1
  fi
  unset _osworld_vm_resolved _osworld_apptainer_image_resolved
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${USER_ROOT}/local/apptainer-test/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${USER_ROOT}/local/apptainer-test/tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" >/dev/null 2>&1 || true
APPTAINER_SHIM_LIB_DIR=""
if [[ -n "${OSWORLD_APPTAINER_BIN:-}" ]]; then
  if _apptainer_usr_root="$(cd "$(dirname "${OSWORLD_APPTAINER_BIN}")/.." && pwd 2>/dev/null)"; then
    APPTAINER_SHIM_LIB_DIR="${_apptainer_usr_root}/lib/apptainer-shims"
  fi
  unset _apptainer_usr_root
fi
if [[ -n "${APPTAINER_SHIM_LIB_DIR}" && ! -d "${APPTAINER_SHIM_LIB_DIR}" ]]; then
  APPTAINER_SHIM_LIB_DIR=""
fi
export APPTAINER_SHIM_LIB_DIR
APPTAINER_SHIM_WRAPPER=""
if [[ -n "${OSWORLD_APPTAINER_BIN:-}" && -n "${APPTAINER_SHIM_LIB_DIR}" ]]; then
  APPTAINER_SHIM_WRAPPER="${APPTAINER_TMPDIR}/osworld-apptainer-shim-${RUN_ID}.sh"
  cat > "${APPTAINER_SHIM_WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
APPTAINER_REAL_BIN='${OSWORLD_APPTAINER_BIN}'
APPTAINER_REAL_SHIM='${APPTAINER_SHIM_LIB_DIR}'
if [[ -n "\${APPTAINER_REAL_SHIM}" && -d "\${APPTAINER_REAL_SHIM}" ]]; then
  export LD_LIBRARY_PATH="\${APPTAINER_REAL_SHIM}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
exec "\${APPTAINER_REAL_BIN}" "\$@"
EOF
  chmod +x "${APPTAINER_SHIM_WRAPPER}"
  export OSWORLD_APPTAINER_BIN="${APPTAINER_SHIM_WRAPPER}"
  export APPTAINER_BIN="${APPTAINER_SHIM_WRAPPER}"
fi
export APPTAINER_SHIM_WRAPPER

if [[ -z "${GPUS_PER_NODE}" ]]; then
  if [[ "${TRAIN_PROFILE}" == "stable-1g-3b-local" ]]; then
    GPUS_PER_NODE="1"
  else
    GPUS_PER_NODE="8"
  fi
fi
# ray.sub reads these from environment when composing per-node srun flags.
export NUM_NODES GPUS_PER_NODE

if [[ "${TRAIN_PROFILE}" == "stable-1g-3b-local" ]]; then
  STABLE_MAX_STEPS="${STABLE_MAX_STEPS:-5}"
  STABLE_MAX_EPOCHS="${STABLE_MAX_EPOCHS:-${STABLE_MAX_STEPS}}"
  STABLE_MODEL_REPO="${STABLE_MODEL_REPO:-Qwen/Qwen2.5-Omni-3B}"
  STABLE_NUM_PROMPTS_PER_STEP="${STABLE_NUM_PROMPTS_PER_STEP:-1}"
  STABLE_NUM_GENERATIONS_PER_PROMPT="${STABLE_NUM_GENERATIONS_PER_PROMPT:-4}"
  # Multimodal prompts are longer than text-only prompts.
  # Keep enough context to avoid prompt-length failures while capping memory.
  STABLE_MAX_TOTAL_SEQUENCE_LENGTH="${STABLE_MAX_TOTAL_SEQUENCE_LENGTH:-2944}"
  STABLE_MAX_MODEL_LEN="${STABLE_MAX_MODEL_LEN:-3072}"
  STABLE_MAX_NEW_TOKENS="${STABLE_MAX_NEW_TOKENS:-24}"
  STABLE_VLLM_GPU_MEMORY_UTILIZATION="${STABLE_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
  STABLE_LOGPROB_CHUNK_SIZE="${STABLE_LOGPROB_CHUNK_SIZE:-256}"
  if [[ -z "${STABLE_TRAIN_GLOBAL_BATCH_SIZE:-}" ]]; then
    stable_train_dp="${NUM_NODES:-1}"
    if ! [[ "${stable_train_dp}" =~ ^[0-9]+$ ]] || (( stable_train_dp < 1 )); then
      stable_train_dp=1
    fi
    # With TP=1 and gpus_per_node=1 in this profile, DP ~= NUM_NODES.
    # Keep train_global_batch_size divisible by DP so Megatron eval defaults are valid.
    STABLE_TRAIN_GLOBAL_BATCH_SIZE="${stable_train_dp}"
    unset stable_train_dp
  fi
  STABLE_DEFER_FP32_LOGITS="${STABLE_DEFER_FP32_LOGITS:-true}"
  STABLE_MEGATRON_ACTIVATION_CHECKPOINTING="${STABLE_MEGATRON_ACTIVATION_CHECKPOINTING:-1}"
  STABLE_EMPTY_UNUSED_MEMORY_LEVEL="${STABLE_EMPTY_UNUSED_MEMORY_LEVEL:-2}"
  STABLE_OPTIMIZER_CPU_OFFLOAD="${STABLE_OPTIMIZER_CPU_OFFLOAD:-true}"
  if [[ -z "${STABLE_OPTIMIZER_OFFLOAD_FRACTION:-}" ]]; then
    # Qwen2.5-Omni path currently validates optimizer offload only at 1.0.
    # For larger VLMs (e.g. Qwen3-VL-8B), prefer partial offload to reduce
    # host-memory pressure on shared nodes.
    if [[ "${STABLE_MODEL_REPO}" == "Qwen/Qwen2.5-Omni-3B" ]]; then
      STABLE_OPTIMIZER_OFFLOAD_FRACTION="1.0"
    else
      STABLE_OPTIMIZER_OFFLOAD_FRACTION="0.5"
    fi
  fi
  STABLE_MAX_IMAGE_SIDE="${STABLE_MAX_IMAGE_SIDE:-896}"
  if [[ "${OSWORLD_VLM_MAX_IMAGE_SIDE}" == "0" ]]; then
    export OSWORLD_VLM_MAX_IMAGE_SIDE="${STABLE_MAX_IMAGE_SIDE}"
  fi
  MODEL_SNAPSHOT_DIR="${MODEL_SNAPSHOT_DIR:-}"

  if [[ -z "${MODEL_SNAPSHOT_DIR}" ]]; then
    snapshot_repo="${STABLE_MODEL_REPO//\//--}"
    snapshot_root="${HF_HOME}/hub/models--${snapshot_repo}/snapshots"
    if [[ -d "${snapshot_root}" ]]; then
      shopt -s nullglob
      snapshot_candidates=("${snapshot_root}"/*)
      shopt -u nullglob
      for candidate in "${snapshot_candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
          MODEL_SNAPSHOT_DIR="${candidate}"
          break
        fi
      done
    fi
  fi

  if [[ -z "${MODEL_SNAPSHOT_DIR}" || ! -d "${MODEL_SNAPSHOT_DIR}" ]]; then
    expected_snapshot_repo="${STABLE_MODEL_REPO//\//--}"
    echo "[FATAL] TRAIN_PROFILE=${TRAIN_PROFILE} requires a local model snapshot." >&2
    echo "        Expected under: ${HF_HOME}/hub/models--${expected_snapshot_repo}/snapshots/" >&2
    echo "        Set MODEL_SNAPSHOT_DIR explicitly, or switch to TRAIN_PROFILE=custom." >&2
    exit 1
  fi

  PROFILE_OVERRIDES="\
grpo.max_num_epochs=${STABLE_MAX_EPOCHS} \
grpo.max_num_steps=${STABLE_MAX_STEPS} \
grpo.num_prompts_per_step=${STABLE_NUM_PROMPTS_PER_STEP} \
grpo.num_generations_per_prompt=${STABLE_NUM_GENERATIONS_PER_PROMPT} \
grpo.val_at_start=false \
grpo.val_period=1000 \
env.vlm.num_workers=1 \
cluster.gpus_per_node=1 \
policy.model_name=${MODEL_SNAPSHOT_DIR} \
policy.tokenizer.name=${MODEL_SNAPSHOT_DIR} \
policy.train_global_batch_size=${STABLE_TRAIN_GLOBAL_BATCH_SIZE} \
policy.generation_batch_size=1 \
policy.logprob_batch_size=1 \
policy.logprob_chunk_size=${STABLE_LOGPROB_CHUNK_SIZE} \
policy.max_total_sequence_length=${STABLE_MAX_TOTAL_SEQUENCE_LENGTH} \
policy.generation.max_new_tokens=${STABLE_MAX_NEW_TOKENS} \
policy.generation.vllm_cfg.max_model_len=${STABLE_MAX_MODEL_LEN} \
policy.generation.vllm_cfg.gpu_memory_utilization=${STABLE_VLLM_GPU_MEMORY_UTILIZATION} \
policy.generation.vllm_cfg.tensor_parallel_size=1 \
policy.megatron_cfg.tensor_model_parallel_size=1 \
policy.megatron_cfg.defer_fp32_logits=${STABLE_DEFER_FP32_LOGITS} \
policy.megatron_cfg.activation_checkpointing=${STABLE_MEGATRON_ACTIVATION_CHECKPOINTING} \
policy.megatron_cfg.empty_unused_memory_level=${STABLE_EMPTY_UNUSED_MEMORY_LEVEL} \
policy.megatron_cfg.optimizer.optimizer_cpu_offload=${STABLE_OPTIMIZER_CPU_OFFLOAD} \
policy.megatron_cfg.optimizer.optimizer_offload_fraction=${STABLE_OPTIMIZER_OFFLOAD_FRACTION} \
checkpointing.save_period=${STABLE_MAX_STEPS} \
logger.wandb_enabled=false"

  export NEMORL_ALLOW_GPU_UNDERSUBSCRIBE="${NEMORL_ALLOW_GPU_UNDERSUBSCRIBE:-1}"
elif [[ "${TRAIN_PROFILE}" != "custom" ]]; then
  echo "[FATAL] Unsupported TRAIN_PROFILE: ${TRAIN_PROFILE}" >&2
  echo "        Supported values: stable-1g-3b-local, custom" >&2
  exit 1
fi

if [[ -n "${PROFILE_OVERRIDES}" && -n "${USER_NEMORL_OVERRIDES}" ]]; then
  NEMORL_OVERRIDES="${PROFILE_OVERRIDES} ${USER_NEMORL_OVERRIDES}"
elif [[ -n "${PROFILE_OVERRIDES}" ]]; then
  NEMORL_OVERRIDES="${PROFILE_OVERRIDES}"
else
  NEMORL_OVERRIDES="${USER_NEMORL_OVERRIDES}"
fi

PREFER_LOCAL_MODEL_SNAPSHOT="${PREFER_LOCAL_MODEL_SNAPSHOT:-1}"
if [[ "${PREFER_LOCAL_MODEL_SNAPSHOT}" == "1" && -n "${NEMORL_OVERRIDES}" ]]; then
  resolved_model_override=""
  for _override_token in ${NEMORL_OVERRIDES}; do
    case "${_override_token}" in
      policy.model_name=*)
        resolved_model_override="${_override_token#policy.model_name=}"
        ;;
    esac
  done

  if [[ -n "${resolved_model_override}" && "${resolved_model_override}" != /* && "${resolved_model_override}" != ./* && "${resolved_model_override}" == */* ]]; then
    _snapshot_repo="${resolved_model_override//\//--}"
    _snapshot_root="${HF_HOME}/hub/models--${_snapshot_repo}/snapshots"
    if [[ -d "${_snapshot_root}" ]]; then
      shopt -s nullglob
      _snapshot_candidates=("${_snapshot_root}"/*)
      shopt -u nullglob
      for _candidate in "${_snapshot_candidates[@]}"; do
        if [[ -d "${_candidate}" ]]; then
          _local_model_snapshot="${_candidate}"
          break
        fi
      done
      if [[ -n "${_local_model_snapshot:-}" ]]; then
        NEMORL_OVERRIDES="${NEMORL_OVERRIDES} policy.model_name=${_local_model_snapshot} policy.tokenizer.name=${_local_model_snapshot}"
        echo "[INFO] Using local snapshot for policy.model_name: ${_local_model_snapshot}" >&2
      fi
      unset _snapshot_candidates
    fi
    unset _snapshot_repo _snapshot_root _candidate _local_model_snapshot
  fi
  unset resolved_model_override _override_token
fi

RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_ROOT}/scripts/run_nemorl_osworld_grpo.sh}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_DATA}" ]]; then
  if [[ "${ALLOW_MISSING_DATA_ON_SUBMIT}" == "1" && -n "${SBATCH_DEPENDENCY}" ]]; then
    echo "[WARN] Train data not found at submit time: ${TRAIN_DATA}" >&2
    echo "       Continuing because dependency is set (${SBATCH_DEPENDENCY}) and ALLOW_MISSING_DATA_ON_SUBMIT=1." >&2
  else
    echo "[FATAL] Train data not found: ${TRAIN_DATA}" >&2
    exit 1
  fi
fi
if [[ ! -f "${VAL_DATA}" ]]; then
  if [[ "${ALLOW_MISSING_DATA_ON_SUBMIT}" == "1" && -n "${SBATCH_DEPENDENCY}" ]]; then
    echo "[WARN] Validation data not found at submit time: ${VAL_DATA}" >&2
    echo "       Continuing because dependency is set (${SBATCH_DEPENDENCY}) and ALLOW_MISSING_DATA_ON_SUBMIT=1." >&2
  else
    echo "[FATAL] Validation data not found: ${VAL_DATA}" >&2
    exit 1
  fi
fi
if [[ ! -x "${RUN_SCRIPT}" ]]; then
  chmod +x "${RUN_SCRIPT}"
fi
if [[ ! -f "${RAY_SUB_SRC}" ]]; then
  echo "[FATAL] ray.sub not found: ${RAY_SUB_SRC}" >&2
  exit 1
fi

mkdir -p "${NEMORL_LOG_ROOT}" "${RESULTS_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}"
cp -f "${RAY_SUB_SRC}" "${NEMORL_LOG_ROOT}/ray.sub"
sed -i \
  -e "s|^#SBATCH --account=.*|#SBATCH --account=${SBATCH_ACCOUNT}|" \
  -e "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${JOB_NAME}|" \
  -e "s|^#SBATCH --partition=.*|#SBATCH --partition=${SBATCH_PARTITION}|" \
  "${NEMORL_LOG_ROOT}/ray.sub"

export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${CHECKPOINT_DIR}' '${LOG_DIR}' && \
if [[ -f '${HF_TOKEN_FILE}' ]]; then _hf_token=\"\$(tr -d '\r\n' < '${HF_TOKEN_FILE}')\"; export HF_TOKEN=\"\${_hf_token}\" HUGGING_FACE_HUB_TOKEN=\"\${_hf_token}\" HUGGINGFACE_HUB_TOKEN=\"\${_hf_token}\"; unset _hf_token; fi && \
export HF_HUB_DISABLE_IMPLICIT_TOKEN=0 && \
export NEMORL_ROOT='${WORKTREE}' PYTHON_BIN='${PYTHON_BIN}' \
OSWORLD_APPTAINER_BIN='${OSWORLD_APPTAINER_BIN:-}' APPTAINER_BIN='${APPTAINER_BIN:-}' \
OSWORLD_APPTAINER_IMAGE='${OSWORLD_APPTAINER_IMAGE:-}' OSWORLD_VM_PATH='${OSWORLD_VM_PATH:-}' APPTAINER_CACHEDIR='${APPTAINER_CACHEDIR:-}' APPTAINER_TMPDIR='${APPTAINER_TMPDIR:-}' \
APPTAINER_SHIM_LIB_DIR='${APPTAINER_SHIM_LIB_DIR:-}' APPTAINER_SHIM_WRAPPER='${APPTAINER_SHIM_WRAPPER:-}' \
NRL_WORKER_RAY_VERSION='${NRL_WORKER_RAY_VERSION}' \
TRAIN_DATA='${TRAIN_DATA}' VAL_DATA='${VAL_DATA}' \
CHECKPOINT_DIR='${CHECKPOINT_DIR}' LOG_DIR='${LOG_DIR}' CONFIG_PATH='${CONFIG_PATH}' && \
bash '${RUN_SCRIPT}' ${NEMORL_OVERRIDES}"

cd "${NEMORL_LOG_ROOT}"

echo "=== Submit NeMo-RL OSWorld train ==="
echo "  Job name:     ${JOB_NAME}"
echo "  Nodes/GPU:    ${NUM_NODES} x ${GPUS_PER_NODE}"
echo "  Partition:    ${SBATCH_PARTITION}  Time: ${SBATCH_TIME}"
if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
  echo "  Dependency:   ${SBATCH_DEPENDENCY}"
fi
echo "  Allow miss:   ${ALLOW_MISSING_DATA_ON_SUBMIT}"
echo "  Worktree:     ${WORKTREE}"
echo "  Config:       ${CONFIG_PATH}"
echo "  Train data:   ${TRAIN_DATA}"
echo "  Val data:     ${VAL_DATA}"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Container:    ${CONTAINER}"
echo "  Setup cmd:    ${SETUP_COMMAND}"
echo "  Worker Ray:   ${NRL_WORKER_RAY_VERSION}"
echo "  OSW apptainer:${OSWORLD_APPTAINER_BIN:-<auto>}"
echo "  OSW image:    ${OSWORLD_APPTAINER_IMAGE:-<docker://happysixd/osworld-docker:latest>}"
echo "  OSW vm path:  ${OSWORLD_VM_PATH:-<auto>}"
echo "  APPT cache:   ${APPTAINER_CACHEDIR}"
echo "  APPT tmp:     ${APPTAINER_TMPDIR}"
echo "  APPT shim lib:${APPTAINER_SHIM_LIB_DIR:-<none>}"
echo "  APPT shim bin:${APPTAINER_SHIM_WRAPPER:-<none>}"
echo "  UV --locked:  ${NRL_UV_RUN_LOCKED}"
echo "  Profile:      ${TRAIN_PROFILE}"
echo "  HF token:     ${HF_TOKEN_FILE}"
if [[ "${TRAIN_PROFILE}" == "stable-1g-3b-local" ]]; then
  echo "  Model path:   ${MODEL_SNAPSHOT_DIR}"
fi
if [[ -n "${NEMORL_OVERRIDES}" ]]; then
  echo "  Overrides:    ${NEMORL_OVERRIDES}"
fi
echo

SBATCH_CMD=(
  sbatch
  --parsable
  --nodes="${NUM_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --job-name="${JOB_NAME}"
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --gres=gpu:"${GPUS_PER_NODE}"
)
if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
  SBATCH_CMD+=(--dependency="${SBATCH_DEPENDENCY}")
fi
SBATCH_CMD+=(ray.sub)

JOB_ID="$("${SBATCH_CMD[@]}")"

echo "Submitted job ${JOB_ID}"
echo "  Slurm log:   ${NEMORL_LOG_ROOT}/slurm-${JOB_ID}.out"
echo "  Driver log:  ${NEMORL_LOG_ROOT}/${JOB_ID}-logs/ray-driver.log"
echo "  Monitor:     tail -f ${NEMORL_LOG_ROOT}/${JOB_ID}-logs/ray-driver.log"
echo "  Stop:        scancel ${JOB_ID}"
