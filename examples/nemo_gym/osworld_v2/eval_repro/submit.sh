#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd -P)"
RAY_SUB="$RL_ROOT/ray.sub"

if (( $# != 1 )); then
  echo "Usage: $0 ENV_FILE" >&2
  exit 2
fi
ENV_FILE="$1"
test -r "$ENV_FILE"
set -a
source "$ENV_FILE"
set +a

if [[ -z "${OPENSANDBOX_DOMAIN:-}" && -n "${OPENSANDBOX_BASE_URL:-}" ]]; then
  opensandbox_host="${OPENSANDBOX_BASE_URL#*://}"
  export OPENSANDBOX_DOMAIN="${opensandbox_host%%/*}"
fi
export OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"

required_vars=(
  CONTAINER
  NANO_OMNI_MODEL_NAME
  NANO_OMNI_CHAT_TEMPLATE
  OSWORLD_EVAL_DATA_ROOT
  OSWORLD_RESULTS_ROOT
  OPENSANDBOX_DOMAIN
  SLURM_ACCOUNT
  SLURM_PARTITION
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required variable is unset: $name" >&2
    exit 2
  fi
done

canonical_file() {
  test -f "$1" || { echo "Missing file: $1" >&2; exit 2; }
  readlink -f "$1"
}
canonical_dir() {
  test -d "$1" || { echo "Missing directory: $1" >&2; exit 2; }
  readlink -f "$1"
}

export CONTAINER="$(canonical_file "$CONTAINER")"
export NANO_OMNI_MODEL_NAME="$(canonical_dir "$NANO_OMNI_MODEL_NAME")"
export NANO_OMNI_CHAT_TEMPLATE="$(canonical_file "$NANO_OMNI_CHAT_TEMPLATE")"
export OSWORLD_EVAL_DATA_ROOT="$(canonical_dir "$OSWORLD_EVAL_DATA_ROOT")"
test -r "$NANO_OMNI_MODEL_NAME/config.json"
test -r "$RAY_SUB"

EXPECTED_CONTAINER_SHA256=1b4edcdeac017210e6abe25885372e025ac111c45aeb038bda334659436eb74e
if [[ "${OSWORLD_VERIFY_CONTAINER_SHA256:-1}" == "1" ]]; then
  actual_sha256="$(sha256sum "$CONTAINER" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$EXPECTED_CONTAINER_SHA256" ]]; then
    echo "Container checksum mismatch: $actual_sha256" >&2
    exit 2
  fi
fi

mkdir -p "$OSWORLD_RESULTS_ROOT"
export OSWORLD_RESULTS_ROOT="$(readlink -f "$OSWORLD_RESULTS_ROOT")"
export OSWORLD_EVAL_CACHE_ROOT="${OSWORLD_EVAL_CACHE_ROOT:-$OSWORLD_RESULTS_ROOT/.eval-cache}"
export OSWORLD_RUNTIME_LIB_DIR="${OSWORLD_RUNTIME_LIB_DIR:-$OSWORLD_EVAL_CACHE_ROOT/runtime-libs}"
mkdir -p "$OSWORLD_EVAL_CACHE_ROOT" "$OSWORLD_RUNTIME_LIB_DIR"
export OSWORLD_EVAL_CACHE_ROOT="$(readlink -f "$OSWORLD_EVAL_CACHE_ROOT")"
export OSWORLD_RUNTIME_LIB_DIR="$(readlink -f "$OSWORLD_RUNTIME_LIB_DIR")"

copy_runtime_lib() {
  local library="$1"
  local destination="$OSWORLD_RUNTIME_LIB_DIR/$library"
  [[ -r "$destination" ]] && return
  local source=""
  if command -v ldconfig >/dev/null 2>&1; then
    source="$(ldconfig -p 2>/dev/null | awk -v lib="$library" '$1 == lib && !found {print $NF; found=1}')"
  fi
  if [[ -z "$source" || ! -r "$source" ]]; then
    for candidate in "/lib/x86_64-linux-gnu/$library" "/usr/lib/x86_64-linux-gnu/$library"; do
      if [[ -r "$candidate" ]]; then
        source="$candidate"
        break
      fi
    done
  fi
  test -r "$source" || { echo "Cannot find $library" >&2; exit 2; }
  cp -fL "$source" "$destination"
}
for library in libopenblas.so.0 libgfortran.so.5 libquadmath.so.0 libgcc_s.so.1; do
  copy_runtime_lib "$library"
done

for script in "$RAY_SUB" "$SCRIPT_DIR/runtime_env.sh" "$SCRIPT_DIR/node_setup.sh" "$SCRIPT_DIR/driver.sh"; do
  bash -n "$script"
done

declare -A identity_mounts=()
mount_specs=("$RL_ROOT:/opt/nemo-rl")
add_identity_mount() {
  local path="$1"
  [[ -d "$path" ]] || path="$(dirname "$path")"
  path="$(readlink -f "$path")"
  for mounted in "${!identity_mounts[@]}"; do
    [[ "$path" == "$mounted" || "$path" == "$mounted/"* ]] && return
  done
  identity_mounts["$path"]=1
  mount_specs+=("$path:$path")
}
add_identity_mount "$NANO_OMNI_MODEL_NAME"
add_identity_mount "$OSWORLD_EVAL_DATA_ROOT"
add_identity_mount "$OSWORLD_RESULTS_ROOT"
add_identity_mount "$OSWORLD_EVAL_CACHE_ROOT"
add_identity_mount "$OSWORLD_RUNTIME_LIB_DIR"

export MOUNTS="$(IFS=,; echo "${mount_specs[*]}")"
export GPUS_PER_NODE=8
export NUM_NODES=1
export CC_DRIVER_VENV="${OSWORLD_DRIVER_VENV:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker}"
export RAY_CLI="$CC_DRIVER_VENV/bin/ray"
export CONTAINER_WORKDIR=/opt/nemo-rl
export SETUP_COMMAND="bash '/opt/nemo-rl/examples/nemo_gym/osworld_v2/eval_repro/node_setup.sh'"
export COMMAND="bash '/opt/nemo-rl/examples/nemo_gym/osworld_v2/eval_repro/driver.sh'"
export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-osworld-kvm}"

timestamp="${OSWORLD_EVAL_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_prefix="${OSWORLD_EVAL_RUN_PREFIX:-osworld-v2-parity-eval}"
series_root="$OSWORLD_RESULTS_ROOT/$run_prefix-$timestamp"
mkdir -p "$series_root"

if [[ "${OSWORLD_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  printf 'dry_run=ok\nseries_root=%s\nmounts=%s\n' "$series_root" "$MOUNTS"
  exit 0
fi

for shard in q01 q02 q03 q04; do
  export OSWORLD_GRPO_VAL_DATA="$OSWORLD_EVAL_DATA_ROOT/validation_nogdrive_361_${shard}.jsonl"
  test -s "$OSWORLD_GRPO_VAL_DATA"
  export OSWORLD_EVAL_RUN_ROOT="$series_root/$shard"
  export OSWORLD_RESULTS_DIR="$OSWORLD_EVAL_RUN_ROOT"
  export BASE_LOG_DIR="$OSWORLD_EVAL_RUN_ROOT"
  mkdir -p "$OSWORLD_EVAL_RUN_ROOT"

  sbatch --parsable \
    --export=ALL \
    --chdir="$RL_ROOT" \
    --nodes=1 \
    --exclusive \
    --gres="${OSWORLD_SBATCH_GRES:-gpu:8}" \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --time="${OSWORLD_JOB_TIME:-04:00:00}" \
    --job-name="osw-v2-eval-${shard}" \
    --output="$OSWORLD_EVAL_RUN_ROOT/slurm-%j.out" \
    "$RAY_SUB" | tee "$OSWORLD_EVAL_RUN_ROOT/submission.txt"
done

echo "series_root=$series_root"
