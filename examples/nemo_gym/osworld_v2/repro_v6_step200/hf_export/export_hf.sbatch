#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J osw-export-hf
# Megatron (NeMo RL DCP) checkpoint -> Hugging Face export for the RPGA eval.
#   REPO=<RL checkout> CONTAINER=<sqsh> CKPT=<.../checkpoints/step_200> OUT=<new hf dir> \
#     sbatch -A <account> -p <partition> hf_export/export_hf.sbatch
# OUT must not exist. The result has 17 safetensors shards (~62G) with tensor keys/shapes/dtypes identical to the
# iter881 base export. After it finishes, replace every non-weight file in OUT (config.json, tokenizer*, processor
# and chat template files, modeling/configuration *.py) with the copies from the iter881 HF export; the Bridge-written
# versions differ in formatting/metadata and the reported eval used the iter881 copies.
set -euo pipefail
: "${REPO:?}" "${CONTAINER:?}" "${CKPT:?}" "${OUT:?}"
[[ -e "$OUT" ]] && { echo "OUT already exists: $OUT" >&2; exit 2; }
B=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
V=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
W=/opt/nemo-rl/examples/nemo_gym/osworld_v2/repro_v6_step200/hf_export
LUSTRE_MOUNT="${LUSTRE_MOUNT:-/lustre:/lustre}"

srun --no-container-mount-home --container-image="$CONTAINER" \
  --container-mounts="$REPO:/opt/nemo-rl,$LUSTRE_MOUNT" \
  bash -c "export PYTHONPATH=/opt/nemo-rl:$B/src:$B/3rdparty/Megatron-LM; \
    $V/bin/torchrun --nproc_per_node=8 $W/export_megatron_to_hf_gpu_wrapper.py \
      --config $CKPT/config.yaml --megatron-ckpt-path $CKPT/policy/weights/iter_0000000 \
      --hf-ckpt-path $OUT --tp 1 --pp 1 --ep 8 --etp 1"
