#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
cd "$OSWORLD_RL_ROOT"

OVERRIDES=(
  "grpo.max_num_steps=${OSWORLD_GRPO_MAX_STEPS:-300}"
  "logger.log_dir=$OSWORLD_RUN_ROOT/exp_001"
  "logger.wandb_enabled=${OSWORLD_WANDB_ENABLED:-true}"
  "logger.wandb.project=${OSWORLD_WANDB_PROJECT:-osworld-rl}"
  "logger.wandb.name=${OSWORLD_WANDB_NAME:-${OSWORLD_RUN_NAME:-osworld-v2}}"
  "checkpointing.checkpoint_must_save_by=${OSWORLD_CHECKPOINT_MUST_SAVE_BY:-00:03:40:00}"
)
if [[ -n "${OSWORLD_SEED:-}" ]]; then
  OVERRIDES+=("grpo.seed=$OSWORLD_SEED")
fi
if [[ -n "${OSWORLD_LR_WARMUP_ITERS:-}" ]]; then
  OVERRIDES+=(
    "policy.megatron_cfg.scheduler.lr_warmup_iters=$OSWORLD_LR_WARMUP_ITERS"
  )
fi
if [[ -n "${OSWORLD_MAX_TOTAL_SEQUENCE_LENGTH:-}" ]]; then
  OVERRIDES+=(
    "policy.max_total_sequence_length=$OSWORLD_MAX_TOTAL_SEQUENCE_LENGTH"
  )
fi
if [[ -n "${OSWORLD_TRAIN_MB_TOKENS:-}" ]]; then
  OVERRIDES+=(
    "policy.sequence_packing.train_mb_tokens=$OSWORLD_TRAIN_MB_TOKENS"
  )
fi
if [[ -n "${OSWORLD_LOGPROB_MB_TOKENS:-}" ]]; then
  OVERRIDES+=(
    "policy.sequence_packing.logprob_mb_tokens=$OSWORLD_LOGPROB_MB_TOKENS"
  )
fi
if [[ -n "${OSWORLD_TEMPERATURE:-}" ]]; then
  OVERRIDES+=(
    "policy.generation.temperature=$OSWORLD_TEMPERATURE"
    "policy.generation.val_temperature=$OSWORLD_TEMPERATURE"
  )
fi
if [[ -n "${OSWORLD_MAX_INFLIGHT_PROMPTS:-}" ]]; then
  OVERRIDES+=(
    "async_rl.max_inflight_prompts=$OSWORLD_MAX_INFLIGHT_PROMPTS"
  )
fi
if [[ -n "${OSWORLD_MAX_BUFFERED_ROLLOUTS:-}" ]]; then
  OVERRIDES+=(
    "async_rl.max_buffered_rollouts=$OSWORLD_MAX_BUFFERED_ROLLOUTS"
  )
fi
if [[ -n "${OSWORLD_LOSS_AGG_MODE:-}" ]]; then
  OVERRIDES+=("loss_fn.loss_agg_mode=$OSWORLD_LOSS_AGG_MODE")
  if [[ "$OSWORLD_LOSS_AGG_MODE" == "seq-mean-token-mean" ]]; then
    OVERRIDES+=("loss_fn.token_level_loss=false")
  fi
fi
if [[ -n "${OSWORLD_CLUSTER_NUM_NODES:-}" ]]; then
  OVERRIDES+=("cluster.num_nodes=$OSWORLD_CLUSTER_NUM_NODES")
fi
if [[ -n "${OSWORLD_GENERATION_NUM_NODES:-}" ]]; then
  OVERRIDES+=(
    "policy.generation.colocated.resources.num_nodes=$OSWORLD_GENERATION_NUM_NODES"
  )
fi

# keep_top_k deletes the resume checkpoint after the next save; keep one
# hardlinked copy of the first resume point (no-op once the target is populated).
# The target must sit inside OSWORLD_CHECKPOINT_DIR: hardlinks cannot cross
# container bind mounts, and only top-level step_* entries are pruned.
if [[ -n "${OSWORLD_PRESERVE_RESUME_CHECKPOINT_TO:-}" ]]; then
  resume_step="$(
    grep -oE '"last_checkpoint_step": *[0-9]+' \
      "$OSWORLD_CHECKPOINT_DIR/latest_checkpoint_status.json" | grep -oE '[0-9]+$'
  )"
  mkdir -p "$OSWORLD_PRESERVE_RESUME_CHECKPOINT_TO"
  if [[ -z "$(ls -A "$OSWORLD_PRESERVE_RESUME_CHECKPOINT_TO")" ]]; then
    cp -al "$OSWORLD_CHECKPOINT_DIR/step_$resume_step" \
      "$OSWORLD_PRESERVE_RESUME_CHECKPOINT_TO/step_$resume_step"
    cp "$OSWORLD_CHECKPOINT_DIR/latest_checkpoint_status.json" \
      "$OSWORLD_PRESERVE_RESUME_CHECKPOINT_TO/"
    echo "preserved resume checkpoint step_$resume_step -> $OSWORLD_PRESERVE_RESUME_CHECKPOINT_TO"
  fi
fi

exec "$OSWORLD_DRIVER_PYTHON" \
  examples/nemo_gym/launch_osworld_v2_cc.py \
  --recipe "${OSWORLD_RECIPE:-flash-b8n8-dr-grpo}" \
  "${OVERRIDES[@]}"
