#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
cd "$LIVE_RL_ROOT"

CONFIG=examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_v2_inference_v1_parity.yaml
test -f "$CONFIG"

exec "$LIVE_DRIVER_PYTHON" \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "$CONFIG" \
  "logger.log_dir=$OSWORLD_EVAL_RUN_ROOT" \
  "policy.generation.vllm_cfg.reasoning_parser_plugin=$LIVE_RL_ROOT/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
