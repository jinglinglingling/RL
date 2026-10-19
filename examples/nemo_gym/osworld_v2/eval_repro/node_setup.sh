#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"

test -f "$LIVE_RL_ROOT/examples/nemo_gym/run_grpo_nemo_gym.py"
test -f "$LIVE_GYM_ROOT/resources_servers/osworld/app.py"
test -f "$LIVE_GYM_ROOT/responses_api_agents/nemotron_osworld/cc_app.py"
test -x "$LIVE_GYM_ROOT/responses_api_agents/osworld_agent/install_optional_runtime_deps.sh"
test -f "$LIVE_RL_ROOT/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"

case "$LIVE_COMPONENT_PYTHON" in
  /root/*|/opt/*) ;;
  *)
    echo "Refusing non-container component Python: $LIVE_COMPONENT_PYTHON" >&2
    exit 2
    ;;
esac

"$LIVE_COMPONENT_PYTHON" -c \
  'import sys; assert sys.version_info[:3] == (3, 13, 14), sys.version'

"$LIVE_DRIVER_PYTHON" - <<'PY'
from pathlib import Path

import nemo_gym
import nemo_rl
import transfer_queue

for module, root in (
    (nemo_rl, Path("/opt/nemo-rl")),
    (nemo_gym, Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym")),
):
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f"{module.__name__} imported from unexpected path: {source}")

print(
    "osworld-v2-eval-preflight-ok",
    Path(nemo_rl.__file__).resolve(),
    Path(nemo_gym.__file__).resolve(),
    Path(transfer_queue.__file__).resolve(),
)
PY
