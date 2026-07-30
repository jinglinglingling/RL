# OSWorld GRPO with the cell-2 OpenSandbox pool

This path combines NeMo-RL's Nemotron Omni GRPO support with NeMo-Gym's
OpenSandbox-backed OSWorld environment. The GPU job calls the remote cell-2 API;
it does not need local KVM access.

## Required environment

```bash
export OPENSANDBOX_DOMAIN=k8s-opensand-opensand-79e2505e2f-79f31e302dbc949f.elb.us-east-2.amazonaws.com
export OPENSANDBOX_API_KEY=<cell-2 key>
export OSWORLD_POOL_REF=osworld-kvm
```

`OPENSANDBOX_DOMAIN` is a host name, without an `http://` prefix. The provider's
configuration supplies the protocol.

Before launching GPUs, validate create, guest execute, screenshot, and cleanup:

```bash
cd 3rdparty/Gym-workspace/Gym
uv run --extra sandbox python scripts/sandbox_runtime_poc/smoke_cell2.py
# After the single-sandbox check, validate the pool's rollout concurrency:
uv run --extra sandbox python scripts/sandbox_runtime_poc/smoke_cell2.py --concurrency 4
cd ../../..
```

## Prepare a four-rollout smoke dataset

```bash
python examples/nemo_gym/prepare_osworld_grpo_data.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/smoke.jsonl \
  --output /tmp/osworld_grpo_smoke.jsonl \
  --limit 1 \
  --num-repeats 4
export OSWORLD_GRPO_TRAIN_DATA=/tmp/osworld_grpo_smoke.jsonl
```

## Run one GRPO step

```bash
uv run python examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-1n8g-megatron.v1.yaml \
  grpo.max_num_steps=1 \
  checkpointing.enabled=false \
  logger.wandb_enabled=false
```

The OSWorld agent uses a sliding three-image context. Consecutive prompts are
therefore not prefix-contiguous. The initial implementation samples one exact
prompt-generation pair from each completed trajectory and gives it the
trajectory-level GRPO advantage. This preserves the rollout conditioning and a
fixed training batch size; `env.nemo_gym.independent_turn_sampling` accepts
`first`, `last`, or `random`.

Infrastructure failures are returned with `mask_sample=true`, so sandbox
allocation, boot, transport, or cleanup failures do not become negative task
examples.
