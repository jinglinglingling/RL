# OSWorld GRPO with the cell-2 OpenSandbox pool

This path combines NeMo-RL's Nemotron Omni GRPO support with NeMo-Gym's
OpenSandbox-backed OSWorld environment. The GPU job calls the remote cell-2 API;
it does not need local KVM access.

## Get the matching RL and Gym revisions

The feature branch pins its matching Gym fork as a Git submodule:

```bash
git clone --recurse-submodules \
  --branch osworld-grpo-3290 \
  https://github.com/jinglinglingling/RL.git
cd RL
```

For an existing checkout, run `git submodule sync --recursive` followed by
`git submodule update --init --recursive`.

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

The Slurm launcher defaults to NVIDIA's internal `batch` partition and shared
NeMo-RL enroot image. Override `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `CONTAINER`,
and `MOUNTS` for another cluster. The Nemotron Omni checkpoint must already be
available in `HF_HOME` when `HF_HUB_OFFLINE=1` (the default).

## Prepare training data

Create a one-task smoke dataset. GRPO creates the configured number of
generations for each row, so input rows should not be duplicated to obtain a
rollout group:

```bash
python examples/nemo_gym/prepare_osworld_grpo_data.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/smoke.jsonl \
  --output /tmp/osworld_grpo_smoke.jsonl \
  --limit 1 \
  --num-repeats 1
export OSWORLD_GRPO_TRAIN_DATA=/tmp/osworld_grpo_smoke.jsonl
```

For a deterministic, domain-stratified train/held-out split of the 361
non-Google-Drive tasks:

```bash
python examples/nemo_gym/tools/split_osworld_grpo_data.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/test_nogdrive.jsonl \
  --output-dir results/osworld-data/split-20260727 \
  --train-fraction 0.8 \
  --seed 20260727
export OSWORLD_GRPO_TRAIN_DATA="$PWD/results/osworld-data/split-20260727/train.jsonl"
```

Do not train on `heldout.jsonl`; use it only for frozen-model evaluation.

## Run a one-step smoke

```bash
uv run python examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-osworld-1n8g-megatron.v1.yaml \
  grpo.max_num_steps=1 \
  checkpointing.enabled=false \
  logger.wandb_enabled=false
```

On Slurm:

```bash
export NUM_NODES=1
export GRPO_MAX_NUM_STEPS=1
export OSWORLD_NUM_GENERATIONS=4
export OSWORLD_TRAIN_GLOBAL_BATCH_SIZE=4
export OSWORLD_MAX_STEPS=5
export RESULTS_DIR="$PWD/results/osworld-grpo-smoke"
export CHECKPOINTING_ENABLED=false
export WANDB_ENABLED=false
bash examples/nemo_gym/slurm/submit_osworld_grpo.sh
```

Check the Slurm log, TensorBoard metrics, and screenshots under
`OSWORLD_DEBUG_TRAJ_DIR` before scaling. The screenshot checker detects missing,
nearly black, or nearly uniform frames:

```bash
python examples/nemo_gym/tools/check_osworld_screenshots.py \
  "$OSWORLD_DEBUG_TRAJ_DIR"
```

## Run formal training

Use a unique result directory, checkpoint directory, and W&B run ID for every
rollout-group configuration. This example runs rollout-4 on two eight-GPU
nodes:

```bash
export OSWORLD_GRPO_TRAIN_DATA="$PWD/results/osworld-data/split-20260727/train.jsonl"
export NUM_NODES=2
export GRPO_MAX_NUM_STEPS=30
export OSWORLD_NUM_GENERATIONS=4
export OSWORLD_TRAIN_GLOBAL_BATCH_SIZE=4
export OSWORLD_MAX_STEPS=15
export RESULTS_DIR="$PWD/results/osworld-grpo-r4-30step"
export CHECKPOINTING_ENABLED=true
export CHECKPOINT_DIR="$RESULTS_DIR/checkpoints"
export CHECKPOINT_SAVE_PERIOD=1
export OSWORLD_DEBUG_TRAJ_DIR="$RESULTS_DIR/debug-trajectories"
export WANDB_ENABLED=true
export WANDB_ENTITY=<wandb entity>
export WANDB_PROJECT=osworld-grpo
export WANDB_RUN_NAME=nemotron-omni-osworld-r4-30step
export WANDB_RUN_ID="$WANDB_RUN_NAME"
export WANDB_RESUME=allow
export WANDB_API_KEY=<wandb API key>
bash examples/nemo_gym/slurm/submit_osworld_grpo.sh
```

For rollout-8, set both `OSWORLD_NUM_GENERATIONS` and
`OSWORLD_TRAIN_GLOBAL_BATCH_SIZE` to `8`, and use new result/checkpoint/W&B
names. The agent limits OpenSandbox work to four concurrent sandboxes, so the
eight trajectories are allocated in waves. Never resume rollout-4 from a
rollout-8 checkpoint or the reverse.

## Resume across a four-hour Slurm limit

Every submission loads the latest complete checkpoint in `CHECKPOINT_DIR`.
Chain jobs with `afterany` so a timeout or transient Cell-2 failure still starts
the next recovery attempt:

```bash
submit_part() {
  local dependency="${1:-}"
  SBATCH_DEPENDENCY="$dependency" \
    bash examples/nemo_gym/slurm/submit_osworld_grpo.sh |
    tee /dev/stderr |
    awk '/Submitted batch job/{print $4}'
}

job_id="$(submit_part)"
for _ in {2..8}; do
  job_id="$(submit_part "$job_id")"
done
echo "Last chained job: $job_id"
```

Keep all configuration and W&B variables exported while constructing the
chain. `checkpointing.keep_top_k=1` retains only the newest complete checkpoint;
one checkpoint is roughly 480 GB for this optimizer configuration, so leave
space for both the current checkpoint and a temporary replacement.

## Rollout and training semantics

The current recipe retains only the latest screenshot while token-aware
multimodal context compaction is unavailable. Consecutive prompts are still not
prefix-contiguous. Training therefore samples one exact prompt-generation pair
from each completed trajectory and assigns it the trajectory-level GRPO
advantage. `env.nemo_gym.independent_turn_sampling` accepts `first`, `last`, or
`random`.

Infrastructure failures are returned with `mask_sample=true`, so sandbox
allocation, boot, transport, or cleanup failures do not become negative task
examples.

Dynamic sampling rejects all-success and all-failure groups because they have
zero reward variance. Task IDs and snapshots are retained in rollout logs so
the resulting learnable-task distribution can be audited.

For frozen-model scoring of the held-out split, follow
`docs/guides/osworld-eval-cell2.md`.
