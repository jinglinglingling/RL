# OSWorld GRPO with NeMo RL

End-to-end online reinforcement learning for multimodal computer-use agents on
[OSWorld](https://os-world.github.io/).

This project connects NeMo RL, NeMo Gym, OpenSandbox, vLLM, and Megatron to
train a vision-language agent directly from interactive desktop trajectories.
It supports multi-node rollout and training, multi-turn GRPO, checkpoint
evaluation, automatic resume, and unattended experiment recovery on Slurm.

## What this project provides

- Online OSWorld interaction instead of offline trajectory imitation
- Group Relative Policy Optimization (GRPO) over complete GUI trajectories
- Multiple tasks and multiple rollouts per optimizer step
- Independent-turn or all-turn policy training
- Multimodal context compaction for long desktop interactions
- Distributed vLLM rollout and Megatron training on Slurm
- OpenSandbox-backed OSWorld VM pools with configurable concurrency
- Periodic checkpoint evaluation and W&B experiment tracking
- Atomic checkpoint validation and safe resume across job segments
- A server-side watchdog with retry, backoff, audit logs, and optional
  Cursor-agent repair

## Current baseline

The active baseline uses:

- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- Environment: OSWorld through NeMo Gym and OpenSandbox
- Algorithm: multi-turn GRPO
- Rollout backend: vLLM
- Training backend: Megatron
- Tiny validation set: 32 mixed-reward OSWorld tasks
- Tiny batch: 8 prompts × 8 rollouts
- Scale: 4 nodes × 8 GPUs
- Compared horizons: 15 and 30 environment turns

The tiny experiments are intended to verify that rewards, advantages,
trajectory weights, optimization, checkpointing, and evaluation behave
correctly before scaling to the full OSWorld task set.

## System flow

```text
OSWorld task
    │
    ▼
OpenSandbox VM ── screenshots/actions ──► NeMo Gym agent
                                                │
                                                ▼
                                      vLLM policy rollout
                                                │
                                      grouped trajectories
                                                │
                                                ▼
                                      GRPO advantages/loss
                                                │
                                                ▼
                                      Megatron optimizer step
                                                │
                                      checkpoint + evaluation
```

## Repository entry points

| Path | Purpose |
| --- | --- |
| `docs/guides/osworld-grpo.md` | End-to-end training guide |
| `docs/guides/osworld-eval-cell2.md` | Cell-2 evaluation guide |
| `examples/nemo_gym/prepare_osworld_grpo_data.py` | Prepare full OSWorld data |
| `examples/nemo_gym/prepare_osworld_overfit_data.py` | Select tiny mixed-reward tasks |
| `examples/nemo_gym/slurm/submit_osworld_grpo.sh` | Main Slurm training launcher |
| `examples/nemo_gym/slurm/submit_osworld_checkpoint_eval.sh` | Evaluate checkpoints |
| `examples/nemo_gym/slurm/submit_osworld_overfit32_fast.sh` | Launch the tiny baseline |
| `examples/nemo_gym/slurm/OSWORLD_NIGHT_WATCHDOG.md` | Unattended monitoring guide |

## Quick start

Clone the repository and initialize its submodules:

```bash
git clone --recurse-submodules \
  ssh://git@gitlab-master.nvidia.com:12051/linglinj/osworld-grpo.git
cd osworld-grpo
```

Review the cluster and sandbox configuration described in the training guide,
then prepare the dataset:

```bash
python examples/nemo_gym/prepare_osworld_grpo_data.py --help
python examples/nemo_gym/prepare_osworld_overfit_data.py --help
```

Launch the 32-task baseline:

```bash
bash examples/nemo_gym/slurm/submit_osworld_overfit32_fast.sh
```

The launcher exposes environment variables for model paths, Slurm resources,
OSWorld pool capacity, rollout concurrency, prompt/rollout batch size, maximum
turns, checkpoint intervals, and resume behavior. See the script and
`docs/guides/osworld-grpo.md` for the required cluster-specific values.

## Resume and fault tolerance

Training checkpoints are considered resumable only when the corresponding
`step_N/training_info.json` is valid and no `tmp_step_N` directory remains.
Actor environments are created on node-local storage and reuse the container
package cache, avoiding cross-node virtual-environment corruption.

For long-running experiments, the watchdog can:

- monitor allowlisted jobs and checkpoint progress
- classify transient infrastructure failures separately from code failures
- resubmit with bounded exponential backoff
- extend successful experiments using fresh validation metrics
- invoke an isolated Cursor SDK repair agent only for unknown/code failures
- keep credentials outside the repository

See
[`examples/nemo_gym/slurm/OSWORLD_NIGHT_WATCHDOG.md`](examples/nemo_gym/slurm/OSWORLD_NIGHT_WATCHDOG.md)
for setup and safety details.

## Documentation

- [OSWorld GRPO training](docs/guides/osworld-grpo.md)
- [OSWorld evaluation on Cell-2](docs/guides/osworld-eval-cell2.md)
- [Night watchdog](examples/nemo_gym/slurm/OSWORLD_NIGHT_WATCHDOG.md)
- [Upstream NeMo RL documentation](https://docs.nvidia.com/nemo/rl/latest/)
- [Upstream NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

## Upstream

This repository is an OSWorld-focused extension of
[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL). The upstream framework,
copyright notices, and license remain in place. Project-specific changes are
concentrated in the OSWorld recipes, NeMo Gym integration, multi-turn GRPO
logic, Slurm launchers, evaluation tools, and watchdog components.
