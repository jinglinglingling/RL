# OSWorld + NeMo-RL End-to-End Pipeline

This folder contains a runnable VLM-RL end-to-end pipeline:

1. Collect trajectories from OSWorld with a Qwen-VL model.
2. Convert trajectories (`traj.jsonl` + screenshots) to NeMo-RL multimodal JSONL.
3. Train with NeMo-RL GRPO using the converted data.

The pipeline is orchestrated by:

- `scripts/e2e_vlm_rl_pipeline.sh`

## Layout

- `scripts/setup_osworld_nemorl_env.sh`: setup dependencies and source paths.
- `scripts/run_qwen_vl_openai_server.sh`: launch Qwen-VL OpenAI-compatible server.
- `scripts/run_osworld_eval_qwen_vl.sh`: run OSWorld multi-env eval.
- `scripts/convert_osworld_results_to_nemorl.py`: convert OSWorld results to JSONL.
- `scripts/run_nemorl_osworld_grpo.py`: NeMo-RL runner that registers custom processor.
- `scripts/run_nemorl_osworld_grpo.sh`: shell entrypoint for training.
- `scripts/e2e_vlm_rl_pipeline.sh`: stage orchestration (setup/collect/convert/train).
- `scripts/slurm/run_osworld_collect_convert_on_node.sh`: run collect+convert on an allocated GPU node.
- `scripts/slurm/submit_osworld_collect_convert.sh`: submit collect+convert as a Slurm job.
- `scripts/slurm/submit_nemorl_osworld_train.sh`: submit NeMo-RL training with ray.sub/container style.
- `scripts/slurm/run_nemorl_osworld_inside_container.sh`: run training inside attached container shell.
- `src/osworld_nemorl/nemorl_processor.py`: custom VLM processor for NeMo-RL.
- `configs/nemorl_osworld_grpo_qwen_vl.yaml`: training config.
- `configs/pipeline.env.example`: template env file for the end-to-end pipeline.
- `configs/slurm.env.example`: template env for sbatch / interactive workflows.

## Prerequisites

- Python 3.10+
- OSWorld source code (default expected path):
  - `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld/third_party/OSWorld`
- NeMo-RL source code (default expected path):
  - `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/Nemo-RL-main-1/RL-merge-2689`

Install helper dependencies for this workspace:

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
python -m pip install -r requirements.txt
```

## Quick Start (End-to-End)

1) Create runtime env file:

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
cp configs/pipeline.env.example configs/pipeline.env
```

2) Start Qwen-VL OpenAI-compatible server in a separate terminal:

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
bash scripts/run_qwen_vl_openai_server.sh
```

3) Run the full pipeline:

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
bash scripts/e2e_vlm_rl_pipeline.sh --stage all
```

You can resume from a stage:

```bash
bash scripts/e2e_vlm_rl_pipeline.sh --stage all --from-stage convert
```

Run only one stage:

```bash
bash scripts/e2e_vlm_rl_pipeline.sh --stage train
```

Pass NeMo-RL Hydra overrides:

```bash
bash scripts/e2e_vlm_rl_pipeline.sh --stage train -- grpo.max_num_steps=100
```

## Slurm / Interactive Workflow

If you usually run via `sbatch` or an interactive GPU node, use these scripts.

### A) Submit OSWorld collect+convert via sbatch

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
source configs/slurm.env.example
bash scripts/slurm/submit_osworld_collect_convert.sh
```

To launch vLLM from your enroot image (ppo-style), pass:

```bash
VLLM_BACKEND=container \
VLLM_CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/enroot-images/nvcr.io/nvidian/nemo-rl:nightly-ultra.squashfs \
bash scripts/slurm/submit_osworld_collect_convert.sh
```

### B) Run collect+convert on an already allocated interactive GPU node

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
source configs/slurm.env.example
bash scripts/slurm/run_osworld_collect_convert_on_node.sh
```

### C) Submit NeMo-RL training in your ray.sub/container style

This mirrors your `Nemo-RL-ppo` style (`sbatch ray.sub` + `export COMMAND=...`).

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
source configs/slurm.env.example
CONTAINER=/path/to/your/nemo-rl.squashfs \
bash scripts/slurm/submit_nemorl_osworld_train.sh
```

### D) Inside attached container shell, run training manually

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
bash scripts/slurm/run_nemorl_osworld_inside_container.sh
```

## Stage-by-Stage Commands

### 1) Run OSWorld evaluation with Qwen-VL

OSWorld's agent routes OpenAI-compatible calls through model names starting with `gpt`.
The easiest setup is serving Qwen-VL with an alias (for example `gpt-4o`) in your API server.

Example (vLLM server side):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name gpt-4o \
  --port 8000 \
  --max-model-len 8192
```

Run OSWorld:

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
OPENAI_BASE_URL="http://127.0.0.1:8000/v1" \
OPENAI_API_KEY="EMPTY" \
bash scripts/run_osworld_eval_qwen_vl.sh
```

Results are written under `runs/osworld-results/`.

### 2) Convert OSWorld trajectories to NeMo-RL JSONL

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
python scripts/convert_osworld_results_to_nemorl.py \
  --results-root runs/osworld-results \
  --examples-root third_party/OSWorld/evaluation_examples/examples \
  --output-train data/nemorl/osworld_train.jsonl \
  --output-val data/nemorl/osworld_val.jsonl
```

### 3) Train with NeMo-RL (GRPO, imitation reward)

This setup uses a custom processor and the built-in `vlm` environment with
`exact_alnum` reward against demonstrated action strings from trajectories.

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/Nemo-RL-Library/osworld
bash scripts/run_nemorl_osworld_grpo.sh
```

You can pass Hydra overrides:

```bash
bash scripts/run_nemorl_osworld_grpo.sh \
  grpo.max_num_steps=100 \
  policy.train_global_batch_size=8
```

## Reference: Two-stage SWE RL (SWE1 pivot + SWE2 agentic)

Based on the recent Gym-side SWE updates, a practical two-stage SWE RL recipe can be summarized as:

### SWE1: Pivot RL (single-step, local decision optimization)

- Core environment/verifier is `resources_servers/swe_pivot`.
- Typical server+agent config is `resources_servers/swe_pivot/configs/swe_pivot.yaml`.
- Rollouts are single-step (`max_steps: 1`) next-action decisions at curated pivot turns.
- Training signal comes from action-level verification (tool category / target / argument similarity), not full end-to-end task completion.
- Goal: quickly improve local fix-design decisions before expensive long-horizon rollouts.

### SWE2: Agentic RL (full-horizon, end-to-end issue solving)

- Core agent is `responses_api_agents/swe_agents` (OpenHands-based harness).
- Typical training config is `responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml`.
- Each sample runs a full coding trajectory (multi-turn tool use inside task containers), then executes benchmark harness tests.
- Reward is end-to-end resolved/not-resolved style signal (`0/1`) after patch evaluation.
- Goal: optimize full issue-resolution behavior under realistic agentic execution.

### Is this based on `mini-swe-agentic`?

- Not directly for the two-stage recipe above.
- In the current Gym tree, the two stages map to `swe_pivot` + `swe_agents` (OpenHands path).
- `responses_api_agents/mini_swe_agent` exists as a separate Mini-SWE-Agent integration, but it is not the core implementation used by the SWE1+SWE2 path above.

### Key differences vs our current OSWorld online GRPO

- OSWorld currently uses `resources_servers/osworld_vlm` + `responses_api_agents/osworld_vlm_agent`.
- OSWorld is multimodal GUI interaction (screenshot observation + desktop actions), while SWE1/2 is code-agent/tool-call centric.
- OSWorld reward is currently task-evaluation driven (mostly sparse `0/1` from `DesktopEnv.evaluate()` in our smoke runs).
- SWE1 adds dense local reward shaping at pivot turns; OSWorld currently does not have an equivalent pivot verifier stage.
- SWE2 and OSWorld are both multi-turn agentic RL in NeMo-Gym, but they differ in environment dynamics, action space, and verifier granularity.
