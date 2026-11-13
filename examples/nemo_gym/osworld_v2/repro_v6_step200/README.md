# Reproducing OSWorld v6 step200 (FlashREINFORCE B32×N4, union2698, AgentEnv)

This branch pins the exact NeMo RL + NeMo Gym code, recipe and launcher used to train the v6 run to `step_200`,
plus the HF export and the evaluation/aggregation procedure behind the reported numbers.

Paths under `/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/` are on the NRT/slurm-1 Lustre
(mirrored on HSG where noted). Data, base model, container and credentials are **not** in git; ask the owner for
read access.

## Expected result

OSWorld eval-361 (no-gdrive), `osworld-v1` template, 4 seeds × 361 tasks, 49,152 context, 200 steps,
T=0.6, top_p=0.95, 7,200 s per-episode budget:

| Model | Pass@1 | Pass@4 | Mean steps |
|---|---|---|---|
| RFC0040 iter881 (base) | 51.74 | 65.10 | 37.0 |
| v6 step200 | **57.90** | **69.53** | 29.3 |

Paired step200 − iter881: Pass@1 +6.16 [+3.58, +8.80], Pass@4 +4.43 [+1.39, +7.76] (task-level bootstrap, 95%).
Per-seed Pass@1 of step200: 59.18 / 57.37 / 59.77 / 55.30, so expect roughly ±2 pt seed noise.

## 1. Code

```bash
git clone --branch linglinj/osworld-v6-step200-repro https://github.com/jinglinglingling/RL.git
cd RL
git submodule update --init 3rdparty/Gym-workspace/Gym 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
(cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge && git submodule update --init 3rdparty/Megatron-LM)
(cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
  git apply ../../../../../examples/nemo_gym/osworld_v2/repro_v6_step200/patches/megatron-lm-1e7598cb-dcp-picklable-exception.patch)
```

| Component | Commit |
|---|---|
| NeMo Gym | `b9cff9f1` on `jinglinglingling/Gym` branch `linglinj/osworld-v6-step200-repro`: the training code (`a71dd777`) plus kill-on-timeout for AgentEnv sandboxes |
| Megatron-Bridge | `5ed9799` (upstream) |
| Megatron-LM | `1e7598cb` (upstream) + `patches/megatron-lm-1e7598cb-dcp-picklable-exception.patch` (only keeps DCP checkpoint errors picklable on Python 3.13; no numerical effect) |

General launcher documentation (prerequisites, Pyxis, AgentEnv TLS, evaluator tools): `../README.md`.

## 2. Inputs

| Input | Location |
|---|---|
| Container | `.../users/linglinj/artifacts/containers/rl-osworld-v2-qualified-69764640.sqsh` (sha256 `1b4edcde…`, checked by `submit.sh`) |
| Base policy | `.../users/yifazhang/rfc0040_k8_s3_hf_native_runtime_recovery_v1_production_20260820/exports/iter_0000881/mcore_to_hf` (+ `chat_template.jinja`) |
| Train data | `.../users/linglinj/artifacts/data/osworld/union2698-astra-assets-20261028/yi-rpga-union2698-exact-20261022/` (`train.jsonl`, 2,698 tasks, `assets/`, `tmp_funcs/`) |
| Canonical harness | `.../users/linglinj/artifacts/data/osworld/yi-rl-band1782-exact-20261021/harness` (agent sha256 `0420a951…`, feedback `ef74f8c3…`). This directory is not shared; use the `harness/` copy from the internal bundle, cloned onto Lustre (same sha256s) |
| Trained checkpoints (HF) | step200: `.../users/linglinj/artifacts/models/v6_step200_hf` (NRT), `.../users/linglinj/osworld-hsg-eval/models/v6_step200_hf` (HSG); iter881: `.../users/linglinj/artifacts/models/rfc0040_iter881` (HSG) |
| Evaluator tools rootfs | `.../users/linglinj/artifacts/cache/osworld-evaluator-tools/rootfs` |
| Sandboxes | AgentEnv cell-3, templates `osworld-v1:1,osworld-slim:1,osworld-v2:1` |

## 3. Training

Recipe: `examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_yi_union2698_agentenv_flash_b32n4.yaml`
(`OSWORLD_RECIPE=yi-union2698-agentenv-flash-b32n4`). The fully resolved config of the original run at step 200 is
in `config_step200_resolved.yaml`.

Key settings: FlashREINFORCE, 32 prompts × 4 generations per step, binary-KL correction gate, no reference KL,
constant lr 5e-6, seed 43, `save_period` 2; 4 nodes × 8 GPUs (1 Megatron policy node, 3 vLLM generation nodes),
80 parallel rollouts, 32 in-flight prompts; agent max 200 turns, `rollout_timeout_s` 3600, RFC0040 K8/S3 context
policy (old screenshots replaced by an omission marker, at most 11 images), 49,152-token context guard with 8,192
reserved for generation. All text history including past thinking is kept, so about 12–13% of rollouts hit the
context guard (median turn 55) at every stage of training; that is expected. Every segment of a context-compacted
rollout is trained with the trajectory's reward.

The original run used two loss-aggregation phases; reproduce them the same way:

1. **Steps 1–149, token-mean.** Fill in `env/phase1_token_mean.env.example` (private copy, `chmod 600`) and run
   `bash examples/nemo_gym/osworld_v2/submit.sh <phase1.env>`. It submits a chain of 4 h segments that resume from
   the last checkpoint; it stops after `step_149`.
2. **Steps 150–200, seq-mean-token-mean.** Fill in `env/phase2_seq_mean_token_mean.env.example` and submit it the
   same way. It resumes from `step_149` and copies it to `checkpoints/preserved_last_token_mean` first.

History of the original run, for reference. The inputs above are the final ones and are what we recommend
using throughout.

- Steps 1–~6 ran with `OSWORLD_GRPO_MAX_STEPS=100`, raised to 200 afterwards. lr is constant, so this does not
  change the schedule.
- Steps 1–~32 read tasks from `.../users/linglinj/artifacts/data/osworld/yi-rpga-union2698-exact-20261022`, with
  Yi's asset root and setup cache.
- Steps ~33–87 used `train.agentenv-safe-v1.jsonl` from that snapshot. It differs only in a best-effort clipboard
  setup for task `os_r4_1158_v2`.
- Steps 88–149 and all of phase 2 used the local-asset copy listed above: same 2,698 rows in the same order, and the
  same harness sha256.

## 4. HF export

`hf_export/export_hf.sbatch` with `hf_export/export_megatron_to_hf_gpu_wrapper.py` (tp1/pp1/ep8/etp1, 1 node).
On slurm-1 set `LUSTRE_MOUNT=/scratch/fsw:/lustre/fs1`. The wrapper only makes the copied read-only `modeling*.py` files writable before the Bridge
post-processing, which otherwise raises `PermissionError` on rank 0 and hangs ranks 1–7. Afterwards, replace all
non-weight files with the copies from the iter881 HF export, as described in the sbatch header. Sanity check:
17 shards, 7,349 tensors with key/shape/dtype identical to iter881. LLM weights differ by about 0.1–0.3% relative,
and the vision and sound towers are unchanged.

## 5. Evaluation (numbers above)

The reported eval used the RPGA OSWorld harness on the HSG cluster (internal repo, commit `eda332d9`). The harness is
not part of this branch. Contract: `rfc0015_v1` agent / `hf_tools` / `feedback_v2`, K8/S3, 200 steps, T=0.6,
top_p=0.95, 16,384 tokens per turn, 49,152 context, `osworld-v1` template (forced by `EVAL_ONLY=1`). One node,
4 TP1 vLLM engines, 32 concurrent episodes, about 1 h 10 min per seed.

On HSG, from the RPGA copy of the internal bundle (cloned onto Lustre), with `R` set to a directory you own and your
own copy of the cell-3 credential file (`endpoint=` / `apikey=` lines, mode 600) and CA:

```bash
cd <bundle>/rpga/scripts/osworld
export R=<your output dir> CELL2_CRED_FILE=<your .cell3_info> AENV_TLS_CA=<your cell3_ca.pem>
F=/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/users/linglinj/osworld-hsg-eval/data/validation_nogdrive_361_gym.jsonl
for s in 1 2 3 4; do
  RUN_ID=<name>_s$s SEED=$s MODEL_PATH=<hf export> VAL_DATA=$F TRAIN_DATA=$F \
  EVAL_BATCH_SIZE=361 ROLLOUT_GENERATE_BATCH_SIZE=361 OSWORLD_ROLLOUT_TIMEOUT_SEC=7200 \
    sbatch -q interactive -t 04:00:00 osworld_eval_hsg.sbatch
done
```

Required harness fix: several eval-361 tasks pass `expected_returncodes` / `on_nonzero` to setup steps, and the
pinned OSWorld `SetupController` rejects them. `eval/rpga_setup_returncode_contract.py` contains the patch applied
inside the agent (same semantics as NeMo Gym `osworld_agent/client.py`). Without it, those tasks fail as infra errors.

Aggregation: `eval/aggregate_paired.py --tasks $F --model step200=<run_s1>,...,<run_s4> --model iter881=...`
prints Pass@1, Pass@4, paired bootstrap CIs and per-domain scores. Four tasks fail with a `killall` setup error for
every model and seed. Excluding them gives 58.55 / 70.31 for step200 and 52.32 / 65.83
for iter881.

The iter881 baseline seeds 1–2 were run with the 3,600 s budget, and their timed-out or returncode-affected tasks
were re-run at 7,200 s. Seeds 3–4 ran at 7,200 s throughout. All step200 seeds ran at 7,200 s.
