# OSWorld 361 x 4 evaluation reproduction

This directory reproduces the paired evaluation completed by Slurm jobs
`20460932/33/35/36` and `20460992/93/94/95`.

## Frozen protocol

- 361 no-Google-Drive OSWorld tasks
- four independent rollouts per task
- four contiguous task shards: 91, 90, 90, and 90 tasks
- `max_steps=100`
- model and agent temperature `0.6`
- `top_p=1.0`
- `max_tokens=16384`
- 32 concurrent environments per shard
- one 8-GPU node per shard
- qualified container SHA-256:
  `1b4edcdeac017210e6abe25885372e025ac111c45aeb038bda334659436eb74e`

## 1. Clone the exact code

```bash
git clone --recurse-submodules \
  --branch osworld-v2-eval-repro \
  https://github.com/jinglinglingling/RL.git
cd RL
git submodule update --init --recursive
git rev-parse HEAD
git submodule status
```

The branch pins the post-run source snapshot of the Gym tree. It is an
evaluation-reproduction branch, not the Gym review branch.

## 2. Download the container

```bash
gh auth login
bash examples/nemo_gym/osworld_v2/download_container.sh /shared/containers
```

The downloader reconstructs the 69.4 GB image from private GitHub Release
assets and verifies the digest above.

## 3. Prepare the exact task shards

Prefer the four byte-identical shard files in the evaluation-evidence Release
asset. If that asset is unavailable, regenerate semantically identical shards:

```bash
python examples/nemo_gym/osworld_v2/eval_repro/prepare_shards.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/test_nogdrive.jsonl \
  --output-dir /shared/data/osworld/eval-parity-shards
```

Verify task IDs and row counts against the evidence bundle. Regenerated JSONL
can have different byte hashes because nested JSON key order is not semantic;
the evidence asset preserves the exact evaluated bytes.

## 4. Configure and submit

```bash
cp examples/nemo_gym/osworld_v2/eval_repro/env.example \
  /shared/private/osworld-eval.env
chmod 600 /shared/private/osworld-eval.env
${EDITOR:-vi} /shared/private/osworld-eval.env

OSWORLD_SUBMIT_DRY_RUN=1 \
  bash examples/nemo_gym/osworld_v2/eval_repro/submit.sh \
  /shared/private/osworld-eval.env

bash examples/nemo_gym/osworld_v2/eval_repro/submit.sh \
  /shared/private/osworld-eval.env
```

For RFC0040 iter 881, point `NANO_OMNI_MODEL_NAME` at its Hugging Face export.
For the SFT control, point it at RFC0037 step400. Keep every other environment
value unchanged.

## 5. Aggregate

After all four jobs complete:

```bash
python examples/nemo_gym/osworld_v2/eval_repro/aggregate_results.py \
  /shared/results/osworld-eval/RUN_PREFIX-TIMESTAMP
```

Expected reference results:

- RFC0040 iter 881: Pass@1 `49.5845%`, Pass@4 `70.6371%`
- RFC0037 step400 SFT: Pass@1 `48.4765%`, Pass@4 `71.1911%`

## Cross-cluster comparison checklist

Compare these before interpreting score differences:

1. RL commit and Gym submodule commit.
2. Container SHA-256.
3. Model `config.json`, `chat_template.jinja`, and
   `model.safetensors.index.json` SHA-256 values.
4. Task manifest and all four shard hashes.
5. Resolved sampling values (`temperature`, `top_p`, token cap, four samples).
6. Agent `max_steps`, concurrency, model-readiness timeout, and Slurm wall time.
7. OpenSandbox pool/template version and VM image.
8. `val_data_step0.jsonl` row counts: 364, 360, 360, 360.

Do not compare only the final accuracy: a task-manifest, template, or sandbox
image mismatch can change the benchmark even when the command line looks alike.
# OSWorld 361 x 4 evaluation reproduction

This directory reproduces the paired evaluation completed on 2026-10-19:

- RFC0040 `iter_0000881`: Pass@1 `49.5845%`, Pass@4 `70.6371%`
- RFC0037 SFT step400: Pass@1 `48.4765%`, Pass@4 `71.1911%`

The eight successful Slurm jobs were `20460932`, `20460933`, `20460935`,
`20460936`, and `20460992` through `20460995`. Each model used 361 tasks,
four rollouts per task, 100 agent steps, temperature 0.6, top-p 1.0, and the
same qualified container.

## Fetch code and container

```bash
git clone --recurse-submodules \
  --branch osworld-v2-eval-repro \
  https://github.com/jinglinglingling/RL.git
cd RL
git submodule update --init --recursive

gh auth login
bash examples/nemo_gym/osworld_v2/download_container.sh /shared/containers
```

The container must hash to:

```text
1b4edcdeac017210e6abe25885372e025ac111c45aeb038bda334659436eb74e
```

The exact reference logs, outputs, task shards, source snapshot, and manifest
are attached to the private release
`jinglinglingling/nemo-rl-molt-osworld-backup@osworld-v2-paired-eval-20261018`.

## Prepare the task shards

```bash
python examples/nemo_gym/osworld_v2/eval_repro/prepare_shards.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/test_nogdrive.jsonl \
  --output-dir /shared/data/osworld/eval-parity-shards
```

Compare the generated `manifest.json` with the reference release before
submitting. The contiguous shard sizes must be `91, 90, 90, 90`.

## Submit one model

```bash
cp examples/nemo_gym/osworld_v2/eval_repro/env.example \
  /shared/private/osworld-eval.env
chmod 600 /shared/private/osworld-eval.env
${EDITOR:-vi} /shared/private/osworld-eval.env

OSWORLD_SUBMIT_DRY_RUN=1 \
  bash examples/nemo_gym/osworld_v2/eval_repro/submit.sh \
  /shared/private/osworld-eval.env

bash examples/nemo_gym/osworld_v2/eval_repro/submit.sh \
  /shared/private/osworld-eval.env
```

This launches four independent one-node jobs. Set `NANO_OMNI_MODEL_NAME`,
`NANO_OMNI_CHAT_TEMPLATE`, and `OSWORLD_EVAL_RUN_PREFIX` for each model.

## Aggregate

```bash
python examples/nemo_gym/osworld_v2/eval_repro/aggregate_results.py \
  /shared/results/osworld-eval/RUN_PREFIX-TIMESTAMP
```

The aggregator requires exactly 361 task groups and 1,444 completed rollouts.

## Cross-server parity checklist

Check these in order:

1. RL commit and Gym submodule commit match this branch and are clean.
2. Container SHA-256 matches the value above.
3. The four shard hashes match the reference `manifest.json`.
4. Model `config.json`, `chat_template.jinja`, and
   `model.safetensors.index.json` hashes match the reference metadata.
5. The resolved recipe keeps temperature 0.6, top-p 1.0, max tokens 16,384,
   max model length 49,152, 100 agent steps, and four generations per task.
6. OpenSandbox uses the same `osworld-kvm` pool and every task reaches its
   evaluator rather than timing out or returning setup errors.
7. Each shard produces 364, 360, 360, and 360 rows respectively before
   comparing Pass@1 or Pass@4.
