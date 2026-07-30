# OSWorld evaluation: local GPUs with cell-2 sandboxes

This is the frozen-model evaluation path. It is intentionally separate from
`osworld-grpo.md`: evaluation does not collect rollout logprobs and does not
update model weights.

## Architecture

The driver and vLLM server run on the local GPU cluster. Only sandbox lifecycle,
desktop actions, screenshots, and evaluator traffic cross the network to the
cell-2 `osworld-kvm` pool:

```text
local GPU node: vLLM 0.23.0 (frozen Nemotron Omni)
        ↑ OpenAI-compatible prompt/screenshot requests
local driver: gym eval run + Nemotron OSWorld agent
        ↕ OpenSandbox HTTP/path proxy
cell-2: osworld-kvm pool → QEMU/KVM desktop VMs
```

## Why run this experiment

Terry's report does not currently support a model-API root cause by itself:

- the best full run was 158/361 (43.8%) versus the 48.28–48.44% reference;
- an exact TP8/H100 Slurm-serving run was 96/229 (41.9%) when stopped, with no
  visible serving improvement;
- the union of two seeds reaches the reference band on the weak domains;
- four transport/evaluator bugs explained much of the original gap.

We should still run a controlled paired comparison. The local endpoint and the
comparison endpoint must use the same checkpoint, task rows, sampling settings,
agent/parser, OSWorld revision, cell-2 pool, and concurrency. Change only
`POLICY_BASE_URL` and `POLICY_API_KEY`.

## Required environment

```bash
export OPENSANDBOX_DOMAIN=k8s-opensand-opensand-79e2505e2f-79f31e302dbc949f.elb.us-east-2.amazonaws.com
export OPENSANDBOX_API_KEY=<cell-2 key>
export OSWORLD_POOL_REF=osworld-kvm
```

## Start the model on a local eight-GPU allocation

Run inside the official `vllm/vllm-openai:v0.23.0` image:

```bash
TP_SIZE=8 PORT=8000 \
  bash examples/nemo_gym/serve_osworld_eval_vllm.sh
```

The launcher pins the public reference checkpoint, `nano_v3` reasoning parser,
xgrammar structured-output backend, float32 Mamba SSM cache, three-image limit,
and the `vllm_local` served model name.

For the one-node TP8/H100 Slurm path, submit both the server and evaluation
driver as one four-hour job:

```bash
export OPENSANDBOX_DOMAIN=<cell-2 host>
export OPENSANDBOX_API_KEY=<cell-2 key>
export OSWORLD_EVAL_INPUT=resources_servers/osworld/data/smoke.jsonl
export OSWORLD_EVAL_OUTPUT=results/osworld-eval/local-tp8-smoke.jsonl
bash examples/nemo_gym/slurm/submit_osworld_eval_local_gpu.sh
```

The wrapper defaults to account `coreai_dlalgo_nemorl`, partition `batch`,
one exclusive node, eight GPUs, and `vllm/vllm-openai:v0.23.0`. All values can
be overridden through environment variables.

## Run the evaluation driver

Start with the four-row smoke set:

```bash
export POLICY_BASE_URL=http://<gpu-node>:8000/v1
export POLICY_API_KEY=EMPTY
export OSWORLD_EVAL_INPUT=resources_servers/osworld/data/smoke.jsonl
export OSWORLD_EVAL_OUTPUT=results/osworld-eval/local-smoke.jsonl
export OSWORLD_EVAL_CONCURRENCY=4
bash examples/nemo_gym/run_osworld_eval.sh
```

Then use a fixed diagnostic subset containing the historically weak `calc`,
`chrome`, and `vlc` tasks. Run the exact same rows at least twice per endpoint
because the reference uses temperature 0.6:

```bash
python examples/nemo_gym/prepare_osworld_eval_subset.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/test_nogdrive.jsonl \
  --output /tmp/osworld_eval_calc_chrome_vlc.jsonl \
  --domains calc chrome vlc \
  --per-domain 4

OSWORLD_EVAL_INPUT=/tmp/osworld_eval_calc_chrome_vlc.jsonl \
OSWORLD_EVAL_OUTPUT=results/osworld-eval/local-seed-1.jsonl \
bash examples/nemo_gym/run_osworld_eval.sh
```

For the API comparison, change only:

```bash
export POLICY_BASE_URL=<comparison OpenAI-compatible /v1 URL>
export POLICY_API_KEY=<comparison key>
export OSWORLD_EVAL_OUTPUT=results/osworld-eval/api-seed-1.jsonl
bash examples/nemo_gym/run_osworld_eval.sh
```

Do not compare unpaired aggregate percentages first. Compare per-task outcomes,
termination reason, action count, parser failures, and saved screenshot/action
trajectories. Enable trajectory capture with:

```bash
export OSWORLD_DEBUG_TRAJ_DIR=results/osworld-eval/trajectories/<run-name>
```

If local and API runs diverge, replay the same divergent task again at low
concurrency before attributing the difference to serving.

Generate the paired task-level comparison with:

```bash
python examples/nemo_gym/compare_osworld_eval_runs.py \
  --local results/osworld-eval/local-seed-1.jsonl \
  --api results/osworld-eval/api-seed-1.jsonl \
  --output results/osworld-eval/local-vs-api-seed-1.json
```
