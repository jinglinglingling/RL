#!/usr/bin/env python3
"""Pass@1 / Pass@4 with paired bootstrap CIs over RPGA OSWorld eval runs.

Each --model takes NAME=RUN_DIR[,RUN_DIR...] with one run directory per seed; every run directory must contain
osworld_results/episodes.jsonl. The first model is the candidate and is compared against each other model.

  python3 aggregate_paired.py --tasks validation_nogdrive_361_gym.jsonl \
    --model step200=$R/runs/hsg_v1_t7200_v6step200_s1,...,$R/runs/hsg_v1_t7200_v6step200_s4 \
    --model iter881=...
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path


def load_run(run_dir):
    episodes = {}
    with open(Path(run_dir) / "osworld_results" / "episodes.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            episodes[row["task_id"]] = row
    return episodes


def bootstrap(values, iters=4000, seed=0):
    rng = random.Random(seed)
    samples = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(iters))
    return st.mean(values), samples[int(0.025 * iters)], samples[int(0.975 * iters) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="eval task jsonl (field 'id' = '<domain>/<task_id>')")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=RUN_DIR[,RUN_DIR...]")
    parser.add_argument("--success-threshold", type=float, default=0.99)
    args = parser.parse_args()

    task_ids, domain = [], {}
    for line in open(args.tasks):
        full_id = json.loads(line)["id"]
        task_id = full_id.split("/")[-1]
        task_ids.append(task_id)
        domain[task_id] = full_id.split("/")[0]

    runs = {}
    for spec in args.model:
        name, dirs = spec.split("=", 1)
        runs[name] = [load_run(d) for d in dirs.split(",")]
        for seed_index, run in enumerate(runs[name]):
            missing = [t for t in task_ids if t not in run]
            if missing:
                raise SystemExit(f"{name} seed {seed_index + 1}: {len(missing)} tasks missing")

    def pass1(name, task):
        return st.mean(run[task]["score"] for run in runs[name])

    def passk(name, task):
        return float(any(run[task]["score"] > args.success_threshold for run in runs[name]))

    names = list(runs)
    print(f"n_tasks={len(task_ids)}")
    for name in names:
        k = len(runs[name])
        steps = st.mean(run[t]["steps"] for run in runs[name] for t in task_ids)
        print(
            f"{name:12s} Pass@1 {100 * st.mean(pass1(name, t) for t in task_ids):6.2f}  "
            f"Pass@{k} {100 * st.mean(passk(name, t) for t in task_ids):6.2f}  mean steps {steps:5.1f}  "
            f"per-seed Pass@1 {[round(100 * st.mean(run[t]['score'] for t in task_ids), 2) for run in runs[name]]}"
        )
    candidate = names[0]
    for other in names[1:]:
        d1 = bootstrap([pass1(candidate, t) - pass1(other, t) for t in task_ids])
        dk = bootstrap([passk(candidate, t) - passk(other, t) for t in task_ids])
        print(
            f"{candidate} - {other}: Pass@1 {100 * d1[0]:+.2f} [{100 * d1[1]:+.2f}, {100 * d1[2]:+.2f}]  "
            f"Pass@k {100 * dk[0]:+.2f} [{100 * dk[1]:+.2f}, {100 * dk[2]:+.2f}]"
        )
    for dom in sorted(set(domain.values())):
        ids = [t for t in task_ids if domain[t] == dom]
        cells = "  ".join(f"{name} {100 * st.mean(pass1(name, t) for t in ids):5.1f}" for name in names)
        print(f"  {dom:18s} n={len(ids):3d}  {cells}")


if __name__ == "__main__":
    main()
