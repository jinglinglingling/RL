# OSWorld night watchdog

This watchdog runs on a Slurm CPU node, so it does not depend on a laptop,
Cursor IDE session, or SSH connection remaining awake.

## One-time Cursor credential

Create a user or service-account API key from Cursor Dashboard → Integrations.
Store it on the server without pasting it into chat or a repository:

```bash
install -d -m 700 ~/.config/cursor
umask 077
read -rsp "Cursor API key: " CURSOR_API_KEY
printf '%s' "${CURSOR_API_KEY}" > ~/.config/cursor/osworld-night-operator.key
unset CURSOR_API_KEY
chmod 600 ~/.config/cursor/osworld-night-operator.key
```

The key is read only by the SDK repair subprocess. It is not copied into the
watchdog config, command line, audit log, or Slurm job name.

## Submit

```bash
export OSWORLD_CELL2_ENV_FILE=/absolute/path/to/private/cell2-opensandbox.env
bash examples/nemo_gym/slurm/submit_osworld_night_watchdog.sh
```

The submit helper creates an isolated, pinned Cursor SDK venv when necessary.
The default job uses the `cpu_long` partition for seven days. Submitting again
while the same watchdog job is queued or running is a no-op.

## Observe

The watchdog writes:

- `results/osworld-night-watchdog/status.json`: current checkpoint and job state
- `results/osworld-night-watchdog/audit.jsonl`: append-only action history
- `results/osworld-night-watchdog/watchdog-<job-id>.out`: process output
- `results/osworld-night-watchdog/agent-results/`: private SDK run summaries

The default config watches the tiny T15 and T30 runs every five minutes.
At step 20 it reads fresh W&B validation metrics and extends a run to step 40
only when accuracy is at least 0.4, natural termination is at least 0.95, and
truncation is at most 0.1.

## Safety behavior

- A non-blocking file lock permits only one watchdog process.
- Only configured job-name prefixes and submit scripts are controlled.
- Known transient failures are retried with exponential backoff and an hourly
  submission cap.
- Code or unknown failures require one Cursor repair run per unique signature.
- The repair agent cannot submit/cancel Slurm jobs and is instructed to edit
  only configured paths.
- The watchdog verifies that Git HEAD did not change, checks changed paths
  against the allowlist, and runs configured validations before resubmission.
- If the API key is absent, unknown failures stop at `needs_agent`; the watchdog
  does not blindly consume more GPUs.
