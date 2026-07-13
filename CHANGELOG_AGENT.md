# Agent Change Log (OSWorld + NeMo-RL)

This file is used as a persistent recovery log so changes can be traced even if a run fails.

## Logging Rules (from now on)
- Record every code change with why it was made.
- Record every smoke/Slurm job ID with result and root failure.
- Keep the latest "next fix target" so work can resume quickly.

## 2026-07-10 Progress Snapshot

### Code Changes
- `third_party/OSWorld/desktop_env/evaluators/metrics/__init__.py`
  - Switched to lazy import for metrics to avoid hard dependency crashes during `DesktopEnv` bootstrap.
- `third_party/OSWorld/desktop_env/evaluators/getters/__init__.py`
  - Switched to lazy import for getters for the same startup-stability reason.
- `third_party/OSWorld/desktop_env/controllers/setup.py`
  - Added local fallback `compare_urls()` implementation to avoid importing full metrics utils path during setup.
- `third_party/OSWorld/desktop_env/providers/apptainer/provider.py`
  - Added writable nginx runtime path mounts (`/var/log/nginx`, `/var/cache/nginx`, `/var/lib/nginx`) for fakeroot mode.
  - Added automatic VLC port remap when configured VLC host port is occupied.
  - Fixed syntax break: completed `list_snapshots()` implementation and imported `List`.
  - Added socket bind-based preflight port checks instead of only `psutil` checks.
  - Added generalized auto-remap for `server/chromium/vnc/vlc` ports.
  - Added hostfwd conflict log parsing and automatic launch retry with remapped ports.

### Recent Job Timeline
- `13662779` (`osw-online-smoke13-newimg-setupcompareurlfix`) -> FAILED
  - Reached rollout and VM image download, then failed on nginx permission (`/var/log/nginx/error.log`).
- `13663389` (`osw-online-smoke14-newimg-nginxpermfix`) -> FAILED
  - Blocked by local syntax error (`IndentationError`) in `provider.py` after edit.
- `13667732` (`osw-online-smoke15-newimg-nginxpermfix-syntaxfix`) -> FAILED
  - `nginx` permission issue gone.
  - New blocker: QEMU host forwarding conflict on port `8080` (`Could not set up host forwarding rule`).
- `13667993` (`osw-online-smoke16-newimg-vlcportremap`) -> FAILED
  - `8080` conflict gone after remap.
  - New blocker: host forwarding conflict moved to port `8006` (VNC).
- `13668535` (`osw-online-smoke17-newimg-portretry`) -> FAILED
  - Verified generalized remap/retry path is active:
    - `vlc_port=8080` remapped to `8083`
    - `vnc_port=8006` remapped to `8007`, then `8008`
  - Still failed with QEMU hostfwd conflict on remapped VNC port (`8008`).

### Current Root Cause
- Apptainer/QEMU hostfwd ports can still race/conflict at VM launch under shared nodes.
- Broader auto-remap/retry logic is implemented, but collision can still persist across retries.

### Next Fix Target
- Move from deterministic incremental remap to randomized/high-range free-port allocation for all forwarded ports.
- If needed, add launch-time temporary socket reservations to reduce race window before QEMU starts.

## 2026-07-10 Continued Debug Log

### Code Changes
- `Nemo-RL-main-1/RL/nemo_rl/distributed/virtual_cluster.py`
  - Fixed `_get_node_ip_and_free_port()` to honor the requested `[port_range_low, port_range_high)` again.
  - Added compatibility fallback for older helper signatures (`TypeError` -> retry without args).
- `scripts/run_nemorl_osworld_online_grpo.sh`
  - Added automatic per-job `cluster.master_port_range_low/high` selection from `SLURM_JOB_ID` to reduce cross-job TCPStore collisions.
- `scripts/run_nemorl_osworld_grpo.sh`
  - Added the same per-job master-port-range selection for offline/converted-data training.
- `third_party/OSWorld/desktop_env/providers/apptainer/provider.py`
  - Added `ssh_forward_port` / `rdp_forward_port` as first-class remap-managed ports.
  - Extended preflight conflict checks and remap pool to include SSH/RDP hostfwd ports.
  - Extended log-based hostfwd conflict parser to remap SSH/RDP conflicts as well.
  - Updated launch logging to print ssh/rdp forward ports for easier diagnosis.
- `third_party/OSWorld/desktop_env/evaluators/getters/__init__.py`
  - Added fallback `get_default_search_engine()` so getter lookup works even if heavy chrome getter module imports fail.

### Recent Job Timeline
- `13674909` (`smoke21`) -> FAILED
  - Main blocker at that point: `torch.distributed` `EADDRINUSE` on `MASTER_PORT`.
- `13676176` (`osw-online-smoke22-masterport-range`) -> FAILED
  - `MASTER_PORT` collision issue resolved (job progressed into rollout/training setup).
  - New blocker: Apptainer hostfwd conflict on RDP port `10006`.
- `13676792` (`osw-online-smoke23-rdp-remap`) -> FAILED
  - RDP hostfwd conflict path no longer the blocker.
  - New blocker: getter lazy-load gap (`AttributeError: get_default_search_engine`).
- `13677782` (`osw-online-smoke24-getter-fallback`) -> COMPLETED
  - Ran through online NeMo-Gym GRPO smoke, completed all `max_steps=5`, and saved checkpoint.
  - No recurrence of:
    - `torch.distributed` `EADDRINUSE`,
    - QEMU hostfwd `10006` collision,
    - missing `get_default_search_engine`.

### Current Root Cause
- Previous blockers (distributed master-port conflict, RDP hostfwd conflict, missing getter fallback) are addressed for smoke scope.

### Next Fix Target
- Run a multi-task / longer online smoke to validate stability beyond single-sample (`MAX_SAMPLES=1`) path.
- Keep monitoring for any remaining optional getter/metric import gaps under other OSWorld task types.

## 2026-07-10 Multi-task Follow-up

### Job Results
- `13680110` (`osw-online-smoke26-multitask3-real`) -> FAILED
  - Reached `Step 2/2` (multi-task path progressed beyond first task).
  - New blocker at `seed_session`: `AttributeError: module 'desktop_env.evaluators.metrics' has no attribute 'is_expected_tabs'`.
  - Related warning showed heavy metric modules still failing to import in minimal env (`rapidfuzz`, `lxml`, etc.), so fallback coverage had to be extended.

### Code Changes
- `third_party/OSWorld/desktop_env/evaluators/metrics/__init__.py`
  - Added lightweight URL normalization helpers for fallback checks.
  - Added fallback `is_expected_tabs` (for chrome open tab URL checks).
  - Added fallback `is_cookie_deleted` (for cookie domain absence checks).

### Next Fix Target
- Re-run the same 3-task smoke set with the new metric fallbacks and verify completion + checkpoint save.

## 2026-07-10 Multi-task Follow-up (2)

### Job Results
- `13681887` (`osw-online-smoke27-multitask3-metricsfix`) -> FAILED
  - Previous `is_expected_tabs` metric error is gone.
  - New blocker at `seed_session`: `AttributeError: module 'desktop_env.evaluators.getters' has no attribute 'get_open_tabs_info'`.
  - This comes from lazy getter imports in minimal env where heavy chrome getter dependencies are unavailable.

### Code Changes
- `third_party/OSWorld/desktop_env/evaluators/getters/__init__.py`
  - Added fallback `get_open_tabs_info` via Chrome DevTools `/json/list` endpoint (no playwright dependency).
  - Added fallback `get_cookie_data` with best-effort sqlite extraction from Chrome Cookies DB.
  - Added helper utilities for fallback path resolution and command output normalization.

### Next Fix Target
- Re-run 3-task smoke again and verify it can pass the open-tabs/cookie tasks without getter AttributeError.

## 2026-07-12 Multi-task Follow-up (3)

### Job Results
- `13683376` (`osw-online-smoke28-multitask3-getterfix`) -> FAILED
  - Progress reached `Epoch 1 / Step 2`.
  - New blocker in setup phase for the tab task:
    - `Setup step 3 failed: _chrome_open_tabs_setup`
    - Root cause: `playwright sync API inside asyncio loop` in NeMo-Gym resources server context.

### Code Changes
- `third_party/OSWorld/desktop_env/controllers/setup.py`
  - Added DevTools HTTP helper methods:
    - `_chrome_list_tabs`
    - `_chrome_open_tab_via_devtools`
    - `_chrome_close_tab_via_devtools`
  - Reworked `_chrome_open_tabs_setup` to use DevTools endpoints (`/json/list`, `/json/new`) instead of `sync_playwright`.
  - Reworked `_chrome_close_tabs_setup` to use DevTools endpoints (`/json/list`, `/json/close/<id>`) instead of `sync_playwright`.
  - Added best-effort blank-tab cleanup to reduce evaluator noise.

### Next Fix Target
- Re-run the same 3-task smoke set and verify the tab setup stage no longer fails inside async server context.

## 2026-07-12 Multi-task Follow-up (4)

### Job Results
- `13827589` (`osw-online-smoke29-multitask3-devtools-tabs`) -> COMPLETED
  - Multi-task online run progressed through:
    - `Epoch 1 / Step 2`
    - `Epoch 2 / Step 2`
    - `Epoch 3` with checkpoint save trigger
  - Confirmed checkpoint save:
    - `Saving checkpoint for step 5...`
    - `Saved checkpoint to .../osw-online-smoke29-multitask3-devtools-tabs/checkpoints/tmp_step_5/policy/weights`
  - Training ended normally:
    - `Max number of steps has been reached, stopping training early`

### Validation Notes
- No recurrence of prior blocker:
  - `playwright sync API inside the asyncio loop` during `_chrome_open_tabs_setup`
- Confirms the DevTools-based tab setup path is viable for this 3-task smoke set.

### Next Fix Target
- Run a longer multi-task online smoke (higher `MAX_STEPS` / more task IDs) to stress stability beyond this checkpointed short run.

## 2026-07-12 Multi-task Follow-up (5)

### Job Results
- `13828541` (`osw-online-smoke30-multitask3-step8`) -> CANCELLED (agent)
  - Submitted with `MAX_STEPS=8`, but profile override still pinned training to 5 steps.
  - Replaced by corrected submission using `STABLE_MAX_STEPS=8`.
- `13828556` (`osw-online-smoke31-multitask3-step8-real`) -> RUNNING
  - Verified overrides now reflect real longer run:
    - `grpo.max_num_epochs=8`
    - `grpo.max_num_steps=8`
    - `checkpointing.save_period=8`

### Next Fix Target
- Monitor `13828556` to completion and confirm long-run stability with checkpoint save at step 8.

## 2026-07-12 Overnight Run Plan

### Submitted Jobs
- `13828556` (`osw-online-smoke31-multitask3-step8-real`) -> RUNNING
  - Current active run for medium-length stability check (`8 steps`).
- `13829681` (`osw-online-long32-step20`) -> PENDING (dependency)
  - Chained with `afterok:13828556` to avoid resource contention.
  - Confirmed long-run overrides:
    - `grpo.max_num_epochs=20`
    - `grpo.max_num_steps=20`
    - `checkpointing.save_period=20`

### Next Fix Target
- After wake-up: check `13828556` final state, then inspect `13829681` progress/failure root cause (if any) and checkpoint path.

## 2026-07-13 Morning Check

### Job Results
- `13828556` (`osw-online-smoke31-multitask3-step8-real`) -> COMPLETED
  - Reached configured training limit (`8` steps) and ended normally.
  - Checkpoint confirmed:
    - `Saving checkpoint for step 8...`
    - `Saved checkpoint to .../osw-online-smoke31-multitask3-step8-real/checkpoints/tmp_step_8/policy/weights`
  - End marker:
    - `Max number of steps has been reached, stopping training early`
- `13829681` (`osw-online-long32-step20`) -> COMPLETED
  - Long run reached configured training limit (`20` steps) and ended normally.
  - Checkpoint confirmed:
    - `Saving checkpoint for step 20...`
    - `Saved checkpoint to .../osw-online-long32-step20/checkpoints/tmp_step_20/policy/weights`
  - End marker:
    - `Max number of steps has been reached, stopping training early`

### Validation Notes
- No `Traceback` and no `POST /seed_session ... 500` detected in the two driver logs.
- The previously fixed async-tab-setup path remains stable under longer runtime.

### Next Fix Target
- Optional: expand task set beyond the current 3-task smoke list for broader cross-task stability.

## 2026-07-13 Training Signal Check + Eval-enabled Run

### 20-step Run Signal Check (`13829681`)
- Parsed 20 `Training Results` entries from driver log.
- Observed ranges:
  - `loss`: `0.0000` to `0.0002`
  - `generation KL`: `0.0067` to `0.2862`
  - `avg reward`: always `0.0000`
  - `mean generation length`: `6` to `24`
- Interpretation:
  - Loop is actively training/updating (KL/length are changing).
  - But effective task reward signal is currently missing on this 3-task smoke mix (all-zero reward).

### New Run Submitted (train + periodic validation)
- `13859796` (`osw-online-long33-step30-val`) -> RUNNING
- Changes vs prior long run:
  - `grpo.max_num_steps=30`, `grpo.max_num_epochs=30`
  - Per-episode action budget in prepared rows increased (`MAX_STEPS=12`)
  - Enabled validation metrics:
    - `grpo.val_at_start=true`
    - `grpo.val_period=5`
- Validation dataset is loaded and active (`1` sample in current smoke split).

### Next Fix Target
- Monitor `13859796` for:
  - periodic `val/*` metrics,
  - reward trend movement away from all-zero,
  - checkpoint save at step 30.

## 2026-07-13 SWE Two-stage Doc Update

### Doc Changes
- Updated `README.md` with a new section:
  - `Reference: Two-stage SWE RL (SWE1 pivot + SWE2 agentic)`.
- Added explicit mapping to current Gym components:
  - SWE1 -> `resources_servers/swe_pivot` + `swe_pivot_tool_simulation_agent` (single-step pivot verification).
  - SWE2 -> `responses_api_agents/swe_agents` (OpenHands-based full-horizon agentic SWE RL).
- Added explicit note on `mini-swe-agentic` relationship:
  - Current SWE1+SWE2 path is not based on `mini_swe_agent` as the core; `mini_swe_agent` is a separate integration path.
- Added OSWorld vs SWE comparison notes to clarify:
  - modality/action-space differences,
  - reward-shape differences (pivot local reward vs OSWorld sparse environment reward),
  - both using NeMo-Gym multi-turn rollout infrastructure.

## 2026-07-13 Qwen3-VL + 50-step Follow-up

### Model Cache / Runtime Prep
- Confirmed official HF model id for requested 8B VLM path:
  - `Qwen/Qwen3-VL-8B-Instruct` (no official `Qwen3.5-VL-8B` repo found under `Qwen/`).
- Model snapshot available locally and complete:
  - `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/linglinj/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  - Blob size check: `17G`.

### Code Changes
- `scripts/slurm/submit_nemorl_osworld_train.sh`
  - Added stable-profile knobs:
    - `STABLE_NUM_PROMPTS_PER_STEP` (default `1`)
    - `STABLE_NUM_GENERATIONS_PER_PROMPT` (default `4`)
  - Wired profile overrides to these env vars:
    - `grpo.num_prompts_per_step=${STABLE_NUM_PROMPTS_PER_STEP}`
    - `grpo.num_generations_per_prompt=${STABLE_NUM_GENERATIONS_PER_PROMPT}`
  - Generalized fatal hint for missing snapshot path to use `STABLE_MODEL_REPO` dynamically (instead of hardcoded `Qwen2.5-Omni-3B`).

### Job Status / Error Check (50-step jobs)
- `13862243` (`osw-online-wandb-v021-step50`) -> FAILED
  - `Ray` mismatch in driver log (`cluster 2.54.0` vs process `2.55.1`).
  - Also hit Megatron config/ckpt compatibility failure for `Qwen2.5-Omni-3B` provider (`Unexpected config keys ...`).
- `13862717` (`osw-online-wandb-v021-step50-fix1`) -> FAILED
  - Still shows repeated `Ray` version mismatch (`2.54.0` vs `2.55.1`).
  - `NemoGym` side eventually reports `Process osworld_vlm finished unexpectedly` during retries.
- Current active run at check time:
  - `13859796` (`osw-online-long33-step30-val`) -> RUNNING.
  - No fatal traceback observed in current pass; reached `validation at step 5`.

### Attempted Resubmission
- Tried launching a new 50-step run with:
  - `Qwen3-VL-8B-Instruct`, `num_generations_per_prompt=4`, `wandb project=vlm_osworld`.
- First submit rejected by Slurm (`Requested time limit is invalid`) because `batch` partition max is `4:00:00`.
- Follow-up submit path requires explicit user approval in current agent policy mode before execution.

## 2026-07-13 Smoke Set Expansion (Sparsity Mitigation)

### Why
- The previous smoke path effectively used `1` or `3` tasks (depending on override), which makes online GRPO signal overly sparse.

### Code / Config Changes
- Added a larger smoke config:
  - `configs/osworld_test_all_smoke20.json`
  - Contains `20` Chrome task IDs (includes the previously stable 3-task subset).
- Updated default online pipeline smoke meta path:
  - `scripts/e2e_vlm_rl_pipeline_online.sh`
  - Default `TEST_ALL_META_PATH` now points to `configs/osworld_test_all_smoke20.json`.
- Updated direct data-prep default:
  - `scripts/prepare_osworld_nemogym_data.py`
  - `--test-all-meta-path` default now points to `configs/osworld_test_all_smoke20.json`.

### Verification
- Re-ran dataset preparation with defaults.
- Result:
  - `total=20`, `train=18`, `val=2`.

## 2026-07-13 CUA-style Reward Gate Alignment (No-core-code Path)

### Static Audit on Current `smoke20`
- Audited evaluator definitions for all 20 tasks in:
  - `configs/osworld_test_all_smoke20.json`
- Findings:
  - `6/20` tasks include evaluator `postconfig` (restart/read-after-restart semantics).
  - `3/20` tasks use known brittle getter/metric paths under minimal env fallback:
    - `default_search_engine`
    - `cookie_data` / `is_cookie_deleted`
    - `open_tabs_info` / `is_expected_tabs`

### New Gate-pass Task Pool
- Added `configs/osworld_test_all_smoke20_gatepass.json`.
- Selection rule (static gate):
  - Exclude tasks with evaluator `postconfig`.
  - Exclude tasks touching above brittle getter/metric paths.
- Output size:
  - `13` Chrome tasks retained from the original `20`.

## 2026-07-13 Smoke20 Runtime Check + W&B Override Fix

### Job Results
- `13864271` (`osw-online-qwen3vl8b-step50-gen4-smoke20`) -> FAILED
  - Failed very early during Hydra override parsing (before rollout).
  - Root cause:
    - Invalid override key path `logger.wandb_project` / `logger.wandb_name`.
    - This config uses nested `logger.wandb.project` and `logger.wandb.name`.
  - Error signature:
    - `ConfigCompositionException: Could not override 'logger.wandb_project'`
    - `omegaconf.errors.ConfigAttributeError: Key 'wandb_project' is not in struct`

### Follow-up
- Resubmitted corrected job:
  - `13864641` (`osw-online-qwen3vl8b-step50-gen4-smoke20-fixwandb`) -> RUNNING
  - Changed override path to:
    - `+logger.wandb.project=vlm_osworld`
    - `+logger.wandb.name=osw-online-qwen3vl8b-step50-gen4-smoke20`
  - Other runtime settings unchanged (`Qwen3-VL-8B-Instruct`, `num_generations_per_prompt=4`, `MAX_SAMPLES=20`, `MAX_STEPS=50`).

### Status Update
- `13864641` -> FAILED (early Hydra parse failure)
  - Root cause:
    - The `+` append prefix is invalid for keys that already exist.
    - `logger.wandb.project` already exists in config; should override directly without `+`.
  - Error signature:
    - `ConfigCompositionException: Could not append to config. An item is already at 'logger.wandb.project'`
    - Suggested fix in traceback: use `logger.wandb.project=...` (or `++...` if force-append semantics are required).

### Next Submission
- Submitted new smoke20 run with corrected W&B overrides and explicit rollout-related knobs:
  - `13865596` (`osw-online-qwen3vl8b-step50-gen4-smoke20-fix2`) -> RUNNING
  - Key overrides pinned:
    - `logger.wandb_enabled=true`
    - `logger.wandb.project=vlm_osworld`
    - `logger.wandb.name=osw-online-qwen3vl8b-step50-gen4-smoke20-fix2`
    - `grpo.num_generations_per_prompt=4`
    - `grpo.num_prompts_per_step=1`
    - `grpo.use_leave_one_out_baseline=true`
    - `env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan=1`
    - data prep max episode steps kept at `MAX_STEPS=50` with `MAX_SAMPLES=20`.

### Status Update
- `13865596` -> FAILED
  - Primary blocker remains Ray version mismatch:
    - cluster daemon side: `Ray 2.54.0`
    - runtime-env workers: `Ray 2.55.1`
  - Secondary cascade:
    - `NemoGym` health-check retries
    - `RuntimeError: Process osworld_vlm finished unexpectedly!`
  - No training loop metrics were emitted before failure (`Training Results` / `Avg Reward` absent).

## 2026-07-13 Ray Version Alignment Hardening

### Problem
- Recurrent warning/error pattern in driver logs:
  - Ray cluster daemon starts as `2.54.0`
  - runtime-env worker venvs get pinned to `NRL_WORKER_RAY_VERSION` (often `2.55.1`)
  - produces `Version mismatch: cluster 2.54.0 vs process 2.55.1`.

### Code Change
- Updated `scripts/slurm/submit_nemorl_osworld_train.sh`:
  - `SETUP_COMMAND` now automatically appends:
    - `/opt/nemo_rl_venv/bin/pip install --quiet --no-input --no-deps --force-reinstall ray==${NRL_WORKER_RAY_VERSION}`
  - This runs before `ray start` on every node via `ray.sub`, aligning the Ray daemon version with worker venv pinning.
  - Guardrail: if user-provided setup command already contains `ray==`, do not append duplicate pin command.
