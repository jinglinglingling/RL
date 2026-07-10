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
