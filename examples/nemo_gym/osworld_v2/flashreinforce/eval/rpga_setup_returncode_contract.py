"""Setup-step returncode contract for the RPGA OSWorld eval agent.

Several tasks of the 361-task eval set pass ``expected_returncodes`` / ``on_nonzero`` to ``execute`` setup and
evaluator steps; the pinned OSWorld ``SetupController._execute_setup`` rejects those keywords, so without this patch
those tasks fail with an infrastructure error. The semantics match NeMo Gym ``osworld_agent/client.py``.

The reported HSG eval applied this inside the agent's one-time setup patch hook, right before ``DesktopEnv.evaluate``
is wrapped:

    from desktop_env.controllers import setup as _setup
    from desktop_env.desktop_env import DesktopEnv
    patch_setup_returncode_contract(_setup)
    orig_evaluate = DesktopEnv.evaluate
    DesktopEnv.evaluate = lambda self, *a, **k: evaluate_score_zero_aware(orig_evaluate, self, *a, **k)
"""

import inspect
import json
import logging
import os
import time

log = logging.getLogger(__name__)


class EvaluatorScoreZero(BaseException):
    """An evaluator command with on_nonzero="score_zero" proved a zero score.

    BaseException so that SetupController.setup's ``except Exception`` cannot re-wrap it as an infrastructure error.
    """


def patch_setup_returncode_contract(setup_module):
    controller = setup_module.SetupController
    current = controller._execute_setup
    if getattr(current, "_returncode_contract", False):
        return
    try:
        if {"expected_returncodes", "on_nonzero"}.issubset(inspect.signature(current).parameters):
            return
    except (TypeError, ValueError):
        pass
    requests = setup_module.requests

    def execute_setup(
        self, command, stdout="", stderr="", shell=False, until=None, expected_returncodes=None, on_nonzero=None
    ):
        if expected_returncodes is None and on_nonzero is None:
            return current(self, command, stdout=stdout, stderr=stderr, shell=shell, until=until)
        if not command:
            raise RuntimeError("Empty setup command")
        if expected_returncodes is None:
            expected_returncodes = [int(until["returncode"])] if until and "returncode" in until else [0]
        elif isinstance(expected_returncodes, int):
            expected_returncodes = [expected_returncodes]
        allowed = {int(code) for code in expected_returncodes}
        if not allowed:
            raise ValueError("expected_returncodes must not be empty")
        if on_nonzero not in {None, "score_zero"}:
            raise ValueError(f"unsupported on_nonzero policy: {on_nonzero!r}")
        replacements = {
            "{CLIENT_PASSWORD}": self.client_password,
            "{SCREEN_WIDTH_HALF}": str(self.screen_width // 2),
            "{SCREEN_HEIGHT_HALF}": str(self.screen_height // 2),
            "{SCREEN_WIDTH}": str(self.screen_width),
            "{SCREEN_HEIGHT}": str(self.screen_height),
        }
        rendered = [command] if isinstance(command, str) else list(command)
        for index, item in enumerate(rendered):
            for old, new in replacements.items():
                item = item.replace(old, new)
            rendered[index] = item
        rendered_command = rendered[0] if isinstance(command, str) else rendered
        payload = json.dumps({"command": rendered_command, "shell": shell})
        headers = {"Content-Type": "application/json"}
        until = until or {}
        failures = 0
        while True:
            result = None
            try:
                response = requests.post(
                    self.http_server + "/setup/execute", headers=headers, data=payload, timeout=130
                )
                if response.status_code == 200:
                    result = response.json()
                    if "returncode" not in result:
                        raise RuntimeError("setup response omitted returncode")
                    if stdout:
                        with open(os.path.join(self.cache_dir, stdout), "w", encoding="utf-8") as handle:
                            handle.write(result.get("output", ""))
                    if stderr:
                        with open(os.path.join(self.cache_dir, stderr), "w", encoding="utf-8") as handle:
                            handle.write(result.get("error", ""))
                else:
                    failures += 1
            except requests.exceptions.RequestException:
                failures += 1
            if failures >= 5:
                raise RuntimeError(f"setup command failed after five request attempts: {rendered_command!r}")
            if result is None:
                continue
            returncode = int(result["returncode"])
            command_text = " ".join(rendered_command) if isinstance(rendered_command, list) else rendered_command
            if returncode not in allowed:
                if on_nonzero == "score_zero" and returncode not in {126, 127}:
                    raise EvaluatorScoreZero(
                        f"evaluator command established score zero with return code {returncode}: {command_text}"
                    )
                raise RuntimeError(
                    f"setup command returned {returncode}; expected {sorted(allowed)}: {command_text}; "
                    f"stdout={result.get('output', '')!r}; stderr={result.get('error', '')!r}"
                )
            if (
                not until
                or ("returncode" in until and returncode == int(until["returncode"]))
                or ("stdout" in until and str(until["stdout"]) in result.get("output", ""))
                or ("stderr" in until and str(until["stderr"]) in result.get("error", ""))
            ):
                return result
            time.sleep(0.3)

    execute_setup._returncode_contract = True
    controller._execute_setup = execute_setup


def evaluate_score_zero_aware(orig_evaluate, env, *args, **kwargs):
    try:
        return orig_evaluate(env, *args, **kwargs)
    except EvaluatorScoreZero as exc:
        log.info("[osworld] evaluator setup established score zero: %s", exc)
        return 0.0
