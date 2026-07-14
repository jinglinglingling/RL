#!/usr/bin/env python3
"""Launch NeMo-RL online GRPO (NeMo-Gym) for OSWorld."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_nemorl_root = project_root.parent / "Nemo-RL-main-1" / "RL"
    default_config = project_root / "configs" / "nemorl_osworld_online_grpo_qwen_vl.yaml"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nemo-rl-root",
        type=Path,
        default=default_nemorl_root,
        help="Path to NeMo-RL repo root.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to NeMo-RL online GRPO config yaml.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def configure_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    token_file = os.environ.get("HF_TOKEN_FILE")
    if not token and token_file:
        token_path = Path(token_file)
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def main() -> None:
    args, passthrough = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_dir = project_root / "src"

    nemo_rl_root = args.nemo_rl_root.resolve()
    config_path = args.config.resolve()
    examples_dir = nemo_rl_root / "examples"
    nemo_gym_examples_dir = examples_dir / "nemo_gym"
    if not nemo_rl_root.exists():
        raise FileNotFoundError(f"NeMo-RL root not found: {nemo_rl_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not (nemo_gym_examples_dir / "run_grpo_nemo_gym.py").exists():
        raise FileNotFoundError(
            "NeMo-Gym GRPO entrypoint not found: "
            f"{nemo_gym_examples_dir / 'run_grpo_nemo_gym.py'}"
        )

    # Make sure local modules and NeMo-RL modules are importable.
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(nemo_rl_root))
    sys.path.insert(0, str(examples_dir))
    sys.path.insert(0, str(nemo_gym_examples_dir))
    configure_hf_token()

    try:
        from run_grpo_nemo_gym import main as run_grpo_nemo_gym_main
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import NeMo-RL online GRPO entrypoint dependencies. "
            f"Missing module: {exc.name!r}. "
            "Run through the Slurm launcher (submit_nemorl_osworld_online_train.sh) "
            "or activate a NeMo-RL environment with full dependencies."
        ) from exc

    sys.argv = [
        "run_grpo_nemo_gym.py",
        "--config",
        str(config_path),
        *passthrough,
    ]
    run_grpo_nemo_gym_main()


if __name__ == "__main__":
    main()
