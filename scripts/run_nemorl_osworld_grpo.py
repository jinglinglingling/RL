#!/usr/bin/env python3
"""Launch NeMo-RL VLM GRPO with OSWorld custom processor registered."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_nemorl_root = project_root.parent / "Nemo-RL-main-1" / "RL-merge-2689"
    default_config = project_root / "configs" / "nemorl_osworld_grpo_qwen_vl.yaml"

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
        help="Path to GRPO config yaml.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def configure_hf_token() -> None:
    """Populate HF auth env vars from env/token file before HF libs load."""

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
        print("[INFO] HF token detected and exported for this run.", file=sys.stderr)
    else:
        print(
            "[WARN] No HF token found in env/HF_TOKEN_FILE; downloads may be rate-limited.",
            file=sys.stderr,
        )


def main() -> None:
    args, passthrough = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_dir = project_root / "src"

    nemo_rl_root = args.nemo_rl_root.resolve()
    config_path = args.config.resolve()
    if not nemo_rl_root.exists():
        raise FileNotFoundError(f"NeMo-RL root not found: {nemo_rl_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Ensure our custom package and NeMo-RL are importable.
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(nemo_rl_root))
    sys.path.insert(0, str(nemo_rl_root / "examples"))
    configure_hf_token()

    from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
    from osworld_nemorl import osworld_vlm_data_processor

    processor_name = "osworld_vlm_data_processor"
    if processor_name not in PROCESSOR_REGISTRY:
        register_processor(processor_name, osworld_vlm_data_processor)

    # Reuse NeMo-RL's reference VLM GRPO script after registry injection.
    from run_vlm_grpo import main as run_vlm_grpo_main

    sys.argv = [
        "run_vlm_grpo.py",
        "--config",
        str(config_path),
        *passthrough,
    ]
    run_vlm_grpo_main()


if __name__ == "__main__":
    main()
