#!/usr/bin/env python3
"""Train an EVT-conditioned action diffusion model."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffusion.src.train import train_action_diffusion
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diffusion_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config override. If omitted, edit DEFAULT_CONFIG_PATH in this script.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH
    train_action_diffusion(load_yaml(cfg_path), config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
