#!/usr/bin/env python3
"""Train the Stage 2 naturalness discriminator."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffusion.src.discriminator.train import train_discriminator
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "discriminator_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild discriminator_dataset.npz before training.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    config = load_yaml(cfg_path)
    if args.rebuild_dataset:
        config.setdefault("data", {})["rebuild"] = True
    train_discriminator(config, config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
