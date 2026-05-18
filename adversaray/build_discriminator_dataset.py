#!/usr/bin/env python3
"""Build the Stage 2 naturalness discriminator dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffusion.src.discriminator.data import build_discriminator_dataset
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "discriminator_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Optional smoke-test cap applied independently to train/val/test Stage 1 samples.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    build_discriminator_dataset(
        load_yaml(cfg_path),
        config_dir=cfg_path.parent,
        max_samples_per_split=int(args.max_samples_per_split),
    )


if __name__ == "__main__":
    main()
