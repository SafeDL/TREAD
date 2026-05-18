#!/usr/bin/env python3
"""Evaluate the Stage 2 naturalness discriminator."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffusion.src.discriminator.eval import evaluate_discriminator
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "discriminator_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--checkpoint", default="checkpoints/best_auc.pt", help="Checkpoint path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None, help="Evaluation split.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    evaluate_discriminator(load_yaml(cfg_path), config_dir=cfg_path.parent, checkpoint=args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
