#!/usr/bin/env python3
"""Train the adversaray Stage 2 naturalness discriminator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.discriminator_train import train_discriminator
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "discriminator_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild dataset before training.")
    parser.add_argument("--epochs", type=int, default=0, help="Optional epoch override for smoke tests.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    if args.rebuild_dataset:
        cfg.setdefault("data", {})["rebuild"] = True
    if args.epochs > 0:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    train_discriminator(cfg, config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()

