#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_train_deepevt.py — 三阶段训练 DeepEVT
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_highd.src.io_utils import load_config, resolve_data_path  # noqa: E402
from tread_deepevt.src.train import train_deepevt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepEVT")
    default_cfg = Path(__file__).resolve().parent / "configs" / "deepevt_following.yaml"
    parser.add_argument("--config", default=str(default_cfg))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = resolve_data_path(cfg["paths"]["output_dir"], args.config)
    train_deepevt(output_dir=output_dir, config=cfg)


if __name__ == "__main__":
    main()
