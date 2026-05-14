#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_build_deepevt_dataset.py — 构建 DeepEVT 数据集 (dataset.npz 等)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_highd.src.io_utils import load_config, resolve_data_path  # noqa: E402
from tread_deepevt.src.data import build_and_save_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DeepEVT dataset")
    default_cfg = Path(__file__).resolve().parent / "configs" / "deepevt_following.yaml"
    parser.add_argument("--config", default=str(default_cfg))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    event_type = cfg.get("event", {}).get("event_type")
    if event_type not in {"following", "cut_in"}:
        raise ValueError(f"config event.event_type must be following/cut_in, got {event_type}")

    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"], args.config)
    events_csv = resolve_data_path(cfg["paths"]["events_csv"], args.config)
    output_dir = resolve_data_path(cfg["paths"]["output_dir"], args.config)

    build_and_save_dataset(
        events_csv=events_csv,
        raw_dir=raw_dir,
        config=cfg,
        output_dir=output_dir,
        event_type=event_type,
    )


if __name__ == "__main__":
    main()
