#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_export_tail_conditions.py — 导出 tail_conditions.csv 供 diffusion 阶段使用
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_highd.src.io_utils import load_config, resolve_data_path  # noqa: E402
from tread_deepevt.src.inference import export_tail_conditions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tail_conditions.csv")
    default_cfg = Path(__file__).resolve().parent / "configs" / "deepevt_following.yaml"
    parser.add_argument("--config", default=str(default_cfg))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(resolve_data_path(cfg["paths"]["output_dir"], args.config))
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / "model.pt"
    tail_levels = cfg.get("training", {}).get("eval_tail_levels", [0.90, 0.95, 0.99])

    export_tail_conditions(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        tail_levels=tuple(float(x) for x in tail_levels),
        include_context_features=True,
    )


if __name__ == "__main__":
    main()
