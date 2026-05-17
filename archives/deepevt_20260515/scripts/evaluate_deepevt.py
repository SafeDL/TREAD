#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_evaluate_deepevt.py — 评估 direct DeepEVT quantiles
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from process_highd.src.io_utils import load_config, resolve_data_path  # noqa: E402
from tread_deepevt.src.evaluate import evaluate_deepevt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepEVT")
    default_cfg = Path(__file__).resolve().parent / "configs" / "deepevt_following.yaml"
    parser.add_argument("--config", default=str(default_cfg))
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint; default <output_dir>/best_model.pt")
    parser.add_argument("--report-name", default=None,
                        help="Output JSON filename under output_dir; default eval_report.json")
    parser.add_argument("--no-quantile-baseline", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    output_dir = Path(resolve_data_path(cfg["paths"]["output_dir"], args.config))
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pt"
    training_cfg = cfg.get("training", {})
    tail_levels = training_cfg.get(
        "eval_tail_levels",
        training_cfg.get("quantile_levels", [0.85, 0.90, 0.95]),
    )

    evaluate_deepevt(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        config=cfg,
        run_quantile_baseline=not args.no_quantile_baseline,
        tail_levels=tuple(float(x) for x in tail_levels),
        report_name=args.report_name,
    )


if __name__ == "__main__":
    main()
