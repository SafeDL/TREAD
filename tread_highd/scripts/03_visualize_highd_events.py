#!/usr/bin/env python
"""
03_visualize_highd_events.py — 可视化事件
==========================================
用法:
  conda activate jzm
  python scripts/03_visualize_highd_events.py --config configs/highd_default.yaml \
      --event_type cut_in --top_k 20 --sort_by risk_score
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tread_highd.io_utils import load_config, resolve_data_path, ensure_dir
from tread_highd.visualization import plot_risk_distribution, plot_ttc_drac_scatter


def main():
    parser = argparse.ArgumentParser(description="TREAD: Visualize highD events")
    parser.add_argument("--config", required=True)
    parser.add_argument("--event_type", default="cut_in", choices=["cut_in", "following"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--sort_by", default="risk_score")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["processed_dir"], args.config)))

    events_path = out_dir / "events.csv"
    if not events_path.exists():
        print(f"events.csv not found at {events_path}. Run 02_build first.")
        return

    df = pd.read_csv(events_path)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    # 风险分布
    plot_risk_distribution(df, args.event_type, str(fig_dir / f"risk_distribution_{args.event_type}.png"))
    plot_ttc_drac_scatter(df, str(fig_dir / "ttc_drac_scatter.png"))

    # Top-K
    sub = df[(df["event_type"] == args.event_type) & (df["is_valid"] == True)]
    if args.sort_by in sub.columns:
        top = sub.nlargest(args.top_k, args.sort_by)
        print(f"\nTop {args.top_k} {args.event_type} events by {args.sort_by}:")
        print(top[["event_id", "recording_id", "ego_id", "target_id",
                    "min_ttc", "max_drac", "risk_score"]].to_string())

    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
