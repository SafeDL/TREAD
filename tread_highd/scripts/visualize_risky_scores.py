#!/usr/bin/env python
"""
visualize_highd_events.py — 可视化事件
==========================================
用法:
  conda activate jzm
  python scripts/visualize_highd_events.py --event_type cut_in --top_k 20 --sort_by risk_score
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Allow running either from the repository root or from tread_highd/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_highd.src.io_utils import load_config, resolve_data_path, ensure_dir


LOWER_IS_RISKIER = {"min_ttc", "min_thw"}


def _top_risky_events(df, sort_by, top_k):
    if sort_by in LOWER_IS_RISKIER:
        return df.nsmallest(top_k, sort_by)
    return df.nlargest(top_k, sort_by)


def main():
    parser = argparse.ArgumentParser(description="TREAD: Visualize highD long-tail events")
    default_config = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--event_type", default="following", choices=["cut_in", "following"])
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--sort_by", default="risk_score")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["processed_dir"], args.config)))

    events_path = out_dir / "events.csv"
    if not events_path.exists():
        print(f"events.csv not found at {events_path}. Run extract_highd_events.py first.")
        return

    df = pd.read_csv(events_path)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    from tread_highd.src.visualization import plot_risk_distribution, plot_ttc_drac_scatter

    # 风险分布
    plot_risk_distribution(df, args.event_type, str(fig_dir / f"risk_distribution_{args.event_type}.png"))
    plot_ttc_drac_scatter(df, str(fig_dir / "ttc_drac_scatter.png"))

    # Top-K
    sub = df[(df["event_type"] == args.event_type) & (df["is_valid"] == True)]
    if args.sort_by in sub.columns:
        top = _top_risky_events(sub, args.sort_by, args.top_k)
        print(f"\nTop {args.top_k} {args.event_type} events by {args.sort_by}:")
        cols = ["event_id", "recording_id", "ego_id", "target_id",
                "min_ttc", "min_thw", "max_drac",
                "ttc_severity", "thw_severity", "drac_severity", "risk_score"]
        cols = [c for c in cols if c in top.columns]
        print(top[cols].to_string())
    else:
        print(f"Column not found: {args.sort_by}")

    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()
