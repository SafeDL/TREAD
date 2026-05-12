#!/usr/bin/env python
"""
visualize_risky_scores.py — 可视化事件风险分布
==========================================
用法:
  conda activate jzm
  python tread_highd/scripts/visualize_risky_scores.py --event_type all
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow running either from the repository root or from tread_highd/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_highd.src.io_utils import load_config, resolve_data_path, ensure_dir


RISK_COLUMNS = [
    ("ttc_severity", "1/min_ttc"),
    ("thw_severity", "1/min_thw"),
    ("drac_severity", "max_drac"),
    ("risk_score", "risk_score"),
]

TAIL_QUANTILES = (0.90, 0.95)


def _with_danger_columns(df, eps=1e-6):
    """Keep all summary columns danger-oriented: larger means riskier."""
    df = df.copy()
    if "min_ttc" in df.columns:
        df["ttc_severity"] = 1.0 / (df["min_ttc"].astype(float) + eps)
    if "min_thw" in df.columns:
        df["thw_severity"] = 1.0 / (df["min_thw"].astype(float) + eps)
    if "max_drac" in df.columns:
        df["drac_severity"] = df["max_drac"].astype(float)
    return df


def _power_law_tail_fit(values, tail_quantile=0.90):
    """Fit log empirical survival on the upper tail as a rough diagnostic."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return {}

    threshold = float(np.quantile(vals, tail_quantile))
    sorted_vals = np.sort(vals)
    n = len(sorted_vals)
    survival = 1.0 - np.arange(1, n + 1) / (n + 1)
    mask = (sorted_vals >= threshold) & (survival > 0)
    x = sorted_vals[mask]
    y = survival[mask]
    if len(np.unique(x)) < 10:
        return {"tail_threshold": threshold, "tail_n": int(mask.sum())}

    # Keep one survival value per x to avoid duplicate-heavy plateaus dominating.
    ux, first_idx = np.unique(x, return_index=True)
    uy = y[first_idx]
    lx = np.log(ux)
    ly = np.log(uy)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "tail_threshold": threshold,
        "tail_n": int(mask.sum()),
        "tail_unique_n": int(len(ux)),
        "tail_loglog_slope": float(slope),
        "tail_survival_alpha": float(-slope),
        "tail_loglog_r2": float(r2),
    }


def _summarize_dataset(events_df, event_types, tail_quantile):
    df = _with_danger_columns(events_df)
    rows = []
    for event_type in event_types:
        sub = df[(df["event_type"] == event_type) & (df["is_valid"] == True)]
        for col, definition in RISK_COLUMNS:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = vals.dropna()
            if len(finite) == 0:
                continue
            quantiles = finite.quantile([0.50, *TAIL_QUANTILES])
            fit = _power_law_tail_fit(finite, tail_quantile)
            q95 = float(quantiles.loc[0.95])
            vmax = float(finite.max())
            rows.append({
                "event_type": event_type,
                "metric": col,
                "definition": definition,
                "n": int(len(finite)),
                "non_positive": int((finite <= 0).sum()),
                "p50": float(quantiles.loc[0.50]),
                "p90": float(quantiles.loc[0.90]),
                "p95": q95,
                "max": vmax,
                "max_over_p95": vmax / q95 if q95 > 0 else np.nan,
                **fit,
            })
    return pd.DataFrame(rows)


def _print_summary(summary_df):
    if summary_df.empty:
        print("No valid risk values found.")
        return
    cols = [
        "event_type", "metric", "definition", "n", "non_positive",
        "p50", "p90", "p95", "max", "max_over_p95",
        "tail_survival_alpha", "tail_loglog_r2",
    ]
    shown = summary_df[[c for c in cols if c in summary_df.columns]].copy()
    for col in shown.select_dtypes(include=[float]).columns:
        shown[col] = shown[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    print("\nDataset-level risk diagnostics (valid events only):")
    print(shown.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="TREAD: Visualize highD long-tail events")
    default_config = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--event_type", default="cut_in", choices=["all", "cut_in", "following"])
    parser.add_argument("--tail_quantile", type=float, default=0.85,
                        help="Lower cutoff used for rough log-log tail diagnostics.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["output_dir"], args.config)))

    events_path = out_dir / "events.csv"
    if not events_path.exists():
        print(f"events.csv not found at {events_path}. Run extract_highd_events.py first.")
        return

    df = pd.read_csv(events_path)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    from tread_highd.src.visualization import (
        plot_risk_distribution,
        plot_survival_curve,
        plot_ttc_drac_scatter,
    )

    event_types = ["following", "cut_in"] if args.event_type == "all" else [args.event_type]
    for event_type in event_types:
        # 风险分布（positive values on log-x, counts on log-y）
        plot_risk_distribution(df, event_type, str(fig_dir / f"risk_distribution_{event_type}.png"))
        # 尾部 survival（1-CDF, log-log）
        plot_survival_curve(df, event_type, str(fig_dir / f"survival_curve_{event_type}.png"))
    plot_ttc_drac_scatter(df, str(fig_dir / "ttc_drac_scatter.png"))

    summary = _summarize_dataset(df, event_types, args.tail_quantile)
    summary_path = fig_dir / "risk_tail_diagnostics.csv"
    summary.to_csv(summary_path, index=False)
    _print_summary(summary)

    print(f"\nFigures saved to {fig_dir}")
    print(f"Diagnostics saved to {summary_path}")


if __name__ == "__main__":
    main()
