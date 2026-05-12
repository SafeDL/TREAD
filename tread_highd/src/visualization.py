"""
visualization.py — 可视化工具
==============================
生成事件轨迹图、风险时序图、风险分布图等可诊断的可视化。
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _with_danger_columns(df, eps=1e-6):
    """Ensure plotting uses larger-is-riskier columns."""
    df = df.copy()
    if "min_ttc" in df.columns:
        df["ttc_severity"] = 1.0 / (df["min_ttc"].astype(float) + eps)
    if "min_thw" in df.columns:
        df["thw_severity"] = 1.0 / (df["min_thw"].astype(float) + eps)
    if "max_drac" in df.columns:
        df["drac_severity"] = df["max_drac"].astype(float)
    return df


def plot_risk_distribution(events_df, event_type, save_path):
    """绘制特定事件类型的 danger-oriented 风险分布。"""
    df = events_df[(events_df["event_type"] == event_type) & events_df["is_valid"]]
    df = _with_danger_columns(df)
    if len(df) == 0:
        logger.warning("No valid %s events for distribution plot.", event_type)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    candidates = [
        ("ttc_severity", "TTC Severity = 1 / min_TTC (larger = riskier)"),
        ("thw_severity", "THW Severity = 1 / min_THW (larger = riskier)"),
        ("drac_severity", "DRAC Severity = max_DRAC (larger = riskier)"),
        ("risk_score", "Risk Score (larger = riskier)"),
    ]
    pairs = [(col, title) for col, title in candidates if col in df.columns]

    for ax, (col, title) in zip(axes.flat, pairs):
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        pos_vals = vals[vals > 0]
        n_non_positive = int((vals <= 0).sum())
        if len(pos_vals) > 0:
            vmin = float(pos_vals.min())
            vmax = float(pos_vals.max())
            if vmax > vmin:
                bins = np.geomspace(vmin, vmax, 51)
            else:
                bins = 10
            ax.hist(pos_vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_xscale("log")
            for q, color in [(0.90, "orange"), (0.95, "red"), (0.99, "darkred")]:
                qv = float(vals.quantile(q))
                if qv > 0:
                    ax.axvline(qv, color=color, ls="--", label=f"P{int(q * 100)}={qv:.3g}")
            ax.legend(fontsize=8)
            ax.set_yscale("log")
        if n_non_positive > 0:
            title = f"{title}\npositive values only; dropped {n_non_positive} non-positive"
        ax.set_title(title)
        ax.set_xlabel("Danger-oriented value (log scale)")
        ax.set_ylabel("Count (log)")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")

    fig.suptitle(f"Risk Distribution — {event_type} (N={len(df)})", fontsize=12)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survival_curve(events_df, event_type, save_path):
    """Empirical survival function 1-CDF on log-log axes for tail diagnosis."""
    df = events_df[(events_df["event_type"] == event_type) & events_df["is_valid"]]
    df = _with_danger_columns(df)
    if len(df) == 0:
        logger.warning("No valid %s events for survival plot.", event_type)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    candidates = [
        ("ttc_severity", "TTC Severity = 1 / min_TTC"),
        ("thw_severity", "THW Severity = 1 / min_THW"),
        ("drac_severity", "DRAC Severity = max_DRAC"),
        ("risk_score", "Risk Score"),
    ]
    pairs = [(col, title) for col, title in candidates if col in df.columns]

    for ax, (col, title) in zip(axes.flat, pairs):
        vals = df[col].dropna().to_numpy()
        vals = vals[vals > 0]
        if len(vals) > 0:
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            survival = 1.0 - np.arange(1, n + 1) / (n + 1)
            ax.loglog(sorted_vals, survival, color="steelblue", lw=1.5)
            for q, color in [(0.90, "orange"), (0.95, "red"), (0.99, "darkred")]:
                qv = float(np.quantile(vals, q))
                ax.axvline(qv, color=color, ls="--", alpha=0.7,
                           label=f"P{int(q * 100)}={qv:.3g}")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("P(X > x)")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")

    fig.suptitle(f"Tail Survival — {event_type} (N={len(df)})", fontsize=12)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ttc_drac_scatter(events_df, save_path):
    """绘制 danger-oriented TTC severity vs DRAC severity 散点图(log-log)。"""
    df = _with_danger_columns(events_df[events_df["is_valid"]])
    if len(df) == 0:
        return
    if "ttc_severity" not in df.columns or "drac_severity" not in df.columns:
        logger.warning("Cannot draw risk scatter without TTC and DRAC columns.")
        return
    x_col = "ttc_severity"
    y_col = "drac_severity"

    fig, ax = plt.subplots(figsize=(8, 6))
    pos_mask = (df[x_col] > 0) & (df[y_col] > 0)
    n_drop = int((~pos_mask).sum())
    df = df[pos_mask]

    for etype, color, marker in [("following", "blue", "o"), ("cut_in", "red", "^")]:
        sub = df[df["event_type"] == etype]
        if len(sub) > 0:
            ax.scatter(sub[x_col], sub[y_col], c=color, marker=marker,
                       alpha=0.3, s=8, label=f"{etype} (n={len(sub)})",
                       linewidths=0)

    # P95/P99 参考线(全体有效正样本)
    for col, axis in [(x_col, "v"), (y_col, "h")]:
        if len(df[col]) > 0:
            for q, color, ls in [(0.95, "red", "--"), (0.99, "darkred", "-.")]:
                qv = float(df[col].quantile(q))
                if axis == "v":
                    ax.axvline(qv, color=color, ls=ls, alpha=0.5,
                               label=f"{col} P{int(q * 100)}={qv:.3g}")
                else:
                    ax.axhline(qv, color=color, ls=ls, alpha=0.5,
                               label=f"{col} P{int(q * 100)}={qv:.3g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TTC Severity = 1 / min_ttc (larger = riskier)")
    ax.set_ylabel("DRAC Severity = max_drac (larger = riskier)")
    title = "Danger-Oriented Risk Scatter (log-log)"
    if n_drop > 0:
        title += f"  |  dropped {n_drop} non-positive points"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
