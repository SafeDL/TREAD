#!/usr/bin/env python3
"""Fit a POT/GPD EVT model to declustered highD longitudinal risk peaks.

No raw-data traversal — reads pre-extracted event scores only.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genpareto

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import setup_logging
from process_highD.src.io_utils import load_config, resolve_data_path
from utils.evt import (
    RETURN_PERIODS,
    empirical_survival,
    fit_evt_model,
)
from utils.highd_exposure import extract_independent_peaks
from utils.io import write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
SCRIPT_DEFAULTS: dict[str, Any] = {
    "log_level": "INFO",
    "selected_method": "B",
    "min_exceedances": 20,
    "max_tail_fraction": 0.25,
    "max_threshold_candidates": 400,
    "min_threshold_exceedance_rate": 0.10,
    "random_seed": 42,
}
logger = logging.getLogger(__name__)


def _paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths_cfg = cfg.get("paths", {})
    highd_events_dir = resolve_data_path(paths_cfg["output_dir"], config_path)
    peak_cfg = cfg.get("evt_peak", {})
    model_path = resolve_data_path(
        peak_cfg.get(
            "model_path",
            "../../../results/highd_following_tail/evt/longitudinal_peak_evt_model.json",
        ),
        config_path,
    )
    return {
        "score_csv": highd_events_dir / "following_event_scores.csv",
        "model": model_path,
        "summary": resolve_data_path(
            peak_cfg.get(
                "summary_path",
                "../../../results/highd_following_tail/evt/longitudinal_peak_evt_summary.json",
            ),
            config_path,
        ),
        "figure_dir": model_path.parent / "figures",
    }


def _load_scored_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD following score cache not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    required = {
        "event_id",
        "recording_id",
        "ego_id",
        "target_id",
        "start_frame",
        "end_frame",
        "anchor_frame",
        "y_long",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    return frame


def _mean_excess_rows(
    values: np.ndarray,
    candidates: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    u_values = np.asarray([row["u"] for row in candidates], dtype=np.float64)
    means = []
    for u_value in u_values:
        excess = values[values > u_value] - u_value
        means.append(float(np.mean(excess)) if excess.size else float("nan"))
    return u_values, np.asarray(means, dtype=np.float64)


def _write_evt_diagnostic_plots(
    figure_dir: Path,
    *,
    model: Any,
    values: np.ndarray,
    collision_critical_level: float,
) -> dict[str, str]:
    """Write standard EVT diagnostic plots (no exposure-dependent plots)."""
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # ---- y_long histogram ----
    x_limit = float(np.quantile(values, 0.999))
    x_limit = max(x_limit, float(model.u) * 1.05, 1.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.hist(values[values <= x_limit], bins=70, color="tab:blue", alpha=0.72)
    ax.axvline(float(model.u), color="black", linestyle="--", label="POT threshold u")
    for period in RETURN_PERIODS:
        key = f"z{period}"
        if key in model.return_levels:
            z_value = float(model.return_levels[key])
            if z_value <= x_limit:
                ax.axvline(z_value, linestyle=":", label=key)
    if float(collision_critical_level) <= x_limit:
        ax.axvline(
            float(collision_critical_level),
            color="tab:red",
            linestyle="-.",
            label="collision critical level",
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0.0, right=x_limit)
    ax.set_xlabel("independent peak y_long")
    ax.set_ylabel("peak count (log)")
    ax.set_title(
        f"highD independent peak risk, clipped at p99.9; max={np.max(values):.2f}"
    )
    ax.legend()
    path = figure_dir / "peak_evt_y_long_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["peak_y_long_histogram"] = str(path)

    # ---- survival fit ----
    sorted_values = np.sort(values[values >= float(model.u)])
    if sorted_values.size:
        empirical = empirical_survival(values, sorted_values)
        model_survival = model.survival(sorted_values)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        focus_max = float(
            max(
                float(model.return_levels.get("z100", model.return_level(100))) * 1.35,
                np.quantile(values, 0.995),
            )
        )
        panels = (
            (axes[0], focus_max, "return-level region"),
            (axes[1], float(np.max(values)), "full tail"),
        )
        for ax, right_limit, title in panels:
            mask = sorted_values <= right_limit
            ax.scatter(
                sorted_values[mask],
                empirical[mask],
                label="empirical survival",
                s=7,
                alpha=0.38,
            )
            ax.plot(
                sorted_values[mask],
                model_survival[mask],
                label="GPD tail survival",
                linewidth=2.0,
                color="tab:orange",
            )
            ax.axvline(float(model.u), color="black", linestyle="--", label="u")
            for period in RETURN_PERIODS:
                key = f"z{period}"
                if key not in model.return_levels:
                    continue
                z_value = float(model.return_levels[key])
                if z_value <= right_limit:
                    survival = float(model.survival(z_value))
                    ax.scatter([z_value], [survival], s=32)
                    ax.annotate(key, (z_value, survival))
            ax.set_yscale("log")
            ax.set_xlabel("independent peak y_long")
            ax.set_title(title)
            ax.grid(True, alpha=0.22)
        axes[0].set_ylabel("P(Y_long_peak > y)")
        axes[1].legend()
        path = figure_dir / "peak_evt_survival_fit.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_survival_fit"] = str(path)

    # ---- threshold stability ----
    candidates = model.threshold_candidates
    if candidates:
        u = np.asarray([row["u"] for row in candidates], dtype=np.float64)
        xi = np.asarray([row["xi"] for row in candidates], dtype=np.float64)
        modified_scale = np.asarray(
            [row["modified_scale"] for row in candidates],
            dtype=np.float64,
        )
        exceedance_rate = np.asarray(
            [row["exceedance_rate"] for row in candidates],
            dtype=np.float64,
        )
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
        for ax in axes:
            ax.axvline(float(model.u), color="black", linestyle="--", linewidth=1.3)
            ax.axvspan(
                float(model.u),
                float(np.max(u)),
                color="tab:green",
                alpha=0.08,
                label="more extreme than selected u",
            )
            ax.grid(True, alpha=0.22)
        axes[0].plot(u, xi, linewidth=1.4)
        axes[0].scatter([float(model.u)], [float(model.xi)], color="black", s=28)
        axes[0].set_ylabel("xi")
        axes[0].set_title("Peak EVT threshold stability")
        axes[1].plot(u, modified_scale, linewidth=1.4, color="tab:green")
        axes[1].set_ylabel("modified scale")
        axes[2].plot(u, exceedance_rate, linewidth=1.4, color="tab:orange")
        axes[2].set_xlabel("threshold u")
        axes[2].set_ylabel("P(peak > u)")
        fig.tight_layout()
        path = figure_dir / "peak_evt_threshold_stability.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_threshold_stability"] = str(path)

        # ---- mean excess ----
        mean_u, mean_excess = _mean_excess_rows(values, candidates)
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        ax.plot(mean_u, mean_excess, linewidth=1.5)
        ax.axvline(float(model.u), color="black", linestyle="--", label="u")
        ax.axvspan(float(model.u), float(np.max(mean_u)), color="tab:green", alpha=0.08)
        ax.set_xlabel("threshold u")
        ax.set_ylabel("mean excess E[Y-u | Y>u]")
        ax.set_title("Peak EVT mean residual life diagnostic")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = figure_dir / "peak_evt_mean_excess.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_mean_excess"] = str(path)

    # ---- tail fit diagnostics (QQ / PP) ----
    excess = np.sort(values[values > float(model.u)] - float(model.u))
    if excess.size:
        empirical_cdf = np.arange(1, excess.size + 1, dtype=np.float64)
        empirical_cdf = empirical_cdf / (excess.size + 1.0)
        gpd_cdf = genpareto.cdf(
            excess,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        gpd_quantiles = genpareto.ppf(
            empirical_cdf,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
        max_q = float(max(np.max(excess), np.max(gpd_quantiles)))
        axes[0].scatter(gpd_quantiles, excess, s=8, alpha=0.40)
        axes[0].plot([0.0, max_q], [0.0, max_q], color="black", linestyle="--")
        axes[0].set_xlabel("GPD theoretical excess quantile")
        axes[0].set_ylabel("empirical excess quantile")
        axes[0].set_title("Peak tail QQ diagnostic")
        axes[0].grid(True, alpha=0.25)
        axes[1].scatter(empirical_cdf, gpd_cdf, s=8, alpha=0.40, color="tab:orange")
        axes[1].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
        axes[1].set_xlabel("empirical CDF of excess")
        axes[1].set_ylabel("GPD CDF of excess")
        axes[1].set_title("Peak tail PP diagnostic")
        axes[1].grid(True, alpha=0.25)
        path = figure_dir / "peak_evt_tail_fit_diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_tail_fit_diagnostics"] = str(path)

    return paths


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    cfg = load_config(DEFAULT_CONFIG_PATH)
    paths = _paths(cfg, DEFAULT_CONFIG_PATH)
    peak_cfg = cfg.get("evt_peak", {})
    decluster_cfg = cfg.get("exposure", {}).get("declustering", {})

    scores = _load_scored_events(paths["score_csv"])

    target_fps = float(cfg.get("sampling", {}).get("target_fps", 25))
    group_keys = tuple(
        str(item)
        for item in decluster_cfg.get("group_keys", ["recording_id", "ego_id"])
    )
    run_length_seconds = float(decluster_cfg.get("run_length_seconds", 5.0))
    peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
    )
    if not peaks:
        raise RuntimeError("No independent highD following risk peaks were extracted")

    y_peaks = np.asarray([row["y_long_max"] for row in peaks], dtype=np.float64)
    model = fit_evt_model(
        y_peaks,
        selected_method=str(SCRIPT_DEFAULTS["selected_method"]),
        min_exceedances=int(SCRIPT_DEFAULTS["min_exceedances"]),
        max_tail_fraction=SCRIPT_DEFAULTS["max_tail_fraction"],
        max_threshold_candidates=int(SCRIPT_DEFAULTS["max_threshold_candidates"]),
        min_threshold_exceedance_rate=float(
            SCRIPT_DEFAULTS["min_threshold_exceedance_rate"]
        ),
        bootstrap_samples=int(peak_cfg.get("bootstrap_samples", 200)),
        random_seed=int(SCRIPT_DEFAULTS["random_seed"]),
    )

    tail_peaks = int(np.sum(y_peaks > float(model.u)))
    collision_critical_level = float(peak_cfg.get("collision_critical_level", 5.0))

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model.to_json(paths["model"])

    figures = _write_evt_diagnostic_plots(
        paths["figure_dir"],
        model=model,
        values=y_peaks,
        collision_critical_level=collision_critical_level,
    )

    write_json(
        paths["summary"],
        {
            "model_path": str(paths["model"]),
            "score_csv": str(paths["score_csv"]),
            "model_type": "gpd_pot_longitudinal_independent_peak_risk",
            "num_independent_peaks": int(len(peaks)),
            "num_tail_peaks": tail_peaks,
            "u": float(model.u),
            "xi": float(model.xi),
            "beta": float(model.beta),
            "exceedance_rate": float(model.exceedance_rate),
            "collision_critical_level": collision_critical_level,
            "collision_critical_level_mode": "fixed_y_long",
            "return_levels": model.return_levels,
            "return_level_ci": model.return_level_ci,
            "declustering_run_length_seconds": run_length_seconds,
            "declustering_group_keys": list(group_keys),
            "figures": figures,
        },
    )
    logger.info(
        "Saved peak EVT model to %s | peaks=%d tail_peaks=%d u=%.6f",
        paths["model"],
        len(peaks),
        tail_peaks,
        model.u,
    )


if __name__ == "__main__":
    main()
