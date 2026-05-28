#!/usr/bin/env python3
"""Fit a POT/GPD EVT model for highD event-level longitudinal risk."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import setup_logging
from scipy.stats import genpareto

from utils.evt import RETURN_PERIODS, empirical_survival, fit_evt_model
from utils.highd_longitudinal import (
    DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    load_highd_event_score_cache,
)
from utils.io import write_csv, write_json


SCRIPT_DEFAULTS: dict[str, Any] = {
    **DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    "events_csv": ROOT / "results" / "highd_events" / "events.csv",
    "event_score_cache_path": ROOT
    / "results"
    / "highd_events"
    / "following_event_scores.csv",
    "evt_model_path": ROOT
    / "results"
    / "highd_evt"
    / "following"
    / "longitudinal_evt_model.json",
    "evt_scores_path": ROOT
    / "results"
    / "highd_evt"
    / "following"
    / "longitudinal_evt_scores.csv",
    "evt_arrays_path": ROOT
    / "results"
    / "highd_evt"
    / "following"
    / "longitudinal_evt_model.npz",
    "threshold_stability_path": ROOT
    / "results"
    / "highd_evt"
    / "following"
    / "threshold_stability.csv",
    "diagnostic_points_path": ROOT
    / "results"
    / "highd_evt"
    / "following"
    / "evt_survival_diagnostic_points.csv",
    "figure_dir": ROOT / "results" / "highd_evt" / "following" / "figures",
    "selected_method": "B",
    "min_exceedances": 20,
    "max_tail_fraction": 0.25,
    "min_threshold_exceedance_rate": 0.10,
    "bootstrap_samples": 200,
    "random_seed": 42,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _score_table(rows: list[dict[str, Any]], model) -> list[dict[str, Any]]:
    keys = (
        "event_index",
        "recording_id",
        "event_id",
        "anchor_frame",
        "available_future_steps",
        "event_future_steps",
        "recorded_min_gap",
        "recorded_min_ttc",
        "recorded_min_thw",
        "recorded_max_drac",
        "min_ego_accel",
        "collision",
        "near_collision",
        "hard_brake",
        "y_long",
        "proxy_risk_score",
        "ttc_risk_score",
        "thw_risk_score",
        "gap_risk_score",
        "drac_risk_score",
        "collision_risk_score",
        "near_collision_risk_score",
        "hard_brake_risk_score",
    )
    y_long = np.asarray([row["y_long"] for row in rows], dtype=np.float64)
    tail_probability = np.asarray(model.survival(y_long), dtype=np.float64)
    risk_score = np.asarray(model.score(y_long), dtype=np.float64)
    out = []
    for idx, row in enumerate(rows):
        item = {key: row[key] for key in keys if key in row}
        item["evt_tail_probability"] = float(tail_probability[idx])
        item["risk_score"] = float(risk_score[idx])
        out.append(item)
    return out


def _load_cached_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            "highD following score cache is required before EVT fitting: "
            f"{path}. Run process_highD/scripts/extract_highd_events.py first."
        )
    rows = load_highd_event_score_cache(path)
    if not rows:
        raise RuntimeError(f"highD following score cache is empty: {path}")
    logger.info("Loaded %d highD following risk rows from %s", len(rows), path)
    return rows


def _load_rows() -> tuple[list[dict[str, Any]], int, str]:
    cache_path = Path(SCRIPT_DEFAULTS["event_score_cache_path"])
    rows = _load_cached_rows(cache_path)
    return rows, 0, "following_event_scores_cache"


def _write_arrays(path: Path, model, y_long: np.ndarray) -> None:
    threshold_rows = model.threshold_candidates
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_long=np.asarray(y_long, dtype=np.float32),
        calibration_values=np.asarray(model.calibration_values, dtype=np.float32),
        threshold_k=np.asarray([row["k"] for row in threshold_rows], dtype=np.float32),
        threshold_u=np.asarray([row["u"] for row in threshold_rows], dtype=np.float32),
        threshold_xi=np.asarray([row["xi"] for row in threshold_rows], dtype=np.float32),
        threshold_beta=np.asarray(
            [row["beta"] for row in threshold_rows],
            dtype=np.float32,
        ),
        return_periods=np.asarray(RETURN_PERIODS, dtype=np.int32),
        return_levels=np.asarray(
            [model.return_levels[f"z{period}"] for period in RETURN_PERIODS],
            dtype=np.float32,
        ),
    )


def _evt_diagnostics(model, y_long: np.ndarray) -> dict[str, Any]:
    values = np.asarray(y_long, dtype=np.float64)
    values = values[np.isfinite(values)]
    excess = np.sort(values[values > float(model.u)] - float(model.u))
    if excess.size:
        gpd_cdf = genpareto.cdf(
            excess,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        empirical_cdf = np.arange(1, excess.size + 1, dtype=np.float64)
        empirical_cdf = empirical_cdf / (excess.size + 1.0)
        cdf_error = gpd_cdf - empirical_cdf
        cdf_rmse = float(np.sqrt(np.mean(np.square(cdf_error))))
        cdf_max_abs = float(np.max(np.abs(cdf_error)))
        gpd_quantiles = genpareto.ppf(
            empirical_cdf,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        qq_error = gpd_quantiles - excess
        qq_rmse = float(np.sqrt(np.mean(np.square(qq_error))))
        qq_scale = float(np.std(excess)) if float(np.std(excess)) > 0.0 else 1.0
        qq_corr = float(np.corrcoef(excess, gpd_quantiles)[0, 1])
        pp_corr = float(np.corrcoef(empirical_cdf, gpd_cdf)[0, 1])
    else:
        cdf_rmse = float("nan")
        cdf_max_abs = float("nan")
        qq_rmse = float("nan")
        qq_scale = float("nan")
        qq_corr = float("nan")
        pp_corr = float("nan")

    diagnostics: dict[str, Any] = {
        "num_calibration_values": int(values.size),
        "num_exceedances": int(excess.size),
        "empirical_exceedance_rate": float(excess.size / max(values.size, 1)),
        "model_exceedance_rate": float(model.exceedance_rate),
        "threshold_quantile": float(np.mean(values <= float(model.u))),
        "gpd_excess_cdf_rmse": cdf_rmse,
        "gpd_excess_cdf_max_abs_error": cdf_max_abs,
        "gpd_excess_qq_rmse": qq_rmse,
        "gpd_excess_qq_rmse_over_std": float(qq_rmse / qq_scale),
        "gpd_excess_qq_correlation": qq_corr,
        "gpd_excess_pp_correlation": pp_corr,
        "return_levels_above_threshold": bool(
            all(float(model.return_levels[f"z{period}"]) > float(model.u) for period in RETURN_PERIODS)
        ),
    }
    for period in RETURN_PERIODS:
        key = f"z{period}"
        z_value = float(model.return_levels[key])
        evt_survival = float(model.survival(z_value))
        empirical = float(empirical_survival(values, z_value))
        diagnostics[f"{key}_evt_survival"] = float(model.survival(z_value))
        diagnostics[f"{key}_evt_score"] = float(model.score(z_value))
        diagnostics[f"{key}_empirical_survival"] = empirical
        diagnostics[f"{key}_survival_abs_error"] = float(abs(empirical - evt_survival))
    return diagnostics


def _write_diagnostic_points(path: Path, model, y_long: np.ndarray) -> None:
    values = np.asarray(y_long, dtype=np.float64)
    values = values[np.isfinite(values)]
    quantiles = np.unique(
        np.concatenate(
            [
                np.linspace(0.50, 0.90, 9),
                np.linspace(0.91, 0.99, 9),
                np.asarray([0.995, 0.999]),
            ]
        )
    )
    y_values = np.quantile(values, quantiles)
    rows = []
    for q, y_value in zip(quantiles, y_values, strict=True):
        rows.append(
            {
                "quantile": float(q),
                "y_long": float(y_value),
                "empirical_survival": float(empirical_survival(values, y_value)),
                "evt_survival": float(model.survival(y_value)),
                "evt_score": float(model.score(y_value)),
            }
        )
    for key, z_value in model.return_levels.items():
        rows.append(
            {
                "quantile": float("nan"),
                "y_long": float(z_value),
                "empirical_survival": float(empirical_survival(values, z_value)),
                "evt_survival": float(model.survival(z_value)),
                "evt_score": float(model.score(z_value)),
                "label": key,
            }
        )
    if rows:
        write_csv(path, rows)


def _mean_excess_rows(values: np.ndarray, candidates: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    u_values = np.asarray([row["u"] for row in candidates], dtype=np.float64)
    means = []
    for u_value in u_values:
        excess = values[values > u_value] - u_value
        means.append(float(np.mean(excess)) if excess.size else float("nan"))
    return u_values, np.asarray(means, dtype=np.float64)


def _write_diagnostic_plots(figure_dir: Path, model, y_long: np.ndarray) -> dict[str, str]:
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(y_long, dtype=np.float64)
    values = values[np.isfinite(values)]
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    x_limit = float(np.quantile(values, 0.999))
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.hist(values[values <= x_limit], bins=70, color="tab:blue", alpha=0.72)
    ax.axvline(float(model.u), color="black", linestyle="--", label="u")
    for period in RETURN_PERIODS:
        z_value = float(model.return_levels[f"z{period}"])
        ax.axvline(z_value, linestyle=":", label=f"z{period}")
    ax.set_yscale("log")
    ax.set_xlim(left=0.0, right=x_limit)
    ax.set_xlabel("y_long")
    ax.set_ylabel("event count (log)")
    ax.set_title(f"highD longitudinal risk, clipped at p99.9; max={np.max(values):.2f}")
    ax.legend()
    path = figure_dir / "evt_y_long_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["y_long_histogram"] = str(path)

    sorted_values = np.sort(values[values >= float(model.u)])
    empirical = empirical_survival(values, sorted_values)
    model_survival = model.survival(sorted_values)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    focus_max = float(max(model.return_levels["z100"] * 1.35, np.quantile(values, 0.995)))
    panels = ((axes[0], focus_max, "return-level region"), (axes[1], float(np.max(values)), "full tail"))
    for ax, right_limit, title in panels:
        mask = sorted_values <= right_limit
        ax.scatter(sorted_values[mask], empirical[mask], label="empirical survival", s=7, alpha=0.38)
        ax.plot(
            sorted_values[mask],
            model_survival[mask],
            label="GPD tail survival",
            linewidth=2.0,
            color="tab:orange",
        )
        ax.axvline(float(model.u), color="black", linestyle="--", label="u")
        for period in RETURN_PERIODS:
            z_value = float(model.return_levels[f"z{period}"])
            if z_value <= right_limit:
                ax.scatter([z_value], [float(model.survival(z_value))], s=32)
                ax.annotate(f"z{period}", (z_value, float(model.survival(z_value))))
        ax.set_yscale("log")
        ax.set_xlabel("y_long")
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("P(Y_long > y)")
    axes[1].legend()
    path = figure_dir / "evt_survival_fit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["survival_fit"] = str(path)

    candidates = model.threshold_candidates
    if candidates:
        u = np.asarray([row["u"] for row in candidates], dtype=np.float64)
        xi = np.asarray([row["xi"] for row in candidates], dtype=np.float64)
        beta = np.asarray([row["beta"] for row in candidates], dtype=np.float64)
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
        axes[0].set_title("Threshold stability")
        axes[1].plot(u, modified_scale, linewidth=1.4, color="tab:green")
        axes[1].set_ylabel("modified scale")
        axes[2].plot(u, exceedance_rate, linewidth=1.4, color="tab:orange")
        axes[2].set_xlabel("threshold u")
        axes[2].set_ylabel("P(Y > u)")
        fig.tight_layout()
        path = figure_dir / "evt_threshold_stability.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["threshold_stability"] = str(path)

        mean_u, mean_excess = _mean_excess_rows(values, candidates)
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        ax.plot(mean_u, mean_excess, linewidth=1.5)
        ax.axvline(float(model.u), color="black", linestyle="--", label="u")
        ax.axvspan(float(model.u), float(np.max(mean_u)), color="tab:green", alpha=0.08)
        ax.set_xlabel("threshold u")
        ax.set_ylabel("mean excess E[Y-u | Y>u]")
        ax.set_title("Mean residual life diagnostic")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = figure_dir / "evt_mean_excess.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["mean_excess"] = str(path)

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
        axes[0].set_title("Tail QQ diagnostic")
        axes[0].grid(True, alpha=0.25)
        axes[1].scatter(empirical_cdf, gpd_cdf, s=8, alpha=0.40, color="tab:orange")
        axes[1].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
        axes[1].set_xlabel("empirical CDF of excess")
        axes[1].set_ylabel("GPD CDF of excess")
        axes[1].set_title("Tail PP diagnostic")
        axes[1].grid(True, alpha=0.25)
        path = figure_dir / "evt_tail_fit_diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["tail_fit_diagnostics"] = str(path)

    return paths


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    rows, skipped, input_source = _load_rows()
    y_long = np.asarray([row["y_long"] for row in rows], dtype=np.float64)
    model = fit_evt_model(
        y_long,
        selected_method=str(SCRIPT_DEFAULTS["selected_method"]),
        min_exceedances=int(SCRIPT_DEFAULTS["min_exceedances"]),
        max_tail_fraction=float(SCRIPT_DEFAULTS["max_tail_fraction"]),
        min_threshold_exceedance_rate=float(
            SCRIPT_DEFAULTS["min_threshold_exceedance_rate"]
        ),
        bootstrap_samples=int(SCRIPT_DEFAULTS["bootstrap_samples"]),
        random_seed=int(SCRIPT_DEFAULTS["random_seed"]),
    )

    model_path = Path(SCRIPT_DEFAULTS["evt_model_path"])
    model.to_json(model_path)
    write_csv(Path(SCRIPT_DEFAULTS["evt_scores_path"]), _score_table(rows, model))
    write_csv(
        Path(SCRIPT_DEFAULTS["threshold_stability_path"]),
        model.threshold_candidates,
    )
    _write_arrays(Path(SCRIPT_DEFAULTS["evt_arrays_path"]), model, y_long)
    _write_diagnostic_points(
        Path(SCRIPT_DEFAULTS["diagnostic_points_path"]),
        model,
        y_long,
    )
    diagnostics = _evt_diagnostics(model, y_long)
    figures = _write_diagnostic_plots(
        Path(SCRIPT_DEFAULTS["figure_dir"]),
        model,
        y_long,
    )
    write_json(
        model_path.with_name("longitudinal_evt_summary.json"),
        {
            "events_csv": str(SCRIPT_DEFAULTS["events_csv"]),
            "event_score_cache_path": str(SCRIPT_DEFAULTS["event_score_cache_path"]),
            "input_source": input_source,
            "evt_model_path": str(model_path),
            "num_events": int(len(rows)),
            "skipped_events": int(skipped),
            "y_long_min": float(np.min(y_long)),
            "y_long_mean": float(np.mean(y_long)),
            "y_long_p95": float(np.percentile(y_long, 95.0)),
            "y_long_max": float(np.max(y_long)),
            "u": float(model.u),
            "xi": float(model.xi),
            "beta": float(model.beta),
            "exceedance_rate": float(model.exceedance_rate),
            "selected_method": model.selected_method,
            "selected_thresholds": model.selected_thresholds,
            "threshold_selection": model.threshold_selection,
            "return_levels": model.return_levels,
            "return_level_ci": model.return_level_ci,
            "diagnostics": diagnostics,
            "figures": figures,
            "scoring_method": (
                "event-level y_long = softmax-pool(1/TTC, 1/THW, 1/gap, "
                "DRAC) plus collision, near-collision, and hard-brake terms; "
                "POT/GPD fitted to highD following-event tail"
            ),
        },
    )
    logger.info(
        "Saved highD longitudinal EVT model to %s with u=%.6f xi=%.6f beta=%.6f",
        model_path,
        model.u,
        model.xi,
        model.beta,
    )


if __name__ == "__main__":
    main()
