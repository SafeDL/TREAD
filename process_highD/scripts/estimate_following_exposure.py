#!/usr/bin/env python3
"""Fit highD following EVT model and estimate independent tail-peak exposure.

This is the single entry-point for following long-tail distribution modeling:
it fits the POT/GPD model, then reads per-recording exposure pre-computed by
extract_highd_events.py and estimates independent tail-event rates.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.io_utils import load_config, resolve_data_path
from process_highD.src.evt_fitting import fit_highd_peak_evt
from utils.evt import (
    fit_gpd_excess,
    gpd_conditional_survival,
    load_evt_model,
    return_level_for_tail_exposure,
)
from utils.highd_exposure import (
    KM_PER_MILE,
    collision_distance_summary,
    extract_independent_peaks,
    peak_rate_summary,
)
from utils.io import write_csv, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "event_id",
    "recording_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "y_long",
}


def _fit_evt_model(config_path: Path) -> None:
    fit_highd_peak_evt(
        config_path=config_path,
        score_filename="following_event_scores.csv",
        peak_config_key="following_evt_peak",
        declustering_config_path=("following_exposure", "declustering"),
        required_columns=REQUIRED_COLUMNS,
        score_column="y_long",
        peak_value_key="y_long_max",
        scenario_label="following",
        summary_model_type="gpd_pot_longitudinal_independent_peak_risk",
        collision_critical_level_mode="fixed_y_long",
    )


def _load_exposure_csv(path: Path) -> list[dict[str, Any]]:
    """Read per-recording exposure pre-computed by extract_highd_events.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Per-recording exposure CSV not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    return frame.to_dict("records")


def _load_scored_events(path: Path, model: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD following score cache not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    scores = pd.read_csv(path)
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
    missing = sorted(required - set(scores.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    y_long = pd.to_numeric(scores["y_long"], errors="coerce").to_numpy()
    scores["risk_score"] = np.asarray(model.score(y_long), dtype=np.float64)
    scores["evt_tail_probability"] = np.asarray(
        model.survival(y_long),
        dtype=np.float64,
    )
    return scores


def _paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths_cfg = cfg["paths"]
    highd_events_dir = resolve_data_path(paths_cfg["output_dir"], config_path)
    exposure_cfg = cfg["following_exposure"]
    peak_cfg = cfg["following_evt_peak"]
    return {
        "exposure_csv": highd_events_dir / "exposure_per_recording.csv",
        "score_csv": highd_events_dir / "following_event_scores.csv",
        "evt_model": resolve_data_path(peak_cfg["model_path"], config_path),
        "output_dir": resolve_data_path(exposure_cfg["output_dir"], config_path),
    }


# ── return-level distance plot helpers ──


def _return_level_curve(
    distances_km: np.ndarray,
    *,
    tail_peak_rate_per_km: float,
    u: float,
    xi: float,
    beta: float,
) -> np.ndarray:
    expected = np.asarray(distances_km, dtype=np.float64) * float(tail_peak_rate_per_km)
    return np.asarray(
        [
            return_level_for_tail_exposure(
                expected_tail_exceedances=float(value),
                u=float(u),
                xi=float(xi),
                beta=float(beta),
            )
            for value in expected
        ],
        dtype=np.float64,
    )


def _bootstrap_distance_curve(
    values: np.ndarray,
    distances_km: np.ndarray,
    *,
    total_exposure_km: float,
    chosen_k: int,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples <= 0 or values.size < max(chosen_k + 1, 10):
        nan = np.full_like(distances_km, np.nan, dtype=np.float64)
        return nan, nan

    rng = np.random.default_rng(int(seed))
    curves: list[np.ndarray] = []
    for _ in range(int(samples)):
        boot = np.sort(rng.choice(values, size=values.size, replace=True))
        k = min(max(5, int(chosen_k)), values.size - 1)
        u = float(boot[boot.size - k - 1])
        excess = boot[boot > u] - u
        if excess.size < 5:
            continue
        try:
            xi, beta = fit_gpd_excess(excess)
        except ValueError:
            continue
        rate_per_km = float(excess.size / max(total_exposure_km, 1.0e-12))
        curves.append(
            _return_level_curve(
                distances_km,
                tail_peak_rate_per_km=rate_per_km,
                u=u,
                xi=xi,
                beta=beta,
            )
        )
    if not curves:
        nan = np.full_like(distances_km, np.nan, dtype=np.float64)
        return nan, nan
    stack = np.stack(curves, axis=0)
    return (
        np.nanquantile(stack, 0.05, axis=0),
        np.nanquantile(stack, 0.95, axis=0),
    )


def _write_return_level_plot(
    figure_dir: Path,
    *,
    values: np.ndarray,
    model: Any,
    total_exposure_km: float,
    collision_critical_level: float,
    collision_return_period_km: float,
    bootstrap_samples: int,
    distance_min_km: float,
    distance_max_km: float,
    random_seed: int,
) -> dict[str, str]:
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_tail = int(np.sum(values > float(model.u)))
    tail_rate_per_km = float(num_tail / max(total_exposure_km, 1.0e-12))

    tail_values = np.sort(values[values > float(model.u)])
    if tail_values.size > 0:
        ranks = np.arange(tail_values.size, 0, -1, dtype=np.float64)
        empirical_return_km = total_exposure_km / ranks
    else:
        tail_values = np.array([], dtype=np.float64)
        empirical_return_km = np.array([], dtype=np.float64)

    min_distance_candidates = [float(distance_min_km)]
    if tail_values.size > 0:
        min_distance_candidates.append(float(np.nanmin(empirical_return_km)))
    if np.isfinite(collision_return_period_km) and collision_return_period_km > 0.0:
        min_distance_candidates.append(float(collision_return_period_km) / 8.0)
    plot_min_km = max(min(min_distance_candidates), 1.0e-3)
    plot_max_km = max(float(distance_max_km), plot_min_km * 10.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    distances_km = np.logspace(
        np.log10(plot_min_km),
        np.log10(plot_max_km),
        360,
    )
    levels = _return_level_curve(
        distances_km,
        tail_peak_rate_per_km=tail_rate_per_km,
        u=float(model.u),
        xi=float(model.xi),
        beta=float(model.beta),
    )
    lower, upper = _bootstrap_distance_curve(
        values,
        distances_km,
        total_exposure_km=total_exposure_km,
        chosen_k=num_tail,
        samples=bootstrap_samples,
        seed=random_seed,
    )

    def draw_panel(
        ax: Any,
        *,
        x_min: float,
        x_max: float,
        y_max: float | None,
        title: str,
        log_y: bool,
        show_legend: bool,
    ) -> None:
        x_mask = (distances_km >= x_min) & (distances_km <= x_max)
        if tail_values.size > 0:
            point_mask = (
                (empirical_return_km >= x_min)
                & (empirical_return_km <= x_max)
                & (tail_values <= y_max if y_max is not None else True)
            )
            ax.scatter(
                empirical_return_km[point_mask],
                tail_values[point_mask],
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                s=18,
                alpha=0.55,
                zorder=3,
                label="empirical tail observations",
            )
        ax.plot(
            distances_km[x_mask],
            levels[x_mask],
            color="green",
            linewidth=2.2,
            label="GPD fit",
        )
        band_mask = x_mask & np.isfinite(lower) & np.isfinite(upper)
        if np.any(band_mask):
            ax.fill_between(
                distances_km[band_mask],
                lower[band_mask],
                upper[band_mask],
                color="tab:red",
                alpha=0.16,
                linewidth=0.0,
                label="90% bootstrap band",
            )
        ax.axhline(
            float(collision_critical_level),
            color="blue",
            linestyle=":",
            linewidth=2.0,
            label="collision critical level",
        )
        if (
            np.isfinite(collision_return_period_km)
            and x_min <= collision_return_period_km <= x_max
        ):
            ax.scatter(
                [collision_return_period_km],
                [collision_critical_level],
                facecolors="none",
                edgecolors="orange",
                linewidths=2.2,
                s=78,
                zorder=5,
                label="estimated critical distance",
            )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if y_max is not None:
            ax.set_ylim(0.0, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Return period distance (km)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.26)
        if show_legend:
            ax.legend(loc="best", fontsize=9)

    focus_x_max = max(
        1.0e3,
        float(collision_return_period_km) * 25.0
        if np.isfinite(collision_return_period_km)
        else 0.0,
    )
    focus_x_max = min(focus_x_max, plot_max_km)
    focus_y_max = max(
        float(collision_critical_level) * 2.2,
        float(model.return_level(100)) * 1.45,
        float(model.u) * 1.3,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    draw_panel(
        axes[0],
        x_min=plot_min_km,
        x_max=focus_x_max,
        y_max=focus_y_max,
        title="critical return-level region",
        log_y=False,
        show_legend=True,
    )
    draw_panel(
        axes[1],
        x_min=plot_min_km,
        x_max=plot_max_km,
        y_max=None,
        title="full extrapolation",
        log_y=True,
        show_legend=False,
    )
    axes[0].set_ylabel("Return level (TREAD y_long)")
    fig.suptitle("Peak-level EVT return level by driving distance")
    path = figure_dir / "peak_evt_return_level_distance.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"peak_evt_return_level_distance": str(path)}


OBSOLETE_RECORDING_PLOTS = (
    "following_exposure_miles_by_recording.png",
    "all_vehicle_exposure_miles_by_recording.png",
    "independent_tail_peaks_by_recording.png",
    "tail_peak_rate_per_mile_by_recording.png",
    "tail_peak_rate_per_all_vehicle_mile_by_recording.png",
)


def _remove_obsolete_recording_plots(figure_dir: Path) -> None:
    """Remove old per-recording bar charts that are no longer reported."""
    for name in OBSOLETE_RECORDING_PLOTS:
        path = figure_dir / name
        if path.exists():
            path.unlink()


# ── collision level ──


def _collision_level_from_config(
    peak_cfg: dict[str, Any],
    *,
    tail_peak_rate_per_km: float,
    model: Any,
) -> tuple[float, dict[str, Any]]:
    if "collision_critical_reference_km" in peak_cfg:
        reference_km = float(peak_cfg["collision_critical_reference_km"])
        if reference_km <= 0.0:
            raise ValueError(
                "following_evt_peak.collision_critical_reference_km must be positive"
            )
        expected_tail_exceedances = float(tail_peak_rate_per_km) * reference_km
        return (
            return_level_for_tail_exposure(
                expected_tail_exceedances=expected_tail_exceedances,
                u=float(model.u),
                xi=float(model.xi),
                beta=float(model.beta),
            ),
            {
                "collision_critical_level_mode": "distance_reference",
                "collision_critical_reference_km": reference_km,
                "collision_critical_expected_tail_exceedances": (
                    expected_tail_exceedances
                ),
            },
        )
    return (
        float(peak_cfg["collision_critical_level"]),
        {
            "collision_critical_level_mode": "fixed_y_long",
            "collision_critical_reference_km": None,
            "collision_critical_expected_tail_exceedances": None,
        },
    )


def _return_period(rate: float) -> float:
    return float(1.0 / rate) if float(rate) > 0.0 else float("inf")


def _log_human_exposure_metrics(
    *,
    model: Any,
    rates: dict[str, float],
    collision_level: float,
    collision_probability_per_peak: float,
    collision_summary: dict[str, float],
) -> None:
    tail_return_miles = _return_period(rates["tail_peak_rate_per_mile"])
    tail_rate_per_km = rates["tail_peak_rate_per_mile"] / KM_PER_MILE
    tail_return_km = _return_period(tail_rate_per_km)
    tail_return_hours = _return_period(rates["tail_peak_rate_per_hour"])
    logger.info(
        (
            "Human highD tail event Y>u: u=%.6g P(Y>u)=%.6g "
            "rate=%.6g/mile %.6g/km %.6g/hour "
            "return=%.6g miles %.6g km %.6g hours"
        ),
        float(model.u),
        float(model.exceedance_rate),
        rates["tail_peak_rate_per_mile"],
        tail_rate_per_km,
        rates["tail_peak_rate_per_hour"],
        tail_return_miles,
        tail_return_km,
        tail_return_hours,
    )
    logger.info(
        (
            "Human highD safety-critical event Y>=%.6g: "
            "P(Y>=level | Y>u)=%.6g P(Y>=level)=%.6g "
            "rate=%.6g/mile %.6g/km %.6g/hour "
            "return=%.6g miles %.6g km %.6g hours"
        ),
        float(collision_level),
        collision_summary["tail_conditional_probability_above_collision_level"],
        float(collision_probability_per_peak),
        collision_summary["highd_safety_critical_intensity_per_mile"],
        collision_summary["highd_safety_critical_intensity_per_km"],
        collision_summary["highd_safety_critical_intensity_per_hour"],
        collision_summary["highd_safety_critical_return_period_miles"],
        collision_summary["highd_safety_critical_return_period_km"],
        collision_summary["highd_safety_critical_return_period_hours"],
    )


# ── main ──


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    _fit_evt_model(DEFAULT_CONFIG_PATH)
    cfg = load_config(DEFAULT_CONFIG_PATH)
    paths = _paths(cfg, DEFAULT_CONFIG_PATH)
    exposure_cfg = cfg["following_exposure"]
    peak_cfg = cfg["following_evt_peak"]
    decluster_cfg = exposure_cfg["declustering"]

    model = load_evt_model(paths["evt_model"])
    scores = _load_scored_events(paths["score_csv"], model)
    exposure_rows = _load_exposure_csv(paths["exposure_csv"])

    total_miles = sum(row["following_ego_miles"] for row in exposure_rows)
    total_hours = sum(row["following_ego_hours"] for row in exposure_rows)
    total_km = total_miles * KM_PER_MILE
    all_vehicle_miles = sum(row["all_vehicle_miles"] for row in exposure_rows)
    all_vehicle_hours = sum(row["all_vehicle_hours"] for row in exposure_rows)

    target_fps = float(cfg["sampling"]["target_fps"])
    group_keys = tuple(str(item) for item in decluster_cfg["group_keys"])
    run_length_seconds = float(decluster_cfg["run_length_seconds"])
    all_peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
    )
    peaks = [
        peak
        for peak in all_peaks
        if float(peak["y_long_max"]) > float(model.u)
    ]

    rates = peak_rate_summary(
        total_exposure_miles=total_miles,
        total_exposure_hours=total_hours,
        num_independent_tail_peaks=len(peaks),
    )
    all_vehicle_rates = peak_rate_summary(
        total_exposure_miles=all_vehicle_miles,
        total_exposure_hours=all_vehicle_hours,
        num_independent_tail_peaks=len(peaks),
    )
    tail_peak_rate_per_km = rates["tail_peak_rate_per_mile"] / KM_PER_MILE

    collision_level, collision_level_meta = _collision_level_from_config(
        peak_cfg,
        tail_peak_rate_per_km=tail_peak_rate_per_km,
        model=model,
    )
    tail_conditional_probability_at_collision = gpd_conditional_survival(
        collision_level,
        u=float(model.u),
        xi=float(model.xi),
        beta=float(model.beta),
    )
    collision_probability_per_peak = float(model.survival(collision_level))
    collision_summary = collision_distance_summary(
        tail_peak_rate_per_mile=rates["tail_peak_rate_per_mile"],
        tail_peak_rate_per_hour=rates["tail_peak_rate_per_hour"],
        tail_conditional_probability_above_collision_level=(
            tail_conditional_probability_at_collision
        ),
    )
    all_vehicle_collision_summary = collision_distance_summary(
        tail_peak_rate_per_mile=all_vehicle_rates["tail_peak_rate_per_mile"],
        tail_peak_rate_per_hour=all_vehicle_rates["tail_peak_rate_per_hour"],
        tail_conditional_probability_above_collision_level=(
            tail_conditional_probability_at_collision
        ),
    )

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    _remove_obsolete_recording_plots(figure_dir)

    y_all_peaks = np.asarray(
        [row["y_long_max"] for row in all_peaks], dtype=np.float64
    )
    _write_return_level_plot(
        figure_dir,
        values=y_all_peaks,
        model=model,
        total_exposure_km=total_km,
        collision_critical_level=collision_level,
        collision_return_period_km=collision_summary[
            "highd_collision_return_period_km"
        ],
        bootstrap_samples=int(peak_cfg["bootstrap_samples"]),
        distance_min_km=float(peak_cfg["distance_plot_min_km"]),
        distance_max_km=float(peak_cfg["distance_plot_max_km"]),
        random_seed=int(peak_cfg["random_seed"]),
    )

    summary = {
        "evt_model_path": str(paths["evt_model"]),
        "evt_tail_threshold_u": float(model.u),
        "collision_critical_level": collision_level,
        "collision_critical_level_mode": collision_level_meta[
            "collision_critical_level_mode"
        ],
        "following_ego_miles": total_miles,
        "following_ego_hours": total_hours,
        "all_vehicle_miles": all_vehicle_miles,
        "all_vehicle_hours": all_vehicle_hours,
        "ego_mile_fraction_of_all_vehicle": (
            float(total_miles / all_vehicle_miles)
            if all_vehicle_miles > 0.0
            else 0.0
        ),
        "num_tail_events_before_declustering": int(
            np.sum(
                pd.to_numeric(scores["y_long"], errors="coerce") > float(model.u)
            )
        ),
        "num_independent_peaks_before_tail_filter": int(len(all_peaks)),
        "num_independent_tail_peaks": int(len(peaks)),
        "tail_peak_rate_per_mile": rates["tail_peak_rate_per_mile"],
        "tail_peak_rate_per_hour": rates["tail_peak_rate_per_hour"],
        "tail_peak_rate_per_km": rates["tail_peak_rate_per_mile"] / KM_PER_MILE,
        "tail_peak_rate_per_all_vehicle_mile": all_vehicle_rates[
            "tail_peak_rate_per_mile"
        ],
        "tail_peak_rate_per_all_vehicle_hour": all_vehicle_rates[
            "tail_peak_rate_per_hour"
        ],
        "tail_threshold_probability_per_independent_peak": float(
            model.exceedance_rate
        ),
        "tail_threshold_return_period_miles": _return_period(
            rates["tail_peak_rate_per_mile"]
        ),
        "tail_threshold_return_period_km": _return_period(
            rates["tail_peak_rate_per_mile"] / KM_PER_MILE
        ),
        "tail_threshold_return_period_hours": _return_period(
            rates["tail_peak_rate_per_hour"]
        ),
        "collision_level_tail_conditional_probability": collision_summary[
            "tail_conditional_probability_above_collision_level"
        ],
        "safety_critical_level_tail_conditional_probability": collision_summary[
            "tail_conditional_probability_above_safety_critical_level"
        ],
        "collision_level_probability_per_independent_peak": (
            collision_probability_per_peak
        ),
        "safety_critical_level_probability_per_independent_peak": (
            collision_probability_per_peak
        ),
        "highd_safety_critical_intensity_per_mile": collision_summary[
            "highd_safety_critical_intensity_per_mile"
        ],
        "highd_safety_critical_return_period_miles": collision_summary[
            "highd_safety_critical_return_period_miles"
        ],
        "highd_safety_critical_intensity_per_km": collision_summary[
            "highd_safety_critical_intensity_per_km"
        ],
        "highd_safety_critical_return_period_km": collision_summary[
            "highd_safety_critical_return_period_km"
        ],
        "highd_safety_critical_intensity_per_hour": collision_summary[
            "highd_safety_critical_intensity_per_hour"
        ],
        "highd_safety_critical_return_period_hours": collision_summary[
            "highd_safety_critical_return_period_hours"
        ],
        "highd_collision_intensity_per_mile": collision_summary[
            "highd_collision_intensity_per_mile"
        ],
        "highd_collision_return_period_miles": collision_summary[
            "highd_collision_return_period_miles"
        ],
        "highd_collision_intensity_per_km": collision_summary[
            "highd_collision_intensity_per_km"
        ],
        "highd_collision_return_period_km": collision_summary[
            "highd_collision_return_period_km"
        ],
        "highd_collision_intensity_per_hour": collision_summary[
            "highd_collision_intensity_per_hour"
        ],
        "highd_collision_return_period_hours": collision_summary[
            "highd_collision_return_period_hours"
        ],
        "collision_intensity_per_all_vehicle_mile": (
            all_vehicle_collision_summary["highd_collision_intensity_per_mile"]
        ),
        "safety_critical_intensity_per_all_vehicle_mile": (
            all_vehicle_collision_summary[
                "highd_safety_critical_intensity_per_mile"
            ]
        ),
        "collision_return_period_all_vehicle_miles": (
            all_vehicle_collision_summary[
                "highd_collision_return_period_miles"
            ]
        ),
        "safety_critical_return_period_all_vehicle_miles": (
            all_vehicle_collision_summary[
                "highd_safety_critical_return_period_miles"
            ]
        ),
        "collision_intensity_per_all_vehicle_hour": (
            all_vehicle_collision_summary[
                "highd_collision_intensity_per_hour"
            ]
        ),
        "safety_critical_intensity_per_all_vehicle_hour": (
            all_vehicle_collision_summary[
                "highd_safety_critical_intensity_per_hour"
            ]
        ),
        "collision_return_period_all_vehicle_hours": (
            all_vehicle_collision_summary[
                "highd_collision_return_period_hours"
            ]
        ),
        "safety_critical_return_period_all_vehicle_hours": (
            all_vehicle_collision_summary[
                "highd_safety_critical_return_period_hours"
            ]
        ),
        "declustering_run_length_seconds": run_length_seconds,
        "declustering_group_keys": list(group_keys),
        "declustering_representative": str(
            decluster_cfg["representative"]
        ),
        "exposure_denominator": "following_ego_miles",
    }

    write_csv(output_dir / "highd_independent_tail_peaks.csv", peaks)
    write_json(output_dir / "highd_exposure_summary.json", summary)
    _log_human_exposure_metrics(
        model=model,
        rates=rates,
        collision_level=collision_level,
        collision_probability_per_peak=collision_probability_per_peak,
        collision_summary=collision_summary,
    )
    logger.info(
        "Wrote highD exposure summary to %s | miles=%.6f peaks=%d rate/mile=%.6g",
        output_dir,
        total_miles,
        len(peaks),
        rates["tail_peak_rate_per_mile"],
    )


if __name__ == "__main__":
    main()
