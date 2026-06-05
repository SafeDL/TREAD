"""Shared POT/GPD peak EVT fitting for highD scenarios."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from process_highD.src.evt_diagnostics import write_evt_diagnostic_plots
from process_highD.src.io_utils import load_config, resolve_data_path
from utils.evt import fit_evt_model
from utils.highd_exposure import extract_independent_peaks
from utils.io import write_json


EVT_FIT_DEFAULTS: dict[str, Any] = {
    "min_exceedances": 20,
    "max_tail_fraction": 0.25,
    "max_threshold_candidates": 400,
    "min_threshold_exceedance_rate": 0.10,
    "random_seed": 42,
}
logger = logging.getLogger(__name__)


def _score_cache_path(cfg: dict[str, Any], config_path: Path, filename: str) -> Path:
    highd_events_dir = resolve_data_path(cfg["paths"]["output_dir"], config_path)
    return highd_events_dir / filename


def _load_scored_events(
    path: Path,
    *,
    required_columns: set[str],
    scenario_label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD {scenario_label} score cache not found: {path}. "
            "Run process_highD/scripts/extract_highd_events.py first."
        )
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    return frame


def fit_highd_peak_evt(
    *,
    config_path: Path,
    score_filename: str,
    peak_config_key: str,
    declustering_config_path: tuple[str, ...],
    required_columns: set[str],
    score_column: str,
    peak_value_key: str,
    scenario_label: str,
    summary_model_type: str,
    collision_critical_level_mode: str,
    model_type: str | None = None,
    summary_extra: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    score_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> None:
    cfg = load_config(config_path)
    peak_cfg = cfg[peak_config_key]
    declustering_config = cfg
    for key in declustering_config_path:
        declustering_config = declustering_config[key]
    score_csv = _score_cache_path(cfg, config_path, score_filename)
    model_path = resolve_data_path(peak_cfg["model_path"], config_path)
    summary_path = resolve_data_path(peak_cfg["summary_path"], config_path)
    figure_dir = model_path.parent / "figures"

    scores = _load_scored_events(
        score_csv,
        required_columns=required_columns,
        scenario_label=scenario_label,
    )
    if score_filter is not None:
        before = len(scores)
        scores = score_filter(scores).copy()
        if scores.empty:
            raise RuntimeError(
                f"No highD {scenario_label} score rows remain after score_filter"
            )
        removed = before - len(scores)
        if removed:
            logger.info(
                "Filtered %d highD %s score rows before EVT fitting",
                removed,
                scenario_label,
            )
    target_fps = float(cfg["sampling"]["target_fps"])
    group_keys = tuple(str(item) for item in declustering_config["group_keys"])
    run_length_seconds = float(declustering_config["run_length_seconds"])
    peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
        score_column=score_column,
    )
    if not peaks:
        raise RuntimeError(
            f"No independent highD {scenario_label} risk peaks were extracted"
        )

    y_peaks = np.asarray([row[peak_value_key] for row in peaks], dtype=np.float64)
    model = fit_evt_model(
        y_peaks,
        min_exceedances=int(EVT_FIT_DEFAULTS["min_exceedances"]),
        max_tail_fraction=EVT_FIT_DEFAULTS["max_tail_fraction"],
        max_threshold_candidates=int(EVT_FIT_DEFAULTS["max_threshold_candidates"]),
        min_threshold_exceedance_rate=float(
            EVT_FIT_DEFAULTS["min_threshold_exceedance_rate"]
        ),
        bootstrap_samples=int(peak_cfg["bootstrap_samples"]),
        random_seed=int(EVT_FIT_DEFAULTS["random_seed"]),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_type is None:
        model.to_json(model_path)
    else:
        model.to_json(model_path, model_type=model_type)

    collision_critical_level = float(peak_cfg["collision_critical_level"])
    figures = write_evt_diagnostic_plots(
        figure_dir,
        model=model,
        values=y_peaks,
        collision_critical_level=collision_critical_level,
        **(plot_kwargs or {}),
    )
    tail_peaks = int(np.sum(y_peaks > float(model.u)))
    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "score_csv": str(score_csv),
        "model_type": summary_model_type,
        "num_independent_peaks": int(len(peaks)),
        "num_tail_peaks": tail_peaks,
        "u": float(model.u),
        "xi": float(model.xi),
        "beta": float(model.beta),
        "exceedance_rate": float(model.exceedance_rate),
        "collision_critical_level": collision_critical_level,
        "collision_critical_level_mode": collision_critical_level_mode,
        "return_levels": model.return_levels,
        "return_level_ci": model.return_level_ci,
        "declustering_run_length_seconds": run_length_seconds,
        "declustering_group_keys": list(group_keys),
        "figures": figures,
    }
    if summary_extra:
        summary.update(summary_extra)
    write_json(summary_path, summary)
    logger.info(
        "Saved %s peak EVT model to %s | peaks=%d tail_peaks=%d u=%.6f",
        scenario_label,
        model_path,
        len(peaks),
        tail_peaks,
        model.u,
    )
