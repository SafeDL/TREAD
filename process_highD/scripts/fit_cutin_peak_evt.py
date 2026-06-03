#!/usr/bin/env python3
"""Fit a POT/GPD EVT model to declustered highD cut-in risk peaks."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import setup_logging
from process_highD.scripts.fit_following_peak_evt import _write_evt_diagnostic_plots
from process_highD.src.io_utils import load_config, resolve_data_path
from utils.evt import fit_evt_model
from utils.highd_exposure import extract_independent_peaks
from utils.io import write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
SCRIPT_DEFAULTS: dict[str, Any] = {
    "log_level": "INFO",
    "min_exceedances": 20,
    "max_tail_fraction": 0.25,
    "max_threshold_candidates": 400,
    "min_threshold_exceedance_rate": 0.10,
    "random_seed": 42,
}
logger = logging.getLogger(__name__)


def _paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths_cfg = cfg["paths"]
    highd_events_dir = resolve_data_path(paths_cfg["output_dir"], config_path)
    cutin_evt_cfg = cfg["cutin_evt_peak"]
    model_path = resolve_data_path(cutin_evt_cfg["model_path"], config_path)
    return {
        "score_csv": highd_events_dir / "cutin_event_scores.csv",
        "model": model_path,
        "summary": resolve_data_path(cutin_evt_cfg["summary_path"], config_path),
        "figure_dir": model_path.parent / "figures",
    }


def _load_scored_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"highD cut-in score cache not found: {path}. "
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
        "y_cutin",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    return frame


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    cfg = load_config(DEFAULT_CONFIG_PATH)
    paths = _paths(cfg, DEFAULT_CONFIG_PATH)
    peak_cfg = cfg["cutin_evt_peak"]
    decluster_cfg = peak_cfg["declustering"]
    scores = _load_scored_events(paths["score_csv"])

    target_fps = float(cfg["sampling"]["target_fps"])
    group_keys = tuple(str(item) for item in decluster_cfg["group_keys"])
    run_length_seconds = float(decluster_cfg["run_length_seconds"])
    peaks = extract_independent_peaks(
        scores,
        run_length_seconds=run_length_seconds,
        fps=target_fps,
        group_keys=group_keys,
        score_column="y_cutin",
    )
    if not peaks:
        raise RuntimeError("No independent highD cut-in risk peaks were extracted")

    y_peaks = np.asarray([row["y_cutin_max"] for row in peaks], dtype=np.float64)
    model = fit_evt_model(
        y_peaks,
        min_exceedances=int(SCRIPT_DEFAULTS["min_exceedances"]),
        max_tail_fraction=SCRIPT_DEFAULTS["max_tail_fraction"],
        max_threshold_candidates=int(SCRIPT_DEFAULTS["max_threshold_candidates"]),
        min_threshold_exceedance_rate=float(
            SCRIPT_DEFAULTS["min_threshold_exceedance_rate"]
        ),
        bootstrap_samples=int(peak_cfg["bootstrap_samples"]),
        random_seed=int(SCRIPT_DEFAULTS["random_seed"]),
    )

    tail_peaks = int(np.sum(y_peaks > float(model.u)))
    collision_critical_level = float(peak_cfg["collision_critical_level"])
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model.to_json(paths["model"], model_type="gpd_pot_cutin_risk")

    figures = _write_evt_diagnostic_plots(
        paths["figure_dir"],
        model=model,
        values=y_peaks,
        collision_critical_level=collision_critical_level,
        risk_variable="Y_cutin",
        histogram_filename="peak_evt_y_cutin_histogram.png",
        histogram_key="peak_y_cutin_histogram",
    )
    write_json(
        paths["summary"],
        {
            "model_path": str(paths["model"]),
            "score_csv": str(paths["score_csv"]),
            "model_type": "gpd_pot_cutin_independent_peak_risk",
            "risk_variable": "y_cutin",
            "num_independent_peaks": int(len(peaks)),
            "num_tail_peaks": tail_peaks,
            "u": float(model.u),
            "xi": float(model.xi),
            "beta": float(model.beta),
            "exceedance_rate": float(model.exceedance_rate),
            "collision_critical_level": collision_critical_level,
            "collision_critical_level_mode": "fixed_y_cutin",
            "return_levels": model.return_levels,
            "return_level_ci": model.return_level_ci,
            "declustering_run_length_seconds": run_length_seconds,
            "declustering_group_keys": list(group_keys),
            "figures": figures,
        },
    )
    logger.info(
        "Saved cut-in peak EVT model to %s | peaks=%d tail_peaks=%d u=%.6f",
        paths["model"],
        len(peaks),
        tail_peaks,
        model.u,
    )


if __name__ == "__main__":
    main()
