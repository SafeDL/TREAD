#!/usr/bin/env python3
"""Fit a POT/GPD EVT model for highD event-level longitudinal risk."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import setup_logging
from utils.evt import RETURN_PERIODS, fit_evt_model
from utils.io import write_csv, write_json

from process_highD.scripts.select_tail_contexts import (
    _build_rows,
    _load_events,
    _runtime_config,
    _score_rows,
)


SCRIPT_DEFAULTS: dict[str, Any] = {
    "raw_dir": ROOT / "highD_dataset" / "Matlab" / "data",
    "events_csv": ROOT / "results" / "highd_events" / "events.csv",
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
    "selected_method": "B",
    "min_exceedances": 20,
    "max_tail_fraction": 0.25,
    "bootstrap_samples": 200,
    "random_seed": 42,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _all_valid_following_events():
    events = _load_events(Path(SCRIPT_DEFAULTS["events_csv"]))
    events = events.copy()
    events["split_index"] = -1
    return events.reset_index(drop=True)


def _score_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "event_index",
        "split_index",
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
    return [{key: row[key] for key in keys if key in row} for row in rows]


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


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    cfg = _runtime_config()
    events = _all_valid_following_events()
    rows, skipped = _build_rows(
        events,
        cfg,
        Path(SCRIPT_DEFAULTS["raw_dir"]),
    )
    _score_rows(rows)
    y_long = np.asarray([row["y_long"] for row in rows], dtype=np.float64)
    model = fit_evt_model(
        y_long,
        selected_method=str(SCRIPT_DEFAULTS["selected_method"]),
        min_exceedances=int(SCRIPT_DEFAULTS["min_exceedances"]),
        max_tail_fraction=float(SCRIPT_DEFAULTS["max_tail_fraction"]),
        bootstrap_samples=int(SCRIPT_DEFAULTS["bootstrap_samples"]),
        random_seed=int(SCRIPT_DEFAULTS["random_seed"]),
    )

    model_path = Path(SCRIPT_DEFAULTS["evt_model_path"])
    model.to_json(model_path)
    write_csv(Path(SCRIPT_DEFAULTS["evt_scores_path"]), _score_table(rows))
    write_csv(
        Path(SCRIPT_DEFAULTS["threshold_stability_path"]),
        model.threshold_candidates,
    )
    _write_arrays(Path(SCRIPT_DEFAULTS["evt_arrays_path"]), model, y_long)
    write_json(
        model_path.with_name("longitudinal_evt_summary.json"),
        {
            "events_csv": str(SCRIPT_DEFAULTS["events_csv"]),
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
            "return_levels": model.return_levels,
            "return_level_ci": model.return_level_ci,
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
