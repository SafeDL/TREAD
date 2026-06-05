#!/usr/bin/env python3
"""Select highD cut-in long-tail contexts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.tail_context_selection import run_tail_context_selection
from utils.highd_cutin import (
    filter_cutin_start_context_rows,
    load_highd_cutin_event_context_cache,
)


CUTIN_TAIL_CONTEXT_CONFIG = {
    "event_context_cache_path": (
        ROOT / "results" / "highd_events" / "cutin_event_contexts.npz"
    ),
    "tail_context_path": (
        ROOT / "results" / "highd_cutin_tail" / "contexts" / "tail_contexts.npz"
    ),
    "independent_tail_peaks_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "exposure"
        / "highd_independent_tail_peaks.csv"
    ),
    "evt_model_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "evt"
        / "cutin_peak_evt_model.json"
    ),
    "evt_summary_path": (
        ROOT
        / "results"
        / "highd_cutin_tail"
        / "evt"
        / "cutin_peak_evt_summary.json"
    ),
    "scenario": "cut_in",
    "risk_value_key": "y_cutin",
    "required_history_steps": 25,
    "context_output_keys": (
        "cross_frame",
        "cutin_start_frame",
        "cutin_end_frame",
        "source_lane",
        "target_lane",
        "completion_gap",
        "post_cutin_min_gap",
        "post_cutin_min_ttc",
        "cutin_gap",
        "cutin_ttc",
        "cutin_time_headway",
        "cutin_lateral_time_gap",
        "max_post_cutin_drac",
        "safety_distance",
        "safety_distance_deficit",
        "cutin_safety_risk_score",
        "post_longitudinal_risk_score",
        "cutin_duration_seconds",
        "cross_lateral_offset",
        "min_abs_lateral_offset",
        "max_abs_lateral_velocity",
        "max_lateral_approach_speed",
        "final_abs_lateral_offset",
        "is_cutin",
        "is_front_cutin",
    ),
    "context_key_dtypes": {
        "y_cutin": "float",
        "cross_frame": "int",
        "cutin_start_frame": "int",
        "cutin_end_frame": "int",
        "source_lane": "int",
        "target_lane": "int",
        "completion_gap": "float",
        "post_cutin_min_gap": "float",
        "post_cutin_min_ttc": "float",
        "cutin_gap": "float",
        "cutin_ttc": "float",
        "cutin_time_headway": "float",
        "cutin_lateral_time_gap": "float",
        "max_post_cutin_drac": "float",
        "safety_distance": "float",
        "safety_distance_deficit": "float",
        "cutin_safety_risk_score": "float",
        "post_longitudinal_risk_score": "float",
        "cutin_duration_seconds": "float",
        "cross_lateral_offset": "float",
        "min_abs_lateral_offset": "float",
        "max_abs_lateral_velocity": "float",
        "max_lateral_approach_speed": "float",
        "final_abs_lateral_offset": "float",
        "is_cutin": "float",
        "is_front_cutin": "float",
    },
    "context_loader": load_highd_cutin_event_context_cache,
    "row_filter": filter_cutin_start_context_rows,
    "fit_evt_hint": "process_highD/scripts/fit_cutin_peak_evt.py",
    "estimate_exposure_hint": "process_highD/scripts/estimate_cutin_exposure.py",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_tail_context_selection(CUTIN_TAIL_CONTEXT_CONFIG)


if __name__ == "__main__":
    main()
