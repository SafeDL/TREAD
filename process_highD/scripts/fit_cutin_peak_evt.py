#!/usr/bin/env python3
"""Fit a POT/GPD EVT model to highD cut-in risk peaks."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.evt_fitting import fit_highd_peak_evt


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
REQUIRED_COLUMNS = {
    "event_id",
    "recording_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "is_cutin",
    "y_cutin",
}


def _semantic_cutin_scores(frame):
    return frame[frame["is_cutin"].astype(float) >= 0.5]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fit_highd_peak_evt(
        config_path=DEFAULT_CONFIG_PATH,
        score_filename="cutin_event_scores.csv",
        peak_config_key="cutin_evt_peak",
        declustering_config_path=("cutin_evt_peak", "declustering"),
        required_columns=REQUIRED_COLUMNS,
        score_column="y_cutin",
        peak_value_key="y_cutin_max",
        scenario_label="cut-in",
        model_type="gpd_pot_cutin_risk",
        summary_model_type="gpd_pot_cutin_independent_peak_risk",
        collision_critical_level_mode="fixed_y_cutin",
        summary_extra={"risk_variable": "y_cutin"},
        score_filter=_semantic_cutin_scores,
        plot_kwargs={
            "risk_variable": "Y_cutin",
            "histogram_filename": "peak_evt_y_cutin_histogram.png",
            "histogram_key": "peak_y_cutin_histogram",
        },
    )


if __name__ == "__main__":
    main()
