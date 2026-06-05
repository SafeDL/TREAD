#!/usr/bin/env python3
"""Fit a POT/GPD EVT model to highD following risk peaks."""
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
    "y_long",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fit_highd_peak_evt(
        config_path=DEFAULT_CONFIG_PATH,
        score_filename="following_event_scores.csv",
        peak_config_key="long_evt_peak",
        declustering_config_path=("exposure", "declustering"),
        required_columns=REQUIRED_COLUMNS,
        score_column="y_long",
        peak_value_key="y_long_max",
        scenario_label="following",
        summary_model_type="gpd_pot_longitudinal_independent_peak_risk",
        collision_critical_level_mode="fixed_y_long",
    )


if __name__ == "__main__":
    main()
