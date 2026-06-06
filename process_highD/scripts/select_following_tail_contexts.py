#!/usr/bin/env python3
"""Select highD car-following long-tail contexts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.tail_context_selection import run_tail_context_selection
from utils.highd_longitudinal import load_highd_event_context_cache


FOLLOWING_TAIL_CONTEXT_CONFIG = {
    "event_context_cache_path": (
        ROOT / "results" / "highd_events" / "following_event_contexts.npz"
    ),
    "tail_context_path": (
        ROOT / "results" / "highd_following_tail" / "contexts" / "tail_contexts.npz"
    ),
    "independent_tail_peaks_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "exposure"
        / "highd_independent_tail_peaks.csv"
    ),
    "evt_model_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "evt"
        / "longitudinal_peak_evt_model.json"
    ),
    "evt_summary_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "evt"
        / "longitudinal_peak_evt_summary.json"
    ),
    "scenario": "following",
    "risk_value_key": "y_long",
    "context_key_dtypes": {
        "y_long": "float",
    },
    "context_loader": load_highd_event_context_cache,
    "fit_evt_hint": "process_highD/scripts/fit_following_peak_evt.py",
    "estimate_exposure_hint": (
        "process_highD/scripts/estimate_following_exposure.py"
    ),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_tail_context_selection(FOLLOWING_TAIL_CONTEXT_CONFIG)


if __name__ == "__main__":
    main()
