"""Synthetic sanity tests for context feature extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_deepevt.src.features import (  # noqa: E402
    CUTIN_FEATURE_KEYS,
    FOLLOWING_FEATURE_KEYS,
    LEAKAGE_KEYS,
    extract_context,
)
from tread_deepevt.src.window_rebuild import NUM_STATE_FEATURES  # noqa: E402


def _synthetic_following_window(T: int = 128) -> np.ndarray:
    """Construct ego-lead trajectory with known gap/relative speed."""
    ego_x = np.arange(T, dtype=np.float32) * 0.8           # 20 m/s -> 0.8 m / 0.04 s
    lead_x = ego_x + 20.0                                  # constant 20m initial gap
    ego_vx = np.full(T, 20.0, dtype=np.float32)
    lead_vx = np.full(T, 18.0, dtype=np.float32)           # slower -> closing
    states = np.zeros((T, 2, NUM_STATE_FEATURES), dtype=np.float32)
    states[:, 0, 0] = ego_x
    states[:, 1, 0] = lead_x
    states[:, 0, 2] = ego_vx
    states[:, 1, 2] = lead_vx
    return states


def test_following_context_initial_values_and_order():
    states = _synthetic_following_window()
    event_row = pd.Series({
        "event_id": "fol_test",
        "event_type": "following",
        "start_frame": 0, "end_frame": 127, "anchor_frame": 64,
    })
    config = {
        "prefix": {"prefix_steps": 25},
        "sampling": {"target_fps": 25},
        "risk": {"epsilon": 1e-6},
        "features": {"forbid_risk_leakage": True},
    }
    vec, keys = extract_context(
        "following", states, event_row, config,
        ego_length=4.5, target_length=4.5,
    )
    assert tuple(keys) == FOLLOWING_FEATURE_KEYS
    assert vec.shape == (len(FOLLOWING_FEATURE_KEYS),)
    f = dict(zip(keys, vec.tolist()))
    # gap_0 = 20 - 4.5 = 15.5
    assert abs(f["gap_0"] - 15.5) < 1e-3
    # relative speed = 20 - 18 = 2
    assert abs(f["relative_speed_0"] - 2.0) < 1e-3
    # ensure no leakage key present
    assert not any(k in LEAKAGE_KEYS for k in keys)


def test_cutin_context_initial_values_and_order():
    T = 128
    ego_x = np.arange(T, dtype=np.float32) * 0.8
    tgt_x = ego_x + 15.0
    ego_vx = np.full(T, 20.0, dtype=np.float32)
    tgt_vx = np.full(T, 21.0, dtype=np.float32)
    states = np.zeros((T, 2, NUM_STATE_FEATURES), dtype=np.float32)
    states[:, 0, 0] = ego_x; states[:, 1, 0] = tgt_x
    states[:, 0, 2] = ego_vx; states[:, 1, 2] = tgt_vx
    states[:, 1, 1] = 2.0     # target lateral offset
    states[:, 1, 3] = 0.5     # target lateral speed
    event_row = pd.Series({
        "event_id": "cin_test", "event_type": "cut_in",
        "start_frame": 0, "end_frame": 127,
        "cutin_start_frame": 30, "cutin_end_frame": 60,
        "cutin_duration": 1.2, "anchor_frame": 45,
    })
    config = {
        "prefix": {"prefix_steps": 25},
        "sampling": {"target_fps": 25},
        "risk": {"epsilon": 1e-6},
        "features": {"forbid_risk_leakage": True},
    }
    vec, keys = extract_context(
        "cut_in", states, event_row, config,
        ego_length=4.5, target_length=4.5,
    )
    assert tuple(keys) == CUTIN_FEATURE_KEYS
    f = dict(zip(keys, vec.tolist()))
    assert abs(f["initial_dx"] - (15.0 - 4.5)) < 1e-3
    assert abs(f["initial_dy"] - 2.0) < 1e-3
    assert abs(f["relative_speed_0"] - (-1.0)) < 1e-3
    assert abs(f["prefix_lateral_speed_mean"] - 0.5) < 1e-3
    assert abs(f["planned_cutin_duration"] - 1.2) < 1e-3
    assert not any(k in LEAKAGE_KEYS for k in keys)
