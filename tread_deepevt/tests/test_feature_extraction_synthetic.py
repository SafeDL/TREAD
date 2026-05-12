"""Synthetic sanity tests for context feature extraction in ego-initial frame."""
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
    extract_context_with_canonical,
)
from tread_deepevt.src.window_rebuild import NUM_STATE_FEATURES  # noqa: E402


def _following_states_in_ego_frame(T: int = 128) -> np.ndarray:
    """Build ego-initial frame states: ego at origin, lead 20m ahead, closing."""
    states = np.zeros((T, 2, NUM_STATE_FEATURES), dtype=np.float32)
    states[:, 0, 2] = 20.0     # ego vx
    states[:, 1, 0] = 20.0     # lead x
    states[:, 1, 2] = 18.0     # lead vx
    return states


def _cutin_states_in_ego_frame(T: int = 128) -> np.ndarray:
    states = np.zeros((T, 2, NUM_STATE_FEATURES), dtype=np.float32)
    states[:, 0, 2] = 20.0
    states[:, 1, 0] = 15.0      # target dx = 15
    states[:, 1, 1] = 2.0       # target dy = 2
    states[:, 1, 2] = 21.0
    states[:, 1, 3] = 0.5       # lateral speed
    return states


def test_following_context_initial_values_and_order():
    states = _following_states_in_ego_frame()
    event_row = pd.Series({
        "event_id": "fol_test",
        "event_type": "following",
        "start_frame": 0, "end_frame": 127, "anchor_frame": 64,
    })
    config = {
        "prefix": {"prefix_steps": 1},
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
    # gap_0 = 20 - 0.5*(4.5 + 4.5) = 15.5
    assert abs(f["gap_0"] - 15.5) < 1e-3
    assert abs(f["relative_speed_0"] - 2.0) < 1e-3
    # initial-context: 不含 prefix-derived 统计量
    assert "gap_slope_prefix" not in keys
    assert "closing_speed_max_prefix" not in keys
    assert "lead_accel_min_prefix" not in keys
    assert not any(k in LEAKAGE_KEYS for k in keys)


def test_cutin_context_initial_values_and_order():
    states = _cutin_states_in_ego_frame()
    event_row = pd.Series({
        "event_id": "cin_test", "event_type": "cut_in",
        "start_frame": 0, "end_frame": 127,
        "cutin_start_frame": 30, "cutin_end_frame": 60,
        "cutin_duration": 1.2, "anchor_frame": 45,
        "source_lane": 2, "target_lane": 3,
    })
    config = {
        "prefix": {"prefix_steps": 1},
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
    # initial-context: 不含 prefix-derived 统计量或 future-leaking 字段
    assert "prefix_lateral_speed_mean" not in keys
    assert "planned_cutin_duration" not in keys
    assert "raw_event_duration" not in keys
    assert not any(k in LEAKAGE_KEYS for k in keys)


def test_extract_context_with_canonical_is_consistent():
    states = _following_states_in_ego_frame()
    event_row = pd.Series({
        "event_id": "fol_test",
        "event_type": "following",
        "start_frame": 0, "end_frame": 127, "anchor_frame": 64,
        "source_lane": 3, "target_lane": 3,
    })
    config = {
        "prefix": {"prefix_steps": 1},
        "sampling": {"target_fps": 25},
        "risk": {"epsilon": 1e-6},
        "features": {"forbid_risk_leakage": True},
    }
    vec, keys, canonical = extract_context_with_canonical(
        "following", states, event_row, config,
        ego_length=4.5, ego_width=1.8,
        target_length=4.5, target_width=1.8,
    )
    # DeepEVT context gap_0 must equal canonical.initial_gap
    g0 = dict(zip(keys, vec.tolist()))["gap_0"]
    assert abs(g0 - canonical.initial_gap) < 1e-5
    # relative speeds agree
    assert abs(dict(zip(keys, vec.tolist()))["relative_speed_0"] - canonical.relative_speed_0) < 1e-5
    # target_center_x0 = gap_0 + 0.5*(L_ego + L_target)
    expected_center_x = g0 + 0.5 * (4.5 + 4.5)
    assert abs(canonical.target_center_x0 - expected_center_x) < 1e-5
    assert canonical.same_lane_initial is True
