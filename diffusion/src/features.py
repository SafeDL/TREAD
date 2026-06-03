"""Leakage-safe history context features for action diffusion."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


FOLLOWING_CONTEXT_KEYS: Tuple[str, ...] = (
    "ego_vx_current",
    "lead_vx_current",
    "gap_current",
    "ego_ax_current",
    "lead_ax_current",
    "gap_change_rate",
    "min_gap_in_prefix",
    "max_closing_speed_in_prefix",
)

CUTIN_CONTEXT_KEYS: Tuple[str, ...] = (
    "ego_vx_current",
    "target_vx_current",
    "target_vy_current",
    "gap_current",
    "lateral_offset_current",
    "ego_ax_current",
    "target_ax_current",
    "target_ay_current",
    "gap_change_rate",
    "lateral_offset_change_rate",
    "relative_vx_trend",
    "relative_vy_trend",
    "target_yaw_rate_current",
    "target_lateral_jerk_current",
    "min_gap_in_prefix",
    "min_abs_lateral_offset_in_prefix",
    "max_abs_target_lateral_velocity_in_prefix",
    "lateral_offset_range_in_prefix",
)


def extract_following_context(
    history: np.ndarray,
    ego_length: float,
    lead_length: float,
    dt: float,
) -> Dict[str, float]:
    """Extract current/history-only car-following context in ego-current frame."""
    states = np.asarray(history, dtype=np.float32)
    ego = states[:, 0]
    lead = states[:, 1]
    gaps = lead[:, 0] - ego[:, 0] - 0.5 * (ego_length + lead_length)
    rel_speed = ego[:, 2] - lead[:, 2]
    elapsed = max((len(states) - 1) * float(dt), 1e-6)
    return {
        "ego_vx_current": float(ego[-1, 2]),
        "lead_vx_current": float(lead[-1, 2]),
        "gap_current": float(gaps[-1]),
        "ego_ax_current": float(ego[-1, 4]),
        "lead_ax_current": float(lead[-1, 4]),
        "gap_change_rate": float((gaps[-1] - gaps[0]) / elapsed),
        "min_gap_in_prefix": float(np.min(gaps)),
        "max_closing_speed_in_prefix": float(np.maximum(rel_speed, 0.0).max()),
    }


def extract_cutin_context(
    history: np.ndarray,
    ego_length: float,
    adv_length: float,
    dt: float,
) -> Dict[str, float]:
    states = np.asarray(history, dtype=np.float32)
    ego = states[:, 0]
    target = states[:, 1]
    gaps = target[:, 0] - ego[:, 0] - 0.5 * (ego_length + adv_length)
    lateral = target[:, 1] - ego[:, 1]
    rel_vx = ego[:, 2] - target[:, 2]
    rel_vy = ego[:, 3] - target[:, 3]
    elapsed = max((len(states) - 1) * float(dt), 1.0e-6)
    eps = 1.0e-6
    heading = np.unwrap(np.arctan2(target[:, 3], np.maximum(target[:, 2], eps)))
    if len(heading) >= 2:
        yaw_rate = float((heading[-1] - heading[-2]) / max(float(dt), eps))
    else:
        yaw_rate = 0.0
    target_ay = target[:, 5].astype(np.float64)
    if len(target_ay) >= 2:
        lateral_jerk = float((target_ay[-1] - target_ay[-2]) / max(float(dt), eps))
    else:
        lateral_jerk = 0.0
    return {
        "ego_vx_current": float(ego[-1, 2]),
        "target_vx_current": float(target[-1, 2]),
        "target_vy_current": float(target[-1, 3]),
        "gap_current": float(gaps[-1]),
        "lateral_offset_current": float(lateral[-1]),
        "ego_ax_current": float(ego[-1, 4]),
        "target_ax_current": float(target[-1, 4]),
        "target_ay_current": float(target[-1, 5]),
        "gap_change_rate": float((gaps[-1] - gaps[0]) / elapsed),
        "lateral_offset_change_rate": float((lateral[-1] - lateral[0]) / elapsed),
        "relative_vx_trend": float((rel_vx[-1] - rel_vx[0]) / elapsed),
        "relative_vy_trend": float((rel_vy[-1] - rel_vy[0]) / elapsed),
        "target_yaw_rate_current": yaw_rate,
        "target_lateral_jerk_current": lateral_jerk,
        "min_gap_in_prefix": float(np.min(gaps)),
        "min_abs_lateral_offset_in_prefix": float(np.min(np.abs(lateral))),
        "max_abs_target_lateral_velocity_in_prefix": float(np.max(np.abs(target[:, 3]))),
        "lateral_offset_range_in_prefix": float(np.max(lateral) - np.min(lateral)),
    }


def extract_context(
    history: np.ndarray,
    ego_length: float,
    adv_length: float,
    dt: float,
    event_type: str = "following",
) -> tuple[np.ndarray, List[str]]:
    if str(event_type) == "cut_in":
        feats = extract_cutin_context(history, ego_length, adv_length, dt)
        keys = list(CUTIN_CONTEXT_KEYS)
    else:
        feats = extract_following_context(history, ego_length, adv_length, dt)
        keys = list(FOLLOWING_CONTEXT_KEYS)
    return np.asarray([feats[k] for k in keys], dtype=np.float32), keys


def extract_following_relative_history(
    history: np.ndarray,
    ego_length: float,
    adv_length: float,
) -> np.ndarray:
    states = np.asarray(history, dtype=np.float32)
    ego = states[:, 0]
    adv = states[:, 1]
    gap = adv[:, 0] - ego[:, 0] - 0.5 * (ego_length + adv_length)
    delta_v = ego[:, 2] - adv[:, 2]
    return np.stack(
        [
            gap,
            delta_v,
        ],
        axis=-1,
    ).astype(np.float32)


def extract_cutin_relative_history(
    history: np.ndarray,
    ego_length: float,
    adv_length: float,
) -> np.ndarray:
    states = np.asarray(history, dtype=np.float32)
    ego = states[:, 0]
    adv = states[:, 1]
    gap = adv[:, 0] - ego[:, 0] - 0.5 * (ego_length + adv_length)
    lateral = adv[:, 1] - ego[:, 1]
    delta_vx = ego[:, 2] - adv[:, 2]
    delta_vy = ego[:, 3] - adv[:, 3]
    return np.stack(
        [
            gap,
            lateral,
            delta_vx,
            delta_vy,
            adv[:, 3],
            adv[:, 5],
        ],
        axis=-1,
    ).astype(np.float32)


def extract_relative_history(
    history: np.ndarray,
    ego_length: float,
    adv_length: float,
    event_type: str = "following",
) -> np.ndarray:
    if str(event_type) == "cut_in":
        return extract_cutin_relative_history(history, ego_length, adv_length)
    return extract_following_relative_history(history, ego_length, adv_length)
