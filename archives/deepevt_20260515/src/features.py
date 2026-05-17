"""
features.py — DeepEVT 上下文特征提取
======================================

输入 ``states`` 已经在 ego-current frame 中 (prefix 末端 ego at origin,
heading +x)，因此可以直接读取 prefix 末端状态作为当前交互状态。

严格禁止以下字段进入 ``context_features``::
    risk_score / min_ttc / min_thw / max_drac /
    ttc_severity / thw_severity / drac_severity /
    risk_percentile / tail_label_*

返回的字典 key 顺序即 feature vector 顺序，并写入
``feature_schema.json`` 以保证训练/评估/推理一致。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 明确禁止作为输入的风险泄漏字段 / future-leaking 字段
LEAKAGE_KEYS: Tuple[str, ...] = (
    "risk_score", "min_ttc", "min_thw", "max_drac",
    "ttc_severity", "thw_severity", "drac_severity",
    "risk_percentile",
    "tail_label_90", "tail_label_95", "tail_label_99",
    # future-derived / post-hoc fields (unless explicitly a controllable parameter)
    "planned_cutin_duration",
    "raw_segment_duration", "raw_event_duration",
)

# 约定的 following / cut-in context 特征顺序。
FOLLOWING_FEATURE_KEYS: Tuple[str, ...] = (
    "ego_vx_current",
    "lead_vx_current",
    "relative_speed_current",
    "gap_current",
    "lateral_offset_current",
    "ego_ax_current",
    "lead_ax_current",
    "gap_change_rate",
    "relative_speed_trend",
    "relative_acceleration",
    "ego_acc_mean_over_prefix",
    "lead_acc_mean_over_prefix",
    "lead_brake_indicator",
    "min_gap_in_prefix",
    "max_closing_speed_in_prefix",
    "lateral_offset_change_rate",
    "lane_width",
    "dt",
    "horizon_steps",
    "prefix_steps",
)
CUTIN_FEATURE_KEYS: Tuple[str, ...] = (
    "ego_vx0",
    "target_vx0",
    "relative_speed_0",
    "target_center_x0",
    "target_center_y0",
    "initial_gap",
    "initial_lateral_offset",
    "target_vy0",
    "target_ax0",
    "target_ay0",
    "lane_width",
    "target_final_y",
    "dt",
    "horizon_steps",
)


def _safe_div(num: float, den: float, default: float, eps: float = 1e-6) -> float:
    return float(num / den) if abs(den) > eps else float(default)


def _state_at(states: np.ndarray, t: int) -> Dict[str, float]:
    row = states[t]
    return {
        "ego_x": float(row[0, 0]), "ego_y": float(row[0, 1]),
        "ego_vx": float(row[0, 2]), "ego_vy": float(row[0, 3]),
        "ego_ax": float(row[0, 4]), "ego_ay": float(row[0, 5]),
        "tgt_x": float(row[1, 0]), "tgt_y": float(row[1, 1]),
        "tgt_vx": float(row[1, 2]), "tgt_vy": float(row[1, 3]),
        "tgt_ax": float(row[1, 4]), "tgt_ay": float(row[1, 5]),
    }


def _prefix_current_index(states: np.ndarray) -> int:
    return max(0, states.shape[0] - 1)


def _elapsed_seconds(num_steps: int, dt: float) -> float:
    return float(max(num_steps - 1, 0) * dt)


# ---------------------------------------------------------------------------
# Following context
# ---------------------------------------------------------------------------

def extract_following_context(
    states: np.ndarray,           # ego-current frame
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
    lane_width: float = 3.75,
) -> Dict[str, float]:
    """从 ego-current frame 短历史状态张量提取 car-following 上下文特征。

    特征只使用 prefix window 内的状态：末端当前状态、短历史趋势和配置常量。
    future risk window 中的 ``risk_score`` / min TTC / max DRAC 等统计量不进入输入。
    """
    current_idx = _prefix_current_index(states)
    s_current = _state_at(states, current_idx)
    dt = 1.0 / max(float(config.get("sampling", {}).get("target_fps", 25)), 1.0)
    horizon_steps = int(config.get("sampling", {}).get("window_length", states.shape[0]))
    prefix_steps = int(states.shape[0])

    ego_x = states[:, 0, 0].astype(np.float64)
    ego_y = states[:, 0, 1].astype(np.float64)
    lead_x = states[:, 1, 0].astype(np.float64)
    lead_y = states[:, 1, 1].astype(np.float64)
    ego_vx = states[:, 0, 2].astype(np.float64)
    lead_vx = states[:, 1, 2].astype(np.float64)
    ego_ax = states[:, 0, 4].astype(np.float64)
    lead_ax = states[:, 1, 4].astype(np.float64)

    gaps = lead_x - ego_x - 0.5 * (ego_length + target_length)
    lateral_offsets = lead_y - ego_y
    relative_speed = ego_vx - lead_vx
    elapsed = _elapsed_seconds(prefix_steps, dt)

    gap_current = float(gaps[-1])
    lateral_offset_current = float(lateral_offsets[-1])
    relative_speed_current = float(relative_speed[-1])
    lead_ax_current = float(lead_ax[-1])

    return {
        "ego_vx_current": s_current["ego_vx"],
        "lead_vx_current": s_current["tgt_vx"],
        "relative_speed_current": relative_speed_current,
        "gap_current": gap_current,
        "lateral_offset_current": lateral_offset_current,
        "ego_ax_current": s_current["ego_ax"],
        "lead_ax_current": lead_ax_current,
        "gap_change_rate": _safe_div(float(gaps[-1] - gaps[0]), elapsed, 0.0),
        "relative_speed_trend": _safe_div(float(relative_speed[-1] - relative_speed[0]), elapsed, 0.0),
        "relative_acceleration": float(s_current["ego_ax"] - lead_ax_current),
        "ego_acc_mean_over_prefix": float(np.mean(ego_ax)),
        "lead_acc_mean_over_prefix": float(np.mean(lead_ax)),
        "lead_brake_indicator": float(np.min(lead_ax) < -0.5),
        "min_gap_in_prefix": float(np.min(gaps)),
        "max_closing_speed_in_prefix": float(np.maximum(relative_speed, 0.0).max()),
        "lateral_offset_change_rate": _safe_div(float(lateral_offsets[-1] - lateral_offsets[0]), elapsed, 0.0),
        "lane_width": float(lane_width),
        "dt": float(dt),
        "horizon_steps": float(horizon_steps),
        "prefix_steps": float(prefix_steps),
    }


# ---------------------------------------------------------------------------
# Cut-in context
# ---------------------------------------------------------------------------

def extract_cutin_context(
    states: np.ndarray,           # ego-current frame
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
    lane_width: float = 3.75,
    target_final_y: float = 0.0,
) -> Dict[str, float]:
    """从 ego-current frame 状态张量提取 cut-in 上下文特征。

    仅使用 prefix 末端当前状态和可由目标车道几何给出的计划横向终点。
    ``planned_cutin_duration`` 不作为模型输入，即使存储在 canonical 中供
    MATLAB 场景实例化使用。
    """
    s0 = _state_at(states, _prefix_current_index(states))
    dt = 1.0 / max(float(config.get("sampling", {}).get("target_fps", 25)), 1.0)
    horizon_steps = int(config.get("sampling", {}).get("window_length", states.shape[0]))

    initial_gap = float(s0["tgt_x"] - 0.5 * (ego_length + target_length))
    initial_dy = float(s0["tgt_y"])

    return {
        "ego_vx0": s0["ego_vx"],
        "target_vx0": s0["tgt_vx"],
        "relative_speed_0": s0["ego_vx"] - s0["tgt_vx"],
        "target_center_x0": s0["tgt_x"],
        "target_center_y0": s0["tgt_y"],
        "initial_gap": initial_gap,
        "initial_lateral_offset": initial_dy,
        "target_vy0": s0["tgt_vy"],
        "target_ax0": s0["tgt_ax"],
        "target_ay0": s0["tgt_ay"],
        "lane_width": float(lane_width),
        "target_final_y": float(target_final_y),
        "dt": dt,
        "horizon_steps": float(horizon_steps),
    }


# ---------------------------------------------------------------------------
# 入口 & 校验
# ---------------------------------------------------------------------------

def feature_keys_for(event_type: str) -> Tuple[str, ...]:
    if event_type == "following":
        return FOLLOWING_FEATURE_KEYS
    if event_type == "cut_in":
        return CUTIN_FEATURE_KEYS
    raise ValueError(f"Unsupported event_type: {event_type}")


def extract_context(
    event_type: str,
    states: np.ndarray,
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
    lane_width: float = 3.75,
    target_final_y: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """按 event_type 调度并返回 (vector, key_order)。"""
    if event_type == "following":
        feats = extract_following_context(
            states, event_row, config, ego_length, target_length, lane_width,
        )
        keys = list(FOLLOWING_FEATURE_KEYS)
    elif event_type == "cut_in":
        feats = extract_cutin_context(
            states, event_row, config, ego_length, target_length,
            lane_width, target_final_y,
        )
        keys = list(CUTIN_FEATURE_KEYS)
    else:
        raise ValueError(f"Unsupported event_type: {event_type}")

    if config.get("features", {}).get("forbid_risk_leakage", True):
        assert_no_leakage(feats)
    vec = np.array([feats[k] for k in keys], dtype=np.float32)
    return vec, keys


def assert_no_leakage(feats: Dict[str, float], forbidden: Sequence[str] = LEAKAGE_KEYS) -> None:
    bad = [k for k in feats if k in forbidden]
    if bad:
        raise ValueError(f"DeepEVT context features contain leakage keys: {bad}")
