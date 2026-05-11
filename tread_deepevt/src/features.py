"""
features.py — DeepEVT 上下文特征提取
======================================

严格禁止以下字段进入 ``context_features``：
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

from tread_highd.src.risk_metrics import compute_gap

logger = logging.getLogger(__name__)

# 明确禁止作为输入的风险泄漏字段
LEAKAGE_KEYS: Tuple[str, ...] = (
    "risk_score", "min_ttc", "min_thw", "max_drac",
    "ttc_severity", "thw_severity", "drac_severity",
    "risk_percentile",
    "tail_label_90", "tail_label_95", "tail_label_99",
)

# 约定的 following / cut-in context 特征顺序
FOLLOWING_FEATURE_KEYS: Tuple[str, ...] = (
    "ego_v0",
    "lead_v0",
    "relative_speed_0",
    "gap_0",
    "ego_accel_0",
    "lead_accel_0",
    "thw_0",
    "gap_slope_prefix",
    "closing_speed_max_prefix",
    "lead_accel_min_prefix",
    "raw_segment_duration",
)

CUTIN_FEATURE_KEYS: Tuple[str, ...] = (
    "ego_v0",
    "target_v0",
    "relative_speed_0",
    "initial_dx",
    "initial_dy",
    "target_vy_0",
    "target_ax_0",
    "target_ay_0",
    "planned_cutin_duration",
    "prefix_lateral_speed_mean",
    "raw_event_duration",
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


def _prefix_slice(states: np.ndarray, prefix_steps: int) -> np.ndarray:
    K = max(1, min(int(prefix_steps), states.shape[0]))
    return states[:K]


# ---------------------------------------------------------------------------
# Following context
# ---------------------------------------------------------------------------

def extract_following_context(
    states: np.ndarray,
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
) -> Dict[str, float]:
    """从 ``[T, 2, F]`` 状态张量提取 car-following 上下文特征。"""
    prefix_steps = int(config.get("prefix", {}).get("prefix_steps", 25))
    eps = float(config.get("risk", {}).get("epsilon", 1e-6))

    s0 = _state_at(states, 0)
    prefix = _prefix_slice(states, prefix_steps)
    K = prefix.shape[0]

    gap_0 = float(
        compute_gap(
            np.array([s0["ego_x"]]), np.array([s0["tgt_x"]]),
            ego_length, target_length,
        )[0]
    )

    # prefix gap 的线性趋势 (斜率, m/s 尺度)
    prefix_gaps = compute_gap(
        prefix[:, 0, 0], prefix[:, 1, 0], ego_length, target_length,
    )
    fps = float(config.get("sampling", {}).get("target_fps", 25))
    if K >= 2 and np.isfinite(prefix_gaps).all():
        # 以秒为自变量，得到 m/s 的物理斜率
        t_axis = np.arange(K, dtype=np.float64) / fps
        slope = float(np.polyfit(t_axis, prefix_gaps.astype(np.float64), 1)[0])
    else:
        slope = 0.0

    closing = prefix[:, 0, 2] - prefix[:, 1, 2]
    closing_max = float(np.maximum(closing, 0.0).max()) if K > 0 else 0.0
    lead_accel_min_prefix = float(prefix[:, 1, 4].min()) if K > 0 else 0.0

    start_f = int(event_row.get("start_frame", 0))
    end_f = int(event_row.get("end_frame", start_f))
    raw_duration = max(end_f - start_f, 0) / max(fps, 1.0)

    feats: Dict[str, float] = {
        "ego_v0": s0["ego_vx"],
        "lead_v0": s0["tgt_vx"],
        "relative_speed_0": s0["ego_vx"] - s0["tgt_vx"],
        "gap_0": gap_0,
        "ego_accel_0": s0["ego_ax"],
        "lead_accel_0": s0["tgt_ax"],
        "thw_0": _safe_div(gap_0, s0["ego_vx"], default=10.0, eps=eps),
        "gap_slope_prefix": slope,
        "closing_speed_max_prefix": closing_max,
        "lead_accel_min_prefix": lead_accel_min_prefix,
        "raw_segment_duration": float(raw_duration),
    }
    return feats


# ---------------------------------------------------------------------------
# Cut-in context
# ---------------------------------------------------------------------------

def extract_cutin_context(
    states: np.ndarray,
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
) -> Dict[str, float]:
    """从 ``[T, 2, F]`` 状态张量提取 cut-in 上下文特征。

    注：``initial_dx`` 使用 **净纵向间距** (与 compute_gap 一致)，
    使其与 following 的 gap_0 在同一物理口径。
    """
    prefix_steps = int(config.get("prefix", {}).get("prefix_steps", 25))
    eps = float(config.get("risk", {}).get("epsilon", 1e-6))

    s0 = _state_at(states, 0)
    prefix = _prefix_slice(states, prefix_steps)

    initial_dx = float(
        compute_gap(
            np.array([s0["ego_x"]]), np.array([s0["tgt_x"]]),
            ego_length, target_length,
        )[0]
    )
    initial_dy = s0["tgt_y"] - s0["ego_y"]

    prefix_lateral_speed_mean = (
        float(np.mean(np.abs(prefix[:, 1, 3]))) if prefix.shape[0] > 0 else 0.0
    )

    fps = float(config.get("sampling", {}).get("target_fps", 25))
    start_f = int(event_row.get("start_frame", 0))
    end_f = int(event_row.get("end_frame", start_f))
    raw_event_duration = max(end_f - start_f, 0) / max(fps, 1.0)

    # cutin_duration 可能在 events.csv 中以秒存储；NaN 时退化为观测的切入长度
    planned = event_row.get("cutin_duration")
    if planned is None or (isinstance(planned, float) and not np.isfinite(planned)):
        cs = event_row.get("cutin_start_frame")
        ce = event_row.get("cutin_end_frame")
        if pd.notna(cs) and pd.notna(ce):
            planned = (int(ce) - int(cs)) / max(fps, 1.0)
        else:
            planned = 0.0
    planned_cutin_duration = float(planned)

    _ = eps  # keep eps reachable for future use without lint warnings
    feats: Dict[str, float] = {
        "ego_v0": s0["ego_vx"],
        "target_v0": s0["tgt_vx"],
        "relative_speed_0": s0["ego_vx"] - s0["tgt_vx"],
        "initial_dx": initial_dx,
        "initial_dy": initial_dy,
        "target_vy_0": s0["tgt_vy"],
        "target_ax_0": s0["tgt_ax"],
        "target_ay_0": s0["tgt_ay"],
        "planned_cutin_duration": planned_cutin_duration,
        "prefix_lateral_speed_mean": prefix_lateral_speed_mean,
        "raw_event_duration": float(raw_event_duration),
    }
    return feats


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
) -> Tuple[np.ndarray, List[str]]:
    """按 event_type 调度并返回 (vector, key_order)。"""
    if event_type == "following":
        feats = extract_following_context(states, event_row, config, ego_length, target_length)
        keys = list(FOLLOWING_FEATURE_KEYS)
    elif event_type == "cut_in":
        feats = extract_cutin_context(states, event_row, config, ego_length, target_length)
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
