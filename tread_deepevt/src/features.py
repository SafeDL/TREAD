"""
features.py — DeepEVT 上下文特征提取
======================================

输入 ``states`` 已经在 ego-initial frame 中 (ego at origin, heading +x)，
因此可以直接读取 ``states[0, 1, 0]`` 作为 target 在 ego 坐标系下的纵向位置。

严格禁止以下字段进入 ``context_features``::
    risk_score / min_ttc / min_thw / max_drac /
    ttc_severity / thw_severity / drac_severity /
    risk_percentile / tail_label_*

返回的字典 key 顺序即 feature vector 顺序，并写入
``feature_schema.json`` 以保证训练/评估/推理一致。每条事件还会得到
一份 ``CanonicalScenarioContext`` 字典 (写到 ``canonical_contexts.json``
和 ``tail_conditions.csv``)，三阶段共享。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .scenario_frame import (
    CUTIN_CONTEXT_TO_CANONICAL,
    FOLLOWING_CONTEXT_TO_CANONICAL,
    CanonicalScenarioContext,
    build_canonical_context,
)

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

# 约定的 following / cut-in context 特征顺序 (与 canonical mapping 同步)
FOLLOWING_FEATURE_KEYS: Tuple[str, ...] = tuple(FOLLOWING_CONTEXT_TO_CANONICAL.keys())
CUTIN_FEATURE_KEYS: Tuple[str, ...] = tuple(CUTIN_CONTEXT_TO_CANONICAL.keys())


def _safe_div(num: float, den: float, default: float, eps: float = 1e-6) -> float:
    return float(num / den) if abs(den) > eps else float(default)


def _prefix_slice(states: np.ndarray, prefix_steps: int) -> np.ndarray:
    K = max(1, min(int(prefix_steps), states.shape[0]))
    return states[:K]


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


# ---------------------------------------------------------------------------
# Following context
# ---------------------------------------------------------------------------

def extract_following_context(
    states: np.ndarray,           # ego-initial frame
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
) -> Dict[str, float]:
    """从 ego-initial frame 状态张量提取 car-following 上下文特征。

    第一版仅使用 t=0 初始状态 (initial-context)，保证 DeepEVT / Diffusion /
    MATLAB 三阶段闭环。prefix-derived 统计量不进入模型输入。
    """
    eps = float(config.get("risk", {}).get("epsilon", 1e-6))
    s0 = _state_at(states, 0)

    gap_0 = float(s0["tgt_x"] - 0.5 * (ego_length + target_length))

    return {
        "ego_v0": s0["ego_vx"],
        "lead_v0": s0["tgt_vx"],
        "relative_speed_0": s0["ego_vx"] - s0["tgt_vx"],
        "gap_0": gap_0,
        "ego_accel_0": s0["ego_ax"],
        "lead_accel_0": s0["tgt_ax"],
        "thw_0": _safe_div(gap_0, s0["ego_vx"], default=10.0, eps=eps),
    }


# ---------------------------------------------------------------------------
# Cut-in context
# ---------------------------------------------------------------------------

def extract_cutin_context(
    states: np.ndarray,           # ego-initial frame
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    target_length: float,
) -> Dict[str, float]:
    """从 ego-initial frame 状态张量提取 cut-in 上下文特征。

    第一版仅使用 t=0 初始状态 (initial-context)。``planned_cutin_duration``
    不作为模型输入 (即使存储在 canonical 中供 MATLAB 场景实例化使用)。
    ``initial_dx`` 使用净纵向间距，与 following 的 gap_0 同口径。
    """
    s0 = _state_at(states, 0)

    initial_dx = float(s0["tgt_x"] - 0.5 * (ego_length + target_length))
    initial_dy = float(s0["tgt_y"])

    return {
        "ego_v0": s0["ego_vx"],
        "target_v0": s0["tgt_vx"],
        "relative_speed_0": s0["ego_vx"] - s0["tgt_vx"],
        "initial_dx": initial_dx,
        "initial_dy": initial_dy,
        "target_vy_0": s0["tgt_vy"],
        "target_ax_0": s0["tgt_ax"],
        "target_ay_0": s0["tgt_ay"],
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


def extract_context_with_canonical(
    event_type: str,
    states: np.ndarray,
    event_row: pd.Series,
    config: dict,
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
) -> Tuple[np.ndarray, List[str], CanonicalScenarioContext]:
    """同时返回 DeepEVT context 向量、key 顺序 与 CanonicalScenarioContext。

    canonical.extras 包含仅作参考的非模型输入量 (如 raw_segment_duration、
    planned_cutin_duration)，供 diffusion / MATLAB 反查场景元信息。
    """
    vec, keys = extract_context(
        event_type, states, event_row, config, ego_length, target_length,
    )
    feats = dict(zip(keys, vec.tolist()))

    fps = float(config.get("sampling", {}).get("target_fps", 25))
    prefix_steps = int(config.get("prefix", {}).get("prefix_steps", 25))

    if event_type == "following":
        start_f = int(event_row.get("start_frame", 0))
        end_f = int(event_row.get("end_frame", start_f))
        raw_duration = max(end_f - start_f, 0) / max(fps, 1.0)
        extras = {
            "thw_0": feats["thw_0"],
            "raw_segment_duration": float(raw_duration),
        }
        planned_cutin = 0.0
        source_lane = event_row.get("source_lane")
        target_lane = event_row.get("target_lane")
    else:  # cut_in
        start_f = int(event_row.get("start_frame", 0))
        end_f = int(event_row.get("end_frame", start_f))
        raw_event_duration = max(end_f - start_f, 0) / max(fps, 1.0)
        planned = event_row.get("cutin_duration")
        if planned is None or (isinstance(planned, float) and not np.isfinite(planned)):
            cs = event_row.get("cutin_start_frame")
            ce = event_row.get("cutin_end_frame")
            if pd.notna(cs) and pd.notna(ce):
                planned = (int(ce) - int(cs)) / max(fps, 1.0)
            else:
                planned = 0.0
        planned_cutin = float(planned)
        extras = {
            "raw_event_duration": float(raw_event_duration),
            "planned_cutin_duration": planned_cutin,
        }
        source_lane = event_row.get("source_lane")
        target_lane = event_row.get("target_lane")

    canonical = build_canonical_context(
        event_id=str(event_row["event_id"]),
        event_type=event_type,
        states_ego_frame=states,
        ego_length=ego_length,
        ego_width=ego_width,
        target_length=target_length,
        target_width=target_width,
        fps=fps,
        prefix_steps=prefix_steps,
        source_lane=int(source_lane) if pd.notna(source_lane) else None,
        target_lane=int(target_lane) if pd.notna(target_lane) else None,
        planned_cutin_duration=planned_cutin,
        extras=extras,
    )
    return vec, keys, canonical


def assert_no_leakage(feats: Dict[str, float], forbidden: Sequence[str] = LEAKAGE_KEYS) -> None:
    bad = [k for k in feats if k in forbidden]
    if bad:
        raise ValueError(f"DeepEVT context features contain leakage keys: {bad}")
