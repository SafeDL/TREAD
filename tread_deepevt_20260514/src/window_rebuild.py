"""
window_rebuild.py — 从 events.csv + raw highD 重建固定 analysis window
=====================================================================

第一阶段 ``events.csv`` 只存储事件元信息，DeepEVT 训练需要每条事件对应
的短历史状态张量 ``states[prefix_steps, actors, state_features]`` (actor0 = ego,
actor1 = target) 以及在固定未来窗口内重新计算的响应风险
``window_risk_score``。

本模块完全复用 tread_highd 的 loader / preprocess / risk_metrics，
保证坐标系、方向统一、净间距计算与第一阶段完全一致。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tread_highd.src.loader import HighDRecording, load_recording
from tread_highd.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)
from tread_highd.src.risk_metrics import (
    compute_drac,
    compute_gap,
    compute_instant_risk,
    compute_thw,
    compute_trajectory_risk,
    compute_ttc,
)

from .scenario_frame import (
    CANONICAL_STATE_FEATURES,
    compute_ego_frame,
    world_to_ego_states,
)

logger = logging.getLogger(__name__)

# 每个 actor 的状态特征顺序 (保持稳定供模型 reshape)。
# 注意: 这里与 canonical frame 字段一一对应；数值含义是 ego-current frame。
STATE_FEATURES: tuple[str, ...] = CANONICAL_STATE_FEATURES
NUM_ACTORS = 2
NUM_STATE_FEATURES = len(STATE_FEATURES)


@dataclass
class WindowSample:
    """单个事件的固定窗口样本。

    重要：``states`` 已经转换到 **ego-current frame** (prefix 末端 ego at origin, heading +x)。
    若需要回到世界坐标系，使用 ``ego_frame`` + ``world_states``。
    """

    event_id: str
    event_type: str
    recording_id: int
    ego_id: int
    target_id: int
    prefix_frames: np.ndarray       # [prefix_steps]
    risk_window_frames: np.ndarray  # [risk_window_steps]
    states: np.ndarray              # [prefix_steps, actors, state_features] in ego-current frame
    world_states: np.ndarray        # [prefix_steps, actors, state_features] in highD world frame
    risk_world_states: np.ndarray   # [risk_window_steps, actors, state_features] in highD world frame
    ego_frame: Dict[str, float]     # origin_x, origin_y, rot_cos, rot_sin
    ego_length: float
    ego_width: float
    target_length: float
    target_width: float
    lane_width: float
    target_final_y: float
    risk_score: float
    min_ttc: float
    min_thw: float
    max_drac: float
    prefix_start_frame: int
    prefix_end_frame: int
    risk_window_start_frame: int
    risk_window_end_frame: int
    anchor_frame: int
    source: str                     # "risk_window" / "anchor_window"


# ---------------------------------------------------------------------------
# 事件过滤
# ---------------------------------------------------------------------------

def filter_events_by_type(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """按 event_type 及第一阶段 is_valid 标记过滤事件行。"""
    df = events[events["event_type"] == event_type].copy()
    if "is_valid" in df.columns:
        valid = df["is_valid"]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[valid]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 帧范围确定
# ---------------------------------------------------------------------------

def get_analysis_frames(event_row: pd.Series, config: dict) -> np.ndarray:
    """返回固定长度 analysis window 的帧索引。

    优先使用第一阶段的 ``risk_window_start_frame`` / ``risk_window_end_frame``。
    若这两个字段不可用或窗口长度不匹配 ``window_length``，则围绕
    ``anchor_frame`` 构建 ``[anchor - pre, anchor + post]``。
    """
    sampling = config.get("sampling", {})
    window_length = int(sampling.get("window_length", 128))
    pre = int(sampling.get("pre_anchor_steps", window_length // 2))
    post = int(sampling.get("post_anchor_steps", window_length - pre - 1))

    rs = event_row.get("risk_window_start_frame")
    re = event_row.get("risk_window_end_frame")
    if pd.notna(rs) and pd.notna(re):
        rs_i = int(rs)
        re_i = int(re)
        span = re_i - rs_i + 1
        if span == window_length:
            return np.arange(rs_i, rs_i + window_length, dtype=np.int64)
        if span > window_length:
            # 风险窗口比分析窗口长 — 围绕 anchor (或风险窗口中心) 裁剪
            anchor = int(event_row.get("anchor_frame", (rs_i + re_i) // 2))
            center = max(rs_i + pre, min(anchor, re_i - (window_length - pre)))
            start = center - pre
            return np.arange(start, start + window_length, dtype=np.int64)
        if span > 0:
            # 以现有风险窗口为中心向两侧扩展
            center = (rs_i + re_i) // 2
            start = center - pre
            return np.arange(start, start + window_length, dtype=np.int64)

    anchor = int(event_row["anchor_frame"])
    start = anchor - pre
    return np.arange(start, start + window_length, dtype=np.int64)


def get_prefix_frames(risk_frames: np.ndarray, config: dict) -> np.ndarray:
    """返回模型可见的短历史帧。

    为兼容当前单帧实现，默认让 prefix 最后一帧与 risk window 起始帧重合。
    因此 ``K=1`` 时仍等价于原始的 initial/current scene 输入；``K>1`` 时
    不再错误读取 risk window 内的后续帧作为历史。
    """
    prefix_steps = int(config.get("prefix", {}).get("prefix_steps", 1))
    if prefix_steps < 1:
        raise ValueError(f"prefix.prefix_steps must be >= 1, got {prefix_steps}")
    end_frame = int(risk_frames[0])
    start_frame = end_frame - prefix_steps + 1
    return np.arange(start_frame, end_frame + 1, dtype=np.int64)


# ---------------------------------------------------------------------------
# 状态张量构建
# ---------------------------------------------------------------------------

def _extract_vehicle_states(
    recording: HighDRecording,
    vehicle_id: int,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
    """返回 vehicle 在指定帧序列上的 ``[time_steps, state_features]`` 状态，若有缺帧返回 None。"""
    try:
        track = recording.get_vehicle_track(int(vehicle_id))
    except KeyError:
        return None
    present = track.index.intersection(frames)
    if len(present) != len(frames):
        return None
    sub = track.loc[frames]
    if "_abnormal" in sub.columns and bool(sub["_abnormal"].any()):
        return None

    out = np.empty((len(frames), NUM_STATE_FEATURES), dtype=np.float32)
    out[:, 0] = sub["x"].values
    out[:, 1] = sub["y"].values
    out[:, 2] = sub["xVelocity"].values
    out[:, 3] = sub["yVelocity"].values if "yVelocity" in sub.columns else 0.0
    out[:, 4] = sub["xAcceleration"].values
    out[:, 5] = sub["yAcceleration"].values if "yAcceleration" in sub.columns else 0.0
    return out


def build_states_from_raw(
    recording: HighDRecording,
    event_row: pd.Series,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
    """返回 ``states[time_steps, actors, state_features]``，actor0=ego, actor1=target。缺帧返回 None。"""
    ego_states = _extract_vehicle_states(recording, int(event_row["ego_id"]), frames)
    if ego_states is None:
        return None
    tgt_states = _extract_vehicle_states(recording, int(event_row["target_id"]), frames)
    if tgt_states is None:
        return None
    return np.stack([ego_states, tgt_states], axis=1).astype(np.float32)


def _lane_width_from_markings(recording: HighDRecording) -> float:
    """Estimate lane width from highD lane marking metadata."""
    widths: List[float] = []
    for key in ("upperLaneMarkings", "lowerLaneMarkings"):
        marks = np.asarray(recording.recording_meta.get(key, []), dtype=float)
        marks = marks[np.isfinite(marks)]
        if len(marks) >= 2:
            widths.extend(np.diff(np.sort(marks)).tolist())
    widths = [float(w) for w in widths if w > 0.5]
    return float(np.median(widths)) if widths else 3.75


def _lane_center_y(recording: HighDRecording, lane_id: object) -> Optional[float]:
    """Best-effort highD lane center y from lane id and lane markings."""
    if lane_id is None or pd.isna(lane_id):
        return None
    try:
        lane = int(lane_id)
    except (TypeError, ValueError):
        return None

    for key in ("upperLaneMarkings", "lowerLaneMarkings"):
        marks = np.asarray(recording.recording_meta.get(key, []), dtype=float)
        marks = np.sort(marks[np.isfinite(marks)])
        if len(marks) < 2:
            continue
        # highD lane ids are 1-based and separated by direction. For the common
        # car lanes, lane id 2 maps to the first interval in each direction.
        idx = lane - 2 if lane <= len(marks) else lane - len(marks) - 2
        if 0 <= idx < len(marks) - 1:
            return float(0.5 * (marks[idx] + marks[idx + 1]))
    return None


# ---------------------------------------------------------------------------
# 固定窗口风险重算
# ---------------------------------------------------------------------------

def recompute_window_risk(
    recording: HighDRecording,
    event_row: pd.Series,
    states: np.ndarray,
    config: dict,
) -> Dict[str, float]:
    """在固定 risk window 内重新计算 risk_score / min_ttc / min_thw / max_drac。"""
    risk_cfg = config.get("risk", {})
    eps = float(risk_cfg.get("epsilon", 1e-6))
    max_ttc = float(risk_cfg.get("max_ttc_clip", 1000.0))
    max_thw = float(risk_cfg.get("max_thw_clip", 200.0))
    soft_lambda = float(risk_cfg.get("softmax_lambda", 10.0))

    meta = recording.tracks_meta
    ego_len = float(meta.loc[int(event_row["ego_id"])]["width"])
    tgt_len = float(meta.loc[int(event_row["target_id"])]["width"])

    ego_x = states[:, 0, 0]
    ego_vx = states[:, 0, 2]
    tgt_x = states[:, 1, 0]
    tgt_vx = states[:, 1, 2]

    gap = compute_gap(ego_x, tgt_x, ego_len, tgt_len)
    ttc = compute_ttc(gap, ego_vx, tgt_vx, max_ttc=max_ttc, eps=eps)
    thw = compute_thw(gap, ego_vx, max_thw=max_thw, eps=eps)
    drac = compute_drac(gap, ego_vx, tgt_vx, eps=eps)

    mask = gap > eps
    if not np.any(mask):
        return {
            "risk_score": float("nan"),
            "min_ttc": float(max_ttc),
            "min_thw": float(max_thw),
            "max_drac": 0.0,
            "valid_risk_frames": 0,
        }

    r_ttc = ttc[mask]
    r_thw = thw[mask]
    r_drac = drac[mask]
    instant = compute_instant_risk(r_ttc, r_thw, r_drac, risk_cfg, eps=eps)
    traj_risk = compute_trajectory_risk(instant, softmax_lambda=soft_lambda)
    return {
        "risk_score": float(traj_risk),
        "min_ttc": float(np.min(r_ttc)),
        "min_thw": float(np.min(r_thw)),
        "max_drac": float(np.max(r_drac)),
        "valid_risk_frames": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# 单事件重建 + 批量构建
# ---------------------------------------------------------------------------

def rebuild_event_window(
    recording: HighDRecording,
    event_row: pd.Series,
    config: dict,
) -> Optional[WindowSample]:
    """单个事件的固定 window 重建。失败返回 None 并记录 debug 原因。"""
    risk_frames = get_analysis_frames(event_row, config)
    prefix_frames = get_prefix_frames(risk_frames, config)
    window_length = int(config.get("sampling", {}).get("window_length", 128))
    if len(risk_frames) != window_length:
        logger.debug(
            "[%s] insufficient_analysis_window: expected %d got %d",
            event_row.get("event_id", "?"), window_length, len(risk_frames),
        )
        return None

    frame_ids = recording.tracks.index.get_level_values("frame")
    f_min = int(frame_ids.min())
    f_max = int(frame_ids.max())
    if int(prefix_frames[0]) < f_min or int(risk_frames[-1]) > f_max:
        logger.debug(
            "[%s] prefix/risk window out of recording range [%d, %d]",
            event_row.get("event_id", "?"), f_min, f_max,
        )
        return None

    prefix_world_states = build_states_from_raw(recording, event_row, prefix_frames)
    risk_world_states = build_states_from_raw(recording, event_row, risk_frames)
    if prefix_world_states is None or risk_world_states is None:
        logger.debug(
            "[%s] missing frames or abnormal tracks",
            event_row.get("event_id", "?"),
        )
        return None

    # 风险计算始终在世界坐标系 (和第一阶段 risk_metrics 完全一致)；
    # 旋转到 ego-current frame 只影响几何表示，不改变 net gap 与速度大小。
    risk = recompute_window_risk(recording, event_row, risk_world_states, config)
    if not np.isfinite(risk["risk_score"]):
        logger.debug("[%s] non-finite window risk", event_row.get("event_id", "?"))
        return None

    # ego-current frame：prefix 最后一帧 ego 几何中心为原点。K=1 时与原实现一致。
    ego_frame = compute_ego_frame(prefix_world_states[-1, 0])
    states_ego = world_to_ego_states(prefix_world_states, ego_frame).astype(np.float32)
    lane_width = _lane_width_from_markings(recording)
    lane_center = _lane_center_y(recording, event_row.get("target_lane"))
    target_final_y = (
        float(lane_center - ego_frame["origin_y"])
        if lane_center is not None
        else float(states_ego[-1, 1, 1])
    )

    meta = recording.tracks_meta
    ego_row_meta = meta.loc[int(event_row["ego_id"])]
    tgt_row_meta = meta.loc[int(event_row["target_id"])]

    source = "risk_window" if pd.notna(event_row.get("risk_window_start_frame")) else "anchor_window"
    return WindowSample(
        event_id=str(event_row["event_id"]),
        event_type=str(event_row["event_type"]),
        recording_id=int(event_row["recording_id"]),
        ego_id=int(event_row["ego_id"]),
        target_id=int(event_row["target_id"]),
        prefix_frames=prefix_frames,
        risk_window_frames=risk_frames,
        states=states_ego,
        world_states=prefix_world_states.astype(np.float32),
        risk_world_states=risk_world_states.astype(np.float32),
        ego_frame=ego_frame,
        ego_length=float(ego_row_meta["width"]),
        ego_width=float(ego_row_meta.get("height", 1.8)),
        target_length=float(tgt_row_meta["width"]),
        target_width=float(tgt_row_meta.get("height", 1.8)),
        lane_width=lane_width,
        target_final_y=target_final_y,
        risk_score=float(risk["risk_score"]),
        min_ttc=float(risk["min_ttc"]),
        min_thw=float(risk["min_thw"]),
        max_drac=float(risk["max_drac"]),
        prefix_start_frame=int(prefix_frames[0]),
        prefix_end_frame=int(prefix_frames[-1]),
        risk_window_start_frame=int(risk_frames[0]),
        risk_window_end_frame=int(risk_frames[-1]),
        anchor_frame=int(event_row["anchor_frame"]),
        source=source,
    )


def prepare_recording(raw_dir: str, recording_id: int, config: dict) -> HighDRecording:
    """加载并按第一阶段相同方式预处理一个 recording。"""
    rec = load_recording(raw_dir, int(recording_id))
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(config.get("sampling", {}).get("target_fps", 25))
    rec = resample_recording(rec, target_fps)
    return rec
