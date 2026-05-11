"""
window_rebuild.py — 从 events.csv + raw highD 重建固定 analysis window
=====================================================================

第一阶段 ``events.csv`` 只存储事件元信息，DeepEVT 训练需要每条事件对应
的固定长度状态张量 ``states[T, 2, F]`` (actor0 = ego, actor1 = target)
以及在同一窗口内重新计算的响应风险 ``window_risk_score``。

本模块完全复用 tread_highd 的 loader / preprocess / risk_metrics，
保证坐标系、方向统一、净间距计算与第一阶段完全一致。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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

logger = logging.getLogger(__name__)

# 每个 actor 的状态特征顺序 (保持稳定供模型 reshape)
STATE_FEATURES: tuple[str, ...] = (
    "x", "y", "vx", "vy", "ax", "ay",
)
NUM_ACTORS = 2
NUM_STATE_FEATURES = len(STATE_FEATURES)


@dataclass
class WindowSample:
    """单个事件的固定窗口样本。"""

    event_id: str
    event_type: str
    recording_id: int
    ego_id: int
    target_id: int
    frames: np.ndarray              # [T]
    states: np.ndarray              # [T, 2, F]
    risk_score: float
    min_ttc: float
    min_thw: float
    max_drac: float
    window_start_frame: int
    window_end_frame: int
    anchor_frame: int
    source: str                     # "risk_window" / "anchor_window"


# ---------------------------------------------------------------------------
# 事件过滤
# ---------------------------------------------------------------------------

def filter_events_by_type(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """按 event_type 及第一阶段 is_valid 标记过滤事件行。"""
    df = events[events["event_type"] == event_type].copy()
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(bool)]
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
        if span >= window_length:
            # 截取前 window_length 帧，保证长度严格一致
            return np.arange(rs_i, rs_i + window_length, dtype=np.int64)
        if span > 0:
            # 以现有风险窗口为中心向两侧扩展
            center = (rs_i + re_i) // 2
            start = center - pre
            return np.arange(start, start + window_length, dtype=np.int64)

    anchor = int(event_row["anchor_frame"])
    start = anchor - pre
    return np.arange(start, start + window_length, dtype=np.int64)


# ---------------------------------------------------------------------------
# 状态张量构建
# ---------------------------------------------------------------------------

def _extract_vehicle_states(
    recording: HighDRecording,
    vehicle_id: int,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
    """返回 vehicle 在指定帧序列上的 ``[T, F]`` 状态，若有缺帧返回 None。"""
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
    """返回 ``states[T, 2, F]`` ，actor0=ego, actor1=target。缺帧返回 None。"""
    ego_states = _extract_vehicle_states(recording, int(event_row["ego_id"]), frames)
    if ego_states is None:
        return None
    tgt_states = _extract_vehicle_states(recording, int(event_row["target_id"]), frames)
    if tgt_states is None:
        return None
    return np.stack([ego_states, tgt_states], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 固定窗口风险重算
# ---------------------------------------------------------------------------

def recompute_window_risk(
    recording: HighDRecording,
    event_row: pd.Series,
    states: np.ndarray,
    config: dict,
) -> Dict[str, float]:
    """在固定 analysis window 内重新计算 risk_score / min_ttc / min_thw / max_drac。"""
    risk_cfg = config.get("risk", {})
    eps = float(risk_cfg.get("epsilon", 1e-6))
    max_ttc = float(risk_cfg.get("max_ttc_clip", 20.0))
    max_thw = float(risk_cfg.get("max_thw_clip", 10.0))
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
    frames = get_analysis_frames(event_row, config)
    window_length = int(config.get("sampling", {}).get("window_length", 128))
    if len(frames) != window_length:
        logger.debug(
            "[%s] insufficient_analysis_window: expected %d got %d",
            event_row.get("event_id", "?"), window_length, len(frames),
        )
        return None

    frame_ids = recording.tracks.index.get_level_values("frame")
    f_min = int(frame_ids.min())
    f_max = int(frame_ids.max())
    if int(frames[0]) < f_min or int(frames[-1]) > f_max:
        logger.debug(
            "[%s] window out of recording range [%d, %d]",
            event_row.get("event_id", "?"), f_min, f_max,
        )
        return None

    states = build_states_from_raw(recording, event_row, frames)
    if states is None:
        logger.debug(
            "[%s] missing frames or abnormal tracks",
            event_row.get("event_id", "?"),
        )
        return None

    risk = recompute_window_risk(recording, event_row, states, config)
    if not np.isfinite(risk["risk_score"]):
        logger.debug("[%s] non-finite window risk", event_row.get("event_id", "?"))
        return None

    source = "risk_window" if pd.notna(event_row.get("risk_window_start_frame")) else "anchor_window"
    return WindowSample(
        event_id=str(event_row["event_id"]),
        event_type=str(event_row["event_type"]),
        recording_id=int(event_row["recording_id"]),
        ego_id=int(event_row["ego_id"]),
        target_id=int(event_row["target_id"]),
        frames=frames,
        states=states,
        risk_score=float(risk["risk_score"]),
        min_ttc=float(risk["min_ttc"]),
        min_thw=float(risk["min_thw"]),
        max_drac=float(risk["max_drac"]),
        window_start_frame=int(frames[0]),
        window_end_frame=int(frames[-1]),
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


def iter_window_samples(
    events_df: pd.DataFrame,
    raw_dir: str,
    config: dict,
    recording_ids: Optional[Iterable[int]] = None,
) -> Iterable[WindowSample]:
    """按 recording 分组迭代重建窗口样本。"""
    if recording_ids is None:
        recording_ids = sorted(events_df["recording_id"].unique().tolist())

    for rid in recording_ids:
        rid_int = int(rid)
        sub = events_df[events_df["recording_id"] == rid_int]
        if len(sub) == 0:
            continue
        try:
            rec = prepare_recording(raw_dir, rid_int, config)
        except Exception as exc:  # noqa: BLE001
            logger.error("Recording %02d load failed: %s", rid_int, exc)
            continue
        for _, row in sub.iterrows():
            try:
                sample = rebuild_event_window(rec, row, config)
            except Exception as exc:  # noqa: BLE001
                logger.warning("event %s rebuild error: %s", row.get("event_id"), exc)
                continue
            if sample is not None:
                yield sample


def build_window_samples(
    events_csv: str | Path,
    raw_dir: str | Path,
    config: dict,
    event_type: str,
    recording_ids: Optional[List[int]] = None,
) -> List[WindowSample]:
    """从 events.csv + raw highD 构建指定 event_type 的全部窗口样本。"""
    events_df = pd.read_csv(events_csv)
    events_df = filter_events_by_type(events_df, event_type)
    logger.info("事件类型 %s: %d 条候选事件", event_type, len(events_df))
    samples = list(iter_window_samples(events_df, str(raw_dir), config, recording_ids))
    logger.info("成功重建窗口样本: %d / %d", len(samples), len(events_df))
    return samples
