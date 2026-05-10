"""
windowing.py — 固定长度窗口构建
=================================
以事件 anchor frame 为中心截取固定长度窗口，并验证窗口有效性。

注意: highD 重采样后帧 ID 不连续，因此使用基于轨迹实际帧索引的方式构建窗口。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .loader import HighDRecording

logger = logging.getLogger(__name__)


def get_window_frames(anchor_frame, pre_steps, post_steps):
    """生成以 anchor 为中心的帧序列 (仅用于连续帧的场景)。

    Returns
    -------
    ndarray, shape (pre_steps + 1 + post_steps,)
    """
    return np.arange(anchor_frame - pre_steps, anchor_frame + post_steps + 1)


def get_window_from_track(recording, ego_id, target_id, anchor_frame, pre_steps, post_steps):
    """从实际轨迹帧中提取窗口帧序列。

    在 ego 和 target 的公共帧中，找到 anchor_frame 位置，
    向前取 pre_steps 帧，向后取 post_steps 帧。

    Returns
    -------
    frames : ndarray or None
        实际帧 ID 序列，长度为 pre_steps + 1 + post_steps。
        如果帧不够则返回 None。
    """
    try:
        ego_track = recording.get_vehicle_track(ego_id)
        tgt_track = recording.get_vehicle_track(target_id)
    except KeyError:
        return None

    if isinstance(ego_track.index, pd.MultiIndex):
        ego_track = ego_track.droplevel("id")
    if isinstance(tgt_track.index, pd.MultiIndex):
        tgt_track = tgt_track.droplevel("id")

    # 公共帧
    common = np.array(sorted(set(ego_track.index) & set(tgt_track.index)))
    if len(common) == 0:
        return None

    # 找到 anchor 在公共帧中的位置
    anchor_idx = np.searchsorted(common, anchor_frame)
    if anchor_idx >= len(common) or common[anchor_idx] != anchor_frame:
        # anchor_frame 不在公共帧中，找最近的
        anchor_idx = np.argmin(np.abs(common - anchor_frame))

    T = pre_steps + 1 + post_steps
    start_idx = anchor_idx - pre_steps
    end_idx = anchor_idx + post_steps + 1  # exclusive

    if start_idx < 0 or end_idx > len(common):
        return None

    return common[start_idx:end_idx]


def validate_window(recording, ego_id, target_id, frames, config):
    """验证窗口是否有效。

    验证条件:
    1. ego 与 target 在所有窗口帧中均存在
    2. gap 不出现大量负值
    3. 加速度不超阈值
    4. 无大量异常帧标记

    Returns
    -------
    (bool, str)
        (True, "") 或 (False, "reason_string")
    """
    if frames is None or len(frames) == 0:
        return False, "no_valid_frames"

    filters = config.get("filters", {})
    max_accel = filters.get("max_abs_accel", 8.0)

    try:
        ego_track = recording.get_vehicle_track(ego_id)
        tgt_track = recording.get_vehicle_track(target_id)
    except KeyError as e:
        return False, f"vehicle_not_found: {e}"

    if isinstance(ego_track.index, pd.MultiIndex):
        ego_track = ego_track.droplevel("id")
    if isinstance(tgt_track.index, pd.MultiIndex):
        tgt_track = tgt_track.droplevel("id")

    ego_frames = set(ego_track.index)
    tgt_frames = set(tgt_track.index)

    # 1. 存在性
    missing_ego = sum(1 for f in frames if f not in ego_frames)
    missing_tgt = sum(1 for f in frames if f not in tgt_frames)

    if missing_ego > 0:
        return False, f"missing_ego_frames: {missing_ego}/{len(frames)}"
    if missing_tgt > 0:
        return False, f"missing_target_frames: {missing_tgt}/{len(frames)}"

    # 2. gap 检查 (允许少量负值)
    ego_x = ego_track.loc[frames, "x"].values
    tgt_x = tgt_track.loc[frames, "x"].values
    gap = tgt_x - ego_x
    neg_ratio = np.mean(gap < 0)
    if neg_ratio > 0.3:
        return False, f"excessive_negative_gap: {neg_ratio:.2%}"

    # 3. 加速度
    ego_ax = ego_track.loc[frames, "xAcceleration"].values
    if np.any(np.abs(ego_ax) > max_accel):
        return False, "ego_abnormal_acceleration"

    # 4. 检查异常标记
    if "_abnormal" in ego_track.columns:
        abnormal_count = ego_track.loc[frames, "_abnormal"].sum()
        if abnormal_count > len(frames) * 0.2:
            return False, f"too_many_abnormal_frames: {abnormal_count}"

    return True, ""
