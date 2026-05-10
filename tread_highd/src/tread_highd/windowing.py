"""
windowing.py — 固定长度窗口构建
=================================
以事件 anchor frame 为中心截取固定长度窗口，并验证窗口有效性。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .loader import HighDRecording

logger = logging.getLogger(__name__)


def get_window_frames(anchor_frame, pre_steps, post_steps):
    """生成以 anchor 为中心的连续帧序列。"""
    return np.arange(anchor_frame - pre_steps, anchor_frame + post_steps + 1)


def get_window_from_track(recording, ego_id, target_id, anchor_frame, pre_steps, post_steps):
    """从 ego 和 target 的公共帧中提取固定长度窗口。

    Returns None if insufficient frames.
    """
    ego_track = recording.get_vehicle_track(ego_id)
    tgt_track = recording.get_vehicle_track(target_id)
    common = np.array(sorted(set(ego_track.index) & set(tgt_track.index)))
    if len(common) == 0:
        return None

    anchor_idx = np.searchsorted(common, anchor_frame)
    if anchor_idx >= len(common) or common[anchor_idx] != anchor_frame:
        anchor_idx = np.argmin(np.abs(common - anchor_frame))

    start_idx = anchor_idx - pre_steps
    end_idx = anchor_idx + post_steps + 1
    if start_idx < 0 or end_idx > len(common):
        return None

    return common[start_idx:end_idx]


def validate_window(recording, ego_id, target_id, frames, config):
    """验证窗口数据质量 (运动学规则，非风险过滤)。

    检查: 帧完整性、gap 合理性、加速度、速度有效性、异常标记。
    """
    if frames is None or len(frames) == 0:
        return False, "no_valid_frames"

    filters = config.get("filters", {})
    max_accel = filters.get("max_abs_accel", 8.0)
    min_speed = filters.get("min_vehicle_speed", 0.0)

    ego_track = recording.get_vehicle_track(ego_id)
    tgt_track = recording.get_vehicle_track(target_id)

    ego_frames = set(ego_track.index)
    tgt_frames = set(tgt_track.index)

    # 帧完整性
    missing_ego = sum(1 for f in frames if f not in ego_frames)
    if missing_ego > 0:
        return False, f"missing_ego_frames: {missing_ego}/{len(frames)}"
    missing_tgt = sum(1 for f in frames if f not in tgt_frames)
    if missing_tgt > 0:
        return False, f"missing_target_frames: {missing_tgt}/{len(frames)}"

    # gap 合理性 (车辆重叠检测)
    gap = tgt_track.loc[frames, "x"].values - ego_track.loc[frames, "x"].values
    if np.mean(gap < 0) > 0.3:
        return False, f"excessive_negative_gap: {np.mean(gap < 0):.2%}"

    # 加速度 (ego + target)
    if np.any(np.abs(ego_track.loc[frames, "xAcceleration"].values) > max_accel):
        return False, "ego_abnormal_acceleration"
    if np.any(np.abs(tgt_track.loc[frames, "xAcceleration"].values) > max_accel):
        return False, "target_abnormal_acceleration"

    # 速度有效性 (方向统一后应为正)
    if min_speed > 0:
        if np.mean(ego_track.loc[frames, "xVelocity"].values < min_speed) > 0.3:
            return False, "ego_invalid_speed"
        if np.mean(tgt_track.loc[frames, "xVelocity"].values < min_speed) > 0.3:
            return False, "target_invalid_speed"

    # 上游异常标记
    if "_abnormal" in ego_track.columns:
        if ego_track.loc[frames, "_abnormal"].sum() > len(frames) * 0.2:
            return False, "too_many_abnormal_frames"

    return True, ""
