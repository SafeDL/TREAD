"""
coordinate.py — ego-centric 坐标转换
======================================
将原始 highD 坐标转换为以 ego 为参考的相对坐标，输出固定维度状态张量。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .loader import HighDRecording
from .schema import EventRecord, NUM_FEATURES, NUM_ACTORS

logger = logging.getLogger(__name__)


def to_ego_centric(ego_track, target_track, frames):
    """将 ego 和 target 转换为 ego-centric 坐标。

    Parameters
    ----------
    ego_track, target_track : pd.DataFrame
        按 frame 索引的轨迹。
    frames : ndarray
        要提取的帧序列。

    Returns
    -------
    ndarray, shape (T, 2, 11)
    ndarray, shape (T, 2) — mask
    """
    T = len(frames)
    A = NUM_ACTORS
    F = NUM_FEATURES
    states = np.zeros((T, A, F), dtype=np.float32)
    mask = np.zeros((T, A), dtype=bool)

    for t, fr in enumerate(frames):
        # ego
        if fr in ego_track.index:
            er = ego_track.loc[fr]
            ego_x = float(er["x"])
            ego_vx = float(er["xVelocity"])
            ego_vy = float(er.get("yVelocity", 0))
            ego_ax = float(er.get("xAcceleration", 0))
            ego_ay = float(er.get("yAcceleration", 0))
            ego_lane = float(er.get("laneId", 0))
            ego_len = float(er.get("width", 4.5))  # highD width = 车长
            ego_wid = float(er.get("height", 1.8))  # highD height = 车宽

            # ego actor: 相对坐标为 0
            states[t, 0, :] = [
                0.0, 0.0, 0.0, 0.0,  # dx, dy, dvx, dvy
                ego_vx, ego_vy, ego_ax, ego_ay,
                ego_lane, ego_len, ego_wid,
            ]
            mask[t, 0] = True

            # target
            if fr in target_track.index:
                tr = target_track.loc[fr]
                tgt_x = float(tr["x"])
                tgt_y = float(tr.get("y", 0))
                ego_y = float(er.get("y", 0))
                tgt_vx = float(tr["xVelocity"])
                tgt_vy = float(tr.get("yVelocity", 0))
                tgt_ax = float(tr.get("xAcceleration", 0))
                tgt_ay = float(tr.get("yAcceleration", 0))
                tgt_lane = float(tr.get("laneId", 0))
                tgt_len = float(tr.get("width", 4.5))
                tgt_wid = float(tr.get("height", 1.8))

                states[t, 1, :] = [
                    tgt_x - ego_x, tgt_y - ego_y,
                    tgt_vx - ego_vx, tgt_vy - ego_vy,
                    tgt_vx, tgt_vy, tgt_ax, tgt_ay,
                    tgt_lane, tgt_len, tgt_wid,
                ]
                mask[t, 1] = True

    return states, mask


def build_state_tensor(event, recording, config, frames=None):
    """为单个事件构建状态张量。

    Parameters
    ----------
    event : EventRecord
    recording : HighDRecording
    config : dict
    frames : ndarray, optional
        如果提供，使用这些帧；否则自动从轨迹中提取。

    Returns
    -------
    states : ndarray, shape (T, 2, 11)
    mask : ndarray, shape (T, 2)
    """
    sampling = config.get("sampling", {})
    pre = sampling.get("pre_anchor_steps", 32)
    post = sampling.get("post_anchor_steps", 31)

    ego_track = recording.get_vehicle_track(event.ego_id)
    tgt_track = recording.get_vehicle_track(event.target_id)

    if isinstance(ego_track.index, pd.MultiIndex):
        ego_track = ego_track.droplevel("id")
    if isinstance(tgt_track.index, pd.MultiIndex):
        tgt_track = tgt_track.droplevel("id")

    if frames is None:
        # 使用实际可用帧
        from .windowing import get_window_from_track
        frames = get_window_from_track(
            recording, event.ego_id, event.target_id,
            event.anchor_frame, pre, post,
        )
        if frames is None:
            # 回退到连续帧范围
            anchor = event.anchor_frame
            frames = np.arange(anchor - pre, anchor + post + 1)

    states, mask = to_ego_centric(ego_track, tgt_track, frames)
    return states, mask

