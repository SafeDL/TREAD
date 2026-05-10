"""
coordinate.py — ego-centric 坐标转换
======================================
将原始 highD 坐标转换为以 ego 为参考的相对坐标，输出固定维度状态张量。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .loader import HighDRecording
from .schema import NUM_FEATURES, NUM_ACTORS


def to_ego_centric(ego_track, target_track, frames):
    """将 ego 和 target 转换为 ego-centric 坐标。

    Returns: states (T, 2, 11), mask (T, 2)
    """
    T = len(frames)
    states = np.zeros((T, NUM_ACTORS, NUM_FEATURES), dtype=np.float32)
    mask = np.zeros((T, NUM_ACTORS), dtype=bool)

    for t, fr in enumerate(frames):
        if fr in ego_track.index:
            er = ego_track.loc[fr]
            ego_x, ego_y = float(er["x"]), float(er.get("y", 0))
            ego_vx, ego_vy = float(er["xVelocity"]), float(er.get("yVelocity", 0))
            ego_ax, ego_ay = float(er.get("xAcceleration", 0)), float(er.get("yAcceleration", 0))

            states[t, 0, :] = [
                0, 0, 0, 0,
                ego_vx, ego_vy, ego_ax, ego_ay,
                float(er.get("laneId", 0)), float(er.get("width", 4.5)), float(er.get("height", 1.8)),
            ]
            mask[t, 0] = True

            if fr in target_track.index:
                tr = target_track.loc[fr]
                tgt_x, tgt_y = float(tr["x"]), float(tr.get("y", 0))
                tgt_vx, tgt_vy = float(tr["xVelocity"]), float(tr.get("yVelocity", 0))

                states[t, 1, :] = [
                    tgt_x - ego_x, tgt_y - ego_y,
                    tgt_vx - ego_vx, tgt_vy - ego_vy,
                    tgt_vx, tgt_vy,
                    float(tr.get("xAcceleration", 0)), float(tr.get("yAcceleration", 0)),
                    float(tr.get("laneId", 0)), float(tr.get("width", 4.5)), float(tr.get("height", 1.8)),
                ]
                mask[t, 1] = True

    return states, mask


def build_state_tensor(event, recording, config, frames=None):
    """为单个事件构建状态张量。"""
    sampling = config.get("sampling", {})
    pre = sampling.get("pre_anchor_steps", 32)
    post = sampling.get("post_anchor_steps", 31)

    ego_track = recording.get_vehicle_track(event.ego_id)
    tgt_track = recording.get_vehicle_track(event.target_id)

    if frames is None:
        from .windowing import get_window_from_track
        frames = get_window_from_track(
            recording, event.ego_id, event.target_id,
            event.anchor_frame, pre, post,
        )
        if frames is None:
            anchor = event.anchor_frame
            frames = np.arange(anchor - pre, anchor + post + 1)

    return to_ego_centric(ego_track, tgt_track, frames)
