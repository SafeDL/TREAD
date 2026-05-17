"""Ego-current coordinate frame helpers for DeepEVT windows."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ego-current frame 中各通道的语义命名 (通用 actor schema,每个 actor 共用同一顺序)。
# 注意: 这些字段描述的是 **ego-current 坐标系下任一 actor** 的状态通道,
# actor 0 = ego、actor 1 = target 时含义一致,不能混入 "_ego" 后缀避免误解。
CANONICAL_STATE_FEATURES: Tuple[str, ...] = (
    "x",   # position x in ego-current frame, +x = ego forward
    "y",   # position y in ego-current frame, +y = ego left
    "vx",
    "vy",
    "ax",
    "ay",
)

# ---------------------------------------------------------------------------
# Ego-current frame 变换
# ---------------------------------------------------------------------------

def compute_ego_frame(
    ego_state_current: np.ndarray, world_heading_x: float = 1.0, world_heading_y: float = 0.0,
) -> Dict[str, float]:
    """从 ego 在 prefix 末端当前帧的状态构造 (origin, rotation) 描述。

    Parameters
    ----------
    ego_state_current : np.ndarray, shape [state_features]
        ego 当前状态 (x, y, vx, vy, ax, ay)。
    world_heading_x, world_heading_y : float
        ego 当前航向在世界坐标系下的方向向量。highD 已经统一为 +x 方向，
        这里默认 (1, 0)。如果未来接入带 yaw 的数据集，可改用 ego 速度向量。

    Returns
    -------
    dict 形如:
        {"origin_x", "origin_y", "rot_cos", "rot_sin"}
    """
    h_norm = float(np.hypot(world_heading_x, world_heading_y))
    if h_norm < 1e-6:
        rot_cos, rot_sin = 1.0, 0.0
    else:
        rot_cos = float(world_heading_x / h_norm)
        rot_sin = float(world_heading_y / h_norm)
    return {
        "origin_x": float(ego_state_current[0]),
        "origin_y": float(ego_state_current[1]),
        "rot_cos": rot_cos,
        "rot_sin": rot_sin,
    }


def world_to_ego_states(states_world: np.ndarray, frame: Dict[str, float]) -> np.ndarray:
    """``states_world`` shape ``[time_steps, actors, state_features]`` 中的 (x, y) 与 (vx, vy)、(ax, ay)
    转到 ego-current frame；返回相同 shape。

    state_features 顺序: (x, y, vx, vy, ax, ay)。其它维度保持原样。
    """
    ox = frame["origin_x"]; oy = frame["origin_y"]
    c = frame["rot_cos"]; s = frame["rot_sin"]
    out = states_world.copy()

    # position
    px = states_world[..., 0] - ox
    py = states_world[..., 1] - oy
    out[..., 0] = c * px + s * py
    out[..., 1] = -s * px + c * py
    # velocity (no translation)
    vx = states_world[..., 2]
    vy = states_world[..., 3]
    out[..., 2] = c * vx + s * vy
    out[..., 3] = -s * vx + c * vy
    # acceleration
    ax = states_world[..., 4]
    ay = states_world[..., 5]
    out[..., 4] = c * ax + s * ay
    out[..., 5] = -s * ax + c * ay
    return out
