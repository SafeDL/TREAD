"""
scenario_frame.py — Canonical Scenario Context Schema
=====================================================

本模块定义 TREAD 三阶段共享的"场景上下文契约":

    DeepEVT.context  ==  Diffusion.condition  ==  MATLAB/RoadRunner.scenario_init

只要三阶段都依赖本文件中的常量与变换函数，就能保证：
    1. 同一事件在三处的 context 数值一致；
    2. context 的物理含义与坐标系一致；
    3. 任何上下文扩展只需在此处统一修改。

关键约定
--------
1. **Ego-current coordinate frame**
   场景坐标系原点位于 ego 在 prefix-window 末端当前帧的几何中心，
   x 轴沿 ego 当前航向 (+x 始终指向 ego forward)。
   highD 数据在 ``preprocess.normalize_driving_direction`` 已经把所有车
   翻转到 +x 行进，因此本文件的旋转部分简化为零；保留旋转矩阵接口
   方便未来扩展到非 highD 数据集 (例如带横摆角的弯道场景)。

2. **Canonical fields**
   ``CanonicalScenarioContext`` 列出三阶段必须共享的当前场景参数。
   DeepEVT context features / diffusion condition / scenario_init.json
   都应该是这些字段、prefix 统计量或 extras 的一一映射。

3. **场景时长 / time horizon**
   每个事件都应导出 ``time_horizon_s`` (risk window 物理时长)
   以及 ``planned_cutin_duration`` (cut-in 专用)，以便 MATLAB 场景
   按相同时长实例化。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Schema version — 三阶段共同读这串字符串校验是否兼容
# ---------------------------------------------------------------------------
SCENARIO_CONTEXT_SCHEMA_VERSION = "1.1.0"

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
# Canonical scenario init schema
# ---------------------------------------------------------------------------

@dataclass
class CanonicalScenarioContext:
    """三阶段共享的场景当前条件。

    所有空间量都在 ego-current frame 中表达 (prefix 末端 ego @ origin, heading = +x)。
    所有时间量单位均为秒。
    """

    # --- 元信息 ---
    event_id: str
    event_type: str            # "following" / "cut_in"
    schema_version: str = SCENARIO_CONTEXT_SCHEMA_VERSION

    # --- ego current state (以原点为基准，因此应当全部为 0) ---
    ego_x0: float = 0.0
    ego_y0: float = 0.0
    ego_v0: float = 0.0
    ego_vy0: float = 0.0
    ego_ax0: float = 0.0
    ego_ay0: float = 0.0
    ego_length: float = 4.5
    ego_width: float = 1.8

    # --- target current state (in ego-current frame) ---
    # MATLAB/RoadRunner actor placement MUST use target_center_x0 / target_center_y0.
    # DeepEVT / risk features use initial_gap (= net longitudinal gap).
    target_center_x0: float = 0.0   # target 几何中心 x (ego-current frame)
    target_center_y0: float = 0.0   # target 几何中心 y (ego-current frame)
    initial_gap: float = 0.0        # 净纵向间距 (= center_x0 - 0.5*(L_ego+L_target))
    initial_lateral_offset: float = 0.0  # 横向偏移 (= center_y0, 因 ego 在 y=0)
    target_dx0: float = 0.0         # [deprecated] 净纵向间距，请用 initial_gap
    target_dy0: float = 0.0         # [deprecated] 横向偏移，请用 target_center_y0
    target_v0: float = 0.0          # target 的 vx (世界量级)，不是相对速度
    target_vy0: float = 0.0
    target_ax0: float = 0.0
    target_ay0: float = 0.0
    target_length: float = 4.5
    target_width: float = 1.8
    relative_speed_0: float = 0.0   # ego_v0 - target_v0

    # --- 车道几何 ---
    source_lane_id: Optional[int] = None
    target_lane_id: Optional[int] = None
    same_lane_initial: bool = True

    # --- 时间维度 ---
    time_horizon_s: float = 0.0     # risk window 总时长
    prefix_horizon_s: float = 0.0   # DeepEVT 编码的 prefix 时长
    planned_cutin_duration: float = 0.0   # cut-in 专用

    # --- 自由扩展项 (event-type-specific) ---
    extras: Dict[str, float] = field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# Canonical context build helpers
# ---------------------------------------------------------------------------

def build_canonical_context(
    *,
    event_id: str,
    event_type: str,
    states_ego_frame: np.ndarray,    # [prefix_steps, actors, state_features] in ego-current frame
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
    fps: float,
    prefix_steps: int,
    analysis_window_steps: Optional[int] = None,
    source_lane: Optional[int] = None,
    target_lane: Optional[int] = None,
    planned_cutin_duration: float = 0.0,
    extras: Optional[Dict[str, float]] = None,
) -> CanonicalScenarioContext:
    """从已对齐到 ego-current frame 的 prefix 状态张量构造 canonical context."""
    if states_ego_frame.ndim != 3 or states_ego_frame.shape[1] != 2:
        raise ValueError("states_ego_frame must be [time_steps, actors, state_features]")
    prefix_len = states_ego_frame.shape[0]
    current_index = max(0, min(prefix_steps, prefix_len) - 1)
    s0_ego = states_ego_frame[current_index, 0]
    s0_tgt = states_ego_frame[current_index, 1]
    horizon_steps = int(analysis_window_steps) if analysis_window_steps is not None else prefix_len

    target_center_x0 = float(s0_tgt[0])
    target_center_y0 = float(s0_tgt[1])
    initial_gap = float(s0_tgt[0] - 0.5 * (ego_length + target_length))
    initial_lateral_offset = float(s0_tgt[1])
    target_dx0 = initial_gap
    target_dy0 = float(s0_tgt[1])

    # 车道是否同道
    same_lane = bool(source_lane == target_lane) if (source_lane is not None and target_lane is not None) else (event_type == "following")

    return CanonicalScenarioContext(
        event_id=event_id,
        event_type=event_type,
        ego_x0=0.0, ego_y0=0.0,
        ego_v0=float(s0_ego[2]), ego_vy0=float(s0_ego[3]),
        ego_ax0=float(s0_ego[4]), ego_ay0=float(s0_ego[5]),
        ego_length=float(ego_length), ego_width=float(ego_width),
        target_center_x0=target_center_x0,
        target_center_y0=target_center_y0,
        initial_gap=initial_gap,
        initial_lateral_offset=initial_lateral_offset,
        target_dx0=target_dx0,
        target_dy0=target_dy0,
        target_v0=float(s0_tgt[2]),
        target_vy0=float(s0_tgt[3]),
        target_ax0=float(s0_tgt[4]),
        target_ay0=float(s0_tgt[5]),
        target_length=float(target_length),
        target_width=float(target_width),
        relative_speed_0=float(s0_ego[2] - s0_tgt[2]),
        source_lane_id=int(source_lane) if source_lane is not None else None,
        target_lane_id=int(target_lane) if target_lane is not None else None,
        same_lane_initial=same_lane,
        time_horizon_s=float(horizon_steps / max(fps, 1.0)),
        prefix_horizon_s=float(min(prefix_steps, prefix_len) / max(fps, 1.0)),
        planned_cutin_duration=float(planned_cutin_duration),
        extras=dict(extras or {}),
    )


# ---------------------------------------------------------------------------
# Canonical context -> DeepEVT context_features 映射 (short-history 版本)
# ---------------------------------------------------------------------------
# 当前状态特征映射到 CanonicalScenarioContext 的 prefix 末端字段；prefix 统计
# 与配置常量映射到 extras，避免和未来风险标签字段混淆。
FOLLOWING_CONTEXT_TO_CANONICAL: Dict[str, str] = {
    "ego_vx_current":                 "ego_v0",
    "lead_vx_current":                "target_v0",
    "relative_speed_current":         "relative_speed_0",
    "gap_current":                    "initial_gap",
    "lateral_offset_current":         "initial_lateral_offset",
    "ego_ax_current":                 "ego_ax0",
    "lead_ax_current":                "target_ax0",
    "gap_change_rate":                "extras.gap_change_rate",
    "relative_speed_trend":           "extras.relative_speed_trend",
    "relative_acceleration":          "extras.relative_acceleration",
    "ego_acc_mean_over_prefix":       "extras.ego_acc_mean_over_prefix",
    "lead_acc_mean_over_prefix":      "extras.lead_acc_mean_over_prefix",
    "lead_brake_indicator":           "extras.lead_brake_indicator",
    "min_gap_in_prefix":              "extras.min_gap_in_prefix",
    "max_closing_speed_in_prefix":    "extras.max_closing_speed_in_prefix",
    "lateral_offset_change_rate":     "extras.lateral_offset_change_rate",
    "lane_width":                     "extras.lane_width",
    "dt":                             "extras.dt",
    "horizon_steps":                  "extras.horizon_steps",
    "prefix_steps":                   "extras.prefix_steps",
}

CUTIN_CONTEXT_TO_CANONICAL: Dict[str, str] = {
    "ego_vx0":                 "ego_v0",
    "target_vx0":              "target_v0",
    "relative_speed_0":        "relative_speed_0",
    "target_center_x0":        "target_center_x0",
    "target_center_y0":        "target_center_y0",
    "initial_gap":             "initial_gap",
    "initial_lateral_offset":  "initial_lateral_offset",
    "target_vy0":              "target_vy0",
    "target_ax0":              "target_ax0",
    "target_ay0":              "target_ay0",
    "lane_width":              "extras.lane_width",
    "target_final_y":          "extras.target_final_y",
    "dt":                      "extras.dt",
    "horizon_steps":           "extras.horizon_steps",
}


def context_to_canonical_mapping(event_type: str) -> Dict[str, str]:
    if event_type == "following":
        return dict(FOLLOWING_CONTEXT_TO_CANONICAL)
    if event_type == "cut_in":
        return dict(CUTIN_CONTEXT_TO_CANONICAL)
    raise ValueError(f"Unsupported event_type: {event_type}")
