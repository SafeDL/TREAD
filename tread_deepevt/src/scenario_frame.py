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
1. **Ego-initial coordinate frame**
   场景坐标系原点位于 ego 在 analysis-window 起始帧的几何中心，
   x 轴沿 ego 初始航向 (+x 始终指向 ego forward)。
   highD 数据在 ``preprocess.normalize_driving_direction`` 已经把所有车
   翻转到 +x 行进，因此本文件的旋转部分简化为零；保留旋转矩阵接口
   方便未来扩展到非 highD 数据集 (例如带横摆角的弯道场景)。

2. **Canonical fields**
   ``CanonicalScenarioContext`` 列出三阶段必须共享的初始场景参数。
   DeepEVT context features / diffusion condition / scenario_init.json
   都应该是这些字段 (或其严格子集) 的一一映射。

3. **场景时长 / time horizon**
   每个事件都应导出 ``time_horizon_s`` (analysis window 物理时长)
   以及 ``planned_cutin_duration`` (cut-in 专用)，以便 MATLAB 场景
   按相同时长实例化。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Schema version — 三阶段共同读这串字符串校验是否兼容
# ---------------------------------------------------------------------------
SCENARIO_CONTEXT_SCHEMA_VERSION = "1.0.0"

# ego-initial frame 中各通道的语义命名 (通用 actor schema,每个 actor 共用同一顺序)。
# 注意: 这些字段描述的是 **ego-initial 坐标系下任一 actor** 的状态通道,
# actor 0 = ego、actor 1 = target 时含义一致,不能混入 "_ego" 后缀避免误解。
CANONICAL_STATE_FEATURES: Tuple[str, ...] = (
    "x",   # position x in ego-initial frame, +x = ego forward
    "y",   # position y in ego-initial frame, +y = ego left
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
    """三阶段共享的场景初始条件。

    所有空间量都在 ego-initial frame 中表达 (ego @ origin, heading = +x)。
    所有时间量单位均为秒。
    """

    # --- 元信息 ---
    event_id: str
    event_type: str            # "following" / "cut_in"
    schema_version: str = SCENARIO_CONTEXT_SCHEMA_VERSION

    # --- ego initial state (以原点为基准，因此应当全部为 0) ---
    ego_x0: float = 0.0
    ego_y0: float = 0.0
    ego_v0: float = 0.0
    ego_vy0: float = 0.0
    ego_ax0: float = 0.0
    ego_ay0: float = 0.0
    ego_length: float = 4.5
    ego_width: float = 1.8

    # --- target initial state (in ego-initial frame) ---
    # MATLAB/RoadRunner actor placement MUST use target_center_x0 / target_center_y0.
    # DeepEVT / risk features use initial_gap (= net longitudinal gap).
    target_center_x0: float = 0.0   # target 几何中心 x (ego-initial frame)
    target_center_y0: float = 0.0   # target 几何中心 y (ego-initial frame)
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
    time_horizon_s: float = 0.0     # analysis window 总时长
    prefix_horizon_s: float = 0.0   # DeepEVT 编码的 prefix 时长
    planned_cutin_duration: float = 0.0   # cut-in 专用

    # --- 自由扩展项 (event-type-specific) ---
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Ego-initial frame 变换
# ---------------------------------------------------------------------------

def compute_ego_initial_frame(
    ego_state_t0: np.ndarray, world_heading_x: float = 1.0, world_heading_y: float = 0.0,
) -> Dict[str, float]:
    """从 ego 在 t=0 的状态构造 (origin, rotation) 描述。

    Parameters
    ----------
    ego_state_t0 : np.ndarray, shape [F]
        ego 在 t=0 的状态 (x, y, vx, vy, ax, ay)。
    world_heading_x, world_heading_y : float
        ego 初始航向在世界坐标系下的方向向量。highD 已经统一为 +x 方向，
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
        "origin_x": float(ego_state_t0[0]),
        "origin_y": float(ego_state_t0[1]),
        "rot_cos": rot_cos,
        "rot_sin": rot_sin,
    }


def world_to_ego_states(states_world: np.ndarray, frame: Dict[str, float]) -> np.ndarray:
    """``states_world`` shape ``[T, A, F]`` 中的 (x, y) 与 (vx, vy)、(ax, ay)
    转到 ego-initial frame；返回相同 shape。

    F 顺序: (x, y, vx, vy, ax, ay)。其它维度保持原样。
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


def ego_to_world_xy(
    xy_ego: np.ndarray, frame: Dict[str, float],
) -> np.ndarray:
    """逆变换 — 把 ego-initial frame 下的 (x, y) 还原到世界坐标。
    给 diffusion 后期生成的轨迹回投到原始 highD 用，完成闭环。
    """
    c = frame["rot_cos"]; s = frame["rot_sin"]
    out = np.empty_like(xy_ego)
    out[..., 0] = c * xy_ego[..., 0] - s * xy_ego[..., 1] + frame["origin_x"]
    out[..., 1] = s * xy_ego[..., 0] + c * xy_ego[..., 1] + frame["origin_y"]
    return out


# ---------------------------------------------------------------------------
# Canonical context build helpers
# ---------------------------------------------------------------------------

def build_canonical_context(
    *,
    event_id: str,
    event_type: str,
    states_ego_frame: np.ndarray,    # [T, 2, F] in ego-initial frame
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
    fps: float,
    prefix_steps: int,
    source_lane: Optional[int] = None,
    target_lane: Optional[int] = None,
    planned_cutin_duration: float = 0.0,
    extras: Optional[Dict[str, float]] = None,
) -> CanonicalScenarioContext:
    """从已对齐到 ego-initial frame 的状态张量构造 canonical context."""
    if states_ego_frame.ndim != 3 or states_ego_frame.shape[1] != 2:
        raise ValueError("states_ego_frame must be [T, 2, F]")
    s0_ego = states_ego_frame[0, 0]
    s0_tgt = states_ego_frame[0, 1]
    T = states_ego_frame.shape[0]

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
        time_horizon_s=float(T / max(fps, 1.0)),
        prefix_horizon_s=float(min(prefix_steps, T) / max(fps, 1.0)),
        planned_cutin_duration=float(planned_cutin_duration),
        extras=dict(extras or {}),
    )


# ---------------------------------------------------------------------------
# Canonical context -> DeepEVT context_features 映射 (initial-context 版本)
# ---------------------------------------------------------------------------
# 每个 DeepEVT context feature 均从 CanonicalScenarioContext 的 t=0 字段
# (或 extras) 一一映射，不依赖 prefix 轨迹。Diffusion 与 MATLAB 解析时
# 只需初始场景参数即可完全复现 DeepEVT 的条件输入。

# 第一版使用 initial-context 特征：所有 context feature 均可从 t=0 状态直接读取，
# 不需要 prefix 轨迹窗口。这保证 DeepEVT / Diffusion / MATLAB 三阶段闭环。
FOLLOWING_CONTEXT_TO_CANONICAL: Dict[str, str] = {
    "ego_v0":                 "ego_v0",
    "lead_v0":                "target_v0",
    "relative_speed_0":       "relative_speed_0",
    "gap_0":                  "initial_gap",
    "ego_accel_0":            "ego_ax0",
    "lead_accel_0":           "target_ax0",
    "thw_0":                  "extras.thw_0",
}

CUTIN_CONTEXT_TO_CANONICAL: Dict[str, str] = {
    "ego_v0":                  "ego_v0",
    "target_v0":               "target_v0",
    "relative_speed_0":        "relative_speed_0",
    "initial_dx":              "initial_gap",
    "initial_dy":              "initial_lateral_offset",
    "target_vy_0":             "target_vy0",
    "target_ax_0":             "target_ax0",
    "target_ay_0":             "target_ay0",
}


def context_to_canonical_mapping(event_type: str) -> Dict[str, str]:
    if event_type == "following":
        return dict(FOLLOWING_CONTEXT_TO_CANONICAL)
    if event_type == "cut_in":
        return dict(CUTIN_CONTEXT_TO_CANONICAL)
    raise ValueError(f"Unsupported event_type: {event_type}")
