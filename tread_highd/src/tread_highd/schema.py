"""
schema.py — 数据结构定义
========================
定义 EventRecord dataclass 以及轨迹张量 schema 常量。
"""

from dataclasses import dataclass, field
from typing import Optional


# ─── 状态特征索引 ───────────────────────────────────────
FEATURE_NAMES = [
    "dx", "dy", "dvx", "dvy",
    "vx", "vy", "ax", "ay",
    "lane_id_normalized", "length", "width",
]
NUM_FEATURES = len(FEATURE_NAMES)  # 11
NUM_ACTORS = 2                      # ego(0) + target(1)
DEFAULT_WINDOW_LENGTH = 64          # T


@dataclass
class EventRecord:
    """单个交互事件的完整描述。"""

    event_id: str = ""
    event_type: str = ""  # "cut_in" 或 "following"
    recording_id: int = -1
    ego_id: int = -1
    target_id: int = -1

    start_frame: int = -1
    end_frame: int = -1
    anchor_frame: int = -1

    # ── cut-in 专用 ──
    cross_frame: Optional[int] = None
    cutin_start_frame: Optional[int] = None
    cutin_end_frame: Optional[int] = None
    source_lane: Optional[int] = None
    target_lane: Optional[int] = None

    # ── 核心风险指标 ──
    min_ttc: float = float("nan")
    min_thw: float = float("nan")
    max_drac: float = float("nan")
    risk_score: float = float("nan")

    # ── 场景统计量 ──
    initial_gap: float = float("nan")
    min_gap: float = float("nan")
    initial_relative_speed: float = float("nan")
    post_cutin_gap: float = float("nan")
    cutin_duration: float = float("nan")

    # ── 质量标记 ──
    is_valid: bool = True
    filter_reason: str = ""
