"""
schema.py — 数据结构定义
========================
定义 EventRecord dataclass 以及轨迹张量 schema 常量。
"""

from dataclasses import dataclass, field
from typing import Optional



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

    # ── 质量标记 ──
    is_valid: bool = True
    filter_reason: str = ""
