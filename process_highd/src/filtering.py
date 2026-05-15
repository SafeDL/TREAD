"""
filtering.py — 事件后处理与风险标注
=====================================
事件筛选仅用语义/运动学规则，风险评分仅作描述性标注，
不在本模块生成 EVT 尾部分位标签。
"""
from __future__ import annotations
import pandas as pd
from dataclasses import asdict


def events_to_dataframe(events):
    """将 EventRecord 列表转换为 DataFrame"""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])
