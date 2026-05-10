"""
filtering.py — 事件后处理与风险标注
=====================================
事件筛选仅用语义/运动学规则，风险评分仅作描述性标注，
不在本模块生成 EVT 尾部分位标签。
"""
from __future__ import annotations
import logging
import pandas as pd
from dataclasses import asdict

logger = logging.getLogger(__name__)


def events_to_dataframe(events):
    """将 EventRecord 列表转换为 DataFrame。"""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])


def filter_events(events_df, config):
    """统计事件有效性 (不做风险值过滤，保持自然暴露分布)。"""
    valid_count = events_df["is_valid"].sum()
    invalid_count = len(events_df) - valid_count

    logger.info("事件有效性: 总计=%d, 有效=%d, 无效=%d",
                len(events_df), valid_count, invalid_count)

    if invalid_count > 0:
        reasons = events_df[~events_df["is_valid"]]["filter_reason"].value_counts()
        for reason, count in reasons.items():
            logger.info("  过滤原因: %s → %d", reason, count)

    return events_df
