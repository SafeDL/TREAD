"""
filtering.py — 事件后处理与风险标注
=====================================
EVT 方法论: 事件筛选仅用语义/运动学规则，风险评分仅作标注，
尾部分位标签供下游 EVT 拟合参考，不作为筛选依据。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import asdict

logger = logging.getLogger(__name__)


def events_to_dataframe(events):
    """将 EventRecord 列表转换为 DataFrame。"""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])


def assign_risk_percentiles(events_df, tail_quantiles=None):
    """按事件类型计算风险百分位数和尾部标签 (仅标注，不过滤)。"""
    if tail_quantiles is None:
        tail_quantiles = [0.90, 0.95, 0.99]

    events_df = events_df.copy()
    events_df["risk_percentile"] = 0.0
    for q in tail_quantiles:
        events_df[f"tail_label_{int(q * 100)}"] = False

    for event_type in events_df["event_type"].unique():
        mask = (events_df["event_type"] == event_type) & events_df["is_valid"]
        scores = events_df.loc[mask, "risk_score"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) == 0:
            continue
        events_df.loc[scores.index, "risk_percentile"] = scores.rank(pct=True).values
        for q in tail_quantiles:
            col = f"tail_label_{int(q * 100)}"
            events_df.loc[scores.index, col] = scores >= scores.quantile(q)

    return events_df


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
