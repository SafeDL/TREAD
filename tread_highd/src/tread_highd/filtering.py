"""
filtering.py — 事件后处理与风险标注
=====================================
对抽取的事件进行运动学有效性检查和风险分层标签生成。

╔══════════════════════════════════════════════════════════════╗
║  EVT 方法论原则 (Extreme Value Theory)                      ║
║                                                              ║
║  • 事件筛选阶段 仅使用 语义/运动学 规则                     ║
║    (同车道、连续帧数、车辆类型、窗口完整性 等)              ║
║                                                              ║
║  • 风险评分 (TTC/THW/DRAC/综合分) 仅作为后处理标注          ║
║    绝不用于过滤事件 — 保留自然暴露分布下的完整事件总体      ║
║                                                              ║
║  • 尾部分位标签 (P90/P95/P99) 为信息性注释                  ║
║    供下游 EVT 拟合时选择阈值参考，不作为筛选依据            ║
║                                                              ║
║  如果在此阶段用风险阈值预筛高危事件，会产生选择偏差         ║
║  (selection bias)，导致后续 EVT 尾部拟合失效。               ║
╚══════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .schema import EventRecord
from dataclasses import asdict

logger = logging.getLogger(__name__)


def events_to_dataframe(events):
    """将 EventRecord 列表转换为 DataFrame。"""
    if not events:
        return pd.DataFrame()
    records = [asdict(e) for e in events]
    return pd.DataFrame(records)


def assign_risk_percentiles(events_df, tail_quantiles=None):
    """按事件类型分别计算风险百分位数并添加尾部标签列。

    注意: 此函数仅做标注 (annotation)，不做过滤 (filtering)。
    所有有效事件（无论风险高低）均保留在数据集中，以维持自然暴露
    分布的完整性，满足 EVT 建模对无偏总体的要求。

    Parameters
    ----------
    events_df : pd.DataFrame
    tail_quantiles : list[float]
        如 [0.90, 0.95, 0.99]，用于生成尾部百分位标签。

    Returns
    -------
    pd.DataFrame
        增加了 risk_percentile 和 tail_label_XX 列的 DataFrame。
    """
    if tail_quantiles is None:
        tail_quantiles = [0.90, 0.95, 0.99]

    events_df = events_df.copy()
    events_df["risk_percentile"] = 0.0

    for event_type in events_df["event_type"].unique():
        mask = (events_df["event_type"] == event_type) & events_df["is_valid"]
        scores = events_df.loc[mask, "risk_score"]
        if len(scores) == 0:
            continue

        # 百分位排名 (0~1)
        percentiles = scores.rank(pct=True)
        events_df.loc[mask, "risk_percentile"] = percentiles.values

        # 尾部分位标签 — 信息性注释，不用于过滤
        for q in tail_quantiles:
            col = f"tail_label_{int(q * 100)}"
            threshold = scores.quantile(q)
            events_df.loc[mask, col] = scores >= threshold

    return events_df


def filter_events(events_df, config):
    """应用运动学有效性检查（不使用风险评分过滤）。

    ╔════════════════════════════════════════════════════════╗
    ║  此函数仅检查数据完整性和运动学合理性。               ║
    ║  绝不根据 risk_score / TTC / DRAC 的数值大小过滤。    ║
    ║  风险评分为 0 或 NaN 的事件仍保留在数据集中。         ║
    ╚════════════════════════════════════════════════════════╝

    Parameters
    ----------
    events_df : pd.DataFrame
    config : dict

    Returns
    -------
    pd.DataFrame
    """
    initial = len(events_df)
    # 已在上游 (windowing / event_extraction) 完成的运动学过滤:
    #   - insufficient_window_frames: 公共帧不足 64 帧
    #   - missing_ego_frames / missing_target_frames: 窗口内帧缺失
    #   - excessive_negative_gap: 间距大量为负（数据对齐异常）
    #   - ego_abnormal_acceleration: 加速度超阈值
    #
    # 此处不再做额外的风险值过滤。
    # 风险评分 (包括 NaN 和 0) 仅作为事件属性保留。

    valid_count = events_df["is_valid"].sum()
    invalid_count = initial - valid_count

    logger.info(
        "事件有效性统计: 总计=%d, 有效=%d, 无效=%d (均为上游运动学过滤)",
        initial, valid_count, invalid_count,
    )

    # 记录无效原因分布
    if invalid_count > 0:
        reasons = events_df[~events_df["is_valid"]]["filter_reason"].value_counts()
        for reason, count in reasons.items():
            logger.info("  过滤原因: %s → %d 个事件", reason, count)

    return events_df
