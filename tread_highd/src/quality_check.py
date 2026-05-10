"""
quality_check.py — 数据质量报告
=================================
生成可诊断的数据质量报告。
"""
from __future__ import annotations
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_quality_report(events_df, output_dir):
    """生成质量报告 JSON。

    Returns
    -------
    dict
        质量报告字典。
    """
    report = {
        "num_recordings": int(events_df["recording_id"].nunique()) if len(events_df) > 0 else 0,
        "num_candidate_cutin": int((events_df["event_type"] == "cut_in").sum()) if len(events_df) > 0 else 0,
        "num_valid_cutin": int(((events_df["event_type"] == "cut_in") & events_df["is_valid"]).sum()) if len(events_df) > 0 else 0,
        "num_candidate_following": int((events_df["event_type"] == "following").sum()) if len(events_df) > 0 else 0,
        "num_valid_following": int(((events_df["event_type"] == "following") & events_df["is_valid"]).sum()) if len(events_df) > 0 else 0,
    }

    # 过滤原因统计
    if len(events_df) > 0 and "filter_reason" in events_df.columns:
        reasons = events_df[~events_df["is_valid"]]["filter_reason"].value_counts().to_dict()
        report["filter_reasons"] = {k: int(v) for k, v in reasons.items()}
    else:
        report["filter_reasons"] = {}

    # 风险分位数
    risk_quantiles = {}
    for etype in ["cut_in", "following"]:
        sub = events_df[(events_df["event_type"] == etype) & events_df["is_valid"]] if len(events_df) > 0 else pd.DataFrame()
        if len(sub) > 0:
            scores = sub["risk_score"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(scores) > 0:
                risk_quantiles[etype] = {
                    "q50": float(scores.quantile(0.50)),
                    "q90": float(scores.quantile(0.90)),
                    "q95": float(scores.quantile(0.95)),
                    "q99": float(scores.quantile(0.99)),
                }
            else:
                risk_quantiles[etype] = {"q50": 0, "q90": 0, "q95": 0, "q99": 0}
        else:
            risk_quantiles[etype] = {"q50": 0, "q90": 0, "q95": 0, "q99": 0}
    report["risk_quantiles"] = risk_quantiles

    # 保存
    out_path = Path(output_dir) / "quality_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("质量报告已保存: %s", out_path)

    return report
