"""
quality_check.py — 数据质量报告
=================================
生成可诊断的数据质量报告。
"""
from __future__ import annotations
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_quality_report(events_df, output_dir):
    """生成质量报告 JSON。

    Returns
    -------
    dict
        质量报告字典。
    """
    has_rows = len(events_df) > 0
    cutin = events_df["event_type"] == "cut_in" if has_rows else []
    following = events_df["event_type"] == "following" if has_rows else []
    valid = events_df["is_valid"] if has_rows else []
    report = {
        "num_recordings": (
            int(events_df["recording_id"].nunique()) if has_rows else 0
        ),
        "num_candidate_cutin": int(cutin.sum()) if has_rows else 0,
        "num_valid_cutin": int((cutin & valid).sum()) if has_rows else 0,
        "num_candidate_following": int(following.sum()) if has_rows else 0,
        "num_valid_following": int((following & valid).sum()) if has_rows else 0,
    }

    # 过滤原因统计
    if len(events_df) > 0 and "filter_reason" in events_df.columns:
        reasons = (
            events_df[~events_df["is_valid"]]["filter_reason"]
            .value_counts()
            .to_dict()
        )
        report["filter_reasons"] = {k: int(v) for k, v in reasons.items()}
    else:
        report["filter_reasons"] = {}

    # 保存
    out_path = Path(output_dir) / "quality_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("质量报告已保存: %s", out_path)

    return report
