#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_highd_events.py — 从 highD 中抽取驾驶事件
=====================================================
输出:
  results/highd_events/events.csv
  results/highd_events/candidate_events.csv
  results/highd_events/invalid_events.csv

用法:
  conda activate tread
  python process_highD/scripts/extract_highd_events.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from process_highD.src.event_extraction import extract_following_segments, extract_cutin_events
from process_highD.src.filtering import events_to_dataframe
from process_highD.src.io_utils import ensure_dir, load_config, resolve_data_path, resolve_recording_ids
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import filter_abnormal_tracks, normalize_driving_direction, resample_recording
from process_highD.src.quality_check import generate_quality_report
from tqdm import tqdm

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
SCRIPT_DEFAULTS = {"log_level": logging.INFO}


def validate_raw_dir(raw_dir: Path) -> None:
    """Fail early when the configured highD raw data directory is missing or empty."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"highD 原始数据目录不存在: {raw_dir}\n"
            "请把 highD CSV 文件放到该目录，或修改配置文件中的 paths.raw_dir。\n"
            "期望文件名示例: 01_tracks.csv, 01_tracksMeta.csv, 01_recordingMeta.csv"
        )
    tracks_files = sorted(raw_dir.glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(
            f"highD 原始数据目录中没有找到 *_tracks.csv: {raw_dir}\n"
            "请确认 raw_dir 指向包含 highD 原始 CSV 的目录。"
        )


def main():
    logging.basicConfig(
        level=SCRIPT_DEFAULTS["log_level"],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("extract")

    config_path = DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    raw_dir_path = resolve_data_path(cfg["paths"]["raw_dir"], config_path)
    validate_raw_dir(raw_dir_path)
    raw_dir = str(raw_dir_path)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["output_dir"], config_path)))
    ensure_dir(out_dir)

    ids = resolve_recording_ids(raw_dir, cfg.get("recordings", {}))
    logger.info("将处理 recording IDs: %s", ids)

    target_fps = cfg.get("sampling", {}).get("target_fps", 10)
    all_events = []

    for rid in tqdm(ids, desc="Extracting events"):
        rec = load_recording(raw_dir, rid)
        rec = normalize_driving_direction(rec)
        rec = filter_abnormal_tracks(rec, cfg)
        rec = resample_recording(rec, target_fps)
        all_events.extend(extract_following_segments(rec, cfg))
        all_events.extend(extract_cutin_events(rec, cfg))

    df = events_to_dataframe(all_events)
    if len(df) > 0:
        valid = df[df["is_valid"]]
        invalid = df[~df["is_valid"]]
        df.to_csv(out_dir / "events.csv", index=False)
        valid.to_csv(out_dir / "candidate_events.csv", index=False)
        invalid.to_csv(out_dir / "invalid_events.csv", index=False)
        generate_quality_report(df, out_dir)
        logger.info(
            "事件总数: %d, 候选事件: %d, 无效事件: %d",
            len(df),
            len(valid),
            len(invalid),
        )
    else:
        logger.warning("未提取到任何事件!")

    logger.info("完成! 输出: %s", out_dir)


if __name__ == "__main__":
    main()
