#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_highd_events.py — 从 highD 中抽取驾驶事件
=====================================================
输出:
  data/events.csv
  data/candidate_events.csv
  data/invalid_events.csv

用法:
  conda activate jzm
  python scripts/extract_highd_events.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running either from the repository root or from tread_highd/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from process_highD.src.io_utils import load_config, resolve_data_path, ensure_dir, resolve_recording_ids
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import normalize_driving_direction, filter_abnormal_tracks, resample_recording
from process_highD.src.event_extraction import extract_following_segments, extract_cutin_events
from process_highD.src.filtering import events_to_dataframe
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="TREAD: Extract highD events")
    default_config = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
    parser.add_argument("--config", default=str(default_config), help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("extract")

    cfg = load_config(args.config)
    raw_dir_path = resolve_data_path(cfg["paths"]["raw_dir"], args.config)
    validate_raw_dir(raw_dir_path)
    raw_dir = str(raw_dir_path)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["output_dir"], args.config)))
    ensure_dir(out_dir)

    ids = resolve_recording_ids(raw_dir, cfg.get("recordings", {}))
    logger.info("将处理 recording IDs: %s", ids)

    target_fps = cfg.get("sampling", {}).get("target_fps", 10)
    all_events = []

    for rid in tqdm(ids, desc="Extracting events"):
        try:
            rec = load_recording(raw_dir, rid)
            rec = normalize_driving_direction(rec)
            rec = filter_abnormal_tracks(rec, cfg)
            rec = resample_recording(rec, target_fps)
            try:
                all_events.extend(extract_following_segments(rec, cfg))
            except Exception as e:
                logger.error("Recording %02d following extraction failed: %s", rid, e)
            try:
                all_events.extend(extract_cutin_events(rec, cfg))
            except Exception as e:
                logger.error("Recording %02d cut-in extraction failed: %s", rid, e)
        except Exception as e:
            logger.error("Recording %02d failed: %s", rid, e)

    df = events_to_dataframe(all_events)
    if len(df) > 0:
        valid = df[df["is_valid"]]
        invalid = df[~df["is_valid"]]
        df.to_csv(out_dir / "events.csv", index=False)
        valid.to_csv(out_dir / "candidate_events.csv", index=False)
        invalid.to_csv(out_dir / "invalid_events.csv", index=False)
        logger.info("事件总数: %d, 候选事件: %d, 无效事件: %d", len(df), len(valid), len(invalid))
    else:
        logger.warning("未提取到任何事件!")

    logger.info("完成! 输出: %s", out_dir)


if __name__ == "__main__":
    main()
