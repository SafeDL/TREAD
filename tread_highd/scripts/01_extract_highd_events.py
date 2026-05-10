#!/usr/bin/env python
"""
01_extract_highd_events.py — 从 highD 中抽取候选事件
=====================================================
输出:
  processed/intermediate/candidate_events.csv
  processed/intermediate/invalid_events.csv

用法:
  conda activate jzm
  python scripts/01_extract_highd_events.py --config configs/highd_default.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

# 将 src 加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tread_highd.io_utils import load_config, resolve_data_path, ensure_dir
from tread_highd.loader import load_recording
from tread_highd.preprocess import normalize_driving_direction, filter_abnormal_tracks, resample_recording
from tread_highd.event_extraction import extract_following_segments, extract_cutin_events
from tread_highd.filtering import events_to_dataframe
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="TREAD: Extract highD events")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--recordings", default="all", help="'all' or comma-separated IDs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("extract")

    cfg = load_config(args.config)
    raw_dir = str(resolve_data_path(cfg["paths"]["raw_dir"], args.config))
    out_dir = Path(str(resolve_data_path(cfg["paths"]["processed_dir"], args.config))) / "intermediate"
    ensure_dir(out_dir)

    # 确定 recording IDs
    if args.recordings == "all":
        ids = sorted(int(p.stem.split("_")[0]) for p in Path(raw_dir).glob("*_tracks.csv"))
        exclude = cfg.get("recordings", {}).get("exclude", [])
        ids = [i for i in ids if i not in exclude]
    else:
        ids = [int(x.strip()) for x in args.recordings.split(",")]

    target_fps = cfg.get("sampling", {}).get("target_fps", 10)
    all_events = []

    for rid in tqdm(ids, desc="Extracting events"):
        try:
            rec = load_recording(raw_dir, rid)
            rec = normalize_driving_direction(rec)
            rec = filter_abnormal_tracks(rec, cfg)
            rec = resample_recording(rec, target_fps)
            fol = extract_following_segments(rec, cfg)
            cin = extract_cutin_events(rec, cfg)
            all_events.extend(fol + cin)
        except Exception as e:
            logger.error("Recording %02d failed: %s", rid, e)

    df = events_to_dataframe(all_events)
    if len(df) > 0:
        valid = df[df["is_valid"]]
        invalid = df[~df["is_valid"]]
        valid.to_csv(out_dir / "candidate_events.csv", index=False)
        invalid.to_csv(out_dir / "invalid_events.csv", index=False)
        logger.info("候选事件: %d, 无效事件: %d", len(valid), len(invalid))
    else:
        logger.warning("未提取到任何事件!")

    logger.info("完成! 输出: %s", out_dir)


if __name__ == "__main__":
    main()
