#!/usr/bin/env python3
"""
extract_highd_events.py — 从 highD 中抽取驾驶事件
=====================================================
输出:
  results/highd_events/events.csv

用法:
  conda activate tread
  python process_highD/scripts/extract_highd_events.py
"""
import logging
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from process_highD.src.event_extraction import extract_cutin_events, extract_following_segments
from process_highD.src.filtering import events_to_dataframe
from process_highD.src.io_utils import ensure_dir, load_config, resolve_data_path, resolve_recording_ids
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import filter_abnormal_tracks, normalize_driving_direction, resample_recording
from process_highD.src.quality_check import generate_quality_report
from utils.highd_exposure import (
    all_vehicle_exposure_for_recording,
    following_exposure_for_recording,
)
from utils.highd_cutin import (
    build_highd_cutin_event_rows_from_recording,
    highd_cutin_options_from_config,
    highd_cutin_score_table,
    save_highd_cutin_event_context_cache,
    score_highd_cutin_event_rows,
)
from utils.highd_longitudinal import (
    build_highd_event_rows_from_recording,
    highd_options_from_config,
    highd_score_table,
    save_highd_event_context_cache,
    score_highd_event_rows,
)
from utils.io import write_csv, write_json

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
SCRIPT_DEFAULTS = {"log_level": logging.INFO}
FOLLOWING_SCORE_CACHE = "following_event_scores.csv"
FOLLOWING_CONTEXT_CACHE = "following_event_contexts.npz"
FOLLOWING_CACHE_SUMMARY = "following_event_cache_summary.json"
CUTIN_SCORE_CACHE = "cutin_event_scores.csv"
CUTIN_CONTEXT_CACHE = "cutin_event_contexts.npz"
CUTIN_CACHE_SUMMARY = "cutin_event_cache_summary.json"
EXPOSURE_PER_RECORDING_CSV = "exposure_per_recording.csv"


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
    risk_options = highd_options_from_config(cfg)
    cutin_risk_options = highd_cutin_options_from_config(cfg)

    target_fps = cfg.get("sampling", {}).get("target_fps", 10)
    all_events = []
    following_rows = []
    following_skipped = 0
    cutin_rows = []
    cutin_skipped = 0
    exposure_rows = []

    for rid in tqdm(ids, desc="Extracting events"):
        rec = load_recording(raw_dir, rid)
        rec = normalize_driving_direction(rec)
        rec = filter_abnormal_tracks(rec, cfg)
        rec = resample_recording(rec, target_fps)
        recording_events = []
        recording_events.extend(extract_following_segments(rec, cfg))
        recording_events.extend(extract_cutin_events(rec, cfg))
        all_events.extend(recording_events)

        recording_df = events_to_dataframe(recording_events)
        if len(recording_df) == 0:
            continue
        following = recording_df[
            (recording_df["event_type"] == "following")
            & (recording_df["is_valid"].astype(bool))
        ]
        cutin = recording_df[
            (recording_df["event_type"] == "cut_in")
            & (recording_df["is_valid"].astype(bool))
        ]

        # 曝光计算：与事件提取在同一遍历中完成，避免二次加载原始数据
        exposure_rows.append({
            **following_exposure_for_recording(
                following.copy() if not following.empty else following,
                recording_id=rid,
                get_track=rec.get_vehicle_track,
                fps=float(rec.recording_meta.get("frameRate", target_fps)),
            ),
            **all_vehicle_exposure_for_recording(
                recording_id=rid,
                vehicle_ids=rec.vehicle_ids(),
                get_track=rec.get_vehicle_track,
                fps=float(rec.recording_meta.get("frameRate", target_fps)),
            ),
        })

        if not following.empty:
            rows, skipped = build_highd_event_rows_from_recording(
                rec,
                following.reset_index(drop=True),
                options=risk_options,
            )
            score_highd_event_rows(rows, options=risk_options)
            following_rows.extend(rows)
            following_skipped += int(skipped)

        if not cutin.empty:
            rows, skipped = build_highd_cutin_event_rows_from_recording(
                rec,
                cutin.reset_index(drop=True),
                options=cutin_risk_options,
            )
            score_highd_cutin_event_rows(rows, options=cutin_risk_options)
            cutin_rows.extend(rows)
            cutin_skipped += int(skipped)

    df = events_to_dataframe(all_events)
    if len(df) > 0:
        valid = df[df["is_valid"]]
        invalid = df[~df["is_valid"]]
        df.to_csv(out_dir / "events.csv", index=False)
        generate_quality_report(df, out_dir)
        if following_rows:
            score_path = out_dir / FOLLOWING_SCORE_CACHE
            context_path = out_dir / FOLLOWING_CONTEXT_CACHE
            write_csv(score_path, highd_score_table(following_rows))
            save_highd_event_context_cache(context_path, following_rows)
            write_json(
                out_dir / FOLLOWING_CACHE_SUMMARY,
                {
                    "score_cache": str(score_path),
                    "context_cache": str(context_path),
                    "num_following_contexts": int(len(following_rows)),
                    "skipped_following_contexts": int(following_skipped),
                },
            )
            logger.info(
                "following 风险缓存: %d 条, skipped=%d, 输出: %s 和 %s",
                len(following_rows),
                following_skipped,
                score_path,
                context_path,
            )
        else:
            logger.warning("没有生成 following 风险/context 缓存")
        if cutin_rows:
            score_path = out_dir / CUTIN_SCORE_CACHE
            context_path = out_dir / CUTIN_CONTEXT_CACHE
            write_csv(score_path, highd_cutin_score_table(cutin_rows))
            save_highd_cutin_event_context_cache(context_path, cutin_rows)
            write_json(
                out_dir / CUTIN_CACHE_SUMMARY,
                {
                    "score_cache": str(score_path),
                    "context_cache": str(context_path),
                    "num_cutin_contexts": int(len(cutin_rows)),
                    "skipped_cutin_contexts": int(cutin_skipped),
                    "risk_variable": "y_cutin",
                },
            )
            logger.info(
                "cut-in 风险缓存: %d 条, skipped=%d, 输出: %s 和 %s",
                len(cutin_rows),
                cutin_skipped,
                score_path,
                context_path,
            )
        else:
            logger.warning("没有生成 cut-in 风险/context 缓存")
        if exposure_rows:
            write_csv(out_dir / EXPOSURE_PER_RECORDING_CSV, exposure_rows)
            logger.info(
                "曝光 per-recording: %d 条记录, 输出: %s",
                len(exposure_rows),
                out_dir / EXPOSURE_PER_RECORDING_CSV,
            )
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
