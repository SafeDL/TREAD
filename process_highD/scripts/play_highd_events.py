#!/usr/bin/env python
"""
play_highd_events.py — sequentially replay extracted highD events as a single mp4.

Usage:
  python scripts/play_highd_events.py
  python scripts/play_highd_events.py --event_type cut_in
  python scripts/play_highd_events.py --event_type following --max_events 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from process_highD.src.io_utils import ensure_dir, load_config, resolve_data_path
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)

LOGGER = logging.getLogger(__name__)
EVENT_ORDER_COLUMNS = ["recording_id", "start_frame", "end_frame", "event_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay highD events in chronological order and save as a single mp4."
    )
    default_config = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--events_csv", default=None, help="Defaults to output_dir/events.csv")
    parser.add_argument(
        "--event_type", default="cut_in", choices=["all", "following", "cut_in"],
        help="Which event type to replay: all, following, or cut_in",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_name", default=None, help="Output filename without extension")
    parser.add_argument("--pre_frames", type=int, default=25)
    parser.add_argument("--post_frames", type=int, default=25)
    parser.add_argument("--view_width", type=float, default=160.0)
    parser.add_argument("--neighbor_margin", type=float, default=20.0)
    parser.add_argument("--tail_frames", type=int, default=50)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--max_events", type=int, default=None)
    return parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def _is_valid_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_events(events_path: Path, event_type: str, max_events: int | None) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    if "is_valid" in events.columns:
        events = events[_is_valid_mask(events["is_valid"])]
    if event_type != "all":
        events = events[events["event_type"] == event_type]
    if events.empty:
        raise ValueError(f"No valid events found for event_type={event_type!r}.")
    order_cols = [c for c in EVENT_ORDER_COLUMNS if c in events.columns]
    events = events.sort_values(order_cols).reset_index(drop=True)
    if max_events is not None:
        events = events.head(max_events)
    return events


def _load_recording(config: dict, config_path: str, recording_id: int):
    raw_dir = resolve_data_path(config["paths"]["raw_dir"], config_path)
    rec = load_recording(str(raw_dir), recording_id)
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(
        config.get("sampling", {}).get("target_fps", rec.recording_meta.get("frameRate", 25))
    )
    return resample_recording(rec, target_fps)


def _safe_int(value, default=None):
    if pd.isna(value):
        return default
    return int(value)


def _frame_sequence(recording, event: pd.Series, pre: int, post: int) -> list[int]:
    start = _safe_int(event.get("start_frame")) - pre
    end = _safe_int(event.get("end_frame")) + post
    available = np.asarray(recording.frame_ids(), dtype=int)
    frames = available[(available >= start) & (available <= end)]
    if len(frames) == 0:
        raise ValueError(f"No frames in window [{start}, {end}].")
    return frames.astype(int).tolist()


def _track_centers(track: pd.DataFrame, frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows = track.loc[track.index.intersection(frames)]
    if rows.empty:
        return np.array([]), np.array([])
    x = rows["x"].to_numpy(float)
    y = -rows["y"].to_numpy(float)
    return x, y


def _lane_groups(recording) -> list[np.ndarray]:
    groups = []
    for key in ["upperLaneMarkings", "lowerLaneMarkings"]:
        vals = np.asarray(recording.recording_meta.get(key, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            groups.append(vals)
    return groups


def _row_box(row: pd.Series) -> tuple[float, float, float, float]:
    w, h = float(row["width"]), float(row["height"])
    return float(row["x"]) - w / 2.0, -float(row["y"]) - h / 2.0, w, h


def _vehicle_style(vid: int, ego_id: int, target_id: int) -> tuple[str, float, int]:
    if vid == ego_id:
        return "#e31a1c", 1.0, 4
    if vid == target_id:
        return "#1f78b4", 1.0, 4
    return "#d9d9d9", 0.55, 1


def _frame_title(event: pd.Series, frame: int, fps: float) -> str:
    anchor = _safe_int(event.get("anchor_frame"))
    cross = _safe_int(event.get("cross_frame"))
    marks = [f"frame={frame}", f"t={(frame - int(event['start_frame'])) / fps:.2f}s"]
    if anchor is not None:
        marks.append(f"anchor={anchor}")
    if cross is not None:
        marks.append(f"cross={cross}")
    return (
        f"{event['event_id']} ({event['event_type']}) | "
        f"ego={int(event['ego_id'])}, target={int(event['target_id'])}\n"
        + ", ".join(marks)
    )


# ── rendering ────────────────────────────────────────────────────────────────
def _build_frame_list(events_df: pd.DataFrame, recording_cache: dict, args) -> list:
    """Return [(recording, event_row, frame_id, within_event_idx), ...] sorted by recording+frame."""
    rows = []
    for _, event in events_df.iterrows():
        rec = recording_cache[int(event["recording_id"])]
        try:
            frames = _frame_sequence(rec, event, args.pre_frames, args.post_frames)
            for fi, frame in enumerate(frames):
                rows.append((rec, event, frame, fi, frames))
        except ValueError as e:
            LOGGER.warning("Skipping event %s: %s", event["event_id"], e)
    return rows


def _render_to_mp4(frame_list: list, args, output_path: Path) -> None:
    from matplotlib.animation import FFMpegWriter
    from tqdm import tqdm

    # Precompute vehicle tracks
    track_cache: dict[tuple, pd.DataFrame] = {}
    for rec, event, *_ in frame_list:
        rid = rec.recording_id
        for vid in (int(event["ego_id"]), int(event["target_id"])):
            if (rid, vid) not in track_cache:
                track_cache[(rid, vid)] = rec.get_vehicle_track(vid)

    fps = float(frame_list[0][0].recording_meta.get("frameRate", 25))
    half_width = args.view_width / 2.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#707070")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("mirrored y (m)")
    title_obj = ax.set_title("")
    ego_line, = ax.plot([], [], color="#e31a1c", lw=1.8, alpha=0.9, zorder=3)
    target_line, = ax.plot([], [], color="#1f78b4", lw=1.8, alpha=0.9, zorder=3)

    lane_artists: list = []
    frame_artists: list = []
    current_rid: list = [None]

    def _refresh_lanes(rec):
        for a in lane_artists:
            a.remove()
        lane_artists.clear()
        groups = _lane_groups(rec)
        markings = np.concatenate(groups) if groups else np.array([])
        ax.set_ylim(
            (-float(np.nanmax(markings)) - 2.0, -float(np.nanmin(markings)) + 2.0)
            if len(markings) else (-20.0, 5.0)
        )
        for group in groups:
            for j, y in enumerate(group):
                is_boundary = j == 0 or j == len(group) - 1
                lane_artists.append(ax.axhline(
                    -float(y), color="white",
                    lw=1.2 if is_boundary else 0.8,
                    ls="-" if is_boundary else "--",
                    alpha=0.9 if is_boundary else 0.65,
                ))

    writer = FFMpegWriter(fps=fps * args.speed, bitrate=2000)
    with writer.saving(fig, str(output_path), dpi=100):
        for rec, event, frame, fi, event_frames in tqdm(frame_list, desc="Rendering", unit="frame"):
            ego_id, target_id, rid = int(event["ego_id"]), int(event["target_id"]), rec.recording_id

            if rid != current_rid[0]:
                _refresh_lanes(rec)
                current_rid[0] = rid

            for a in frame_artists:
                a.remove()
            frame_artists.clear()

            ego_track = track_cache[(rid, ego_id)]
            target_track = track_cache[(rid, target_id)]

            if frame in ego_track.index:
                center_x = float(ego_track.loc[frame, "x"])
            elif frame in target_track.index:
                center_x = float(target_track.loc[frame, "x"])
            else:
                center_x = args.view_width / 2.0
            xlim = (center_x - half_width, center_x + half_width)
            ax.set_xlim(*xlim)

            frame_df = rec.get_frame(frame)
            if not frame_df.empty:
                visible = frame_df[
                    (frame_df["x"] >= xlim[0] - args.neighbor_margin)
                    & (frame_df["x"] <= xlim[1] + args.neighbor_margin)
                ]
                for idx, row in visible.iterrows():
                    vid = int(idx[0]) if isinstance(idx, tuple) else int(idx)
                    color, alpha, zorder = _vehicle_style(vid, ego_id, target_id)
                    x, y, w, h = _row_box(row)
                    frame_artists.append(ax.add_patch(
                        Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                                   lw=0.6, alpha=alpha, zorder=zorder)
                    ))
                    if vid in {ego_id, target_id}:
                        label = "ego" if vid == ego_id else "target"
                        frame_artists.append(ax.text(
                            x, y + h + 0.25, f"{label} {vid}", fontsize=8, color="black", zorder=5,
                            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
                        ))

            trail = event_frames[max(0, fi - args.tail_frames): fi + 1]
            ego_line.set_data(*_track_centers(ego_track, trail))
            target_line.set_data(*_track_centers(target_track, trail))
            title_obj.set_text(_frame_title(event, frame, fps))

            writer.grab_frame()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    output_dir_root = resolve_data_path(cfg["paths"]["output_dir"], args.config)
    events_path = Path(args.events_csv) if args.events_csv else output_dir_root / "events.csv"
    output_dir = Path(args.output_dir) if args.output_dir else output_dir_root / "figures" / "event_playbacks"

    if not events_path.exists():
        raise FileNotFoundError(f"events.csv not found: {events_path}")

    events = _load_events(events_path, args.event_type, args.max_events)
    LOGGER.info("Loaded %d valid %s events", len(events), args.event_type)

    recording_cache: dict = {}
    for rid in sorted(events["recording_id"].unique()):
        rid = int(rid)
        LOGGER.info("Loading recording %02d ...", rid)
        recording_cache[rid] = _load_recording(cfg, args.config, rid)

    frame_list = _build_frame_list(events, recording_cache, args)
    LOGGER.info("Total frames to render: %d", len(frame_list))
    if not frame_list:
        LOGGER.error("No frames to render.")
        return

    name = args.output_name or f"events_{args.event_type}"
    output_path = output_dir / f"{name}.mp4"
    ensure_dir(output_dir)

    LOGGER.info("Saving %s ...", output_path)
    _render_to_mp4(frame_list, args, output_path)
    LOGGER.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
