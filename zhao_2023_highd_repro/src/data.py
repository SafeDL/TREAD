from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_repo_imports, output_dir, resolve_path, save_json


ensure_repo_imports()

from process_highD.src.lane_utils import are_adjacent_lanes, detect_lane_changes, parse_lane_markings
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import filter_abnormal_tracks, normalize_driving_direction, resample_recording


@dataclass
class CutinEvent:
    event_id: str
    recording_id: int
    ego_id: int
    target_id: int
    source_lane: int
    target_lane: int
    start_frame: int
    cross_frame: int
    end_frame: int
    duration_s: float
    post_gap_m: float
    lane_width_m: float
    start_speed_mps: float
    ego_speed_mps: float


def _is_car(meta_row: pd.Series) -> bool:
    return str(meta_row.get("class", "")).strip().lower() == "car"


def _vehicle_length(meta: pd.DataFrame, vehicle_id: int) -> float:
    return float(meta.loc[int(vehicle_id), "width"])


def _progress_bounds(
    track: pd.DataFrame,
    lane_info: dict,
    source_lane: int,
    target_lane: int,
    cross_frame: int,
) -> tuple[int, int, np.ndarray] | None:
    lanes = lane_info.get("lanes", {})
    source = lanes.get(int(source_lane))
    target = lanes.get(int(target_lane))
    if source is None or target is None:
        return None
    denom = float(target["center"]) - float(source["center"])
    if abs(denom) < 1.0e-6:
        return None
    frames = track.index.to_numpy(dtype=np.int64)
    y = track["y"].to_numpy(dtype=float)
    progress = (y - float(source["center"])) / denom
    cross_pos = int(np.searchsorted(frames, int(cross_frame)))
    cross_pos = min(max(cross_pos, 0), len(frames) - 1)
    before = np.flatnonzero(progress[:cross_pos + 1] <= 0.10)
    after = np.flatnonzero(progress[cross_pos:] >= 0.90) + cross_pos
    if len(before) == 0 or len(after) == 0:
        return None
    start = int(frames[before[-1]])
    end = int(frames[after[0]])
    if end <= start:
        return None
    return start, end, progress.astype(np.float32)


def _gap_at(recording, meta: pd.DataFrame, ego_id: int, target_id: int, frame: int) -> float | None:
    try:
        ego = recording.get_vehicle_track(int(ego_id)).loc[int(frame)]
        target = recording.get_vehicle_track(int(target_id)).loc[int(frame)]
    except KeyError:
        return None
    ego_len = _vehicle_length(meta, ego_id)
    target_len = _vehicle_length(meta, target_id)
    return float(target["x"] - ego["x"] - 0.5 * (ego_len + target_len))


def _has_vehicle_between(recording, ego_id: int, target_id: int, frame: int, lane_id: int) -> bool:
    try:
        ego = recording.get_vehicle_track(int(ego_id)).loc[int(frame)]
        target = recording.get_vehicle_track(int(target_id)).loc[int(frame)]
    except KeyError:
        return True
    ego_x = float(ego["x"])
    target_x = float(target["x"])
    if target_x <= ego_x:
        return True
    frame_df = recording.get_frame(int(frame))
    for vid in frame_df.index.get_level_values("id").unique():
        if int(vid) in {int(ego_id), int(target_id)}:
            continue
        row = frame_df.loc[(vid, int(frame))]
        if int(row["laneId"]) != int(lane_id):
            continue
        if ego_x < float(row["x"]) < target_x:
            return True
    return False


def _match_ego(recording, meta: pd.DataFrame, target_id: int, target_lane: int, frame: int, cfg: dict) -> int | None:
    track = recording.get_vehicle_track(int(target_id))
    if frame in track.index:
        fid = int(track.loc[int(frame), "followingId"])
        if fid in meta.index and _is_car(meta.loc[fid]):
            gap = _gap_at(recording, meta, fid, target_id, frame)
            if gap is not None and gap > float(cfg["min_post_cutin_gap"]):
                if not _has_vehicle_between(recording, fid, target_id, frame, target_lane):
                    return int(fid)

    target_x = float(track.loc[int(frame), "x"])
    candidates: list[tuple[int, float]] = []
    frame_df = recording.get_frame(int(frame))
    for vid in frame_df.index.get_level_values("id").unique():
        vid = int(vid)
        if vid == int(target_id) or vid not in meta.index or not _is_car(meta.loc[vid]):
            continue
        row = frame_df.loc[(vid, int(frame))]
        if int(row["laneId"]) != int(target_lane):
            continue
        gap = target_x - float(row["x"])
        if gap > 0:
            candidates.append((vid, gap))
    candidates.sort(key=lambda item: item[1])
    for ego_id, _ in candidates:
        net_gap = _gap_at(recording, meta, ego_id, target_id, frame)
        if net_gap is None:
            continue
        if not (float(cfg["min_post_cutin_gap"]) < net_gap < float(cfg["max_post_cutin_gap"])):
            continue
        if not _has_vehicle_between(recording, ego_id, target_id, frame, target_lane):
            return int(ego_id)
    return None


def _sample_track(track: pd.DataFrame, start: int, end: int, points: int) -> np.ndarray:
    frames = track.index.to_numpy(dtype=float)
    wanted = np.linspace(float(start), float(end), int(points))
    cols = ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    out = np.zeros((int(points), 6), dtype=np.float32)
    for idx, col in enumerate(cols):
        values = track[col].to_numpy(dtype=float)
        out[:, idx] = np.interp(wanted, frames, values).astype(np.float32)
    return out


def _make_local_target(raw: np.ndarray) -> np.ndarray:
    local = raw.astype(np.float32).copy()
    local[:, 0] -= local[0, 0]
    local[:, 1] -= local[0, 1]
    return local


def _paper_sequence(local_target: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            local_target[:, 1],
            local_target[:, 2],
        ],
        axis=-1,
    ).astype(np.float32)


def _recording_ids(config: dict) -> list[int]:
    ids = config["data"].get("recording_ids", [])
    if ids:
        return [int(item) for item in ids]
    return list(range(1, 61))


def build_emergency_cutin_dataset(config: dict) -> dict[str, Path]:
    cfg = config["data"]
    root_out = output_dir(config)
    raw_dir = resolve_path(config["paths"]["highd_dir"])
    events: list[CutinEvent] = []
    trajectories: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    paper_sequences: list[np.ndarray] = []

    for rid in _recording_ids(config):
        recording = load_recording(str(raw_dir), int(rid))
        recording = normalize_driving_direction(recording)
        recording = filter_abnormal_tracks(recording, config)
        recording = resample_recording(recording, int(cfg["target_fps"]))
        fps = float(recording.recording_meta.get("frameRate", cfg["target_fps"]))
        meta = recording.tracks_meta
        lanes = parse_lane_markings(recording.recording_meta)
        counter = 0

        for vid in meta.index:
            if not _is_car(meta.loc[vid]) or int(meta.loc[vid].get("numLaneChanges", 0)) < 1:
                continue
            track = recording.get_vehicle_track(int(vid))
            if "_abnormal" in track.columns and bool(track["_abnormal"].any()):
                continue
            lane_changes = detect_lane_changes(track, int(vid), int(cfg["min_stable_steps"]))
            for lc in lane_changes:
                if not are_adjacent_lanes(int(lc["from_lane"]), int(lc["to_lane"]), lanes):
                    continue
                bounds = _progress_bounds(
                    track,
                    lanes,
                    int(lc["from_lane"]),
                    int(lc["to_lane"]),
                    int(lc["cross_frame"]),
                )
                if bounds is None:
                    continue
                start, end, _progress = bounds
                duration = (end - start) / max(fps, 1.0)
                if duration < float(cfg["min_lane_change_seconds"]) or duration > float(cfg["max_lane_change_seconds"]):
                    continue
                if start not in track.index or end not in track.index:
                    continue
                ego_id = _match_ego(recording, meta, int(vid), int(lc["to_lane"]), int(lc["cross_frame"]), cfg)
                if ego_id is None:
                    continue
                ego_track = recording.get_vehicle_track(int(ego_id))
                needed = [start, int(lc["cross_frame"]), end]
                if any(frame not in ego_track.index or frame not in track.index for frame in needed):
                    continue
                if "_abnormal" in ego_track.columns and bool(ego_track.loc[start:end, "_abnormal"].any()):
                    continue
                post_gap = _gap_at(recording, meta, ego_id, int(vid), int(lc["cross_frame"]))
                if post_gap is None:
                    continue
                lane_width = float(lanes["lanes"].get(int(lc["to_lane"]), {}).get("width", 3.75))
                target_raw = _sample_track(track, start, end, int(cfg["trajectory_points"]))
                target_local = _make_local_target(target_raw)
                if abs(float(target_local[-1, 1])) < 0.65 * lane_width:
                    continue
                ego_row = ego_track.loc[int(start)]
                target_row = track.loc[int(start)]
                counter += 1
                event = CutinEvent(
                    event_id=f"zhao_cutin_{rid:02d}_{counter:05d}",
                    recording_id=int(rid),
                    ego_id=int(ego_id),
                    target_id=int(vid),
                    source_lane=int(lc["from_lane"]),
                    target_lane=int(lc["to_lane"]),
                    start_frame=int(start),
                    cross_frame=int(lc["cross_frame"]),
                    end_frame=int(end),
                    duration_s=float(duration),
                    post_gap_m=float(post_gap),
                    lane_width_m=float(lane_width),
                    start_speed_mps=float(target_row["xVelocity"]),
                    ego_speed_mps=float(ego_row["xVelocity"]),
                )
                events.append(event)
                trajectories.append(target_local)
                paper_sequences.append(_paper_sequence(target_local))
                conditions.append(
                    np.asarray(
                        [
                            event.duration_s,
                            event.start_speed_mps,
                            event.ego_speed_mps - event.start_speed_mps,
                            event.post_gap_m,
                            event.lane_width_m,
                            target_local[-1, 1],
                        ],
                        dtype=np.float32,
                    )
                )

    if not trajectories:
        raise RuntimeError("No emergency cut-in events were extracted. Relax config/data filters.")

    events_df = pd.DataFrame([asdict(item) for item in events])
    events_path = root_out / "events.csv"
    dataset_path = root_out / "emergency_cutin_dataset.npz"
    summary_path = root_out / "data_summary.json"
    events_df.to_csv(events_path, index=False)
    np.savez_compressed(
        dataset_path,
        trajectories=np.stack(trajectories).astype(np.float32),
        conditions=np.stack(conditions).astype(np.float32),
        paper_sequences=np.stack(paper_sequences).astype(np.float32),
        event_id=events_df["event_id"].to_numpy(dtype=str),
    )
    save_json(
        summary_path,
        {
            "event_count": int(len(events)),
            "recording_ids": _recording_ids(config),
            "trajectory_points": int(cfg["trajectory_points"]),
            "duration_mean_s": float(events_df["duration_s"].mean()),
            "duration_max_s": float(events_df["duration_s"].max()),
            "post_gap_mean_m": float(events_df["post_gap_m"].mean()),
        },
    )
    return {"events": events_path, "dataset": dataset_path, "summary": summary_path}
