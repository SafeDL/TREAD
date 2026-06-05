"""Build sliding-window action diffusion datasets from highD events."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from process_highD.src.loader import HighDRecording, load_recording
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)

from .features import extract_context, extract_relative_history
from .normalization import apply_normalizers, fit_dataset_normalizers
from .scenario_frame import compute_ego_frame, world_to_ego_states
from .types import (
    CUTIN_RELATIVE_HISTORY_KEYS,
    CUTIN_ACCEL_ACTION_KEYS,
    CUTIN_JERK_STEER_ACTION_KEYS,
    CUTIN_TRAJECTORY_KEYS,
    FOLLOWING_ACCEL_ACTION_KEYS,
    FOLLOWING_JERK_ACTION_KEYS,
    FOLLOWING_RELATIVE_HISTORY_KEYS,
    EventType,
    NUM_ACTORS,
    NUM_STATE_FEATURES,
    STATE_FEATURES,
)
from .utils import save_json

logger = logging.getLogger(__name__)


SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}
INDEX_TO_SPLIT = {v: k for k, v in SPLIT_TO_INDEX.items()}


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path
    events_csv: Path
    output_dir: Path


def _optional_path(
    value: str | None,
    *,
    base: Path,
) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _event_value(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


def action_keys_for(
    event_type: EventType | str,
    action_representation: str = "acceleration",
) -> Tuple[str, ...]:
    if _event_value(event_type) == EventType.FOLLOWING.value:
        if str(action_representation).lower() == "jerk":
            return FOLLOWING_JERK_ACTION_KEYS
        if str(action_representation).lower() == "acceleration":
            return FOLLOWING_ACCEL_ACTION_KEYS
        raise ValueError(
            "Unsupported following action_representation: "
            f"{action_representation}"
        )
    if _event_value(event_type) == EventType.CUT_IN.value:
        if str(action_representation).lower() in {"jerk_steer_rate", "jerk"}:
            return CUTIN_JERK_STEER_ACTION_KEYS
        if str(action_representation).lower() in {
            "maneuver_acceleration",
            "acceleration",
            "ax_ay",
        }:
            return CUTIN_ACCEL_ACTION_KEYS
        if str(action_representation).lower() in {
            "maneuver_trajectory",
            "trajectory",
        }:
            return CUTIN_TRAJECTORY_KEYS
        raise ValueError(
            "Unsupported cut-in action_representation: "
            f"{action_representation}"
        )
    raise ValueError(f"Unsupported event_type: {event_type}")


def prepare_recording(raw_dir: str | Path, recording_id: int, config: dict) -> HighDRecording:
    rec = load_recording(str(raw_dir), int(recording_id))
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(config["sampling"]["target_fps"])
    rec = resample_recording(rec, target_fps)
    return rec


def _extract_vehicle_states(
    recording: HighDRecording,
    vehicle_id: int,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        track = recording.get_vehicle_track(int(vehicle_id))
    except KeyError:
        return None
    present = track.index.intersection(frames)
    if len(present) != len(frames):
        return None
    sub = track.loc[frames]
    if "_abnormal" in sub.columns and bool(sub["_abnormal"].any()):
        return None
    out = np.zeros((len(frames), NUM_STATE_FEATURES), dtype=np.float32)
    out[:, 0] = sub["x"].values
    out[:, 1] = sub["y"].values
    out[:, 2] = sub["xVelocity"].values
    out[:, 3] = sub["yVelocity"].values if "yVelocity" in sub.columns else 0.0
    out[:, 4] = sub["xAcceleration"].values
    out[:, 5] = sub["yAcceleration"].values if "yAcceleration" in sub.columns else 0.0
    return out


def _build_world_states(
    recording: HighDRecording,
    event_row: pd.Series,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
    ego = _extract_vehicle_states(recording, int(event_row["ego_id"]), frames)
    adv = _extract_vehicle_states(recording, int(event_row["target_id"]), frames)
    if ego is None or adv is None:
        return None
    return np.stack([ego, adv], axis=1).astype(np.float32)


def _vehicle_length_from_meta(meta: pd.DataFrame, vehicle_id: int) -> float:
    """highD `width` is the longitudinal bounding-box size; `height` is lateral width."""
    return float(meta.loc[int(vehicle_id)]["width"])


def _savgol_smooth_1d(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(y) < 3:
        return y.astype(np.float32)
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w < 3:
        return y.astype(np.float32)
    p = min(max(int(polyorder), 0), w - 1)
    half = w // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, p + 1, increasing=True)
    coeff = np.linalg.pinv(design)[0]
    padded = np.pad(y, (half, half), mode="edge")
    out = np.convolve(padded, coeff[::-1], mode="valid")
    return out.astype(np.float32)


def _smooth_velocity(values: np.ndarray, action_cfg: dict) -> np.ndarray:
    smoothing = action_cfg["smoothing"]
    method = str(smoothing["method"]).lower()
    if method in {"none", "raw"}:
        return np.asarray(values, dtype=np.float32)
    if method != "savgol":
        raise ValueError(f"Unsupported action smoothing method: {method}")
    return _savgol_smooth_1d(
        np.asarray(values, dtype=np.float32),
        int(smoothing["window"]),
        int(smoothing["polyorder"]),
    )


def _following_actions(
    history_world_states: np.ndarray,
    future_world_states: np.ndarray,
    config: dict,
    dt: float,
) -> np.ndarray:
    action_cfg = config["action"]
    source = str(action_cfg["source"]).lower()
    representation = str(action_cfg["representation"]).lower()
    ax_min = float(action_cfg["ax_min"])
    ax_max = float(action_cfg["ax_max"])
    jerk_abs_max = float(action_cfg["jerk_abs_max"])
    if source == "raw_acceleration":
        ax = future_world_states[:, 1, 4].astype(np.float32)
    elif source == "smoothed_velocity_diff":
        lead_vx = np.concatenate(
            [history_world_states[:, 1, 2], future_world_states[:, 1, 2]]
        ).astype(np.float32)
        smooth_vx = _smooth_velocity(lead_vx, action_cfg)
        ax_all = np.diff(smooth_vx) / max(float(dt), 1e-6)
        start = len(history_world_states) - 1
        stop = start + len(future_world_states)
        ax = ax_all[start:stop]
    else:
        raise ValueError(f"Unsupported action.source: {source}")
    ax = np.clip(ax, ax_min, ax_max).astype(np.float32)
    if representation == "acceleration":
        return ax.reshape(-1, 1)
    if representation == "jerk":
        if source == "smoothed_velocity_diff" and len(history_world_states) >= 2:
            lead_vx = np.concatenate(
                [history_world_states[:, 1, 2], future_world_states[:, 1, 2]]
            ).astype(np.float32)
            smooth_vx = _smooth_velocity(lead_vx, action_cfg)
            ax_all = np.diff(smooth_vx) / max(float(dt), 1e-6)
            prev_ax = float(ax_all[max(len(history_world_states) - 2, 0)])
        else:
            prev_ax = float(history_world_states[-1, 1, 4])
        jx = np.diff(np.concatenate([[prev_ax], ax])) / max(float(dt), 1e-6)
        return np.clip(jx, -jerk_abs_max, jerk_abs_max).astype(np.float32).reshape(-1, 1)
    raise ValueError(f"Unsupported action.representation: {representation}")


def _cutin_actions(
    history_world_states: np.ndarray,
    future_world_states: np.ndarray,
    config: dict,
    dt: float,
) -> np.ndarray:
    action_cfg = config["action"]
    source = str(action_cfg["source"]).lower()
    representation = str(action_cfg["representation"]).lower()
    if representation not in {"jerk_steer_rate", "jerk"}:
        raise ValueError(f"Unsupported cut-in action.representation: {representation}")

    ax_min = float(action_cfg["ax_min"])
    ax_max = float(action_cfg["ax_max"])
    jerk_abs_max = float(action_cfg["jerk_abs_max"])
    steering_rate_abs_max = float(action_cfg.get("steering_rate_abs_max", 1.0))
    wheelbase = max(float(action_cfg.get("wheelbase", 5.0)), 1.0e-6)

    if source == "raw_acceleration":
        ax = future_world_states[:, 1, 4].astype(np.float32)
    elif source == "smoothed_velocity_diff":
        target_vx = np.concatenate(
            [history_world_states[:, 1, 2], future_world_states[:, 1, 2]]
        ).astype(np.float32)
        smooth_vx = _smooth_velocity(target_vx, action_cfg)
        ax_all = np.diff(smooth_vx) / max(float(dt), 1.0e-6)
        start = len(history_world_states) - 1
        stop = start + len(future_world_states)
        ax = ax_all[start:stop]
    else:
        raise ValueError(f"Unsupported action.source: {source}")
    ax = np.clip(ax, ax_min, ax_max).astype(np.float32)

    prev_ax = float(history_world_states[-1, 1, 4])
    jx = np.diff(np.concatenate([[prev_ax], ax])) / max(float(dt), 1.0e-6)
    jx = np.clip(jx, -jerk_abs_max, jerk_abs_max).astype(np.float32)

    all_target = np.concatenate(
        [history_world_states[-1:, 1], future_world_states[:, 1]],
        axis=0,
    ).astype(np.float64)
    heading = np.unwrap(
        np.arctan2(all_target[:, 3], np.maximum(all_target[:, 2], 1.0e-6))
    )
    yaw_rate = np.diff(heading) / max(float(dt), 1.0e-6)
    speed = np.hypot(all_target[:-1, 2], all_target[:-1, 3])
    steering = np.arctan2(wheelbase * yaw_rate, np.maximum(speed, 1.0e-6))
    prev_heading = float(
        np.arctan2(
            history_world_states[-1, 1, 3],
            max(float(history_world_states[-1, 1, 2]), 1.0e-6),
        )
    )
    if len(history_world_states) >= 2:
        prev_prev_heading = float(
            np.arctan2(
                history_world_states[-2, 1, 3],
                max(float(history_world_states[-2, 1, 2]), 1.0e-6),
            )
        )
    else:
        prev_prev_heading = prev_heading
    prev_yaw_rate = (prev_heading - prev_prev_heading) / max(float(dt), 1.0e-6)
    prev_speed = float(
        np.hypot(history_world_states[-1, 1, 2], history_world_states[-1, 1, 3])
    )
    prev_steering = float(
        np.arctan2(wheelbase * prev_yaw_rate, max(prev_speed, 1.0e-6))
    )
    steering_rate = np.diff(np.concatenate([[prev_steering], steering])) / max(
        float(dt),
        1.0e-6,
    )
    steering_rate = np.clip(
        steering_rate,
        -steering_rate_abs_max,
        steering_rate_abs_max,
    ).astype(np.float32)
    return np.stack([jx, steering_rate], axis=-1).astype(np.float32)


def _stride_for_split(dataset_cfg: dict, split_idx: int) -> int:
    split = INDEX_TO_SPLIT[int(split_idx)]
    key = f"{split}_stride"
    return int(dataset_cfg.get(key, dataset_cfg.get("stride", 5)))


def _select_event_samples(samples: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    selected = np.linspace(0, len(samples) - 1, int(limit), dtype=np.int64)
    return [samples[int(i)] for i in selected]


def _clamp_anchor_t(
    desired_t: int,
    start: int,
    end: int,
    history_steps: int,
    horizon_steps: int,
) -> int | None:
    lo = int(start) + int(history_steps) - 1
    hi = int(end) - int(horizon_steps)
    if hi < lo:
        return None
    return min(max(int(desired_t), lo), hi)


def _cutin_anchor_candidates(
    row: pd.Series,
    *,
    history_steps: int,
    horizon_steps: int,
    mode: str = "maneuver_start",
) -> list[tuple[str, int]]:
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    cross = int(row.get("cross_frame", row.get("anchor_frame", start)))
    cutin_start = int(row.get("cutin_start_frame", cross))
    cutin_end = int(row.get("cutin_end_frame", cross))
    mode = str(mode).lower()
    if mode == "maneuver_start":
        return [("maneuver_start", int(cutin_start))]
    if mode != "three_phase_anchors":
        raise ValueError(f"Unsupported cut-in dataset.phase_sampling: {mode}")
    desired = [
        ("pre_cross", cross - max(1, horizon_steps // 2)),
        ("crossing", cross - 1),
        ("post_cross", cutin_end - max(1, horizon_steps // 4)),
    ]
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for phase, target_t in desired:
        t = _clamp_anchor_t(
            target_t,
            start,
            end,
            history_steps,
            horizon_steps,
        )
        if t is None:
            continue
        key = (phase, int(t))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _exact_anchor_if_valid(
    phase_label: str,
    anchor_t: int,
    *,
    start: int,
    end: int,
    history_steps: int,
    horizon_steps: int,
) -> list[tuple[str, int]]:
    lo = int(start) + int(history_steps) - 1
    hi = int(end) - int(horizon_steps)
    t = int(anchor_t)
    if lo <= t <= hi:
        return [(str(phase_label), t)]
    return []


def _cutin_control_anchor_candidates(
    row: pd.Series,
    *,
    history_steps: int,
    horizon_steps: int,
    stride: int,
    mode: str,
) -> list[tuple[str, int]]:
    """Return ordered short-control anchors for cut-in action-prior training.

    ``cutin_sequence`` intentionally keeps diverse windows from the complete
    extracted cut-in event, in temporal order. ``cutin_maneuver`` is the stricter
    ablation that only samples anchors between cut-in start and cut-in end.
    """
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    cross = int(row.get("cross_frame", row.get("anchor_frame", start)))
    cutin_start = int(row.get("cutin_start_frame", cross))
    cutin_end = int(row.get("cutin_end_frame", cross))
    mode = str(mode).lower()
    if mode in {"cutin_start", "maneuver_start"}:
        return _exact_anchor_if_valid(
            "cutin_start",
            cutin_start,
            start=start,
            end=end,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
    if mode not in {"cutin_sequence", "cutin_maneuver", "sliding"}:
        raise ValueError(f"Unsupported cut-in dataset.phase_sampling: {mode}")

    if mode == "cutin_maneuver":
        lo = max(cutin_start, start + history_steps - 1)
        hi = min(max(cutin_start, cutin_end - 1), end - horizon_steps)
    else:
        lo = start + history_steps - 1
        hi = end - horizon_steps
    if hi < lo:
        return []

    out: list[tuple[str, int]] = []
    step = max(int(stride), 1)
    for t in range(int(lo), int(hi) + 1, step):
        if t < cutin_start:
            phase = "pre_maneuver"
        elif t < cross:
            phase = "pre_cross"
        elif t < cutin_end:
            phase = "crossing"
        else:
            phase = "post_cross"
        out.append((phase, int(t)))
    if out and out[-1][1] != int(hi):
        if hi < cutin_start:
            phase = "pre_maneuver"
        elif hi < cross:
            phase = "pre_cross"
        elif hi < cutin_end:
            phase = "crossing"
        else:
            phase = "post_cross"
        out.append((phase, int(hi)))
    return out


def _cutin_metadata_for_anchor(
    row: pd.Series,
    *,
    anchor_t: int,
    phase_label: str,
    fps: float,
) -> dict[str, float | str]:
    cross = int(row.get("cross_frame", row.get("anchor_frame", anchor_t)))
    cutin_start = int(row.get("cutin_start_frame", cross))
    cutin_end = int(row.get("cutin_end_frame", cross))
    denom = max(float(cutin_end - cutin_start), 1.0)
    source_lane = int(row.get("source_lane", 0))
    target_lane = int(row.get("target_lane", source_lane))
    return {
        "phase_label": str(phase_label),
        "time_to_cross_s": float((cross - int(anchor_t)) / max(float(fps), 1.0e-6)),
        "time_to_cutin_end_s": float((cutin_end - int(anchor_t)) / max(float(fps), 1.0e-6)),
        "cutin_progress": float((int(anchor_t) - cutin_start) / denom),
        "lane_change_direction": float(np.sign(target_lane - source_lane)),
    }


def _cutin_trajectory_targets(
    history_local: np.ndarray,
    future_local: np.ndarray,
) -> np.ndarray:
    target0 = history_local[-1, 1].astype(np.float32)
    target = future_local[:, 1].astype(np.float32)
    return np.stack(
        [
            target[:, 0] - target0[0],
            target[:, 1] - target0[1],
            target[:, 2],
            target[:, 3],
        ],
        axis=-1,
    ).astype(np.float32)


def _cutin_acceleration_targets(
    future_local: np.ndarray,
    config: dict,
) -> np.ndarray:
    action_cfg = config["action"]
    target = future_local[:, 1].astype(np.float32)
    ax = np.clip(
        target[:, 4],
        float(action_cfg.get("ax_min", -8.0)),
        float(action_cfg.get("ax_max", 4.0)),
    )
    ay = np.clip(
        target[:, 5],
        -float(action_cfg.get("ay_abs_max", 4.0)),
        float(action_cfg.get("ay_abs_max", 4.0)),
    )
    return np.stack([ax, ay], axis=-1).astype(np.float32)


def _rebase_cached_cutin_window(
    history_states: np.ndarray,
    future_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-anchor cached local states at the current history endpoint.

    Cut-in trajectory caches are stored in the maneuver-start ego frame. For
    sliding short-control windows, the diffusion context should still be in the
    current ego frame of each window, which is just a translation in highD's
    road-aligned coordinates.
    """
    combined = np.concatenate([history_states, future_states], axis=0).astype(
        np.float32
    )
    frame = compute_ego_frame(history_states[-1, 0])
    rebased = world_to_ego_states(combined, frame).astype(np.float32)
    return (
        rebased[: len(history_states)],
        rebased[len(history_states):],
    )


def _resolve_paths(config: dict, config_dir: str | Path | None) -> DatasetPaths:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths = config.get("paths", {})
    missing = [key for key in ("raw_dir", "events_csv", "output_dir") if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    output_dir = (base / paths["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return DatasetPaths(
        raw_dir=(base / paths["raw_dir"]).resolve(),
        events_csv=(base / paths["events_csv"]).resolve(),
        output_dir=output_dir,
    )


def _load_valid_events(paths: DatasetPaths, event_type: str, config: dict) -> pd.DataFrame:
    events = pd.read_csv(paths.events_csv)
    events = events[events["event_type"] == event_type].copy()
    if "is_valid" in events.columns:
        valid = events["is_valid"]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        events = events[valid].copy()
    events = events.reset_index(drop=True)
    if events.empty:
        raise RuntimeError(
            f"No valid events found for event_type={event_type} "
            f"in {paths.events_csv}"
        )

    max_recordings = int(config.get("dataset", {}).get("max_recordings", 0))
    if max_recordings > 0:
        keep_rids = sorted(events["recording_id"].unique().tolist())[:max_recordings]
        events = events[events["recording_id"].isin(keep_rids)].reset_index(drop=True)
        logger.warning("dataset.max_recordings=%d: using recordings=%s", max_recordings, keep_rids)
    return events


def _cutin_trajectory_cache_path(
    config: dict,
    *,
    config_dir: str | Path | None,
    events_csv: Path,
) -> Path:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths_cfg = config.get("paths", {})
    configured = _optional_path(
        paths_cfg.get("cutin_trajectory_cache")
        or paths_cfg.get("event_trajectory_cache"),
        base=base,
    )
    if configured is not None:
        return configured
    return events_csv.parent / "cutin_event_trajectories.npz"


def _following_segment_cache_path(
    config: dict,
    *,
    config_dir: str | Path | None,
    events_csv: Path,
) -> Path:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths_cfg = config.get("paths", {})
    configured = _optional_path(
        paths_cfg.get("following_segment_cache")
        or paths_cfg.get("event_segment_cache"),
        base=base,
    )
    if configured is not None:
        return configured
    return events_csv.parent / "following_event_segments.npz"


def _load_compatible_following_segment_cache(
    path: Path,
    *,
    target_fps: float,
) -> dict[str, Any] | None:
    if not path.exists():
        logger.info(
            "Following segment cache not found; rebuilding from raw highD: %s",
            path,
        )
        return None
    with np.load(path, allow_pickle=True) as archive:
        files = set(archive.files)
        required = {
            "event_id",
            "offset",
            "length",
            "frames",
            "world_states",
            "ego_length",
            "adv_length",
        }
        missing = sorted(required - files)
        if missing:
            logger.warning(
                "Following segment cache is missing %s; rebuilding from raw highD: %s",
                missing,
                path,
            )
            return None
        data = {key: archive[key] for key in required}
        if "target_fps" in files:
            data["target_fps"] = archive["target_fps"]
    states = data["world_states"]
    if states.ndim != 3 or states.shape[1:] != (NUM_ACTORS, NUM_STATE_FEATURES):
        logger.warning(
            "Following segment cache world_states shape mismatch %s; "
            "rebuilding from raw highD: %s",
            tuple(states.shape),
            path,
        )
        return None
    cached_fps = float(data["target_fps"].item()) if "target_fps" in data else target_fps
    if abs(cached_fps - float(target_fps)) > 1.0e-6:
        logger.warning(
            (
                "Following segment cache target_fps=%.6g does not match "
                "dataset target_fps=%.6g; rebuilding from raw highD"
            ),
            cached_fps,
            target_fps,
        )
        return None
    event_ids = [str(value) for value in data["event_id"]]
    index = {event_id: idx for idx, event_id in enumerate(event_ids)}
    logger.info(
        "Using following segment cache: %s (%d events)",
        path,
        len(index),
    )
    return {"path": path, "data": data, "index": index}


def _load_compatible_cutin_trajectory_cache(
    path: Path,
    *,
    history_steps: int,
    horizon_steps: int,
    target_fps: float,
) -> dict[str, Any] | None:
    if not path.exists():
        logger.info(
            "Cut-in trajectory cache not found; cache is required: %s",
            path,
        )
        return None
    with np.load(path, allow_pickle=True) as archive:
        files = set(archive.files)
        required = {
            "event_id",
            "context_states",
            "future_states",
            "anchor_frame",
            "cross_frame",
            "cutin_end_frame",
            "ego_length",
            "adv_length",
        }
        missing = sorted(required - files)
        if missing:
            logger.warning(
                "Cut-in trajectory cache is missing %s; cache is required: %s",
                missing,
                path,
            )
            return None
        data = {key: archive[key] for key in required}
        if "target_fps" in files:
            data["target_fps"] = archive["target_fps"]
    context_states = data["context_states"]
    future_states = data["future_states"]
    if (
        context_states.ndim != 4
        or future_states.ndim != 4
        or int(context_states.shape[1]) != int(history_steps)
        or int(future_states.shape[1]) < int(horizon_steps)
    ):
        logger.warning(
            (
                "Cut-in trajectory cache shape mismatch: context=%s "
                "future=%s requested_history=%d "
                "requested_horizon=%d"
            ),
            tuple(context_states.shape),
            tuple(future_states.shape),
            history_steps,
            horizon_steps,
        )
        return None
    cached_fps = float(data["target_fps"].item()) if "target_fps" in data else target_fps
    if abs(cached_fps - float(target_fps)) > 1.0e-6:
        logger.warning(
            (
                "Cut-in trajectory cache target_fps=%.6g does not match "
                "dataset target_fps=%.6g; cache is required"
            ),
            cached_fps,
            target_fps,
        )
        return None
    event_ids = [str(value) for value in data["event_id"]]
    index = {event_id: idx for idx, event_id in enumerate(event_ids)}
    logger.info(
        "Using cut-in trajectory cache: %s (%d events)",
        path,
        len(index),
    )
    return {"path": path, "data": data, "index": index}


def _split_by_recording(
    recording_ids: Iterable[int],
    cfg: dict,
) -> Tuple[Dict[int, int], Dict[str, object]]:
    split_cfg = cfg["splits"]
    seed = int(split_cfg["random_seed"])
    train_r = float(split_cfg["train_ratio"])
    val_r = float(split_cfg["val_ratio"])
    test_r = float(split_cfg["test_ratio"])
    total = max(train_r + val_r + test_r, 1e-6)
    train_r, val_r = train_r / total, val_r / total
    ids = sorted({int(r) for r in recording_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n >= 3:
        n_train = max(1, int(round(train_r * n)))
        n_val = max(1, int(round(val_r * n)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
    else:
        n_train = max(1, n - 1)
        n_val = max(0, n - n_train)
    mapping: Dict[int, int] = {}
    for rid in ids[:n_train]:
        mapping[int(rid)] = SPLIT_TO_INDEX["train"]
    for rid in ids[n_train:n_train + n_val]:
        mapping[int(rid)] = SPLIT_TO_INDEX["val"]
    for rid in ids[n_train + n_val:]:
        mapping[int(rid)] = SPLIT_TO_INDEX["test"]
    split_meta = {
        "strategy": "recording",
        "random_seed": seed,
        "train_recording_ids": [int(r) for r in ids[:n_train]],
        "val_recording_ids": [int(r) for r in ids[n_train:n_train + n_val]],
        "test_recording_ids": [int(r) for r in ids[n_train + n_val:]],
    }
    return mapping, split_meta


def build_action_dataset(config: dict, *, config_dir: str | Path | None = None) -> dict:
    """Build ``dataset.npz`` for one event type.

    For car-following, each sample contains a lead longitudinal action sequence.
    For default cut-in control priors, each sample is a short sliding window
    action plan, either ``[ax, ay]`` or ``[jx, steering_rate]``. Maneuver-level
    cut-in experiments can still build full-event ``[ax, ay]`` or
    ``[dx, dy, vx, vy]`` targets from the trajectory cache.
    """
    event_type = str(config.get("event", {}).get("event_type", "following"))
    if event_type not in {EventType.FOLLOWING.value, EventType.CUT_IN.value}:
        raise NotImplementedError(f"Unsupported event_type={event_type}")

    paths = _resolve_paths(config, config_dir)
    events = _load_valid_events(paths, event_type, config)

    sample_cfg = config["sampling"]
    fps = float(sample_cfg["target_fps"])
    dt = 1.0 / max(fps, 1.0)
    history_steps = int(config["context"]["history_steps"])
    horizon_steps = int(config["generation"]["horizon_steps"])
    dataset_cfg = config.get("dataset", {})
    max_windows_per_event = int(dataset_cfg.get("max_windows_per_event", 0))
    min_gap = float(dataset_cfg.get("min_current_gap", 0.5))
    action_representation = str(config["action"]["representation"]).lower()
    phase_sampling = str(dataset_cfg.get("phase_sampling", "sliding"))
    is_cutin_maneuver = (
        event_type == EventType.CUT_IN.value
        and action_representation
        in {"maneuver_acceleration", "maneuver_trajectory", "trajectory"}
    )
    use_cutin_trajectory_cache = bool(
        dataset_cfg.get("use_event_trajectory_cache", is_cutin_maneuver)
    )
    following_segment_cache = None
    following_segment_cache_path: Path | None = None
    cutin_trajectory_cache = None
    cutin_trajectory_cache_path: Path | None = None
    if event_type == EventType.FOLLOWING.value and bool(
        dataset_cfg.get("use_event_segment_cache", True)
    ):
        following_segment_cache_path = _following_segment_cache_path(
            config,
            config_dir=config_dir,
            events_csv=paths.events_csv,
        )
        following_segment_cache = _load_compatible_following_segment_cache(
            following_segment_cache_path,
            target_fps=fps,
        )
    elif (
        event_type == EventType.CUT_IN.value
        and use_cutin_trajectory_cache
    ):
        cutin_trajectory_cache_path = _cutin_trajectory_cache_path(
            config,
            config_dir=config_dir,
            events_csv=paths.events_csv,
        )
        cutin_trajectory_cache = _load_compatible_cutin_trajectory_cache(
            cutin_trajectory_cache_path,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            target_fps=fps,
        )
    elif event_type == EventType.CUT_IN.value and is_cutin_maneuver:
        raise RuntimeError(
            "Maneuver-level cut-in dataset requires the trajectory cache generated by "
            "extract_highd_events.py with dataset.phase_sampling='maneuver_start'. "
            "Use action.representation='ax_ay' or 'jerk_steer_rate' for short "
            "rolling-control sliding windows."
        )

    if (
        event_type == EventType.CUT_IN.value
        and use_cutin_trajectory_cache
        and cutin_trajectory_cache is None
    ):
        raise RuntimeError(
            "Cut-in dataset is configured to use the trajectory cache, but the "
            "cache is missing or incompatible. Rerun "
            "process_highD/scripts/extract_highd_events.py or set "
            "dataset.use_event_trajectory_cache=false to rebuild from raw highD."
        )
    if event_type == EventType.CUT_IN.value and is_cutin_maneuver:
        if cutin_trajectory_cache is None:
            raise RuntimeError(
                "Cut-in trajectory cache is missing or incompatible. Rerun "
                "process_highD/scripts/extract_highd_events.py before building "
                "the diffusion dataset."
            )
    if event_type == EventType.CUT_IN.value and cutin_trajectory_cache is not None:
        cached_event_ids = set(cutin_trajectory_cache["index"].keys())
        before = len(events)
        events = events[
            events["event_id"].astype(str).isin(cached_event_ids)
        ].reset_index(drop=True)
        missing = before - len(events)
        if missing > 0:
            logger.info(
                "Skipping %d cut-in events not present in trajectory cache; "
                "raw highD fallback is disabled while cache is active",
                missing,
            )
        if events.empty:
            raise RuntimeError(
                "Cut-in trajectory cache is active, but none of the valid "
                "events are present in the cache."
            )

    rid_split, split_meta = _split_by_recording(events["recording_id"].tolist(), config)
    grouped = events.groupby("recording_id")
    arrays: Dict[str, list] = {
        "context_states": [],
        "future_states": [],
        "context_features": [],
        "relative_history": [],
        "actions": [],
        "split_index": [],
        "recording_id": [],
        "event_id": [],
        "anchor_frame": [],
        "ego_length": [],
        "adv_length": [],
    }
    if event_type == EventType.CUT_IN.value:
        arrays.update(
            {
                "trajectory_targets": [],
                "future_cross_index": [],
                "future_cutin_end_index": [],
                "cross_mask": [],
                "cutin_end_mask": [],
                "phase_label": [],
            }
        )
    context_keys: List[str] | None = None
    skipped = 0
    for rid, rows in grouped:
        recording: HighDRecording | None = None
        meta: pd.DataFrame | None = None
        for _, row in rows.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            split_idx = rid_split[int(rid)]
            stride = _stride_for_split(dataset_cfg, split_idx)
            cached_event_idx: int | None = None
            cached_following_event_idx: int | None = None
            if following_segment_cache is not None:
                cached_following_event_idx = following_segment_cache["index"].get(
                    str(row["event_id"])
                )
            if cutin_trajectory_cache is not None:
                cached_event_idx = cutin_trajectory_cache["index"].get(
                    str(row["event_id"])
                )
            if event_type == EventType.CUT_IN.value and not is_cutin_maneuver:
                candidate_items = _cutin_control_anchor_candidates(
                    row,
                    history_steps=history_steps,
                    horizon_steps=horizon_steps,
                    stride=stride,
                    mode=phase_sampling,
                )
            elif event_type == EventType.CUT_IN.value and is_cutin_maneuver:
                candidate_items = _cutin_anchor_candidates(
                    row,
                    history_steps=history_steps,
                    horizon_steps=horizon_steps,
                    mode=phase_sampling,
                )
            else:
                candidate_items = [
                    ("sliding", int(t))
                    for t in range(
                        start + history_steps - 1,
                        end - horizon_steps + 1,
                        max(stride, 1),
                    )
                ]
            event_samples: list[dict] = []
            for phase_label, t in candidate_items:
                frames = np.arange(t - history_steps + 1, t + horizon_steps + 1, dtype=np.int64)
                history_local: np.ndarray | None = None
                future_local: np.ndarray | None = None
                history_world: np.ndarray | None = None
                future_world: np.ndarray | None = None
                ego_len: float
                adv_len: float
                if cached_following_event_idx is not None:
                    cache_data = following_segment_cache["data"]
                    offset = int(
                        cache_data["offset"][cached_following_event_idx]
                    )
                    length = int(
                        cache_data["length"][cached_following_event_idx]
                    )
                    segment_frames = cache_data["frames"][offset:offset + length]
                    pos0 = int(np.searchsorted(segment_frames, int(frames[0])))
                    pos1 = pos0 + int(len(frames))
                    if (
                        0 <= pos0
                        and pos1 <= length
                        and np.array_equal(segment_frames[pos0:pos1], frames)
                    ):
                        states = cache_data["world_states"][
                            offset + pos0:offset + pos1
                        ].astype(np.float32)
                        history_world = states[:history_steps]
                        future_world = states[history_steps:]
                        ego_frame = compute_ego_frame(history_world[-1, 0])
                        history_local = world_to_ego_states(
                            history_world,
                            ego_frame,
                        ).astype(np.float32)
                        future_local = world_to_ego_states(
                            future_world,
                            ego_frame,
                        ).astype(np.float32)
                        ego_len = float(
                            cache_data["ego_length"][cached_following_event_idx]
                        )
                        adv_len = float(
                            cache_data["adv_length"][cached_following_event_idx]
                        )
                        cross_frame = int(t)
                        end_frame = int(t)
                    else:
                        cached_following_event_idx = None

                if cached_event_idx is not None and history_local is None:
                    cache_data = cutin_trajectory_cache["data"]
                    cache_anchor = int(cache_data["anchor_frame"][cached_event_idx])
                    if is_cutin_maneuver and int(t) == cache_anchor:
                        cross_frame = int(
                            cache_data["cross_frame"][cached_event_idx]
                        )
                        end_frame = int(
                            cache_data["cutin_end_frame"][cached_event_idx]
                        )
                        history_local = cache_data["context_states"][
                            cached_event_idx
                        ].astype(np.float32)
                        future_local = cache_data["future_states"][
                            cached_event_idx,
                            :horizon_steps,
                        ].astype(np.float32)
                        ego_len = float(cache_data["ego_length"][cached_event_idx])
                        adv_len = float(cache_data["adv_length"][cached_event_idx])
                    elif not is_cutin_maneuver:
                        context_cached = cache_data["context_states"][
                            cached_event_idx
                        ].astype(np.float32)
                        future_cached = cache_data["future_states"][
                            cached_event_idx
                        ].astype(np.float32)
                        combined = np.concatenate(
                            [context_cached, future_cached],
                            axis=0,
                        )
                        anchor_pos = (
                            int(context_cached.shape[0])
                            - 1
                            + int(t)
                            - int(cache_anchor)
                        )
                        hist_start = anchor_pos - history_steps + 1
                        fut_stop = anchor_pos + horizon_steps + 1
                        if hist_start < 0 or fut_stop > int(combined.shape[0]):
                            cached_event_idx = None
                        else:
                            history_cached = combined[
                                hist_start:anchor_pos + 1
                            ].astype(np.float32)
                            future_cached_window = combined[
                                anchor_pos + 1:fut_stop
                            ].astype(np.float32)
                            history_local, future_local = _rebase_cached_cutin_window(
                                history_cached,
                                future_cached_window,
                            )
                            history_world = history_local
                            future_world = future_local
                            cross_frame = int(
                                cache_data["cross_frame"][cached_event_idx]
                            )
                            end_frame = int(
                                cache_data["cutin_end_frame"][cached_event_idx]
                            )
                            ego_len = float(
                                cache_data["ego_length"][cached_event_idx]
                            )
                            adv_len = float(
                                cache_data["adv_length"][cached_event_idx]
                            )
                    else:
                        cached_event_idx = None

                if history_local is None or future_local is None:
                    if (
                        event_type == EventType.CUT_IN.value
                        and (
                            is_cutin_maneuver
                            or cutin_trajectory_cache is not None
                        )
                    ):
                        skipped += 1
                        continue
                    if recording is None or meta is None:
                        recording = prepare_recording(paths.raw_dir, int(rid), config)
                        meta = recording.tracks_meta
                    ego_len = _vehicle_length_from_meta(meta, int(row["ego_id"]))
                    adv_len = _vehicle_length_from_meta(meta, int(row["target_id"]))
                    if event_type == EventType.CUT_IN.value:
                        cross_frame = int(row.get("cross_frame", row.get("anchor_frame", t)))
                        end_frame = int(row.get("cutin_end_frame", cross_frame))
                    else:
                        cross_frame = int(t)
                        end_frame = int(t)
                    states = _build_world_states(recording, row, frames)
                    if states is None:
                        skipped += 1
                        continue
                    history_world = states[:history_steps]
                    future_world = states[history_steps:]
                    ego_frame = compute_ego_frame(history_world[-1, 0])
                    history_local = world_to_ego_states(history_world, ego_frame).astype(np.float32)
                    future_local = world_to_ego_states(future_world, ego_frame).astype(np.float32)
                elif event_type == EventType.CUT_IN.value:
                    cross_frame = int(
                        row.get("cross_frame", row.get("anchor_frame", t))
                    )
                    end_frame = int(row.get("cutin_end_frame", cross_frame))
                else:
                    cross_frame = int(t)
                    end_frame = int(t)
                gap_now = (
                    history_local[-1, 1, 0]
                    - history_local[-1, 0, 0]
                    - 0.5 * (ego_len + adv_len)
                )
                if gap_now < min_gap:
                    skipped += 1
                    continue
                cutin_extra: dict[str, object] = {}
                context_metadata = None
                if event_type == EventType.CUT_IN.value:
                    trajectory_targets = _cutin_trajectory_targets(
                        history_local,
                        future_local,
                    )
                    if action_representation in {
                        "maneuver_acceleration",
                        "acceleration",
                        "ax_ay",
                    }:
                        actions = _cutin_acceleration_targets(
                            future_local,
                            config,
                        )
                    elif action_representation in {"jerk_steer_rate", "jerk"}:
                        if history_world is None or future_world is None:
                            raise RuntimeError(
                                "Cut-in short-control samples require world states"
                            )
                        actions = _cutin_actions(
                            history_world,
                            future_world,
                            config,
                            dt,
                        )
                    else:
                        actions = trajectory_targets
                    future_cross_index = int(cross_frame - (int(t) + 1))
                    future_end_index = int(end_frame - (int(t) + 1))
                    if is_cutin_maneuver:
                        context_metadata = _cutin_metadata_for_anchor(
                            row,
                            anchor_t=int(t),
                            phase_label=phase_label,
                            fps=fps,
                        )
                    cutin_extra = {
                        "trajectory_targets": trajectory_targets,
                        "future_cross_index": future_cross_index,
                        "future_cutin_end_index": future_end_index,
                        "cross_mask": float(0 <= future_cross_index < horizon_steps),
                        "cutin_end_mask": float(0 <= future_end_index < horizon_steps),
                        "phase_label": str(phase_label),
                    }
                else:
                    if history_world is None or future_world is None:
                        raise RuntimeError(
                            "Internal error: following sample missing world states"
                        )
                    actions = _following_actions(
                        history_world,
                        future_world,
                        config,
                        dt,
                    )
                relative_history = extract_relative_history(
                    history_local,
                    ego_len,
                    adv_len,
                    event_type=event_type,
                )
                if not np.all(np.isfinite(actions)):
                    skipped += 1
                    continue
                context_vec, keys = extract_context(
                    history_local,
                    ego_len,
                    adv_len,
                    dt,
                    event_type=event_type,
                    metadata=context_metadata,
                )
                if context_keys is None:
                    context_keys = keys
                event_samples.append(
                    {
                        "context_states": history_local,
                        "future_states": future_local,
                        "context_features": context_vec,
                        "relative_history": relative_history,
                        "actions": actions,
                        "split_index": split_idx,
                        "recording_id": int(rid),
                        "event_id": str(row["event_id"]),
                        "anchor_frame": int(t),
                        "ego_length": float(ego_len),
                        "adv_length": float(adv_len),
                        **cutin_extra,
                    }
                )
            for sample in _select_event_samples(event_samples, max_windows_per_event):
                for key in arrays:
                    arrays[key].append(sample[key])

    if not arrays["actions"]:
        raise RuntimeError(
            "No diffusion training samples were built. "
            "Check window sizes and raw data paths."
        )

    out_arrays = {
        "context_states": np.asarray(arrays["context_states"], dtype=np.float32),
        "future_states": np.asarray(arrays["future_states"], dtype=np.float32),
        "context_features": np.asarray(arrays["context_features"], dtype=np.float32),
        "relative_history": np.asarray(arrays["relative_history"], dtype=np.float32),
        "actions": np.asarray(arrays["actions"], dtype=np.float32),
        "split_index": np.asarray(arrays["split_index"], dtype=np.int8),
        "recording_id": np.asarray(arrays["recording_id"], dtype=np.int16),
        "event_id": np.asarray(arrays["event_id"], dtype=object),
        "anchor_frame": np.asarray(arrays["anchor_frame"], dtype=np.int64),
        "ego_length": np.asarray(arrays["ego_length"], dtype=np.float32),
        "adv_length": np.asarray(arrays["adv_length"], dtype=np.float32),
    }
    if event_type == EventType.CUT_IN.value:
        out_arrays.update(
            {
                "trajectory_targets": np.asarray(
                    arrays["trajectory_targets"],
                    dtype=np.float32,
                ),
                "future_cross_index": np.asarray(
                    arrays["future_cross_index"],
                    dtype=np.int64,
                ),
                "future_cutin_end_index": np.asarray(
                    arrays["future_cutin_end_index"],
                    dtype=np.int64,
                ),
                "cross_mask": np.asarray(arrays["cross_mask"], dtype=np.float32),
                "cutin_end_mask": np.asarray(
                    arrays["cutin_end_mask"],
                    dtype=np.float32,
                ),
                "phase_label": np.asarray(arrays["phase_label"], dtype=object),
            }
        )
    train_mask = out_arrays["split_index"] == SPLIT_TO_INDEX["train"]
    stats = fit_dataset_normalizers(
        out_arrays["context_states"],
        out_arrays["context_features"],
        out_arrays["actions"],
        train_mask,
        out_arrays["relative_history"],
    )
    norm_arrays = apply_normalizers(out_arrays, stats)
    train_keys = [
        "context_states",
        "context_features",
        "relative_history",
        "actions",
        "split_index",
    ]
    if event_type == EventType.CUT_IN.value:
        train_keys.extend(
            [
                "future_states",
                "trajectory_targets",
                "future_cross_index",
                "future_cutin_end_index",
                "cross_mask",
                "cutin_end_mask",
                "phase_label",
            ]
        )
    train_arrays = {
        key: norm_arrays[key]
        for key in train_keys
    }

    np.savez_compressed(paths.output_dir / "dataset.npz", **out_arrays)
    np.savez_compressed(paths.output_dir / "dataset_normalized.npz", **train_arrays)
    schema = {
        "event_type": event_type,
        "state_features": list(STATE_FEATURES),
        "future_state_features": list(STATE_FEATURES),
        "future_state_frame": "anchor_ego_local",
        "num_actors": NUM_ACTORS,
        "context_keys": context_keys or [],
        "relative_history_keys": list(
            CUTIN_RELATIVE_HISTORY_KEYS
            if event_type == EventType.CUT_IN.value
            else FOLLOWING_RELATIVE_HISTORY_KEYS
        ),
        "action_representation": action_representation,
        "action_keys": list(action_keys_for(event_type, action_representation)),
        "generation_target": (
            "maneuver_acceleration"
            if event_type == EventType.CUT_IN.value
            and action_representation == "maneuver_acceleration"
            else "maneuver_trajectory"
            if event_type == EventType.CUT_IN.value
            and action_representation in {"maneuver_trajectory", "trajectory"}
            else "action"
        ),
        "following_segment_cache_path": str(following_segment_cache_path or "")
        if event_type == EventType.FOLLOWING.value
        else "",
        "following_segment_cache_used": bool(following_segment_cache is not None)
        if event_type == EventType.FOLLOWING.value
        else False,
        "cutin_anchor_sampling": phase_sampling
        if event_type == EventType.CUT_IN.value
        else "",
        "cutin_anchor_scope": (
            "complete_cutin_event_ordered"
            if event_type == EventType.CUT_IN.value
            and not is_cutin_maneuver
            and phase_sampling.lower() in {"cutin_sequence", "sliding"}
            else "cutin_start_to_cutin_end"
            if event_type == EventType.CUT_IN.value
            and not is_cutin_maneuver
            and phase_sampling.lower() == "cutin_maneuver"
            else "cutin_start"
            if event_type == EventType.CUT_IN.value
            and not is_cutin_maneuver
            and phase_sampling.lower() in {"cutin_start", "maneuver_start"}
            else "maneuver_level"
            if event_type == EventType.CUT_IN.value
            else ""
        ),
        "cutin_trajectory_cache_path": str(cutin_trajectory_cache_path or "")
        if event_type == EventType.CUT_IN.value
        else "",
        "cutin_trajectory_cache_used": bool(cutin_trajectory_cache is not None)
        if event_type == EventType.CUT_IN.value
        else False,
        "trajectory_keys": list(CUTIN_TRAJECTORY_KEYS)
        if event_type == EventType.CUT_IN.value
        else [],
        "derived_action_keys": list(CUTIN_JERK_STEER_ACTION_KEYS)
        if event_type == EventType.CUT_IN.value
        and action_representation in {"jerk_steer_rate", "jerk"}
        else ["jx", "jy"]
        if event_type == EventType.CUT_IN.value
        else [],
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "dt": dt,
        "skipped_windows": skipped,
        "num_samples": int(out_arrays["actions"].shape[0]),
        "split_counts": {
            name: int(np.sum(out_arrays["split_index"] == idx))
            for name, idx in SPLIT_TO_INDEX.items()
        },
    }
    save_json(schema, paths.output_dir / "feature_schema.json")
    save_json(stats, paths.output_dir / "normalization_stats.json")
    save_json(split_meta, paths.output_dir / "train_val_test_split.json")
    logger.info(
        "Built %d samples at %s; skipped=%d",
        out_arrays["actions"].shape[0],
        paths.output_dir,
        skipped,
    )
    return {
        "arrays": out_arrays,
        "schema": schema,
        "stats": stats,
        "output_dir": paths.output_dir,
    }


def load_normalized_dataset(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "dataset_normalized.npz"
    if not path.exists():
        raise FileNotFoundError(f"Normalized diffusion dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
