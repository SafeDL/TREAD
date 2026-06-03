"""Shared highD cut-in reconstruction and event-level risk scoring."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.risk import longitudinal_proxy_from_series, longitudinal_series_from_states

from .highd_longitudinal import (
    DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    PASSENGER_CAR_CLASS,
    _event_context,
)


logger = logging.getLogger(__name__)


HIGHD_CUTIN_SCORE_KEYS = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "cross_frame",
    "cutin_start_frame",
    "cutin_end_frame",
    "source_lane",
    "target_lane",
    "ego_length",
    "adv_length",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "completion_gap",
    "post_cutin_min_gap",
    "post_cutin_min_ttc",
    "cutin_duration_seconds",
    "cross_lateral_offset",
    "min_abs_lateral_offset",
    "max_abs_lateral_velocity",
    "collision",
    "near_collision",
    "y_cutin",
)


DEFAULT_HIGHD_CUTIN_CONFIG: dict[str, Any] = {
    **DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    "lateral_intrusion_weight": 1.5,
    "lateral_offset_eps": 0.25,
    "lateral_offset_scale": 1.0,
    "lateral_velocity_scale": 1.0,
    "cutin_duration_scale": 2.0,
    "cutin_duration_min_seconds": 0.1,
}


def highd_cutin_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(DEFAULT_HIGHD_CUTIN_CONFIG)
    if overrides:
        options.update(overrides)
    return options


def highd_cutin_options_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = highd_cutin_options(overrides)
    sampling = config.get("sampling", {})
    filters = config.get("filters", {})
    cutin = config.get("cutin_risk", {})
    for source, mapping in (
        (sampling, {"target_fps": "target_fps"}),
        (
            filters,
            {
                "max_abs_accel": "max_abs_accel",
                "max_abs_jerk": "max_abs_jerk",
                "max_position_jump": "max_position_jump",
                "min_vehicle_speed": "min_vehicle_speed",
                "require_passenger_car_ego": "require_passenger_car_ego",
                "require_passenger_car_lead": "require_passenger_car_lead",
            },
        ),
        (
            cutin,
            {
                "lateral_intrusion_weight": "lateral_intrusion_weight",
                "lateral_offset_eps": "lateral_offset_eps",
                "lateral_offset_scale": "lateral_offset_scale",
                "lateral_velocity_scale": "lateral_velocity_scale",
                "cutin_duration_scale": "cutin_duration_scale",
                "cutin_duration_min_seconds": "cutin_duration_min_seconds",
            },
        ),
    ):
        for source_key, target_key in mapping.items():
            if source_key in source:
                options[target_key] = source[source_key]
    return options


def _completion_frame(row: pd.Series) -> int:
    if "cutin_end_frame" in row and pd.notna(row["cutin_end_frame"]):
        return int(row["cutin_end_frame"])
    if "cross_frame" in row and pd.notna(row["cross_frame"]):
        return int(row["cross_frame"])
    return int(row["anchor_frame"])


def _cutin_raw_motion_metrics(
    recording: Any,
    row: pd.Series,
    *,
    fps: float,
    ego_length: float,
    adv_length: float,
) -> dict[str, float]:
    ego_id = int(row["ego_id"])
    target_id = int(row["target_id"])
    start = int(row.get("cutin_start_frame", row["anchor_frame"]))
    end = int(row.get("cutin_end_frame", row["anchor_frame"]))
    cross = int(row.get("cross_frame", row["anchor_frame"]))
    frames = np.arange(min(start, end), max(start, end) + 1, dtype=np.int64)
    try:
        ego = recording.get_vehicle_track(ego_id)
        target = recording.get_vehicle_track(target_id)
        common = [
            int(frame)
            for frame in frames
            if frame in ego.index and frame in target.index
        ]
    except KeyError:
        common = []
    if common:
        ego_sub = ego.loc[common]
        target_sub = target.loc[common]
        lateral = target_sub["y"].astype(float).to_numpy() - ego_sub["y"].astype(float).to_numpy()
        if "yVelocity" in target_sub.columns:
            rel_vy = target_sub["yVelocity"].astype(float).to_numpy()
            if "yVelocity" in ego_sub.columns:
                rel_vy = rel_vy - ego_sub["yVelocity"].astype(float).to_numpy()
        else:
            rel_vy = np.zeros(len(common), dtype=np.float64)
        min_abs_lateral = float(np.min(np.abs(lateral)))
        max_abs_vy = float(np.max(np.abs(rel_vy)))
    else:
        min_abs_lateral = float("nan")
        max_abs_vy = float("nan")

    try:
        ego_cross = recording.get_vehicle_track(ego_id).loc[cross]
        target_cross = recording.get_vehicle_track(target_id).loc[cross]
        cross_lateral = float(target_cross["y"] - ego_cross["y"])
    except KeyError:
        cross_lateral = float("nan")

    completion = _completion_frame(row)
    try:
        ego_end = recording.get_vehicle_track(ego_id).loc[completion]
        target_end = recording.get_vehicle_track(target_id).loc[completion]
        completion_gap = float(target_end["x"] - ego_end["x"]) - 0.5 * (
            float(ego_length) + float(adv_length)
        )
    except KeyError:
        completion_gap = float("nan")

    duration_steps = max(0, int(end) - int(start))
    duration_seconds = float(duration_steps / max(float(fps), 1.0e-6))
    return {
        "completion_gap": completion_gap,
        "cutin_duration_seconds": duration_seconds,
        "cross_lateral_offset": cross_lateral,
        "min_abs_lateral_offset": min_abs_lateral,
        "max_abs_lateral_velocity": max_abs_vy,
    }


def _cutin_metrics(
    recording: Any,
    row: pd.Series,
    context: np.ndarray,
    future: np.ndarray,
    ego_length: float,
    adv_length: float,
    options: dict[str, Any],
) -> dict[str, Any]:
    series = longitudinal_series_from_states(future, ego_length, adv_length)
    gap = series["gap"]
    ttc = series["ttc"]
    thw = series["thw"]
    drac = series["drac"]
    initial_ego = context[-1, 0]
    initial_target = context[-1, 1]
    initial_gap = initial_target[0] - initial_ego[0] - 0.5 * (
        float(ego_length) + float(adv_length)
    )
    raw = _cutin_raw_motion_metrics(
        recording,
        row,
        fps=float(options["target_fps"]),
        ego_length=ego_length,
        adv_length=adv_length,
    )
    return {
        "initial_gap": float(initial_gap),
        "initial_closing_speed": float(initial_ego[2] - initial_target[2]),
        "recorded_min_gap": float(np.min(gap)),
        "recorded_min_ttc": float(np.min(np.clip(ttc, 0.0, 1000.0))),
        "post_cutin_min_gap": float(np.min(gap)),
        "post_cutin_min_ttc": float(np.min(np.clip(ttc, 0.0, 1000.0))),
        "_gap_series": gap.astype(np.float32),
        "_ttc_series": np.clip(ttc, 0.0, 1000.0).astype(np.float32),
        "_thw_series": np.clip(thw, 0.0, 1000.0).astype(np.float32),
        "_drac_series": np.clip(drac, 0.0, 1000.0).astype(np.float32),
        "_ego_speed_series": series["ego_speed"].astype(np.float32),
        "_lead_speed_series": series["lead_speed"].astype(np.float32),
        "_ego_accel_series": series["ego_accel"].astype(np.float32),
        **raw,
    }


def build_highd_cutin_event_rows_from_recording(
    recording: Any,
    events: pd.DataFrame,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    opts = highd_cutin_options(options)
    history_steps = int(opts["history_steps"])
    min_future_steps = int(opts["min_future_steps"])
    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in events.iterrows():
        ego_id = int(row["ego_id"])
        target_id = int(row["target_id"])
        ego_meta = recording.tracks_meta.loc[ego_id]
        target_meta = recording.tracks_meta.loc[target_id]
        ego_class = str(ego_meta.get("class", "")).strip().lower()
        target_class = str(target_meta.get("class", "")).strip().lower()
        if bool(opts.get("require_passenger_car_ego", True)) and ego_class != PASSENGER_CAR_CLASS:
            skipped += 1
            continue
        if bool(opts.get("require_passenger_car_lead", True)) and target_class != PASSENGER_CAR_CLASS:
            skipped += 1
            continue
        item = _event_context(recording, row, history_steps, min_future_steps)
        if item is None:
            skipped += 1
            continue
        metrics = _cutin_metrics(
            recording,
            row,
            item["context_states"],
            item["future_states"],
            float(item["ego_length"]),
            float(item["adv_length"]),
            opts,
        )
        rows.append(
            {
                "recording_id": int(row["recording_id"]),
                "event_id": str(row["event_id"]),
                "ego_id": ego_id,
                "target_id": target_id,
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "anchor_frame": int(row["anchor_frame"]),
                "cross_frame": int(row.get("cross_frame", row["anchor_frame"])),
                "cutin_start_frame": int(row.get("cutin_start_frame", row["anchor_frame"])),
                "cutin_end_frame": int(row.get("cutin_end_frame", row["anchor_frame"])),
                "source_lane": int(row.get("source_lane", -1)),
                "target_lane": int(row.get("target_lane", -1)),
                "context_states": item["context_states"],
                "ego_length": float(item["ego_length"]),
                "adv_length": float(item["adv_length"]),
                **metrics,
            }
        )
    return rows, skipped


def score_highd_cutin_event_rows(
    rows: list[dict[str, Any]],
    *,
    options: dict[str, Any] | None = None,
) -> None:
    opts = highd_cutin_options(options)
    cfg = {
        "closed_loop_risk": {
            "collision_bonus": float(opts["collision_bonus"]),
            "near_collision_weight": float(opts["near_collision_weight"]),
            "hard_brake_weight": float(opts["hard_brake_weight"]),
            "hard_brake_threshold": float(opts["hard_brake_threshold"]),
        },
        "longitudinal_risk_scoring": {
            "ttc_weight": float(opts["w_ttc"]),
            "thw_weight": float(opts["w_thw"]),
            "gap_weight": float(opts["w_gap"]),
            "drac_weight": float(opts["w_drac"]),
            "ttc_scale": float(opts["ttc_scale"]),
            "thw_scale": float(opts["thw_scale"]),
            "gap_scale": float(opts["gap_scale"]),
            "drac_scale": float(opts["drac_scale"]),
            "ttc_eps": float(opts["ttc_eps"]),
            "thw_eps": float(opts["thw_eps"]),
            "gap_eps": float(opts["gap_eps"]),
            "pool_beta": float(opts["pool_beta"]),
        },
    }
    near_gap_threshold = float(opts["near_collision_gap"])
    hard_brake_threshold = float(opts["hard_brake_threshold"])
    lateral_weight = float(opts["lateral_intrusion_weight"])
    lateral_offset_eps = max(float(opts["lateral_offset_eps"]), 1.0e-6)
    lateral_offset_scale = max(float(opts["lateral_offset_scale"]), 1.0e-6)
    lateral_velocity_scale = max(float(opts["lateral_velocity_scale"]), 1.0e-6)
    duration_scale = max(float(opts["cutin_duration_scale"]), 1.0e-6)
    min_duration = max(float(opts["cutin_duration_min_seconds"]), 1.0e-6)

    for row in rows:
        series = {
            "gap": row["_gap_series"],
            "ttc": row["_ttc_series"],
            "thw": row["_thw_series"],
            "drac": row["_drac_series"],
            "ego_speed": row["_ego_speed_series"],
            "lead_speed": row["_lead_speed_series"],
            "ego_accel": row["_ego_accel_series"],
        }
        proxy = longitudinal_proxy_from_series(
            series,
            cfg,
            scoring_section="longitudinal_risk_scoring",
        )
        min_gap = float(np.min(row["_gap_series"]))
        min_ego_accel = float(np.min(row["_ego_accel_series"]))
        collision = float(min_gap <= 0.0)
        near_collision = float(min_gap < near_gap_threshold)
        hard_brake = max(0.0, hard_brake_threshold - min_ego_accel) / max(
            abs(hard_brake_threshold),
            1.0e-6,
        )
        lateral_offset = float(row.get("min_abs_lateral_offset", np.nan))
        if not np.isfinite(lateral_offset):
            lateral_offset = abs(float(row.get("cross_lateral_offset", 0.0)))
        lateral_velocity = float(row.get("max_abs_lateral_velocity", 0.0))
        if not np.isfinite(lateral_velocity):
            lateral_velocity = 0.0
        duration = max(float(row.get("cutin_duration_seconds", 0.0)), min_duration)
        lateral_objective = (
            1.0 / max(lateral_offset / lateral_offset_scale, lateral_offset_eps)
            + lateral_velocity / lateral_velocity_scale
            + duration_scale / duration
        )
        lateral_score = lateral_weight * lateral_objective
        collision_score = float(opts["collision_bonus"]) * collision
        near_score = float(opts["near_collision_weight"]) * near_collision
        hard_score = float(opts["hard_brake_weight"]) * hard_brake
        y_cutin = (
            collision_score
            + near_score
            + float(proxy["proxy_risk_score"])
            + hard_score
            + lateral_score
        )
        row["collision"] = collision
        row["near_collision"] = near_collision
        row["y_cutin"] = float(y_cutin)


def highd_cutin_score_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in HIGHD_CUTIN_SCORE_KEYS if key in row} for row in rows]


def save_highd_cutin_event_context_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "context_states": np.asarray([row["context_states"] for row in rows], dtype=np.float32),
    }
    for key in HIGHD_CUTIN_SCORE_KEYS:
        if key in rows[0]:
            payload[key] = np.asarray([row[key] for row in rows])
    np.savez_compressed(path, **payload)


def load_highd_cutin_event_context_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"highD cut-in event context cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "context_states" not in data.files:
        raise KeyError(f"{path} is missing context_states")
    arrays = {
        key: data[key]
        for key in ("context_states", *HIGHD_CUTIN_SCORE_KEYS)
        if key in data.files
    }
    count = int(arrays["context_states"].shape[0])
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        row: dict[str, Any] = {
            "context_states": arrays["context_states"][idx].astype(np.float32),
        }
        for key in HIGHD_CUTIN_SCORE_KEYS:
            if key not in arrays:
                continue
            value = arrays[key][idx]
            if isinstance(value, np.generic):
                value = value.item()
            row[key] = value
        rows.append(row)
    return rows
