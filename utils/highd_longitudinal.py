"""Shared highD following-event reconstruction and longitudinal risk scoring."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffusion.src.data import (
    _build_world_states,
    _vehicle_length_from_meta,
)
from diffusion.src.features import extract_scenario_condition
from diffusion.src.scenario_frame import compute_ego_frame, world_to_ego_states

from .risk import apply_closed_loop_risk, longitudinal_series_from_states


logger = logging.getLogger(__name__)

PASSENGER_CAR_CLASS = "car"


HIGHD_EVENT_SCORE_KEYS = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "ego_length",
    "adv_length",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "collision",
    "near_collision",
    "y_long",
)


DEFAULT_HIGHD_LONGITUDINAL_CONFIG: dict[str, Any] = {
    "history_steps": 10,
    "min_future_steps": 125,
    "w_ttc": 2.0,
    "w_thw": 1.0,
    "w_gap": 1.0,
    "w_drac": 2.0,
    "ttc_scale": 1.0,
    "thw_scale": 1.0,
    "gap_scale": 1.0,
    "drac_scale": 5.0,
    "ttc_eps": 0.2,
    "thw_eps": 0.2,
    "gap_eps": 0.5,
    "pool_beta": 8.0,
    "collision_bonus": 5.0,
    "near_collision_weight": 1.0,
    "hard_brake_weight": 1.0,
    "hard_brake_threshold": -4.0,
    "near_collision_gap": 2.0,
    "target_fps": 25,
    "max_abs_accel": 8.0,
    "max_abs_jerk": 30.0,
    "max_position_jump": 5.0,
    "min_vehicle_speed": 0.0,
    "require_passenger_car_ego": True,
    "require_passenger_car_lead": True,
}


def highd_longitudinal_options(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(DEFAULT_HIGHD_LONGITUDINAL_CONFIG)
    if overrides:
        options.update(overrides)
    return options


def highd_options_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = highd_longitudinal_options(overrides)
    sampling = config.get("sampling", {})
    filters = config.get("filters", {})
    following = config.get("following", {})
    for source, mapping in (
        (
            sampling,
            {
                "target_fps": "target_fps",
            },
        ),
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
            following,
            {
                "context_history_steps": "history_steps",
                "min_future_steps": "min_future_steps",
            },
        ),
    ):
        for source_key, target_key in mapping.items():
            if source_key in source:
                options[target_key] = source[source_key]
    return options


def _safe_frame_value(row: pd.Series | dict[str, Any], key: str, default: int) -> int:
    value = row.get(key, default)
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def _optional_frame_value(row: pd.Series | dict[str, Any], key: str) -> int | None:
    value = row.get(key, None)
    if value is None or pd.isna(value):
        return None
    return int(value)


def _event_context(
    recording: Any,
    row: pd.Series,
    history_steps: int,
    min_future_steps: int,
    *,
    event_type: str = "following",
) -> dict[str, Any] | None:
    anchor = int(row["anchor_frame"])
    available_future_steps = max(0, int(row["end_frame"]) - anchor)
    if available_future_steps < int(min_future_steps):
        return None
    future_steps = int(min_future_steps)
    if future_steps <= 0:
        return None
    history_frames = np.arange(
        anchor - int(history_steps) + 1,
        anchor + 1,
        dtype=np.int64,
    )
    future_frames = np.arange(
        anchor + 1,
        anchor + future_steps + 1,
        dtype=np.int64,
    )
    frames = np.concatenate([history_frames, future_frames])
    states = _build_world_states(recording, row, frames)
    if states is None:
        return None
    ego_len = _vehicle_length_from_meta(
        recording.tracks_meta,
        int(row["ego_id"]),
    )
    adv_len = _vehicle_length_from_meta(
        recording.tracks_meta,
        int(row["target_id"]),
    )
    history_world = states[:history_steps]
    future_world = states[history_steps:]
    ego_frame = compute_ego_frame(history_world[-1, 0])
    local_history = world_to_ego_states(history_world, ego_frame).astype(np.float32)
    local_future = world_to_ego_states(future_world, ego_frame).astype(np.float32)
    initial_states = local_history[-1]
    metadata = {
        "anchor_frame": anchor,
        "cross_frame": _safe_frame_value(row, "cross_frame", anchor),
        "cutin_start_frame": _safe_frame_value(row, "cutin_start_frame", anchor),
        "cutin_end_frame": _safe_frame_value(row, "cutin_end_frame", anchor),
    }
    scenario_conditions, _keys = extract_scenario_condition(
        initial_states,
        local_future,
        ego_len,
        adv_len,
        event_type=event_type,
        dt=1.0 / max(float(recording.recording_meta.get("frameRate", 25)), 1.0),
        metadata=metadata,
    )
    return {
        "scenario_conditions": scenario_conditions.astype(np.float32),
        "initial_states": initial_states.astype(np.float32),
        "future_states": local_future,
        "ego_length": float(ego_len),
        "adv_length": float(adv_len),
        "available_future_steps": int(available_future_steps),
    }


def _interaction_metrics(
    initial_states: np.ndarray,
    future: np.ndarray,
    ego_length: float,
    adv_length: float,
) -> dict[str, Any]:
    series = longitudinal_series_from_states(future, ego_length, adv_length)
    gap = series["gap"]
    ttc = series["ttc"]
    initial_ego = initial_states[0]
    initial_lead = initial_states[1]
    initial_gap = initial_lead[0] - initial_ego[0]
    initial_gap -= 0.5 * (ego_length + adv_length)
    return {
        "initial_gap": float(initial_gap),
        "initial_closing_speed": float(initial_ego[2] - initial_lead[2]),
        "recorded_min_gap": float(np.min(gap)),
        "recorded_min_ttc": float(np.min(np.clip(ttc, 0.0, 1000.0))),
        "_gap_series": gap.astype(np.float32),
        "_ego_speed_series": series["ego_speed"].astype(np.float32),
        "_lead_speed_series": series["lead_speed"].astype(np.float32),
        "_ego_accel_series": series["ego_accel"].astype(np.float32),
    }


def build_highd_event_rows_from_recording(
    recording: Any,
    events: pd.DataFrame,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    opts = highd_longitudinal_options(options)
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
        if (
            bool(opts.get("require_passenger_car_ego", True))
            and ego_class != PASSENGER_CAR_CLASS
        ):
            skipped += 1
            continue
        if (
            bool(opts.get("require_passenger_car_lead", True))
            and target_class != PASSENGER_CAR_CLASS
        ):
            skipped += 1
            continue
        item = _event_context(
            recording,
            row,
            history_steps,
            min_future_steps,
            event_type="following",
        )
        if item is None:
            skipped += 1
            continue
        metrics = _interaction_metrics(
            item["initial_states"],
            item["future_states"],
            float(item["ego_length"]),
            float(item["adv_length"]),
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
                "scenario_conditions": item["scenario_conditions"],
                "initial_states": item["initial_states"],
                "ego_length": float(item["ego_length"]),
                "adv_length": float(item["adv_length"]),
                **metrics,
            }
        )
    return rows, skipped


def build_highd_following_segment_rows_from_recording(
    recording: Any,
    events: pd.DataFrame,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Build full following segments for diffusion sliding-window reuse."""
    opts = highd_longitudinal_options(options)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in events.iterrows():
        ego_id = int(row["ego_id"])
        target_id = int(row["target_id"])
        ego_meta = recording.tracks_meta.loc[ego_id]
        target_meta = recording.tracks_meta.loc[target_id]
        ego_class = str(ego_meta.get("class", "")).strip().lower()
        target_class = str(target_meta.get("class", "")).strip().lower()
        if (
            bool(opts.get("require_passenger_car_ego", True))
            and ego_class != PASSENGER_CAR_CLASS
        ):
            skipped += 1
            continue
        if (
            bool(opts.get("require_passenger_car_lead", True))
            and target_class != PASSENGER_CAR_CLASS
        ):
            skipped += 1
            continue
        frames = np.arange(
            int(row["start_frame"]),
            int(row["end_frame"]) + 1,
            dtype=np.int64,
        )
        states = _build_world_states(recording, row, frames)
        if states is None:
            skipped += 1
            continue
        rows.append(
            {
                "recording_id": int(row["recording_id"]),
                "event_id": str(row["event_id"]),
                "ego_id": ego_id,
                "target_id": target_id,
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "anchor_frame": int(row["anchor_frame"]),
                "frames": frames,
                "world_states": states.astype(np.float32),
                "ego_length": float(
                    _vehicle_length_from_meta(recording.tracks_meta, ego_id)
                ),
                "adv_length": float(
                    _vehicle_length_from_meta(recording.tracks_meta, target_id)
                ),
            }
        )
    return rows, skipped


def highd_risk_config(
    *,
    options: dict[str, Any] | None = None,
    evt_model_path: Path | None = None,
) -> dict[str, Any]:
    opts = highd_longitudinal_options(options)
    cfg: dict[str, Any] = {
        "closed_loop_risk": {
            "collision_bonus": float(opts["collision_bonus"]),
            "near_collision_weight": float(opts["near_collision_weight"]),
            "hard_brake_weight": float(opts["hard_brake_weight"]),
            "hard_brake_threshold": float(opts["hard_brake_threshold"]),
            "lead_physics_weight": 0.0,
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
    if evt_model_path is not None:
        cfg["evt"] = {
            "score_space": "evt",
            "return_period": 100,
            "model_path": str(evt_model_path),
        }
    return cfg


def score_highd_event_rows(
    rows: list[dict[str, Any]],
    *,
    options: dict[str, Any] | None = None,
    evt_model_path: Path | None = None,
) -> None:
    opts = highd_longitudinal_options(options)
    cfg = highd_risk_config(options=opts, evt_model_path=evt_model_path)
    near_gap_threshold = float(opts["near_collision_gap"])
    for row in rows:
        trace = [
            {
                "gap": float(gap),
                "ego_speed": float(ego_speed),
                "lead_speed": float(lead_speed),
                "ego_accel": float(ego_accel),
            }
            for gap, ego_speed, lead_speed, ego_accel in zip(
                row["_gap_series"],
                row["_ego_speed_series"],
                row["_lead_speed_series"],
                row["_ego_accel_series"],
                strict=True,
            )
        ]
        min_gap = float(np.min(row["_gap_series"]))
        min_ego_accel = float(np.min(row["_ego_accel_series"]))
        metrics = {
            "collision": float(min_gap <= 0.0),
            "near_collision": float(min_gap < near_gap_threshold),
            "min_ego_accel": min_ego_accel,
            "lead_physics_penalty": 0.0,
        }
        apply_closed_loop_risk(
            metrics,
            trace,
            cfg,
            scoring_section="longitudinal_risk_scoring",
        )

        row["collision"] = float(metrics["collision"])
        row["near_collision"] = float(metrics["near_collision"])
        row["y_long"] = float(metrics["y_long"])


def highd_score_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in HIGHD_EVENT_SCORE_KEYS if key in row} for row in rows]


def save_highd_event_context_cache(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in rows],
            dtype=np.float32,
        ),
        "initial_states": np.asarray(
            [row["initial_states"] for row in rows],
            dtype=np.float32,
        ),
    }
    for key in HIGHD_EVENT_SCORE_KEYS:
        if key in rows[0]:
            payload[key] = np.asarray([row[key] for row in rows])
    np.savez_compressed(path, **payload)


def save_highd_following_segment_cache(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    target_fps: float,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(row["frames"]) for row in rows], dtype=np.int64)
    offsets = np.zeros(len(rows), dtype=np.int64)
    if len(rows) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    payload: dict[str, np.ndarray] = {
        "target_fps": np.asarray(float(target_fps), dtype=np.float32),
        "offset": offsets,
        "length": lengths,
        "frames": np.concatenate(
            [row["frames"] for row in rows],
            axis=0,
        ).astype(np.int64),
        "world_states": np.concatenate(
            [row["world_states"] for row in rows],
            axis=0,
        ).astype(np.float32),
    }
    for key in ("event_id", "ego_length", "adv_length"):
        payload[key] = np.asarray([row[key] for row in rows])
    np.savez_compressed(path, **payload)


def load_highd_event_context_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"highD event context cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "scenario_conditions" not in data.files or "initial_states" not in data.files:
        raise KeyError(f"{path} is missing scenario_conditions/initial_states")
    arrays = {
        key: data[key]
        for key in ("scenario_conditions", "initial_states", *HIGHD_EVENT_SCORE_KEYS)
        if key in data.files
    }
    scenario_conditions = arrays["scenario_conditions"]
    count = int(scenario_conditions.shape[0])
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        row: dict[str, Any] = {
            "scenario_conditions": scenario_conditions[idx].astype(np.float32),
            "initial_states": arrays["initial_states"][idx].astype(np.float32),
        }
        for key in HIGHD_EVENT_SCORE_KEYS:
            if key not in arrays:
                continue
            value = arrays[key][idx]
            if isinstance(value, np.generic):
                value = value.item()
            row[key] = value
        rows.append(row)
    return rows
