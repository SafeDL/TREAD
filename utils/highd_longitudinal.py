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
    prepare_recording,
)
from diffusion.src.scenario_frame import compute_ego_frame, world_to_ego_states

from .risk import apply_closed_loop_risk, longitudinal_series_from_states


logger = logging.getLogger(__name__)


HIGHD_EVENT_SCORE_KEYS = (
    "event_index",
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "available_future_steps",
    "event_future_steps",
    "ego_length",
    "adv_length",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "recorded_min_thw",
    "recorded_max_drac",
    "min_ego_accel",
    "collision",
    "near_collision",
    "hard_brake",
    "y_long",
    "risk_score",
    "proxy_risk_score",
    "ttc_objective",
    "thw_objective",
    "gap_objective",
    "drac_objective",
    "ttc_risk_score",
    "thw_risk_score",
    "gap_risk_score",
    "drac_risk_score",
    "collision_risk_score",
    "near_collision_risk_score",
    "hard_brake_severity",
    "hard_brake_risk_score",
    "evt_tail_probability",
    "evt_return_level_target",
    "evt_failure_threshold",
)


DEFAULT_HIGHD_LONGITUDINAL_CONFIG: dict[str, Any] = {
    "history_steps": 10,
    "min_future_steps": 5,
    "random_seed": 42,
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
}


def highd_longitudinal_options(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(DEFAULT_HIGHD_LONGITUDINAL_CONFIG)
    if overrides:
        options.update(overrides)
    return options


def highd_runtime_config(
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opts = highd_longitudinal_options(options)
    return {
        "sampling": {"target_fps": float(opts["target_fps"])},
        "filters": {
            "max_abs_accel": float(opts["max_abs_accel"]),
            "max_abs_jerk": float(opts["max_abs_jerk"]),
            "max_position_jump": float(opts["max_position_jump"]),
            "min_vehicle_speed": float(opts["min_vehicle_speed"]),
        },
    }


def highd_options_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = highd_longitudinal_options(overrides)
    sampling = config.get("sampling", {})
    filters = config.get("filters", {})
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
            },
        ),
    ):
        for source_key, target_key in mapping.items():
            if source_key in source:
                options[target_key] = source[source_key]
    return options


def load_following_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"highD event CSV not found: {path}")
    events = pd.read_csv(path)
    required = {
        "event_id",
        "event_type",
        "recording_id",
        "ego_id",
        "target_id",
        "start_frame",
        "end_frame",
        "anchor_frame",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    events = events[events["event_type"] == "following"].copy()
    if "is_valid" in events.columns:
        valid = events["is_valid"]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        events = events[valid].copy()
    if events.empty:
        raise RuntimeError(f"No valid following events found in {path}")
    return events.reset_index(drop=True)


def all_following_events_for_evt(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["available_future_steps"] = (
        out["end_frame"].astype(int) - out["anchor_frame"].astype(int)
    )
    return out.reset_index(drop=True)


def _event_context(
    recording: Any,
    row: pd.Series,
    history_steps: int,
    min_future_steps: int,
) -> dict[str, Any] | None:
    anchor = int(row["anchor_frame"])
    available_future_steps = max(0, int(row["end_frame"]) - anchor)
    if available_future_steps < int(min_future_steps):
        return None
    future_steps = available_future_steps
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
    return {
        "context_states": world_to_ego_states(history_world, ego_frame).astype(
            np.float32
        ),
        "future_states": world_to_ego_states(future_world, ego_frame).astype(
            np.float32
        ),
        "ego_length": float(ego_len),
        "adv_length": float(adv_len),
        "available_future_steps": int(available_future_steps),
        "event_future_steps": int(future_steps),
    }


def _interaction_metrics(
    context: np.ndarray,
    future: np.ndarray,
    ego_length: float,
    adv_length: float,
) -> dict[str, Any]:
    series = longitudinal_series_from_states(future, ego_length, adv_length)
    gap = series["gap"]
    closing = series["closing_speed"]
    ttc = series["ttc"]
    thw = series["thw"]
    drac = series["drac"]
    initial_ego = context[-1, 0]
    initial_lead = context[-1, 1]
    initial_gap = initial_lead[0] - initial_ego[0]
    initial_gap -= 0.5 * (ego_length + adv_length)
    return {
        "initial_gap": float(initial_gap),
        "initial_closing_speed": float(initial_ego[2] - initial_lead[2]),
        "recorded_min_gap": float(np.min(gap)),
        "recorded_min_ttc": float(np.min(np.clip(ttc, 0.0, 1000.0))),
        "recorded_min_thw": float(np.min(np.clip(thw, 0.0, 1000.0))),
        "recorded_max_drac": float(np.max(np.clip(drac, 0.0, 1000.0))),
        "_gap_series": gap.astype(np.float32),
        "_ttc_series": np.clip(ttc, 0.0, 1000.0).astype(np.float32),
        "_thw_series": np.clip(thw, 0.0, 1000.0).astype(np.float32),
        "_drac_series": np.clip(drac, 0.0, 1000.0).astype(np.float32),
        "_closing_series": closing.astype(np.float32),
        "_ego_speed_series": series["ego_speed"].astype(np.float32),
        "_lead_speed_series": series["lead_speed"].astype(np.float32),
        "_ego_accel_series": series["ego_accel"].astype(np.float32),
    }


def build_highd_event_rows(
    events: pd.DataFrame,
    cfg: dict[str, Any],
    raw_dir: Path,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    opts = highd_longitudinal_options(options)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for rid, group in events.groupby("recording_id", sort=True):
        recording = prepare_recording(raw_dir, int(rid), cfg)
        group_rows, group_skipped = build_highd_event_rows_from_recording(
            recording,
            group,
            options=opts,
        )
        rows.extend(group_rows)
        skipped += group_skipped
    if not rows:
        raise RuntimeError("No highD contexts could be built")
    if skipped:
        logger.warning("Skipped %d events with incomplete highD states", skipped)
    return rows, skipped


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
    for event_index, row in events.iterrows():
        item = _event_context(
            recording,
            row,
            history_steps,
            min_future_steps,
        )
        if item is None:
            skipped += 1
            continue
        metrics = _interaction_metrics(
            item["context_states"],
            item["future_states"],
            float(item["ego_length"]),
            float(item["adv_length"]),
        )
        rows.append(
            {
                "event_index": int(event_index),
                "recording_id": int(row["recording_id"]),
                "event_id": str(row["event_id"]),
                "ego_id": int(row["ego_id"]),
                "target_id": int(row["target_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "anchor_frame": int(row["anchor_frame"]),
                "available_future_steps": int(item["available_future_steps"]),
                "event_future_steps": int(item["event_future_steps"]),
                "context_states": item["context_states"],
                "ego_length": float(item["ego_length"]),
                "adv_length": float(item["adv_length"]),
                **metrics,
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
    hard_brake_threshold = float(opts["hard_brake_threshold"])
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
        row["hard_brake"] = float(min_ego_accel <= hard_brake_threshold)
        row["min_ego_accel"] = float(min_ego_accel)
        row["y_long"] = float(metrics["y_long"])
        row["risk_score"] = float(metrics["risk_score"])
        row["proxy_risk_score"] = float(metrics["proxy_risk_score"])
        row["ttc_objective"] = float(metrics["ttc_objective"])
        row["thw_objective"] = float(metrics["thw_objective"])
        row["gap_objective"] = float(metrics["gap_objective"])
        row["drac_objective"] = float(metrics["drac_objective"])
        row["ttc_risk_score"] = float(metrics["ttc_risk_score"])
        row["thw_risk_score"] = float(metrics["thw_risk_score"])
        row["gap_risk_score"] = float(metrics["gap_risk_score"])
        row["drac_risk_score"] = float(metrics["drac_risk_score"])
        row["collision_risk_score"] = float(metrics["collision_risk_score"])
        row["near_collision_risk_score"] = float(metrics["near_collision_risk_score"])
        row["hard_brake_severity"] = float(metrics["hard_brake_severity"])
        row["hard_brake_risk_score"] = float(metrics["hard_brake_risk_score"])
        row["evt_tail_probability"] = float(metrics["evt_tail_probability"])
        row["evt_return_level_target"] = float(metrics["evt_return_level_target"])
        row["evt_failure_threshold"] = float(metrics["evt_failure_threshold"])


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
        "context_states": np.asarray(
            [row["context_states"] for row in rows],
            dtype=np.float32,
        ),
    }
    for key in HIGHD_EVENT_SCORE_KEYS:
        if key in rows[0]:
            payload[key] = np.asarray([row[key] for row in rows])
    np.savez_compressed(path, **payload)


def load_highd_event_context_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"highD event context cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "context_states" not in data.files:
        raise KeyError(f"{path} is missing context_states")
    arrays = {
        key: data[key]
        for key in ("context_states", *HIGHD_EVENT_SCORE_KEYS)
        if key in data.files
    }
    context_states = arrays["context_states"]
    count = int(context_states.shape[0])
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        row: dict[str, Any] = {
            "context_states": context_states[idx].astype(np.float32),
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


def load_highd_event_score_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"highD event score cache not found: {path}")
    frame = pd.read_csv(path)
    if "y_long" not in frame.columns:
        raise KeyError(f"{path} is missing y_long")
    return frame.to_dict(orient="records")
