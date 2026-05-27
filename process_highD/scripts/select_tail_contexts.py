#!/usr/bin/env python3
"""Select shared long-tail highD contexts for adversarial and subset tasks."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import (
    SPLIT_TO_INDEX,
    _build_world_states,
    _split_by_recording,
    _vehicle_length_from_meta,
    prepare_recording,
)
from diffusion.src.scenario_frame import compute_ego_frame, world_to_ego_states
from diffusion.src.utils import setup_logging
from utils.io import write_csv, write_json
from utils.risk import apply_closed_loop_risk, longitudinal_series_from_states


SCRIPT_DEFAULTS: dict[str, Any] = {
    "raw_dir": ROOT / "highD_dataset" / "Matlab" / "data",
    "events_csv": ROOT / "results" / "highd_events" / "events.csv",
    "tail_context_path": (
        ROOT / "results" / "highd_tail_contexts" / "following" / "tail_contexts.npz"
    ),
    "tail_score_path": (
        ROOT / "results" / "highd_tail_contexts" / "following" / "tail_scores.npz"
    ),
    "evt_model_path": (
        ROOT / "results" / "highd_evt" / "following" / "longitudinal_evt_model.json"
    ),
    "split": "val",
    "num_contexts": 32,
    "tail_quantile": 0.90,
    "max_score_contexts": 0,
    "history_steps": 10,
    "min_future_steps": 5,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
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
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _runtime_config() -> dict[str, Any]:
    return {
        "sampling": {"target_fps": float(SCRIPT_DEFAULTS["target_fps"])},
        "filters": {
            "max_abs_accel": float(SCRIPT_DEFAULTS["max_abs_accel"]),
            "max_abs_jerk": float(SCRIPT_DEFAULTS["max_abs_jerk"]),
            "max_position_jump": float(SCRIPT_DEFAULTS["max_position_jump"]),
            "min_vehicle_speed": float(SCRIPT_DEFAULTS["min_vehicle_speed"]),
        },
        "splits": {
            "train_ratio": float(SCRIPT_DEFAULTS["train_ratio"]),
            "val_ratio": float(SCRIPT_DEFAULTS["val_ratio"]),
            "test_ratio": float(SCRIPT_DEFAULTS["test_ratio"]),
            "random_seed": int(SCRIPT_DEFAULTS["random_seed"]),
        },
    }


def _load_events(path: Path) -> pd.DataFrame:
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


def _filter_events(events: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    split = str(SCRIPT_DEFAULTS["split"])
    max_contexts = int(SCRIPT_DEFAULTS["max_score_contexts"])
    seed = int(SCRIPT_DEFAULTS["random_seed"])
    rid_split, split_meta = _split_by_recording(
        events["recording_id"].tolist(),
        cfg,
    )
    events = events.copy()
    events["split_index"] = events["recording_id"].map(lambda rid: rid_split[int(rid)])
    events["available_future_steps"] = events["end_frame"].astype(int) - events[
        "anchor_frame"
    ].astype(int)
    events = events[events["split_index"] == SPLIT_TO_INDEX[split]].reset_index(
        drop=True
    )
    if events.empty:
        raise RuntimeError(f"No valid following events found in {split} split")
    if max_contexts > 0 and len(events) > max_contexts:
        events = events.sample(
            n=max_contexts,
            random_state=seed,
        ).sort_index()
    logger.info(
        "Using %d valid %s following events from split metadata %s",
        len(events),
        split,
        {
            key: split_meta[key]
            for key in sorted(split_meta)
            if key.endswith("_recording_ids")
        },
    )
    return events.reset_index(drop=True)


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
        "context_states": world_to_ego_states(
            history_world,
            ego_frame,
        ).astype(np.float32),
        "future_states": world_to_ego_states(
            future_world,
            ego_frame,
        ).astype(np.float32),
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
    series = longitudinal_series_from_states(
        future,
        ego_length,
        adv_length,
    )
    gap = series["gap"]
    closing = series["closing_speed"]
    ttc = series["ttc"]
    thw = series["thw"]
    drac = series["drac"]
    initial_ego = context[-1, 0]
    initial_lead = context[-1, 1]
    initial_gap = initial_lead[0] - initial_ego[0] - 0.5 * (ego_length + adv_length)
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


def _build_rows(
    events: pd.DataFrame,
    cfg: dict[str, Any],
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    history_steps = int(SCRIPT_DEFAULTS["history_steps"])
    min_future_steps = int(SCRIPT_DEFAULTS["min_future_steps"])
    rows: list[dict[str, Any]] = []
    skipped = 0
    for rid, group in events.groupby("recording_id", sort=True):
        recording = prepare_recording(raw_dir, int(rid), cfg)
        for event_index, row in group.iterrows():
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
                    "split_index": int(row["split_index"]),
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
    if not rows:
        raise RuntimeError("No highD contexts could be built")
    if skipped:
        logger.warning("Skipped %d events with incomplete highD states", skipped)
    return rows, skipped


def _risk_config(evt_model_path: Path | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "closed_loop_risk": {
            "collision_bonus": float(SCRIPT_DEFAULTS["collision_bonus"]),
            "near_collision_weight": float(SCRIPT_DEFAULTS["near_collision_weight"]),
            "hard_brake_weight": float(SCRIPT_DEFAULTS["hard_brake_weight"]),
            "hard_brake_threshold": float(SCRIPT_DEFAULTS["hard_brake_threshold"]),
            "lead_physics_weight": 0.0,
        },
        "longitudinal_risk_scoring": {
            "ttc_weight": float(SCRIPT_DEFAULTS["w_ttc"]),
            "thw_weight": float(SCRIPT_DEFAULTS["w_thw"]),
            "gap_weight": float(SCRIPT_DEFAULTS["w_gap"]),
            "drac_weight": float(SCRIPT_DEFAULTS["w_drac"]),
            "ttc_scale": float(SCRIPT_DEFAULTS["ttc_scale"]),
            "thw_scale": float(SCRIPT_DEFAULTS["thw_scale"]),
            "gap_scale": float(SCRIPT_DEFAULTS["gap_scale"]),
            "drac_scale": float(SCRIPT_DEFAULTS["drac_scale"]),
            "ttc_eps": float(SCRIPT_DEFAULTS["ttc_eps"]),
            "thw_eps": float(SCRIPT_DEFAULTS["thw_eps"]),
            "gap_eps": float(SCRIPT_DEFAULTS["gap_eps"]),
            "pool_beta": float(SCRIPT_DEFAULTS["pool_beta"]),
        },
    }
    if evt_model_path is not None:
        cfg["evt"] = {
            "score_space": "evt",
            "return_period": 100,
            "model_path": str(evt_model_path),
        }
    return cfg


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    evt_model_path: Path | None = None,
) -> None:
    cfg = _risk_config(evt_model_path)
    near_gap_threshold = float(SCRIPT_DEFAULTS["near_collision_gap"])
    hard_brake_threshold = float(SCRIPT_DEFAULTS["hard_brake_threshold"])
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


def _save_outputs(rows: list[dict[str, Any]], skipped: int) -> None:
    tail_context_path = Path(SCRIPT_DEFAULTS["tail_context_path"])
    tail_score_path = Path(SCRIPT_DEFAULTS["tail_score_path"])
    tail_context_path.parent.mkdir(parents=True, exist_ok=True)
    tail_score_path.parent.mkdir(parents=True, exist_ok=True)

    tail_quantile = float(SCRIPT_DEFAULTS["tail_quantile"])
    num_contexts = int(SCRIPT_DEFAULTS["num_contexts"])
    score = np.asarray(
        [row["risk_score"] for row in rows],
        dtype=np.float32,
    )
    threshold = float(np.quantile(score[np.isfinite(score)], tail_quantile))
    tail_idx = np.where(score >= threshold)[0]
    tail_idx = tail_idx[np.argsort(score[tail_idx])[::-1]]
    if num_contexts > 0:
        tail_idx = tail_idx[:num_contexts]
    if tail_idx.size == 0:
        raise RuntimeError(f"No tail contexts found at quantile {tail_quantile}")
    selected = [rows[int(idx)] for idx in tail_idx]

    score_keys = (
        "split_index",
        "recording_id",
        "event_id",
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
    score_rows = [{key: row[key] for key in score_keys} for row in rows]
    write_csv(tail_score_path.with_suffix(".csv"), score_rows)
    np.savez_compressed(
        tail_score_path,
        **{key: np.asarray([row[key] for row in rows]) for key in score_keys},
    )

    payload: dict[str, np.ndarray] = {
        "context_states": np.asarray(
            [row["context_states"] for row in selected],
            dtype=np.float32,
        ),
        "source_type": np.asarray(
            ["highd_event_tail"] * len(selected),
            dtype=object,
        ),
        "tail_threshold": np.asarray(
            [threshold] * len(selected),
            dtype=np.float32,
        ),
    }
    for key in score_keys:
        payload[key] = np.asarray([row[key] for row in selected])
    np.savez_compressed(tail_context_path, **payload)

    write_json(
        tail_context_path.with_name("tail_context_summary.json"),
        {
            "events_csv": str(SCRIPT_DEFAULTS["events_csv"]),
            "tail_scores": str(tail_score_path),
            "tail_contexts": str(tail_context_path),
            "split": str(SCRIPT_DEFAULTS["split"]),
            "num_scored_events": int(len(rows)),
            "num_contexts": int(len(selected)),
            "skipped_events": int(skipped),
            "tail_quantile": tail_quantile,
            "tail_threshold": threshold,
            "score_min": float(np.min(score)),
            "score_mean": float(np.mean(score)),
            "score_p95": float(np.percentile(score, 95.0)),
            "score_max": float(np.max(score)),
            "scoring_method": (
                "shared RSS-free longitudinal risk over the anchor-to-end "
                "event suffix: softmax-pool(1/TTC, 1/THW, 1/gap, DRAC) "
                "plus collision, near-collision, and hard-brake terms"
            ),
            "min_future_steps": int(SCRIPT_DEFAULTS["min_future_steps"]),
            "score_components": [
                "ttc_risk_score",
                "thw_risk_score",
                "gap_risk_score",
                "drac_risk_score",
                "collision_risk_score",
                "near_collision_risk_score",
                "hard_brake_risk_score",
            ],
            "score_weights": {
                "w_ttc": float(SCRIPT_DEFAULTS["w_ttc"]),
                "w_thw": float(SCRIPT_DEFAULTS["w_thw"]),
                "w_gap": float(SCRIPT_DEFAULTS["w_gap"]),
                "w_drac": float(SCRIPT_DEFAULTS["w_drac"]),
                "collision_bonus": float(SCRIPT_DEFAULTS["collision_bonus"]),
                "near_collision_weight": float(
                    SCRIPT_DEFAULTS["near_collision_weight"]
                ),
                "hard_brake_weight": float(SCRIPT_DEFAULTS["hard_brake_weight"]),
            },
            "available_future_steps_min_selected": int(
                min(row["available_future_steps"] for row in selected)
            ),
            "available_future_steps_max_selected": int(
                max(row["available_future_steps"] for row in selected)
            ),
        },
    )
    logger.info(
        "Wrote %d shared tail contexts to %s",
        len(selected),
        tail_context_path,
    )


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    cfg = _runtime_config()
    events = _load_events(Path(SCRIPT_DEFAULTS["events_csv"]))
    filtered = _filter_events(events, cfg)
    rows, skipped = _build_rows(filtered, cfg, Path(SCRIPT_DEFAULTS["raw_dir"]))
    evt_model_path = Path(SCRIPT_DEFAULTS["evt_model_path"])
    if not evt_model_path.exists():
        logger.warning("EVT model not found at %s; tail risk_score uses raw y_long", evt_model_path)
        evt_model_path = None
    _score_rows(rows, evt_model_path=evt_model_path)
    _save_outputs(rows, skipped)


if __name__ == "__main__":
    main()
