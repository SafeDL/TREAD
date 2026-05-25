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
from utils.risk import percentile_rank


SCRIPT_DEFAULTS: dict[str, Any] = {
    "raw_dir": ROOT / "highD_dataset" / "Matlab" / "data",
    "events_csv": ROOT / "results" / "highd_events" / "events.csv",
    "tail_context_path": (
        ROOT / "results" / "highd_tail_contexts" / "following" / "tail_contexts.npz"
    ),
    "tail_score_path": (
        ROOT / "results" / "highd_tail_contexts" / "following" / "tail_scores.npz"
    ),
    "split": "val",
    "num_contexts": 32,
    "tail_quantile": 0.90,
    "max_score_contexts": 0,
    "history_steps": 10,
    "near_term_steps": 50,
    "min_future_steps": 5,
    "tail_top_fraction": 0.10,
    "tail_min_topk": 3,
    "w_event": 0.70,
    "w_near": 0.30,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    "w_ttc": 1.0,
    "w_thw": 1.0,
    "w_gap": 1.0,
    "w_drac": 1.0,
    "w_dv": 1.0,
    "eps": 1.0e-3,
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
    events["split_index"] = events["recording_id"].map(
        lambda rid: rid_split[int(rid)]
    )
    events["available_future_steps"] = (
        events["end_frame"].astype(int) - events["anchor_frame"].astype(int)
    )
    events = events[
        events["split_index"] == SPLIT_TO_INDEX[split]
    ].reset_index(drop=True)
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
    ego = future[:, 0]
    lead = future[:, 1]
    gap = lead[:, 0] - ego[:, 0] - 0.5 * (ego_length + adv_length)
    closing = ego[:, 2] - lead[:, 2]
    valid_gap = gap > 1.0e-6
    positive_closing = closing > 1.0e-6
    ttc = np.where(
        valid_gap & positive_closing,
        gap / np.maximum(closing, 1.0e-6),
        1000.0,
    )
    thw = np.where(
        valid_gap & (ego[:, 2] > 1.0e-6),
        gap / np.maximum(ego[:, 2], 1.0e-6),
        1000.0,
    )
    drac = np.where(
        valid_gap & positive_closing,
        np.square(closing) / np.maximum(2.0 * gap, 1.0e-6),
        0.0,
    )
    initial_ego = context[-1, 0]
    initial_lead = context[-1, 1]
    initial_gap = (
        initial_lead[0]
        - initial_ego[0]
        - 0.5 * (ego_length + adv_length)
    )
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
                    "available_future_steps": int(
                        item["available_future_steps"]
                    ),
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


def _top_fraction_mean(
    values: np.ndarray,
    *,
    fraction: float,
    min_topk: int,
) -> tuple[float, int]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0
    k = max(int(min_topk), int(np.ceil(float(fraction) * arr.size)))
    k = min(k, int(arr.size))
    top = np.partition(arr, int(arr.size) - k)[-k:]
    return float(np.mean(top)), int(k)


def _split_flat(values: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        out.append(values[start:end])
        start = end
    return out


def _score_rows(rows: list[dict[str, Any]]) -> None:
    eps = float(SCRIPT_DEFAULTS["eps"])
    top_fraction = float(SCRIPT_DEFAULTS["tail_top_fraction"])
    min_topk = int(SCRIPT_DEFAULTS["tail_min_topk"])
    near_term_steps = int(SCRIPT_DEFAULTS["near_term_steps"])
    weights = {
        "ttc_component": float(SCRIPT_DEFAULTS["w_ttc"]),
        "thw_component": float(SCRIPT_DEFAULTS["w_thw"]),
        "gap_component": float(SCRIPT_DEFAULTS["w_gap"]),
        "drac_component": float(SCRIPT_DEFAULTS["w_drac"]),
        "dv_component": float(SCRIPT_DEFAULTS["w_dv"]),
    }
    event_weight = float(SCRIPT_DEFAULTS["w_event"])
    near_weight = float(SCRIPT_DEFAULTS["w_near"])
    weight_sum = max(event_weight + near_weight, eps)
    event_weight /= weight_sum
    near_weight /= weight_sum

    lengths = [int(row["event_future_steps"]) for row in rows]
    raw_series = {
        "ttc_component": 1.0
        / np.maximum(
            np.concatenate([row["_ttc_series"] for row in rows]),
            eps,
        ),
        "thw_component": 1.0
        / np.maximum(
            np.concatenate([row["_thw_series"] for row in rows]),
            eps,
        ),
        "gap_component": 1.0
        / np.maximum(
            np.concatenate([row["_gap_series"] for row in rows]),
            eps,
        ),
        "drac_component": np.maximum(
            0.0,
            np.concatenate([row["_drac_series"] for row in rows]),
        ),
        "dv_component": np.maximum(
            0.0,
            np.concatenate([row["_closing_series"] for row in rows]),
        ),
    }
    ranked_series = {
        key: _split_flat(percentile_rank(value), lengths)
        for key, value in raw_series.items()
    }

    for idx, row in enumerate(rows):
        event_components: dict[str, float] = {}
        near_components: dict[str, float] = {}
        event_topk = 0
        near_topk = 0
        near_count = min(near_term_steps, int(row["event_future_steps"]))
        for key, values_by_row in ranked_series.items():
            values = values_by_row[idx]
            event_value, event_k = _top_fraction_mean(
                values,
                fraction=top_fraction,
                min_topk=min_topk,
            )
            near_value, near_k = _top_fraction_mean(
                values[:near_count],
                fraction=top_fraction,
                min_topk=min_topk,
            )
            event_components[key] = event_value
            near_components[key] = near_value
            event_topk = max(event_topk, event_k)
            near_topk = max(near_topk, near_k)

        event_score = sum(
            weights[key] * value for key, value in event_components.items()
        )
        near_score = sum(
            weights[key] * value for key, value in near_components.items()
        )
        final_score = event_weight * event_score + near_weight * near_score

        row["near_term_steps"] = int(near_count)
        row["event_tail_topk"] = int(event_topk)
        row["near_tail_topk"] = int(near_topk)
        row["event_tail_score"] = float(event_score)
        row["near_tail_score"] = float(near_score)
        row["criticality_score"] = float(final_score)
        for key, value in event_components.items():
            row[key] = float(value)
            row[f"event_{key}"] = float(value)
        for key, value in near_components.items():
            row[f"near_{key}"] = float(value)

        near_gap = row["_gap_series"][:near_count]
        near_ttc = row["_ttc_series"][:near_count]
        near_thw = row["_thw_series"][:near_count]
        near_drac = row["_drac_series"][:near_count]
        row["near_min_gap"] = float(np.min(near_gap))
        row["near_min_ttc"] = float(np.min(near_ttc))
        row["near_min_thw"] = float(np.min(near_thw))
        row["near_max_drac"] = float(np.max(near_drac))


def _save_outputs(rows: list[dict[str, Any]], skipped: int) -> None:
    tail_context_path = Path(SCRIPT_DEFAULTS["tail_context_path"])
    tail_score_path = Path(SCRIPT_DEFAULTS["tail_score_path"])
    tail_context_path.parent.mkdir(parents=True, exist_ok=True)
    tail_score_path.parent.mkdir(parents=True, exist_ok=True)

    tail_quantile = float(SCRIPT_DEFAULTS["tail_quantile"])
    num_contexts = int(SCRIPT_DEFAULTS["num_contexts"])
    score = np.asarray(
        [row["criticality_score"] for row in rows],
        dtype=np.float32,
    )
    threshold = float(np.quantile(score[np.isfinite(score)], tail_quantile))
    tail_idx = np.where(score >= threshold)[0]
    tail_idx = tail_idx[np.argsort(score[tail_idx])[::-1]]
    if num_contexts > 0:
        tail_idx = tail_idx[:num_contexts]
    if tail_idx.size == 0:
        raise RuntimeError(
            f"No tail contexts found at quantile {tail_quantile}"
        )
    selected = [rows[int(idx)] for idx in tail_idx]

    score_keys = (
        "split_index",
        "recording_id",
        "event_id",
        "anchor_frame",
        "available_future_steps",
        "event_future_steps",
        "near_term_steps",
        "event_tail_topk",
        "near_tail_topk",
        "ego_length",
        "adv_length",
        "initial_gap",
        "initial_closing_speed",
        "recorded_min_gap",
        "recorded_min_ttc",
        "recorded_min_thw",
        "recorded_max_drac",
        "near_min_gap",
        "near_min_ttc",
        "near_min_thw",
        "near_max_drac",
        "event_tail_score",
        "near_tail_score",
        "ttc_component",
        "thw_component",
        "gap_component",
        "drac_component",
        "dv_component",
        "event_ttc_component",
        "event_thw_component",
        "event_gap_component",
        "event_drac_component",
        "event_dv_component",
        "near_ttc_component",
        "near_thw_component",
        "near_gap_component",
        "near_drac_component",
        "near_dv_component",
        "criticality_score",
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
                "per-frame percentile risk with top-fraction aggregation "
                "over the anchor-to-end event suffix, blended with a "
                "near-term top-fraction score"
            ),
            "min_future_steps": int(SCRIPT_DEFAULTS["min_future_steps"]),
            "near_term_steps": int(SCRIPT_DEFAULTS["near_term_steps"]),
            "tail_top_fraction": float(SCRIPT_DEFAULTS["tail_top_fraction"]),
            "tail_min_topk": int(SCRIPT_DEFAULTS["tail_min_topk"]),
            "event_score_weight": float(SCRIPT_DEFAULTS["w_event"]),
            "near_score_weight": float(SCRIPT_DEFAULTS["w_near"]),
            "score_components": [
                "event_tail_score",
                "near_tail_score",
                "event_ttc_component",
                "event_thw_component",
                "event_gap_component",
                "event_drac_component",
                "event_dv_component",
                "near_ttc_component",
                "near_thw_component",
                "near_gap_component",
                "near_drac_component",
                "near_dv_component",
            ],
            "score_weights": {
                "w_ttc": float(SCRIPT_DEFAULTS["w_ttc"]),
                "w_thw": float(SCRIPT_DEFAULTS["w_thw"]),
                "w_gap": float(SCRIPT_DEFAULTS["w_gap"]),
                "w_drac": float(SCRIPT_DEFAULTS["w_drac"]),
                "w_dv": float(SCRIPT_DEFAULTS["w_dv"]),
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
    _score_rows(rows)
    _save_outputs(rows, skipped)


if __name__ == "__main__":
    main()
