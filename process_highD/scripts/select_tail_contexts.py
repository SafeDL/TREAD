#!/usr/bin/env python3
"""Select shared long-tail highD contexts for adversarial and subset tasks."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import setup_logging
from utils.evt import load_evt_model
from utils.highd_longitudinal import (
    DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    load_highd_event_context_cache,
)
from utils.io import write_json


PASSENGER_CAR_CLASS = "car"
SOURCE_INDEPENDENT_TAIL_PEAK = "highd_independent_tail_peak"
SOURCE_EVENT_TAIL = "highd_event_tail"
SOURCE_TAIL_FEATURE_KDE_KNN = "highd_tail_feature_kde_knn"
CONTEXT_METHOD_EMPIRICAL = "empirical"
CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN = "tail_feature_kde_knn"
LOG_GAP_IDX = 0
EGO_SPEED_IDX = 1
LEAD_SPEED_IDX = 2
CLOSING_SPEED_IDX = 3
EGO_ACCEL_IDX = 4
LEAD_ACCEL_IDX = 5
LATERAL_OFFSET_IDX = 6


SCRIPT_DEFAULTS: dict[str, Any] = {
    **DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    "event_context_cache_path": (
        ROOT / "results" / "highd_events" / "following_event_contexts.npz"
    ),
    "tail_context_path": (
        ROOT / "results" / "highd_following_tail" / "contexts" / "tail_contexts.npz"
    ),
    "independent_tail_peaks_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "exposure"
        / "highd_independent_tail_peaks.csv"
    ),
    "evt_model_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "evt"
        / "longitudinal_peak_evt_model.json"
    ),
    "evt_summary_path": (
        ROOT
        / "results"
        / "highd_following_tail"
        / "evt"
        / "longitudinal_peak_evt_summary.json"
    ),
    "collision_critical_level": 5.0,
    "evt_target_mode": "collision_critical_level",
    "context_source": "independent_tail_peaks",
    "num_contexts": 0,
    "require_passenger_car_ego": True,
    "passenger_car_max_length": 6.0,
    "context_generation_method": CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN,
    "num_synthetic_contexts": 7500,
    "include_empirical_contexts": True,
    "tail_feature_bandwidth": 0.20,
    "tail_feature_knn_clip_quantile": 0.01,
    "selection_random_seed": 42,
    "evt_return_period": 100,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)

_CONTEXT_KEYS: tuple[str, ...] = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "ego_class",
    "target_class",
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
    "risk_score",
    "evt_tail_probability",
    "peak_id",
    "representative_event_id",
    "base_context_index",
    "base_event_id",
    "synthetic_context",
    "context_model_method",
    "context_feature_distance",
)

_INT_CONTEXT_KEYS = {
    "recording_id",
    "ego_id",
    "target_id",
    "anchor_frame",
    "base_context_index",
    "synthetic_context",
}

_FLOAT_CONTEXT_KEYS = {
    "ego_length",
    "adv_length",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "collision",
    "near_collision",
    "y_long",
    "risk_score",
    "evt_tail_probability",
    "context_feature_distance",
}

_STRING_CONTEXT_KEYS = {
    "event_id",
    "ego_class",
    "target_class",
    "peak_id",
    "representative_event_id",
    "base_event_id",
    "context_model_method",
}

_TAIL_FEATURE_NAMES: tuple[str, ...] = (
    "log_initial_gap",
    "ego_speed",
    "lead_speed",
    "closing_speed",
    "ego_accel",
    "lead_accel",
    "lateral_offset",
)


def _collision_critical_level() -> float:
    summary_path = Path(SCRIPT_DEFAULTS["evt_summary_path"])
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if "collision_critical_level" in summary:
            return float(summary["collision_critical_level"])
    return SCRIPT_DEFAULTS["collision_critical_level"]


def _apply_evt_scoring(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], Path]:
    """Score rows with the EVT model and return per-row scores + dataset constants."""
    evt_model_path = Path(SCRIPT_DEFAULTS["evt_model_path"])
    if not evt_model_path.exists():
        raise FileNotFoundError(
            "EVT model is required before tail context selection: "
            f"{evt_model_path}. Run "
            "process_highD/scripts/fit_longitudinal_peak_evt.py first."
        )
    model = load_evt_model(evt_model_path)
    return_period = int(SCRIPT_DEFAULTS["evt_return_period"])
    collision_critical_level = _collision_critical_level()
    if str(SCRIPT_DEFAULTS.get("evt_target_mode")) == "collision_critical_level":
        target = collision_critical_level
    else:
        target = float(model.return_level(return_period))
    failure_threshold = float(model.score(target))
    tail_threshold_u = float(model.u)
    tail_threshold_score = float(model.score(tail_threshold_u))
    exceedance_rate = float(model.exceedance_rate)

    y_long = np.asarray([row["y_long"] for row in rows], dtype=np.float64)
    risk_score = np.asarray(model.score(y_long), dtype=np.float64)
    tail_probability = np.asarray(model.survival(y_long), dtype=np.float64)

    for idx, row in enumerate(rows):
        row["risk_score"] = float(risk_score[idx])
        row["evt_tail_probability"] = float(tail_probability[idx])

    # Return dataset-level constants separately (written once in summary, not per row).
    evt_meta = {
        "evt_tail_threshold_u": tail_threshold_u,
        "evt_tail_threshold_score": tail_threshold_score,
        "evt_exceedance_rate": exceedance_rate,
        "evt_return_period": return_period,
        "evt_return_level_target": target,
        "evt_failure_threshold": failure_threshold,
        "evt_model_path": str(evt_model_path),
        "evt_target_mode": str(SCRIPT_DEFAULTS.get("evt_target_mode")),
        "collision_critical_level": collision_critical_level,
    }
    return evt_meta, evt_model_path


def _load_cached_rows() -> list[dict[str, Any]]:
    cache_path = Path(SCRIPT_DEFAULTS["event_context_cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(
            "highD following context cache is required before tail selection: "
            f"{cache_path}. Run process_highD/scripts/extract_highd_events.py first."
        )
    rows = load_highd_event_context_cache(cache_path)
    if not rows:
        raise RuntimeError(f"highD following context cache is empty: {cache_path}")
    logger.info("Loaded %d highD following contexts from %s", len(rows), cache_path)
    return rows


def _load_rows() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows = _load_cached_rows()
    rows = _filter_passenger_car_ego(rows)
    evt_meta, _ = _apply_evt_scoring(rows)
    return rows, evt_meta, "following_event_context_cache"


def _filter_passenger_car_ego(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not bool(SCRIPT_DEFAULTS.get("require_passenger_car_ego", True)):
        return rows
    max_length = float(SCRIPT_DEFAULTS.get("passenger_car_max_length", 6.0))
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        ego_class = str(row.get("ego_class", "")).strip().lower()
        if ego_class:
            is_car = ego_class == PASSENGER_CAR_CLASS
        else:
            # Legacy caches may not include vehicle class metadata.
            is_car = float(row.get("ego_length", float("inf"))) <= max_length
        if is_car:
            kept.append(row)
        else:
            dropped += 1
    if dropped:
        logger.info(
            "Filtered %d contexts whose ego vehicle is not a passenger car",
            dropped,
        )
    if not kept:
        raise RuntimeError("No passenger-car ego following contexts remain")
    return kept


def _tail_feature(row: dict[str, Any]) -> np.ndarray:
    states = np.asarray(row["context_states"], dtype=np.float32)
    ego = states[-1, 0]
    lead = states[-1, 1]
    ego_length = float(row.get("ego_length", 4.8))
    lead_length = float(row.get("adv_length", 4.8))
    gap = float(lead[0] - ego[0] - 0.5 * (ego_length + lead_length))
    gap = max(gap, 0.2)
    ego_speed = max(float(ego[2]), 0.0)
    lead_speed = max(float(lead[2]), 0.0)
    return np.asarray(
        [
            np.log(gap),
            ego_speed,
            lead_speed,
            ego_speed - lead_speed,
            float(ego[4]),
            float(lead[4]),
            float(lead[1] - ego[1]),
        ],
        dtype=np.float64,
    )


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_tail_feature(row) for row in rows], axis=0)


def _reconstruct_context_from_feature(
    base_row: dict[str, Any],
    target_feature: np.ndarray,
) -> np.ndarray:
    states = np.asarray(base_row["context_states"], dtype=np.float32).copy()
    base_feature = _tail_feature(base_row)

    target_gap = float(np.exp(float(target_feature[LOG_GAP_IDX])))
    base_gap = float(np.exp(float(base_feature[LOG_GAP_IDX])))
    states[:, 1, 0] += np.float32(target_gap - base_gap)

    for actor, feature_idx in ((0, EGO_SPEED_IDX), (1, LEAD_SPEED_IDX)):
        target_speed = max(float(target_feature[feature_idx]), 0.0)
        base_speed = max(float(base_feature[feature_idx]), 0.0)
        states[:, actor, 2] += np.float32(target_speed - base_speed)
        states[-1, actor, 2] = np.float32(target_speed)

    for actor, feature_idx in ((0, EGO_ACCEL_IDX), (1, LEAD_ACCEL_IDX)):
        delta_accel = float(target_feature[feature_idx] - base_feature[feature_idx])
        states[:, actor, 4] = np.clip(
            states[:, actor, 4] + np.float32(delta_accel),
            -8.0,
            4.0,
        )

    states[:, 1, 1] += np.float32(
        target_feature[LATERAL_OFFSET_IDX] - base_feature[LATERAL_OFFSET_IDX]
    )
    return states


def _sample_tail_feature_contexts(
    rows: list[dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if not rows:
        raise RuntimeError("Cannot sample synthetic tail contexts from an empty pool")

    features = _feature_matrix(rows)
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    standardized = (features - center) / scale

    q = float(SCRIPT_DEFAULTS.get("tail_feature_knn_clip_quantile", 0.01))
    q = min(max(q, 0.0), 0.49)
    lower = np.quantile(features, q, axis=0)
    upper = np.quantile(features, 1.0 - q, axis=0)
    bandwidth = max(float(SCRIPT_DEFAULTS.get("tail_feature_bandwidth", 0.20)), 0.0)
    if len(rows) > 1 and bandwidth > 0.0:
        cov = np.cov(standardized, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0], dtype=np.float64) * 1.0e-4
    else:
        cov = np.eye(features.shape[1], dtype=np.float64) * 1.0e-4

    sampled: list[dict[str, Any]] = []
    for idx in range(int(count)):
        seed_idx = int(rng.integers(0, len(rows)))
        z = standardized[seed_idx].copy()
        if bandwidth > 0.0:
            z += rng.multivariate_normal(
                np.zeros(features.shape[1], dtype=np.float64),
                cov * bandwidth * bandwidth,
            )
        target_feature = np.clip(center + z * scale, lower, upper)
        target_standardized = (target_feature - center) / scale
        distance = np.sum(
            (standardized - target_standardized[None, :]) ** 2,
            axis=1,
        )
        base_idx = int(np.argmin(distance))
        base = rows[base_idx]
        item = dict(base)
        item["context_states"] = _reconstruct_context_from_feature(
            base,
            target_feature,
        )
        item["source_type"] = SOURCE_TAIL_FEATURE_KDE_KNN
        item["event_id"] = (
            f"synthetic_tail_{idx:05d}_base_{base.get('event_id', base_idx)}"
        )
        item["base_context_index"] = base_idx
        item["base_event_id"] = str(base.get("event_id", ""))
        item["synthetic_context"] = 1
        item["context_model_method"] = CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN
        item["context_feature_distance"] = float(np.sqrt(distance[base_idx]))
        new_feature = _tail_feature(item)
        item["initial_gap"] = float(np.exp(new_feature[LOG_GAP_IDX]))
        item["initial_closing_speed"] = float(new_feature[CLOSING_SPEED_IDX])
        sampled.append(item)
    return sampled


def _independent_peak_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peaks_path = Path(SCRIPT_DEFAULTS["independent_tail_peaks_path"])
    if not peaks_path.exists():
        raise FileNotFoundError(
            "Independent highD tail peaks are required for strict tail context "
            f"selection: {peaks_path}. Run "
            "process_highD/scripts/estimate_highd_exposure.py first, or set "
            "SCRIPT_DEFAULTS['context_source']='raw_tail_events' for diagnostic "
            "legacy selection."
        )
    import pandas as pd

    peaks = pd.read_csv(peaks_path)
    required = {"representative_event_id", "peak_id"}
    missing = sorted(required - set(peaks.columns))
    if missing:
        raise KeyError(f"{peaks_path} is missing required columns: {missing}")

    by_event_id = {str(row["event_id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    missing_events: list[str] = []
    for _, peak in peaks.iterrows():
        event_id = str(peak["representative_event_id"])
        source = by_event_id.get(event_id)
        if source is None:
            missing_events.append(event_id)
            continue
        item = dict(source)
        for key, value in peak.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            item[key] = value
        selected.append(item)
    if missing_events:
        logger.warning(
            "Skipped %d independent peaks whose representative events were "
            "not found in the context cache",
            len(missing_events),
        )
    if not selected:
        raise RuntimeError(
            "No independent tail peak contexts could be matched from "
            f"{peaks_path}"
        )
    return selected


def _raw_tail_rows(
    rows: list[dict[str, Any]],
    tail_threshold_u: float,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if np.isfinite(float(row["y_long"]))
        and float(row["y_long"]) > tail_threshold_u
    ]


def _save_outputs(
    rows: list[dict[str, Any]],
    evt_meta: dict[str, Any],
    input_source: str,
) -> None:
    tail_context_path = Path(SCRIPT_DEFAULTS["tail_context_path"])
    tail_context_path.parent.mkdir(parents=True, exist_ok=True)

    num_contexts = int(SCRIPT_DEFAULTS["num_contexts"])
    score = np.asarray(
        [row["risk_score"] for row in rows],
        dtype=np.float32,
    )
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise RuntimeError("No finite highD tail risk scores were produced")

    tail_threshold_u = float(evt_meta["evt_tail_threshold_u"])
    context_source = str(
        SCRIPT_DEFAULTS.get("context_source", "independent_tail_peaks")
    )
    if context_source == "independent_tail_peaks":
        candidate_rows = _independent_peak_rows(rows)
        source_type = SOURCE_INDEPENDENT_TAIL_PEAK
        tail_selection_method = "evt_pot_threshold_declustered_peaks"
        context_distribution = "uniform over selected highD independent tail peaks"
    elif context_source == "raw_tail_events":
        candidate_rows = _raw_tail_rows(rows, tail_threshold_u)
        source_type = SOURCE_EVENT_TAIL
        tail_selection_method = "evt_pot_threshold"
        context_distribution = "uniform over selected highD raw tail following events"
    else:
        raise ValueError(f"Unsupported context_source: {context_source}")

    num_available_tail_contexts = int(len(candidate_rows))
    for row in candidate_rows:
        row["source_type"] = source_type
        row["synthetic_context"] = 0
        row["context_model_method"] = CONTEXT_METHOD_EMPIRICAL
        row["context_feature_distance"] = 0.0
        row["base_context_index"] = -1
        row["base_event_id"] = ""
    if num_contexts > 0:
        rng = np.random.default_rng(int(SCRIPT_DEFAULTS["selection_random_seed"]))
        sample_size = min(num_contexts, num_available_tail_contexts)
        chosen = rng.choice(
            np.arange(num_available_tail_contexts),
            size=sample_size,
            replace=False,
        )
        candidate_rows = [candidate_rows[int(idx)] for idx in chosen]
    if not candidate_rows:
        raise RuntimeError(
            "No tail contexts found above EVT POT threshold "
            f"u={tail_threshold_u:.6g}"
        )
    selected = candidate_rows
    context_generation_method = str(
        SCRIPT_DEFAULTS.get(
            "context_generation_method",
            CONTEXT_METHOD_EMPIRICAL,
        )
    )
    num_synthetic_contexts = int(SCRIPT_DEFAULTS.get("num_synthetic_contexts", 0))
    if context_generation_method == CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN:
        rng = np.random.default_rng(int(SCRIPT_DEFAULTS["selection_random_seed"]))
        synthetic_rows = _sample_tail_feature_contexts(
            candidate_rows,
            count=num_synthetic_contexts,
            rng=rng,
        )
        selected = (
            candidate_rows
            if bool(SCRIPT_DEFAULTS.get("include_empirical_contexts", True))
            else []
        ) + synthetic_rows
        context_distribution = (
            "empirical highD tail contexts plus KDE-smoothed low-dimensional "
            "tail-feature samples reconstructed by nearest-neighbor contexts"
        )
    elif context_generation_method != CONTEXT_METHOD_EMPIRICAL:
        raise ValueError(
            f"Unsupported context_generation_method: {context_generation_method}"
        )
    tail_sampling_method = (
        "uniform_random_without_replacement"
        if num_contexts > 0
        else f"all_{context_source}"
    )

    collision_critical_level = evt_meta["collision_critical_level"]

    payload: dict[str, np.ndarray] = {
        "context_states": np.asarray(
            [row["context_states"] for row in selected],
            dtype=np.float32,
        ),
        "source_type": np.asarray(
            [row.get("source_type", source_type) for row in selected],
            dtype=object,
        ),
        "tail_threshold": np.asarray(tail_threshold_u, dtype=np.float32),
        "collision_critical_level": np.asarray(
            collision_critical_level,
            dtype=np.float32,
        ),
    }
    for key in _CONTEXT_KEYS:
        if all(key in row for row in selected):
            if key in _INT_CONTEXT_KEYS:
                dtype = np.int32
            elif key in _FLOAT_CONTEXT_KEYS:
                dtype = np.float32
            elif key in _STRING_CONTEXT_KEYS:
                dtype = object
            else:
                dtype = None
            payload[key] = np.asarray(
                [row[key] for row in selected],
                dtype=dtype,
            )
    np.savez_compressed(tail_context_path, **payload)

    selected_fraction = len(candidate_rows) / max(num_available_tail_contexts, 1)
    write_json(
        tail_context_path.with_name("tail_context_summary.json"),
        {
            **evt_meta,
            "input_source": input_source,
            "context_source": context_source,
            "tail_contexts": str(tail_context_path),
            "context_distribution": context_distribution,
            "num_scored_events": int(len(rows)),
            "num_available_tail_contexts": num_available_tail_contexts,
            "num_contexts": int(len(selected)),
            "num_empirical_contexts": int(len(candidate_rows)),
            "num_synthetic_contexts": int(
                sum(int(row.get("synthetic_context", 0)) for row in selected)
            ),
            "selected_tail_fraction": float(selected_fraction),
            "tail_selection_method": tail_selection_method,
            "tail_sampling_method": tail_sampling_method,
            "context_generation_method": context_generation_method,
            "tail_feature_names": list(_TAIL_FEATURE_NAMES),
            "tail_feature_bandwidth": float(
                SCRIPT_DEFAULTS.get("tail_feature_bandwidth", 0.20)
            ),
            "selection_random_seed": int(SCRIPT_DEFAULTS["selection_random_seed"]),
            "tail_threshold": tail_threshold_u,
            "score_min": float(np.min(finite_score)),
            "score_mean": float(np.mean(finite_score)),
            "score_p95": float(np.percentile(finite_score, 95.0)),
            "score_max": float(np.max(finite_score)),
            "min_future_steps": int(SCRIPT_DEFAULTS["min_future_steps"]),
            "available_future_steps_min_selected": int(
                min(row["available_future_steps"] for row in selected)
            ),
            "available_future_steps_max_selected": int(
                max(row["available_future_steps"] for row in selected)
            ),
        },
    )
    logger.info(
        "Wrote %d highD tail contexts from %d available tail events to %s",
        len(selected),
        num_available_tail_contexts,
        tail_context_path,
    )


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    rows, evt_meta, input_source = _load_rows()
    _save_outputs(rows, evt_meta, input_source)


if __name__ == "__main__":
    main()
