#!/usr/bin/env python3
"""Select shared long-tail highD contexts for adversarial and subset tasks."""
from __future__ import annotations

import logging
import json
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
)

_INT_CONTEXT_KEYS = {
    "recording_id",
    "ego_id",
    "target_id",
    "anchor_frame",
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
}

_STRING_CONTEXT_KEYS = {
    "event_id",
    "peak_id",
    "representative_event_id",
}


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
    evt_meta, _ = _apply_evt_scoring(rows)
    return rows, evt_meta, "following_event_context_cache"


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
        source_type = "highd_independent_tail_peak"
        tail_selection_method = "evt_pot_threshold_declustered_peaks"
        context_distribution = "uniform over selected highD independent tail peaks"
    elif context_source == "raw_tail_events":
        candidate_rows = _raw_tail_rows(rows, tail_threshold_u)
        source_type = "highd_event_tail"
        tail_selection_method = "evt_pot_threshold"
        context_distribution = "uniform over selected highD raw tail following events"
    else:
        raise ValueError(f"Unsupported context_source: {context_source}")

    num_available_tail_contexts = int(len(candidate_rows))
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
        "source_type": np.asarray(source_type, dtype=object),
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

    selected_fraction = len(selected) / max(num_available_tail_contexts, 1)
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
            "selected_tail_fraction": float(selected_fraction),
            "tail_selection_method": tail_selection_method,
            "tail_sampling_method": tail_sampling_method,
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
