#!/usr/bin/env python3
"""Select shared long-tail highD contexts for adversarial and subset tasks."""
from __future__ import annotations

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
from utils.io import write_csv, write_json


SCRIPT_DEFAULTS: dict[str, Any] = {
    **DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    "events_csv": ROOT / "results" / "highd_events" / "events.csv",
    "event_context_cache_path": (
        ROOT / "results" / "highd_events" / "following_event_contexts.npz"
    ),
    "tail_context_path": (
        ROOT / "results" / "highd_tail_contexts" / "following" / "tail_contexts.npz"
    ),
    "tail_score_path": (
        ROOT / "results" / "highd_tail_contexts" / "following" / "tail_scores.npz"
    ),
    "evt_model_path": (
        ROOT / "results" / "highd_evt" / "following" / "longitudinal_evt_model.json"
    ),
    "num_contexts": 500,
    "selection_random_seed": 42,
    "evt_return_period": 100,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _apply_evt_scoring(rows: list[dict[str, Any]]) -> Path:
    evt_model_path = Path(SCRIPT_DEFAULTS["evt_model_path"])
    if not evt_model_path.exists():
        raise FileNotFoundError(
            "EVT model is required before tail context selection: "
            f"{evt_model_path}. Run process_highD/scripts/fit_longitudinal_evt.py first."
        )
    model = load_evt_model(evt_model_path)
    return_period = int(SCRIPT_DEFAULTS["evt_return_period"])
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
        row["evt_return_period"] = return_period
        row["evt_return_level_target"] = target
        row["evt_failure_threshold"] = failure_threshold
        row["evt_tail_threshold_u"] = tail_threshold_u
        row["evt_tail_threshold_score"] = tail_threshold_score
        row["evt_exceedance_rate"] = exceedance_rate
    return evt_model_path


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


def _load_rows() -> tuple[list[dict[str, Any]], int, str]:
    rows = _load_cached_rows()
    _apply_evt_scoring(rows)
    return rows, 0, "following_event_context_cache"


def _save_outputs(rows: list[dict[str, Any]], skipped: int, input_source: str) -> None:
    tail_context_path = Path(SCRIPT_DEFAULTS["tail_context_path"])
    tail_score_path = Path(SCRIPT_DEFAULTS["tail_score_path"])
    tail_context_path.parent.mkdir(parents=True, exist_ok=True)
    tail_score_path.parent.mkdir(parents=True, exist_ok=True)

    num_contexts = int(SCRIPT_DEFAULTS["num_contexts"])
    score = np.asarray(
        [row["risk_score"] for row in rows],
        dtype=np.float32,
    )
    y_long = np.asarray(
        [row["y_long"] for row in rows],
        dtype=np.float32,
    )
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise RuntimeError("No finite highD tail risk scores were produced")
    tail_threshold_u = float(rows[0]["evt_tail_threshold_u"])
    tail_threshold_score = float(rows[0]["evt_tail_threshold_score"])
    evt_exceedance_rate = float(rows[0]["evt_exceedance_rate"])
    tail_idx = np.where(np.isfinite(y_long) & (y_long > tail_threshold_u))[0]
    num_available_tail_contexts = int(tail_idx.size)
    if num_contexts > 0:
        rng = np.random.default_rng(int(SCRIPT_DEFAULTS["selection_random_seed"]))
        sample_size = min(num_contexts, num_available_tail_contexts)
        tail_idx = rng.choice(tail_idx, size=sample_size, replace=False)
    if tail_idx.size == 0:
        raise RuntimeError(
            "No tail contexts found above EVT POT threshold "
            f"u={tail_threshold_u:.6g}"
        )
    selected = [rows[int(idx)] for idx in tail_idx]
    tail_sampling_method = (
        "uniform_random_without_replacement"
        if num_contexts > 0
        else "all_evt_tail_contexts"
    )

    score_keys = (
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
        "evt_return_period",
        "evt_return_level_target",
        "evt_failure_threshold",
        "evt_tail_threshold_u",
        "evt_tail_threshold_score",
        "evt_exceedance_rate",
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
            [tail_threshold_u] * len(selected),
            dtype=np.float32,
        ),
        "tail_score_threshold": np.asarray(
            [tail_threshold_score] * len(selected),
            dtype=np.float32,
        ),
        "tail_selection_method": np.asarray(
            ["evt_pot_threshold"] * len(selected),
            dtype=object,
        ),
        "tail_sampling_method": np.asarray(
            [tail_sampling_method] * len(selected),
            dtype=object,
        ),
    }
    for key in score_keys:
        payload[key] = np.asarray([row[key] for row in selected])
    np.savez_compressed(tail_context_path, **payload)

    selected_fraction = len(selected) / max(num_available_tail_contexts, 1)
    write_json(
        tail_context_path.with_name("tail_context_summary.json"),
        {
            "events_csv": str(SCRIPT_DEFAULTS["events_csv"]),
            "event_context_cache_path": str(SCRIPT_DEFAULTS["event_context_cache_path"]),
            "input_source": input_source,
            "tail_scores": str(tail_score_path),
            "tail_contexts": str(tail_context_path),
            "context_distribution": (
                "uniform over selected highD tail following events"
            ),
            "event_level_distribution": "all valid highD following events; no train/val/test split",
            "num_scored_events": int(len(rows)),
            "num_available_tail_contexts": num_available_tail_contexts,
            "num_contexts": int(len(selected)),
            "selected_tail_fraction": float(selected_fraction),
            "skipped_events": int(skipped),
            "tail_selection_method": "evt_pot_threshold",
            "tail_sampling_method": tail_sampling_method,
            "selection_random_seed": int(SCRIPT_DEFAULTS["selection_random_seed"]),
            "tail_threshold_space": "y_long",
            "tail_threshold": tail_threshold_u,
            "evt_tail_threshold_u": tail_threshold_u,
            "evt_tail_threshold_score": tail_threshold_score,
            "evt_exceedance_rate": evt_exceedance_rate,
            "evt_return_period": int(SCRIPT_DEFAULTS["evt_return_period"]),
            "evt_return_level_target": float(rows[0]["evt_return_level_target"]),
            "evt_failure_threshold": float(rows[0]["evt_failure_threshold"]),
            "score_min": float(np.min(finite_score)),
            "score_mean": float(np.mean(finite_score)),
            "score_p95": float(np.percentile(finite_score, 95.0)),
            "score_max": float(np.max(finite_score)),
            "scoring_method": (
                "shared RSS-free longitudinal risk over the anchor-to-end "
                "event suffix: softmax-pool(1/TTC, 1/THW, 1/gap, DRAC) "
                "plus collision, near-collision, and hard-brake terms; "
                "risk_score is S_EVT(y_long); tail events are y_long > "
                "the fitted EVT POT threshold u"
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
        "Wrote %d highD tail contexts from %d available tail events to %s",
        len(selected),
        num_available_tail_contexts,
        tail_context_path,
    )


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    rows, skipped, input_source = _load_rows()
    _save_outputs(rows, skipped, input_source)


if __name__ == "__main__":
    main()
