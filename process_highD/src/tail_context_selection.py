"""Shared highD long-tail context selection engine."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from utils.evt import load_evt_model
from utils.io import write_json


SOURCE_INDEPENDENT_TAIL_PEAK = "highd_independent_tail_peak"
SOURCE_TAIL_FEATURE_KDE_KNN = "highd_tail_feature_kde_knn"
CONTEXT_METHOD_EMPIRICAL = "empirical"
CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN = "tail_feature_kde_knn"
LOG_GAP_IDX = 0
EGO_SPEED_IDX = 1
ADV_SPEED_IDX = 2
CLOSING_SPEED_IDX = 3
EGO_ACCEL_IDX = 4
ADV_ACCEL_IDX = 5
LATERAL_OFFSET_IDX = 6


COMMON_SELECTION_DEFAULTS: dict[str, Any] = {
    "evt_target_mode": "collision_critical_level",
    "empirical_context_limit": None,
    "context_generation_method": CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN,
    "num_synthetic_contexts": 500,
    "include_empirical_contexts": True,
    "tail_feature_bandwidth": 0.20,
    "tail_feature_knn_clip_quantile": 0.01,
    "selection_random_seed": 42,
    "evt_return_period": 100,
    "min_future_steps": 5,
}
logger = logging.getLogger(__name__)

_COMMON_CONTEXT_KEYS: tuple[str, ...] = (
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

_COMMON_CONTEXT_KEY_DTYPES = {
    "recording_id": "int",
    "event_id": "str",
    "ego_id": "int",
    "target_id": "int",
    "anchor_frame": "int",
    "ego_length": "float",
    "adv_length": "float",
    "initial_gap": "float",
    "initial_closing_speed": "float",
    "recorded_min_gap": "float",
    "recorded_min_ttc": "float",
    "collision": "float",
    "near_collision": "float",
    "risk_score": "float",
    "evt_tail_probability": "float",
    "peak_id": "str",
    "representative_event_id": "str",
    "base_context_index": "int",
    "base_event_id": "str",
    "synthetic_context": "int",
    "context_model_method": "str",
    "context_feature_distance": "float",
}

_DTYPE_ALIASES = {
    "int": np.int32,
    "float": np.float32,
    "str": object,
    "object": object,
}

_SYNTHETIC_CONTEXT_FIELDS = {
    "base_context_index": -1,
    "base_event_id": "",
    "synthetic_context": 0,
    "context_model_method": CONTEXT_METHOD_EMPIRICAL,
    "context_feature_distance": 0.0,
}

_SYNTHETIC_CONTEXT_DTYPES = {
    "base_context_index": "int",
    "base_event_id": "str",
    "synthetic_context": "int",
    "context_model_method": "str",
    "context_feature_distance": "float",
}


def _unique_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _resolve_np_dtype(label: str | type | None) -> object | type | None:
    if label is None:
        return None
    if isinstance(label, str):
        return _DTYPE_ALIASES.get(label, label)
    return label


def _context_key_dtypes(config: dict[str, Any]) -> dict[str, object | type | None]:
    configured = dict(config.get("context_key_dtypes", {}))
    labels = {
        **_COMMON_CONTEXT_KEY_DTYPES,
        **_SYNTHETIC_CONTEXT_DTYPES,
        **configured,
    }
    return {key: _resolve_np_dtype(value) for key, value in labels.items()}


def _context_output_keys(config: dict[str, Any]) -> tuple[str, ...]:
    risk_key = str(config["risk_value_key"])
    configured = tuple(str(key) for key in config.get("context_output_keys", ()))
    return _unique_keys(_COMMON_CONTEXT_KEYS + (risk_key,) + configured)


_TAIL_FEATURE_NAMES: tuple[str, ...] = (
    "log_initial_gap",
    "ego_speed",
    "adv_speed",
    "closing_speed",
    "ego_accel",
    "adv_accel",
    "lateral_offset",
)


def _merged_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {**COMMON_SELECTION_DEFAULTS, **config}
    required = {
        "event_context_cache_path",
        "tail_context_path",
        "independent_tail_peaks_path",
        "evt_model_path",
        "evt_summary_path",
        "scenario",
        "risk_value_key",
        "context_loader",
    }
    missing = sorted(key for key in required if key not in cfg)
    if missing:
        raise KeyError(f"Tail context selection config missing keys: {missing}")
    return cfg


def _collision_critical_level(config: dict[str, Any]) -> float:
    summary_path = Path(config["evt_summary_path"])
    if not summary_path.exists():
        raise FileNotFoundError(
            "EVT summary is required before tail context selection: "
            f"{summary_path}"
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if "collision_critical_level" not in summary:
        raise KeyError(f"{summary_path} is missing collision_critical_level")
    return float(summary["collision_critical_level"])


def _apply_evt_scoring(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Score rows with the EVT model and return per-row scores + dataset constants."""
    evt_model_path = Path(config["evt_model_path"])
    if not evt_model_path.exists():
        raise FileNotFoundError(
            "EVT model is required before tail context selection: "
            f"{evt_model_path}. Run "
            f"{config.get('fit_evt_hint', 'the scenario EVT fitting script')} first."
        )
    model = load_evt_model(evt_model_path)
    return_period = int(config["evt_return_period"])
    collision_critical_level = _collision_critical_level(config)
    if str(config["evt_target_mode"]) == "collision_critical_level":
        target = collision_critical_level
    else:
        target = float(model.return_level(return_period))
    failure_threshold = float(model.score(target))
    tail_threshold_u = float(model.u)
    tail_threshold_score = float(model.score(tail_threshold_u))
    exceedance_rate = float(model.exceedance_rate)

    risk_key = str(config["risk_value_key"])
    values = np.asarray([row[risk_key] for row in rows], dtype=np.float64)
    risk_score = np.asarray(model.score(values), dtype=np.float64)
    tail_probability = np.asarray(model.survival(values), dtype=np.float64)

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
        "evt_target_mode": str(config["evt_target_mode"]),
        "collision_critical_level": collision_critical_level,
        "risk_value_key": risk_key,
        "scenario": str(config["scenario"]),
    }
    return evt_meta, evt_model_path


def _load_cached_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    cache_path = Path(config["event_context_cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(
            "highD event context cache is required before tail selection: "
            f"{cache_path}. Run process_highD/scripts/extract_highd_events.py first."
        )
    loader = config["context_loader"]
    if not callable(loader):
        raise TypeError("Tail context selection config context_loader must be callable")
    rows = loader(cache_path)
    if not rows:
        raise RuntimeError(f"highD event context cache is empty: {cache_path}")
    row_filter = config.get("row_filter")
    if row_filter is not None:
        if not callable(row_filter):
            raise TypeError("Tail context selection config row_filter must be callable")
        before = len(rows)
        rows = row_filter(rows)
        if not rows:
            raise RuntimeError(
                f"No highD {config['scenario']} contexts remain after row_filter: "
                f"{cache_path}"
            )
        removed = before - len(rows)
        if removed:
            logger.info(
                "Filtered %d cached highD %s contexts before tail selection",
                removed,
                config["scenario"],
            )
    required_history_steps = config.get("required_history_steps")
    if required_history_steps is not None:
        states = np.asarray(rows[0]["context_states"], dtype=np.float32)
        actual_history_steps = int(states.shape[0])
        expected_history_steps = int(required_history_steps)
        if actual_history_steps != expected_history_steps:
            raise ValueError(
                f"{config['scenario']} context cache has history_steps="
                f"{actual_history_steps}, expected {expected_history_steps}: "
                f"{cache_path}. Rebuild the highD event cache first with "
                "python process_highD/scripts/extract_highd_events.py"
            )
    logger.info(
        "Loaded %d highD %s contexts from %s",
        len(rows),
        config["scenario"],
        cache_path,
    )
    return rows


def _load_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows = _load_cached_rows(config)
    evt_meta, _ = _apply_evt_scoring(rows, config)
    return rows, evt_meta, f"{config['scenario']}_event_context_cache"


def _tail_feature(row: dict[str, Any]) -> np.ndarray:
    states = np.asarray(row["context_states"], dtype=np.float32)
    ego = states[-1, 0]
    adv = states[-1, 1]
    ego_length = float(row["ego_length"])
    adv_length = float(row["adv_length"])
    gap = float(adv[0] - ego[0] - 0.5 * (ego_length + adv_length))
    gap = max(gap, 0.2)
    ego_speed = max(float(np.hypot(ego[2], ego[3])), 0.0)
    adv_speed = max(float(np.hypot(adv[2], adv[3])), 0.0)
    return np.asarray(
        [
            np.log(gap),
            ego_speed,
            adv_speed,
            ego_speed - adv_speed,
            float(ego[4]),
            float(adv[4]),
            float(adv[1] - ego[1]),
        ],
        dtype=np.float64,
    )


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_tail_feature(row) for row in rows], axis=0)


def _scale_velocity_vector(
    states: np.ndarray,
    *,
    actor: int,
    target_speed: float,
    base_speed: float,
) -> None:
    if base_speed > 1.0e-6:
        scale = np.float32(target_speed / base_speed)
        states[:, actor, 2:4] *= scale
        return
    states[:, actor, 2] = np.float32(target_speed)
    states[:, actor, 3] = np.float32(0.0)


def _reconstruct_context_from_feature(
    base_row: dict[str, Any],
    target_feature: np.ndarray,
) -> np.ndarray:
    states = np.asarray(base_row["context_states"], dtype=np.float32).copy()
    base_feature = _tail_feature(base_row)

    target_gap = float(np.exp(float(target_feature[LOG_GAP_IDX])))
    base_gap = float(np.exp(float(base_feature[LOG_GAP_IDX])))
    states[:, 1, 0] += np.float32(target_gap - base_gap)

    for actor, feature_idx in ((0, EGO_SPEED_IDX), (1, ADV_SPEED_IDX)):
        target_speed = max(float(target_feature[feature_idx]), 0.0)
        base_speed = max(float(base_feature[feature_idx]), 0.0)
        _scale_velocity_vector(
            states,
            actor=actor,
            target_speed=target_speed,
            base_speed=base_speed,
        )

    for actor, feature_idx in ((0, EGO_ACCEL_IDX), (1, ADV_ACCEL_IDX)):
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
    config: dict[str, Any],
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

    q = float(config["tail_feature_knn_clip_quantile"])
    q = min(max(q, 0.0), 0.49)
    lower = np.quantile(features, q, axis=0)
    upper = np.quantile(features, 1.0 - q, axis=0)
    bandwidth = max(float(config["tail_feature_bandwidth"]), 0.0)
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
            f"synthetic_tail_{idx:05d}_base_{base['event_id']}"
        )
        item["base_context_index"] = base_idx
        item["base_event_id"] = str(base["event_id"])
        item["synthetic_context"] = 1
        item["context_model_method"] = CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN
        item["context_feature_distance"] = float(np.sqrt(distance[base_idx]))
        new_feature = _tail_feature(item)
        item["initial_gap"] = float(np.exp(new_feature[LOG_GAP_IDX]))
        item["initial_closing_speed"] = float(new_feature[CLOSING_SPEED_IDX])
        sampled.append(item)
    return sampled


def _independent_peak_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    peaks_path = Path(config["independent_tail_peaks_path"])
    if not peaks_path.exists():
        estimate_script = str(
            config.get(
                "estimate_exposure_hint",
                "the scenario exposure estimation script",
            )
        )
        raise FileNotFoundError(
            "Independent highD tail peaks are required for strict tail context "
            f"selection: {peaks_path}. Run "
            f"{estimate_script} first."
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
        raise KeyError(
            "Independent peaks reference events not found in the context cache: "
            f"{missing_events[:10]} (total={len(missing_events)})"
        )
    if not selected:
        raise RuntimeError(
            "No independent tail peak contexts could be matched from "
            f"{peaks_path}"
        )
    return selected


def _save_outputs(
    rows: list[dict[str, Any]],
    evt_meta: dict[str, Any],
    input_source: str,
    config: dict[str, Any],
) -> None:
    tail_context_path = Path(config["tail_context_path"])
    tail_context_path.parent.mkdir(parents=True, exist_ok=True)

    empirical_context_limit = config["empirical_context_limit"]
    if empirical_context_limit is not None:
        empirical_context_limit = int(empirical_context_limit)
        if empirical_context_limit <= 0:
            raise ValueError("empirical_context_limit must be positive or None")
    score = np.asarray(
        [row["risk_score"] for row in rows],
        dtype=np.float32,
    )
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise RuntimeError("No finite highD tail risk scores were produced")

    tail_threshold_u = float(evt_meta["evt_tail_threshold_u"])
    context_source = "independent_tail_peaks"
    candidate_rows = _independent_peak_rows(rows, config)
    source_type = SOURCE_INDEPENDENT_TAIL_PEAK
    tail_selection_method = "evt_pot_threshold_declustered_peaks"
    context_distribution = "uniform over selected highD independent tail peaks"

    num_available_tail_contexts = int(len(candidate_rows))
    for row in candidate_rows:
        row["source_type"] = source_type
        row.update(_SYNTHETIC_CONTEXT_FIELDS)
    if empirical_context_limit is not None:
        rng = np.random.default_rng(int(config["selection_random_seed"]))
        sample_size = min(empirical_context_limit, num_available_tail_contexts)
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
        config["context_generation_method"]
    )
    num_synthetic_contexts = int(config["num_synthetic_contexts"])
    if context_generation_method == CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN:
        rng = np.random.default_rng(int(config["selection_random_seed"]))
        synthetic_rows = _sample_tail_feature_contexts(
            candidate_rows,
            count=num_synthetic_contexts,
            rng=rng,
            config=config,
        )
        selected = (
            candidate_rows
            if bool(config["include_empirical_contexts"])
            else []
        ) + synthetic_rows
        context_distribution = (
            "empirical highD tail contexts plus KDE-smoothed low-dimensional "
            "tail-feature perturbations reconstructed by nearest-neighbor contexts"
        )
    elif context_generation_method != CONTEXT_METHOD_EMPIRICAL:
        raise ValueError(
            f"Unsupported context_generation_method: {context_generation_method}"
    )
    tail_sampling_method = (
        "uniform_random_without_replacement"
        if empirical_context_limit is not None
        else f"all_{context_source}"
    )
    selected_score = np.asarray(
        [row["risk_score"] for row in selected],
        dtype=np.float32,
    )

    collision_critical_level = evt_meta["collision_critical_level"]

    payload: dict[str, np.ndarray] = {
        "context_states": np.asarray(
            [row["context_states"] for row in selected],
            dtype=np.float32,
        ),
        "source_type": np.asarray(
            [row["source_type"] for row in selected],
            dtype=object,
        ),
        "tail_threshold": np.asarray(tail_threshold_u, dtype=np.float32),
        "collision_critical_level": np.asarray(
            collision_critical_level,
            dtype=np.float32,
        ),
    }
    context_key_dtypes = _context_key_dtypes(config)
    for key in _context_output_keys(config):
        if all(key in row for row in selected):
            payload[key] = np.asarray(
                [row[key] for row in selected],
                dtype=context_key_dtypes.get(key),
            )
    np.savez_compressed(tail_context_path, **payload)

    num_output_contexts = int(len(selected))
    num_output_synthetic_contexts = int(
        sum(int(row["synthetic_context"]) for row in selected)
    )
    num_output_empirical_contexts = num_output_contexts - num_output_synthetic_contexts
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
            "num_output_contexts": num_output_contexts,
            "empirical_context_limit": empirical_context_limit,
            "num_empirical_contexts": int(len(candidate_rows)),
            "num_output_empirical_contexts": num_output_empirical_contexts,
            "num_synthetic_contexts": num_output_synthetic_contexts,
            "selected_tail_fraction": float(selected_fraction),
            "tail_selection_method": tail_selection_method,
            "tail_sampling_method": tail_sampling_method,
            "context_generation_method": context_generation_method,
            "tail_feature_names": list(_TAIL_FEATURE_NAMES),
            "tail_feature_bandwidth": float(config["tail_feature_bandwidth"]),
            "selection_random_seed": int(config["selection_random_seed"]),
            "scenario": str(config["scenario"]),
            "risk_value_key": str(config["risk_value_key"]),
            "tail_threshold": tail_threshold_u,
            "score_min": float(np.min(selected_score)),
            "score_mean": float(np.mean(selected_score)),
            "score_p95": float(np.percentile(selected_score, 95.0)),
            "score_max": float(np.max(selected_score)),
            "min_future_steps": int(config["min_future_steps"]),
        },
    )
    logger.info(
        (
            "Wrote %d %s tail contexts to %s | scored_events=%d, "
            "available_real_tail_peaks=%d, output_real=%d, output_synthetic=%d"
        ),
        num_output_contexts,
        config["scenario"],
        tail_context_path,
        len(rows),
        num_available_tail_contexts,
        num_output_empirical_contexts,
        num_output_synthetic_contexts,
    )


def run_tail_context_selection(config: dict[str, Any]) -> None:
    """Build long-tail context space from an explicit scenario config."""
    cfg = _merged_config(config)
    rows, evt_meta, input_source = _load_rows(cfg)
    _save_outputs(rows, evt_meta, input_source, cfg)
