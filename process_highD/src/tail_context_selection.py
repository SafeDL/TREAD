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
SOURCE_TAIL_GAUSSIAN_COPULA = "highd_tail_gaussian_copula"
CONTEXT_METHOD_EMPIRICAL = "empirical"
CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN = "tail_feature_kde_knn"
CONTEXT_METHOD_GAUSSIAN_COPULA = "gaussian_copula"
FOLLOWING_EGO_VX_IDX = 0
FOLLOWING_LOG_GAP_IDX = 1
FOLLOWING_DELTA_V_IDX = 2
FOLLOWING_LEAD_AX_IDX = 3
CUTIN_EGO_VX_IDX = 0
CUTIN_LOG_GAP_IDX = 1
CUTIN_LATERAL_OFFSET_IDX = 2
CUTIN_DELTA_VX_IDX = 3
CUTIN_TARGET_VY_IDX = 4
CUTIN_TARGET_AY_IDX = 5
CUTIN_FINAL_LATERAL_OFFSET_IDX = 6
CUTIN_TIME_TO_CROSS_IDX = 7
CUTIN_TARGET_SPEED_CHANGE_IDX = 8
CUTIN_TARGET_SLOPE_AT_CROSS_IDX = 9


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
    "min_future_steps": 125,
    "copula_correlation_regularization": 1.0e-4,
    "copula_marginal_clip_quantile": 0.01,
    "generate_diffusion_rollouts": False,
    "num_diffusion_scenarios": 0,
    "diffusion_checkpoint_path": "checkpoints/best_noise_mse.pt",
    "generated_scenarios_path": None,
    "diffusion_batch_size": 256,
    "diffusion_inference_steps": None,
    "diffusion_device": "auto",
    "diffusion_seed": 42,
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


FOLLOWING_TAIL_FEATURE_NAMES: tuple[str, ...] = (
    "ego_vx_0",
    "log_initial_gap",
    "initial_delta_v",
    "lead_ax_0",
    "lead_speed_change",
    "lead_min_ax",
    "lead_braking_duration",
)

CUTIN_TAIL_FEATURE_NAMES: tuple[str, ...] = (
    "ego_vx_0",
    "log_initial_gap",
    "initial_lateral_offset",
    "initial_delta_vx",
    "target_vy_0",
    "target_ay_0",
    "final_lateral_offset",
    "time_to_cross",
    "target_speed_change",
    "target_slope_at_cross",
)


def _tail_feature_names(config: dict[str, Any]) -> tuple[str, ...]:
    return (
        CUTIN_TAIL_FEATURE_NAMES
        if str(config["scenario"]) == "cut_in"
        else FOLLOWING_TAIL_FEATURE_NAMES
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
    if "scenario_conditions" not in rows[0] or "initial_states" not in rows[0]:
        raise KeyError(
            f"{config['scenario']} context cache is not anchor-scenario. "
            f"Rebuild it first with python process_highD/scripts/extract_highd_events.py: "
            f"{cache_path}"
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


def _tail_feature(row: dict[str, Any], scenario: str) -> np.ndarray:
    conditions = np.asarray(row["scenario_conditions"], dtype=np.float64)
    gap = max(float(conditions[1]), 0.2)
    if str(scenario) == "cut_in":
        return np.asarray(
            [
                float(conditions[0]),
                np.log(gap),
                float(conditions[2]),
                float(conditions[3]),
                float(conditions[4]),
                float(conditions[5]),
                float(conditions[6]),
                float(conditions[7]),
                float(conditions[8]),
                float(conditions[9]),
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            float(conditions[0]),
            np.log(gap),
            float(conditions[2]),
            float(conditions[3]),
            float(conditions[4]),
            float(conditions[5]),
            float(conditions[6]),
        ],
        dtype=np.float64,
    )


def _feature_matrix(rows: list[dict[str, Any]], scenario: str) -> np.ndarray:
    return np.stack([_tail_feature(row, scenario) for row in rows], axis=0)


def _reconstruct_initial_from_feature(
    base_row: dict[str, Any],
    target_feature: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(base_row["initial_states"], dtype=np.float32).copy()
    ego_length = float(base_row["ego_length"])
    adv_length = float(base_row["adv_length"])

    if str(scenario) == "cut_in":
        target_gap = float(np.exp(float(target_feature[CUTIN_LOG_GAP_IDX])))
        ego_vx = max(float(target_feature[CUTIN_EGO_VX_IDX]), 0.0)
        delta_vx = float(target_feature[CUTIN_DELTA_VX_IDX])
        states[1, 0] = np.float32(
            states[0, 0] + 0.5 * (ego_length + adv_length) + target_gap
        )
        states[0, 2] = np.float32(ego_vx)
        states[1, 2] = np.float32(max(ego_vx - delta_vx, 0.0))
        states[1, 1] = np.float32(
            states[0, 1] + float(target_feature[CUTIN_LATERAL_OFFSET_IDX])
        )
        states[1, 3] = np.float32(float(target_feature[CUTIN_TARGET_VY_IDX]))
        states[1, 5] = np.float32(
            np.clip(float(target_feature[CUTIN_TARGET_AY_IDX]), -4.0, 4.0)
        )
        scenario_conditions = np.asarray(
            [
                ego_vx,
                target_gap,
                float(target_feature[CUTIN_LATERAL_OFFSET_IDX]),
                delta_vx,
                float(target_feature[CUTIN_TARGET_VY_IDX]),
                float(target_feature[CUTIN_TARGET_AY_IDX]),
                float(target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX]),
                float(target_feature[CUTIN_TIME_TO_CROSS_IDX]),
                float(target_feature[CUTIN_TARGET_SPEED_CHANGE_IDX]),
                float(target_feature[CUTIN_TARGET_SLOPE_AT_CROSS_IDX]),
            ],
            dtype=np.float32,
        )
        return scenario_conditions, states.astype(np.float32)

    target_gap = float(np.exp(float(target_feature[FOLLOWING_LOG_GAP_IDX])))
    ego_vx = max(float(target_feature[FOLLOWING_EGO_VX_IDX]), 0.0)
    delta_v = float(target_feature[FOLLOWING_DELTA_V_IDX])
    states[1, 0] = np.float32(states[0, 0] + 0.5 * (ego_length + adv_length) + target_gap)
    states[0, 2] = np.float32(ego_vx)
    states[1, 2] = np.float32(max(ego_vx - delta_v, 0.0))
    states[1, 4] = np.float32(
        np.clip(float(target_feature[FOLLOWING_LEAD_AX_IDX]), -8.0, 4.0)
    )
    scenario_conditions = np.asarray(
        [
            ego_vx,
            target_gap,
            delta_v,
            float(states[1, 4]),
            float(target_feature[4]),
            float(target_feature[5]),
            max(float(target_feature[6]), 0.0),
        ],
        dtype=np.float32,
    )
    return scenario_conditions, states.astype(np.float32)


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

    scenario = str(config["scenario"])
    features = _feature_matrix(rows, scenario)
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
        if str(scenario) == "cut_in":
            target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX] = np.clip(
                target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX], -1.0, 1.0
            )
        target_standardized = (target_feature - center) / scale
        distance = np.sum(
            (standardized - target_standardized[None, :]) ** 2,
            axis=1,
        )
        base_idx = int(np.argmin(distance))
        base = rows[base_idx]
        item = dict(base)
        scenario_conditions, initial_states = _reconstruct_initial_from_feature(
            base,
            target_feature,
            scenario,
        )
        item["scenario_conditions"] = scenario_conditions
        item["initial_states"] = initial_states
        item["source_type"] = SOURCE_TAIL_FEATURE_KDE_KNN
        item["event_id"] = (
            f"synthetic_tail_{idx:05d}_base_{base['event_id']}"
        )
        item["base_context_index"] = base_idx
        item["base_event_id"] = str(base["event_id"])
        item["synthetic_context"] = 1
        item["context_model_method"] = CONTEXT_METHOD_TAIL_FEATURE_KDE_KNN
        item["context_feature_distance"] = float(np.sqrt(distance[base_idx]))
        new_feature = _tail_feature(item, scenario)
        if str(scenario) == "cut_in":
            item["initial_gap"] = float(np.exp(new_feature[CUTIN_LOG_GAP_IDX]))
            item["initial_closing_speed"] = float(new_feature[CUTIN_DELTA_VX_IDX])
        else:
            item["initial_gap"] = float(np.exp(new_feature[FOLLOWING_LOG_GAP_IDX]))
            item["initial_closing_speed"] = float(new_feature[FOLLOWING_DELTA_V_IDX])
        sampled.append(item)
    return sampled


def _normal_score_pseudo_observations(features: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    ranks = np.empty_like(features, dtype=np.float64)
    n = int(features.shape[0])
    for col in range(int(features.shape[1])):
        order = np.argsort(features[:, col], kind="mergesort")
        ranks[order, col] = np.arange(1, n + 1, dtype=np.float64)
    u = ranks / float(n + 1)
    return ndtri(np.clip(u, 1.0e-6, 1.0 - 1.0e-6))


def _sample_gaussian_copula_contexts(
    rows: list[dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) < 2:
        raise RuntimeError(
            "Gaussian copula tail context sampling requires at least 2 rows"
        )

    from scipy.special import ndtr

    scenario = str(config["scenario"])
    features = _feature_matrix(rows, scenario)
    z = _normal_score_pseudo_observations(features)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.atleast_2d(corr).astype(np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    reg = max(float(config["copula_correlation_regularization"]), 0.0)
    corr = corr + np.eye(corr.shape[0], dtype=np.float64) * reg
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1.0e-8, None)
    corr = (eigvecs * eigvals[None, :]) @ eigvecs.T
    denom = np.sqrt(np.clip(np.diag(corr), 1.0e-12, None))
    corr = corr / denom[:, None] / denom[None, :]

    q = float(config["copula_marginal_clip_quantile"])
    q = min(max(q, 0.0), 0.49)
    lower = np.quantile(features, q, axis=0)
    upper = np.quantile(features, 1.0 - q, axis=0)
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    standardized = (features - center) / scale

    sampled_z = rng.multivariate_normal(
        np.zeros(features.shape[1], dtype=np.float64),
        corr,
        size=int(count),
        check_valid="ignore",
    )
    sampled_u = np.clip(ndtr(sampled_z), 1.0e-6, 1.0 - 1.0e-6)

    sampled: list[dict[str, Any]] = []
    for idx in range(int(count)):
        target_feature = np.asarray(
            [
                np.quantile(features[:, col], sampled_u[idx, col])
                for col in range(features.shape[1])
            ],
            dtype=np.float64,
        )
        target_feature = np.clip(target_feature, lower, upper)
        if str(scenario) == "cut_in":
            target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX] = np.clip(
                target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX], -1.0, 1.0
            )
        target_standardized = (target_feature - center) / scale
        distance = np.sum(
            (standardized - target_standardized[None, :]) ** 2,
            axis=1,
        )
        base_idx = int(np.argmin(distance))
        base = rows[base_idx]
        item = dict(base)
        scenario_conditions, initial_states = _reconstruct_initial_from_feature(
            base,
            target_feature,
            scenario,
        )
        item["scenario_conditions"] = scenario_conditions
        item["initial_states"] = initial_states
        item["source_type"] = SOURCE_TAIL_GAUSSIAN_COPULA
        item["event_id"] = (
            f"gaussian_copula_tail_{idx:05d}_base_{base['event_id']}"
        )
        item["base_context_index"] = base_idx
        item["base_event_id"] = str(base["event_id"])
        item["synthetic_context"] = 1
        item["context_model_method"] = CONTEXT_METHOD_GAUSSIAN_COPULA
        item["context_feature_distance"] = float(np.sqrt(distance[base_idx]))
        new_feature = _tail_feature(item, scenario)
        if scenario == "cut_in":
            item["initial_gap"] = float(np.exp(new_feature[CUTIN_LOG_GAP_IDX]))
            item["initial_closing_speed"] = float(new_feature[CUTIN_DELTA_VX_IDX])
        else:
            item["initial_gap"] = float(np.exp(new_feature[FOLLOWING_LOG_GAP_IDX]))
            item["initial_closing_speed"] = float(new_feature[FOLLOWING_DELTA_V_IDX])
        sampled.append(item)
    return sampled


def _diffusion_generated_path(config: dict[str, Any], tail_context_path: Path) -> Path:
    configured = config.get("generated_scenarios_path")
    if configured is not None:
        return Path(configured)
    return tail_context_path.parent / "diffusion_generated_scenarios.npz"


def _generate_diffusion_rollouts(
    selected: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    tail_context_path: Path,
) -> dict[str, Any] | None:
    if not bool(config["generate_diffusion_rollouts"]):
        return None
    if str(config["scenario"]) != "cut_in":
        raise ValueError(
            "Diffusion rollout generation is currently implemented for cut_in"
        )
    if "diffusion_dataset_dir" not in config:
        raise KeyError(
            "Tail context config requires diffusion_dataset_dir when "
            "generate_diffusion_rollouts=true"
        )

    import torch

    from diffusion.src.kinematics import integrate_cutin_acceleration_actions
    from diffusion.src.utils import set_seed
    from utils.diffusion_adapter import DiffusionPriorAdapter
    from utils.normalization import denormalize_torch, normalize_numpy

    output_path = _diffusion_generated_path(config, tail_context_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    natural_dir = Path(config["diffusion_dataset_dir"])
    checkpoint = Path(config["diffusion_checkpoint_path"])
    set_seed(int(config["diffusion_seed"]))
    adapter = DiffusionPriorAdapter.load(
        natural_dir,
        checkpoint,
        device=str(config["diffusion_device"]),
    )
    schema = adapter.schema
    if str(schema.get("event_type", "")).lower() != "cut_in":
        raise RuntimeError(
            "Configured diffusion checkpoint is not a cut-in prior: "
            f"{schema.get('event_type')}"
        )

    requested = int(config["num_diffusion_scenarios"])
    if requested <= 0:
        requested = int(len(selected))
    rng = np.random.default_rng(int(config["diffusion_seed"]))
    replace = requested > len(selected)
    context_indices = rng.choice(
        np.arange(len(selected)),
        size=requested,
        replace=replace,
    )
    conditions = np.asarray(
        [selected[int(idx)]["scenario_conditions"] for idx in context_indices],
        dtype=np.float32,
    )
    initial_states = np.asarray(
        [selected[int(idx)]["initial_states"] for idx in context_indices],
        dtype=np.float32,
    )
    normalized_conditions = normalize_numpy(
        conditions,
        adapter.stats,
        "scenario_conditions",
    )
    batch_size = max(int(config["diffusion_batch_size"]), 1)
    inference_steps = config["diffusion_inference_steps"]
    if inference_steps is not None:
        inference_steps = int(inference_steps)

    actions: list[np.ndarray] = []
    guidance_scale = float(config.get("diffusion_guidance_scale", 0.0))
    adapter.model.eval()
    with torch.no_grad():
        for start in range(0, requested, batch_size):
            end = min(start + batch_size, requested)
            cond = torch.from_numpy(normalized_conditions[start:end]).float().to(
                adapter.device
            )
            if guidance_scale > 0.0:
                sample = adapter.model.sample_ddim_with_guidance(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                )
            else:
                sample = adapter.model.sample_ddim(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                )
            decoded = denormalize_torch(sample, adapter.stats, "actions")
            actions.append(decoded.detach().cpu().numpy().astype(np.float32))
    action_array = np.concatenate(actions, axis=0)
    action_cfg = adapter.config.get("action", {})
    projection_cfg = adapter.config.get("trajectory_projection", {})
    trajectories = integrate_cutin_acceleration_actions(
        initial_states,
        action_array,
        float(schema["dt"]),
        ax_min=float(action_cfg.get("ax_min", -8.0)),
        ax_max=float(action_cfg.get("ax_max", 4.0)),
        ay_abs_max=float(action_cfg.get("ay_abs_max", 4.0)),
        speed_min=float(projection_cfg.get("speed_min", 0.0)),
        speed_max=float(projection_cfg.get("speed_max", 50.0)),
    )
    np.savez_compressed(
        output_path,
        context_index=context_indices.astype(np.int64),
        scenario_conditions=conditions.astype(np.float32),
        initial_states=initial_states.astype(np.float32),
        actions=action_array.astype(np.float32),
        target_trajectory=trajectories.astype(np.float32),
        source_type=np.asarray(
            [selected[int(idx)].get("source_type", "") for idx in context_indices],
            dtype=object,
        ),
        base_event_id=np.asarray(
            [selected[int(idx)].get("base_event_id", "") for idx in context_indices],
            dtype=object,
        ),
    )
    summary = {
        "generated_scenarios": str(output_path),
        "num_generated_scenarios": int(requested),
        "diffusion_dataset_dir": str(natural_dir),
        "diffusion_checkpoint_path": str(checkpoint),
        "diffusion_inference_steps": inference_steps,
        "diffusion_batch_size": batch_size,
        "diffusion_seed": int(config["diffusion_seed"]),
        "sampler": "ddim",
    }
    write_json(
        output_path.with_name("diffusion_generated_scenarios_summary.json"),
        summary,
    )
    logger.info(
        "Wrote %d cut-in diffusion generated scenarios to %s",
        requested,
        output_path,
    )
    return summary


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
    elif context_generation_method == CONTEXT_METHOD_GAUSSIAN_COPULA:
        rng = np.random.default_rng(int(config["selection_random_seed"]))
        synthetic_rows = _sample_gaussian_copula_contexts(
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
            "empirical highD independent tail peaks plus samples from a "
            "Gaussian-copula joint distribution over diffusion scenario "
            "condition variables"
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
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in selected],
            dtype=np.float32,
        ),
        "initial_states": np.asarray(
            [row["initial_states"] for row in selected],
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
    diffusion_summary = _generate_diffusion_rollouts(
        selected,
        config=config,
        tail_context_path=tail_context_path,
    )

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
            "tail_feature_names": list(_tail_feature_names(config)),
            "tail_feature_bandwidth": float(config["tail_feature_bandwidth"]),
            "copula_correlation_regularization": float(
                config["copula_correlation_regularization"]
            ),
            "copula_marginal_clip_quantile": float(
                config["copula_marginal_clip_quantile"]
            ),
            "diffusion_generation": diffusion_summary,
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
