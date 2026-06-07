"""Cut-in EVT tail context generation and visualization."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffusion.src.features import CUTIN_SCENARIO_CONDITION_KEYS
from process_highD.src.io_utils import load_config
from utils.evt import load_evt_model
from utils.highd_cutin import load_highd_cutin_event_context_cache
from utils.io import write_json


logger = logging.getLogger(__name__)

SOURCE_EMPIRICAL_TAIL = "highd_evt_independent_tail_peak"
SOURCE_COPULA_CONTEXT = "highd_evt_gaussian_copula_context"
CONDITION_KEYS: tuple[str, ...] = CUTIN_SCENARIO_CONDITION_KEYS
TAIL_FEATURE_NAMES: tuple[str, ...] = (
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
INTRINSIC_TRAJECTORY_METRIC_NAMES: tuple[str, ...] = (
    "lane_entry_time",
    "longitudinal_displacement",
    "total_lateral_displacement",
    "lateral_progress_toward_ego_lane",
    "final_abs_lateral_offset",
    "target_speed_change",
    "max_abs_longitudinal_accel",
    "max_abs_lateral_velocity",
    "mean_abs_lateral_accel",
)
LANE_CHANGE_RATE_NAMES: tuple[str, ...] = (
    "lane_entry_rate",
    "post_entry_retention_rate",
    "completed_lane_change_rate",
)
VARIABLE_EPS = 1.0e-8

CONTEXT_ARRAY_KEYS: tuple[str, ...] = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
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
    "cutin_gap",
    "cutin_ttc",
    "cutin_time_headway",
    "cutin_lateral_time_gap",
    "max_post_cutin_drac",
    "safety_distance",
    "safety_distance_deficit",
    "cutin_duration_seconds",
    "cross_lateral_offset",
    "min_abs_lateral_offset",
    "max_abs_lateral_velocity",
    "max_lateral_approach_speed",
    "final_abs_lateral_offset",
    "cutin_safety_risk_score",
    "is_cutin",
    "is_front_cutin",
    "collision",
    "near_collision",
    "y_cutin",
    "risk_score",
    "evt_tail_probability",
    "peak_id",
    "representative_event_id",
    "base_event_id",
    "base_context_index",
    "synthetic_context",
    "source_type",
)


def _path(config: dict[str, Any], key: str) -> Path:
    return Path(config[key]).resolve()


def _read_evt_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"EVT summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_semantic_contexts(path: Path) -> list[dict[str, Any]]:
    rows = load_highd_cutin_event_context_cache(path)
    if not rows:
        raise RuntimeError(f"Cut-in context cache is empty: {path}")
    invalid = [
        str(row.get("event_id", idx))
        for idx, row in enumerate(rows)
        if float(row.get("is_cutin", 0.0)) < 0.5
    ]
    if invalid:
        raise RuntimeError(
            "Cut-in context cache contains non semantic cut-in rows: "
            f"{invalid[:10]} (total={len(invalid)})"
        )
    return rows


def _score_rows_with_evt(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    model = load_evt_model(model_path)
    summary = _read_evt_summary(summary_path)
    values = np.asarray([row["y_cutin"] for row in rows], dtype=np.float64)
    scores = np.asarray(model.score(values), dtype=np.float64)
    survival = np.asarray(model.survival(values), dtype=np.float64)
    for idx, row in enumerate(rows):
        row["risk_score"] = float(scores[idx])
        row["evt_tail_probability"] = float(survival[idx])
    return {
        "evt_model_path": str(model_path),
        "evt_summary_path": str(summary_path),
        "evt_tail_threshold_u": float(model.u),
        "evt_tail_threshold_score": float(model.score(float(model.u))),
        "evt_exceedance_rate": float(model.exceedance_rate),
        "collision_critical_level": float(summary["collision_critical_level"]),
        "risk_value_key": "y_cutin",
    }


def _select_independent_tail_peaks(
    rows: list[dict[str, Any]],
    peaks_path: Path,
) -> list[dict[str, Any]]:
    if not peaks_path.exists():
        raise FileNotFoundError(
            "Independent cut-in tail peaks not found: "
            f"{peaks_path}. Run process_highD/scripts/estimate_cutin_exposure.py first."
        )
    peaks = pd.read_csv(peaks_path)
    required = {"representative_event_id", "peak_id", "y_cutin_max"}
    missing = sorted(required - set(peaks.columns))
    if missing:
        raise KeyError(f"{peaks_path} is missing required columns: {missing}")

    rows_by_event = {str(row["event_id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    missing_events: list[str] = []
    for _, peak in peaks.iterrows():
        event_id = str(peak["representative_event_id"])
        base = rows_by_event.get(event_id)
        if base is None:
            missing_events.append(event_id)
            continue
        item = dict(base)
        for key, value in peak.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            item[key] = value
        item["source_type"] = SOURCE_EMPIRICAL_TAIL
        item["base_event_id"] = str(item["event_id"])
        item["base_context_index"] = -1
        item["synthetic_context"] = 0
        selected.append(item)
    if missing_events:
        raise KeyError(
            "Independent cut-in peaks reference events missing from context cache: "
            f"{missing_events[:10]} (total={len(missing_events)})"
        )
    if not selected:
        raise RuntimeError(f"No independent cut-in tail peaks matched {peaks_path}")
    return selected


def _tail_feature(row: dict[str, Any]) -> np.ndarray:
    cond = np.asarray(row["scenario_conditions"], dtype=np.float64)
    gap = max(float(cond[1]), 0.2)
    return np.asarray(
        [
            float(cond[0]),
            np.log(gap),
            float(cond[2]),
            float(cond[3]),
            float(cond[4]),
            float(cond[5]),
            float(cond[6]),
            float(cond[7]),
            float(cond[8]),
            float(cond[9]),
        ],
        dtype=np.float64,
    )


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_tail_feature(row) for row in rows], axis=0)


def _nearest_base(features: np.ndarray, target: np.ndarray) -> int:
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    z = (features - center) / scale
    target_z = (target - center) / scale
    return int(np.argmin(np.sum((z - target_z[None, :]) ** 2, axis=1)))


def _reconstruct_cutin_state(
    base_row: dict[str, Any],
    feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(base_row["initial_states"], dtype=np.float32).copy()
    ego_length = float(base_row["ego_length"])
    target_length = float(base_row["adv_length"])
    ego_vx = max(float(feature[0]), 0.0)
    gap = float(np.exp(float(feature[1])))
    lateral_offset = float(feature[2])
    delta_vx = float(feature[3])
    target_vx = max(ego_vx - delta_vx, 0.0)

    states[0, 2] = np.float32(ego_vx)
    states[1, 0] = np.float32(states[0, 0] + 0.5 * (ego_length + target_length) + gap)
    states[1, 1] = np.float32(states[0, 1] + lateral_offset)
    states[1, 2] = np.float32(target_vx)
    states[1, 3] = np.float32(float(feature[4]))
    states[1, 5] = np.float32(np.clip(float(feature[5]), -4.0, 4.0))

    conditions = np.asarray(
        [
            ego_vx,
            gap,
            lateral_offset,
            delta_vx,
            float(feature[4]),
            float(feature[5]),
            float(feature[6]),
            float(feature[7]),
            float(feature[8]),
            float(feature[9]),
        ],
        dtype=np.float32,
    )
    return conditions, states.astype(np.float32)


def _compute_intrinsic_trajectory_metrics(
    target_trajectory: np.ndarray,
    initial_states: np.ndarray,
    *,
    dt: float,
    lane_threshold: float = 1.0,
    post_cutin_window_seconds: float = 3.0,
) -> dict[str, np.ndarray]:
    """Compute lane-change trajectory metrics intrinsic to the target vehicle.

    These metrics characterise the lane-change manoeuvre itself (lateral motion,
    longitudinal speed profile, trajectory smoothness) and do NOT depend on ego
    interaction behaviour.  They are suitable for evaluating open-loop generated
    trajectories against real-world tail events.

    Parameters
    ----------
    target_trajectory: [B, H, 6]  future states of the target vehicle (t=1..H).
    initial_states:    [B, 2, 6]  initial ego + target states at t=0.
    dt:                seconds per step.
    """
    target_vy = target_trajectory[:, :, 3]
    target_vx = target_trajectory[:, :, 2]
    target_ay = target_trajectory[:, :, 5]
    batch = int(target_trajectory.shape[0])
    horizon = int(target_trajectory.shape[1])

    ego_y = initial_states[:, 0, 1]
    initial_lateral = initial_states[:, 1, 1] - ego_y
    relative_lateral = target_trajectory[:, :, 1] - ego_y[:, None]
    abs_lateral = np.abs(relative_lateral)
    initial_abs_lateral = np.abs(initial_lateral)
    lane_entry = abs_lateral <= float(lane_threshold)
    has_lane_entry = np.any(lane_entry, axis=1)
    first_lane_entry = np.argmax(lane_entry, axis=1)
    lane_entry_time = np.full(batch, np.nan, dtype=np.float64)
    lane_entry_time[has_lane_entry] = (
        first_lane_entry[has_lane_entry].astype(np.float64) + 1.0
    ) * float(dt)
    post_steps = max(
        1,
        int(np.ceil(float(post_cutin_window_seconds) / max(float(dt), 1.0e-6))),
    )
    post_entry_retention = np.zeros(batch, dtype=bool)
    for idx in np.flatnonzero(has_lane_entry):
        start = int(first_lane_entry[idx])
        stop = min(horizon, start + post_steps)
        post_entry_retention[idx] = bool(
            np.max(abs_lateral[idx, start:stop]) <= float(lane_threshold)
        )
    abs_with_initial = np.concatenate(
        [initial_abs_lateral[:, None], abs_lateral],
        axis=1,
    )
    approach_speed = np.maximum(
        -np.diff(abs_with_initial, axis=1) / max(float(dt), 1.0e-6),
        0.0,
    )
    has_approach = np.ones(batch, dtype=bool)
    for idx in range(batch):
        if initial_abs_lateral[idx] > float(lane_threshold):
            stop = int(first_lane_entry[idx]) + 1 if has_lane_entry[idx] else horizon
            has_approach[idx] = bool(
                stop > 0 and float(np.max(approach_speed[idx, :stop])) >= 0.05
            )
    completed_lane_change = has_lane_entry & post_entry_retention & has_approach

    return {
        "lane_entry_time": lane_entry_time,
        "lateral_progress_toward_ego_lane": (
            initial_abs_lateral - abs_lateral[:, -1]
        ),
        "longitudinal_displacement": (
            target_trajectory[:, -1, 0] - initial_states[:, 1, 0]
        ),
        "total_lateral_displacement": np.abs(
            target_trajectory[:, -1, 1] - initial_states[:, 1, 1]
        ),
        "minimum_abs_lateral_offset": np.min(abs_lateral, axis=1),
        "final_abs_lateral_offset": abs_lateral[:, -1],
        "max_abs_lateral_velocity": np.max(np.abs(target_vy), axis=1),
        "max_abs_longitudinal_accel": np.max(
            np.abs(target_trajectory[:, :, 4]), axis=1
        ),
        "mean_abs_lateral_accel": np.mean(np.abs(target_ay), axis=1),
        "target_speed_change": target_vx[:, -1] - initial_states[:, 1, 2],
        "lane_entry_rate": has_lane_entry.astype(np.float64),
        "post_entry_retention_rate": post_entry_retention.astype(np.float64),
        "completed_lane_change_rate": completed_lane_change.astype(np.float64),
    }


def _normal_score_pseudo_observations(features: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    ranks = np.empty_like(features, dtype=np.float64)
    n = int(features.shape[0])
    for col in range(int(features.shape[1])):
        order = np.argsort(features[:, col], kind="mergesort")
        ranks[order, col] = np.arange(1, n + 1, dtype=np.float64)
    u = ranks / float(n + 1)
    return ndtri(np.clip(u, 1.0e-6, 1.0 - 1.0e-6))


def _conditions_to_tail_features(conditions: np.ndarray) -> np.ndarray:
    features = np.asarray(conditions, dtype=np.float64).copy()
    if features.ndim != 2 or features.shape[1] != len(TAIL_FEATURE_NAMES):
        raise ValueError(
            "scenario conditions must have shape [N, "
            f"{len(TAIL_FEATURE_NAMES)}], got {tuple(features.shape)}"
        )
    features[:, 1] = np.log(np.maximum(features[:, 1], 0.2))
    return features


def _plot_condition_distribution_comparison(
    empirical_conditions: np.ndarray,
    generated_input_conditions: np.ndarray,
    generated_realized_conditions: np.ndarray,
    *,
    histogram_path: Path,
) -> dict[str, Any]:
    empirical_features = _conditions_to_tail_features(empirical_conditions)
    input_features = _conditions_to_tail_features(generated_input_conditions)
    realized_features = _conditions_to_tail_features(generated_realized_conditions)
    variable = _variable_mask(empirical_features)
    names = tuple(
        name for name, keep in zip(TAIL_FEATURE_NAMES, variable) if bool(keep)
    )
    empirical_variable = empirical_features[:, variable]
    input_variable = input_features[:, variable]
    realized_variable = realized_features[:, variable]
    plt = _matplotlib()
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.15))
    flat_axes = np.asarray(axes).reshape(-1)
    metrics: dict[str, Any] = {}
    for idx, name in enumerate(names):
        ax = flat_axes[idx]
        real = empirical_variable[:, idx]
        sampled = input_variable[:, idx]
        realized = realized_variable[:, idx]
        joined = np.concatenate(
            [
                real[np.isfinite(real)],
                sampled[np.isfinite(sampled)],
                realized[np.isfinite(realized)],
            ]
        )
        metrics[name] = {
            "sampled_input_vs_evt_tail": _distribution_metrics(real, sampled),
            "diffusion_realized_vs_evt_tail": _distribution_metrics(real, realized),
        }
        if joined.size:
            lo, hi = np.percentile(joined, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1.0e-9:
                lo, hi = float(np.min(joined)), float(np.max(joined))
            bins = np.linspace(lo, hi, 34) if hi > lo + 1.0e-9 else 30
            ax.hist(real, bins=bins, density=True, alpha=0.42, label="EVT tail")
            ax.hist(
                sampled,
                bins=bins,
                density=True,
                alpha=0.35,
                label="Copula sampled input",
            )
            ax.hist(
                realized,
                bins=bins,
                density=True,
                alpha=0.35,
                label="Diffusion realized",
            )
            if hi > lo + 1.0e-9:
                ax.set_xlim(lo, hi)
        ax.set_title(name)
        ax.grid(alpha=0.25)
    for ax in flat_axes[len(names) :]:
        ax.axis("off")
    flat_axes[0].legend(fontsize=8)
    fig.tight_layout()
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(histogram_path, dpi=180)
    plt.close(fig)
    return {
        "variable_feature_names": list(names),
        "feature_histograms": metrics,
        "figure_interpretation": (
            "Each panel compares one tail-feature marginal: empirical EVT-tail "
            "events, Gaussian-copula sampled diffusion inputs, and conditions "
            "realized after integrating diffusion-generated trajectories."
        ),
    }


def _variable_mask(features: np.ndarray) -> np.ndarray:
    return np.std(np.asarray(features, dtype=np.float64), axis=0) > VARIABLE_EPS


def _fit_gaussian_copula(
    features: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    if features.shape[1] == 0:
        raise RuntimeError("Gaussian copula has no variable feature dimensions")
    z = _normal_score_pseudo_observations(features)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr += np.eye(corr.shape[0], dtype=np.float64) * max(float(regularization), 0.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1.0e-8, None)
    corr = (eigvecs * eigvals[None, :]) @ eigvecs.T
    diag = np.sqrt(np.clip(np.diag(corr), 1.0e-12, None))
    corr = corr / diag[:, None] / diag[None, :]
    np.fill_diagonal(corr, 1.0)
    return corr


def _sample_condition_distribution(
    tail_rows: list[dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
    clip_quantile: float,
    regularization: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if len(tail_rows) < 2:
        raise RuntimeError("Gaussian copula requires at least two tail contexts")
    features = _feature_matrix(tail_rows)
    variable = _variable_mask(features)
    variable_features = features[:, variable]
    logger.info(
        "Fitting Gaussian copula on %d tail events × %d variable features …",
        len(tail_rows),
        variable_features.shape[1],
    )
    corr_variable = _fit_gaussian_copula(
        variable_features,
        regularization=regularization,
    )
    logger.info("Copula fitted — sampling %d synthetic conditions …", int(count))
    corr = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    corr[np.ix_(variable, variable)] = corr_variable
    np.fill_diagonal(corr, 1.0)

    from scipy.special import ndtr

    q = min(max(float(clip_quantile), 0.0), 0.49)
    lower = np.quantile(variable_features, q, axis=0)
    upper = np.quantile(variable_features, 1.0 - q, axis=0)
    time_to_cross_idx = TAIL_FEATURE_NAMES.index("time_to_cross")
    time_to_cross_support = np.unique(features[:, time_to_cross_idx])
    sampled_z = rng.multivariate_normal(
        np.zeros(variable_features.shape[1], dtype=np.float64),
        corr_variable,
        size=int(count),
        check_valid="ignore",
    )
    sampled_u = np.clip(ndtr(sampled_z), 1.0e-6, 1.0 - 1.0e-6)

    out: list[dict[str, Any]] = []
    center = np.median(features, axis=0)
    for idx in range(int(count)):
        variable_target = np.asarray(
            [
                np.quantile(variable_features[:, col], sampled_u[idx, col])
                for col in range(variable_features.shape[1])
            ],
            dtype=np.float64,
        )
        variable_target = np.clip(variable_target, lower, upper)
        target_feature = center.copy()
        target_feature[variable] = variable_target
        # Enforce cut-in semantic: target must end in ego's lane.
        target_feature[6] = np.clip(target_feature[6], -1.0, 1.0)
        target_feature[time_to_cross_idx] = float(
            time_to_cross_support[
                int(np.argmin(np.abs(time_to_cross_support - target_feature[time_to_cross_idx])))
            ]
        )
        base_idx = _nearest_base(features, target_feature)
        base = tail_rows[base_idx]
        conditions, states = _reconstruct_cutin_state(base, target_feature)
        item = dict(base)
        item["scenario_conditions"] = conditions
        item["initial_states"] = states
        item["source_type"] = SOURCE_COPULA_CONTEXT
        item["event_id"] = f"cutin_copula_{idx:05d}_base_{base['event_id']}"
        item["base_event_id"] = str(base["event_id"])
        item["base_context_index"] = base_idx
        item["synthetic_context"] = 1
        item["initial_gap"] = float(conditions[1])
        item["initial_closing_speed"] = float(conditions[3])
        item["y_cutin"] = np.nan
        item["risk_score"] = np.nan
        item["evt_tail_probability"] = np.nan
        out.append(item)
    return out, corr, variable


def _save_condition_distribution(
    path: Path,
    *,
    empirical_rows: list[dict[str, Any]],
    corr: np.ndarray,
    variable_mask: np.ndarray,
    evt_meta: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = _feature_matrix(empirical_rows)
    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in empirical_rows],
            dtype=np.float32,
        ),
        "tail_features": features.astype(np.float32),
        "condition_keys": np.asarray(CONDITION_KEYS, dtype=object),
        "tail_feature_names": np.asarray(TAIL_FEATURE_NAMES, dtype=object),
        "copula_correlation": corr.astype(np.float32),
        "copula_variable_mask": np.asarray(variable_mask, dtype=bool),
        "copula_marginal_values": features.astype(np.float32),
        "source_event_id": np.asarray(
            [row["event_id"] for row in empirical_rows],
            dtype=object,
        ),
        "event_id": np.asarray(
            [row["event_id"] for row in empirical_rows],
            dtype=object,
        ),
        "recording_id": np.asarray(
            [row["recording_id"] for row in empirical_rows],
            dtype=np.int32,
        ),
        "synthetic_context": np.zeros(len(empirical_rows), dtype=np.int8),
        "source_type": np.asarray(
            [SOURCE_EMPIRICAL_TAIL for _ in empirical_rows],
            dtype=object,
        ),
        "source_peak_id": np.asarray(
            [row["peak_id"] for row in empirical_rows],
            dtype=object,
        ),
        "copula_marginal_clip_quantile": np.asarray(
            float(config["copula_marginal_clip_quantile"]),
            dtype=np.float32,
        ),
        "copula_correlation_regularization": np.asarray(
            float(config["copula_correlation_regularization"]),
            dtype=np.float32,
        ),
        "tail_threshold": np.asarray(evt_meta["evt_tail_threshold_u"], dtype=np.float32),
        "collision_critical_level": np.asarray(
            evt_meta["collision_critical_level"],
            dtype=np.float32,
        ),
    }
    np.savez_compressed(path, **payload)


def _matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    return left, right


def _distribution_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    from scipy.stats import ks_2samp, wasserstein_distance

    a, b = _clean_pair(real, generated)
    if a.size == 0 or b.size == 0:
        return {
            "real_mean": float("nan"),
            "generated_mean": float("nan"),
            "mean_delta": float("nan"),
            "ks_statistic": float("nan"),
            "wasserstein": float("nan"),
        }
    ks = ks_2samp(a, b)
    return {
        "real_mean": float(np.mean(a)),
        "generated_mean": float(np.mean(b)),
        "mean_delta": float(np.mean(b) - np.mean(a)),
        "real_std": float(np.std(a)),
        "generated_std": float(np.std(b)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(wasserstein_distance(a, b)),
    }


def _plot_hist_grid(
    real: np.ndarray,
    generated: np.ndarray,
    names: tuple[str, ...],
    path: Path,
    *,
    real_label: str,
    generated_label: str,
) -> dict[str, dict[str, float]]:
    plt = _matplotlib()
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.1))
    flat_axes = np.asarray(axes).reshape(-1)
    metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        ax = flat_axes[idx]
        a, b = _clean_pair(real[:, idx], generated[:, idx])
        metrics[name] = _distribution_metrics(a, b)
        if a.size and b.size:
            joined = np.concatenate([a, b])
            lo, hi = np.percentile(joined, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1.0e-9:
                lo, hi = float(np.min(joined)), float(np.max(joined))
            bins = np.linspace(lo, hi, 36) if hi > lo + 1.0e-9 else 30
            ax.hist(a, bins=bins, density=True, alpha=0.45, label=real_label)
            ax.hist(b, bins=bins, density=True, alpha=0.45, label=generated_label)
            if hi > lo + 1.0e-9:
                ax.set_xlim(lo, hi)
        ax.set_title(name)
        ax.grid(alpha=0.25)
    for ax in flat_axes[len(names) :]:
        ax.axis("off")
    flat_axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return metrics


def _generate_diffusion_scenarios(
    sampled_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    import torch

    from diffusion.src.kinematics import integrate_cutin_acceleration_actions
    from diffusion.src.utils import set_seed
    from utils.diffusion_adapter import DiffusionPriorAdapter
    from utils.normalization import denormalize_torch, normalize_numpy

    requested = int(config["num_diffusion_scenarios"])
    if requested <= 0:
        raise ValueError("num_diffusion_scenarios must be positive")
    if requested > len(sampled_rows):
        raise ValueError(
            "num_diffusion_scenarios cannot exceed num_condition_samples in the "
            "cut-in tail pipeline"
        )

    output_path = _path(config, "generated_scenarios_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    natural_dir = _path(config, "diffusion_dataset_dir")
    checkpoint = Path(config["diffusion_checkpoint_path"])
    set_seed(int(config["diffusion_seed"]))
    adapter = DiffusionPriorAdapter.load(
        natural_dir,
        checkpoint,
        device=str(config["diffusion_device"]),
    )
    diffusion_cfg = load_config(_path(config, "diffusion_config_path"))
    _validate_diffusion_alignment(
        adapter=adapter,
        diffusion_config=diffusion_cfg,
        inference_steps=int(config["diffusion_inference_steps"]),
    )
    x0_clip_abs = float(diffusion_cfg.get("diffusion", {}).get("x0_clip_abs", 0.0))
    adapter.model.denoiser.cfg.x0_clip_abs = x0_clip_abs
    schema = adapter.schema

    batch_size = max(int(config["diffusion_batch_size"]), 1)
    inference_steps = int(config["diffusion_inference_steps"])
    guidance_scale = float(config.get("diffusion_guidance_scale", 0.0))
    rejection_cfg = dict(config.get("diffusion_rejection", {}))
    rejection_enabled = bool(rejection_cfg.get("enabled", False))
    selected = sampled_rows if rejection_enabled else sampled_rows[:requested]
    if requested > len(selected) and not rejection_enabled:
        raise ValueError(
            "num_diffusion_scenarios cannot exceed num_condition_samples in the "
            "cut-in tail pipeline"
        )

    action_stats = adapter.stats["actions"]
    action_mean = torch.tensor(
        action_stats["mean"],
        dtype=torch.float32,
        device=adapter.device,
    ).view(1, 1, -1)
    action_std = torch.tensor(
        action_stats["std"],
        dtype=torch.float32,
        device=adapter.device,
    ).view(1, 1, -1)
    guidance_config = _cutin_guidance_config(adapter.config, config)

    accepted_conditions: list[np.ndarray] = []
    accepted_initial_states: list[np.ndarray] = []
    accepted_actions: list[np.ndarray] = []
    accepted_trajectories: list[np.ndarray] = []
    accepted_ego_lengths: list[np.ndarray] = []
    accepted_adv_lengths: list[np.ndarray] = []
    accepted_base_event_ids: list[np.ndarray] = []
    accepted_source_types: list[np.ndarray] = []
    generated_total = 0
    semantic_total = 0
    overlap_total = 0
    starts_outside_total = 0
    lateral_progress_total = 0
    post_remain_total = 0
    front_total = 0
    collision_free_total = 0
    condition_error_totals = {
        "final_lateral_offset": 0.0,
        "target_speed_change": 0.0,
        "target_slope_at_cross": 0.0,
    }
    batches_per_pass = int(np.ceil(len(selected) / batch_size))
    max_cycles = 1
    if rejection_enabled:
        multiplier = float(rejection_cfg.get("candidate_multiplier", 1.0))
        max_cycles = max(1, int(np.ceil(multiplier)))
    total_slots = len(selected) * max_cycles
    total_batches = int(np.ceil(total_slots / batch_size))
    log_interval = max(1, total_batches // 20)  # log ~20 times across the run
    logger.info(
        "Generating %d scenarios (batch_size=%d, batches=%d×%d cycles=%d, "
        "guidance=%.3f, rejection=%s) …",
        requested,
        batch_size,
        batches_per_pass,
        max_cycles,
        total_batches,
        guidance_scale,
        "on" if rejection_enabled else "off",
    )
    adapter.model.eval()
    with torch.no_grad():
        for global_start in range(0, total_slots, batch_size):
            if sum(chunk.shape[0] for chunk in accepted_actions) >= requested:
                break
            start = global_start % len(selected)
            end = min(start + batch_size, len(selected))
            batch_rows = selected[start:end]
            conditions = np.asarray(
                [row["scenario_conditions"] for row in batch_rows],
                dtype=np.float32,
            )
            initial_states = np.asarray(
                [row["initial_states"] for row in batch_rows],
                dtype=np.float32,
            )
            normalized_conditions = normalize_numpy(
                conditions,
                adapter.stats,
                "scenario_conditions",
            )
            cond = torch.from_numpy(normalized_conditions).float().to(
                adapter.device
            )
            raw_cond = torch.from_numpy(conditions).float().to(adapter.device)
            init_tensor = torch.from_numpy(initial_states).float().to(adapter.device)
            ego_lengths_batch = np.asarray(
                [row["ego_length"] for row in batch_rows],
                dtype=np.float64,
            )
            adv_lengths_batch = np.asarray(
                [row["adv_length"] for row in batch_rows],
                dtype=np.float64,
            )
            if guidance_scale > 0.0:
                sample = adapter.model.sample_ddim_with_guidance(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_context={
                        "scenario_conditions": raw_cond,
                        "initial_states": init_tensor,
                        "action_mean": action_mean,
                        "action_std": action_std,
                        "ego_length": torch.from_numpy(ego_lengths_batch.astype(np.float32)).to(adapter.device),
                        "adv_length": torch.from_numpy(adv_lengths_batch.astype(np.float32)).to(adapter.device),
                    },
                    guidance_config=guidance_config,
                )
            else:
                sample = adapter.model.sample_ddim(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                )
            decoded = denormalize_torch(sample, adapter.stats, "actions")
            action_batch = decoded.detach().cpu().numpy().astype(np.float32)
            action_cfg = adapter.config["action"]
            projection_cfg = adapter.config.get("trajectory_projection", {})
            trajectories = integrate_cutin_acceleration_actions(
                initial_states,
                action_batch,
                float(schema["dt"]),
                ax_min=float(action_cfg["ax_min"]),
                ax_max=float(action_cfg["ax_max"]),
                ay_abs_max=float(action_cfg["ay_abs_max"]),
                speed_min=float(projection_cfg.get("speed_min", 0.0)),
                speed_max=float(projection_cfg.get("speed_max", 50.0)),
            )
            masks = _semantic_cutin_mask(
                target_trajectory=trajectories,
                initial_states=initial_states,
                ego_length=ego_lengths_batch,
                adv_length=adv_lengths_batch,
                dt=float(schema["dt"]),
                config=config,
            )
            generated_total += int(len(batch_rows))
            semantic_total += int(np.sum(masks["semantic_cutin"]))
            overlap_total += int(np.sum(masks["has_overlap"]))
            starts_outside_total += int(np.sum(masks["starts_outside_ego_lane"]))
            lateral_progress_total += int(np.sum(masks["has_lateral_progress"]))
            post_remain_total += int(np.sum(masks["post_remain"]))
            front_total += int(np.sum(masks["front_at_overlap"]))
            collision_free_total += int(np.sum(masks["collision_free"]))
            realized_batch = _realized_conditions_from_arrays(
                conditions,
                initial_states,
                trajectories,
                dt=float(schema["dt"]),
            )
            for key in condition_error_totals:
                col = CONDITION_KEYS.index(key)
                condition_error_totals[key] += float(
                    np.sum(np.abs(realized_batch[:, col] - conditions[:, col]))
                )
            batch_idx = global_start // batch_size
            if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
                accepted_so_far = sum(chunk.shape[0] for chunk in accepted_actions)
                logger.info(
                    "  batch %4d/%d | generated=%d accepted=%d/%d | "
                    "overlap=%.1f%% post-remain=%.1f%% front=%.1f%% "
                    "collision-free=%.1f%%",
                    batch_idx + 1,
                    total_batches,
                    generated_total,
                    accepted_so_far,
                    requested,
                    100.0 * overlap_total / max(generated_total, 1),
                    100.0 * post_remain_total / max(generated_total, 1),
                    100.0 * front_total / max(generated_total, 1),
                    100.0 * collision_free_total / max(generated_total, 1),
                )
            keep = masks["accepted"] if rejection_enabled else np.ones(len(batch_rows), dtype=bool)
            if rejection_enabled and not np.any(keep):
                continue
            remaining = requested - sum(chunk.shape[0] for chunk in accepted_actions)
            keep_indices = np.flatnonzero(keep)[:remaining]
            accepted_conditions.append(conditions[keep_indices])
            accepted_initial_states.append(initial_states[keep_indices])
            accepted_actions.append(action_batch[keep_indices])
            accepted_trajectories.append(trajectories[keep_indices])
            accepted_ego_lengths.append(ego_lengths_batch[keep_indices].astype(np.float32))
            accepted_adv_lengths.append(adv_lengths_batch[keep_indices].astype(np.float32))
            accepted_base_event_ids.append(
                np.asarray([row["base_event_id"] for row in batch_rows], dtype=object)[keep_indices]
            )
            accepted_source_types.append(
                np.asarray([row["source_type"] for row in batch_rows], dtype=object)[keep_indices]
            )
    if not accepted_actions:
        raise RuntimeError(
            "No diffusion scenarios were accepted. Relax diffusion_rejection "
            "thresholds or increase num_condition_samples/candidate multiplier."
        )
    logger.info(
        "Generation complete — %d total candidates, %d accepted",
        generated_total,
        sum(chunk.shape[0] for chunk in accepted_actions),
    )
    conditions = np.concatenate(accepted_conditions, axis=0)
    initial_states = np.concatenate(accepted_initial_states, axis=0)
    action_array = np.concatenate(accepted_actions, axis=0)
    trajectories = np.concatenate(accepted_trajectories, axis=0)
    ego_lengths = np.concatenate(accepted_ego_lengths, axis=0).astype(np.float64)
    adv_lengths = np.concatenate(accepted_adv_lengths, axis=0).astype(np.float64)
    base_event_id = np.concatenate(accepted_base_event_ids, axis=0)
    source_type = np.concatenate(accepted_source_types, axis=0)
    accepted_count = int(action_array.shape[0])
    if rejection_enabled and accepted_count < requested:
        logger.warning(
            "Diffusion semantic rejection accepted %d/%d requested scenarios from %d candidates",
            accepted_count,
            requested,
            generated_total,
        )
    # Post-generation validity statistics.
    ego_y = initial_states[:, 0, 1].astype(np.float64)
    target_final_y = trajectories[:, -1, 1].astype(np.float64)
    final_lateral_offset = target_final_y - ego_y
    valid_lateral = np.abs(final_lateral_offset) < 1.5
    # Ego constant-speed assumption: ego_x(t) = ego_x0 + ego_vx0 * t
    t_final = float(schema["dt"]) * float(int(trajectories.shape[1]))
    ego_x_final = initial_states[:, 0, 0].astype(np.float64) + initial_states[:, 0, 2].astype(np.float64) * t_final
    gap_final = (
        trajectories[:, -1, 0].astype(np.float64)
        - ego_x_final
        - 0.5 * (ego_lengths + adv_lengths)
    )
    valid_gap = gap_final > 0.0
    valid = valid_lateral & valid_gap
    logger.info(
        "Validity: %.1f%% overall | %.1f%% lateral-complete | %.1f%% collision-free",
        100.0 * float(np.mean(valid)),
        100.0 * float(np.mean(valid_lateral)),
        100.0 * float(np.mean(valid_gap)),
    )
    final_masks = _semantic_cutin_mask(
        target_trajectory=trajectories,
        initial_states=initial_states,
        ego_length=ego_lengths,
        adv_length=adv_lengths,
        dt=float(schema["dt"]),
        config=config,
    )
    realized_conditions = _realized_conditions_from_arrays(
        conditions,
        initial_states,
        trajectories,
        dt=float(schema["dt"]),
    )
    logger.info(
        "Semantic cut-in: %.1f%% accepted output | %.1f%% overlap | %.1f%% post-remain | %.1f%% front-at-overlap",
        100.0 * float(np.mean(final_masks["semantic_cutin"])),
        100.0 * float(np.mean(final_masks["has_overlap"])),
        100.0 * float(np.mean(final_masks["post_remain"])),
        100.0 * float(np.mean(final_masks["front_at_overlap"])),
    )
    np.savez_compressed(
        output_path,
        scenario_conditions=conditions.astype(np.float32),
        initial_states=initial_states.astype(np.float32),
        actions=action_array.astype(np.float32),
        target_trajectory=trajectories.astype(np.float32),
        ego_length=ego_lengths.astype(np.float32),
        adv_length=adv_lengths.astype(np.float32),
        base_event_id=base_event_id,
        source_type=source_type,
        condition_keys=np.asarray(CONDITION_KEYS, dtype=object),
        realized_scenario_conditions=realized_conditions.astype(np.float32),
        semantic_cutin=final_masks["semantic_cutin"].astype(np.int8),
        rejection_accepted=final_masks["accepted"].astype(np.int8),
        has_lane_entry=final_masks["has_overlap"].astype(np.int8),
        has_lateral_approach=final_masks["has_approach"].astype(np.int8),
        starts_outside_ego_lane=final_masks["starts_outside_ego_lane"].astype(np.int8),
        has_lateral_progress=final_masks["has_lateral_progress"].astype(np.int8),
        post_entry_retention=final_masks["post_remain"].astype(np.int8),
        front_at_lane_entry=final_masks["front_at_overlap"].astype(np.int8),
        collision_free=final_masks["collision_free"].astype(np.int8),
    )
    output_condition_adherence_mae = {
        key: float(
            np.mean(
                np.abs(
                    realized_conditions[:, CONDITION_KEYS.index(key)]
                    - conditions[:, CONDITION_KEYS.index(key)]
                )
            )
        )
        for key in condition_error_totals
    }
    summary = {
        "generated_scenarios": str(output_path),
        "num_generated_scenarios": accepted_count,
        "num_requested_scenarios": requested,
        "diffusion_dataset_dir": str(natural_dir),
        "diffusion_checkpoint_path": str(checkpoint),
        "diffusion_inference_steps": inference_steps,
        "trained_diffusion_steps": int(adapter.model.num_steps),
        "diffusion_batch_size": batch_size,
        "diffusion_seed": int(config["diffusion_seed"]),
        "sampler": "ddim",
        "x0_clip_abs": x0_clip_abs,
        "guidance_scale": guidance_scale,
        "rejection_enabled": rejection_enabled,
        "rejection_candidates_evaluated": generated_total,
        "rejection_acceptance_rate": (
            float(accepted_count / generated_total) if generated_total else 0.0
        ),
        "candidate_semantic_cutin_rate": (
            float(semantic_total / generated_total) if generated_total else 0.0
        ),
        "candidate_overlap_rate": (
            float(overlap_total / generated_total) if generated_total else 0.0
        ),
        "candidate_starts_outside_ego_lane_rate": (
            float(starts_outside_total / generated_total) if generated_total else 0.0
        ),
        "candidate_lateral_progress_rate": (
            float(lateral_progress_total / generated_total) if generated_total else 0.0
        ),
        "candidate_post_remain_rate": (
            float(post_remain_total / generated_total) if generated_total else 0.0
        ),
        "candidate_front_at_overlap_rate": (
            float(front_total / generated_total) if generated_total else 0.0
        ),
        "candidate_collision_free_rate": (
            float(collision_free_total / generated_total) if generated_total else 0.0
        ),
        "candidate_marginal_failure_rates": {
            "no_lane_entry": (
                float(1.0 - overlap_total / generated_total) if generated_total else 0.0
            ),
            "starts_inside_ego_lane": (
                float(1.0 - starts_outside_total / generated_total)
                if generated_total else 0.0
            ),
            "insufficient_lateral_progress": (
                float(1.0 - lateral_progress_total / generated_total)
                if generated_total else 0.0
            ),
            "no_post_entry_retention": (
                float(1.0 - post_remain_total / generated_total) if generated_total else 0.0
            ),
            "not_front_at_lane_entry": (
                float(1.0 - front_total / generated_total) if generated_total else 0.0
            ),
            "collision": (
                float(1.0 - collision_free_total / generated_total) if generated_total else 0.0
            ),
        },
        "candidate_condition_adherence_mae": {
            key: float(value / generated_total) if generated_total else float("nan")
            for key, value in condition_error_totals.items()
        },
        "output_semantic_cutin_rate": float(np.mean(final_masks["semantic_cutin"])),
        "output_rejection_accepted_rate": float(np.mean(final_masks["accepted"])),
        "output_condition_adherence_mae": output_condition_adherence_mae,
        "model_condition_inputs": ["scenario_conditions"],
        "initial_states_role": "trajectory integration initial state, not denoiser input",
    }
    write_json(output_path.with_name("diffusion_generated_scenarios_summary.json"), summary)
    return output_path, summary


def _validate_diffusion_alignment(
    *,
    adapter: Any,
    diffusion_config: dict[str, Any],
    inference_steps: int,
) -> None:
    schema = adapter.schema
    checkpoint_config = adapter.config
    errors: list[str] = []
    if str(schema.get("event_type", "")).lower() != "cut_in":
        errors.append(f"schema.event_type={schema.get('event_type')!r}")
    if list(schema.get("condition_keys", [])) != list(CONDITION_KEYS):
        errors.append(f"schema.condition_keys={schema.get('condition_keys')!r}")
    if list(schema.get("action_keys", [])) != ["ax", "ay"]:
        errors.append(f"schema.action_keys={schema.get('action_keys')!r}")
    if str(schema.get("action_representation", "")).lower() != "ax_ay":
        errors.append(
            f"schema.action_representation={schema.get('action_representation')!r}"
        )
    expected_horizon = int(diffusion_config["sequence"]["horizon_steps"])
    if int(schema.get("horizon_steps", -1)) != expected_horizon:
        errors.append(
            f"schema.horizon_steps={schema.get('horizon_steps')!r}, "
            f"config.sequence.horizon_steps={expected_horizon}"
        )
    expected_dt = 1.0 / float(diffusion_config["sampling"]["target_fps"])
    if abs(float(schema.get("dt", -1.0)) - expected_dt) > 1.0e-9:
        errors.append(f"schema.dt={schema.get('dt')!r}, expected_dt={expected_dt}")
    expected_steps = int(diffusion_config["diffusion"]["steps"])
    if int(adapter.model.num_steps) != expected_steps:
        errors.append(
            f"checkpoint diffusion_steps={adapter.model.num_steps}, "
            f"config.diffusion.steps={expected_steps}"
        )
    if int(inference_steps) != int(adapter.model.num_steps):
        errors.append(
            f"diffusion_inference_steps={inference_steps}, "
            f"checkpoint diffusion_steps={adapter.model.num_steps}"
        )
    for section in ("event", "sequence", "action", "model"):
        if checkpoint_config.get(section) != diffusion_config.get(section):
            errors.append(f"checkpoint config section {section!r} differs from natural_cutin.yaml")
    checkpoint_diffusion = dict(checkpoint_config.get("diffusion", {}))
    expected_diffusion = dict(diffusion_config.get("diffusion", {}))
    checkpoint_diffusion.pop("x0_clip_abs", None)
    expected_diffusion.pop("x0_clip_abs", None)
    if checkpoint_diffusion != expected_diffusion:
        errors.append("checkpoint config section 'diffusion' differs from natural_cutin.yaml")
    if errors:
        raise RuntimeError("Cut-in diffusion configuration is not aligned: " + "; ".join(errors))


def _semantic_cutin_mask(
    *,
    target_trajectory: np.ndarray,
    initial_states: np.ndarray,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    dt: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    cfg = dict(config.get("diffusion_rejection", {}))
    overlap_threshold = float(cfg.get("lateral_overlap_threshold", 1.0))
    lane_threshold = float(cfg.get("cutin_lateral_offset", overlap_threshold))
    min_initial_lateral_offset = float(
        cfg.get("min_initial_lateral_offset", lane_threshold)
    )
    min_lateral_progress = float(cfg.get("min_lateral_progress", 0.0))
    min_approach_speed = float(cfg.get("min_lateral_approach_speed", 0.05))
    min_front_gap = float(cfg.get("min_cutin_front_gap", 0.0))
    post_seconds = float(cfg.get("post_cutin_window_seconds", 3.0))
    require_collision_free = bool(cfg.get("require_collision_free", True))

    traj = np.asarray(target_trajectory, dtype=np.float64)
    init = np.asarray(initial_states, dtype=np.float64)
    batch, horizon = int(traj.shape[0]), int(traj.shape[1])
    ego_y = init[:, 0, 1]
    lateral = traj[:, :, 1] - ego_y[:, None]
    abs_lateral = np.abs(lateral)
    initial_abs = np.abs(init[:, 1, 1] - ego_y)
    abs_with_initial = np.concatenate([initial_abs[:, None], abs_lateral], axis=1)
    approach_speed = np.maximum(-np.diff(abs_with_initial, axis=1) / max(float(dt), 1.0e-6), 0.0)
    min_abs_lateral = np.min(abs_lateral, axis=1)
    starts_outside_ego_lane = initial_abs >= min_initial_lateral_offset
    has_lateral_progress = (initial_abs - min_abs_lateral) >= min_lateral_progress

    has_overlap = np.any(abs_lateral <= overlap_threshold, axis=1)
    first_overlap = np.argmax(abs_lateral <= overlap_threshold, axis=1)
    post_steps = max(1, int(np.ceil(post_seconds / max(float(dt), 1.0e-6))))
    has_approach = np.ones(batch, dtype=bool)
    post_remain = np.zeros(batch, dtype=bool)
    for idx in range(batch):
        if initial_abs[idx] > lane_threshold:
            end = int(first_overlap[idx]) + 1 if has_overlap[idx] else horizon
            has_approach[idx] = bool(
                end > 0 and float(np.max(approach_speed[idx, :end])) >= min_approach_speed
            )
        if has_overlap[idx]:
            start = int(first_overlap[idx])
            stop = min(horizon, start + post_steps)
            post_remain[idx] = bool(np.max(abs_lateral[idx, start:stop]) <= lane_threshold)

    t = (np.arange(1, horizon + 1, dtype=np.float64) * float(dt))[None, :]
    ego_x = init[:, 0, 0, None] + init[:, 0, 2, None] * t
    gap = traj[:, :, 0] - ego_x - 0.5 * (
        np.asarray(ego_length, dtype=np.float64)[:, None]
        + np.asarray(adv_length, dtype=np.float64)[:, None]
    )
    front_at_overlap = np.zeros(batch, dtype=bool)
    for idx in range(batch):
        if has_overlap[idx]:
            front_at_overlap[idx] = bool(gap[idx, int(first_overlap[idx])] >= min_front_gap)
    collision_free = np.min(gap, axis=1) > 0.0
    semantic = (
        starts_outside_ego_lane
        & has_lateral_progress
        & has_overlap
        & has_approach
        & post_remain
        & front_at_overlap
    )
    accepted = semantic & collision_free if require_collision_free else semantic
    return {
        "accepted": accepted,
        "semantic_cutin": semantic,
        "has_overlap": has_overlap,
        "has_approach": has_approach,
        "starts_outside_ego_lane": starts_outside_ego_lane,
        "has_lateral_progress": has_lateral_progress,
        "post_remain": post_remain,
        "front_at_overlap": front_at_overlap,
        "collision_free": collision_free,
        "final_lateral": lateral[:, -1],
        "final_gap": gap[:, -1],
    }


def _cutin_guidance_config(adapter_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    action_cfg = adapter_config.get("action", {})
    projection_cfg = adapter_config.get("trajectory_projection", {})
    reject_cfg = config.get("diffusion_rejection", {})
    guidance_cfg = dict(config.get("diffusion_guidance", {}))
    return {
        **reject_cfg,
        **guidance_cfg,
        "ax_min": float(action_cfg.get("ax_min", -8.0)),
        "ax_max": float(action_cfg.get("ax_max", 4.0)),
        "ay_abs_max": float(action_cfg.get("ay_abs_max", 4.0)),
        "lateral_jerk_abs_max": float(action_cfg.get("lateral_jerk_abs_max", 8.0)),
        "speed_min": float(projection_cfg.get("speed_min", 0.0)),
        "speed_max": float(projection_cfg.get("speed_max", 50.0)),
    }


def _generated_semantic_metrics(generated_path: Path, dt: float) -> dict[str, np.ndarray]:
    """Compute intrinsic lane-change trajectory metrics for generated scenarios.

    Only metrics that describe the target vehicle's own manoeuvre are included.
    Interaction-dependent quantities (gap, TTC) are intentionally excluded
    because ego behaviour in open-loop generation is a constant-speed placeholder
    and does not represent any real AV stack.
    """
    data = np.load(generated_path, allow_pickle=True)
    return _compute_intrinsic_trajectory_metrics(
        target_trajectory=data["target_trajectory"].astype(np.float64),
        initial_states=data["initial_states"].astype(np.float64),
        dt=float(dt),
    )


def _realized_conditions_from_arrays(
    input_conditions: np.ndarray,
    initial_states: np.ndarray,
    target_trajectory: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Replace outcome-dependent input conditions with realized trajectory values."""
    conditions = np.asarray(input_conditions, dtype=np.float64).copy()
    initial = np.asarray(initial_states, dtype=np.float64)
    trajectory = np.asarray(target_trajectory, dtype=np.float64)
    conditions[:, CONDITION_KEYS.index("final_lateral_offset")] = (
        trajectory[:, -1, 1] - initial[:, 0, 1]
    )
    conditions[:, CONDITION_KEYS.index("target_speed_change")] = (
        trajectory[:, -1, 2] - initial[:, 1, 2]
    )
    cross_idx = np.clip(
        np.rint(
            conditions[:, CONDITION_KEYS.index("time_to_cross")]
            / max(float(dt), 1.0e-6)
        ).astype(np.int64)
        - 1,
        0,
        trajectory.shape[1] - 1,
    )
    cross_vx = trajectory[np.arange(trajectory.shape[0]), cross_idx, 2]
    cross_vy = trajectory[np.arange(trajectory.shape[0]), cross_idx, 3]
    conditions[:, CONDITION_KEYS.index("target_slope_at_cross")] = (
        cross_vy / (np.abs(cross_vx) + 1.0e-3)
    )
    return conditions


def _generated_realized_conditions(
    generated: Any,
    *,
    dt: float,
) -> np.ndarray:
    if "realized_scenario_conditions" in generated.files:
        return generated["realized_scenario_conditions"].astype(np.float64)
    return _realized_conditions_from_arrays(
        generated["scenario_conditions"],
        generated["initial_states"],
        generated["target_trajectory"],
        dt=dt,
    )


def _real_semantic_matrix(
    rows: list[dict[str, Any]],
    *,
    dataset_dir: Path,
    dt: float,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    """Intrinsic lane-change trajectory metrics from real tail event states.

    Loads the diffusion dataset to access the target vehicle's recorded future
    trajectory and computes metrics that characterise the lane-change manoeuvre
    independently of ego behaviour.
    """
    dataset_path = Path(dataset_dir) / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cut-in diffusion dataset not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    idx_arr = _dataset_indices_for_tail_rows(data, rows)
    initial = data["initial_states"][idx_arr].astype(np.float64)
    future = data["future_states"][idx_arr].astype(np.float64)
    metrics = _compute_intrinsic_trajectory_metrics(
        target_trajectory=future[:, :, 1, :],
        initial_states=initial,
        dt=float(dt),
    )
    values = np.stack(
        [metrics[name] for name in INTRINSIC_TRAJECTORY_METRIC_NAMES], axis=1
    )
    return INTRINSIC_TRAJECTORY_METRIC_NAMES, values, metrics


def _dataset_indices_for_tail_rows(
    data: Any,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    by_event_id: dict[str, list[int]] = {}
    for idx, event_id in enumerate(data["event_id"].tolist()):
        by_event_id.setdefault(str(event_id), []).append(int(idx))
    dataset_anchor = (
        data["anchor_frame"].astype(np.int64)
        if "anchor_frame" in data.files
        else None
    )
    indices: list[int] = []
    for row in rows:
        event_id = str(row["event_id"])
        candidates = by_event_id.get(event_id)
        if not candidates:
            raise KeyError(f"Tail event {event_id} not found in diffusion dataset")
        if dataset_anchor is not None and "anchor_frame" in row:
            anchor = int(row["anchor_frame"])
            chosen = min(
                candidates,
                key=lambda idx: abs(int(dataset_anchor[idx]) - anchor),
            )
        else:
            chosen = candidates[0]
        indices.append(int(chosen))
    return np.asarray(indices, dtype=np.int64)


def _real_tail_trajectory_arrays(
    rows: list[dict[str, Any]],
    *,
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(dataset_dir) / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cut-in diffusion dataset not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    idx_arr = _dataset_indices_for_tail_rows(data, rows)
    initial = data["initial_states"][idx_arr].astype(np.float64)
    target = data["future_states"][idx_arr, :, 1, :].astype(np.float64)
    return initial, target


def _generated_semantic_matrix(
    metrics: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Extract intrinsic metric matrix from generated-scenario metric dict."""
    values = np.stack(
        [metrics[name] for name in INTRINSIC_TRAJECTORY_METRIC_NAMES], axis=1
    )
    return INTRINSIC_TRAJECTORY_METRIC_NAMES, values


def _plot_lateral_trajectory_comparison(
    *,
    empirical_rows: list[dict[str, Any]],
    generated: Any,
    dataset_dir: Path,
    path: Path,
    dt: float,
    max_traces: int = 120,
) -> dict[str, Any]:
    real_initial, real_target = _real_tail_trajectory_arrays(
        empirical_rows,
        dataset_dir=dataset_dir,
    )
    gen_initial = generated["initial_states"].astype(np.float64)
    gen_target = generated["target_trajectory"].astype(np.float64)
    real_direction = np.sign(real_initial[:, 0, 1] - real_initial[:, 1, 1])
    gen_direction = np.sign(gen_initial[:, 0, 1] - gen_initial[:, 1, 1])
    real_direction = np.where(real_direction == 0.0, 1.0, real_direction)
    gen_direction = np.where(gen_direction == 0.0, 1.0, gen_direction)
    real_y = (real_target[:, :, 1] - real_initial[:, 1, 1:2]) * real_direction[:, None]
    gen_y = (gen_target[:, :, 1] - gen_initial[:, 1, 1:2]) * gen_direction[:, None]
    horizon = min(real_y.shape[1], gen_y.shape[1])
    real_y = real_y[:, :horizon]
    gen_y = gen_y[:, :horizon]
    t = np.arange(horizon, dtype=np.float64) * float(dt)

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    rng = np.random.default_rng(42)
    for values, color, label in (
        (real_y, "tab:blue", "EVT tail"),
        (gen_y, "tab:orange", "Diffusion generated"),
    ):
        sample_count = min(max_traces, values.shape[0])
        if sample_count > 0:
            sample_idx = rng.choice(values.shape[0], size=sample_count, replace=False)
            ax.plot(
                t,
                values[sample_idx].T,
                color=color,
                alpha=0.045,
                linewidth=0.7,
            )
        median = np.nanmedian(values, axis=0)
        lower = np.nanquantile(values, 0.25, axis=0)
        upper = np.nanquantile(values, 0.75, axis=0)
        ax.fill_between(t, lower, upper, color=color, alpha=0.16, linewidth=0.0)
        ax.plot(t, median, color=color, linewidth=2.2, label=f"{label} median")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("time from anchor (s)")
    ax.set_ylabel("lateral progress toward ego lane (m)")
    ax.set_title("Cut-in lateral trajectory family, direction aligned")
    ax.grid(alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "real_num_trajectories": int(real_y.shape[0]),
        "generated_num_trajectories": int(gen_y.shape[0]),
        "horizon_steps": int(horizon),
        "real_final_lateral_displacement_mean": float(np.nanmean(real_y[:, -1])),
        "generated_final_lateral_displacement_mean": float(np.nanmean(gen_y[:, -1])),
    }


def _write_visualizations(
    *,
    empirical_rows: list[dict[str, Any]],
    generated_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    dt: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_paths = {
        "cutin_manoeuvre_tail_vs_generated": output_dir / "cutin_manoeuvre_tail_vs_generated.png",
        "cutin_lateral_trajectory_tail_vs_generated": output_dir / "cutin_lateral_trajectory_tail_vs_generated.png",
        "scenario_condition_tail_vs_copula_sampled": output_dir / "scenario_condition_tail_vs_copula_sampled.png",
    }
    for obsolete_name in (
        "scenario_condition_copula_correlation.png",
        "scenario_condition_copula_joint_density.png",
    ):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    generated_metrics = _generated_semantic_metrics(generated_path, dt)
    semantic_names, real_semantic, real_metrics = _real_semantic_matrix(
        empirical_rows, dataset_dir=dataset_dir, dt=dt
    )
    _, generated_semantic = _generated_semantic_matrix(generated_metrics)
    semantic_metrics = _plot_hist_grid(
        real_semantic,
        generated_semantic,
        semantic_names,
        condition_paths["cutin_manoeuvre_tail_vs_generated"],
        real_label="EVT tail",
        generated_label="Diffusion generated",
    )
    lane_change_rates = {
        name: {
            "real_rate": float(np.mean(real_metrics[name])),
            "generated_rate": float(np.mean(generated_metrics[name])),
            "rate_delta": float(
                np.mean(generated_metrics[name]) - np.mean(real_metrics[name])
            ),
        }
        for name in LANE_CHANGE_RATE_NAMES
    }
    generated = np.load(generated_path, allow_pickle=True)
    trajectory_family_metrics = _plot_lateral_trajectory_comparison(
        empirical_rows=empirical_rows,
        generated=generated,
        dataset_dir=dataset_dir,
        path=condition_paths["cutin_lateral_trajectory_tail_vs_generated"],
        dt=dt,
    )
    condition_metrics = _plot_condition_distribution_comparison(
        np.asarray(
            [row["scenario_conditions"] for row in empirical_rows],
            dtype=np.float64,
        ),
        generated["scenario_conditions"].astype(np.float64),
        _generated_realized_conditions(generated, dt=dt),
        histogram_path=condition_paths["scenario_condition_tail_vs_copula_sampled"],
    )
    report = {
        "figures": {key: str(value) for key, value in condition_paths.items()},
        "cutin_manoeuvre_metrics": semantic_metrics,
        "cutin_lateral_trajectory_family": trajectory_family_metrics,
        "lane_change_event_rates": lane_change_rates,
        "scenario_condition_distribution": condition_metrics,
        "note": (
            "The main similarity assessment is cut-in specific: lane entry, "
            "post-entry retention, lateral progress/displacement, final lane "
            "position, entry timing, lateral motion, and target speed change. "
            "The scenario-condition histogram compares empirical EVT-tail "
            "conditions, Gaussian-copula sampled diffusion inputs, and conditions "
            "realized by diffusion trajectories. "
            "Interaction-dependent gap/TTC metrics are excluded because ego uses "
            "a constant-speed open-loop placeholder."
        ),
    }
    write_json(output_dir / "distribution_similarity_summary.json", report)
    return report


def run_cutin_tail_generation(config: dict[str, Any]) -> None:
    required = {
        "event_context_cache_path",
        "condition_distribution_path",
        "independent_tail_peaks_path",
        "evt_model_path",
        "evt_summary_path",
        "num_condition_samples",
        "num_diffusion_scenarios",
        "diffusion_dataset_dir",
        "diffusion_checkpoint_path",
        "diffusion_config_path",
        "generated_scenarios_path",
        "diffusion_batch_size",
        "diffusion_inference_steps",
        "diffusion_device",
        "diffusion_seed",
        "selection_random_seed",
        "copula_marginal_clip_quantile",
        "copula_correlation_regularization",
    }
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise KeyError(f"Cut-in tail generation config missing keys: {missing}")

    rows = _load_semantic_contexts(_path(config, "event_context_cache_path"))
    evt_meta = _score_rows_with_evt(
        rows,
        model_path=_path(config, "evt_model_path"),
        summary_path=_path(config, "evt_summary_path"),
    )
    empirical_tail = _select_independent_tail_peaks(
        rows,
        _path(config, "independent_tail_peaks_path"),
    )
    rng = np.random.default_rng(int(config["selection_random_seed"]))
    condition_sample_count = int(config["num_condition_samples"])
    rejection_cfg = dict(config.get("diffusion_rejection", {}))
    if bool(rejection_cfg.get("enabled", False)):
        multiplier = max(float(rejection_cfg.get("candidate_multiplier", 1.0)), 1.0)
        condition_sample_count = max(
            condition_sample_count,
            int(np.ceil(int(config["num_diffusion_scenarios"]) * multiplier)),
        )
    sampled_conditions, corr, variable_mask = _sample_condition_distribution(
        empirical_tail,
        count=condition_sample_count,
        rng=rng,
        clip_quantile=float(config["copula_marginal_clip_quantile"]),
        regularization=float(config["copula_correlation_regularization"]),
    )
    condition_distribution_path = _path(config, "condition_distribution_path")
    _save_condition_distribution(
        condition_distribution_path,
        empirical_rows=empirical_tail,
        corr=corr,
        variable_mask=variable_mask,
        evt_meta=evt_meta,
        config=config,
    )
    generated_path, diffusion_summary = _generate_diffusion_scenarios(
        sampled_conditions,
        config=config,
    )
    visual_summary = _write_visualizations(
        empirical_rows=empirical_tail,
        generated_path=generated_path,
        dataset_dir=_path(config, "diffusion_dataset_dir"),
        output_dir=generated_path.parent / "figures",
        dt=0.04,
    )
    summary = {
        **evt_meta,
        "condition_distribution": str(condition_distribution_path),
        "num_evt_tail_conditions": int(len(empirical_tail)),
        "num_condition_samples": int(len(sampled_conditions)),
        "num_diffusion_scenarios": int(diffusion_summary["num_generated_scenarios"]),
        "num_requested_diffusion_scenarios": int(config["num_diffusion_scenarios"]),
        "condition_keys": list(CONDITION_KEYS),
        "tail_feature_names": list(TAIL_FEATURE_NAMES),
        "dynamic_tail_feature_names": [
            name for name, keep in zip(TAIL_FEATURE_NAMES, variable_mask) if bool(keep)
        ],
        "constant_tail_feature_values": {
            name: float(_feature_matrix(empirical_tail)[0, idx])
            for idx, (name, keep) in enumerate(zip(TAIL_FEATURE_NAMES, variable_mask))
            if not bool(keep)
        },
        "condition_distribution_model": (
            "Gaussian copula fitted on EVT declustered independent cut-in tail "
            "scenario_conditions"
        ),
        "model_condition_inputs": ["scenario_conditions"],
        "initial_states_role": "trajectory integration initial state, not denoiser input",
        "diffusion_generation": diffusion_summary,
        "visualization": visual_summary,
        "selection_random_seed": int(config["selection_random_seed"]),
        "copula_marginal_clip_quantile": float(config["copula_marginal_clip_quantile"]),
        "copula_correlation_regularization": float(
            config["copula_correlation_regularization"]
        ),
        "diffusion_rejection": dict(config.get("diffusion_rejection", {})),
        "diffusion_guidance": dict(config.get("diffusion_guidance", {})),
    }
    write_json(
        condition_distribution_path.with_name("scenario_condition_distribution_summary.json"),
        summary,
    )
    logger.info(
        "Wrote scenario-condition distribution from %d EVT cut-in tail events, "
        "%d sampled conditions, and %d diffusion scenarios",
        len(empirical_tail),
        len(sampled_conditions),
        int(diffusion_summary["num_generated_scenarios"]),
    )
