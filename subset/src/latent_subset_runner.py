#!/usr/bin/env python3
"""Shared subset simulation implementation."""
from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_json, save_json
from utils.context import context_from_npz, load_context_npz
from utils.evt import load_evt_model
from utils.io import resolve_path, write_csv
from subset.src.closed_loop_runner import (
    ClosedLoopCutInRunner,
    ClosedLoopFollowingRunner,
)
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from subset.src.latent_evaluator import LatentMpcEpisodeEvaluator
from subset.src.subset_simulation import (
    SubsetLevel,
    run_subset_simulation,
)


logger = logging.getLogger(__name__)
SOURCE_INDEPENDENT_TAIL_PEAK = "highd_independent_tail_peak"
SOURCE_TAIL_FEATURE_KDE_KNN = "highd_tail_feature_kde_knn"
SOURCE_TAIL_GAUSSIAN_COPULA = "highd_tail_gaussian_copula"
TAIL_DISTRIBUTION_SOURCE_TYPES = {
    SOURCE_INDEPENDENT_TAIL_PEAK,
    SOURCE_TAIL_FEATURE_KDE_KNN,
    SOURCE_TAIL_GAUSSIAN_COPULA,
}
_WORKER_EVALUATOR: LatentMpcEpisodeEvaluator | None = None


def _worker_init(torch_num_threads: int) -> None:
    if torch_num_threads <= 0:
        return
    try:
        import torch

        torch.set_num_threads(int(torch_num_threads))
    except Exception as exc:  # 防御性工作进程初始化设置
        logger.warning("Could not set worker torch threads: %s", exc)


def _worker_evaluate_task(
    task: tuple[int, int, np.ndarray],
) -> tuple[int, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("Multiprocessing worker evaluator is not initialized")
    sample_idx, context_index, latent = task
    return int(sample_idx), _WORKER_EVALUATOR.evaluate(
        int(context_index),
        np.asarray(latent, dtype=np.float32),
    )


class _MultiprocessPopulationEvaluator:
    """Evaluate independent level populations in forked CPU workers."""

    def __init__(
        self,
        evaluator: LatentMpcEpisodeEvaluator,
        *,
        num_workers: int,
        chunksize: int,
        worker_torch_num_threads: int,
    ) -> None:
        self.evaluator = evaluator
        self.num_workers = int(num_workers)
        self.chunksize = max(1, int(chunksize))
        self.worker_torch_num_threads = int(worker_torch_num_threads)
        self.pool: mp.pool.Pool | None = None

    def __enter__(self) -> "_MultiprocessPopulationEvaluator":
        global _WORKER_EVALUATOR
        _WORKER_EVALUATOR = self.evaluator
        context = mp.get_context("fork")
        self.pool = context.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(self.worker_torch_num_threads,),
        )
        logger.info(
            "Enabled multiprocessing population evaluation workers=%d chunksize=%d",
            self.num_workers,
            self.chunksize,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        global _WORKER_EVALUATOR
        if self.pool is not None:
            if exc_type is None:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
            self.pool = None
        _WORKER_EVALUATOR = None

    def evaluate_many(
        self,
        context_indices: np.ndarray,
        latents: np.ndarray,
        level: int,
    ) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list]:
        if self.pool is None:
            raise RuntimeError("Multiprocessing population evaluator is not active")
        total = int(latents.shape[0])
        scores = np.zeros(total, dtype=np.float64)
        actions: list[np.ndarray | None] = [None] * total
        metrics: list[dict[str, float] | None] = [None] * total
        traces: list[list[dict[str, float]] | None] = [None] * total
        tasks = (
            (idx, int(context_indices[idx]), latents[idx].copy())
            for idx in range(total)
        )
        interval = max(1, total // 10)
        done = 0
        for sample_idx, result in self.pool.imap_unordered(
            _worker_evaluate_task,
            tasks,
            chunksize=self.chunksize,
        ):
            scores[sample_idx] = float(result.score)
            actions[sample_idx] = result.actions.astype(np.float32, copy=True)
            item_metrics = dict(result.metrics)
            item_metrics["context_index"] = float(context_indices[sample_idx])
            metrics[sample_idx] = item_metrics
            traces[sample_idx] = list(result.trace)
            done += 1
            if done == total or done % interval == 0:
                logger.info(
                    "Subset level %d evaluated %d/%d samples",
                    level,
                    done,
                    total,
                )

        return (
            scores,
            [item for item in actions if item is not None],
            [item for item in metrics if item is not None],
            [item for item in traces if item is not None],
        )


def _paths(config: dict[str, Any], base: Path) -> dict[str, Path]:
    paths = config.get("paths", {})
    required = ("tail_context_path", "evt_model_path")
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    output_value = config.get("subset_simulation", {}).get("output_dir")
    if not output_value:
        raise KeyError("Config subset_simulation.output_dir is required")
    resolved = {
        "tail_contexts": resolve_path(paths["tail_context_path"], base),
        "evt_model": resolve_path(paths["evt_model_path"], base),
        "output_dir": resolve_path(str(output_value), base),
    }
    if "exposure_summary_path" in paths:
        resolved["exposure_summary"] = resolve_path(
            paths["exposure_summary_path"],
            base,
        )
    return resolved


def _load_contexts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Subset tail contexts not found: {path}")
    raw = load_context_npz(path)
    count = int(raw["scenario_conditions"].shape[0])
    return [context_from_npz(raw, idx) for idx in range(count)]


def _validate_context_schema(
    contexts: list[dict[str, Any]],
    sampler: FrozenDiffusionSampler,
    context_path: Path,
) -> None:
    if not contexts:
        raise ValueError(f"Subset tail context file is empty: {context_path}")
    states = np.asarray(contexts[0]["initial_states"], dtype=np.float32)
    conditions = np.asarray(contexts[0]["scenario_conditions"], dtype=np.float32)
    if states.shape != (2, 6):
        raise ValueError(
            "Subset initial_states must have shape [num_actors, state_features], "
            f"got {tuple(states.shape)} in {context_path}"
        )
    cfg = sampler.prior.model.denoiser.cfg
    expected_dim = int(cfg.scenario_condition_dim)
    if conditions.ndim != 1 or int(conditions.shape[0]) != expected_dim:
        raise ValueError(
            "Subset tail context scenario condition schema does not match the "
            f"diffusion checkpoint: got {tuple(conditions.shape)}, expected "
            f"({expected_dim},). Context file: {context_path}. Rebuild the "
            "highD event cache and tail contexts with the current settings: "
            "python process_highD/scripts/extract_highd_events.py && "
            "python process_highD/scripts/select_following_tail_contexts.py or "
            "python process_highD/scripts/select_cutin_tail_contexts.py"
        )
    logger.info(
        "Loaded %d subset contexts from %s with scenario_condition_dim=%d",
        len(contexts),
        context_path,
        expected_dim,
    )


def _multiprocess_population_evaluator(
    evaluator: LatentMpcEpisodeEvaluator,
    sampler: FrozenDiffusionSampler,
    config: dict[str, Any],
) -> _MultiprocessPopulationEvaluator | None:
    parallel_cfg = config.get("parallel", {})
    subset_cfg = config.get("subset_simulation", {})
    num_workers = int(
        parallel_cfg.get(
            "population_num_workers",
            subset_cfg.get("population_num_workers", 1),
        )
    )
    if num_workers <= 1:
        return None
    device_type = str(getattr(sampler.prior.device, "type", sampler.prior.device))
    if device_type == "cuda":
        logger.warning(
            "Disabling multiprocessing population evaluation on CUDA device; "
            "forked CUDA workers are unsafe. Use one worker or set training.device "
            "to cpu for CPU multiprocessing."
        )
        return None
    return _MultiprocessPopulationEvaluator(
        evaluator,
        num_workers=num_workers,
        chunksize=int(parallel_cfg.get("population_chunksize", 1)),
        worker_torch_num_threads=int(
            parallel_cfg.get("worker_torch_num_threads", 1)
        ),
    )


def _evt_failure_threshold(
    path: Path,
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"EVT model is required before subset simulation: {path}")
    evt_cfg = config.setdefault("evt", {})
    evt_cfg["model_path"] = str(path)
    evt_cfg["score_space"] = str(evt_cfg.get("score_space", "evt"))
    target_mode = str(evt_cfg.get("target_mode", "return_period"))
    return_period = int(evt_cfg.get("return_period", 100))
    model = load_evt_model(path)
    if target_mode == "collision_critical_level":
        z_target = float(evt_cfg.get("collision_critical_level", 5.0))
    elif target_mode == "return_period":
        z_target = float(model.return_level(return_period))
    else:
        raise ValueError(f"Unsupported evt.target_mode: {target_mode}")
    failure_threshold = float(model.score(z_target))
    return failure_threshold, {
        "evt_target_mode": target_mode,
        "evt_return_period": float(return_period),
        "evt_return_level_target": z_target,
        "evt_failure_threshold": failure_threshold,
        "evt_model_u": float(model.u),
        "evt_model_xi": float(model.xi),
        "evt_model_beta": float(model.beta),
        "evt_model_exceedance_rate": float(model.exceedance_rate),
    }


def _metric_array(
    levels: list[SubsetLevel],
    key: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for level in levels:
        values = [float(item.get(key, np.nan)) for item in level.metrics]
        rows.append(np.asarray(values, dtype=np.float32))
    return np.stack(rows, axis=0)


def _actions_array(levels: list[SubsetLevel]) -> tuple[np.ndarray, np.ndarray]:
    max_steps = max(
        int(action.shape[0]) for level in levels for action in level.actions
    )
    max_dim = max(int(action.shape[1]) for level in levels for action in level.actions)
    shape = (len(levels), len(levels[0].actions), max_steps, max_dim)
    actions = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape[:3], dtype=np.float32)
    for level_idx, level in enumerate(levels):
        for sample_idx, action in enumerate(level.actions):
            steps = int(action.shape[0])
            dim = int(action.shape[1])
            actions[level_idx, sample_idx, :steps, :dim] = action
            mask[level_idx, sample_idx, :steps] = 1.0
    return actions, mask


def _save_samples(result, output_dir: Path) -> None:
    levels = result.levels
    actions, action_mask = _actions_array(levels)
    np.savez_compressed(
        output_dir / "latent_subset_samples.npz",
        context_indices=np.stack(
            [level.context_indices for level in levels],
            axis=0,
        ),
        latents=np.stack([level.latents for level in levels], axis=0),
        scores=np.stack([level.scores for level in levels], axis=0),
        thresholds=np.asarray(
            [level.threshold for level in levels],
            dtype=np.float32,
        ),
        acceptance_rate=np.asarray(
            [level.acceptance_rate for level in levels],
            dtype=np.float32,
        ),
        accepted_mask=np.stack([level.accepted for level in levels], axis=0),
        actions=actions,
        action_mask=action_mask,
        collision=_metric_array(levels, "collision"),
        min_gap=_metric_array(levels, "min_gap"),
        min_ttc=_metric_array(levels, "min_ttc"),
        physical_feasible=_metric_array(levels, "physical_feasible"),
        y_long=_metric_array(levels, "y_long"),
        y_cutin=_metric_array(levels, "y_cutin"),
        cutin_safety_risk_score=_metric_array(levels, "cutin_safety_risk_score"),
        cutin_time_headway=_metric_array(levels, "cutin_time_headway"),
        cutin_lateral_time_gap=_metric_array(levels, "cutin_lateral_time_gap"),
        safety_distance_deficit=_metric_array(levels, "safety_distance_deficit"),
        max_post_cutin_drac=_metric_array(levels, "max_post_cutin_drac"),
        min_abs_lateral_offset=_metric_array(levels, "min_abs_lateral_offset"),
        final_abs_lateral_offset=_metric_array(levels, "final_abs_lateral_offset"),
        max_lateral_approach_speed=_metric_array(levels, "max_lateral_approach_speed"),
        lateral_overlap_fraction=_metric_array(levels, "lateral_overlap_fraction"),
        is_cutin=_metric_array(levels, "is_cutin"),
        is_front_cutin=_metric_array(levels, "is_front_cutin"),
        evt_tail_probability=_metric_array(levels, "evt_tail_probability"),
    )


def _top_cases(
    result,
    contexts: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    metric_keys = (
        "collision",
        "near_collision",
        "min_gap",
        "min_ttc",
        "risk_score",
        "y_long",
        "y_cutin",
        "evt_tail_probability",
        "physical_feasible",
        "cutin_safety_risk_score",
        "cutin_gap",
        "cutin_ttc",
        "cutin_time_headway",
        "cutin_lateral_time_gap",
        "safety_distance",
        "safety_distance_deficit",
        "max_post_cutin_drac",
        "min_abs_lateral_offset",
        "final_abs_lateral_offset",
        "max_lateral_approach_speed",
        "lateral_overlap_fraction",
        "is_cutin",
        "is_front_cutin",
    )
    for level in result.levels:
        for idx, score in enumerate(level.scores):
            context_index = int(level.context_indices[idx])
            context = contexts[context_index]
            cases.append(
                {
                    "level": int(level.level),
                    "sample_index": int(idx),
                    "context_index": context_index,
                    "recording_id": context.get("recording_id"),
                    "event_id": context.get("event_id"),
                    "score": float(score),
                    "metrics": {
                        key: float(level.metrics[idx][key])
                        for key in metric_keys
                        if key in level.metrics[idx]
                    },
                }
            )
    cases.sort(key=lambda item: float(item["score"]), reverse=True)
    return cases[:top_k]


def _probability_uncertainty(
    result,
    *,
    num_samples: int,
    p0: float,
) -> dict[str, float]:
    level_power = max(len(result.levels) - 1, 0)
    scale = float(p0) ** level_power
    q = float(result.final_failure_fraction)
    n = max(int(num_samples), 1)
    conditional_se = float(np.sqrt(max(q * (1.0 - q), 0.0) / n))
    se = scale * conditional_se
    probability = float(result.probability)
    lower = max(0.0, probability - 1.96 * se)
    upper = min(1.0, probability + 1.96 * se)
    rel = float(se / probability) if probability > 0.0 else float("inf")
    return {
        "probability_standard_error": float(se),
        "probability_ci95_lower": float(lower),
        "probability_ci95_upper": float(upper),
        "conditional_final_fraction_standard_error": conditional_se,
        "relative_standard_error": rel,
        "uncertainty_method": (
            "binomial final-level approximation; ignores MCMC correlation"
        ),
    }


def _uniqueness_stats(
    context_indices: np.ndarray,
    latents: np.ndarray,
) -> dict[str, float]:
    num_samples = max(int(latents.shape[0]), 1)
    context_counter: dict[int, int] = {}
    state_counter: dict[tuple[int, bytes], int] = {}
    for idx in range(int(latents.shape[0])):
        context = int(context_indices[idx])
        state = (context, np.ascontiguousarray(latents[idx]).tobytes())
        context_counter[context] = context_counter.get(context, 0) + 1
        state_counter[state] = state_counter.get(state, 0) + 1
    context_counts = np.asarray(list(context_counter.values()), dtype=np.int64)
    state_counts = np.asarray(list(state_counter.values()), dtype=np.int64)
    return {
        "unique_contexts": float(len(context_counter)),
        "largest_context_count": float(np.max(context_counts)),
        "largest_context_share": float(np.max(context_counts) / num_samples),
        "unique_states": float(len(state_counter)),
        "largest_state_count": float(np.max(state_counts)),
        "largest_state_share": float(np.max(state_counts) / num_samples),
    }


def _level_stats(result, failure_threshold: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for level in result.levels:
        scores = np.asarray(level.scores, dtype=np.float64)
        uniqueness = _uniqueness_stats(level.context_indices, level.latents)
        rows.append(
            {
                "level": float(level.level),
                "num_samples": float(len(scores)),
                "score_min": float(np.min(scores)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "score_p50": float(np.quantile(scores, 0.50)),
                "score_p90": float(np.quantile(scores, 0.90)),
                "score_p95": float(np.quantile(scores, 0.95)),
                "score_max": float(np.max(scores)),
                "subset_threshold": float(level.threshold),
                "failure_fraction": float(np.mean(scores >= float(failure_threshold))),
                "acceptance_rate": float(level.acceptance_rate),
                **uniqueness,
            }
        )
    return rows


def _reliability_thresholds(
    config: dict[str, Any],
    *,
    num_contexts: int,
    num_samples: int,
) -> dict[str, float]:
    subset_cfg = config.get("subset_simulation", {})
    min_context_absolute = int(subset_cfg.get("reliability_min_unique_contexts", 10))
    min_unique_contexts = min(int(num_contexts), min_context_absolute)
    min_state_fraction = float(
        subset_cfg.get("reliability_min_unique_state_fraction", 0.50)
    )
    min_unique_states = max(1, int(np.ceil(min_state_fraction * num_samples)))
    return {
        "min_unique_contexts": float(min_unique_contexts),
        "min_unique_states": float(min_unique_states),
        "max_largest_context_share": float(
            subset_cfg.get("reliability_max_largest_context_share", 0.30)
        ),
        "max_largest_state_share": float(
            subset_cfg.get("reliability_max_largest_state_share", 0.10)
        ),
        "min_acceptance_rate": float(
            subset_cfg.get("reliability_min_acceptance_rate", 0.10)
        ),
    }


def _reliability_assessment(
    level_stats: list[dict[str, float]],
    config: dict[str, Any],
    *,
    num_contexts: int,
    num_samples: int,
) -> dict[str, Any]:
    thresholds = _reliability_thresholds(
        config,
        num_contexts=num_contexts,
        num_samples=num_samples,
    )
    if not level_stats:
        return {
            "status": "fail",
            "reason": ["no subset levels were produced"],
            "thresholds": thresholds,
        }
    final = dict(level_stats[-1])
    failures: list[str] = []
    warnings: list[str] = []

    if final["unique_contexts"] < thresholds["min_unique_contexts"]:
        failures.append(
            "unique_contexts "
            f"{final['unique_contexts']:.0f} < {thresholds['min_unique_contexts']:.0f}"
        )
    if final["unique_states"] < thresholds["min_unique_states"]:
        failures.append(
            "unique_states "
            f"{final['unique_states']:.0f} < {thresholds['min_unique_states']:.0f}"
        )
    if final["largest_context_share"] > thresholds["max_largest_context_share"]:
        failures.append(
            "largest_context_share "
            f"{final['largest_context_share']:.3f} > "
            f"{thresholds['max_largest_context_share']:.3f}"
        )
    if final["largest_state_share"] > thresholds["max_largest_state_share"]:
        failures.append(
            "largest_state_share "
            f"{final['largest_state_share']:.3f} > "
            f"{thresholds['max_largest_state_share']:.3f}"
        )

    acceptance_level: dict[str, float] | None = None
    for row in reversed(level_stats):
        candidate = float(row.get("acceptance_rate", np.nan))
        if np.isfinite(candidate) and row.get("level", 0.0) > 0.0:
            acceptance_level = row
            break
    acceptance = (
        float(acceptance_level["acceptance_rate"])
        if acceptance_level is not None
        else float("nan")
    )
    if np.isfinite(acceptance):
        if acceptance < thresholds["min_acceptance_rate"]:
            failures.append(
                "acceptance_rate "
                f"{acceptance:.3f} < {thresholds['min_acceptance_rate']:.3f}"
            )
    elif final.get("level", 0.0) > 0.0:
        warnings.append("no transition acceptance_rate is available")

    status = "fail" if failures else ("warning" if warnings else "pass")
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "thresholds": thresholds,
        "assessed_level": int(final.get("level", -1)),
        "observed": {
            "unique_contexts": final.get("unique_contexts"),
            "unique_states": final.get("unique_states"),
            "largest_context_share": final.get("largest_context_share"),
            "largest_state_share": final.get("largest_state_share"),
            "acceptance_rate": acceptance,
            "acceptance_rate_level": (
                acceptance_level.get("level") if acceptance_level is not None else None
            ),
            "final_level_acceptance_rate": final.get("acceptance_rate"),
        },
    }


def _context_tail_thresholds(contexts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for context in contexts:
        for key in ("tail_threshold", "evt_tail_threshold_u"):
            if key not in context:
                continue
            try:
                value = float(context[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
            break
    return values


def _context_collision_levels(contexts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for context in contexts:
        if "collision_critical_level" not in context:
            continue
        try:
            value = float(context["collision_critical_level"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _mileage_return_period(
    result,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
    evt_target: dict[str, float],
    reliability: dict[str, Any],
    probability_estimate_kind: str,
    exposure_summary_path: Path | None,
) -> dict[str, Any]:
    cfg = config.get("mileage_return_period", {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    strictness_failures: list[str] = []
    exposure: dict[str, Any] = {}
    if exposure_summary_path is None:
        strictness_failures.append("paths.exposure_summary_path is not configured")
    elif not exposure_summary_path.exists():
        strictness_failures.append(f"exposure_summary_missing={exposure_summary_path}")
    else:
        exposure = load_json(exposure_summary_path)

    event_type = str(config.get("event", {}).get("event_type", "following"))
    is_cutin = event_type == "cut_in"
    expected_denominator = "all_vehicle_miles" if is_cutin else "following_ego_miles"
    primary_label = "cut-in all-vehicle" if is_cutin else "following ego"
    risk_label = "Y_cutin_sim" if is_cutin else "Y_long_sim"
    total_miles = float(
        exposure.get(
            "all_vehicle_miles" if is_cutin else "following_ego_miles",
            0.0,
        )
    )
    total_hours = float(
        exposure.get(
            "all_vehicle_hours" if is_cutin else "following_ego_hours",
            0.0,
        )
    )
    all_vehicle_miles = float(exposure.get("all_vehicle_miles", 0.0))
    all_vehicle_hours = float(exposure.get("all_vehicle_hours", 0.0))
    if total_miles <= 0.0:
        strictness_failures.append("total_exposure_miles <= 0")
    if not is_cutin and all_vehicle_miles <= 0.0:
        strictness_failures.append("total_all_vehicle_miles <= 0")
    if str(exposure.get("exposure_denominator", "")) != expected_denominator:
        strictness_failures.append(
            f"exposure_denominator!={expected_denominator}"
        )

    probability = float(result.probability)
    if not np.isfinite(probability) or probability < 0.0:
        strictness_failures.append("subset_probability is not finite and non-negative")

    tail_rate_per_mile = float(exposure.get("tail_peak_rate_per_mile", 0.0))
    tail_rate_per_hour = float(exposure.get("tail_peak_rate_per_hour", 0.0))
    tail_rate_per_all_vehicle_mile = float(
        exposure.get(
            "tail_peak_rate_per_all_vehicle_mile",
            tail_rate_per_mile if is_cutin else 0.0,
        )
    )
    tail_rate_per_all_vehicle_hour = float(
        exposure.get(
            "tail_peak_rate_per_all_vehicle_hour",
            tail_rate_per_hour if is_cutin else 0.0,
        )
    )

    def _periods(rate_per_mile: float, rate_per_hour: float) -> dict[str, float]:
        intensity_per_mile = float(max(rate_per_mile, 0.0) * probability)
        intensity_per_hour = float(max(rate_per_hour, 0.0) * probability)
        return {
            "intensity_per_mile": intensity_per_mile,
            "return_period_miles": (
                float(1.0 / intensity_per_mile)
                if intensity_per_mile > 0.0
                else float("inf")
            ),
            "intensity_per_km": float(intensity_per_mile / 1.609344),
            "return_period_km": (
                float(1.609344 / intensity_per_mile)
                if intensity_per_mile > 0.0
                else float("inf")
            ),
            "intensity_per_hour": intensity_per_hour,
            "return_period_hours": (
                float(1.0 / intensity_per_hour)
                if intensity_per_hour > 0.0
                else float("inf")
            ),
        }

    primary_periods = _periods(tail_rate_per_mile, tail_rate_per_hour)
    all_vehicle_periods = _periods(
        tail_rate_per_all_vehicle_mile,
        tail_rate_per_all_vehicle_hour,
    )

    def _ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0 or not np.isfinite(denominator):
            return float("nan")
        return float(numerator / denominator)

    target_mode = str(evt_target.get("evt_target_mode", "return_period"))
    human_reference: dict[str, Any] | None = None
    if target_mode == "collision_critical_level":
        highd_intensity_mile = float(
            exposure.get(
                "highd_safety_critical_intensity_per_mile",
                exposure.get("highd_collision_intensity_per_mile", np.nan),
            )
        )
        highd_intensity_km = float(
            exposure.get(
                "highd_safety_critical_intensity_per_km",
                exposure.get(
                    "highd_collision_intensity_per_km",
                    highd_intensity_mile / 1.609344,
                ),
            )
        )
        highd_intensity_hour = float(
            exposure.get(
                "highd_safety_critical_intensity_per_hour",
                exposure.get("highd_collision_intensity_per_hour", np.nan),
            )
        )
        highd_return_miles = float(
            exposure.get(
                "highd_safety_critical_return_period_miles",
                exposure.get("highd_collision_return_period_miles", np.nan),
            )
        )
        highd_return_km = float(
            exposure.get(
                "highd_safety_critical_return_period_km",
                exposure.get(
                    "highd_collision_return_period_km",
                    highd_return_miles * 1.609344,
                ),
            )
        )
        highd_return_hours = float(
            exposure.get(
                "highd_safety_critical_return_period_hours",
                exposure.get("highd_collision_return_period_hours", np.nan),
            )
        )
        human_reference = {
            "interpretation": (
                f"Human highD {event_type} reference at the same EVT "
                f"safety-critical level and {expected_denominator} denominator."
            ),
            "highd_safety_critical_intensity_per_mile": highd_intensity_mile,
            "highd_safety_critical_intensity_per_km": highd_intensity_km,
            "highd_safety_critical_intensity_per_hour": highd_intensity_hour,
            "highd_safety_critical_return_period_miles": highd_return_miles,
            "highd_safety_critical_return_period_km": highd_return_km,
            "highd_safety_critical_return_period_hours": highd_return_hours,
            "highd_collision_intensity_per_mile": highd_intensity_mile,
            "highd_collision_intensity_per_km": highd_intensity_km,
            "highd_collision_intensity_per_hour": highd_intensity_hour,
            "highd_collision_return_period_miles": highd_return_miles,
            "highd_collision_return_period_km": highd_return_km,
            "highd_collision_return_period_hours": highd_return_hours,
            "ads_to_highd_intensity_ratio_per_mile": _ratio(
                primary_periods["intensity_per_mile"],
                highd_intensity_mile,
            ),
            "ads_to_highd_intensity_ratio_per_hour": _ratio(
                primary_periods["intensity_per_hour"],
                highd_intensity_hour,
            ),
            "ads_return_period_over_highd_return_period_miles": _ratio(
                primary_periods["return_period_miles"],
                highd_return_miles,
            ),
            "ads_return_period_over_highd_return_period_hours": _ratio(
                primary_periods["return_period_hours"],
                highd_return_hours,
            ),
        }

    if bool(cfg.get("require_tail_threshold_match", True)):
        exposure_u = exposure.get("evt_tail_threshold_u")
        if exposure_u is None:
            strictness_failures.append("exposure evt_tail_threshold_u is missing")
        else:
            exposure_u = float(exposure_u)
            evt_model_u = float(evt_target.get("evt_model_u", np.nan))
            tol = float(cfg.get("tail_threshold_abs_tol", 1.0e-6))
            if not np.isfinite(evt_model_u) or abs(exposure_u - evt_model_u) > tol:
                strictness_failures.append(
                    "exposure evt_tail_threshold_u does not match subset EVT model u"
                )
            context_thresholds = _context_tail_thresholds(contexts)
            if not context_thresholds:
                strictness_failures.append("tail context threshold metadata is missing")
            elif max(abs(value - exposure_u) for value in context_thresholds) > tol:
                strictness_failures.append(
                    "tail context threshold does not match exposure evt_tail_threshold_u"
                )
    collision_level = evt_target.get("evt_return_level_target")
    if target_mode == "collision_critical_level":
        exposure_collision = exposure.get("collision_critical_level")
        tol = float(cfg.get("tail_threshold_abs_tol", 1.0e-6))
        if exposure_collision is None:
            strictness_failures.append("exposure collision_critical_level is missing")
        elif abs(float(exposure_collision) - float(collision_level)) > tol:
            strictness_failures.append(
                "exposure collision_critical_level does not match subset target"
            )
        context_collision = _context_collision_levels(contexts)
        if not context_collision:
            strictness_failures.append(
                "tail context collision critical metadata is missing"
            )
        elif (
            max(abs(value - float(collision_level)) for value in context_collision)
            > tol
        ):
            strictness_failures.append(
                "tail context collision critical level does not match subset target"
            )

    if bool(cfg.get("require_independent_peak_contexts", True)):
        source_types = {str(context.get("source_type", "")) for context in contexts}
        if (
            not source_types
            or not source_types.issubset(TAIL_DISTRIBUTION_SOURCE_TYPES)
        ):
            strictness_failures.append(
                "tail_context_source!="
                "independent_tail_peak_or_tail_feature_distribution "
                f"({sorted(source_types)})"
            )

    if bool(cfg.get("require_subset_reliability_pass", True)):
        if reliability.get("status") != "pass":
            strictness_failures.append(
                f"subset_reliability_status={reliability.get('status')}"
            )
    if probability_estimate_kind != "standard_subset_estimate":
        strictness_failures.append(
            f"probability_estimate_kind={probability_estimate_kind}"
        )

    return {
        "enabled": True,
        "event_type": event_type,
        "risk_label": risk_label,
        "primary_exposure_label": primary_label,
        "exposure_denominator": expected_denominator,
        "exposure_summary_path": (
            str(exposure_summary_path) if exposure_summary_path is not None else None
        ),
        "ads_exceedance_probability_conditional": probability,
        "tail_peak_rate_per_mile": tail_rate_per_mile,
        "ads_extreme_risk_intensity_per_mile": primary_periods[
            "intensity_per_mile"
        ],
        "ads_safety_critical_intensity_per_mile": primary_periods[
            "intensity_per_mile"
        ],
        "ads_return_period_miles": primary_periods["return_period_miles"],
        "ads_safety_critical_return_period_miles": primary_periods[
            "return_period_miles"
        ],
        "ads_extreme_risk_intensity_per_km": primary_periods[
            "intensity_per_km"
        ],
        "ads_safety_critical_intensity_per_km": primary_periods[
            "intensity_per_km"
        ],
        "ads_return_period_km": primary_periods["return_period_km"],
        "ads_safety_critical_return_period_km": primary_periods[
            "return_period_km"
        ],
        "tail_peak_rate_per_hour": tail_rate_per_hour,
        "ads_extreme_risk_intensity_per_hour": primary_periods[
            "intensity_per_hour"
        ],
        "ads_safety_critical_intensity_per_hour": primary_periods[
            "intensity_per_hour"
        ],
        "ads_return_period_hours": primary_periods["return_period_hours"],
        "ads_safety_critical_return_period_hours": primary_periods[
            "return_period_hours"
        ],
        "all_highd_vehicle_background": {
            "total_all_vehicle_miles": all_vehicle_miles,
            "total_all_vehicle_km": float(
                exposure.get("all_vehicle_km", all_vehicle_miles * 1.609344)
            ),
            "total_all_vehicle_hours": all_vehicle_hours,
            "following_ego_mile_fraction_of_all_vehicle_miles": float(
                exposure.get("ego_mile_fraction_of_all_vehicle", 0.0)
            ),
            "tail_peak_rate_per_all_vehicle_mile": tail_rate_per_all_vehicle_mile,
            "tail_peak_rate_per_all_vehicle_km": float(
                exposure.get(
                    "tail_peak_rate_per_all_vehicle_km",
                    tail_rate_per_all_vehicle_mile / 1.609344,
                )
            ),
            "tail_peak_rate_per_all_vehicle_hour": (
                tail_rate_per_all_vehicle_hour
            ),
            "ads_extreme_risk_intensity_per_all_vehicle_mile": (
                all_vehicle_periods["intensity_per_mile"]
            ),
            "ads_safety_critical_intensity_per_all_vehicle_mile": (
                all_vehicle_periods["intensity_per_mile"]
            ),
            "ads_return_period_all_vehicle_miles": all_vehicle_periods[
                "return_period_miles"
            ],
            "ads_safety_critical_return_period_all_vehicle_miles": (
                all_vehicle_periods["return_period_miles"]
            ),
            "ads_extreme_risk_intensity_per_all_vehicle_km": (
                all_vehicle_periods["intensity_per_km"]
            ),
            "ads_safety_critical_intensity_per_all_vehicle_km": (
                all_vehicle_periods["intensity_per_km"]
            ),
            "ads_return_period_all_vehicle_km": all_vehicle_periods[
                "return_period_km"
            ],
            "ads_safety_critical_return_period_all_vehicle_km": (
                all_vehicle_periods["return_period_km"]
            ),
            "ads_extreme_risk_intensity_per_all_vehicle_hour": (
                all_vehicle_periods["intensity_per_hour"]
            ),
            "ads_safety_critical_intensity_per_all_vehicle_hour": (
                all_vehicle_periods["intensity_per_hour"]
            ),
            "ads_return_period_all_vehicle_hours": all_vehicle_periods[
                "return_period_hours"
            ],
            "ads_safety_critical_return_period_all_vehicle_hours": (
                all_vehicle_periods["return_period_hours"]
            ),
        },
        "human_highd_reference": human_reference,
        "human_highd_following_reference": (
            human_reference if not is_cutin else None
        ),
        "human_highd_cutin_reference": human_reference if is_cutin else None,
        "evt_target_mode": target_mode,
        "collision_critical_level": (
            float(collision_level)
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_intensity_per_mile_at_x_c": (
            primary_periods["intensity_per_mile"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_return_period_miles_at_x_c": (
            primary_periods["return_period_miles"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_intensity_per_km_at_x_c": (
            primary_periods["intensity_per_km"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_return_period_km_at_x_c": (
            primary_periods["return_period_km"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_intensity_per_hour_at_x_c": (
            primary_periods["intensity_per_hour"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_safety_critical_return_period_hours_at_x_c": (
            primary_periods["return_period_hours"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_collision_intensity_per_mile": (
            primary_periods["intensity_per_mile"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_collision_return_period_miles": (
            primary_periods["return_period_miles"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_collision_intensity_per_km": (
            primary_periods["intensity_per_km"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_collision_return_period_km": (
            primary_periods["return_period_km"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_collision_intensity_per_hour": (
            primary_periods["intensity_per_hour"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "ads_collision_return_period_hours": (
            primary_periods["return_period_hours"]
            if target_mode == "collision_critical_level"
            else None
        ),
        "primary_exposure_miles": total_miles,
        "primary_exposure_hours": total_hours,
        "following_ego_miles": float(exposure.get("following_ego_miles", 0.0)),
        "following_ego_hours": float(exposure.get("following_ego_hours", 0.0)),
        "all_vehicle_miles": all_vehicle_miles,
        "all_vehicle_hours": all_vehicle_hours,
        "num_independent_tail_peaks": int(
            exposure.get("num_independent_tail_peaks", 0)
        ),
        "strict_mileage_interpretation": not strictness_failures,
        "strictness_failures": strictness_failures,
    }


def _write_level_stats(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    write_csv(path, rows)


def _write_diagnostic_plots(
    result,
    output_dir: Path,
    failure_threshold: float,
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Could not write subset diagnostic plots: %s", exc)
        return {}

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for level in result.levels:
        ax.hist(
            level.scores,
            bins=24,
            alpha=0.45,
            label=f"level {level.level}",
        )
    ax.axvline(
        float(failure_threshold),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="failure threshold",
    )
    ax.set_xlabel("risk score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "subset_score_histograms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["score_histograms"] = str(path)

    return paths


def _log_mileage_return_period(
    summary: dict[str, Any],
    result: Any,
    logger: logging.Logger,
) -> None:
    """将里程回报周期指标打印到控制台。"""
    mileage = summary.get("mileage_return_period", {})
    if not mileage.get("enabled"):
        return

    probability = float(result.probability)
    target_mode = str(mileage.get("evt_target_mode", "return_period"))
    collision_level = mileage.get("collision_critical_level")
    strict = bool(mileage.get("strict_mileage_interpretation", False))
    strict_note = "" if strict else " [非严格解释]"
    risk_label = str(mileage.get("risk_label", "Y_long_sim"))
    primary_label = str(mileage.get("primary_exposure_label", "following ego"))
    event_type = str(mileage.get("event_type", "following"))

    logger.info("=" * 72)
    logger.info("📊 里程回报周期 (Mileage Return Period) 分析%s", strict_note)
    logger.info("=" * 72)
    logger.info(
        "  子集概率 P(%s > threshold | tail peak): %.6g",
        risk_label,
        probability,
    )
    logger.info(
        "  尾部峰值率:        %.4f /mile | %.4f /km | %.2f /hour",
        mileage["tail_peak_rate_per_mile"],
        mileage.get(
            "tail_peak_rate_per_km",
            mileage["tail_peak_rate_per_mile"] / 1.609344,
        ),
        mileage.get("tail_peak_rate_per_hour", float("nan")),
    )
    logger.info("  ── %s 里程 ──", primary_label)
    logger.info(
        "  safety-critical 强度: %.4g /mile | %.4g /km | %.4g /hour",
        mileage["ads_extreme_risk_intensity_per_mile"],
        mileage["ads_extreme_risk_intensity_per_km"],
        mileage["ads_extreme_risk_intensity_per_hour"],
    )
    logger.info(
        "  回报周期:          %.1f miles | %.1f km | %.1f hours",
        mileage["ads_return_period_miles"],
        mileage["ads_return_period_km"],
        mileage["ads_return_period_hours"],
    )

    all_bg = mileage.get("all_highd_vehicle_background", {})
    primary_uses_all_vehicle = event_type == "cut_in" and (
        str(mileage.get("exposure_denominator", "")) == "all_vehicle_miles"
    )
    if all_bg and not primary_uses_all_vehicle:
        logger.info("  ── all highD vehicle 背景 ──")
        logger.info(
            "  following ego 占全车比例: %.3f",
            float(
                all_bg.get(
                    "following_ego_mile_fraction_of_all_vehicle_miles",
                    0.0,
                )
            )
        )
        logger.info(
            "  尾部峰值率(全车):  %.4f /mile | %.4f /km | %.2f /hour",
            all_bg["tail_peak_rate_per_all_vehicle_mile"],
            all_bg["tail_peak_rate_per_all_vehicle_km"],
            all_bg["tail_peak_rate_per_all_vehicle_hour"],
        )
        logger.info(
            "  safety-critical 强度(全车): %.4g /mile | %.4g /km | %.4g /hour",
            all_bg["ads_extreme_risk_intensity_per_all_vehicle_mile"],
            all_bg["ads_extreme_risk_intensity_per_all_vehicle_km"],
            all_bg["ads_extreme_risk_intensity_per_all_vehicle_hour"],
        )
        logger.info(
            "  回报周期(全车):    %.1f miles | %.1f km | %.1f hours",
            all_bg["ads_return_period_all_vehicle_miles"],
            all_bg["ads_return_period_all_vehicle_km"],
            all_bg["ads_return_period_all_vehicle_hours"],
        )

    if target_mode == "collision_critical_level" and collision_level is not None:
        logger.info("  ── safety-critical 阈值等效 ──")
        logger.info(
            "  目标阈值:          collision_critical_level = %.6g；"
            "上方 ADS 强度/回报周期即该阈值结果",
            float(collision_level),
        )
        human = (
            mileage.get("human_highd_cutin_reference")
            or mileage.get("human_highd_following_reference")
            or mileage.get("human_highd_reference")
            or {}
        )
        if human:
            logger.info("  ── highD 人类驾驶基线对比 ──")
            logger.info(
                "  highD safety-critical 强度: %.4g /mile | %.4g /km | %.4g /hour",
                float(
                    human.get(
                        "highd_safety_critical_intensity_per_mile",
                        float("nan"),
                    )
                ),
                float(
                    human.get("highd_safety_critical_intensity_per_km", float("nan"))
                ),
                float(
                    human.get(
                        "highd_safety_critical_intensity_per_hour",
                        float("nan"),
                    )
                ),
            )
            logger.info(
                "  highD safety-critical 回报周期: %.1f miles | %.1f km | %.1f hours",
                float(
                    human.get(
                        "highd_safety_critical_return_period_miles",
                        float("nan"),
                    )
                ),
                float(
                    human.get(
                        "highd_safety_critical_return_period_km",
                        float("nan"),
                    )
                ),
                float(
                    human.get(
                        "highd_safety_critical_return_period_hours",
                        float("nan"),
                    )
                ),
            )
            logger.info(
                "  ADS/highD强度比:   %.3f /mile | %.3f /hour",
                float(
                    human.get("ads_to_highd_intensity_ratio_per_mile", float("nan"))
                ),
                float(
                    human.get("ads_to_highd_intensity_ratio_per_hour", float("nan"))
                ),
            )

    failures = mileage.get("strictness_failures", [])
    if failures:
        logger.warning(
            "  严格性检查未通过 (%d): %s",
            len(failures),
            "; ".join(str(f) for f in failures),
        )
    logger.info("=" * 72)


def _summary(
    result,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
    failure_threshold: float,
    evt_target: dict[str, float],
    level_stats: list[dict[str, float]],
    figures: dict[str, str],
    exposure_summary_path: Path | None,
) -> dict[str, Any]:
    subset_cfg = config.get("subset_simulation", {})
    uncertainty = _probability_uncertainty(
        result,
        num_samples=subset_cfg.get("num_samples", 100),
        p0=subset_cfg.get("p0", 0.1),
    )
    reliability = _reliability_assessment(
        level_stats,
        config,
        num_contexts=len(contexts),
        num_samples=subset_cfg.get("num_samples", 100),
    )
    source_types = {str(context.get("source_type", "")) for context in contexts}
    target_mode = str(evt_target.get("evt_target_mode", "return_period"))
    event_type = str(config.get("event", {}).get("event_type", "following"))
    risk_label = "Y_cutin_sim" if event_type == "cut_in" else "Y_long_sim"
    if source_types == {SOURCE_INDEPENDENT_TAIL_PEAK}:
        if target_mode == "collision_critical_level":
            probability_target = (
                f"P_context,z({risk_label} > x_c | o in highD independent tail peaks)"
            )
        else:
            probability_target = (
                f"P_context,z({risk_label} > z_m | o in highD independent tail peaks)"
            )
    elif source_types and source_types.issubset(TAIL_DISTRIBUTION_SOURCE_TYPES):
        if target_mode == "collision_critical_level":
            probability_target = (
                f"P_context,z({risk_label} > x_c | "
                "o sampled from highD tail scenario-condition distribution)"
            )
        else:
            probability_target = (
                f"P_context,z({risk_label} > z_m | "
                "o sampled from highD tail scenario-condition distribution)"
            )
    elif source_types == {"highd_event_tail"}:
        probability_target = (
            f"P_context,z({risk_label} > z_m | o in highD tail contexts)"
        )
    else:
        probability_target = f"P_context,z({risk_label} > z_m | configured contexts)"
    strict_probability = reliability.get("status") == "pass"
    if strict_probability:
        probability_estimate_kind = "standard_subset_estimate"
    else:
        probability_estimate_kind = "low_reliability_standard_estimate"
    mileage_return_period = _mileage_return_period(
        result,
        contexts,
        config,
        evt_target,
        reliability,
        probability_estimate_kind,
        exposure_summary_path,
    )
    return_period = int(evt_target.get("evt_return_period", 100))
    if target_mode == "collision_critical_level":
        failure_event = (
            f"{risk_label} > x_c "
            f"({float(evt_target['evt_return_level_target']):.6g})"
        )
    else:
        failure_event = f"{risk_label} > z{return_period}"
    execution_mode = str(
        config.get("event", {}).get("execution_mode", "rolling_mpc")
    )
    summary = {
        "probability": float(result.probability),
        **uncertainty,
        "reliability": reliability,
        "probability_target": probability_target,
        "probability_estimate_kind": probability_estimate_kind,
        "strict_probability_interpretation": strict_probability,
        "mileage_return_period": mileage_return_period,
        "failure_event": failure_event,
        "score_space": config.get("evt", {}).get("score_space", "evt"),
        **evt_target,
        "failure_threshold": failure_threshold,
        "final_failure_fraction": result.final_failure_fraction,
        "thresholds": [level.threshold for level in result.levels],
        "acceptance_rates": [level.acceptance_rate for level in result.levels],
        "level_stats": level_stats,
        "figures": figures,
        "num_levels": len(result.levels),
        "num_contexts": len(contexts),
        "num_samples": subset_cfg.get("num_samples", 100),
        "p0": subset_cfg.get("p0", 0.1),
        "proposal_std": subset_cfg.get("proposal_std", 0.35),
        "context_refresh_prob": subset_cfg.get("context_refresh_prob", 0.1),
        "mh_retries_per_sample": subset_cfg.get("mh_retries_per_sample", 4),
        "max_levels": int(subset_cfg.get("max_levels", 8)),
        "episode_steps": int(
            config.get(
                "_effective_episode_steps",
                config.get("env", {}).get("episode_steps", 200),
            )
        ),
        "execution_mode": execution_mode,
    }
    summary["commit_steps_max"] = int(
        config.get("env", {}).get("commit_steps_max", 10)
    )
    return summary


def run_subset_from_config(
    config: dict[str, Any],
    config_dir: Path,
    *,
    expected_event_type: str | None = None,
) -> Path:
    base = Path(config_dir)
    paths = _paths(config, base)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = _load_contexts(paths["tail_contexts"])
    failure_threshold, evt_target = _evt_failure_threshold(paths["evt_model"], config)
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=base)
    _validate_context_schema(contexts, sampler, paths["tail_contexts"])
    event_type = str(
        sampler.prior.schema.get(
            "event_type",
            config.get("event", {}).get("event_type", "following"),
        )
    )
    if expected_event_type is not None and event_type != expected_event_type:
        raise ValueError(
            f"Expected {expected_event_type} diffusion prior/config, got {event_type}"
        )
    if event_type == "cut_in":
        execution_mode = str(
            config.get("event", {}).get("execution_mode", "rolling_control")
        )
        if execution_mode != "rolling_control":
            raise ValueError(
                "Cut-in subset simulation requires event.execution_mode to be "
                "'rolling_control'"
            )
        runner = ClosedLoopCutInRunner(sampler, config)
    else:
        runner = ClosedLoopFollowingRunner(sampler, config)
    evaluator = LatentMpcEpisodeEvaluator(
        sampler,
        runner,
        contexts,
        config,
        inference_steps=int(
            config.get("sampling", {}).get("eval_diffusion_steps", 100)
        ),
    )
    config["_effective_episode_steps"] = int(evaluator.episode_steps)
    subset_cfg = config.get("subset_simulation", {})
    target_label = (
        "x_c"
        if str(evt_target.get("evt_target_mode")) == "collision_critical_level"
        else "z_m"
    )
    logger.info(
        (
            "Running mixed-context subset simulation contexts=%d "
            "samples=%d p0=%.3f max_levels=%d threshold=%.6f "
            "%s=%.6f latent_shape=%s proposal_std=%.3f "
            "context_refresh_prob=%.3f mh_retries=%d"
        ),
        len(contexts),
        subset_cfg.get("num_samples", 100),
        subset_cfg.get("p0", 0.1),
        subset_cfg.get("max_levels", 8),
        failure_threshold,
        target_label,
        evt_target["evt_return_level_target"],
        evaluator.latent_shape,
        subset_cfg.get("proposal_std", 0.35),
        subset_cfg.get("context_refresh_prob", 0.1),
        subset_cfg.get("mh_retries_per_sample", 4),
    )
    population_evaluator = _multiprocess_population_evaluator(
        evaluator,
        sampler,
        config,
    )
    with ExitStack() as stack:
        evaluate_many = None
        if population_evaluator is not None:
            evaluate_many = stack.enter_context(population_evaluator).evaluate_many
        result = run_subset_simulation(
            evaluator.evaluate,
            context_count=evaluator.context_count,
            latent_shape=evaluator.latent_shape,
            num_samples=subset_cfg.get("num_samples", 100),
            p0=subset_cfg.get("p0", 0.1),
            max_levels=subset_cfg.get("max_levels", 8),
            proposal_std=subset_cfg.get("proposal_std", 0.35),
            context_refresh_prob=subset_cfg.get("context_refresh_prob", 0.1),
            failure_threshold=failure_threshold,
            seed=config.get("training", {}).get("seed", 42),
            mh_retries_per_sample=subset_cfg.get("mh_retries_per_sample", 4),
            evaluate_many=evaluate_many,
        )
    _save_samples(result, output_dir)
    level_stats = _level_stats(result, failure_threshold)
    _write_level_stats(output_dir / "latent_subset_level_stats.csv", level_stats)
    reliability = _reliability_assessment(
        level_stats,
        config,
        num_contexts=len(contexts),
        num_samples=int(subset_cfg.get("num_samples", 100)),
    )
    message = (
        "Subset reliability %s at level %d | unique_contexts=%.0f "
        "unique_states=%.0f largest_context_share=%.3f "
        "largest_state_share=%.3f acceptance_rate=%s"
    )
    observed = reliability.get("observed", {})
    acceptance = observed.get("acceptance_rate")
    acceptance_text = (
        f"{float(acceptance):.3f}"
        if isinstance(acceptance, (int, float)) and np.isfinite(float(acceptance))
        else "nan"
    )
    log_fn = logger.info if reliability["status"] == "pass" else logger.warning
    log_fn(
        message,
        reliability["status"],
        reliability.get("assessed_level", -1),
        float(observed.get("unique_contexts", np.nan)),
        float(observed.get("unique_states", np.nan)),
        float(observed.get("largest_context_share", np.nan)),
        float(observed.get("largest_state_share", np.nan)),
        acceptance_text,
    )
    if reliability.get("failures"):
        logger.warning(
            "Subset reliability failures: %s",
            "; ".join(str(item) for item in reliability["failures"]),
        )
    if reliability.get("warnings"):
        logger.warning(
            "Subset reliability warnings: %s",
            "; ".join(str(item) for item in reliability["warnings"]),
        )
    figures = _write_diagnostic_plots(
        result,
        output_dir,
        failure_threshold,
    )
    summary = _summary(
        result,
        contexts,
        config,
        failure_threshold,
        evt_target,
        level_stats,
        figures,
        paths.get("exposure_summary"),
    )
    save_json(summary, output_dir / "latent_subset_summary.json")

    # ── 里程回报周期控制台打印 ──
    _log_mileage_return_period(summary, result, logger)
    save_json(
        _top_cases(result, contexts),
        output_dir / "latent_subset_top_cases.json",
    )
    return output_dir / "latent_subset_summary.json"
