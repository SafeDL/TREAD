"""Random-walk Metropolis subset simulation in latent/context space."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .latent_evaluator import LatentEvaluation


EvaluateFn = Callable[[int, np.ndarray], LatentEvaluation]
logger = logging.getLogger(__name__)


@dataclass
class SubsetLevel:
    level: int
    context_indices: np.ndarray
    latents: np.ndarray
    scores: np.ndarray
    actions: list[np.ndarray]
    metrics: list[dict[str, float]]
    traces: list[list[dict[str, float]]]
    threshold: float
    accepted: np.ndarray
    acceptance_rate: float


@dataclass
class SubsetSimulationResult:
    levels: list[SubsetLevel]
    probability: float
    final_failure_fraction: float
    failure_threshold: float


def standard_normal_log_prob(z: np.ndarray) -> float:
    value = np.asarray(z, dtype=np.float64)
    return float(-0.5 * np.sum(value * value))


def _evaluate_population(
    context_indices: np.ndarray,
    latents: np.ndarray,
    evaluate: EvaluateFn,
    *,
    level: int,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]], list]:
    scores: list[float] = []
    actions: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    traces: list[list[dict[str, float]]] = []
    total = int(latents.shape[0])
    interval = max(1, total // 10)
    for idx, latent in enumerate(latents):
        context_index = int(context_indices[idx])
        result = evaluate(context_index, latent)
        scores.append(float(result.score))
        actions.append(result.actions)
        item_metrics = dict(result.metrics)
        item_metrics["context_index"] = float(context_index)
        metrics.append(item_metrics)
        traces.append(result.trace)
        done = idx + 1
        if done == total or done % interval == 0:
            logger.info(
                "Subset level %d evaluated %d/%d samples",
                level,
                done,
                total,
            )
    return np.asarray(scores, dtype=np.float64), actions, metrics, traces


def _elite_indices(scores: np.ndarray, elite_count: int) -> np.ndarray:
    if elite_count <= 0:
        raise ValueError("elite_count must be positive")
    order = np.argsort(scores)[::-1]
    return order[:elite_count].astype(np.int64)


def _mh_next(
    current_context: int,
    current_z: np.ndarray,
    current_score: float,
    evaluate: EvaluateFn,
    rng: np.random.Generator,
    *,
    context_count: int,
    proposal_std: float,
    context_refresh_prob: float,
    threshold: float,
) -> tuple[int, np.ndarray, float, bool]:
    if rng.random() < context_refresh_prob:
        proposal_context = int(rng.integers(0, int(context_count)))
        proposal_z = rng.standard_normal(current_z.shape).astype(np.float32)
        proposal_eval = evaluate(proposal_context, proposal_z)
        if float(proposal_eval.score) < threshold:
            return current_context, current_z, current_score, False
        return proposal_context, proposal_z, float(proposal_eval.score), True

    proposal_z = current_z + proposal_std * rng.standard_normal(
        current_z.shape
    )
    proposal_z = proposal_z.astype(np.float32)
    proposal_eval = evaluate(current_context, proposal_z)
    if float(proposal_eval.score) < threshold:
        return current_context, current_z, current_score, False

    log_alpha = standard_normal_log_prob(proposal_z)
    log_alpha -= standard_normal_log_prob(current_z)
    if np.log(rng.random()) <= min(0.0, log_alpha):
        return current_context, proposal_z, float(proposal_eval.score), True
    return current_context, current_z, current_score, False


def run_subset_simulation(
    evaluate: EvaluateFn,
    *,
    context_count: int,
    latent_shape: tuple[int, int, int],
    num_samples: int,
    p0: float,
    max_levels: int,
    proposal_std: float,
    context_refresh_prob: float,
    failure_threshold: float,
    seed: int,
) -> SubsetSimulationResult:
    if context_count <= 0:
        raise ValueError("context_count must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    if proposal_std <= 0.0:
        raise ValueError("proposal_std must be positive")
    if not 0.0 <= context_refresh_prob <= 1.0:
        raise ValueError("context_refresh_prob must be in [0, 1]")

    rng = np.random.default_rng(int(seed))
    elite_count = max(1, int(round(float(num_samples) * float(p0))))
    if elite_count >= num_samples:
        raise ValueError("p0 leaves no non-elite samples")

    context_indices = rng.integers(
        0,
        int(context_count),
        size=int(num_samples),
        dtype=np.int64,
    )
    latents = rng.standard_normal((num_samples, *latent_shape))
    latents = latents.astype(np.float32)
    levels: list[SubsetLevel] = []
    probability = float("nan")
    final_failure_fraction = 0.0

    for level_idx in range(max_levels):
        logger.info("Subset level %d started", level_idx)
        scores, actions, metrics, traces = _evaluate_population(
            context_indices,
            latents,
            evaluate,
            level=level_idx,
        )
        threshold = float(np.quantile(scores, 1.0 - float(p0)))
        accepted = np.ones(num_samples, dtype=np.float32)
        acceptance_rate = 1.0 if level_idx == 0 else float("nan")
        levels.append(
            SubsetLevel(
                level=level_idx,
                context_indices=context_indices.copy(),
                latents=latents.copy(),
                scores=scores.copy(),
                actions=actions,
                metrics=metrics,
                traces=traces,
                threshold=threshold,
                accepted=accepted,
                acceptance_rate=acceptance_rate,
            )
        )

        final_failure_fraction = float(
            np.mean(scores >= float(failure_threshold))
        )
        logger.info(
            (
                "Subset level %d threshold %.6f score_min %.6f "
                "score_mean %.6f score_max %.6f failure_fraction %.6f"
            ),
            level_idx,
            threshold,
            float(np.min(scores)),
            float(np.mean(scores)),
            float(np.max(scores)),
            final_failure_fraction,
        )
        if threshold >= failure_threshold or level_idx == max_levels - 1:
            probability = (float(p0) ** level_idx) * final_failure_fraction
            break

        elite_idx = _elite_indices(scores, elite_count)
        next_contexts: list[int] = []
        next_latents: list[np.ndarray] = []
        next_accepted: list[float] = []
        chain_len = int(np.ceil(num_samples / elite_count))
        for seed_idx in elite_idx:
            current_context = int(context_indices[seed_idx])
            current_z = latents[seed_idx].copy()
            current_score = float(scores[seed_idx])
            for _step in range(chain_len):
                current_context, current_z, current_score, is_accepted = (
                    _mh_next(
                        current_context,
                        current_z,
                        current_score,
                        evaluate,
                        rng,
                        context_count=context_count,
                        proposal_std=proposal_std,
                        context_refresh_prob=context_refresh_prob,
                        threshold=threshold,
                    )
                )
                next_contexts.append(int(current_context))
                next_latents.append(current_z.copy())
                next_accepted.append(float(is_accepted))
                if len(next_latents) >= num_samples:
                    break
            if len(next_latents) >= num_samples:
                break

        context_indices = np.asarray(next_contexts, dtype=np.int64)
        latents = np.asarray(next_latents, dtype=np.float32)
        levels[-1].accepted = np.asarray(next_accepted, dtype=np.float32)
        levels[-1].acceptance_rate = float(np.mean(levels[-1].accepted))
        logger.info(
            "Subset level %d MH acceptance_rate %.6f",
            level_idx,
            levels[-1].acceptance_rate,
        )

    logger.info(
        "Subset simulation finished probability %.8g after %d levels",
        probability,
        len(levels),
    )
    return SubsetSimulationResult(
        levels=levels,
        probability=float(probability),
        final_failure_fraction=float(final_failure_fraction),
        failure_threshold=float(failure_threshold),
    )
