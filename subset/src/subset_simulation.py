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


def _state_key(context: int, z: np.ndarray) -> tuple[int, bytes]:
    return int(context), np.ascontiguousarray(z).tobytes()


def _unique_state_count(
    context_indices: np.ndarray,
    latents: np.ndarray,
) -> int:
    return len(
        {
            _state_key(int(context_indices[idx]), latents[idx])
            for idx in range(int(latents.shape[0]))
        }
    )


def _diverse_elite_indices(
    scores: np.ndarray,
    context_indices: np.ndarray,
    latents: np.ndarray,
    *,
    threshold: float,
    elite_count: int,
) -> np.ndarray:
    if elite_count <= 0:
        raise ValueError("elite_count must be positive")
    order = np.argsort(scores)[::-1]
    eligible = [int(idx) for idx in order if float(scores[idx]) >= threshold]
    selected: list[int] = []
    seen_states: set[tuple[int, bytes]] = set()
    seen_contexts: set[int] = set()

    for idx in eligible:
        context = int(context_indices[idx])
        key = _state_key(context, latents[idx])
        if key in seen_states or context in seen_contexts:
            continue
        selected.append(idx)
        seen_states.add(key)
        seen_contexts.add(context)
        if len(selected) >= elite_count:
            return np.asarray(selected, dtype=np.int64)

    for idx in eligible:
        context = int(context_indices[idx])
        key = _state_key(context, latents[idx])
        if key in seen_states:
            continue
        selected.append(idx)
        seen_states.add(key)
        if len(selected) >= elite_count:
            break
    return np.asarray(selected, dtype=np.int64)


def _mh_proposal(
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
) -> tuple[int, np.ndarray, float] | None:
    if rng.random() < context_refresh_prob:
        proposal_context = int(rng.integers(0, int(context_count)))
        proposal_z = rng.standard_normal(current_z.shape).astype(np.float32)
        proposal_eval = evaluate(proposal_context, proposal_z)
        if float(proposal_eval.score) < threshold:
            return None
        return proposal_context, proposal_z, float(proposal_eval.score)

    proposal_z = current_z + proposal_std * rng.standard_normal(
        current_z.shape
    )
    proposal_z = proposal_z.astype(np.float32)
    proposal_eval = evaluate(current_context, proposal_z)
    if float(proposal_eval.score) < threshold:
        return None

    log_alpha = standard_normal_log_prob(proposal_z)
    log_alpha -= standard_normal_log_prob(current_z)
    if np.log(rng.random()) <= min(0.0, log_alpha):
        return current_context, proposal_z, float(proposal_eval.score)
    return None


def _fresh_above_threshold(
    evaluate: EvaluateFn,
    rng: np.random.Generator,
    *,
    context_count: int,
    latent_shape: tuple[int, int, int],
    threshold: float,
) -> tuple[int, np.ndarray, float] | None:
    proposal_context = int(rng.integers(0, int(context_count)))
    proposal_z = rng.standard_normal(latent_shape).astype(np.float32)
    proposal_eval = evaluate(proposal_context, proposal_z)
    if float(proposal_eval.score) < threshold:
        return None
    return proposal_context, proposal_z, float(proposal_eval.score)


def _build_next_population(
    context_indices: np.ndarray,
    latents: np.ndarray,
    scores: np.ndarray,
    evaluate: EvaluateFn,
    rng: np.random.Generator,
    *,
    context_count: int,
    latent_shape: tuple[int, int, int],
    num_samples: int,
    threshold: float,
    elite_count: int,
    proposal_std: float,
    context_refresh_prob: float,
    mh_retries_per_sample: int,
    refresh_attempts_per_sample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    elite_idx = _diverse_elite_indices(
        scores,
        context_indices,
        latents,
        threshold=threshold,
        elite_count=elite_count,
    )
    if elite_idx.size == 0:
        raise RuntimeError(
            "No elite samples met the subset threshold; cannot build next level"
        )

    chain_states: list[tuple[int, np.ndarray, float]] = [
        (
            int(context_indices[idx]),
            latents[idx].copy(),
            float(scores[idx]),
        )
        for idx in elite_idx
    ]
    next_contexts: list[int] = []
    next_latents: list[np.ndarray] = []
    next_accepted: list[float] = []

    for context, z, _score in chain_states:
        next_contexts.append(int(context))
        next_latents.append(z.copy())
        next_accepted.append(0.0)
        if len(next_latents) >= num_samples:
            break

    cursor = 0
    while len(next_latents) < num_samples:
        chain_idx = cursor % len(chain_states)
        current_context, current_z, current_score = chain_states[chain_idx]
        accepted_state: tuple[int, np.ndarray, float] | None = None
        for _attempt in range(max(1, int(mh_retries_per_sample))):
            accepted_state = _mh_proposal(
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
            if accepted_state is not None:
                chain_states[chain_idx] = accepted_state
                break

        if accepted_state is None:
            for _attempt in range(max(0, int(refresh_attempts_per_sample))):
                accepted_state = _fresh_above_threshold(
                    evaluate,
                    rng,
                    context_count=context_count,
                    latent_shape=latent_shape,
                    threshold=threshold,
                )
                if accepted_state is not None:
                    chain_states.append(accepted_state)
                    break

        if accepted_state is None:
            accepted_state = chain_states[chain_idx]
            is_accepted = 0.0
        else:
            is_accepted = 1.0

        context, z, _score = accepted_state
        next_contexts.append(int(context))
        next_latents.append(z.copy())
        next_accepted.append(is_accepted)
        cursor += 1

    return (
        np.asarray(next_contexts, dtype=np.int64),
        np.asarray(next_latents, dtype=np.float32),
        np.asarray(next_accepted, dtype=np.float32),
        float(np.mean(next_accepted)),
    )


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
    mh_retries_per_sample: int = 4,
    refresh_attempts_per_sample: int = 4,
    min_next_unique_contexts: int = 2,
    min_next_unique_states: int = 2,
    stop_on_collapse: bool = True,
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
    if mh_retries_per_sample <= 0:
        raise ValueError("mh_retries_per_sample must be positive")
    if refresh_attempts_per_sample < 0:
        raise ValueError("refresh_attempts_per_sample must be non-negative")

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

        context_indices_next, latents_next, accepted, acceptance_rate = (
            _build_next_population(
                context_indices,
                latents,
                scores,
                evaluate,
                rng,
                context_count=context_count,
                latent_shape=latent_shape,
                num_samples=num_samples,
                threshold=threshold,
                elite_count=elite_count,
                proposal_std=proposal_std,
                context_refresh_prob=context_refresh_prob,
                mh_retries_per_sample=mh_retries_per_sample,
                refresh_attempts_per_sample=refresh_attempts_per_sample,
            )
        )
        levels[-1].accepted = accepted
        levels[-1].acceptance_rate = acceptance_rate
        next_unique_contexts = int(np.unique(context_indices_next).shape[0])
        next_unique_states = _unique_state_count(
            context_indices_next,
            latents_next,
        )
        logger.info(
            (
                "Subset level %d MH acceptance_rate %.6f "
                "next_unique_contexts=%d next_unique_states=%d"
            ),
            level_idx,
            levels[-1].acceptance_rate,
            next_unique_contexts,
            next_unique_states,
        )
        if (
            bool(stop_on_collapse)
            and (
                next_unique_contexts < int(min_next_unique_contexts)
                or next_unique_states < int(min_next_unique_states)
            )
        ):
            probability = (float(p0) ** level_idx) * final_failure_fraction
            logger.warning(
                (
                    "Stopping subset simulation before level %d to avoid "
                    "Markov-chain collapse: next_unique_contexts=%d "
                    "next_unique_states=%d probability=%.8g"
                ),
                level_idx + 1,
                next_unique_contexts,
                next_unique_states,
                probability,
            )
            break

        context_indices = context_indices_next
        latents = latents_next

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
