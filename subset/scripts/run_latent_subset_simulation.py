#!/usr/bin/env python3
"""Run multi-context rolling latent-space subset simulation."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import resolve_path, write_csv
from utils.context import context_from_npz, load_context_npz
from diffusion.src.utils import load_yaml, save_json, setup_logging
from utils.evt import load_evt_model
from subset.src.closed_loop_runner import ClosedLoopFollowingRunner
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from subset.src.latent_evaluator import LatentMpcEpisodeEvaluator
from subset.src.subset_simulation import (
    SubsetLevel,
    run_subset_simulation,
)


DEFAULT_CONFIG_PATH = (
    ROOT / "subset" / "scripts" / "configs" / "latent_subset_simulation.yaml"
)
SCRIPT_DEFAULTS = {"log_level": "INFO"}
logger = logging.getLogger(__name__)


def _paths(config: dict[str, Any], base: Path) -> dict[str, Path]:
    paths = config.get("paths", {})
    required = ("tail_context_path", "evt_model_path")
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    output_value = config.get("subset_simulation", {}).get("output_dir")
    if not output_value:
        raise KeyError("Config subset_simulation.output_dir is required")
    return {
        "tail_contexts": resolve_path(paths["tail_context_path"], base),
        "evt_model": resolve_path(paths["evt_model_path"], base),
        "output_dir": resolve_path(str(output_value), base),
    }


def _load_contexts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Subset tail contexts not found: {path}")
    raw = load_context_npz(path)
    count = int(raw["context_states"].shape[0])
    return [context_from_npz(raw, idx) for idx in range(count)]


def _evt_failure_threshold(
    path: Path,
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"EVT model is required before subset simulation: {path}")
    evt_cfg = config.setdefault("evt", {})
    evt_cfg["model_path"] = str(path)
    evt_cfg["score_space"] = str(evt_cfg.get("score_space", "evt"))
    return_period = int(evt_cfg.get("return_period", 100))
    model = load_evt_model(path)
    z_target = float(model.return_level(return_period))
    failure_threshold = float(model.score(z_target))
    return failure_threshold, {
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
        "proxy_risk_score",
        "evt_tail_probability",
        "evt_return_level_target",
        "evt_failure_threshold",
        "ttc_objective",
        "thw_objective",
        "drac_objective",
        "gap_objective",
        "physical_feasible",
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


def _context_usage(levels: list[SubsetLevel], context_count: int) -> list[int]:
    if not levels:
        return [0] * context_count
    values = np.asarray(levels[-1].context_indices, dtype=np.int64)
    counts = np.bincount(values, minlength=context_count)
    return [int(item) for item in counts[:context_count]]


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

    acceptance = float(final.get("acceptance_rate", np.nan))
    if np.isfinite(acceptance) and final.get("level", 0.0) > 0.0:
        if acceptance < thresholds["min_acceptance_rate"]:
            failures.append(
                "acceptance_rate "
                f"{acceptance:.3f} < {thresholds['min_acceptance_rate']:.3f}"
            )
    elif final.get("level", 0.0) > 0.0:
        warnings.append("final-level acceptance_rate is unavailable")

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
            "acceptance_rate": final.get("acceptance_rate"),
        },
    }


def _write_level_stats(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    write_csv(path, rows)


def _write_diagnostic_plots(
    result,
    output_dir: Path,
    failure_threshold: float,
    context_count: int,
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

    fig, ax = plt.subplots(figsize=(7, 4))
    levels = [level.level for level in result.levels]
    thresholds = [level.threshold for level in result.levels]
    failure = [
        float(np.mean(level.scores >= float(failure_threshold)))
        for level in result.levels
    ]
    ax.plot(levels, thresholds, marker="o", label="subset threshold")
    ax.axhline(
        float(failure_threshold),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="failure threshold",
    )
    ax.set_xlabel("level")
    ax.set_ylabel("score")
    ax2 = ax.twinx()
    ax2.plot(levels, failure, marker="s", color="tab:red", label="failure fraction")
    ax2.set_ylabel("failure fraction")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    path = figure_dir / "subset_threshold_progression.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["threshold_progression"] = str(path)

    final_counts = np.bincount(
        np.asarray(result.levels[-1].context_indices, dtype=np.int64),
        minlength=context_count,
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(context_count), final_counts)
    ax.set_xlabel("context index")
    ax.set_ylabel("final level count")
    fig.tight_layout()
    path = figure_dir / "subset_final_context_usage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["final_context_usage"] = str(path)

    return paths


def _summary(
    result,
    contexts: list[dict[str, Any]],
    config: dict[str, Any],
    failure_threshold: float,
    evt_target: dict[str, float],
    level_stats: list[dict[str, float]],
    figures: dict[str, str],
) -> dict[str, Any]:
    subset_cfg = config.get("subset_simulation", {})
    uncertainty = _probability_uncertainty(
        result,
        num_samples=int(subset_cfg.get("num_samples", 100)),
        p0=float(subset_cfg.get("p0", 0.1)),
    )
    reliability = _reliability_assessment(
        level_stats,
        config,
        num_contexts=len(contexts),
        num_samples=int(subset_cfg.get("num_samples", 100)),
    )
    source_types = {str(context.get("source_type", "")) for context in contexts}
    if source_types == {"highd_event_tail"}:
        probability_target = (
            "P_context,z(Y_long_sim > z_m | o in highD tail contexts)"
        )
    else:
        probability_target = "P_context,z(Y_long_sim > z_m | configured contexts)"
    return {
        "probability": float(result.probability),
        **uncertainty,
        "reliability": reliability,
        "probability_target": probability_target,
        "score_space": str(config.get("evt", {}).get("score_space", "evt")),
        **evt_target,
        "failure_threshold": float(failure_threshold),
        "final_failure_fraction": float(result.final_failure_fraction),
        "thresholds": [float(level.threshold) for level in result.levels],
        "acceptance_rates": [float(level.acceptance_rate) for level in result.levels],
        "level_stats": level_stats,
        "figures": figures,
        "num_levels": len(result.levels),
        "num_contexts": len(contexts),
        "context_usage_final_level": _context_usage(
            result.levels,
            len(contexts),
        ),
        "num_samples": int(subset_cfg.get("num_samples", 100)),
        "p0": float(subset_cfg.get("p0", 0.1)),
        "proposal_std": float(subset_cfg.get("proposal_std", 0.35)),
        "context_refresh_prob": float(subset_cfg.get("context_refresh_prob", 0.1)),
        "mh_retries_per_sample": int(subset_cfg.get("mh_retries_per_sample", 4)),
        "refresh_attempts_per_sample": int(
            subset_cfg.get("refresh_attempts_per_sample", 4)
        ),
        "min_next_unique_contexts": int(subset_cfg.get("min_next_unique_contexts", 4)),
        "min_next_unique_states": int(subset_cfg.get("min_next_unique_states", 10)),
        "stop_on_collapse": bool(subset_cfg.get("stop_on_collapse", True)),
        "max_levels": int(subset_cfg.get("max_levels", 8)),
        "episode_steps": int(config.get("env", {}).get("episode_steps", 200)),
        "commit_steps_max": int(config.get("env", {}).get("commit_steps_max", 10)),
    }


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    base = DEFAULT_CONFIG_PATH.parent
    config = load_yaml(DEFAULT_CONFIG_PATH)
    paths = _paths(config, base)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = _load_contexts(paths["tail_contexts"])
    failure_threshold, evt_target = _evt_failure_threshold(paths["evt_model"], config)
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=base)
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
    subset_cfg = config.get("subset_simulation", {})
    logger.info(
        (
            "Running mixed-context subset simulation contexts=%d "
            "samples=%d p0=%.3f max_levels=%d threshold=%.6f "
            "z_m=%.6f latent_shape=%s"
        ),
        len(contexts),
        int(subset_cfg.get("num_samples", 100)),
        float(subset_cfg.get("p0", 0.1)),
        int(subset_cfg.get("max_levels", 8)),
        failure_threshold,
        evt_target["evt_return_level_target"],
        evaluator.latent_shape,
    )
    result = run_subset_simulation(
        evaluator.evaluate,
        context_count=evaluator.context_count,
        latent_shape=evaluator.latent_shape,
        num_samples=int(subset_cfg.get("num_samples", 100)),
        p0=float(subset_cfg.get("p0", 0.1)),
        max_levels=int(subset_cfg.get("max_levels", 8)),
        proposal_std=float(subset_cfg.get("proposal_std", 0.35)),
        context_refresh_prob=float(subset_cfg.get("context_refresh_prob", 0.1)),
        failure_threshold=failure_threshold,
        seed=int(config.get("training", {}).get("seed", 42)),
        mh_retries_per_sample=int(subset_cfg.get("mh_retries_per_sample", 4)),
        refresh_attempts_per_sample=int(
            subset_cfg.get("refresh_attempts_per_sample", 4)
        ),
        min_next_unique_contexts=int(subset_cfg.get("min_next_unique_contexts", 4)),
        min_next_unique_states=int(subset_cfg.get("min_next_unique_states", 10)),
        stop_on_collapse=bool(subset_cfg.get("stop_on_collapse", True)),
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
        len(contexts),
    )
    save_json(
        _summary(
            result,
            contexts,
            config,
            failure_threshold,
            evt_target,
            level_stats,
            figures,
        ),
        output_dir / "latent_subset_summary.json",
    )
    save_json(
        _top_cases(result, contexts),
        output_dir / "latent_subset_top_cases.json",
    )
    logger.info(
        "Saved latent subset simulation result to %s probability %.6g",
        output_dir,
        result.probability,
    )


if __name__ == "__main__":
    main()
