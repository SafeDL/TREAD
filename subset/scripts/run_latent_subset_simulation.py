#!/usr/bin/env python3
"""Run multi-context rolling latent-space subset simulation."""
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

from utils.io import resolve_path, write_csv
from utils.context import context_from_npz, load_context_npz
from diffusion.src.utils import load_yaml, save_json, setup_logging
from subset.src.closed_loop_runner import ClosedLoopFollowingRunner
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from subset.src.latent_evaluator import LatentMpcEpisodeEvaluator
from subset.src.subset_simulation import (
    SubsetLevel,
    run_subset_simulation,
)


DEFAULT_CONFIG_PATH = (
    ROOT
    / "subset"
    / "scripts"
    / "configs"
    / "latent_subset_simulation.yaml"
)
SCRIPT_DEFAULTS = {"log_level": "INFO"}
logger = logging.getLogger(__name__)


def _paths(config: dict[str, Any], base: Path) -> dict[str, Path]:
    paths = config.get("paths", {})
    required = ("tail_context_path", "pilot_threshold_path")
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    output_value = config.get("subset_simulation", {}).get("output_dir")
    if not output_value:
        raise KeyError("Config subset_simulation.output_dir is required")
    return {
        "tail_contexts": resolve_path(paths["tail_context_path"], base),
        "pilot_threshold": resolve_path(paths["pilot_threshold_path"], base),
        "output_dir": resolve_path(str(output_value), base),
    }


def _load_contexts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Subset tail contexts not found: {path}")
    raw = load_context_npz(path)
    count = int(raw["context_states"].shape[0])
    return [context_from_npz(raw, idx) for idx in range(count)]


def _load_failure_threshold(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(
            "Pilot threshold is required before subset simulation: "
            f"{path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "failure_threshold" not in payload:
        raise KeyError(f"{path} is missing failure_threshold")
    return float(payload["failure_threshold"])


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
        int(action.shape[0])
        for level in levels
        for action in level.actions
    )
    max_dim = max(
        int(action.shape[1])
        for level in levels
        for action in level.actions
    )
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
        collision_valid=_metric_array(levels, "collision_valid"),
        min_gap=_metric_array(levels, "min_gap"),
        min_ttc=_metric_array(levels, "min_ttc"),
        min_rss_margin=_metric_array(levels, "min_rss_margin"),
        physical_feasible=_metric_array(levels, "physical_feasible"),
    )


def _top_cases(
    result,
    contexts: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    metric_keys = (
        "collision",
        "collision_valid",
        "near_collision",
        "min_gap",
        "min_ttc",
        "min_rss_margin",
        "risk_score",
        "proxy_risk_score",
        "relative_rss_objective",
        "ttc_objective",
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


def _level_stats(result, failure_threshold: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for level in result.levels:
        scores = np.asarray(level.scores, dtype=np.float64)
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
                "failure_fraction": float(
                    np.mean(scores >= float(failure_threshold))
                ),
                "acceptance_rate": float(level.acceptance_rate),
                "unique_contexts": float(
                    len(set(int(x) for x in level.context_indices))
                ),
                "unique_latents": float(
                    np.unique(
                        np.ascontiguousarray(level.latents).reshape(
                            level.latents.shape[0],
                            -1,
                        ),
                        axis=0,
                    ).shape[0]
                ),
            }
        )
    return rows


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
    level_stats: list[dict[str, float]],
    figures: dict[str, str],
) -> dict[str, Any]:
    subset_cfg = config.get("subset_simulation", {})
    uncertainty = _probability_uncertainty(
        result,
        num_samples=int(subset_cfg.get("num_samples", 100)),
        p0=float(subset_cfg.get("p0", 0.1)),
    )
    return {
        "probability": float(result.probability),
        **uncertainty,
        "probability_target": (
            "P_context,z(score > threshold | selected tail contexts)"
        ),
        "failure_threshold": float(failure_threshold),
        "final_failure_fraction": float(result.final_failure_fraction),
        "thresholds": [float(level.threshold) for level in result.levels],
        "acceptance_rates": [
            float(level.acceptance_rate) for level in result.levels
        ],
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
        "context_refresh_prob": float(
            subset_cfg.get("context_refresh_prob", 0.1)
        ),
        "max_levels": int(subset_cfg.get("max_levels", 8)),
        "episode_steps": int(config.get("env", {}).get("episode_steps", 200)),
        "commit_steps_max": int(
            config.get("env", {}).get("commit_steps_max", 10)
        ),
    }


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    base = DEFAULT_CONFIG_PATH.parent
    config = load_yaml(DEFAULT_CONFIG_PATH)
    paths = _paths(config, base)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = _load_contexts(paths["tail_contexts"])
    failure_threshold = _load_failure_threshold(paths["pilot_threshold"])
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
            "latent_shape=%s"
        ),
        len(contexts),
        int(subset_cfg.get("num_samples", 100)),
        float(subset_cfg.get("p0", 0.1)),
        int(subset_cfg.get("max_levels", 8)),
        failure_threshold,
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
        context_refresh_prob=float(
            subset_cfg.get("context_refresh_prob", 0.1)
        ),
        failure_threshold=failure_threshold,
        seed=int(config.get("training", {}).get("seed", 42)),
    )
    _save_samples(result, output_dir)
    level_stats = _level_stats(result, failure_threshold)
    _write_level_stats(output_dir / "latent_subset_level_stats.csv", level_stats)
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
