#!/usr/bin/env python3
"""Optional pilot score diagnostics for EVT-calibrated subset simulation."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from utils.context import context_from_npz, load_context_npz
from utils.io import resolve_path, write_csv, write_json
from subset.src.closed_loop_runner import ClosedLoopFollowingRunner
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from subset.src.latent_evaluator import LatentMpcEpisodeEvaluator


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
    required = (
        "tail_context_path",
        "pilot_diagnostic_path",
        "pilot_scores_path",
    )
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    return {
        "tail_contexts": resolve_path(paths["tail_context_path"], base),
        "diagnostic": resolve_path(paths["pilot_diagnostic_path"], base),
        "scores": resolve_path(paths["pilot_scores_path"], base),
    }


def _load_contexts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Subset tail contexts not found: {path}")
    raw = load_context_npz(path)
    return [
        context_from_npz(raw, idx)
        for idx in range(int(raw["context_states"].shape[0]))
    ]


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    base = DEFAULT_CONFIG_PATH.parent
    config = load_yaml(DEFAULT_CONFIG_PATH)
    paths = _paths(config, base)
    contexts = _load_contexts(paths["tail_contexts"])
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
    pilot_cfg = config.get("pilot_diagnostic", {})
    samples_per_context = int(pilot_cfg.get("samples_per_context", 16))
    upper_quantile = float(pilot_cfg.get("upper_quantile", 0.95))
    lower_quantile = float(pilot_cfg.get("lower_quantile", 0.90))
    rng = np.random.default_rng(int(config.get("training", {}).get("seed", 42)))
    rows: list[dict[str, Any]] = []
    for context_idx, context in enumerate(contexts):
        logger.info(
            "Pilot context %d/%d event=%s",
            context_idx + 1,
            len(contexts),
            context.get("event_id"),
        )
        for sample_idx in range(samples_per_context):
            latent = rng.standard_normal(evaluator.latent_shape).astype(
                np.float32
            )
            result = evaluator.evaluate(context_idx, latent)
            rows.append(
                {
                    "context_index": int(context_idx),
                    "sample_index": int(sample_idx),
                    "recording_id": context.get("recording_id"),
                    "event_id": context.get("event_id"),
                    "score": float(result.score),
                    "collision": float(result.metrics.get("collision", 0.0)),
                    "near_collision": float(
                        result.metrics.get("near_collision", 0.0)
                    ),
                    "min_gap": float(result.metrics.get("min_gap", np.nan)),
                    "min_ttc": float(result.metrics.get("min_ttc", np.nan)),
                }
            )
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    q90 = float(np.quantile(scores, lower_quantile))
    q95 = float(np.quantile(scores, upper_quantile))
    write_csv(paths["scores"], rows)
    write_json(
        paths["diagnostic"],
        {
            "diagnostic_only": True,
            "note": (
                "Pilot quantiles are score diagnostics only; subset final "
                "failure threshold is defined by the EVT return level."
            ),
            "upper_quantile": float(upper_quantile),
            "lower_quantile": float(lower_quantile),
            "score_q90": float(q90),
            "score_q95": float(q95),
            "score_min": float(np.min(scores)),
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
            "num_contexts": int(len(contexts)),
            "samples_per_context": int(samples_per_context),
            "num_samples": int(len(rows)),
            "collision_rate": float(
                np.mean([row["collision"] for row in rows])
            ),
            "near_collision_rate": float(
                np.mean([row["near_collision"] for row in rows])
            ),
        },
    )
    logger.info("Saved pilot score diagnostics to %s", paths["diagnostic"])


if __name__ == "__main__":
    main()
