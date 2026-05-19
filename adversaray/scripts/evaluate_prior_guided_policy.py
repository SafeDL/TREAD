#!/usr/bin/env python3
"""Evaluate a prior-guided policy in closed-loop highway-env rollouts."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler
from adversaray.src.prior_guided_train import (
    evaluate_prior_guided_policy,
    recorded_future_metrics,
    recorded_future_series,
    rollout_distance_metrics,
)
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _attach_runtime_paths(cfg: dict, base: Path) -> None:
    paths = cfg.get("paths", {})
    cfg["_runtime"] = {
        "config_dir": str(base),
        "natural_dataset_dir": str((base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()),
        "output_dir": str((base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()),
        "highd_events_csv": str((base / paths.get("highd_events_csv", "../../../data/highd_events/events.csv")).resolve()),
        "highd_raw_dir": str((base / paths.get("highd_raw_dir", "../../../highD_dataset/Matlab/data")).resolve()),
        "highd_config": str(
            (base / paths.get("highd_config", "../../../process_highD/scripts/configs/highd_default.yaml")).resolve()
        ),
    }


def _resolve_tail_score_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _select_tail_contexts(
    idx: np.ndarray,
    *,
    split: str,
    tail_score_path: Path,
    tail_min_quantile: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not tail_score_path.exists():
        raise FileNotFoundError(f"Tail score file not found: {tail_score_path}")
    data = _load_npz(tail_score_path)
    if "dataset_index" not in data or "criticality_score" not in data:
        raise KeyError(f"{tail_score_path} must contain 'dataset_index' and 'criticality_score'")
    score_idx = np.asarray(data["dataset_index"], dtype=np.int64)
    score = np.asarray(data["criticality_score"], dtype=np.float64)
    if score_idx.shape[0] != score.shape[0]:
        raise ValueError(
            f"{tail_score_path} has mismatched dataset_index and criticality_score lengths: "
            f"{score_idx.shape[0]} vs {score.shape[0]}"
        )
    split_mask = np.isin(score_idx, np.asarray(idx, dtype=np.int64))
    if not np.any(split_mask):
        raise RuntimeError(
            f"Tail score file {tail_score_path} does not cover split '{split}'. "
            "Build split-specific scores, for example: "
            "python adversaray/scripts/build_tail_context_scores.py "
            f"--split {split} --output-dir data/adversaray/following/prior_guided/{split}_tail_scores"
        )
    split_idx = score_idx[split_mask]
    split_score = score[split_mask]
    finite = np.isfinite(split_score)
    if not np.any(finite):
        raise RuntimeError(f"Tail score file {tail_score_path} has no finite criticality_score values for split '{split}'")
    quantile = float(tail_min_quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"--tail-min-quantile must be in [0, 1], got {quantile}")
    threshold = float(np.quantile(split_score[finite], quantile))
    candidate_mask = finite & (split_score >= threshold)
    if not np.any(candidate_mask):
        raise RuntimeError(
            f"Tail score file {tail_score_path} produced no tail candidates for split '{split}' at quantile {quantile}"
        )
    candidate_idx = split_idx[candidate_mask]
    candidate_score = split_score[candidate_mask]
    order = np.argsort(-candidate_score, kind="mergesort")
    selected = candidate_idx[order].astype(np.int64)
    metadata = {
        "context_selection": "tail",
        "tail_score_path": str(tail_score_path),
        "tail_min_quantile": quantile,
        "tail_threshold": threshold,
        "tail_candidate_count": int(selected.size),
    }
    return selected, metadata


def _comparison_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    mapping = {
        "collision_rate": "collision_rate",
        "valid_collision_rate": "collision_valid_rate",
        "invalid_collision_rate": "invalid_collision_rate",
        "near_collision_rate": "near_collision_rate",
        "hard_brake_rate": "hard_brake_rate",
        "min_ttc_mean": "min_ttc_mean",
        "min_ttc_p05": "min_ttc_p05",
        "min_gap_mean": "min_gap_mean",
        "min_gap_p05": "min_gap_p05",
        "min_rss_margin_mean": "min_rss_margin_mean",
        "min_rss_margin_p05": "min_rss_margin_p05",
        "prior_kl_mean": "prior_kl_mean",
        "prior_kl_p95": "prior_kl_p95",
        "guidance_norm_mean": "guidance_norm_mean",
        "guidance_norm_p95": "guidance_norm_p95",
        "lead_accel_mean": "lead_accel_mean_mean",
        "lead_accel_std": "lead_accel_std_mean",
        "lead_accel_min": "lead_accel_min_p05",
        "lead_accel_max": "lead_accel_max_p95",
        "lead_jerk_abs_mean": "lead_jerk_abs_mean_mean",
        "lead_jerk_abs_max": "lead_jerk_abs_max_p95",
        "lead_speed_mean": "lead_speed_mean_mean",
        "action_clip_rate": "action_clip_rate_mean",
        "jerk_violation_rate": "jerk_violation_rate_mean",
        "speed_negative_rate": "speed_negative_rate_mean",
    }
    return {f"{prefix}_{out_key}": float(metrics.get(in_key, float("nan"))) for out_key, in_key in mapping.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--policy-checkpoint", default="", help="Optional policy checkpoint override.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-contexts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-guidance", action="store_true", help="Evaluate the frozen diffusion prior only.")
    parser.add_argument("--compare-frozen-prior", action=argparse.BooleanOptionalAction, default=True, help="Evaluate frozen prior and guided policy on the same contexts.")
    parser.add_argument("--commit-steps", type=int, default=50, help="Evaluation replan cadence override.")
    parser.add_argument("--tail-val", action="store_true", help="Evaluate the highest-criticality subset of the selected split.")
    parser.add_argument("--tail-score-path", default="", help="Path to context_tail_scores.npz covering the selected split.")
    parser.add_argument("--tail-min-quantile", type=float, default=0.9, help="Criticality quantile threshold for --tail-val.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    cfg.setdefault("env", {})["commit_steps_max"] = int(args.commit_steps)
    base = cfg_path.parent
    _attach_runtime_paths(cfg, base)
    paths = cfg.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()
    if args.policy_checkpoint:
        cfg.setdefault("paths", {})["policy_checkpoint"] = args.policy_checkpoint
    else:
        default_ckpt = output_dir / "checkpoints" / "best_delta_reward.pt"
        if default_ckpt.exists():
            cfg.setdefault("paths", {})["policy_checkpoint"] = str(default_ckpt)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_npz(natural_dir / "dataset.npz")
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[args.split])[0]
    selection_metadata: dict[str, object] = {
        "context_selection": "default",
        "tail_score_path": None,
        "tail_min_quantile": None,
        "tail_threshold": None,
        "tail_candidate_count": int(len(idx)),
    }
    if args.tail_val:
        training = cfg.get("training", {})
        tail_score_value = str(args.tail_score_path or training.get("tail_score_path", "") or "").strip()
        if not tail_score_value:
            raise ValueError("--tail-val requires --tail-score-path or training.tail_score_path")
        tail_score_path = _resolve_tail_score_path(tail_score_value, base)
        tail_min_quantile = (
            float(args.tail_min_quantile)
            if args.tail_min_quantile is not None
            else float(training.get("tail_min_quantile", 0.95))
        )
        idx, selection_metadata = _select_tail_contexts(
            idx,
            split=args.split,
            tail_score_path=tail_score_path,
            tail_min_quantile=tail_min_quantile,
        )
    evaluated_context_count = int(min(len(idx), int(args.num_contexts)))
    selection_metadata["evaluated_context_count"] = evaluated_context_count
    recorded_metrics = recorded_future_metrics(raw, idx, max_contexts=int(args.num_contexts), config=cfg)
    if args.compare_frozen_prior:
        prior_cfg = copy.deepcopy(cfg)
        prior_cfg.setdefault("policy", {})["enabled"] = False
        prior_sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()
        prior_metrics = evaluate_prior_guided_policy(
            prior_sampler,
            prior_cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
            return_rows=True,
        )
        prior_rows = prior_metrics.pop("_rows", [])
        guided_sampler = PriorGuidedDiffusionSampler.from_config(cfg, config_dir=base).eval()
        guided_metrics = evaluate_prior_guided_policy(
            guided_sampler,
            cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
            return_rows=True,
        )
        guided_rows = guided_metrics.pop("_rows", [])
        recorded_series = recorded_future_series(raw, idx, max_contexts=int(args.num_contexts), config=cfg)
        paired_delta_keys = (
            "reward",
            "rss_reward",
            "gap_reward",
            "ttc_reward",
            "min_rss_margin",
            "min_gap",
            "min_ttc",
            "action_clip_rate",
            "jerk_violation_rate",
            "speed_negative_rate",
            "lead_physics_penalty",
            "prior_kl_per_plan",
            "guidance_norm_per_plan",
        )
        delta_metrics = {}
        for key in paired_delta_keys:
            prior_key = f"{key}_mean"
            guided_key = f"{key}_mean"
            if prior_key in prior_metrics and guided_key in guided_metrics:
                delta_metrics[f"{key}_delta_mean"] = float(guided_metrics[guided_key] - prior_metrics[prior_key])
        metrics = {
            **_comparison_metrics("prior", prior_metrics),
            **_comparison_metrics("guided", guided_metrics),
            **delta_metrics,
            **rollout_distance_metrics(recorded_series, "prior", prior_rows),
            **rollout_distance_metrics(recorded_series, "guided", guided_rows),
            "prior_kl_mean": float(guided_metrics.get("prior_kl_mean", float("nan"))),
            "guidance_norm_mean": float(guided_metrics.get("guidance_norm_mean", float("nan"))),
            "recorded_future": recorded_metrics,
            "prior_raw": prior_metrics,
            "guided_raw": guided_metrics,
        }
    else:
        if args.disable_guidance:
            cfg.setdefault("policy", {})["enabled"] = False
        sampler = PriorGuidedDiffusionSampler.from_config(cfg, config_dir=base).eval()
        metrics = evaluate_prior_guided_policy(
            sampler,
            cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
        )
        metrics["recorded_future"] = recorded_metrics
    save_json(
        {
            "split": args.split,
            "mode": "compare" if args.compare_frozen_prior else ("frozen_prior" if args.disable_guidance else "guided"),
            "num_contexts": evaluated_context_count,
            **selection_metadata,
            "metrics": metrics,
        },
        output_dir / "prior_guided_eval_summary.json",
    )


if __name__ == "__main__":
    main()
