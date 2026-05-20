#!/usr/bin/env python3
"""Evaluate a prior-guided policy in closed-loop highway-env rollouts."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
from adversaray.src.config_utils import apply_rss_config_override
from adversaray.src.king_gradient_guidance import optimize_action_plan_king
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler
from adversaray.src.prior_guided_train import (
    _batch_observation_for_contexts,
    _context,
    _summarize_rows,
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


def _expert_rows_from_dataset(
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
) -> dict[str, float]:
    was_training = sampler.policy.training
    sampler.eval()
    runner = ClosedLoopFollowingRunner(sampler, config)
    rows: list[dict[str, Any]] = []
    for idx in indices[:max_contexts]:
        result = runner.rollout_pre_sampled_plan(_context(raw, int(idx)), np.asarray(raw["expert_plan"][idx], dtype=np.float32))
        rows.append(
            {
                "reward": float(result.reward),
                "prior_kl": float(result.prior_kl_sum.detach().cpu()),
                "guidance_norm": float(result.guidance_norm_sum.detach().cpu()),
                "trace": result.trace,
                **result.metrics,
            }
        )
    sampler.train(was_training)
    summary = _summarize_rows(rows)
    summary["_rows"] = rows  # type: ignore[assignment]
    return summary


def _pair_delta_metrics(
    lhs_name: str,
    rhs_name: str,
    lhs_metrics: dict[str, float],
    rhs_metrics: dict[str, float],
) -> dict[str, float]:
    keys = (
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
    )
    out: dict[str, float] = {}
    for key in keys:
        lhs_key = f"{key}_mean"
        rhs_key = f"{key}_mean"
        if lhs_key in lhs_metrics and rhs_key in rhs_metrics:
            out[f"{lhs_name}_{rhs_name}_{key}_delta_mean"] = float(lhs_metrics[lhs_key] - rhs_metrics[rhs_key])
    return out


def _king_proxy_config(sampler: PriorGuidedDiffusionSampler, config: dict[str, Any]) -> dict[str, Any]:
    proxy_config = copy.deepcopy(sampler.prior.config)
    proxy_config["king_gradient"] = copy.deepcopy(config.get("king_gradient", {}))
    proxy_config["rss"] = copy.deepcopy(config.get("rss", {}))
    proxy_config["physics"] = copy.deepcopy(config.get("physics", {}))
    return proxy_config


def _rollout_row(result: Any) -> dict[str, float]:
    return {
        "reward": float(result.reward),
        "prior_kl": float(result.prior_kl_sum.detach().cpu()),
        "guidance_norm": float(result.guidance_norm_sum.detach().cpu()),
        "trace": result.trace,
        **result.metrics,
    }


def _paired_king_row(
    prior_result: Any,
    king_result: Any,
    prior_plan: np.ndarray,
    king_plan: np.ndarray,
    king_diag: dict[str, Any],
    pos: int,
) -> dict[str, float]:
    prior = _rollout_row(prior_result)
    king = _rollout_row(king_result)
    row: dict[str, float] = {}
    for key, value in prior.items():
        if isinstance(value, (int, float, np.floating)):
            row[f"prior_{key}"] = float(value)
    for key, value in king.items():
        if isinstance(value, (int, float, np.floating)):
            row[f"king_{key}"] = float(value)
    for key in (
        "reward",
        "rss_reward",
        "gap_reward",
        "ttc_reward",
        "min_rss_margin",
        "min_gap",
        "min_ttc",
        "collision",
        "collision_valid",
        "near_collision",
        "action_clip_rate",
        "jerk_violation_rate",
        "speed_negative_rate",
        "lead_physics_penalty",
    ):
        row[f"{key}_delta"] = float(king.get(key, 0.0) - prior.get(key, 0.0))
    diff = np.asarray(king_plan, dtype=np.float32) - np.asarray(prior_plan, dtype=np.float32)
    row["action_l2"] = float(np.sqrt(np.mean(np.square(diff))))
    row["proxy_risk_before"] = float(king_diag["risk_before"][pos].detach().cpu())
    row["proxy_risk_after"] = float(king_diag["risk_after"][pos].detach().cpu())
    row["proxy_risk_delta"] = row["proxy_risk_after"] - row["proxy_risk_before"]
    row["proxy_nat_penalty"] = float(king_diag["naturalness_penalty"][pos].detach().cpu())
    row["proxy_physics_penalty"] = float(king_diag["physics_penalty"][pos].detach().cpu())
    return row


def evaluate_king_gradient_policy(
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
    seed: int,
    return_rows: bool = False,
) -> dict[str, Any]:
    was_training = sampler.policy.training
    was_enabled = sampler.schedule.enabled
    sampler.eval()
    sampler.set_guidance_enabled(False)
    runner = ClosedLoopFollowingRunner(sampler, config)
    device = sampler.prior.device
    proxy_config = _king_proxy_config(sampler, config)
    rows: list[dict[str, float]] = []
    for offset, idx_value in enumerate(indices[:max_contexts]):
        ctx = _context(raw, int(idx_value))
        batch, prepared = _batch_observation_for_contexts(runner, [ctx])
        with torch.no_grad():
            prior_sample = sampler.sample_batch(batch, seed=[int(seed) + offset])
            raw_context = sampler.prior.decode_context_states(batch["context_states"].to(device).float())
        king_diag = optimize_action_plan_king(
            prior_sample.raw_actions.to(device),
            raw_context,
            batch.get("ego_length"),
            batch.get("adv_length"),
            sampler.prior.schema,
            proxy_config,
        )
        prior_plan = prior_sample.raw_actions[0].detach().cpu().numpy().astype(np.float32)
        king_plan = king_diag["adv_actions"][0].detach().cpu().numpy().astype(np.float32)
        prior_result = runner.rollout_pre_sampled_plan(
            prepared[0],
            prior_plan,
            log_prob_sum=prior_sample.trajectory_log_prob[0],
            prior_kl_sum=prior_sample.prior_kl[0],
            guidance_norm_sum=prior_sample.guidance_norm[0],
        )
        king_result = runner.rollout_pre_sampled_plan(prepared[0], king_plan)
        rows.append(_paired_king_row(prior_result, king_result, prior_plan, king_plan, king_diag, 0))
    sampler.set_guidance_enabled(was_enabled)
    sampler.train(was_training)
    summary = _summarize_rows(rows)
    if return_rows:
        summary["_rows"] = rows  # type: ignore[assignment]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--policy-checkpoint", default="", help="Optional policy checkpoint override.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-contexts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--king-gradient", action="store_true", help="Evaluate frozen prior plans after KING-style action-gradient optimization.")
    parser.add_argument("--disable-guidance", action="store_true", help="Evaluate the frozen diffusion prior only.")
    parser.add_argument("--compare-frozen-prior", action=argparse.BooleanOptionalAction, default=True, help="Evaluate frozen prior and guided policy on the same contexts.")
    parser.add_argument("--commit-steps", type=int, default=50, help="Evaluation replan cadence override.")
    parser.add_argument("--tail-val", action="store_true", help="Evaluate the highest-criticality subset of the selected split.")
    parser.add_argument("--tail-score-path", default="", help="Path to context_tail_scores.npz covering the selected split.")
    parser.add_argument("--tail-min-quantile", type=float, default=0.9, help="Criticality quantile threshold for --tail-val.")
    parser.add_argument("--synthetic-val", action="store_true", help="Evaluate EVT synthetic tail contexts instead of highD split contexts.")
    parser.add_argument("--synthetic-context-path", default="", help="Path to synthetic_tail_contexts.npz for --synthetic-val.")
    parser.add_argument("--expert-plan-dataset", default="", help="Optional adversarial_plan_dataset.npz for searched expert upper-bound comparison.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.synthetic_val and args.tail_val:
        raise ValueError("--synthetic-val and --tail-val are mutually exclusive")
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    apply_rss_config_override(cfg, cfg_path.parent)
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
    expert_eval = bool(str(args.expert_plan_dataset or "").strip())
    synthetic_path: Path | None = None
    expert_dataset_path: Path | None = None
    synthetic_eval = bool(args.synthetic_val)
    if expert_eval:
        expert_dataset_path = _resolve_tail_score_path(str(args.expert_plan_dataset), base)
        raw = _load_npz(expert_dataset_path)
        if "context_states" not in raw or "expert_plan" not in raw:
            raise KeyError(f"{expert_dataset_path} must contain context_states and expert_plan")
        idx = np.arange(raw["context_states"].shape[0], dtype=np.int64)
        synthetic_eval = True
    elif synthetic_eval:
        training = cfg.get("training", {})
        synthetic_value = str(args.synthetic_context_path or training.get("synthetic_context_path", "") or "").strip()
        if not synthetic_value:
            raise ValueError("--synthetic-val requires --synthetic-context-path or training.synthetic_context_path")
        synthetic_path = _resolve_tail_score_path(synthetic_value, base)
        raw = _load_npz(synthetic_path)
        if "context_states" not in raw:
            raise KeyError(f"{synthetic_path} must contain context_states")
        idx = np.arange(raw["context_states"].shape[0], dtype=np.int64)
    else:
        raw = _load_npz(natural_dir / "dataset.npz")
        idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[args.split])[0]
    selection_metadata: dict[str, object] = {
        "context_selection": "expert_plan_dataset" if expert_eval else ("synthetic_tail" if synthetic_eval else "default"),
        "tail_score_path": None,
        "tail_min_quantile": None,
        "tail_threshold": None,
        "tail_candidate_count": int(len(idx)),
        "synthetic_context_path": str(synthetic_path) if synthetic_path is not None else None,
        "expert_plan_dataset": str(expert_dataset_path) if expert_eval else None,
    }
    if expert_eval and args.tail_val:
        raise ValueError("--tail-val is not compatible with --expert-plan-dataset")
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
    recorded_metrics = (
        {}
        if synthetic_eval
        else recorded_future_metrics(raw, idx, max_contexts=int(args.num_contexts), config=cfg)
    )
    if args.king_gradient:
        prior_cfg = copy.deepcopy(cfg)
        prior_cfg.setdefault("policy", {})["enabled"] = False
        prior_cfg.setdefault("paths", {})["policy_checkpoint"] = ""
        sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()
        metrics = evaluate_king_gradient_policy(
            sampler,
            prior_cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
            return_rows=True,
        )
        king_rows = metrics.pop("_rows", [])
        metrics["recorded_future"] = recorded_metrics
        if not synthetic_eval:
            recorded_series = recorded_future_series(raw, idx, max_contexts=int(args.num_contexts), config=cfg)
            prior_rows = [
                {key.removeprefix("prior_"): value for key, value in row.items() if key.startswith("prior_")}
                for row in king_rows
            ]
            guided_rows = [
                {key.removeprefix("king_"): value for key, value in row.items() if key.startswith("king_")}
                for row in king_rows
            ]
            metrics.update(rollout_distance_metrics(recorded_series, "prior", prior_rows))
            metrics.update(rollout_distance_metrics(recorded_series, "king", guided_rows))
    elif args.compare_frozen_prior:
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
        delta_metrics.update(_pair_delta_metrics("guided", "prior", guided_metrics, prior_metrics))
        distance_metrics = {}
        if not synthetic_eval:
            recorded_series = recorded_future_series(raw, idx, max_contexts=int(args.num_contexts), config=cfg)
            distance_metrics = {
                **rollout_distance_metrics(recorded_series, "prior", prior_rows),
                **rollout_distance_metrics(recorded_series, "guided", guided_rows),
            }
        expert_metrics: dict[str, Any] = {}
        if expert_eval:
            expert_metrics = _expert_rows_from_dataset(
                prior_sampler,
                prior_cfg,
                raw,
                idx,
                max_contexts=int(args.num_contexts),
            )
            expert_metrics.pop("_rows", [])
            delta_metrics.update(_pair_delta_metrics("expert", "prior", expert_metrics, prior_metrics))
            delta_metrics.update(_pair_delta_metrics("expert", "guided", expert_metrics, guided_metrics))
        metrics = {
            **_comparison_metrics("prior", prior_metrics),
            **_comparison_metrics("guided", guided_metrics),
            **(_comparison_metrics("expert", expert_metrics) if expert_eval else {}),
            **delta_metrics,
            **distance_metrics,
            "prior_kl_mean": float(guided_metrics.get("prior_kl_mean", float("nan"))),
            "guidance_norm_mean": float(guided_metrics.get("guidance_norm_mean", float("nan"))),
            "recorded_future": recorded_metrics,
            "prior_raw": prior_metrics,
            "guided_raw": guided_metrics,
            **({"expert_raw": expert_metrics} if expert_eval else {}),
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
    selection_name = "expert_plan" if expert_eval else ("synthetic_tail" if synthetic_eval else ("highd_tail" if args.tail_val else "normal"))
    mode = "king_gradient" if args.king_gradient else ("compare" if args.compare_frozen_prior else ("frozen_prior" if args.disable_guidance else "guided"))
    filename = (
        output_dir / f"king_gradient_eval_{selection_name}_summary.json"
        if args.king_gradient
        else output_dir / f"prior_guided_eval_{selection_name}_summary.json"
    )
    save_json(
        {
            "split": args.split,
            "mode": mode,
            "num_contexts": evaluated_context_count,
            **selection_metadata,
            "metrics": metrics,
        },
        filename,
    )


if __name__ == "__main__":
    main()
