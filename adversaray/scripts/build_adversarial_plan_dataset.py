#!/usr/bin/env python3
"""Search fixed adversarial expert plans near the frozen diffusion prior."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner  # noqa: E402
from adversaray.src.config_utils import apply_rss_config_override  # noqa: E402
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler  # noqa: E402
from adversaray.src.prior_guided_train import (  # noqa: E402
    _batch_observation_for_contexts,
    _context,
    _fixed_plan_rollout_worker,
    _load_npz,
    _resolve_paths,
    _sample_weighted_without_replacement,
)
from diffusion.src.data import SPLIT_TO_INDEX  # noqa: E402
from diffusion.src.utils import load_yaml, save_json, setup_logging  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _as_float(result: Any, key: str, default: float = 0.0) -> float:
    return float(result["metrics"].get(key, default))


def _rollout_plans(
    runner: ClosedLoopFollowingRunner,
    config: dict[str, Any],
    context: dict[str, Any],
    plans: np.ndarray,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    plans = np.asarray(plans, dtype=np.float32)
    if plans.ndim == 2:
        plans = plans[None]
    if int(workers) <= 0:
        out: list[dict[str, Any]] = []
        for plan in plans:
            result = runner.rollout_pre_sampled_plan(context, plan)
            out.append(
                {
                    "reward": float(result.reward),
                    "metrics": dict(result.metrics),
                    "trace": result.trace,
                    "num_generated_plans": int(result.num_generated_plans),
                }
            )
        return out
    payload_base = {
        "config": config,
        "schema": runner.sampler.prior.schema,
        "prior_config": runner.sampler.prior.config,
        "history_steps": int(runner.sampler.prior.model.denoiser.cfg.history_steps),
        "horizon_steps": int(runner.sampler.prior.model.denoiser.cfg.horizon_steps),
    }
    payloads = [{**payload_base, "context": context, "plan": plan} for plan in plans]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(_fixed_plan_rollout_worker, payloads))


def _sample_prior_plans(
    sampler: PriorGuidedDiffusionSampler,
    runner: ClosedLoopFollowingRunner,
    context: dict[str, Any],
    *,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    was_enabled = sampler.schedule.enabled
    sampler.set_guidance_enabled(False)
    try:
        batch, prepared = _batch_observation_for_contexts(runner, [context])
        with torch.no_grad():
            sample = sampler.sample_batch(batch, num_samples=int(num_samples), seed=int(seed))
        return sample.raw_actions.detach().cpu().numpy().astype(np.float32), prepared[0]
    finally:
        sampler.set_guidance_enabled(was_enabled)


def _plan_smoothness(plan: np.ndarray) -> float:
    arr = np.asarray(plan, dtype=np.float32)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.mean(np.square(arr[1:] - arr[:-1])))


def _physics_score(row: dict[str, Any]) -> float:
    metrics = row["metrics"]
    return float(
        metrics.get("lead_physics_penalty", 0.0)
        + metrics.get("action_clip_rate", 0.0)
        + metrics.get("jerk_violation_rate", 0.0)
        + metrics.get("speed_negative_rate", 0.0)
    )


def _objective(
    row: dict[str, Any],
    plan: np.ndarray,
    prior_plan: np.ndarray,
    cfg: dict[str, Any],
) -> float:
    risk_reward = float(row["metrics"].get("risk_reward", row["reward"]))
    action_penalty = float(np.sqrt(np.mean(np.square(np.asarray(plan, dtype=np.float32) - prior_plan))))
    smooth_penalty = _plan_smoothness(plan)
    physics_penalty = _physics_score(row)
    return float(
        risk_reward
        - float(cfg.get("lambda_action", 0.05)) * action_penalty
        - float(cfg.get("lambda_smooth", 0.01)) * smooth_penalty
        - float(cfg.get("lambda_physics", 1.0)) * physics_penalty
    )


def _is_physical(row: dict[str, Any], cfg: dict[str, Any], reward_cfg: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    if float(metrics.get("invalid_initial_context", 0.0)) > 0.0:
        return False
    physics_gate = float(cfg.get("physics_gate", reward_cfg.get("physics_gate", 10.0)))
    jerk_gate = float(cfg.get("jerk_violation_gate", reward_cfg.get("jerk_violation_gate", 0.2)))
    clip_gate = float(cfg.get("action_clip_rate_gate", 0.2))
    speed_gate = float(cfg.get("speed_negative_rate_gate", 0.0))
    return bool(
        float(metrics.get("lead_physics_penalty", 0.0)) <= physics_gate
        and float(metrics.get("jerk_violation_rate", 0.0)) <= jerk_gate
        and float(metrics.get("action_clip_rate", 0.0)) <= clip_gate
        and float(metrics.get("speed_negative_rate", 0.0)) <= speed_gate
    )


def _select_best(
    rows: list[dict[str, Any]],
    plans: np.ndarray,
    prior_plan: np.ndarray,
    prior_reward: float,
    cfg: dict[str, Any],
    reward_cfg: dict[str, Any],
) -> tuple[int, float, bool]:
    objectives = np.asarray([_objective(row, plans[pos], prior_plan, cfg) for pos, row in enumerate(rows)], dtype=np.float64)
    legal = np.asarray([_is_physical(row, cfg, reward_cfg) for row in rows], dtype=bool)
    reward_delta = np.asarray([float(row["reward"]) - prior_reward for row in rows], dtype=np.float64)
    usable = legal & np.isfinite(reward_delta)
    if np.any(usable):
        usable_idx = np.where(usable)[0]
        best = int(usable_idx[np.argmax(reward_delta[usable_idx])])
    else:
        best = int(np.nanargmax(objectives))
    success = bool(usable[best] and reward_delta[best] > float(cfg.get("min_reward_delta", 0.0)))
    return best, float(objectives[best]), success


def _random_search(
    sampler: PriorGuidedDiffusionSampler,
    runner: ClosedLoopFollowingRunner,
    config: dict[str, Any],
    context: dict[str, Any],
    prior_plan: np.ndarray,
    prior_reward: float,
    search_cfg: dict[str, Any],
    *,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, Any], float, bool]:
    plans, prepared = _sample_prior_plans(
        sampler,
        runner,
        context,
        num_samples=int(search_cfg.get("num_candidates", 128)),
        seed=int(seed),
    )
    rows = _rollout_plans(runner, config, prepared, plans, workers=workers)
    best, objective, success = _select_best(rows, plans, prior_plan, prior_reward, search_cfg, config.get("reward", {}))
    return plans[best], rows[best], objective, success


def _cem_search(
    runner: ClosedLoopFollowingRunner,
    config: dict[str, Any],
    context: dict[str, Any],
    prior_plan: np.ndarray,
    prior_reward: float,
    search_cfg: dict[str, Any],
    *,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, Any], float, bool]:
    rng = np.random.default_rng(int(seed))
    num_candidates = int(search_cfg.get("num_candidates", 128))
    iters = int(search_cfg.get("cem_iters", 5))
    elite_count = max(1, int(round(num_candidates * float(search_cfg.get("elite_frac", 0.1)))))
    mean = np.asarray(prior_plan, dtype=np.float32).copy()
    std = np.full_like(mean, float(search_cfg.get("action_noise_std", 0.5)), dtype=np.float32)
    best_plan = mean.copy()
    best_row = _rollout_plans(runner, config, context, best_plan, workers=workers)[0]
    best_objective = _objective(best_row, best_plan, prior_plan, search_cfg)
    best_delta = float(best_row["reward"]) - prior_reward
    best_success = _is_physical(best_row, search_cfg, config.get("reward", {})) and (
        best_delta > float(search_cfg.get("min_reward_delta", 0.0))
    )

    for _ in range(max(iters, 1)):
        noise = rng.normal(0.0, 1.0, size=(num_candidates, *mean.shape)).astype(np.float32)
        plans = mean[None] + noise * std[None]
        rows = _rollout_plans(runner, config, context, plans, workers=workers)
        objectives = np.asarray([_objective(row, plans[pos], prior_plan, search_cfg) for pos, row in enumerate(rows)], dtype=np.float64)
        order = np.argsort(-objectives, kind="mergesort")
        elites = plans[order[:elite_count]]
        mean = np.mean(elites, axis=0).astype(np.float32)
        std = np.maximum(np.std(elites, axis=0).astype(np.float32), 1e-3)
        candidate_best, objective, success = _select_best(rows, plans, prior_plan, prior_reward, search_cfg, config.get("reward", {}))
        candidate_delta = float(rows[candidate_best]["reward"]) - prior_reward
        should_replace = False
        if success and (not best_success or candidate_delta > best_delta):
            should_replace = True
        elif not best_success and objective > best_objective:
            should_replace = True
        if should_replace:
            best_plan = plans[candidate_best].astype(np.float32)
            best_row = rows[candidate_best]
            best_objective = objective
            best_delta = candidate_delta
            best_success = success
    return best_plan, best_row, float(best_objective), bool(best_success)


def _load_contexts(config: dict[str, Any], config_dir: Path, natural_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    training = config.get("training", {})
    search_cfg = config.get("plan_search", {})
    synthetic_value = str(training.get("synthetic_context_path", "") or "").strip()
    if not synthetic_value:
        raise ValueError("training.synthetic_context_path is required for adversarial plan search")
    synthetic = _load_npz(_resolve(synthetic_value, config_dir))
    if "context_states" not in synthetic:
        raise KeyError("Synthetic context file must contain context_states")
    if not bool(search_cfg.get("include_highd_tail", False)):
        return synthetic

    split = str(search_cfg.get("highd_split", training.get("split", "train")))
    split_idx = np.where(natural_raw["split_index"] == SPLIT_TO_INDEX[split])[0]
    tail_path_value = str(training.get("tail_score_path", "") or "").strip()
    if not tail_path_value:
        raise ValueError("plan_search.include_highd_tail=true requires training.tail_score_path")
    tail = _load_npz(_resolve(tail_path_value, config_dir))
    if "dataset_index" not in tail or "criticality_score" not in tail:
        raise KeyError("Tail score file must contain dataset_index and criticality_score")
    score_idx = np.asarray(tail["dataset_index"], dtype=np.int64)
    score = np.asarray(tail["criticality_score"], dtype=np.float64)
    mask = np.isin(score_idx, split_idx) & np.isfinite(score)
    threshold = float(np.quantile(score[mask], float(search_cfg.get("highd_tail_min_quantile", training.get("tail_min_quantile", 0.9)))))
    mask &= score >= threshold
    weight_key = "tail_sampling_weight" if "tail_sampling_weight" in tail else "tail_weight"
    weights = np.asarray(tail[weight_key][mask], dtype=np.float64) if weight_key in tail else score[mask]
    rng = np.random.default_rng(int(training.get("seed", 42)))
    count = int(search_cfg.get("highd_tail_contexts", 0))
    selected = _sample_weighted_without_replacement(score_idx[mask], weights, size=count, rng=rng)
    highd = {key: value[selected] for key, value in natural_raw.items() if isinstance(value, np.ndarray) and value.shape[:1] == natural_raw["context_states"].shape[:1]}
    return _concat_context_sets(synthetic, highd)


def _concat_context_sets(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(first)
    n_first = int(first["context_states"].shape[0])
    n_second = int(second["context_states"].shape[0])
    keys = set(first) | set(second)
    for key in keys:
        if key in first and key in second and first[key].shape[1:] == second[key].shape[1:]:
            out[key] = np.concatenate([first[key], second[key]], axis=0)
        elif key == "source_type":
            a = np.asarray(first.get(key, np.full(n_first, "synthetic", dtype="U16"))).astype("U32")
            b = np.asarray(second.get(key, np.full(n_second, "highd_tail", dtype="U16"))).astype("U32")
            out[key] = np.concatenate([a, b], axis=0)
    if "source_type" not in out:
        out["source_type"] = np.concatenate(
            [np.full(n_first, "synthetic", dtype="U16"), np.full(n_second, "highd_tail", dtype="U16")],
            axis=0,
        )
    return out


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    successful = np.asarray([row["successful"] for row in rows], dtype=np.float64)
    expert_physics_bad = np.asarray([row["expert_physics_violation"] for row in rows], dtype=np.float64)
    out = {
        "num_contexts": int(len(rows)),
        "num_successful_contexts": int(np.sum(successful)),
        "success_rate": float(np.mean(successful)) if rows else 0.0,
        "physics_violation_rate": float(np.mean(expert_physics_bad)) if rows else 0.0,
    }
    for key, src in (
        ("mean_prior_reward", "prior_reward"),
        ("mean_expert_reward", "expert_reward"),
        ("mean_reward_delta", "reward_delta"),
        ("action_clip_rate", "expert_action_clip_rate"),
        ("jerk_violation_rate", "expert_jerk_violation_rate"),
        ("speed_negative_rate", "expert_speed_negative_rate"),
        ("prior_min_gap_mean", "prior_min_gap"),
        ("prior_min_ttc_mean", "prior_min_ttc"),
        ("prior_min_rss_margin_mean", "prior_min_rss_margin"),
    ):
        values = np.asarray([row[src] for row in rows], dtype=np.float64)
        out[key] = float(np.nanmean(values)) if values.size else float("nan")
    for key, src in (
        ("p05_min_gap_expert", "expert_min_gap"),
        ("p05_min_ttc_expert", "expert_min_ttc"),
        ("p05_min_rss_margin_expert", "expert_min_rss_margin"),
        ("prior_min_gap_p05", "prior_min_gap"),
        ("prior_min_ttc_p05", "prior_min_ttc"),
        ("prior_min_rss_margin_p05", "prior_min_rss_margin"),
    ):
        values = np.asarray([row[src] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(np.percentile(values, 5.0)) if values.size else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--method", choices=("random", "cem"), default="", help="Override plan_search.method.")
    parser.add_argument("--num-contexts", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--num-candidates", type=int, default=0, help="Override plan_search.num_candidates.")
    parser.add_argument("--cem-iters", type=int, default=0, help="Override plan_search.cem_iters.")
    parser.add_argument("--output", default="", help="Optional output dataset path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    config = load_yaml(cfg_path)
    apply_rss_config_override(config, cfg_path.parent)
    natural_dir, _diffusion_ckpt, output_dir = _resolve_paths(config, cfg_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_npz(natural_dir / "dataset.npz")
    contexts_raw = _load_contexts(config, cfg_path.parent, raw)

    search_cfg = copy.deepcopy(config.get("plan_search", {}))
    if args.method:
        search_cfg["method"] = args.method
    if args.num_candidates > 0:
        search_cfg["num_candidates"] = int(args.num_candidates)
    if args.cem_iters > 0:
        search_cfg["cem_iters"] = int(args.cem_iters)
    max_contexts = int(args.num_contexts or search_cfg.get("max_contexts", contexts_raw["context_states"].shape[0]))
    max_contexts = min(max_contexts, int(contexts_raw["context_states"].shape[0]))
    workers = max(int(search_cfg.get("rollout_workers", 0)), 0)
    seed = int(config.get("training", {}).get("seed", 42))

    prior_cfg = copy.deepcopy(config)
    prior_cfg.setdefault("policy", {})["enabled"] = False
    sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=cfg_path.parent).eval()
    runner = ClosedLoopFollowingRunner(sampler, prior_cfg)

    output: dict[str, list[Any]] = {
        "context_states": [],
        "ego_length": [],
        "adv_length": [],
        "source_type": [],
        "prior_plan": [],
        "expert_plan": [],
        "prior_reward": [],
        "expert_reward": [],
        "reward_delta": [],
        "expert_min_gap": [],
        "expert_min_ttc": [],
        "expert_min_rss_margin": [],
        "expert_action_clip_rate": [],
        "expert_jerk_violation_rate": [],
        "expert_speed_negative_rate": [],
        "successful": [],
        "prior_seed": [],
        "prior_min_gap": [],
        "prior_min_ttc": [],
        "prior_min_rss_margin": [],
        "search_objective": [],
    }
    summary_rows: list[dict[str, float]] = []
    method = str(search_cfg.get("method", "cem")).lower()
    if method not in {"random", "cem"}:
        raise ValueError(f"Unsupported plan_search.method={method!r}; expected random or cem")

    for pos in range(max_contexts):
        ctx = _context(contexts_raw, pos)
        prior_seed = seed + pos
        prior_plans, prepared = _sample_prior_plans(sampler, runner, ctx, num_samples=1, seed=prior_seed)
        prior_plan = prior_plans[0]
        prior_row = _rollout_plans(runner, prior_cfg, prepared, prior_plan, workers=workers)[0]
        prior_reward = float(prior_row["reward"])

        if method == "random":
            expert_plan, expert_row, objective, successful = _random_search(
                sampler,
                runner,
                prior_cfg,
                prepared,
                prior_plan,
                prior_reward,
                search_cfg,
                seed=prior_seed + 100000,
                workers=workers,
            )
        else:
            expert_plan, expert_row, objective, successful = _cem_search(
                runner,
                prior_cfg,
                prepared,
                prior_plan,
                prior_reward,
                search_cfg,
                seed=prior_seed + 100000,
                workers=workers,
            )

        expert_reward = float(expert_row["reward"])
        source = str(prepared.get("source_type", "synthetic"))
        output["context_states"].append(np.asarray(prepared["raw_context_states"], dtype=np.float32))
        output["ego_length"].append(float(prepared.get("ego_length", 4.8)))
        output["adv_length"].append(float(prepared.get("adv_length", 4.8)))
        output["source_type"].append(source)
        output["prior_plan"].append(prior_plan.astype(np.float32))
        output["expert_plan"].append(np.asarray(expert_plan, dtype=np.float32))
        output["prior_reward"].append(prior_reward)
        output["expert_reward"].append(expert_reward)
        output["reward_delta"].append(expert_reward - prior_reward)
        output["expert_min_gap"].append(_as_float(expert_row, "min_gap", np.nan))
        output["expert_min_ttc"].append(_as_float(expert_row, "min_ttc", np.nan))
        output["expert_min_rss_margin"].append(_as_float(expert_row, "min_rss_margin", np.nan))
        output["expert_action_clip_rate"].append(_as_float(expert_row, "action_clip_rate", np.nan))
        output["expert_jerk_violation_rate"].append(_as_float(expert_row, "jerk_violation_rate", np.nan))
        output["expert_speed_negative_rate"].append(_as_float(expert_row, "speed_negative_rate", np.nan))
        output["successful"].append(float(successful))
        output["prior_seed"].append(float(prior_seed))
        output["prior_min_gap"].append(_as_float(prior_row, "min_gap", np.nan))
        output["prior_min_ttc"].append(_as_float(prior_row, "min_ttc", np.nan))
        output["prior_min_rss_margin"].append(_as_float(prior_row, "min_rss_margin", np.nan))
        output["search_objective"].append(float(objective))

        expert_physical = _is_physical(expert_row, search_cfg, prior_cfg.get("reward", {}))
        row = {
            "successful": float(successful),
            "expert_physics_violation": float(not expert_physical),
            "prior_reward": prior_reward,
            "expert_reward": expert_reward,
            "reward_delta": expert_reward - prior_reward,
            "expert_min_gap": _as_float(expert_row, "min_gap", np.nan),
            "expert_min_ttc": _as_float(expert_row, "min_ttc", np.nan),
            "expert_min_rss_margin": _as_float(expert_row, "min_rss_margin", np.nan),
            "expert_action_clip_rate": _as_float(expert_row, "action_clip_rate", np.nan),
            "expert_jerk_violation_rate": _as_float(expert_row, "jerk_violation_rate", np.nan),
            "expert_speed_negative_rate": _as_float(expert_row, "speed_negative_rate", np.nan),
            "prior_min_gap": _as_float(prior_row, "min_gap", np.nan),
            "prior_min_ttc": _as_float(prior_row, "min_ttc", np.nan),
            "prior_min_rss_margin": _as_float(prior_row, "min_rss_margin", np.nan),
        }
        summary_rows.append(row)
        logger.info(
            "context=%d/%d source=%s delta=%.4f success=%s",
            pos + 1,
            max_contexts,
            source,
            expert_reward - prior_reward,
            successful,
        )

    out_path = Path(args.output) if args.output else output_dir / "adversarial_plan_dataset.npz"
    if not out_path.is_absolute():
        out_path = (cfg_path.parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        context_states=np.asarray(output["context_states"], dtype=np.float32),
        ego_length=np.asarray(output["ego_length"], dtype=np.float32),
        adv_length=np.asarray(output["adv_length"], dtype=np.float32),
        source_type=np.asarray(output["source_type"]).astype("U32"),
        prior_plan=np.asarray(output["prior_plan"], dtype=np.float32),
        expert_plan=np.asarray(output["expert_plan"], dtype=np.float32),
        prior_reward=np.asarray(output["prior_reward"], dtype=np.float32),
        expert_reward=np.asarray(output["expert_reward"], dtype=np.float32),
        reward_delta=np.asarray(output["reward_delta"], dtype=np.float32),
        expert_min_gap=np.asarray(output["expert_min_gap"], dtype=np.float32),
        expert_min_ttc=np.asarray(output["expert_min_ttc"], dtype=np.float32),
        expert_min_rss_margin=np.asarray(output["expert_min_rss_margin"], dtype=np.float32),
        expert_action_clip_rate=np.asarray(output["expert_action_clip_rate"], dtype=np.float32),
        expert_jerk_violation_rate=np.asarray(output["expert_jerk_violation_rate"], dtype=np.float32),
        expert_speed_negative_rate=np.asarray(output["expert_speed_negative_rate"], dtype=np.float32),
        successful=np.asarray(output["successful"], dtype=np.float32),
        prior_seed=np.asarray(output["prior_seed"], dtype=np.int64),
        prior_min_gap=np.asarray(output["prior_min_gap"], dtype=np.float32),
        prior_min_ttc=np.asarray(output["prior_min_ttc"], dtype=np.float32),
        prior_min_rss_margin=np.asarray(output["prior_min_rss_margin"], dtype=np.float32),
        search_objective=np.asarray(output["search_objective"], dtype=np.float32),
    )
    summary = {
        **_summary(summary_rows),
        "dataset_path": str(out_path),
        "method": method,
        "num_candidates": int(search_cfg.get("num_candidates", 128)),
        "cem_iters": int(search_cfg.get("cem_iters", 5)),
    }
    save_json(summary, out_path.with_name("adversarial_plan_dataset_summary.json"))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
