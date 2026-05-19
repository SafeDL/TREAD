"""REINFORCE training loop for prior-regularized guided diffusion."""
from __future__ import annotations

import csv
import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_json, save_json, select_device, set_seed

from .closed_loop_runner import ClosedLoopFollowingRunner, RolloutResult
from .diffusion_adapter import DiffusionPriorAdapter
from .guidance_policy import GuidancePolicy, GuidancePolicyConfig
from .prior_guided_sampler import PriorGuidedDiffusionSampler, PriorGuidedSampleResult
from .risk_utils import actions_to_accel_jerk

logger = logging.getLogger(__name__)

REWARD_COMPONENT_KEYS = (
    "risk_reward",
    "collision_reward",
    "ttc_reward",
    "gap_reward",
    "rss_reward",
    "hard_brake_reward",
    "near_collision_reward",
    "invalid_collision_reward",
    "physics_penalty_reward",
    "lead_physics_penalty",
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _resolve_paths(config: dict[str, Any], config_dir: str | Path | None) -> tuple[Path, Path, Path]:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths = config.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    diffusion_ckpt = Path(paths.get("diffusion_checkpoint", "checkpoints/best_noise_mse.pt"))
    if not diffusion_ckpt.is_absolute():
        diffusion_ckpt = (base / diffusion_ckpt).resolve()
        if not diffusion_ckpt.exists():
            diffusion_ckpt = (natural_dir / paths.get("diffusion_checkpoint", "checkpoints/best_noise_mse.pt")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()
    config["_runtime"] = {
        "config_dir": str(base),
        "natural_dataset_dir": str(natural_dir),
        "diffusion_checkpoint": str(diffusion_ckpt),
        "output_dir": str(output_dir),
        "highd_events_csv": str((base / paths.get("highd_events_csv", "../../../data/highd_events/events.csv")).resolve()),
        "highd_raw_dir": str((base / paths.get("highd_raw_dir", "../../../highD_dataset/Matlab/data")).resolve()),
        "highd_config": str(
            (base / paths.get("highd_config", "../../../process_highD/scripts/configs/highd_default.yaml")).resolve()
        ),
    }
    return natural_dir, diffusion_ckpt, output_dir


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _make_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard unavailable: %s", exc)
        return None
    return SummaryWriter(str(output_dir / "runs"))


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    ego_lengths = raw.get("ego_length")
    adv_lengths = raw.get("adv_length")
    context = {
        "raw_context_states": raw["context_states"][idx],
        "ego_length": float(ego_lengths[idx]) if ego_lengths is not None else 4.8,
        "adv_length": float(adv_lengths[idx]) if adv_lengths is not None else 4.8,
    }
    for key in ("recording_id", "event_id", "anchor_frame"):
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    return context


def _save_checkpoint(
    path: Path,
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    schema: dict[str, Any],
    epoch: int,
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": sampler.policy.state_dict(),
            "config": config,
            "schema": schema,
            "epoch": int(epoch),
            "summary": summary,
        },
        path,
    )


def _bucket(values: np.ndarray, bins: int = 4) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or np.allclose(finite.min(), finite.max()):
        return np.zeros(arr.shape, dtype=np.int64)
    edges = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(arr, edges, right=False).astype(np.int64)


def _sample_context_indices(
    raw: dict[str, np.ndarray],
    train_idx: np.ndarray,
    *,
    max_train_contexts: int,
    rng: np.random.Generator,
    mode: str,
    training: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> np.ndarray:
    if max_train_contexts <= 0 or len(train_idx) <= max_train_contexts:
        return np.asarray(train_idx, dtype=np.int64)
    size = min(int(max_train_contexts), len(train_idx))
    if mode == "tail_mixture":
        selected = _sample_tail_mixture_indices(
            raw,
            train_idx,
            size=size,
            rng=rng,
            training=training or {},
            config_dir=config_dir,
        )
        if selected.size > 0:
            return np.sort(selected.astype(np.int64))
        if bool((training or {}).get("require_tail_scores", False)):
            raise RuntimeError(
                "context_sampling='tail_mixture' requires usable tail scores, but none were found for the current split. "
                "Build context_tail_scores.npz or set training.require_tail_scores=false for development fallback."
            )
        logger.warning("tail_mixture requested but no usable tail scores were found; falling back to stratified sampling")
        mode = "stratified"
    if mode != "stratified" or "context_states" not in raw:
        return np.sort(rng.choice(train_idx, size=size, replace=False)).astype(np.int64)

    context = np.asarray(raw["context_states"][train_idx], dtype=np.float32)
    ego_length = np.asarray(raw["ego_length"][train_idx] if "ego_length" in raw else np.full(len(train_idx), 4.8), dtype=np.float32)
    adv_length = np.asarray(raw["adv_length"][train_idx] if "adv_length" in raw else np.full(len(train_idx), 4.8), dtype=np.float32)
    initial_gap = context[:, -1, 1, 0] - context[:, -1, 0, 0] - 0.5 * (ego_length + adv_length)
    closing_speed = context[:, -1, 0, 2] - context[:, -1, 1, 2]
    gap_bucket = _bucket(initial_gap)
    closing_bucket = _bucket(closing_speed)
    recording = raw["recording_id"][train_idx] if "recording_id" in raw else np.full(len(train_idx), -1)
    event = raw["event_id"][train_idx] if "event_id" in raw else np.full(len(train_idx), -1)

    groups: dict[tuple[str, str, int, int], list[int]] = {}
    for pos, idx in enumerate(train_idx):
        key = (str(recording[pos]), str(event[pos]), int(gap_bucket[pos]), int(closing_bucket[pos]))
        groups.setdefault(key, []).append(int(idx))
    if len(groups) <= 1:
        return np.sort(rng.choice(train_idx, size=size, replace=False)).astype(np.int64)

    queues = [rng.permutation(np.asarray(items, dtype=np.int64)).tolist() for items in groups.values()]
    order = rng.permutation(len(queues)).tolist()
    selected: list[int] = []
    while len(selected) < size and order:
        next_order: list[int] = []
        for group_idx in order:
            if queues[group_idx]:
                selected.append(int(queues[group_idx].pop()))
                if len(selected) >= size:
                    break
            if queues[group_idx]:
                next_order.append(group_idx)
        order = rng.permutation(next_order).tolist() if next_order else []
    return np.sort(np.asarray(selected, dtype=np.int64))


def _resolve_training_path(path_value: str, config_dir: str | Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    return (base / path).resolve()


def _sample_weighted_without_replacement(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if size <= 0 or len(values) == 0:
        return np.asarray([], dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(values), dtype=np.float64)
    probabilities = weights / float(weights.sum())
    return rng.choice(values, size=min(size, len(values)), replace=False, p=probabilities).astype(np.int64)


def _sample_tail_mixture_indices(
    raw: dict[str, np.ndarray],
    train_idx: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
    training: dict[str, Any],
    config_dir: str | Path | None,
) -> np.ndarray:
    tail_path_value = str(training.get("tail_score_path", "") or "")
    if not tail_path_value:
        return np.asarray([], dtype=np.int64)
    tail_path = _resolve_training_path(tail_path_value, config_dir)
    if not tail_path.exists():
        return np.asarray([], dtype=np.int64)
    data = np.load(tail_path, allow_pickle=True)
    weight_key = "tail_sampling_weight" if "tail_sampling_weight" in data else "tail_weight"
    if "dataset_index" not in data or weight_key not in data:
        return np.asarray([], dtype=np.int64)
    score_idx = np.asarray(data["dataset_index"], dtype=np.int64)
    train_set = set(int(x) for x in np.asarray(train_idx, dtype=np.int64))
    mask = np.asarray([int(x) in train_set for x in score_idx], dtype=bool)
    if not np.any(mask):
        return np.asarray([], dtype=np.int64)
    available_idx = score_idx[mask]
    weights = np.asarray(data[weight_key][mask], dtype=np.float64)
    score = np.asarray(data["criticality_score"][mask] if "criticality_score" in data else weights, dtype=np.float64)
    min_quantile = float(training.get("tail_min_quantile", 0.9))
    threshold = float(np.quantile(score[np.isfinite(score)], min_quantile)) if np.any(np.isfinite(score)) else float("-inf")
    tail_mask = score >= threshold
    tail_candidates = available_idx[tail_mask]
    tail_weights = weights[tail_mask]
    total_fraction = (
        float(training.get("tail_fraction", 0.6))
        + float(training.get("random_fraction", 0.2))
        + float(training.get("stratified_fraction", 0.2))
    )
    if total_fraction <= 0.0:
        total_fraction = 1.0
    tail_count = int(round(size * float(training.get("tail_fraction", 0.6)) / total_fraction))
    random_count = int(round(size * float(training.get("random_fraction", 0.2)) / total_fraction))
    strat_count = max(0, size - tail_count - random_count)
    temperature = max(float(training.get("tail_weight_temperature", 1.0)), 1e-6)
    tail_weights = np.power(np.maximum(tail_weights, 0.0), 1.0 / temperature)
    selected: list[int] = []
    selected.extend(_sample_weighted_without_replacement(tail_candidates, tail_weights, size=tail_count, rng=rng).tolist())
    remaining = np.asarray([idx for idx in train_idx if int(idx) not in set(selected)], dtype=np.int64)
    if random_count > 0 and len(remaining) > 0:
        selected.extend(rng.choice(remaining, size=min(random_count, len(remaining)), replace=False).astype(np.int64).tolist())
    remaining = np.asarray([idx for idx in train_idx if int(idx) not in set(selected)], dtype=np.int64)
    if strat_count > 0 and len(remaining) > 0:
        stratified = _sample_context_indices(
            raw,
            remaining,
            max_train_contexts=min(strat_count, len(remaining)),
            rng=rng,
            mode="stratified",
        )
        selected.extend(stratified.astype(np.int64).tolist())
    if len(selected) < size:
        remaining = np.asarray([idx for idx in train_idx if int(idx) not in set(selected)], dtype=np.int64)
        if len(remaining) > 0:
            selected.extend(rng.choice(remaining, size=min(size - len(selected), len(remaining)), replace=False).astype(np.int64).tolist())
    return np.asarray(selected[:size], dtype=np.int64)


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"reward_mean": float("nan")}
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and key not in keys:
                keys.append(key)
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_p05"] = float(np.percentile(values, 5.0))
        out[f"{key}_p95"] = float(np.percentile(values, 95.0))
    for key in ("collision", "collision_valid", "invalid_collision", "near_collision", "hard_brake", "invalid_initial_context"):
        mean_key = f"{key}_mean"
        if mean_key in out:
            out[f"{key}_rate"] = out[mean_key]
    return out


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n = max(len(x), len(y))
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    xp = np.interp(q, (np.arange(len(x), dtype=np.float64) + 0.5) / len(x), x)
    yp = np.interp(q, (np.arange(len(y), dtype=np.float64) + 0.5) / len(y), y)
    return float(np.mean(np.abs(xp - yp)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _series_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p05": float(np.percentile(arr, 5.0)),
        f"{prefix}_p95": float(np.percentile(arr, 95.0)),
    }


def _schema_for_recorded_metrics(config: dict[str, Any]) -> dict[str, Any]:
    runtime_dir = config.get("_runtime", {}).get("natural_dataset_dir")
    if runtime_dir:
        schema_path = Path(runtime_dir) / "feature_schema.json"
        if schema_path.exists():
            return load_json(schema_path)
    return {
        "action_representation": config.get("action", {}).get("representation", "acceleration"),
        "dt": float(config.get("env", {}).get("dt", 1.0 / 25.0)),
    }


def recorded_future_series(
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    if "future_states" not in raw:
        return {}
    idx = np.asarray(indices[:max_contexts], dtype=np.int64)
    if idx.size == 0:
        return {}
    future = np.asarray(raw["future_states"][idx], dtype=np.float32)
    ego = future[:, :, 0]
    lead = future[:, :, 1]
    ego_length = np.asarray(raw["ego_length"][idx] if "ego_length" in raw else np.full(idx.size, 4.8), dtype=np.float32)
    lead_length = np.asarray(raw["adv_length"][idx] if "adv_length" in raw else np.full(idx.size, 4.8), dtype=np.float32)
    gap = lead[:, :, 0] - ego[:, :, 0] - 0.5 * (ego_length[:, None] + lead_length[:, None])
    closing = ego[:, :, 2] - lead[:, :, 2]
    ttc = np.where(closing > 1e-6, gap / np.maximum(closing, 1e-6), 1000.0)
    if "actions" in raw and "context_states" in raw:
        schema = _schema_for_recorded_metrics(config)
        lead_accel, lead_jerk = actions_to_accel_jerk(raw["actions"][idx], raw["context_states"][idx], schema, config)
    else:
        dt = float(config.get("env", {}).get("dt", 1.0 / 25.0))
        lead_accel = lead[:, :, 4]
        lead_jerk = np.diff(lead_accel, axis=1) / max(dt, 1e-6) if lead_accel.shape[1] > 1 else np.zeros_like(lead_accel)
    return {
        "real_gap": gap.reshape(-1),
        "real_min_gap": np.min(gap, axis=1),
        "real_final_gap": gap[:, -1],
        "real_min_ttc": np.min(np.clip(ttc, 0.0, 1000.0), axis=1),
        "real_lead_speed": lead[:, :, 2].reshape(-1),
        "real_lead_accel": lead_accel.reshape(-1),
        "real_lead_jerk_abs": np.abs(lead_jerk).reshape(-1),
    }


def recorded_future_metrics(
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
    config: dict[str, Any],
) -> dict[str, float]:
    series = recorded_future_series(raw, indices, max_contexts=max_contexts, config=config)
    if not series:
        return {"available": 0.0}
    near_gap = float(config.get("reward", {}).get("near_collision_gap", 2.0))
    out = {
        "available": 1.0,
        "num_contexts": float(len(series["real_min_gap"])),
        "real_collision_rate": float(np.mean(series["real_gap"] <= 0.0)),
        "real_near_collision_rate": float(np.mean(series["real_gap"] < near_gap)),
    }
    for key in ("real_min_gap", "real_final_gap", "real_min_ttc", "real_lead_speed", "real_lead_accel", "real_lead_jerk_abs"):
        out.update(_series_summary(series[key], key))
    return out


def rollout_distance_metrics(recorded: dict[str, np.ndarray], prefix: str, rows: list[dict[str, float]]) -> dict[str, float]:
    if not recorded or not rows:
        return {}
    gen_min_gap = np.asarray([row.get("min_gap", np.nan) for row in rows], dtype=np.float64)
    gen_final_gap = np.asarray([row.get("final_gap", row.get("min_gap", np.nan)) for row in rows], dtype=np.float64)
    gen_min_ttc = np.asarray([row.get("min_ttc", np.nan) for row in rows], dtype=np.float64)
    trace_steps = [step for row in rows for step in row.get("trace", []) if isinstance(step, dict)]
    gen_lead_accel = np.asarray([step.get("lead_accel", np.nan) for step in trace_steps], dtype=np.float64)
    gen_lead_jerk_abs = np.asarray([abs(float(step.get("lead_jerk", np.nan))) for step in trace_steps], dtype=np.float64)
    pairs = {
        "min_gap": ("real_min_gap", gen_min_gap, "wasserstein"),
        "final_gap": ("real_final_gap", gen_final_gap, "wasserstein"),
        "lead_accel": ("real_lead_accel", gen_lead_accel, "wasserstein"),
        "lead_jerk_abs": ("real_lead_jerk_abs", gen_lead_jerk_abs, "wasserstein"),
        "min_ttc": ("real_min_ttc", gen_min_ttc, "ks"),
    }
    out: dict[str, float] = {}
    for name, (real_key, generated, metric) in pairs.items():
        real = recorded.get(real_key, np.asarray([]))
        value = _ks_statistic(real, generated) if metric == "ks" else _wasserstein_1d(real, generated)
        out[f"real_vs_{prefix}_{name}_{metric}"] = value
    return out


@torch.no_grad()
def evaluate_prior_guided_policy(
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
    seed: int,
    return_rows: bool = False,
) -> dict[str, float]:
    was_training = sampler.policy.training
    sampler.eval()
    runner = ClosedLoopFollowingRunner(sampler, config)
    rows: list[dict[str, float]] = []
    for offset, idx in enumerate(indices[:max_contexts]):
        result = runner.rollout(_context(raw, int(idx)), seed=int(seed) + offset)
        rows.append(
            {
                "reward": result.reward,
                "prior_kl": float(result.prior_kl_sum.detach().cpu()),
                "guidance_norm": float(result.guidance_norm_sum.detach().cpu()),
                "trace": result.trace,
                **result.metrics,
            }
        )
    sampler.train(was_training)
    summary = _summarize_rows(rows)
    if return_rows:
        summary["_rows"] = rows  # type: ignore[assignment]
    return summary


def _rollout_row(result: Any) -> dict[str, float]:
    return {
        "reward": float(result.reward),
        "prior_kl": float(result.prior_kl_sum.detach().cpu()),
        "prior_kl_per_plan": float(result.prior_kl_per_plan.detach().cpu()),
        "prior_kl_per_step": float(result.prior_kl_per_step.detach().cpu()),
        "guidance_norm": float(result.guidance_norm_sum.detach().cpu()),
        "guidance_norm_per_plan": float(result.guidance_norm_per_plan.detach().cpu()),
        **result.metrics,
    }


def _paired_row(prior_result: Any, guided_result: Any) -> dict[str, float]:
    prior = _rollout_row(prior_result)
    guided = _rollout_row(guided_result)
    row: dict[str, float] = {}
    for key, value in prior.items():
        if isinstance(value, (int, float, np.floating)):
            row[f"prior_{key}"] = float(value)
    for key, value in guided.items():
        if isinstance(value, (int, float, np.floating)):
            row[f"guided_{key}"] = float(value)
    delta_keys = (
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
    for key in delta_keys:
        row[f"{key}_delta"] = float(guided.get(key, 0.0) - prior.get(key, 0.0))
    row["reward_delta"] = float(guided["reward"] - prior["reward"])
    row["rss_risk_improvement"] = float(guided.get("rss_reward", 0.0) - prior.get("rss_reward", 0.0))
    row["rss_margin_degradation"] = float(prior.get("min_rss_margin", 0.0) - guided.get("min_rss_margin", 0.0))
    row["invalid_initial_context"] = float(max(prior.get("invalid_initial_context", 0.0), guided.get("invalid_initial_context", 0.0)))
    return row


def _batch_observation_for_contexts(
    runner: ClosedLoopFollowingRunner,
    contexts: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    observations: list[dict[str, np.ndarray]] = []
    prepared_contexts: list[dict[str, Any]] = []
    ego_lengths: list[float] = []
    adv_lengths: list[float] = []
    for ctx in contexts:
        raw_context = np.asarray(ctx["raw_context_states"], dtype=np.float32).copy()
        raw_context[:, :, 1] = 0.0
        ego_length = float(ctx.get("ego_length", 4.8))
        lead_length = float(ctx.get("adv_length", ctx.get("lead_length", 4.8)))
        rebuilt = runner._maybe_reconstruct_highd_context(ctx, ego_length, lead_length)
        if rebuilt is not None:
            raw_context, ego_length, lead_length = rebuilt
            raw_context[:, :, 1] = 0.0
        initial_gap = float(raw_context[-1, 1, 0] - raw_context[-1, 0, 0] - 0.5 * (ego_length + lead_length))
        if initial_gap <= runner.initial_gap_min and not runner.skip_invalid_initial_context:
            raw_context[-1, 1, 0] = raw_context[-1, 0, 0] + 0.5 * (ego_length + lead_length) + runner.initial_gap_min
        history_world: deque[np.ndarray] = deque(maxlen=runner.history_steps)
        for item in raw_context[-runner.history_steps :]:
            v = np.asarray(item, dtype=np.float32).copy()
            v[:, 1] = 0.0
            history_world.append(v)
        observations.append(runner._build_observation(history_world, ego_length, lead_length))
        prepared = dict(ctx)
        prepared["raw_context_states"] = raw_context
        prepared["ego_length"] = ego_length
        prepared["adv_length"] = lead_length
        prepared_contexts.append(prepared)
        ego_lengths.append(ego_length)
        adv_lengths.append(lead_length)
    batch = {
        "context_states": torch.from_numpy(np.stack([obs["context_states"] for obs in observations], axis=0)).float(),
        "context_features": torch.from_numpy(np.stack([obs["context_features"] for obs in observations], axis=0)).float(),
        "relative_history": torch.from_numpy(np.stack([obs["relative_history"] for obs in observations], axis=0)).float(),
        "ego_length": torch.tensor(ego_lengths, dtype=torch.float32),
        "adv_length": torch.tensor(adv_lengths, dtype=torch.float32),
    }
    return batch, prepared_contexts


def sample_batch_plans(
    sampler: PriorGuidedDiffusionSampler,
    runner: ClosedLoopFollowingRunner,
    contexts: list[dict[str, Any]],
    seeds: list[int] | None,
) -> tuple[PriorGuidedSampleResult, np.ndarray, list[dict[str, Any]]]:
    batch, prepared_contexts = _batch_observation_for_contexts(runner, contexts)
    sample = sampler.sample_batch(batch, seed=seeds)
    plans = sample.raw_actions.detach().cpu().numpy().astype(np.float32)
    return sample, plans, prepared_contexts


def _rollout_sampled_plan(
    runner: ClosedLoopFollowingRunner,
    context: dict[str, Any],
    plans: np.ndarray,
    sample: PriorGuidedSampleResult,
    pos: int,
) -> Any:
    return runner.rollout_pre_sampled_plan(
        context,
        plans[pos],
        log_prob_sum=sample.trajectory_log_prob[pos],
        prior_kl_sum=sample.prior_kl[pos],
        guidance_norm_sum=sample.guidance_norm[pos],
    )


def _worker_runner(
    config: dict[str, Any],
    schema: dict[str, Any],
    prior_config: dict[str, Any],
    *,
    history_steps: int,
    horizon_steps: int,
) -> ClosedLoopFollowingRunner:
    prior = SimpleNamespace(
        device=torch.device("cpu"),
        schema=schema,
        config=prior_config,
        model=SimpleNamespace(denoiser=SimpleNamespace(cfg=SimpleNamespace(history_steps=history_steps, horizon_steps=horizon_steps))),
    )
    return ClosedLoopFollowingRunner(SimpleNamespace(prior=prior), config)


def _fixed_plan_rollout_worker(payload: dict[str, Any]) -> dict[str, Any]:
    runner = _worker_runner(
        payload["config"],
        payload["schema"],
        payload["prior_config"],
        history_steps=int(payload["history_steps"]),
        horizon_steps=int(payload["horizon_steps"]),
    )
    result = runner.rollout_pre_sampled_plan(
        payload["context"],
        payload["plan"],
        log_prob_sum=torch.tensor(float(payload.get("log_prob_sum", 0.0)), dtype=torch.float32),
        prior_kl_sum=torch.tensor(float(payload.get("prior_kl_sum", 0.0)), dtype=torch.float32),
        guidance_norm_sum=torch.tensor(float(payload.get("guidance_norm_sum", 0.0)), dtype=torch.float32),
    )
    return {"reward": float(result.reward), "metrics": result.metrics, "trace": result.trace, "num_generated_plans": int(result.num_generated_plans)}


def _result_from_worker(
    worker_row: dict[str, Any],
    sample: PriorGuidedSampleResult,
    pos: int,
    *,
    episode_steps: int,
) -> RolloutResult:
    log_prob_sum = sample.trajectory_log_prob[pos]
    prior_kl_sum = sample.prior_kl[pos]
    guidance_norm_sum = sample.guidance_norm[pos]
    num_generated = max(int(worker_row.get("num_generated_plans", 1)), 1)
    metrics = dict(worker_row["metrics"])
    metrics.update(
        {
            "prior_kl": float(prior_kl_sum.detach().cpu()),
            "prior_kl_per_plan": float((prior_kl_sum / num_generated).detach().cpu()),
            "prior_kl_per_step": float((prior_kl_sum / max(int(episode_steps), 1)).detach().cpu()),
            "guidance_norm": float(guidance_norm_sum.detach().cpu()),
            "guidance_norm_per_plan": float((guidance_norm_sum / num_generated).detach().cpu()),
        }
    )
    return RolloutResult(
        reward=float(worker_row["reward"]),
        metrics=metrics,
        log_prob_sum=log_prob_sum,
        prior_kl_sum=prior_kl_sum,
        prior_kl_per_plan=prior_kl_sum / num_generated,
        prior_kl_per_step=prior_kl_sum / max(int(episode_steps), 1),
        guidance_norm_sum=guidance_norm_sum,
        guidance_norm_per_plan=guidance_norm_sum / num_generated,
        num_generated_plans=num_generated,
        trace=worker_row.get("trace", []),
    )


def _parallel_fixed_plan_results(
    sample: PriorGuidedSampleResult,
    plans: np.ndarray,
    contexts: list[dict[str, Any]],
    runner: ClosedLoopFollowingRunner,
    config: dict[str, Any],
    *,
    workers: int,
) -> list[RolloutResult]:
    payload_base = {
        "config": config,
        "schema": runner.sampler.prior.schema,
        "prior_config": runner.sampler.prior.config,
        "history_steps": int(runner.sampler.prior.model.denoiser.cfg.history_steps),
        "horizon_steps": int(runner.sampler.prior.model.denoiser.cfg.horizon_steps),
    }
    payloads = [
        {
            **payload_base,
            "context": ctx,
            "plan": plans[pos],
            "log_prob_sum": float(sample.trajectory_log_prob[pos].detach().cpu()),
            "prior_kl_sum": float(sample.prior_kl[pos].detach().cpu()),
            "guidance_norm_sum": float(sample.guidance_norm[pos].detach().cpu()),
        }
        for pos, ctx in enumerate(contexts)
    ]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        rows = list(pool.map(_fixed_plan_rollout_worker, payloads))
    return [_result_from_worker(row, sample, pos, episode_steps=runner.episode_steps) for pos, row in enumerate(rows)]


def _sample_paired_batch_rollouts(
    sampler: PriorGuidedDiffusionSampler,
    runner: ClosedLoopFollowingRunner,
    contexts: list[dict[str, Any]],
    seeds: list[int] | None,
    *,
    workers: int = 0,
) -> list[tuple[Any, Any]] | None:
    was_enabled = sampler.schedule.enabled
    try:
        sampler.set_guidance_enabled(False)
        with torch.no_grad():
            prior_sample, prior_plans, prepared_contexts = sample_batch_plans(sampler, runner, contexts, seeds)
        sampler.set_guidance_enabled(True)
        guided_sample, guided_plans, _ = sample_batch_plans(sampler, runner, prepared_contexts, seeds)
        if int(workers) > 0:
            prior_results = _parallel_fixed_plan_results(
                prior_sample,
                prior_plans,
                prepared_contexts,
                runner,
                sampler.config,
                workers=int(workers),
            )
            guided_results = _parallel_fixed_plan_results(
                guided_sample,
                guided_plans,
                prepared_contexts,
                runner,
                sampler.config,
                workers=int(workers),
            )
            return list(zip(prior_results, guided_results))
        return [
            (
                _rollout_sampled_plan(runner, ctx, prior_plans, prior_sample, pos),
                _rollout_sampled_plan(runner, ctx, guided_plans, guided_sample, pos),
            )
            for pos, ctx in enumerate(prepared_contexts)
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch plan sampling failed; falling back to scalar rollouts: %s", exc)
        return None
    finally:
        sampler.set_guidance_enabled(was_enabled)


@torch.no_grad()
def evaluate_paired_prior_guided_policy(
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
    seed: int,
    return_rows: bool = False,
) -> dict[str, float]:
    was_training = sampler.policy.training
    was_enabled = sampler.schedule.enabled
    sampler.eval()
    runner = ClosedLoopFollowingRunner(sampler, config)
    rows: list[dict[str, float]] = []
    for offset, idx in enumerate(indices[:max_contexts]):
        ctx = _context(raw, int(idx))
        seed_i = int(seed) + offset
        sampler.set_guidance_enabled(False)
        prior_result = runner.rollout(ctx, seed=seed_i)
        sampler.set_guidance_enabled(True)
        guided_result = runner.rollout(ctx, seed=seed_i)
        rows.append(_paired_row(prior_result, guided_result))
    sampler.set_guidance_enabled(was_enabled)
    sampler.train(was_training)
    summary = _summarize_rows(rows)
    if return_rows:
        summary["_rows"] = rows  # type: ignore[assignment]
    return summary


def _selection_score(summary: dict[str, float], training: dict[str, Any]) -> float:
    alpha = float(training.get("selection_alpha_prior_kl", 0.05))
    beta = float(training.get("selection_beta_physics", 0.1))
    reward_delta = float(summary.get("val_reward_delta_mean", summary.get("reward_delta_mean", summary.get("reward_mean", 0.0))))
    kl = float(summary.get("val_guided_prior_kl_per_plan_mean", summary.get("guided_prior_kl_per_plan_mean", summary.get("prior_kl_per_plan_mean", 0.0))))
    physics = float(
        summary.get(
            "val_guided_lead_physics_penalty_mean",
            summary.get("guided_lead_physics_penalty_mean", summary.get("lead_physics_penalty_mean", 0.0)),
        )
    )
    return float(reward_delta - alpha * kl - beta * physics)


def train_prior_guided_policy(config: dict[str, Any], *, config_dir: str | Path | None = None) -> dict[str, Any]:
    training = config.get("training", {})
    set_seed(int(training.get("seed", 42)))
    natural_dir, diffusion_ckpt, output_dir = _resolve_paths(config, config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_npz(natural_dir / "dataset.npz")
    schema = load_json(natural_dir / "feature_schema.json")
    split_index = raw["split_index"]
    all_train_idx = np.where(split_index == SPLIT_TO_INDEX[str(training.get("split", "train"))])[0]
    val_idx = np.where(split_index == SPLIT_TO_INDEX[str(training.get("val_split", "val"))])[0]
    max_train_contexts = int(training.get("max_train_contexts", 0))
    resample_contexts_each_epoch = bool(training.get("resample_contexts_each_epoch", True))
    rng = np.random.default_rng(int(training.get("seed", 42)))
    if len(all_train_idx) == 0:
        raise RuntimeError("No training contexts found for prior-guided policy")
    train_idx = all_train_idx
    if not resample_contexts_each_epoch:
        train_idx = _sample_context_indices(
            raw,
            all_train_idx,
            max_train_contexts=max_train_contexts,
            rng=rng,
            mode=str(training.get("context_sampling", "stratified")),
            training=training,
            config_dir=config.get("_runtime", {}).get("config_dir", config_dir),
        )

    device = select_device(training.get("device", "auto"))
    prior = DiffusionPriorAdapter.load(natural_dir, diffusion_ckpt, device=device)
    policy = GuidancePolicy(GuidancePolicyConfig.from_prior(prior.model.denoiser.cfg, config))
    sampler = PriorGuidedDiffusionSampler(prior, policy, config).train(True)
    runner = ClosedLoopFollowingRunner(sampler, config)
    optimizer = torch.optim.AdamW(
        sampler.policy.parameters(),
        lr=float(training.get("lr", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = int(training.get("epochs", 20))
    batch_size = int(training.get("batch_size", 4))
    grad_clip = float(training.get("grad_clip", 1.0))
    lambda_prior = float(training.get("lambda_prior", 0.01))
    baseline_beta = float(training.get("baseline_ema_beta", 0.9))
    reward_clip = float(training.get("reward_clip", 0.0))
    eval_contexts = int(training.get("eval_contexts", 16))
    paired_prior_baseline = bool(training.get("paired_prior_baseline", False))
    paired_same_seed = bool(training.get("paired_same_seed", True))
    paired_reward_mode = str(training.get("paired_reward_mode", "delta"))
    paired_abs_weight = float(training.get("paired_abs_weight", 0.1))
    paired_abs_warmup_epochs = int(training.get("paired_abs_warmup_epochs", 0))
    horizon_steps = int(getattr(sampler.prior.model.denoiser.cfg, "horizon_steps", 0))
    batch_plan_sampling = bool(training.get("batch_plan_sampling", True)) and paired_prior_baseline
    rollout_workers = max(int(training.get("rollout_workers", 0)), 0)
    if batch_plan_sampling and runner.episode_steps > horizon_steps:
        logger.warning(
            "Batch plan sampling disabled because episode_steps=%d exceeds diffusion horizon=%d",
            runner.episode_steps,
            horizon_steps,
        )
        batch_plan_sampling = False
    if batch_plan_sampling and runner.commit_steps_max < runner.episode_steps:
        logger.warning(
            "Batch plan sampling disabled because commit_steps_max=%d is less than episode_steps=%d",
            runner.commit_steps_max,
            runner.episode_steps,
        )
        batch_plan_sampling = False
    if rollout_workers > 0 and not batch_plan_sampling:
        logger.warning("rollout_workers=%d requested but fixed-plan batch sampling is disabled; using serial scalar rollouts", rollout_workers)
        rollout_workers = 0
    writer = _make_writer(output_dir, bool(training.get("tensorboard", True)))
    history: list[dict[str, Any]] = []
    baseline: float | None = None
    best_reward = float("-inf")
    best_delta_reward = float("-inf")
    best_selection_score = float("-inf")
    global_step = 0

    epoch_context_budget = min(len(all_train_idx), max_train_contexts) if max_train_contexts > 0 else len(all_train_idx)
    logger.info("Training prior-guided policy on %s with %d available contexts; epoch budget=%d", device, len(all_train_idx), epoch_context_budget)
    for epoch in range(1, epochs + 1):
        if resample_contexts_each_epoch:
            epoch_train_idx = _sample_context_indices(
                raw,
                all_train_idx,
                max_train_contexts=max_train_contexts,
                rng=rng,
                mode=str(training.get("context_sampling", "stratified")),
                training=training,
                config_dir=config.get("_runtime", {}).get("config_dir", config_dir),
            )
        else:
            epoch_train_idx = train_idx
        if len(epoch_train_idx) == 0:
            raise RuntimeError("No training contexts found for prior-guided policy")
        shuffled = rng.permutation(epoch_train_idx)
        epoch_rows: list[dict[str, float]] = []
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            valid_results: list[Any] = []
            valid_rewards: list[float] = []
            batch_rewards: list[float] = []
            batch_rewards_clipped: list[float] = []
            batch_prior: list[float] = []
            batch_prior_per_plan: list[float] = []
            batch_prior_per_step: list[float] = []
            batch_guidance: list[float] = []
            batch_log_prob: list[float] = []
            batch_metrics: list[dict[str, float]] = []
            batch_pair_rows: list[dict[str, float]] = []
            valid_advantages: list[float] = []
            prior_loss_metric = str(training.get("prior_kl_loss_metric", "prior_kl_per_plan"))
            batch_contexts = [_context(raw, int(idx)) for idx in batch]
            paired_batch_rollouts: list[tuple[Any, Any]] | None = None
            if paired_prior_baseline and batch_plan_sampling:
                batch_seeds = (
                    [int(training.get("seed", 42)) + global_step + pos for pos in range(len(batch))]
                    if paired_same_seed
                    else None
                )
                paired_batch_rollouts = _sample_paired_batch_rollouts(
                    sampler,
                    runner,
                    batch_contexts,
                    batch_seeds,
                    workers=rollout_workers,
                )
            for batch_pos, ctx in enumerate(batch_contexts):
                if paired_prior_baseline:
                    if paired_batch_rollouts is not None:
                        prior_result, guided_result = paired_batch_rollouts[batch_pos]
                    else:
                        seed_i = (int(training.get("seed", 42)) + global_step) if paired_same_seed else None
                        was_enabled = sampler.schedule.enabled
                        sampler.set_guidance_enabled(False)
                        with torch.no_grad():
                            prior_result = runner.rollout(ctx, seed=seed_i)
                        sampler.set_guidance_enabled(True)
                        guided_result = runner.rollout(ctx, seed=seed_i if paired_same_seed else None)
                        sampler.set_guidance_enabled(was_enabled)
                    reward = float(guided_result.reward)
                    prior_reward = float(prior_result.reward)
                    reward_delta = reward - prior_reward
                    use_abs_warmup = paired_abs_warmup_epochs > 0 and epoch <= paired_abs_warmup_epochs
                    advantage_value = reward_delta
                    if paired_reward_mode == "delta_plus_abs" or use_abs_warmup:
                        advantage_value = reward_delta + paired_abs_weight * reward
                    reward_for_loss = float(np.clip(advantage_value, -reward_clip, reward_clip)) if reward_clip > 0.0 else float(advantage_value)
                    result = guided_result
                    batch_pair_rows.append(_paired_row(prior_result, guided_result))
                else:
                    result = runner.rollout(ctx, seed=None)
                    reward = float(result.reward)
                    reward_for_loss = float(np.clip(reward, -reward_clip, reward_clip)) if reward_clip > 0.0 else reward
                batch_rewards.append(reward)
                batch_rewards_clipped.append(reward_for_loss)
                batch_prior.append(float(result.prior_kl_sum.detach().cpu()))
                batch_prior_per_plan.append(float(result.prior_kl_per_plan.detach().cpu()))
                batch_prior_per_step.append(float(result.prior_kl_per_step.detach().cpu()))
                batch_guidance.append(float(result.guidance_norm_sum.detach().cpu()))
                batch_log_prob.append(float(result.log_prob_sum.detach().cpu()))
                batch_metrics.append(result.metrics)
                global_step += 1
                if result.metrics.get("invalid_initial_context", 0.0) > 0.0:
                    continue
                valid_results.append(result)
                valid_rewards.append(reward_for_loss)
                valid_advantages.append(reward_for_loss)
                if writer is not None:
                    writer.add_scalar("rollout/reward", reward, global_step)
                    writer.add_scalar("rollout/reward_for_loss", reward_for_loss, global_step)
                    writer.add_scalar("rollout/prior_kl", batch_prior[-1], global_step)
                    writer.add_scalar("rollout/guidance_norm", batch_guidance[-1], global_step)
            losses: list[torch.Tensor] = []
            if valid_results:
                reward_tensor = torch.tensor(valid_rewards, dtype=torch.float32, device=device)
                if paired_prior_baseline:
                    advantages = torch.tensor(valid_advantages, dtype=torch.float32, device=device)
                    if bool(training.get("normalize_paired_advantages", False)) and advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
                else:
                    batch_baseline = float(reward_tensor.mean().detach().cpu())
                    baseline = batch_baseline if baseline is None else baseline_beta * baseline + (1.0 - baseline_beta) * batch_baseline
                    advantages = reward_tensor - float(baseline)
                if (not paired_prior_baseline) and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
                for advantage, result in zip(advantages, valid_results):
                    prior_penalty = getattr(result, prior_loss_metric, result.prior_kl_per_plan)
                    loss = -advantage.detach() * result.log_prob_sum + lambda_prior * prior_penalty
                    if loss.requires_grad:
                        losses.append(loss)
            if losses:
                batch_loss = torch.stack(losses).mean()
                batch_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(sampler.policy.parameters(), grad_clip)
                optimizer.step()
            else:
                batch_loss = torch.zeros((), dtype=torch.float32, device=device)
            prior_kl_mean = float(np.mean(batch_prior)) if batch_prior else 0.0
            prior_kl_per_plan_mean = float(np.mean(batch_prior_per_plan)) if batch_prior_per_plan else 0.0
            prior_kl_per_step_mean = float(np.mean(batch_prior_per_step)) if batch_prior_per_step else 0.0
            if prior_loss_metric == "prior_kl_sum":
                prior_kl_loss_mean = prior_kl_mean
            elif prior_loss_metric == "prior_kl_per_step":
                prior_kl_loss_mean = prior_kl_per_step_mean
            else:
                prior_kl_loss_mean = prior_kl_per_plan_mean
            reward_abs_mean = float(np.mean(np.abs(batch_rewards_clipped))) if batch_rewards_clipped else 0.0
            if paired_prior_baseline and batch_pair_rows:
                row = {"epoch": float(epoch), **_summarize_rows(batch_pair_rows)}
                row.update(
                    {
                        "reward_mean": float(row.get("guided_reward_mean", np.mean(batch_rewards))),
                        "reward_guided_mean": float(row.get("guided_reward_mean", np.mean(batch_rewards))),
                        "reward_prior_mean": float(row.get("prior_reward_mean", 0.0)),
                        "reward_delta_mean": float(row.get("reward_delta_mean", 0.0)),
                        "reward_for_loss_mean": float(np.mean(batch_rewards_clipped)),
                        "rss_reward_guided_mean": float(row.get("guided_rss_reward_mean", 0.0)),
                        "rss_reward_prior_mean": float(row.get("prior_rss_reward_mean", 0.0)),
                        "rss_reward_delta_mean": float(row.get("rss_reward_delta_mean", 0.0)),
                        "rss_risk_improvement_mean": float(row.get("rss_risk_improvement_mean", 0.0)),
                        "min_rss_margin_guided_mean": float(row.get("guided_min_rss_margin_mean", 0.0)),
                        "min_rss_margin_prior_mean": float(row.get("prior_min_rss_margin_mean", 0.0)),
                        "min_rss_margin_delta_mean": float(row.get("min_rss_margin_delta_mean", 0.0)),
                        "rss_margin_degradation_mean": float(row.get("rss_margin_degradation_mean", 0.0)),
                        "min_gap_guided_mean": float(row.get("guided_min_gap_mean", 0.0)),
                        "min_gap_prior_mean": float(row.get("prior_min_gap_mean", 0.0)),
                        "min_gap_delta_mean": float(row.get("min_gap_delta_mean", 0.0)),
                        "min_ttc_guided_mean": float(row.get("guided_min_ttc_mean", 0.0)),
                        "min_ttc_prior_mean": float(row.get("prior_min_ttc_mean", 0.0)),
                        "min_ttc_delta_mean": float(row.get("min_ttc_delta_mean", 0.0)),
                        "prior_kl_mean": prior_kl_mean,
                        "prior_kl_per_plan_mean": prior_kl_per_plan_mean,
                        "prior_kl_per_step_mean": prior_kl_per_step_mean,
                        "prior_kl_penalty_mean": float(lambda_prior * prior_kl_loss_mean),
                        "prior_kl_reward_ratio": float(prior_kl_loss_mean / max(reward_abs_mean, 1e-6)),
                        "guidance_norm_mean": float(np.mean(batch_guidance)),
                        "guidance_norm_per_plan_mean": float(np.mean([m.get("guidance_norm_per_plan", 0.0) for m in batch_metrics])),
                        "num_generated_plans_mean": float(np.mean([m.get("num_generated_plans", 0.0) for m in batch_metrics])),
                        "trajectory_log_prob_mean": float(np.mean(batch_log_prob)),
                        "loss": float(batch_loss.detach().cpu()) if losses else 0.0,
                        "collision_rate": float(row.get("guided_collision_mean", 0.0)),
                        "collision_valid_rate": float(row.get("guided_collision_valid_mean", 0.0)),
                        "invalid_collision_rate": float(row.get("guided_invalid_collision_mean", 0.0)),
                        "near_collision_rate": float(row.get("guided_near_collision_mean", 0.0)),
                        "hard_brake_rate": float(row.get("guided_hard_brake_mean", 0.0)),
                        "invalid_initial_context_rate": float(row.get("invalid_initial_context_mean", 0.0)),
                        "action_clip_rate": float(row.get("guided_action_clip_rate_mean", 0.0)),
                        "jerk_violation_rate": float(row.get("guided_jerk_violation_rate_mean", 0.0)),
                        "speed_negative_rate": float(row.get("guided_speed_negative_rate_mean", 0.0)),
                    }
                )
            else:
                row = {
                    "epoch": float(epoch),
                    "reward_mean": float(np.mean(batch_rewards)),
                    "reward_for_loss_mean": float(np.mean(batch_rewards_clipped)),
                    "prior_kl_mean": prior_kl_mean,
                    "prior_kl_per_plan_mean": prior_kl_per_plan_mean,
                    "prior_kl_per_step_mean": prior_kl_per_step_mean,
                    "prior_kl_penalty_mean": float(lambda_prior * prior_kl_loss_mean),
                    "prior_kl_reward_ratio": float(prior_kl_loss_mean / max(reward_abs_mean, 1e-6)),
                    "guidance_norm_mean": float(np.mean(batch_guidance)),
                    "guidance_norm_per_plan_mean": float(np.mean([m.get("guidance_norm_per_plan", 0.0) for m in batch_metrics])),
                    "num_generated_plans_mean": float(np.mean([m.get("num_generated_plans", 0.0) for m in batch_metrics])),
                    "trajectory_log_prob_mean": float(np.mean(batch_log_prob)),
                    "loss": float(batch_loss.detach().cpu()) if losses else 0.0,
                    "collision_rate": float(np.mean([m["collision"] for m in batch_metrics])),
                    "collision_valid_rate": float(np.mean([m.get("collision_valid", 0.0) for m in batch_metrics])),
                    "invalid_collision_rate": float(np.mean([m.get("invalid_collision", 0.0) for m in batch_metrics])),
                    "near_collision_rate": float(np.mean([m.get("near_collision", 0.0) for m in batch_metrics])),
                    "hard_brake_rate": float(np.mean([m.get("hard_brake", 0.0) for m in batch_metrics])),
                    "invalid_initial_context_rate": float(np.mean([m.get("invalid_initial_context", 0.0) for m in batch_metrics])),
                    "min_ttc_mean": float(np.mean([m["min_ttc"] for m in batch_metrics])),
                    "min_gap_mean": float(np.mean([m["min_gap"] for m in batch_metrics])),
                    "min_rss_margin_mean": float(np.mean([m["min_rss_margin"] for m in batch_metrics])),
                    "action_clip_rate": float(np.mean([m.get("action_clip_rate", 0.0) for m in batch_metrics])),
                    "jerk_violation_rate": float(np.mean([m.get("jerk_violation_rate", 0.0) for m in batch_metrics])),
                    "speed_negative_rate": float(np.mean([m.get("speed_negative_rate", 0.0) for m in batch_metrics])),
                    "naturalness_gate_mean": float(np.mean([m.get("naturalness_gate", 1.0) for m in batch_metrics])),
                    **{f"{key}_mean": float(np.mean([m.get(key, 0.0) for m in batch_metrics])) for key in REWARD_COMPONENT_KEYS},
                }
            epoch_rows.append(row)
        epoch_summary = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        val_metrics: dict[str, float] = {}
        if len(val_idx) > 0 and (epoch == 1 or epoch % int(training.get("eval_every_epochs", 5)) == 0 or epoch == epochs):
            if paired_prior_baseline:
                val_metrics = evaluate_paired_prior_guided_policy(
                    sampler,
                    config,
                    raw,
                    val_idx,
                    max_contexts=eval_contexts,
                    seed=int(training.get("seed", 42)) + 1000 + epoch,
                )
            else:
                val_metrics = evaluate_prior_guided_policy(
                    sampler,
                    config,
                    raw,
                    val_idx,
                    max_contexts=eval_contexts,
                    seed=int(training.get("seed", 42)) + 1000 + epoch,
                )
        history_row = {"epoch": epoch, **epoch_summary, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(history_row)
        if writer is not None:
            for key, value in epoch_summary.items():
                if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                    writer.add_scalar(f"epoch/{key}", float(value), epoch)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                    writer.add_scalar(f"eval/{key}", float(value), epoch)
        if paired_prior_baseline:
            score = float(val_metrics.get("guided_reward_mean", epoch_summary.get("reward_guided_mean", epoch_summary["reward_mean"])))
            delta_score = float(val_metrics.get("reward_delta_mean", epoch_summary.get("reward_delta_mean", 0.0)))
        else:
            score = float(val_metrics.get("reward_mean", epoch_summary["reward_mean"]))
            delta_score = score
        selection_score = _selection_score({**epoch_summary, **{f"val_{k}": v for k, v in val_metrics.items()}}, training)
        checkpoint_summary = {"train": epoch_summary, "val": val_metrics}
        if score > best_reward:
            best_reward = score
            _save_checkpoint(output_dir / "checkpoints" / "best_reward.pt", sampler, config, schema, epoch, checkpoint_summary)
        if delta_score > best_delta_reward:
            best_delta_reward = delta_score
            _save_checkpoint(output_dir / "checkpoints" / "best_delta_reward.pt", sampler, config, schema, epoch, checkpoint_summary)
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            _save_checkpoint(output_dir / "checkpoints" / "best_selection_score.pt", sampler, config, schema, epoch, checkpoint_summary)
        _save_checkpoint(output_dir / "checkpoints" / "last.pt", sampler, config, schema, epoch, checkpoint_summary)
        if epoch == 1 or epoch % int(training.get("log_every_epochs", 1)) == 0 or epoch == epochs:
            logger.info(
                "epoch=%03d reward=%.4f delta=%.4f prior_kl=%.4f collision=%.3f score=%.4f selection=%.4f",
                epoch,
                epoch_summary["reward_mean"],
                epoch_summary.get("reward_delta_mean", float("nan")),
                epoch_summary["prior_kl_mean"],
                epoch_summary["collision_rate"],
                score,
                selection_score,
            )

    _write_history_csv(output_dir / "training_history.csv", history)
    summary = {
        "best_reward": best_reward,
        "best_delta_reward": best_delta_reward,
        "best_selection_score": best_selection_score,
        "epochs_completed": epochs,
        "history": history,
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "training_summary.json")
    if writer is not None:
        writer.close()
    return summary
