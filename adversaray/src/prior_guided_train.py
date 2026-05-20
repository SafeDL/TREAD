"""Shared context and evaluation helpers for prior/KING-guided rollouts."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.utils import load_json

from .closed_loop_runner import ClosedLoopFollowingRunner
from .prior_guided_sampler import PriorGuidedDiffusionSampler
from .risk_utils import actions_to_accel_jerk


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    ego_lengths = raw.get("ego_length")
    adv_lengths = raw.get("adv_length")
    context: dict[str, Any] = {
        "raw_context_states": raw["context_states"][idx],
        "ego_length": float(ego_lengths[idx]) if ego_lengths is not None else 4.8,
        "adv_length": float(adv_lengths[idx]) if adv_lengths is not None else 4.8,
    }
    for key in (
        "recording_id",
        "event_id",
        "anchor_frame",
        "source_type",
        "anchor_dataset_index",
        "target_gap",
        "target_ttc",
        "target_rss_margin",
        "criticality_score",
    ):
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    return context


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
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


def rollout_distance_metrics(recorded: dict[str, np.ndarray], prefix: str, rows: list[dict[str, Any]]) -> dict[str, float]:
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
) -> dict[str, Any]:
    was_training = sampler.policy.training
    sampler.eval()
    runner = ClosedLoopFollowingRunner(sampler, config)
    rows: list[dict[str, Any]] = []
    for offset, idx in enumerate(indices[:max_contexts]):
        result = runner.rollout(_context(raw, int(idx)), seed=int(seed) + offset)
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
    if return_rows:
        summary["_rows"] = rows
    return summary


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
