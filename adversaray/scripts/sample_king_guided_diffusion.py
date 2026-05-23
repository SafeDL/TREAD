#!/usr/bin/env python3
"""Sample frozen-prior plans and optimize them with KING-style gradients."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
from adversaray.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from adversaray.src.king_gradient_guidance import optimize_action_plan_king
from adversaray.src.context_utils import _context, _load_npz
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "king_guided_following.yaml"
SCRIPT_DEFAULTS = {
    "split": "val",
    "num_contexts": 256,
    "seed": 42,
    "output_name": "king_guided_samples.npz",
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _attach_runtime_paths(cfg: dict[str, Any], base: Path) -> None:
    paths = cfg.get("paths", {})
    runtime: dict[str, str] = {"config_dir": str(base)}
    for key, value in paths.items():
        runtime[key] = str(_resolve(value, base))
    cfg["_runtime"] = runtime


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _split_indices(raw: dict[str, np.ndarray], split: str) -> np.ndarray:
    if "split_index" not in raw:
        raise KeyError(
            "Tail contexts must contain split_index; rebuild them with "
            "prepare_king_guided_contexts.py."
        )
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[split])[0]
    idx = idx.astype(np.int64)
    if idx.size == 0:
        raise RuntimeError(f"No tail contexts found for split '{split}'")
    return idx


def _select_raw_contexts(
    cfg: dict[str, Any],
    base: Path,
    split: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    training = cfg.get("training", {})
    tail_context_value = str(training.get("tail_context_path", "") or "").strip()
    if not tail_context_value:
        raise ValueError(
            "training.tail_context_path must be set for KING sampling"
        )
    path = _resolve(tail_context_value, base)
    raw = _load_npz(path)
    required = {"context_states", "split_index"}
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")
    return raw, _split_indices(raw, split), "tail_natural"


def _array_mean(arrays: dict[str, np.ndarray], key: str) -> float:
    value = arrays.get(key)
    if value is None or value.size == 0:
        return float("nan")
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _masked_action_l2(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    diff = np.square(
        np.asarray(left, dtype=np.float32)
        - np.asarray(right, dtype=np.float32)
    )
    per_step = np.mean(diff, axis=-1)
    if mask is None:
        return np.sqrt(np.mean(per_step, axis=1))
    weights = np.asarray(mask, dtype=np.float32)
    denom = np.maximum(np.sum(weights, axis=1), 1.0)
    return np.sqrt(np.sum(per_step * weights, axis=1) / denom)


def _sample_summary(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    reference_key = (
        "king_reference_actions"
        if "king_reference_actions" in arrays
        else "prior_actions"
    )
    reference_actions = np.asarray(arrays[reference_key], dtype=np.float32)
    king_actions = np.asarray(arrays["king_actions"], dtype=np.float32)
    action_l2 = _masked_action_l2(
        king_actions,
        reference_actions,
        arrays.get("action_mask"),
    )
    risk_before = _array_mean(arrays, "risk_before")
    risk_after = _array_mean(arrays, "risk_after")
    return {
        "risk_before_mean": risk_before,
        "risk_after_mean": risk_after,
        "risk_delta_mean": risk_after - risk_before,
        "gap_before_mean": _array_mean(arrays, "gap_before"),
        "gap_after_mean": _array_mean(arrays, "gap_after"),
        "ttc_before_mean": _array_mean(arrays, "ttc_before"),
        "ttc_after_mean": _array_mean(arrays, "ttc_after"),
        "rss_before_mean": _array_mean(arrays, "rss_before"),
        "rss_after_mean": _array_mean(arrays, "rss_after"),
        "naturalness_penalty_mean": _array_mean(arrays, "naturalness_penalty"),
        "physics_penalty_mean": _array_mean(arrays, "physics_penalty"),
        "n1_action_residual_mean": _array_mean(arrays, "n1_action_residual"),
        "naturalness_violation_mean": _array_mean(arrays, "naturalness_violation"),
        "lambda_naturalness_final_mean": _array_mean(arrays, "lambda_naturalness_final"),
        "ego_accel_min_mean": _array_mean(arrays, "ego_accel_min"),
        "ego_accel_mean": _array_mean(arrays, "ego_accel_mean"),
        "ego_speed_min_mean": _array_mean(arrays, "ego_speed_min"),
        "ego_speed_mean": _array_mean(arrays, "ego_speed_mean"),
        "action_l2_mean": float(np.mean(action_l2)) if action_l2.size else float("nan"),
        "event_steps_mean": _array_mean(arrays, "event_steps"),
        "num_plans_king_mean": _array_mean(arrays, "num_plans_king"),
    }


def _event_steps(ctx: dict[str, Any], cfg: dict[str, Any]) -> int:
    if "event_steps" in ctx:
        steps = int(ctx["event_steps"])
    else:
        steps = int(cfg.get("env", {}).get("episode_steps", 50))
    if steps <= 0:
        raise ValueError(f"event_steps must be positive, got {steps}")
    return steps


def _to_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(np.asarray(value, dtype=np.float64).mean())


def _king_scalar_summary(result: dict[str, Any]) -> dict[str, float]:
    keys = (
        "risk_before",
        "risk_after",
        "rss_before",
        "rss_after",
        "ttc_before",
        "ttc_after",
        "gap_before",
        "gap_after",
        "naturalness_penalty",
        "physics_penalty",
        "rss_objective",
        "ttc_objective",
        "drac_objective",
        "gap_objective",
        "min_rss_margin",
        "min_ttc",
        "min_gap",
        "prior_rss_objective",
        "prior_ttc_objective",
        "prior_drac_objective",
        "prior_gap_objective",
        "prior_min_rss_margin",
        "prior_min_ttc",
        "prior_min_gap",
        "ego_accel_min",
        "ego_accel_mean",
        "ego_speed_min",
        "ego_speed_mean",
        "prior_ego_accel_min",
        "prior_ego_accel_mean",
        "prior_ego_speed_min",
        "prior_ego_speed_mean",
        "n1_action_residual",
        "naturalness_violation",
        "lambda_naturalness_final",
    )
    out: dict[str, float] = {}
    for key in keys:
        if key not in result:
            continue
        value = _to_scalar(result[key])
        if np.isfinite(value):
            out[key] = value
    return out


def _mean_plan_summaries(
    summaries: list[dict[str, float]],
    key: str,
) -> float:
    values = np.asarray(
        [item.get(key, np.nan) for item in summaries],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _pad_actions(
    sequences: list[np.ndarray],
    *,
    max_steps: int | None = None,
    action_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("Cannot pad an empty action sequence list")
    if max_steps is None:
        max_steps = max(int(seq.shape[0]) for seq in sequences)
    if action_dim is None:
        action_dim = max(int(seq.shape[1]) for seq in sequences)
    padded = np.zeros((len(sequences), max_steps, action_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_steps), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        steps = int(seq.shape[0])
        dim = int(seq.shape[1])
        padded[idx, :steps, :dim] = seq.astype(np.float32)
        mask[idx, :steps] = 1.0
    return padded, mask


def _make_king_plan_callback(
    *,
    sampler: FrozenDiffusionSampler,
    cfg: dict[str, Any],
    ego_length: float,
    adv_length: float,
    seed: int,
):
    device = sampler.prior.device

    def callback(
        obs: dict[str, np.ndarray],
        plan_id: int,
        step: int,
    ) -> dict[str, Any]:
        context_states = torch.from_numpy(obs["context_states"][None]).float()
        context_features = torch.from_numpy(
            obs["context_features"][None]
        ).float()
        relative_history = torch.from_numpy(
            obs["relative_history"][None]
        ).float()
        local_seed = int(seed) + int(plan_id) * 1009 + int(step)
        with torch.no_grad():
            prior_sample = sampler.sample(
                context_states,
                context_features,
                relative_history,
                ego_length=torch.tensor([ego_length], dtype=torch.float32),
                adv_length=torch.tensor([adv_length], dtype=torch.float32),
                num_samples=1,
                seed=local_seed,
            )
        raw_context = torch.from_numpy(obs["raw_context_states"][None]).to(
            device=device,
            dtype=torch.float32,
        )
        result = optimize_action_plan_king(
            prior_sample.raw_actions.to(device),
            raw_context,
            torch.tensor([ego_length], dtype=torch.float32, device=device),
            torch.tensor([adv_length], dtype=torch.float32, device=device),
            sampler.prior.schema,
            cfg,
        )
        return {
            "plan": _tensor_to_numpy(result["adv_actions"])[0],
            "prior_plan": _tensor_to_numpy(result["prior_actions"])[0],
            "summary": _king_scalar_summary(result),
        }

    return callback


def _sample_receding_case(
    *,
    runner: ClosedLoopFollowingRunner,
    sampler: FrozenDiffusionSampler,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    steps = _event_steps(ctx, cfg)
    prior_result = runner.rollout(
        ctx,
        seed=seed,
        episode_steps=steps,
    )
    callback = _make_king_plan_callback(
        sampler=sampler,
        cfg=cfg,
        ego_length=float(ctx["ego_length"]),
        adv_length=float(ctx["adv_length"]),
        seed=seed,
    )
    king_result = runner.rollout(
        ctx,
        seed=seed,
        episode_steps=steps,
        plan_callback=callback,
    )
    if prior_result.actions is None or king_result.actions is None:
        raise RuntimeError("Closed-loop rollout did not return executed actions")
    if king_result.prior_actions is None:
        raise RuntimeError("KING rollout did not return reference prior actions")
    summaries = king_result.plan_summaries
    scalars = {
        "risk_before": _mean_plan_summaries(summaries, "risk_before"),
        "risk_after": _mean_plan_summaries(summaries, "risk_after"),
        "closed_loop_risk_before": float(prior_result.closed_loop_risk),
        "closed_loop_risk_after": float(king_result.closed_loop_risk),
        "num_plans_prior": float(prior_result.num_generated_plans),
        "num_plans_king": float(king_result.num_generated_plans),
    }
    for key in (
        "rss_before",
        "rss_after",
        "ttc_before",
        "ttc_after",
        "gap_before",
        "gap_after",
        "naturalness_penalty",
        "physics_penalty",
        "rss_objective",
        "ttc_objective",
        "drac_objective",
        "gap_objective",
        "min_rss_margin",
        "min_ttc",
        "min_gap",
        "prior_rss_objective",
        "prior_ttc_objective",
        "prior_drac_objective",
        "prior_gap_objective",
        "prior_min_rss_margin",
        "prior_min_ttc",
        "prior_min_gap",
        "ego_accel_min",
        "ego_accel_mean",
        "ego_speed_min",
        "ego_speed_mean",
        "prior_ego_accel_min",
        "prior_ego_accel_mean",
        "prior_ego_speed_min",
        "prior_ego_speed_mean",
        "n1_action_residual",
        "naturalness_violation",
        "lambda_naturalness_final",
    ):
        scalars[key] = _mean_plan_summaries(summaries, key)
    return {
        "prior_actions": prior_result.actions,
        "king_actions": king_result.actions,
        "king_reference_actions": king_result.prior_actions,
        "event_steps": int(steps),
        "scalars": scalars,
    }


def main() -> None:
    setup_logging(SCRIPT_DEFAULTS["log_level"])

    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    _attach_runtime_paths(cfg, base)
    paths = cfg.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    output_dir = _resolve(paths["output_dir"], base)
    output_path = output_dir / str(SCRIPT_DEFAULTS["output_name"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split = str(SCRIPT_DEFAULTS["split"])
    raw, idx, source_name = _select_raw_contexts(cfg, base, split)
    max_contexts = min(int(SCRIPT_DEFAULTS["num_contexts"]), int(idx.size))
    selected = idx[:max_contexts]
    if max_contexts <= 0:
        raise ValueError("No contexts selected for KING sampling")

    sampler = FrozenDiffusionSampler.from_config(cfg, config_dir=base).eval()
    runner = ClosedLoopFollowingRunner(sampler, cfg)

    contexts = [_context(raw, int(item)) for item in selected]
    output: dict[str, Any] = {
        "context_states": np.stack(
            [
                np.asarray(ctx["raw_context_states"], dtype=np.float32)
                for ctx in contexts
            ],
            axis=0,
        ),
        "ego_length": np.asarray(
            [float(ctx["ego_length"]) for ctx in contexts],
            dtype=np.float32,
        ),
        "adv_length": np.asarray(
            [float(ctx["adv_length"]) for ctx in contexts],
            dtype=np.float32,
        ),
        "event_steps": np.asarray(
            [_event_steps(ctx, cfg) for ctx in contexts],
            dtype=np.int64,
        ),
    }
    scalar_keys = (
        "risk_before",
        "risk_after",
        "closed_loop_risk_before",
        "closed_loop_risk_after",
        "rss_before",
        "rss_after",
        "ttc_before",
        "ttc_after",
        "gap_before",
        "gap_after",
        "naturalness_penalty",
        "physics_penalty",
        "rss_objective",
        "ttc_objective",
        "drac_objective",
        "gap_objective",
        "min_rss_margin",
        "min_ttc",
        "min_gap",
        "prior_rss_objective",
        "prior_ttc_objective",
        "prior_drac_objective",
        "prior_gap_objective",
        "prior_min_rss_margin",
        "prior_min_ttc",
        "prior_min_gap",
        "ego_accel_min",
        "ego_accel_mean",
        "ego_speed_min",
        "ego_speed_mean",
        "prior_ego_accel_min",
        "prior_ego_accel_mean",
        "prior_ego_speed_min",
        "prior_ego_speed_mean",
        "n1_action_residual",
        "naturalness_violation",
        "lambda_naturalness_final",
        "num_plans_prior",
        "num_plans_king",
    )

    prior_sequences: list[np.ndarray] = []
    king_sequences: list[np.ndarray] = []
    reference_sequences: list[np.ndarray] = []
    scalar_values: dict[str, list[float]] = {key: [] for key in scalar_keys}
    for pos, ctx in enumerate(contexts):
        seed = int(SCRIPT_DEFAULTS["seed"]) + int(pos)
        result = _sample_receding_case(
            runner=runner,
            sampler=sampler,
            cfg=cfg,
            ctx=ctx,
            seed=seed,
        )
        prior_sequences.append(result["prior_actions"])
        king_sequences.append(result["king_actions"])
        reference_sequences.append(result["king_reference_actions"])
        for key in scalar_keys:
            scalar_values[key].append(
                float(result["scalars"].get(key, np.nan))
            )

        logger.info(
            "KING event %d/%d steps=%d risk %.4f -> %.4f",
            pos + 1,
            max_contexts,
            int(result["event_steps"]),
            float(result["scalars"].get("closed_loop_risk_before", np.nan)),
            float(result["scalars"].get("closed_loop_risk_after", np.nan)),
        )

    max_action_steps = max(
        max(seq.shape[0] for seq in prior_sequences),
        max(seq.shape[0] for seq in king_sequences),
        max(seq.shape[0] for seq in reference_sequences),
    )
    action_dim = max(
        max(seq.shape[1] for seq in prior_sequences),
        max(seq.shape[1] for seq in king_sequences),
        max(seq.shape[1] for seq in reference_sequences),
    )
    prior_actions, action_mask = _pad_actions(
        prior_sequences,
        max_steps=int(max_action_steps),
        action_dim=int(action_dim),
    )
    king_actions, king_mask = _pad_actions(
        king_sequences,
        max_steps=int(max_action_steps),
        action_dim=int(action_dim),
    )
    reference_actions, reference_mask = _pad_actions(
        reference_sequences,
        max_steps=int(max_action_steps),
        action_dim=int(action_dim),
    )
    if not np.array_equal(king_mask, reference_mask):
        raise RuntimeError("KING reference action mask does not match output")
    output["prior_actions"] = prior_actions
    output["king_actions"] = king_actions
    output["king_reference_actions"] = reference_actions
    output["action_mask"] = king_mask
    output["prior_action_mask"] = action_mask
    output["king_action_mask"] = king_mask
    for key, values in scalar_values.items():
        output[key] = np.asarray(values, dtype=np.float32)

    arrays = dict(output)
    rename = {
        "rss_objective": "rss_objective_after",
        "ttc_objective": "ttc_objective_after",
        "drac_objective": "drac_objective_after",
        "gap_objective": "gap_objective_after",
        "min_rss_margin": "min_rss_margin_after",
        "min_ttc": "min_ttc_after",
        "min_gap": "min_gap_after",
        "prior_rss_objective": "rss_objective_before",
        "prior_ttc_objective": "ttc_objective_before",
        "prior_drac_objective": "drac_objective_before",
        "prior_gap_objective": "gap_objective_before",
        "prior_min_rss_margin": "min_rss_margin_before",
        "prior_min_ttc": "min_ttc_before",
        "prior_min_gap": "min_gap_before",
    }
    arrays = {rename.get(key, key): value for key, value in arrays.items()}
    arrays["dataset_index"] = selected.astype(np.int64)
    arrays["source_name"] = np.asarray([source_name] * max_contexts)
    np.savez_compressed(output_path, **arrays)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    save_json(
        {
            "split": split,
            "source": source_name,
            "num_contexts": max_contexts,
            "output_path": str(output_path),
            **_sample_summary(arrays),
        },
        summary_path,
    )
    logger.info("Saved KING-guided samples to %s", output_path)
    logger.info("Saved KING-guided sample summary to %s", summary_path)


if __name__ == "__main__":
    main()
