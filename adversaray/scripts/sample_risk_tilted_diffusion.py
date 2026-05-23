#!/usr/bin/env python3
"""Sample receding-horizon risk-tilted diffusion trajectories."""
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

from adversaray.src.adversary_dynamics import integrate_adversary_actions_torch
from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
from adversaray.src.context_utils import _context, _load_npz
from adversaray.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from adversaray.src.king_gradient_guidance import compute_king_risk
from adversaray.src.physics_losses import physical_violation_penalty
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "king_guided_following.yaml"
)
SCRIPT_DEFAULTS = {
    "split": "val",
    "num_contexts": 256,
    "seed": 42,
    "output_name": "risk_tilted_samples.npz",
    "log_level": "INFO",
}
RISK_TILTED_DEFAULTS = {
    "enabled": True,
    "late_fraction": 0.40,
    "num_late_steps": 0,
    "guidance_scale": 20.0,
    "scale_schedule": "linear_ramp",
    "guidance_variance_mode": "posterior_variance",
    "max_grad_norm": 1.0,
    "normalize_grad": True,
    "scale_by_sqrt_dim": True,
    "apply_at_t0": False,
    "lambda_phys": 0.2,
    "lambda_action_l2": 0.0,
    "min_grad_norm": 1.0e-12,
    "nan_to_num": True,
    "save_guidance_diagnostics": True,
}
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _effective_risk_tilted_config(cfg: dict[str, Any]) -> dict[str, Any]:
    tilted_cfg = dict(RISK_TILTED_DEFAULTS)
    tilted_cfg.update(dict(cfg.get("risk_tilted_diffusion", {})))
    tilted_cfg["enabled"] = True
    return tilted_cfg


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
            "training.tail_context_path must be set for risk-tilted sampling"
        )
    path = _resolve(tail_context_value, base)
    raw = _load_npz(path)
    required = {"context_states", "split_index"}
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")
    return raw, _split_indices(raw, split), "tail_natural"


def _event_steps(ctx: dict[str, Any], cfg: dict[str, Any]) -> int:
    if "event_steps" in ctx:
        steps = int(ctx["event_steps"])
    else:
        steps = int(cfg.get("env", {}).get("episode_steps", 50))
    if steps <= 0:
        raise ValueError(f"event_steps must be positive, got {steps}")
    return steps


def _diagnostics_for_actions(
    actions: torch.Tensor,
    raw_context: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    sampler: FrozenDiffusionSampler,
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    kin = integrate_adversary_actions_torch(
        actions,
        raw_context,
        ego_length,
        adv_length,
        sampler.prior.schema,
        cfg,
    )
    risk, risk_diag = compute_king_risk(kin, cfg)
    physics, physics_diag = physical_violation_penalty(kin, cfg)
    return {
        "risk_objective": risk.detach(),
        "physics_penalty": physics.detach(),
        **{key: value.detach() for key, value in risk_diag.items()},
        **{key: value.detach() for key, value in physics_diag.items()},
    }


def _to_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(np.asarray(value, dtype=np.float64).mean())


def _prefixed_plan_summary(
    prefix: str,
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, float]:
    mapping = {
        "risk_objective": f"{prefix}_risk_objective",
        "min_gap": f"{prefix}_min_gap",
        "min_ttc": f"{prefix}_min_ttc",
        "min_rss_margin": f"{prefix}_min_rss_margin",
        "physics_penalty": f"{prefix}_physics_penalty",
        "negative_speed_rate": f"{prefix}_negative_speed_rate",
        "jerk_violation_rate": f"{prefix}_jerk_violation_rate",
        "ax_violation_rate": f"{prefix}_ax_violation_rate",
    }
    out: dict[str, float] = {}
    for src, dst in mapping.items():
        if src not in diagnostics:
            continue
        value = _to_scalar(diagnostics[src])
        if np.isfinite(value):
            out[dst] = value
    return out


def _guidance_summary(
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, float]:
    mapping = {
        "guidance_steps": "tilted_guidance_steps",
        "guidance_risk": "tilted_guidance_risk_mean",
        "guidance_physics": "tilted_guidance_physics_mean",
        "guidance_grad_norm": "tilted_guidance_grad_norm_mean",
        "guidance_scale": "tilted_guidance_scale_mean",
        "guidance_variance_multiplier": (
            "tilted_guidance_variance_multiplier_mean"
        ),
        "guidance_effective_scale": "tilted_guidance_effective_scale_mean",
    }
    out: dict[str, float] = {}
    for src, dst in mapping.items():
        if src not in diagnostics:
            continue
        value = _to_scalar(diagnostics[src])
        if np.isfinite(value):
            out[dst] = value
    return out


def _make_tilted_plan_callback(
    *,
    sampler: FrozenDiffusionSampler,
    cfg: dict[str, Any],
    risk_tilted_config: dict[str, Any],
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
                risk_tilted=False,
            )
        tilted_sample = sampler.sample(
            context_states,
            context_features,
            relative_history,
            ego_length=torch.tensor([ego_length], dtype=torch.float32),
            adv_length=torch.tensor([adv_length], dtype=torch.float32),
            num_samples=1,
            seed=local_seed,
            risk_tilted=True,
            risk_tilted_config=risk_tilted_config,
        )
        raw_context = torch.from_numpy(obs["raw_context_states"][None]).to(
            device=device,
            dtype=torch.float32,
        )
        ego_len = torch.tensor([ego_length], dtype=torch.float32, device=device)
        adv_len = torch.tensor([adv_length], dtype=torch.float32, device=device)
        prior_diag = _diagnostics_for_actions(
            prior_sample.raw_actions.to(device),
            raw_context,
            ego_len,
            adv_len,
            sampler,
            cfg,
        )
        tilted_diag = _diagnostics_for_actions(
            tilted_sample.raw_actions.to(device),
            raw_context,
            ego_len,
            adv_len,
            sampler,
            cfg,
        )
        summary = {
            **_prefixed_plan_summary("prior", prior_diag),
            **_prefixed_plan_summary("tilted", tilted_diag),
            **_guidance_summary(tilted_sample.diagnostics),
        }
        return {
            "plan": _tensor_to_numpy(tilted_sample.raw_actions)[0],
            "prior_plan": _tensor_to_numpy(prior_sample.raw_actions)[0],
            "summary": summary,
        }

    return callback


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


def _sample_receding_case(
    *,
    runner: ClosedLoopFollowingRunner,
    sampler: FrozenDiffusionSampler,
    cfg: dict[str, Any],
    risk_tilted_config: dict[str, Any],
    ctx: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    steps = _event_steps(ctx, cfg)
    prior_result = runner.rollout(
        ctx,
        seed=seed,
        episode_steps=steps,
    )
    callback = _make_tilted_plan_callback(
        sampler=sampler,
        cfg=cfg,
        risk_tilted_config=risk_tilted_config,
        ego_length=float(ctx["ego_length"]),
        adv_length=float(ctx["adv_length"]),
        seed=seed,
    )
    tilted_result = runner.rollout(
        ctx,
        seed=seed,
        episode_steps=steps,
        plan_callback=callback,
    )
    if prior_result.actions is None or tilted_result.actions is None:
        raise RuntimeError("Closed-loop rollout did not return executed actions")
    if tilted_result.prior_actions is None:
        raise RuntimeError(
            "Risk-tilted rollout did not return reference prior actions"
        )
    summaries = tilted_result.plan_summaries
    scalars = {
        "closed_loop_risk_prior": float(prior_result.closed_loop_risk),
        "closed_loop_risk_tilted": float(tilted_result.closed_loop_risk),
        "num_plans_prior": float(prior_result.num_generated_plans),
        "num_plans_tilted": float(tilted_result.num_generated_plans),
    }
    for key in (
        "prior_risk_objective",
        "prior_min_gap",
        "prior_min_ttc",
        "prior_min_rss_margin",
        "prior_physics_penalty",
        "prior_negative_speed_rate",
        "prior_jerk_violation_rate",
        "prior_ax_violation_rate",
        "tilted_risk_objective",
        "tilted_min_gap",
        "tilted_min_ttc",
        "tilted_min_rss_margin",
        "tilted_physics_penalty",
        "tilted_negative_speed_rate",
        "tilted_jerk_violation_rate",
        "tilted_ax_violation_rate",
        "tilted_guidance_steps",
        "tilted_guidance_risk_mean",
        "tilted_guidance_physics_mean",
        "tilted_guidance_grad_norm_mean",
        "tilted_guidance_scale_mean",
        "tilted_guidance_variance_multiplier_mean",
        "tilted_guidance_effective_scale_mean",
    ):
        scalars[key] = _mean_plan_summaries(summaries, key)
    return {
        "prior_actions": prior_result.actions,
        "tilted_actions": tilted_result.actions,
        "tilted_reference_actions": tilted_result.prior_actions,
        "event_steps": int(steps),
        "scalars": scalars,
    }


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


def _sample_summary(
    arrays: dict[str, np.ndarray],
    risk_tilted_config: dict[str, Any],
) -> dict[str, Any]:
    prior_risk = _array_mean(arrays, "prior_risk_objective")
    tilted_risk = _array_mean(arrays, "tilted_risk_objective")
    action_l2 = _masked_action_l2(
        arrays["tilted_actions"],
        arrays["tilted_reference_actions"],
        arrays.get("tilted_action_mask"),
    )
    return {
        "prior_risk_mean": prior_risk,
        "tilted_risk_mean": tilted_risk,
        "tilted_minus_prior_risk_mean": tilted_risk - prior_risk,
        "prior_closed_loop_risk_mean": _array_mean(
            arrays,
            "closed_loop_risk_prior",
        ),
        "tilted_closed_loop_risk_mean": _array_mean(
            arrays,
            "closed_loop_risk_tilted",
        ),
        "tilted_minus_prior_closed_loop_risk_mean": (
            _array_mean(arrays, "closed_loop_risk_tilted")
            - _array_mean(arrays, "closed_loop_risk_prior")
        ),
        "prior_min_gap_mean": _array_mean(arrays, "prior_min_gap"),
        "tilted_min_gap_mean": _array_mean(arrays, "tilted_min_gap"),
        "prior_min_ttc_mean": _array_mean(arrays, "prior_min_ttc"),
        "tilted_min_ttc_mean": _array_mean(arrays, "tilted_min_ttc"),
        "prior_min_rss_margin_mean": _array_mean(
            arrays,
            "prior_min_rss_margin",
        ),
        "tilted_min_rss_margin_mean": _array_mean(
            arrays,
            "tilted_min_rss_margin",
        ),
        "tilted_physics_penalty_mean": _array_mean(
            arrays,
            "tilted_physics_penalty",
        ),
        "tilted_guidance_variance_multiplier_mean": _array_mean(
            arrays,
            "tilted_guidance_variance_multiplier_mean",
        ),
        "tilted_guidance_effective_scale_mean": _array_mean(
            arrays,
            "tilted_guidance_effective_scale_mean",
        ),
        "tilted_action_l2_from_reference_mean": float(np.mean(action_l2)),
        "event_steps_mean": _array_mean(arrays, "event_steps"),
        "num_plans_tilted_mean": _array_mean(arrays, "num_plans_tilted"),
        "risk_tilted_diffusion": dict(risk_tilted_config),
    }


def main() -> None:
    setup_logging(SCRIPT_DEFAULTS["log_level"])

    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    paths = cfg.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    output_dir = _resolve(paths["output_dir"], base)
    output_path = output_dir / str(SCRIPT_DEFAULTS["output_name"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    risk_tilted_config = _effective_risk_tilted_config(cfg)

    split = str(SCRIPT_DEFAULTS["split"])
    raw, idx, source_name = _select_raw_contexts(cfg, base, split)
    max_contexts = min(int(SCRIPT_DEFAULTS["num_contexts"]), int(idx.size))
    selected = idx[:max_contexts]
    if max_contexts <= 0:
        raise ValueError("No contexts selected for risk-tilted sampling")

    sampler = FrozenDiffusionSampler.from_config(cfg, config_dir=base).eval()
    if any(param.requires_grad for param in sampler.prior.model.parameters()):
        raise RuntimeError("Frozen diffusion prior has trainable parameters")
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
        "closed_loop_risk_prior",
        "closed_loop_risk_tilted",
        "num_plans_prior",
        "num_plans_tilted",
        "prior_risk_objective",
        "prior_min_gap",
        "prior_min_ttc",
        "prior_min_rss_margin",
        "prior_physics_penalty",
        "prior_negative_speed_rate",
        "prior_jerk_violation_rate",
        "prior_ax_violation_rate",
        "tilted_risk_objective",
        "tilted_min_gap",
        "tilted_min_ttc",
        "tilted_min_rss_margin",
        "tilted_physics_penalty",
        "tilted_negative_speed_rate",
        "tilted_jerk_violation_rate",
        "tilted_ax_violation_rate",
        "tilted_guidance_steps",
        "tilted_guidance_risk_mean",
        "tilted_guidance_physics_mean",
        "tilted_guidance_grad_norm_mean",
        "tilted_guidance_scale_mean",
        "tilted_guidance_variance_multiplier_mean",
        "tilted_guidance_effective_scale_mean",
    )

    prior_sequences: list[np.ndarray] = []
    tilted_sequences: list[np.ndarray] = []
    reference_sequences: list[np.ndarray] = []
    scalar_values: dict[str, list[float]] = {key: [] for key in scalar_keys}
    for pos, ctx in enumerate(contexts):
        seed = int(SCRIPT_DEFAULTS["seed"]) + int(pos)
        result = _sample_receding_case(
            runner=runner,
            sampler=sampler,
            cfg=cfg,
            risk_tilted_config=risk_tilted_config,
            ctx=ctx,
            seed=seed,
        )
        prior_sequences.append(result["prior_actions"])
        tilted_sequences.append(result["tilted_actions"])
        reference_sequences.append(result["tilted_reference_actions"])
        for key in scalar_keys:
            scalar_values[key].append(
                float(result["scalars"].get(key, np.nan))
            )
        logger.info(
            "Risk-tilted event %d/%d steps=%d risk %.4f -> %.4f",
            pos + 1,
            max_contexts,
            int(result["event_steps"]),
            float(result["scalars"].get("closed_loop_risk_prior", np.nan)),
            float(result["scalars"].get("closed_loop_risk_tilted", np.nan)),
        )

    max_action_steps = max(
        max(seq.shape[0] for seq in prior_sequences),
        max(seq.shape[0] for seq in tilted_sequences),
        max(seq.shape[0] for seq in reference_sequences),
    )
    action_dim = max(
        max(seq.shape[1] for seq in prior_sequences),
        max(seq.shape[1] for seq in tilted_sequences),
        max(seq.shape[1] for seq in reference_sequences),
    )
    prior_actions, prior_mask = _pad_actions(
        prior_sequences,
        max_steps=int(max_action_steps),
        action_dim=int(action_dim),
    )
    tilted_actions, tilted_mask = _pad_actions(
        tilted_sequences,
        max_steps=int(max_action_steps),
        action_dim=int(action_dim),
    )
    reference_actions, reference_mask = _pad_actions(
        reference_sequences,
        max_steps=int(max_action_steps),
        action_dim=int(action_dim),
    )
    if not np.array_equal(tilted_mask, reference_mask):
        raise RuntimeError(
            "Risk-tilted reference action mask does not match output"
        )
    output["prior_actions"] = prior_actions
    output["tilted_actions"] = tilted_actions
    output["tilted_reference_actions"] = reference_actions
    output["action_mask"] = tilted_mask
    output["prior_action_mask"] = prior_mask
    output["tilted_action_mask"] = tilted_mask
    for key, values in scalar_values.items():
        output[key] = np.asarray(values, dtype=np.float32)

    arrays = dict(output)
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
            **_sample_summary(arrays, risk_tilted_config),
        },
        summary_path,
    )
    logger.info("Saved risk-tilted samples to %s", output_path)
    logger.info("Saved risk-tilted sample summary to %s", summary_path)


if __name__ == "__main__":
    main()
