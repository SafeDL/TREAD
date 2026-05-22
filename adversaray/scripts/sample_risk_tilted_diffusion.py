#!/usr/bin/env python3
"""Sample prior and risk-tilted diffusion plans."""
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
from adversaray.src.context_utils import (
    _batch_observation_for_contexts,
    _context,
    _load_npz,
)
from adversaray.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from adversaray.src.king_gradient_guidance import compute_king_risk
from adversaray.src.physics_losses import physical_violation_penalty
from adversaray.src.adversary_dynamics import integrate_adversary_actions_torch
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
    "batch_size": 16,
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


def _append_tensor(
    out: dict[str, list[np.ndarray]],
    key: str,
    value: torch.Tensor,
) -> None:
    out.setdefault(key, []).append(_tensor_to_numpy(value))


def _effective_risk_tilted_config(cfg: dict[str, Any]) -> dict[str, Any]:
    tilted_cfg = dict(cfg.get("risk_tilted_diffusion", {}))
    tilted_cfg.update(RISK_TILTED_DEFAULTS)
    return tilted_cfg


def _split_indices(raw: dict[str, np.ndarray], split: str) -> np.ndarray:
    if "split_index" not in raw:
        raise KeyError(
            "Tail contexts must contain split_index; rebuild contexts first."
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
    tail_context_value = str(
        training.get("tail_context_path", "") or ""
    ).strip()
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


def _append_plan_diagnostics(
    output: dict[str, list[np.ndarray]],
    prefix: str,
    diagnostics: dict[str, torch.Tensor],
) -> None:
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
    for src, dst in mapping.items():
        if src in diagnostics:
            _append_tensor(output, dst, diagnostics[src])


def _append_guidance_diagnostics(
    output: dict[str, list[np.ndarray]],
    diagnostics: dict[str, torch.Tensor],
) -> None:
    if "guidance_steps" in diagnostics:
        _append_tensor(
            output,
            "tilted_guidance_steps",
            diagnostics["guidance_steps"],
        )
    for key, out_key in (
        ("guidance_risk", "tilted_guidance_risk_mean"),
        ("guidance_physics", "tilted_guidance_physics_mean"),
        ("guidance_grad_norm", "tilted_guidance_grad_norm_mean"),
        ("guidance_scale", "tilted_guidance_scale_mean"),
        (
            "guidance_variance_multiplier",
            "tilted_guidance_variance_multiplier_mean",
        ),
        (
            "guidance_effective_scale",
            "tilted_guidance_effective_scale_mean",
        ),
    ):
        if key in diagnostics:
            _append_tensor(output, out_key, diagnostics[key].mean(dim=0))


def _array_mean(arrays: dict[str, np.ndarray], key: str) -> float:
    value = arrays.get(key)
    if value is None or value.size == 0:
        return float("nan")
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _action_l2(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    diff = lhs - rhs
    return np.sqrt(np.mean(np.square(diff), axis=tuple(range(1, diff.ndim))))


def _sample_summary(
    arrays: dict[str, np.ndarray],
    risk_tilted_config: dict[str, Any],
) -> dict[str, Any]:
    prior_risk = _array_mean(arrays, "prior_risk_objective")
    tilted_risk = _array_mean(arrays, "tilted_risk_objective")
    summary: dict[str, Any] = {
        "prior_risk_mean": prior_risk,
        "tilted_risk_mean": tilted_risk,
        "tilted_minus_prior_risk_mean": tilted_risk - prior_risk,
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
        "tilted_action_l2_from_prior_mean": float(
            np.mean(
                _action_l2(arrays["tilted_actions"], arrays["prior_actions"])
            )
        ),
        "risk_tilted_diffusion": dict(risk_tilted_config),
    }
    return summary


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
    device = sampler.prior.device

    output: dict[str, list[np.ndarray]] = {
        "context_states": [],
        "ego_length": [],
        "adv_length": [],
        "prior_actions": [],
        "tilted_actions": [],
    }

    batch_size = max(int(SCRIPT_DEFAULTS["batch_size"]), 1)
    for start in range(0, max_contexts, batch_size):
        batch_indices = selected[start : start + batch_size]
        contexts = [_context(raw, int(item)) for item in batch_indices]
        batch, prepared_contexts = _batch_observation_for_contexts(
            runner,
            contexts,
        )
        seeds = [
            int(SCRIPT_DEFAULTS["seed"]) + start + pos
            for pos in range(len(prepared_contexts))
        ]
        with torch.no_grad():
            prior_sample = sampler.sample_batch(
                batch,
                seed=seeds,
                risk_tilted=False,
            )
        tilted_sample = sampler.sample_batch(
            batch,
            seed=seeds,
            risk_tilted=True,
            risk_tilted_config=risk_tilted_config,
        )
        with torch.no_grad():
            raw_context = sampler.prior.decode_context_states(
                batch["context_states"].to(device).float()
            )

        ego_length = batch.get("ego_length")
        adv_length = batch.get("adv_length")
        prior_diag = _diagnostics_for_actions(
            prior_sample.raw_actions.to(device),
            raw_context,
            ego_length,
            adv_length,
            sampler,
            cfg,
        )
        tilted_diag = _diagnostics_for_actions(
            tilted_sample.raw_actions.to(device),
            raw_context,
            ego_length,
            adv_length,
            sampler,
            cfg,
        )

        output["context_states"].append(
            np.stack(
                [ctx["raw_context_states"] for ctx in prepared_contexts],
                axis=0,
            ).astype(np.float32)
        )
        output["ego_length"].append(
            np.asarray(
                [ctx["ego_length"] for ctx in prepared_contexts],
                dtype=np.float32,
            )
        )
        output["adv_length"].append(
            np.asarray(
                [ctx["adv_length"] for ctx in prepared_contexts],
                dtype=np.float32,
            )
        )
        output["prior_actions"].append(
            _tensor_to_numpy(prior_sample.raw_actions)
        )
        output["tilted_actions"].append(
            _tensor_to_numpy(tilted_sample.raw_actions)
        )
        _append_plan_diagnostics(output, "prior", prior_diag)
        _append_plan_diagnostics(output, "tilted", tilted_diag)
        _append_guidance_diagnostics(output, tilted_sample.diagnostics)

        logger.info(
            "Risk-tilted batch %d-%d/%d risk prior %.4f -> tilted %.4f",
            start + 1,
            start + len(prepared_contexts),
            max_contexts,
            float(prior_diag["risk_objective"].mean().cpu()),
            float(tilted_diag["risk_objective"].mean().cpu()),
        )

    arrays = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in output.items()
    }
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
