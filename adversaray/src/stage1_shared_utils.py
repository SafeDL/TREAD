"""Shared diagnostics for Stage 1 proposal training and scenario banks."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from .ego_surrogate import IDMSurrogateParams
from .guidance_losses import physical_violation_penalty
from .king_gradient_guidance import _king_config, _physics_config, compute_king_risk
from .torch_kinematics import integrate_following_actions_torch


RISK_TYPE_LABELS = ("low_gap", "low_ttc", "rss_violation", "high_drac", "hard_brake_inducing")


def rollout_proxy_diagnostics(
    actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict[str, Any],
    config: dict[str, Any],
    *,
    ego_surrogate_params: IDMSurrogateParams | None = None,
) -> dict[str, torch.Tensor]:
    king_cfg = _king_config(config)
    kin = integrate_following_actions_torch(
        actions,
        context_states,
        ego_length,
        adv_length,
        schema,
        config,
        ego_surrogate_params=ego_surrogate_params,
    )
    risk, risk_diag = compute_king_risk(kin, config)
    physics, physics_diag = physical_violation_penalty(kin, _physics_config(config, king_cfg))
    closing_speed = kin.ego_velocity - kin.velocity
    drac = torch.where(
        closing_speed > 0.0,
        closing_speed.square() / torch.clamp(2.0 * kin.gap, min=max(float(king_cfg.get("gap_eps", 0.5)), 1e-6)),
        torch.zeros_like(closing_speed),
    )
    return {
        "risk_objective": risk,
        "physics_penalty": physics,
        "min_ego_acceleration": torch.min(kin.ego_acceleration, dim=1).values,
        "drac": torch.max(drac, dim=1).values,
        **risk_diag,
        **physics_diag,
    }


def classify_risk_types(diag: dict[str, torch.Tensor]) -> torch.Tensor:
    scores = torch.stack(
        [
            torch.clamp((5.0 - diag["min_gap"]) / 5.0, min=0.0),
            torch.clamp((3.0 - diag["min_ttc"]) / 3.0, min=0.0),
            torch.clamp(-diag["min_rss_margin"] / 20.0, min=0.0),
            torch.clamp(diag["drac"] / 8.0, min=0.0),
            torch.clamp((-diag["min_ego_acceleration"] - 3.0) / 5.0, min=0.0),
        ],
        dim=-1,
    )
    return torch.argmax(scores, dim=-1)


def risk_type_summary(risk_type: np.ndarray | torch.Tensor) -> dict[str, Any]:
    values = risk_type.detach().cpu().numpy() if isinstance(risk_type, torch.Tensor) else np.asarray(risk_type)
    values = values.reshape(-1).astype(np.int64)
    total = max(int(values.size), 1)
    count = {label: int(np.sum(values == idx)) for idx, label in enumerate(RISK_TYPE_LABELS)}
    ratio = {label: float(count[label] / total) for label in RISK_TYPE_LABELS}
    probs = np.asarray([ratio[label] for label in RISK_TYPE_LABELS], dtype=np.float64)
    probs = probs[probs > 0.0]
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    return {
        "risk_type_entropy": entropy,
        "risk_type_count": count,
        "risk_type_ratio": ratio,
    }


def risk_type_names(risk_type: np.ndarray | torch.Tensor) -> np.ndarray:
    values = risk_type.detach().cpu().numpy() if isinstance(risk_type, torch.Tensor) else np.asarray(risk_type)
    return np.asarray([RISK_TYPE_LABELS[int(item)] for item in values.reshape(-1)])


def latent_diversity_reward(delta_actions: torch.Tensor, *, contexts: int, candidates_per_context: int) -> torch.Tensor:
    if contexts <= 0 or candidates_per_context <= 1:
        return torch.zeros((), dtype=delta_actions.dtype, device=delta_actions.device)
    shaped = delta_actions.reshape(int(contexts), int(candidates_per_context), -1)
    distances = torch.cdist(shaped, shaped, p=2)
    mask = ~torch.eye(int(candidates_per_context), dtype=torch.bool, device=delta_actions.device)[None]
    return distances[mask.expand_as(distances)].mean()


def risk_coverage_reward(diag: dict[str, torch.Tensor]) -> torch.Tensor:
    scores = torch.stack(
        [
            torch.clamp((5.0 - diag["min_gap"]) / 5.0, min=0.0),
            torch.clamp((3.0 - diag["min_ttc"]) / 3.0, min=0.0),
            torch.clamp(-diag["min_rss_margin"] / 20.0, min=0.0),
            torch.clamp(diag["drac"] / 8.0, min=0.0),
            torch.clamp((-diag["min_ego_acceleration"] - 3.0) / 5.0, min=0.0),
        ],
        dim=-1,
    )
    soft_types = torch.softmax(scores, dim=-1).mean(dim=0)
    return -torch.sum(soft_types * torch.log(torch.clamp(soft_types, min=1e-8)))


def action_template_diversity_reward(template_params: torch.Tensor) -> torch.Tensor:
    if template_params.shape[0] <= 1:
        return torch.zeros((), dtype=template_params.dtype, device=template_params.device)
    selected = template_params[:, :3]
    return selected.std(dim=0, unbiased=False).mean()


def tensor_stats(values: np.ndarray | torch.Tensor, prefix: str) -> dict[str, float]:
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    arr = arr.astype(np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p05": float(np.percentile(arr, 5.0)),
        f"{prefix}_p95": float(np.percentile(arr, 95.0)),
    }


def template_diversity_summary(template_params: np.ndarray | torch.Tensor) -> dict[str, float]:
    arr = template_params.detach().cpu().numpy() if isinstance(template_params, torch.Tensor) else np.asarray(template_params)
    out: dict[str, float] = {}
    for idx, key in enumerate(("brake_start", "brake_duration", "brake_intensity")):
        out.update(tensor_stats(arr[..., idx], key))
    return out


def update_counter(counter: Counter[str], risk_type: np.ndarray | torch.Tensor) -> None:
    for name in risk_type_names(risk_type):
        counter[str(name)] += 1
