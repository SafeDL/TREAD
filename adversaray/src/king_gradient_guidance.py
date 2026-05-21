"""KING-style differentiable action-plan guidance for following scenarios."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .physics_losses import physical_violation_penalty
from .rss import RSSConfig, rss_margin, softmax_pool
from .torch_kinematics import FollowingKinematics, integrate_following_actions_torch


def _king_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("king_gradient", {}))
    defaults = {
        "enabled": True,
        "num_steps": 50,
        "step_size": 2.0,
        "grad_clip_norm": 1.0,
        "rss_weight": 3.0,
        "ttc_weight": 1.0,
        "drac_weight": 1.0,
        "gap_weight": 0.5,
        "rss_scale": 100.0,
        "ttc_scale": 1.0,
        "drac_scale": 5.0,
        "gap_scale": 20.0,
        "ttc_eps": 0.2,
        "gap_eps": 0.5,
        "gap_target": 20.0,
        "pool_beta": 8.0,
        "lambda_nat": 0.0,
        "lambda_phys": 0.2,
        "jerk_min": -12.0,
        "jerk_max": 12.0,
        "accel_min": -8.0,
        "accel_max": 4.0,
        "speed_min": 0.0,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def _require_jerk_schema(schema: dict[str, Any], config: dict[str, Any]) -> None:
    rep = str(schema.get("action_representation", config.get("action", {}).get("representation", "jerk"))).lower()
    if rep != "jerk":
        raise ValueError(f"KING gradient guidance v1 expects raw jerk actions, got action_representation={rep!r}")


def _physics_config(config: dict[str, Any], king_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    physics = dict(out.get("physics", {}))
    physics["ax_min"] = float(king_cfg.get("accel_min", physics.get("ax_min", -8.0)))
    physics["ax_max"] = float(king_cfg.get("accel_max", physics.get("ax_max", 4.0)))
    physics["jerk_abs_max"] = max(
        abs(float(king_cfg.get("jerk_min", -12.0))),
        abs(float(king_cfg.get("jerk_max", 12.0))),
    )
    physics["speed_min"] = float(king_cfg.get("speed_min", physics.get("speed_min", 0.0)))
    out["physics"] = physics
    return out


def _safe_min_ttc(kin: FollowingKinematics, king_cfg: dict[str, Any]) -> torch.Tensor:
    closing_speed = kin.ego_velocity - kin.velocity
    eps = float(king_cfg.get("ttc_eps", 0.2))
    ttc = torch.where(
        closing_speed > 0.0,
        kin.gap / torch.clamp(closing_speed, min=max(eps, 1e-6)),
        torch.full_like(kin.gap, 1000.0),
    )
    return torch.min(torch.clamp(ttc, min=0.0, max=1000.0), dim=1).values


def compute_king_risk(
    kin: FollowingKinematics,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the differentiable KING proxy risk for an integrated plan."""
    king_cfg = _king_config(config)
    rss_cfg = RSSConfig.from_config(config)
    margin, safe = rss_margin(kin, rss_cfg)
    pool_beta = float(king_cfg.get("pool_beta", rss_cfg.pool_beta))
    temperature = max(float(rss_cfg.temperature), 1e-6)

    rss_per_t = F.softplus((safe - kin.gap) / temperature)
    rss_objective_raw = softmax_pool(rss_per_t, beta=pool_beta, dim=1)

    closing_speed = kin.ego_velocity - kin.velocity
    ttc_eps = max(float(king_cfg.get("ttc_eps", 0.2)), 1e-6)
    gap_eps = max(float(king_cfg.get("gap_eps", 0.5)), 1e-6)
    positive_closing = closing_speed > 0.0

    ttc = torch.where(
        positive_closing,
        torch.clamp(kin.gap, min=gap_eps) / torch.clamp(closing_speed, min=ttc_eps),
        torch.full_like(kin.gap, 1000.0),
    )
    ttc_per_t = torch.where(positive_closing, 1.0 / torch.clamp(ttc, min=ttc_eps), torch.zeros_like(ttc))
    ttc_objective_raw = softmax_pool(ttc_per_t, beta=pool_beta, dim=1)

    drac = closing_speed.square() / torch.clamp(2.0 * kin.gap, min=gap_eps)
    drac = torch.where(positive_closing, drac, torch.zeros_like(drac))
    drac_objective_raw = softmax_pool(drac, beta=pool_beta, dim=1)

    gap_target = float(king_cfg.get("gap_target", 20.0))
    gap_per_t = F.softplus(gap_target - kin.gap)
    gap_objective_raw = softmax_pool(gap_per_t, beta=pool_beta, dim=1)

    rss_scale = max(float(king_cfg.get("rss_scale", 100.0)), 1e-6)
    ttc_scale = max(float(king_cfg.get("ttc_scale", 1.0)), 1e-6)
    drac_scale = max(float(king_cfg.get("drac_scale", 5.0)), 1e-6)
    gap_scale = max(float(king_cfg.get("gap_scale", 20.0)), 1e-6)
    rss_objective = rss_objective_raw / rss_scale
    ttc_objective = ttc_objective_raw / ttc_scale
    drac_objective = drac_objective_raw / drac_scale
    gap_objective = gap_objective_raw / gap_scale

    objective = (
        float(king_cfg.get("rss_weight", 3.0)) * rss_objective
        + float(king_cfg.get("ttc_weight", 1.0)) * ttc_objective
        + float(king_cfg.get("drac_weight", 1.0)) * drac_objective
        + float(king_cfg.get("gap_weight", 0.5)) * gap_objective
    )
    return objective, {
        "rss_objective": rss_objective,
        "ttc_objective": ttc_objective,
        "drac_objective": drac_objective,
        "gap_objective": gap_objective,
        "risk_objective": objective,
        "rss_margin": margin,
        "rss_safe_distance": safe,
        "min_rss_margin": torch.min(margin, dim=1).values,
        "min_gap": torch.min(kin.gap, dim=1).values,
        "min_ttc": _safe_min_ttc(kin, king_cfg),
    }


def _diagnostics_for_actions(
    actions: torch.Tensor,
    raw_context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    king_cfg = _king_config(config)
    kin = integrate_following_actions_torch(actions, raw_context_states, ego_length, adv_length, schema, config)
    risk, risk_diag = compute_king_risk(kin, config)
    physics, physics_diag = physical_violation_penalty(kin, _physics_config(config, king_cfg))
    diagnostics = {
        "risk_objective": risk,
        "physics_penalty": physics,
        **risk_diag,
        **physics_diag,
    }
    diagnostics.update(
        {
            "ego_accel_min": torch.min(kin.ego_acceleration, dim=1).values,
            "ego_accel_mean": torch.mean(kin.ego_acceleration, dim=1),
            "ego_speed_min": torch.min(kin.ego_velocity, dim=1).values,
            "ego_speed_mean": torch.mean(kin.ego_velocity, dim=1),
        }
    )
    return diagnostics


def optimize_action_plan_king(
    prior_actions: torch.Tensor,
    raw_context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Optimize a frozen-prior raw jerk plan with differentiable risk gradients."""
    if prior_actions.ndim != 3 or prior_actions.shape[-1] < 1:
        raise ValueError(f"Expected prior_actions shape [B,H,1+], got {tuple(prior_actions.shape)}")
    _require_jerk_schema(schema, config)

    king_cfg = _king_config(config)
    device = prior_actions.device
    raw_context_states = raw_context_states.to(device=device, dtype=prior_actions.dtype)
    ego_length = None if ego_length is None else ego_length.to(device=device, dtype=prior_actions.dtype)
    adv_length = None if adv_length is None else adv_length.to(device=device, dtype=prior_actions.dtype)

    j0 = prior_actions.detach()
    j = j0.clone().detach().requires_grad_(True)
    num_steps = max(int(king_cfg.get("num_steps", 50)), 0) if bool(king_cfg.get("enabled", True)) else 0
    step_size = float(king_cfg.get("step_size", 2.0))
    grad_clip_norm = float(king_cfg.get("grad_clip_norm", 1.0))
    jerk_min = float(king_cfg.get("jerk_min", -12.0))
    jerk_max = float(king_cfg.get("jerk_max", 12.0))
    lambda_nat = float(king_cfg.get("lambda_nat", 0.0))
    lambda_phys = float(king_cfg.get("lambda_phys", 0.2))
    physics_cfg = _physics_config(config, king_cfg)
    trace: list[dict[str, float]] = []

    for step in range(num_steps):
        kin = integrate_following_actions_torch(j, raw_context_states, ego_length, adv_length, schema, config)
        risk, risk_diag = compute_king_risk(kin, config)
        naturalness = (j - j0).square().flatten(1).mean(dim=1)
        physics, _physics_diag = physical_violation_penalty(kin, physics_cfg)
        objective = risk - lambda_nat * naturalness - lambda_phys * physics
        grad = torch.autograd.grad(objective.sum(), j, retain_graph=False, create_graph=False)[0]
        grad_norm = torch.clamp(grad.flatten(1).norm(dim=1), min=1e-12)
        if grad_clip_norm > 0.0:
            grad = grad * (grad_clip_norm / grad_norm).view(-1, 1, 1)
            grad_norm = grad.flatten(1).norm(dim=1)
        with torch.no_grad():
            trace.append(
                {
                    "step": float(step),
                    "objective": float(objective.detach().mean().cpu()),
                    "risk": float(risk.detach().mean().cpu()),
                    "rss_objective": float(risk_diag["rss_objective"].detach().mean().cpu()),
                    "ttc_objective": float(risk_diag["ttc_objective"].detach().mean().cpu()),
                    "drac_objective": float(risk_diag["drac_objective"].detach().mean().cpu()),
                    "gap_objective": float(risk_diag["gap_objective"].detach().mean().cpu()),
                    "naturalness_penalty": float(naturalness.detach().mean().cpu()),
                    "physics_penalty": float(physics.detach().mean().cpu()),
                    "grad_norm": float(grad_norm.detach().mean().cpu()),
                    "ego_accel_mean": float(kin.ego_acceleration.detach().mean().cpu()),
                    "ego_speed_mean": float(kin.ego_velocity.detach().mean().cpu()),
                }
            )
            j = torch.clamp(j + step_size * grad, min=jerk_min, max=jerk_max).detach().requires_grad_(True)

    adv_actions = j.detach()
    final_diag = _diagnostics_for_actions(adv_actions, raw_context_states, ego_length, adv_length, schema, config)
    prior_diag = _diagnostics_for_actions(j0, raw_context_states, ego_length, adv_length, schema, config)
    naturalness = (adv_actions - j0).square().flatten(1).mean(dim=1)
    return {
        "adv_actions": adv_actions,
        "prior_actions": j0,
        "risk_trace": trace,
        "risk_before": prior_diag["risk_objective"].detach(),
        "risk_after": final_diag["risk_objective"].detach(),
        "rss_before": prior_diag["min_rss_margin"].detach(),
        "rss_after": final_diag["min_rss_margin"].detach(),
        "ttc_before": prior_diag["min_ttc"].detach(),
        "ttc_after": final_diag["min_ttc"].detach(),
        "gap_before": prior_diag["min_gap"].detach(),
        "gap_after": final_diag["min_gap"].detach(),
        "naturalness_penalty": naturalness.detach(),
        "physics_penalty": final_diag["physics_penalty"].detach(),
        **{key: value.detach() for key, value in final_diag.items()},
        **{f"prior_{key}": value.detach() for key, value in prior_diag.items()},
    }
