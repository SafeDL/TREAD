"""Differentiable longitudinal proxy risk for Stage 1 proposals."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .rss import RSSConfig, rss_margin, softmax_pool
from .torch_kinematics import FollowingKinematics


def proxy_risk_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("proxy_risk", {}))
    defaults = {
        "rss_weight": 3.0,
        "ttc_weight": 1.0,
        "drac_weight": 1.0,
        "gap_weight": 0.5,
        "ttc_eps": 0.2,
        "gap_eps": 0.5,
        "gap_target": 20.0,
        "pool_beta": 8.0,
        "accel_min": -8.0,
        "accel_max": 4.0,
        "jerk_min": -12.0,
        "jerk_max": 12.0,
        "speed_min": 0.0,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def physics_config(config: dict[str, Any], risk_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    physics = dict(out.get("physics", {}))
    physics["ax_min"] = float(risk_cfg.get("accel_min", physics.get("ax_min", -8.0)))
    physics["ax_max"] = float(risk_cfg.get("accel_max", physics.get("ax_max", 4.0)))
    physics["jerk_abs_max"] = max(
        abs(float(risk_cfg.get("jerk_min", -12.0))),
        abs(float(risk_cfg.get("jerk_max", 12.0))),
    )
    physics["speed_min"] = float(risk_cfg.get("speed_min", physics.get("speed_min", 0.0)))
    out["physics"] = physics
    return out


def _safe_min_ttc(kin: FollowingKinematics, risk_cfg: dict[str, Any]) -> torch.Tensor:
    closing_speed = kin.ego_velocity - kin.velocity
    eps = float(risk_cfg.get("ttc_eps", 0.2))
    ttc = torch.where(
        closing_speed > 0.0,
        kin.gap / torch.clamp(closing_speed, min=max(eps, 1e-6)),
        torch.full_like(kin.gap, 1000.0),
    )
    return torch.min(torch.clamp(ttc, min=0.0, max=1000.0), dim=1).values


def compute_proxy_risk(
    kin: FollowingKinematics,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    risk_cfg = proxy_risk_config(config)
    rss_cfg = RSSConfig.from_config(config)
    margin, safe = rss_margin(kin, rss_cfg)
    pool_beta = float(risk_cfg.get("pool_beta", rss_cfg.pool_beta))
    temperature = max(float(rss_cfg.temperature), 1e-6)

    rss_per_t = F.softplus((safe - kin.gap) / temperature)
    rss_objective = softmax_pool(rss_per_t, beta=pool_beta, dim=1)

    closing_speed = kin.ego_velocity - kin.velocity
    ttc_eps = max(float(risk_cfg.get("ttc_eps", 0.2)), 1e-6)
    gap_eps = max(float(risk_cfg.get("gap_eps", 0.5)), 1e-6)
    positive_closing = closing_speed > 0.0

    ttc = torch.where(
        positive_closing,
        torch.clamp(kin.gap, min=gap_eps) / torch.clamp(closing_speed, min=ttc_eps),
        torch.full_like(kin.gap, 1000.0),
    )
    ttc_per_t = torch.where(positive_closing, 1.0 / torch.clamp(ttc, min=ttc_eps), torch.zeros_like(ttc))
    ttc_objective = softmax_pool(ttc_per_t, beta=pool_beta, dim=1)

    drac = closing_speed.square() / torch.clamp(2.0 * kin.gap, min=gap_eps)
    drac = torch.where(positive_closing, drac, torch.zeros_like(drac))
    drac_objective = softmax_pool(drac, beta=pool_beta, dim=1)

    gap_target = float(risk_cfg.get("gap_target", 20.0))
    gap_objective = softmax_pool(F.softplus(gap_target - kin.gap), beta=pool_beta, dim=1)

    objective = (
        float(risk_cfg.get("rss_weight", 3.0)) * rss_objective
        + float(risk_cfg.get("ttc_weight", 1.0)) * ttc_objective
        + float(risk_cfg.get("drac_weight", 1.0)) * drac_objective
        + float(risk_cfg.get("gap_weight", 0.5)) * gap_objective
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
        "min_ttc": _safe_min_ttc(kin, risk_cfg),
    }
