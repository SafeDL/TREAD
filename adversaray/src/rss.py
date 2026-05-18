"""Differentiable longitudinal RSS objectives."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .torch_kinematics import FollowingKinematics


@dataclass(frozen=True)
class RSSConfig:
    response_time: float = 1.0
    ego_max_accel: float = 2.0
    ego_min_brake: float = 4.0
    lead_max_brake: float = 6.0
    temperature: float = 1.0
    pool_beta: float = 8.0

    @classmethod
    def from_config(cls, config: dict) -> "RSSConfig":
        cfg = config.get("rss", config)
        return cls(
            response_time=float(cfg.get("response_time", 1.0)),
            ego_max_accel=float(cfg.get("ego_max_accel", 2.0)),
            ego_min_brake=float(cfg.get("ego_min_brake", 4.0)),
            lead_max_brake=float(cfg.get("lead_max_brake", 6.0)),
            temperature=float(cfg.get("temperature", 1.0)),
            pool_beta=float(cfg.get("pool_beta", 8.0)),
        )


def rss_safe_distance(ego_velocity: torch.Tensor, lead_velocity: torch.Tensor, cfg: RSSConfig) -> torch.Tensor:
    rho = cfg.response_time
    ego_after_response = ego_velocity + rho * cfg.ego_max_accel
    ego_distance = ego_velocity * rho + 0.5 * cfg.ego_max_accel * rho * rho
    ego_brake_distance = ego_after_response.square() / max(2.0 * cfg.ego_min_brake, 1e-6)
    lead_brake_distance = lead_velocity.square() / max(2.0 * cfg.lead_max_brake, 1e-6)
    return torch.clamp(ego_distance + ego_brake_distance - lead_brake_distance, min=0.0)


def rss_margin(kin: FollowingKinematics, cfg: RSSConfig) -> tuple[torch.Tensor, torch.Tensor]:
    safe = rss_safe_distance(kin.ego_velocity, torch.clamp(kin.velocity, min=0.0), cfg)
    return kin.gap - safe, safe


def softmax_pool(x: torch.Tensor, beta: float = 8.0, dim: int = 1) -> torch.Tensor:
    weights = torch.softmax(float(beta) * x, dim=dim)
    return torch.sum(weights * x, dim=dim)


def rss_criticality_objective(kin: FollowingKinematics, cfg: RSSConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    margin, safe = rss_margin(kin, cfg)
    violation = F.softplus((safe - kin.gap) / max(cfg.temperature, 1e-6))
    objective = softmax_pool(violation, beta=cfg.pool_beta, dim=1)
    return objective, {
        "rss_margin": margin,
        "rss_safe_distance": safe,
        "rss_violation_soft": violation,
        "min_rss_margin": torch.min(margin, dim=1).values,
        "rss_violation_rate": (margin < 0.0).to(kin.gap.dtype).mean(dim=1),
    }

