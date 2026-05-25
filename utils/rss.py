"""Shared RSS helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RSSConfig:
    response_time: float = 0.458
    ego_max_accel: float = 2.389
    ego_min_brake: float = 2.136
    lead_max_brake: float = 7.625
    temperature: float = 1.0
    pool_beta: float = 8.0
    mode: str = "delta"
    delta_temperature: float = 1.0
    relative_safe_min: float = 1.0
    improper_temperature: float = 1.0

    @classmethod
    def from_config(cls, config: dict) -> "RSSConfig":
        cfg = config.get("rss", config)
        return cls(
            response_time=float(cfg.get("response_time", cls.response_time)),
            ego_max_accel=float(cfg.get("ego_max_accel", cls.ego_max_accel)),
            ego_min_brake=float(cfg.get("ego_min_brake", cls.ego_min_brake)),
            lead_max_brake=float(
                cfg.get("lead_max_brake", cls.lead_max_brake)
            ),
            temperature=float(cfg.get("temperature", cls.temperature)),
            pool_beta=float(cfg.get("pool_beta", cls.pool_beta)),
            mode=str(cfg.get("mode", cls.mode)).lower(),
            delta_temperature=float(
                cfg.get("delta_temperature", cls.delta_temperature)
            ),
            relative_safe_min=float(
                cfg.get("relative_safe_min", cls.relative_safe_min)
            ),
            improper_temperature=float(
                cfg.get("improper_temperature", cls.improper_temperature)
            ),
        )


def rss_safe_distance(
    ego_velocity: torch.Tensor,
    lead_velocity: torch.Tensor,
    cfg: RSSConfig,
) -> torch.Tensor:
    rho = cfg.response_time
    ego_after_response = ego_velocity + rho * cfg.ego_max_accel
    ego_distance = ego_velocity * rho + 0.5 * cfg.ego_max_accel * rho * rho
    ego_brake_distance = ego_after_response.square() / max(
        2.0 * cfg.ego_min_brake,
        1e-6,
    )
    lead_brake_distance = lead_velocity.square() / max(
        2.0 * cfg.lead_max_brake,
        1e-6,
    )
    return torch.clamp(
        ego_distance + ego_brake_distance - lead_brake_distance,
        min=0.0,
    )


def rss_safe_distance_np(
    ego_velocity: np.ndarray,
    lead_velocity: np.ndarray,
    cfg: RSSConfig,
) -> np.ndarray:
    ego_velocity = np.asarray(ego_velocity, dtype=np.float64)
    lead_velocity = np.asarray(lead_velocity, dtype=np.float64)
    rho = float(cfg.response_time)
    ego_after_response = ego_velocity + rho * float(cfg.ego_max_accel)
    ego_distance = (
        ego_velocity * rho
        + 0.5 * float(cfg.ego_max_accel) * rho * rho
    )
    ego_brake_distance = np.square(ego_after_response) / max(
        2.0 * float(cfg.ego_min_brake),
        1e-6,
    )
    lead_brake_distance = np.square(lead_velocity) / max(
        2.0 * float(cfg.lead_max_brake),
        1e-6,
    )
    return np.maximum(
        ego_distance + ego_brake_distance - lead_brake_distance,
        0.0,
    )


def rss_margin(kin: Any, cfg: RSSConfig) -> tuple[torch.Tensor, torch.Tensor]:
    safe = rss_safe_distance(
        kin.ego_velocity,
        torch.clamp(kin.velocity, min=0.0),
        cfg,
    )
    return kin.gap - safe, safe


def softmax_pool(
    x: torch.Tensor,
    beta: float = 8.0,
    dim: int = 1,
) -> torch.Tensor:
    weights = torch.softmax(float(beta) * x, dim=dim)
    return torch.sum(weights * x, dim=dim)

