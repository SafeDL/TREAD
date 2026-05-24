"""Enhanced RSS objectives for relative and responsibility-aware risk."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .adversary_dynamics import FollowingKinematics
from .rss import RSSConfig, rss_margin, softmax_pool


def _temperature(value: float) -> float:
    return max(float(value), 1.0e-6)


def relative_rss_margin(
    kin: FollowingKinematics,
    cfg: RSSConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    margin, safe = rss_margin(kin, cfg)
    denom = torch.clamp(safe, min=float(cfg.relative_safe_min))
    return margin / denom, safe


def relative_rss_objective(
    kin: FollowingKinematics,
    cfg: RSSConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rel_margin, safe = relative_rss_margin(kin, cfg)
    violation = F.softplus(-rel_margin / _temperature(cfg.temperature))
    objective = softmax_pool(violation, beta=cfg.pool_beta, dim=1)
    return objective, {
        "relative_rss_margin": rel_margin,
        "relative_rss_violation_soft": violation,
        "relative_rss_safe_distance": safe,
        "min_relative_rss_margin": torch.min(rel_margin, dim=1).values,
    }


def delta_rss_objective(
    kin_adv: FollowingKinematics,
    kin_ref: FollowingKinematics | None,
    cfg: RSSConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if kin_ref is None:
        raise ValueError("delta RSS mode requires reference_kin")
    margin_adv, _safe_adv = rss_margin(kin_adv, cfg)
    margin_ref, _safe_ref = rss_margin(kin_ref, cfg)
    if margin_adv.shape != margin_ref.shape:
        raise ValueError(
            "delta RSS requires matching adversary/reference horizons, "
            f"got {tuple(margin_adv.shape)} and {tuple(margin_ref.shape)}"
        )
    delta_margin = margin_ref.detach() - margin_adv
    temp = _temperature(cfg.delta_temperature)
    centered = F.softplus(delta_margin / temp) - F.softplus(
        torch.zeros((), dtype=delta_margin.dtype, device=delta_margin.device)
    )
    violation = torch.clamp(centered, min=0.0)
    objective = softmax_pool(violation, beta=cfg.pool_beta, dim=1)
    return objective, {
        "delta_rss_margin": delta_margin,
        "delta_rss_violation_soft": violation,
        "max_delta_rss_margin": torch.max(delta_margin, dim=1).values,
    }


def rss_improper_response_score(
    kin: FollowingKinematics,
    cfg: RSSConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rel_margin, _safe = relative_rss_margin(kin, cfg)
    danger = F.softplus(-rel_margin / _temperature(cfg.temperature))
    brake_threshold = -float(cfg.ego_min_brake)
    if kin.ego_action_accel is not None:
        ego_accel = kin.ego_action_accel
    else:
        ego_accel = kin.ego_acceleration
    insufficient_brake = torch.sigmoid(
        (ego_accel - brake_threshold) / _temperature(cfg.improper_temperature)
    )
    improper = danger * insufficient_brake
    objective = softmax_pool(improper, beta=cfg.pool_beta, dim=1)
    return objective, {
        "rss_improper_soft": improper,
        "rss_improper_fraction": (
            (rel_margin < 0.0) & (ego_accel > brake_threshold)
        ).to(kin.gap.dtype).mean(dim=1),
    }
