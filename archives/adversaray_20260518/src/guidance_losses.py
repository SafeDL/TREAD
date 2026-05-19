"""Differentiable physics penalties for guided denoising."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .torch_kinematics import FollowingKinematics


def physical_violation_penalty(kin: FollowingKinematics, config: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config.get("physics", config.get("action", {}))
    speed_min = float(cfg.get("speed_min", 0.0))
    ax_min = float(cfg.get("ax_min", -8.0))
    ax_max = float(cfg.get("ax_max", 4.0))
    jerk_abs_max = float(cfg.get("jerk_abs_max", 12.0))
    acceleration_low = F.softplus(ax_min - kin.acceleration).square()
    acceleration_high = F.softplus(kin.acceleration - ax_max).square()
    if kin.acceleration.shape[1] <= 1:
        continuity = torch.zeros((kin.acceleration.shape[0],), dtype=kin.acceleration.dtype, device=kin.acceleration.device)
    else:
        first_jump = (kin.acceleration[:, 0] - kin.lead_initial_acceleration).square()
        smooth = (kin.acceleration[:, 1:] - kin.acceleration[:, :-1]).square().mean(dim=1)
        continuity = first_jump + smooth
    pieces = {
        "negative_speed_penalty": F.softplus(speed_min - kin.velocity).square().mean(dim=1),
        "acceleration_bound_penalty": (acceleration_low + acceleration_high).mean(dim=1),
        "jerk_bound_penalty": F.softplus(torch.abs(kin.jerk) - jerk_abs_max).square().mean(dim=1),
        "trajectory_discontinuity_penalty": continuity,
    }
    total = sum(pieces.values())
    diagnostics = {
        **pieces,
        "negative_speed_rate": (kin.velocity < speed_min).to(kin.velocity.dtype).mean(dim=1),
        "ax_violation_rate": ((kin.acceleration < ax_min) | (kin.acceleration > ax_max)).to(kin.acceleration.dtype).mean(dim=1),
        "jerk_violation_rate": (torch.abs(kin.jerk) > jerk_abs_max).to(kin.jerk.dtype).mean(dim=1),
        "trajectory_continuity": continuity,
    }
    return total, diagnostics
