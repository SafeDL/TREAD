"""Physical feasibility diagnostics for adversary trajectories."""
from __future__ import annotations

import torch

from .adversary_dynamics import FollowingKinematics


def physical_violation_penalty(
    kin: FollowingKinematics,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config.get("physics", config.get("action", {}))
    speed_min = float(cfg.get("speed_min", 0.0))
    speed_max = float(cfg.get("speed_max", float("inf")))
    ax_min = float(cfg.get("ax_min", -8.0))
    ax_max = float(cfg.get("ax_max", 4.0))
    jerk_abs_max = float(cfg.get("jerk_abs_max", 12.0))
    steering_abs_max = float(cfg.get("steering_abs_max", float("inf")))
    acceleration_low = torch.relu(ax_min - kin.acceleration).square()
    acceleration_high = torch.relu(kin.acceleration - ax_max).square()
    speed_low = torch.relu(speed_min - kin.velocity).square()
    speed_high = torch.relu(kin.velocity - speed_max).square()
    jerk_bound = torch.relu(torch.abs(kin.jerk) - jerk_abs_max).square()
    if kin.adversary_steering is None:
        steering_bound = torch.zeros_like(kin.velocity)
        steering_violation = torch.zeros_like(kin.velocity)
    else:
        steering_bound = torch.relu(
            torch.abs(kin.adversary_steering) - steering_abs_max
        ).square()
        steering_violation = (
            torch.abs(kin.adversary_steering) > steering_abs_max
        ).to(kin.velocity.dtype)

    if kin.acceleration.shape[1] <= 1:
        continuity = torch.zeros(
            (kin.acceleration.shape[0],),
            dtype=kin.acceleration.dtype,
            device=kin.acceleration.device,
        )
    else:
        first_jump = (
            kin.acceleration[:, 0] - kin.lead_initial_acceleration
        ).square()
        smooth = (
            kin.acceleration[:, 1:] - kin.acceleration[:, :-1]
        ).square().mean(dim=1)
        continuity = first_jump + smooth

    pieces = {
        "speed_bound_penalty": (speed_low + speed_high).mean(dim=1),
        "acceleration_bound_penalty": (
            acceleration_low + acceleration_high
        ).mean(dim=1),
        "jerk_bound_penalty": jerk_bound.mean(dim=1),
        "steering_bound_penalty": steering_bound.mean(dim=1),
    }
    total = sum(pieces.values())
    diagnostics = {
        **pieces,
        "negative_speed_penalty": speed_low.mean(dim=1),
        "negative_speed_rate": (kin.velocity < speed_min).to(
            kin.velocity.dtype
        ).mean(dim=1),
        "speed_high_violation_rate": (kin.velocity > speed_max).to(
            kin.velocity.dtype
        ).mean(dim=1),
        "ax_violation_rate": (
            (kin.acceleration < ax_min) | (kin.acceleration > ax_max)
        ).to(kin.acceleration.dtype).mean(dim=1),
        "jerk_violation_rate": (torch.abs(kin.jerk) > jerk_abs_max).to(
            kin.jerk.dtype
        ).mean(dim=1),
        "steering_violation_rate": steering_violation.mean(dim=1),
        "trajectory_discontinuity_penalty": continuity,
        "trajectory_continuity": continuity,
    }
    return total, diagnostics
