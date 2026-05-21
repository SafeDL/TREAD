"""Differentiable longitudinal kinematics for following scenarios."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FollowingKinematics:
    jerk: torch.Tensor
    acceleration: torch.Tensor
    velocity: torch.Tensor
    displacement: torch.Tensor
    gap: torch.Tensor
    ego_velocity: torch.Tensor
    lead_initial_acceleration: torch.Tensor


def _dt(schema: dict, config: dict | None = None) -> float:
    config = config or {}
    return float(schema.get("dt", config.get("sampling", {}).get("dt", 0.04)))


def _representation(schema: dict, config: dict | None = None) -> str:
    config = config or {}
    return str(schema.get("action_representation", config.get("action", {}).get("representation", "jerk"))).lower()


def integrate_following_actions_torch(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict,
    config: dict | None = None,
) -> FollowingKinematics:
    """Integrate lead-car actions against a nominal constant-velocity ego.

    ``future_actions`` are expected in physical action units, not normalized.
    The function intentionally avoids NumPy so guidance can backpropagate into
    the action sequence or into the diffusion state that produced it.
    """
    if future_actions.ndim != 3 or future_actions.shape[-1] < 1:
        raise ValueError(f"Expected future_actions shape [B,H,1+], got {tuple(future_actions.shape)}")
    if context_states.ndim != 4 or context_states.shape[2] < 2 or context_states.shape[-1] < 5:
        raise ValueError(f"Expected context_states shape [B,T,2,state_dim>=5], got {tuple(context_states.shape)}")
    config = config or {}
    dt = _dt(schema, config)
    rep = _representation(schema, config)
    lead0 = context_states[:, -1, 1]
    ego0 = context_states[:, -1, 0]
    prev_ax = lead0[:, 4]
    if rep == "jerk":
        jerk = future_actions[:, :, 0]
        acceleration = prev_ax[:, None] + torch.cumsum(jerk, dim=1) * dt
    elif rep == "acceleration":
        acceleration = future_actions[:, :, 0]
        prev = torch.cat([prev_ax[:, None], acceleration[:, :-1]], dim=1)
        jerk = (acceleration - prev) / max(dt, 1e-6)
    else:
        raise ValueError(f"Unsupported action representation: {rep}")

    v0 = torch.clamp(lead0[:, 2], min=0.0)
    velocity = v0[:, None] + torch.cumsum(acceleration, dim=1) * dt
    v_before = torch.cat([v0[:, None], velocity[:, :-1]], dim=1)
    displacement = torch.cumsum(v_before * dt + 0.5 * acceleration * dt * dt, dim=1)

    if ego_length is None:
        ego_length = torch.full((future_actions.shape[0],), 4.8, dtype=future_actions.dtype, device=future_actions.device)
    if adv_length is None:
        adv_length = torch.full((future_actions.shape[0],), 4.8, dtype=future_actions.dtype, device=future_actions.device)
    half_lengths = 0.5 * (ego_length.to(dtype=future_actions.dtype, device=future_actions.device) + adv_length.to(dtype=future_actions.dtype, device=future_actions.device))
    gap0 = lead0[:, 0] - ego0[:, 0] - half_lengths
    steps = torch.arange(1, future_actions.shape[1] + 1, dtype=future_actions.dtype, device=future_actions.device)
    ego_velocity = torch.clamp(ego0[:, 2], min=0.0)
    ego_dx = ego_velocity[:, None] * (steps[None, :] * dt)
    gap = gap0[:, None] + displacement - ego_dx
    return FollowingKinematics(
        jerk=jerk,
        acceleration=acceleration,
        velocity=velocity,
        displacement=displacement,
        gap=gap,
        ego_velocity=ego_velocity[:, None].expand_as(velocity),
        lead_initial_acceleration=prev_ax,
    )

