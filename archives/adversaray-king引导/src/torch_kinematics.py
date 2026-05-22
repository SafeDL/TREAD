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
    ego_acceleration: torch.Tensor
    ego_displacement: torch.Tensor


def _dt(schema: dict, config: dict | None = None) -> float:
    config = config or {}
    return float(schema.get("dt", config.get("sampling", {}).get("dt", 0.04)))


def _representation(schema: dict, config: dict | None = None) -> str:
    config = config or {}
    return str(schema.get("action_representation", config.get("action", {}).get("representation", "jerk"))).lower()


def _idm_config(config: dict | None = None) -> dict[str, float]:
    config = config or {}
    env_cfg = config.get("env", {})
    idm_cfg = config.get("idm", {})
    return {
        "desired_speed": float(idm_cfg.get("desired_speed", env_cfg.get("ego_target_speed", 30.0))),
        "max_accel": float(idm_cfg.get("max_accel", 2.0)),
        "comfortable_brake": float(idm_cfg.get("comfortable_brake", 2.0)),
        "min_gap": float(idm_cfg.get("min_gap", 2.0)),
        "desired_headway": float(idm_cfg.get("desired_headway", 1.0)),
        "delta": float(idm_cfg.get("delta", 4.0)),
        "accel_min": float(idm_cfg.get("accel_min", -6.0)),
        "accel_max": float(idm_cfg.get("accel_max", 3.0)),
    }


def integrate_following_actions_torch(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict,
    config: dict | None = None,
) -> FollowingKinematics:
    """Integrate lead-car actions against a differentiable IDM ego proxy.

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
    ego_length = ego_length.to(dtype=future_actions.dtype, device=future_actions.device)
    adv_length = adv_length.to(dtype=future_actions.dtype, device=future_actions.device)
    half_lengths = 0.5 * (ego_length + adv_length)
    gap0 = lead0[:, 0] - ego0[:, 0] - half_lengths
    ego_v0 = torch.clamp(ego0[:, 2], min=0.0)
    idm = _idm_config(config)
    desired_speed = max(float(idm["desired_speed"]), 1e-3)
    max_accel = max(float(idm["max_accel"]), 1e-3)
    comfortable_brake = max(float(idm["comfortable_brake"]), 1e-3)
    min_gap = max(float(idm["min_gap"]), 1e-3)
    desired_headway = max(float(idm["desired_headway"]), 0.0)
    delta = max(float(idm["delta"]), 1e-3)
    accel_min = float(idm["accel_min"])
    accel_max = float(idm["accel_max"])
    interaction_scale = max(2.0 * (max_accel * comfortable_brake) ** 0.5, 1e-6)

    ego_v = ego_v0
    ego_x = torch.zeros_like(ego_v0)
    ego_velocity_parts: list[torch.Tensor] = []
    ego_acceleration_parts: list[torch.Tensor] = []
    ego_displacement_parts: list[torch.Tensor] = []
    gap_parts: list[torch.Tensor] = []

    for t in range(future_actions.shape[1]):
        lead_v_t = torch.clamp(v_before[:, t], min=0.0)
        gap_t = gap0 + (displacement[:, t - 1] if t > 0 else torch.zeros_like(gap0)) - ego_x
        delta_v = ego_v - lead_v_t
        dynamic_gap = torch.clamp(ego_v * desired_headway + ego_v * delta_v / interaction_scale, min=0.0)
        desired_gap = min_gap + dynamic_gap
        free_road = torch.pow(torch.clamp(ego_v / desired_speed, min=0.0), delta)
        interaction = torch.square(desired_gap / torch.clamp(gap_t, min=1e-3))
        ego_accel = max_accel * (1.0 - free_road - interaction)
        ego_accel = torch.clamp(ego_accel, min=accel_min, max=accel_max)
        ego_dx = torch.clamp(ego_v * dt + 0.5 * ego_accel * dt * dt, min=0.0)
        ego_x = ego_x + ego_dx
        ego_v = torch.clamp(ego_v + ego_accel * dt, min=0.0)

        ego_acceleration_parts.append(ego_accel)
        ego_velocity_parts.append(ego_v)
        ego_displacement_parts.append(ego_x)
        gap_parts.append(gap0 + displacement[:, t] - ego_x)

    ego_acceleration = torch.stack(ego_acceleration_parts, dim=1)
    ego_velocity = torch.stack(ego_velocity_parts, dim=1)
    ego_displacement = torch.stack(ego_displacement_parts, dim=1)
    gap = torch.stack(gap_parts, dim=1)
    return FollowingKinematics(
        jerk=jerk,
        acceleration=acceleration,
        velocity=velocity,
        displacement=displacement,
        gap=gap,
        ego_velocity=ego_velocity,
        lead_initial_acceleration=prev_ax,
        ego_acceleration=ego_acceleration,
        ego_displacement=ego_displacement,
    )
