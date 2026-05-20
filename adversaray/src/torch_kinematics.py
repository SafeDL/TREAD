"""Differentiable longitudinal kinematics for following scenarios."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ego_surrogate import IDMSurrogateParams, ego_surrogate_config


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


def _surrogate_type(config: dict | None = None) -> str:
    cfg = ego_surrogate_config(config)
    return str(cfg.get("type", "constant_velocity")).lower()


def _idm_acceleration(
    gap: torch.Tensor,
    ego_velocity: torch.Tensor,
    lead_velocity: torch.Tensor,
    params: IDMSurrogateParams,
) -> torch.Tensor:
    gap = torch.clamp(gap, min=0.1)
    v_ego = torch.clamp(ego_velocity, min=0.0)
    v_lead = torch.clamp(lead_velocity, min=0.0)
    desired_speed = torch.clamp(params.desired_speed, min=1e-3)
    max_accel = torch.clamp(params.max_accel, min=1e-3)
    comfortable_brake = torch.clamp(params.comfortable_brake, min=1e-3)
    delta_v = v_ego - v_lead
    braking_term = v_ego * delta_v / torch.clamp(2.0 * torch.sqrt(max_accel * comfortable_brake), min=1e-3)
    desired_gap = params.min_gap + v_ego * params.desired_headway + braking_term
    desired_gap = torch.clamp(desired_gap, min=params.min_gap)
    free_road = torch.pow(v_ego / desired_speed, torch.clamp(params.delta, min=1.0))
    interaction = torch.square(desired_gap / gap)
    return max_accel * (1.0 - free_road - interaction)


def integrate_following_actions_torch(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict,
    config: dict | None = None,
    ego_surrogate_params: IDMSurrogateParams | None = None,
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
    ego_velocity0 = torch.clamp(ego0[:, 2], min=0.0)
    surrogate_type = "idm" if ego_surrogate_params is not None else _surrogate_type(config)
    if surrogate_type in {"constant_velocity", "constant"}:
        ego_dx = ego_velocity0[:, None] * (steps[None, :] * dt)
        gap = gap0[:, None] + displacement - ego_dx
        ego_velocity = ego_velocity0[:, None].expand_as(velocity)
        ego_acceleration = torch.zeros_like(velocity)
        ego_displacement = ego_dx
    elif surrogate_type == "idm":
        if ego_surrogate_params is None:
            raise ValueError("ego_surrogate_params is required when ego_surrogate.type='idm'")
        params = ego_surrogate_params.to(device=future_actions.device, dtype=future_actions.dtype)
        x_ego = torch.zeros_like(ego_velocity0)
        v_ego = ego_velocity0
        prev_ego_accel = torch.clamp(ego0[:, 4], min=-8.0, max=4.0)
        gap_steps: list[torch.Tensor] = []
        ego_velocity_steps: list[torch.Tensor] = []
        ego_acceleration_steps: list[torch.Tensor] = []
        ego_displacement_steps: list[torch.Tensor] = []
        for t in range(future_actions.shape[1]):
            raw_accel = _idm_acceleration(gap0 + displacement[:, t] - x_ego, v_ego, velocity[:, t], params)
            response_time = torch.clamp(params.response_time, min=0.0)
            response_alpha = torch.where(
                response_time > 1e-6,
                torch.clamp(torch.as_tensor(dt, dtype=future_actions.dtype, device=future_actions.device) / response_time, max=1.0),
                torch.ones_like(response_time),
            )
            ego_accel = prev_ego_accel + response_alpha * (raw_accel - prev_ego_accel)
            v_ego = torch.clamp(v_ego + ego_accel * dt, min=0.0)
            x_ego = x_ego + v_ego * dt
            gap_t = gap0 + displacement[:, t] - x_ego
            gap_steps.append(gap_t)
            ego_velocity_steps.append(v_ego)
            ego_acceleration_steps.append(ego_accel)
            ego_displacement_steps.append(x_ego)
            prev_ego_accel = ego_accel
        gap = torch.stack(gap_steps, dim=1)
        ego_velocity = torch.stack(ego_velocity_steps, dim=1)
        ego_acceleration = torch.stack(ego_acceleration_steps, dim=1)
        ego_displacement = torch.stack(ego_displacement_steps, dim=1)
    else:
        raise ValueError(f"Unsupported ego_surrogate.type: {surrogate_type}")
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
