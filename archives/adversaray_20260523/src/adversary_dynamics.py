"""Differentiable adversary kinematics for following scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    adversary_x: torch.Tensor | None = None
    adversary_y: torch.Tensor | None = None
    adversary_yaw: torch.Tensor | None = None
    adversary_steering: torch.Tensor | None = None
    ego_x: torch.Tensor | None = None
    ego_y: torch.Tensor | None = None
    ego_yaw: torch.Tensor | None = None
    ego_action_accel: torch.Tensor | None = None
    ego_action_steering: torch.Tensor | None = None


def _dt(schema: dict, config: dict | None = None) -> float:
    config = config or {}
    return float(schema.get("dt", config.get("sampling", {}).get("dt", 0.04)))


def _representation(schema: dict, config: dict | None = None) -> str:
    config = config or {}
    action_cfg = config.get("action", {})
    rep = schema.get("action_representation", action_cfg.get("representation"))
    return str(rep or "jerk").lower()


def _dynamics_config(config: dict | None = None) -> dict[str, Any]:
    config = config or {}
    cfg = dict(config.get("dynamics", {}))
    cfg.setdefault("model", "longitudinal")
    cfg.setdefault("wheelbase", 5.0)
    cfg.setdefault("steering_abs_max", 0.5)
    cfg.setdefault("steering_rate_abs_max", 1.0)
    cfg.setdefault(
        "speed_min",
        config.get("physics", {}).get("speed_min", 0.0),
    )
    cfg.setdefault("speed_max", config.get("env", {}).get("speed_limit", 40.0))
    cfg.setdefault("accel_min", config.get("physics", {}).get("ax_min", -8.0))
    cfg.setdefault("accel_max", config.get("physics", {}).get("ax_max", 4.0))
    return cfg


def _validate_inputs(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
) -> None:
    if future_actions.ndim != 3 or future_actions.shape[-1] < 1:
        raise ValueError(
            "Expected future_actions shape [B,H,1+], "
            f"got {tuple(future_actions.shape)}"
        )
    valid_context = (
        context_states.ndim == 4
        and context_states.shape[2] >= 2
        and context_states.shape[-1] >= 5
    )
    if not valid_context:
        raise ValueError(
            "Expected context_states shape [B,T,2,state_dim>=5], "
            f"got {tuple(context_states.shape)}"
        )


def _lengths(
    future_actions: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ego_length is None:
        ego_length = torch.full(
            (future_actions.shape[0],),
            4.8,
            dtype=future_actions.dtype,
            device=future_actions.device,
        )
    if adv_length is None:
        adv_length = torch.full(
            (future_actions.shape[0],),
            4.8,
            dtype=future_actions.dtype,
            device=future_actions.device,
        )
    ego_length = ego_length.to(
        dtype=future_actions.dtype,
        device=future_actions.device,
    )
    adv_length = adv_length.to(
        dtype=future_actions.dtype,
        device=future_actions.device,
    )
    return ego_length, adv_length


def _heading_from_state(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] < 4:
        return torch.zeros_like(state[:, 0])
    speed = torch.hypot(state[:, 2], state[:, 3])
    yaw = torch.atan2(state[:, 3], state[:, 2])
    return torch.where(speed > 1e-4, yaw, torch.zeros_like(yaw))


def _state_speed(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] >= 4:
        return torch.hypot(state[:, 2], state[:, 3])
    return torch.clamp(state[:, 2], min=0.0)


def _longitudinal_acceleration(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] < 6:
        return state[:, 4]
    yaw = _heading_from_state(state)
    return state[:, 4] * torch.cos(yaw) + state[:, 5] * torch.sin(yaw)


def _apply_highway_idm_ego_response(
    kin: FollowingKinematics,
    context_states: torch.Tensor,
    ego_length: torch.Tensor,
    adv_length: torch.Tensor,
    schema: dict,
    config: dict | None,
) -> FollowingKinematics:
    cfg = dict((config or {}).get("ego_response", {}))
    model = str(cfg.get("model", "highway_idm")).lower()
    if model != "highway_idm":
        raise ValueError(f"Unknown ego_response.model: {model}")
    if kin.adversary_x is None or kin.adversary_y is None:
        raise ValueError("highway_idm ego response requires adversary poses")
    from .highway_idm_ego import rollout_highway_idm_ego_trace

    ego_trace = rollout_highway_idm_ego_trace(
        context_states=context_states,
        adversary_x=kin.adversary_x.detach(),
        adversary_y=kin.adversary_y.detach(),
        adversary_yaw=(
            torch.zeros_like(kin.adversary_x)
            if kin.adversary_yaw is None
            else kin.adversary_yaw.detach()
        ),
        adversary_speed=kin.velocity.detach(),
        adversary_accel=kin.acceleration.detach(),
        adversary_steering=(
            torch.zeros_like(kin.adversary_x)
            if kin.adversary_steering is None
            else kin.adversary_steering.detach()
        ),
        ego_length=ego_length.detach(),
        adv_length=adv_length.detach(),
        schema=schema,
        config=config or {},
    )
    ego_x = ego_trace["ego_x"].to(kin.velocity.device, kin.velocity.dtype)
    ego_y = ego_trace["ego_y"].to(kin.velocity.device, kin.velocity.dtype)
    ego_velocity = ego_trace["ego_speed"].to(
        kin.velocity.device,
        kin.velocity.dtype,
    )
    ego_acceleration = ego_trace["ego_accel"].to(
        kin.velocity.device,
        kin.velocity.dtype,
    )
    half_lengths = 0.5 * (ego_length + adv_length)
    gap = kin.adversary_x - ego_x - half_lengths[:, None]
    return FollowingKinematics(
        jerk=kin.jerk,
        acceleration=kin.acceleration,
        velocity=kin.velocity,
        displacement=kin.displacement,
        gap=gap,
        ego_velocity=ego_velocity.detach(),
        lead_initial_acceleration=kin.lead_initial_acceleration,
        ego_acceleration=ego_acceleration.detach(),
        ego_displacement=(
            ego_x - context_states[:, -1, 0, 0][:, None]
        ).detach(),
        adversary_x=kin.adversary_x,
        adversary_y=kin.adversary_y,
        adversary_yaw=kin.adversary_yaw,
        adversary_steering=kin.adversary_steering,
        ego_x=ego_x.detach(),
        ego_y=ego_y.detach(),
        ego_yaw=ego_trace["ego_yaw"].to(
            kin.velocity.device,
            kin.velocity.dtype,
        ).detach(),
        ego_action_accel=ego_trace["ego_action_accel"].to(
            kin.velocity.device,
            kin.velocity.dtype,
        ).detach(),
        ego_action_steering=ego_trace["ego_action_steering"].to(
            kin.velocity.device,
            kin.velocity.dtype,
        ).detach(),
    )


def integrate_longitudinal_actions_torch(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict,
    config: dict | None = None,
) -> FollowingKinematics:
    """Integrate longitudinal adversary actions before IDM ego replay."""
    _validate_inputs(future_actions, context_states)
    config = config or {}
    dt = _dt(schema, config)
    rep = _representation(schema, config)
    dyn_cfg = _dynamics_config(config)
    lead0 = context_states[:, -1, 1]
    ego0 = context_states[:, -1, 0]
    prev_ax = _longitudinal_acceleration(lead0)
    accel_min = float(dyn_cfg["accel_min"])
    accel_max = float(dyn_cfg["accel_max"])
    speed_min = float(dyn_cfg["speed_min"])
    speed_max = float(dyn_cfg["speed_max"])
    if rep == "jerk":
        jerk = future_actions[:, :, 0]
        acceleration = prev_ax[:, None] + torch.cumsum(jerk, dim=1) * dt
    elif rep == "acceleration":
        acceleration = torch.clamp(
            future_actions[:, :, 0],
            min=accel_min,
            max=accel_max,
        )
        prev = torch.cat([prev_ax[:, None], acceleration[:, :-1]], dim=1)
        jerk = (acceleration - prev) / max(dt, 1e-6)
    else:
        raise ValueError(f"Unsupported action representation: {rep}")
    acceleration = torch.clamp(
        acceleration,
        min=accel_min,
        max=accel_max,
    )

    v0 = torch.clamp(
        _state_speed(lead0),
        min=speed_min,
        max=speed_max,
    )
    velocity = torch.clamp(
        v0[:, None] + torch.cumsum(acceleration, dim=1) * dt,
        min=speed_min,
        max=speed_max,
    )
    v_before = torch.cat([v0[:, None], velocity[:, :-1]], dim=1)
    displacement = torch.cumsum(0.5 * (v_before + velocity) * dt, dim=1)

    ego_length, adv_length = _lengths(future_actions, ego_length, adv_length)
    half_lengths = 0.5 * (ego_length + adv_length)
    ego_v0 = torch.clamp(_state_speed(ego0), min=0.0)
    ego_accel0 = _longitudinal_acceleration(ego0)
    lead_yaw0 = _heading_from_state(lead0)
    lead_x = lead0[:, 0, None] + displacement
    lead_y = lead0[:, 1, None].expand_as(lead_x)
    lead_yaw = lead_yaw0[:, None].expand_as(lead_x)
    steering = torch.zeros_like(lead_x)
    ego_velocity = ego_v0[:, None].expand_as(velocity)
    ego_acceleration = ego_accel0[:, None].expand_as(velocity)
    ego_displacement = torch.zeros_like(velocity)
    ego_yaw = _heading_from_state(ego0)[:, None].expand_as(velocity)
    ego_x = ego0[:, 0, None].expand_as(velocity)
    ego_y = ego0[:, 1, None].expand_as(velocity)
    gap = lead_x - ego_x - half_lengths[:, None]
    kin = FollowingKinematics(
        jerk=jerk,
        acceleration=acceleration,
        velocity=velocity,
        displacement=displacement,
        gap=gap,
        ego_velocity=ego_velocity,
        lead_initial_acceleration=prev_ax,
        ego_acceleration=ego_acceleration,
        ego_displacement=ego_displacement,
        adversary_x=lead_x,
        adversary_y=lead_y,
        adversary_yaw=lead_yaw,
        adversary_steering=steering,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        ego_action_accel=ego_acceleration,
        ego_action_steering=torch.zeros_like(ego_acceleration),
    )
    return _apply_highway_idm_ego_response(
        kin,
        context_states,
        ego_length,
        adv_length,
        schema,
        config,
    )


def integrate_kinematic_bicycle_actions_torch(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict,
    config: dict | None = None,
) -> FollowingKinematics:
    """Integrate jerk and steering-rate actions with a bicycle model."""
    _validate_inputs(future_actions, context_states)
    config = config or {}
    dt = _dt(schema, config)
    rep = _representation(schema, config)
    dyn_cfg = _dynamics_config(config)
    lead0 = context_states[:, -1, 1]
    ego0 = context_states[:, -1, 0]
    prev_accel = _longitudinal_acceleration(lead0)
    accel_min = float(dyn_cfg["accel_min"])
    accel_max = float(dyn_cfg["accel_max"])
    if rep == "jerk":
        jerk = future_actions[:, :, 0]
    elif rep == "acceleration":
        acceleration_actions = torch.clamp(
            future_actions[:, :, 0],
            min=accel_min,
            max=accel_max,
        )
        prev = torch.cat(
            [prev_accel[:, None], acceleration_actions[:, :-1]],
            dim=1,
        )
        jerk = (acceleration_actions - prev) / max(dt, 1e-6)
    else:
        raise ValueError(f"Unsupported action representation: {rep}")

    if future_actions.shape[-1] > 1:
        steering_rate = future_actions[:, :, 1]
    else:
        steering_rate = torch.zeros_like(jerk)
    steering_rate = torch.clamp(
        steering_rate,
        min=-float(dyn_cfg["steering_rate_abs_max"]),
        max=float(dyn_cfg["steering_rate_abs_max"]),
    )

    x = lead0[:, 0]
    y = lead0[:, 1]
    yaw = _heading_from_state(lead0)
    velocity = torch.clamp(
        _state_speed(lead0),
        min=float(dyn_cfg["speed_min"]),
        max=float(dyn_cfg["speed_max"]),
    )
    acceleration = prev_accel
    steering = torch.zeros_like(velocity)
    x0 = x
    wheelbase = max(float(dyn_cfg["wheelbase"]), 1e-6)
    steering_abs_max = float(dyn_cfg["steering_abs_max"])
    speed_min = float(dyn_cfg["speed_min"])
    speed_max = float(dyn_cfg["speed_max"])

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    yaws: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    steerings: list[torch.Tensor] = []

    for t in range(future_actions.shape[1]):
        if rep == "jerk":
            acceleration = torch.clamp(
                acceleration + jerk[:, t] * dt,
                min=accel_min,
                max=accel_max,
            )
        else:
            acceleration = torch.clamp(
                future_actions[:, t, 0],
                min=accel_min,
                max=accel_max,
            )
        steering = torch.clamp(
            steering + steering_rate[:, t] * dt,
            min=-steering_abs_max,
            max=steering_abs_max,
        )
        x = x + velocity * torch.cos(yaw) * dt
        y = y + velocity * torch.sin(yaw) * dt
        yaw = yaw + velocity / wheelbase * torch.tan(steering) * dt
        velocity = torch.clamp(
            velocity + acceleration * dt,
            min=speed_min,
            max=speed_max,
        )
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
        velocities.append(velocity)
        accelerations.append(acceleration)
        steerings.append(steering)

    adversary_x = torch.stack(xs, dim=1)
    adversary_y = torch.stack(ys, dim=1)
    adversary_yaw = torch.stack(yaws, dim=1)
    velocity_out = torch.stack(velocities, dim=1)
    acceleration_out = torch.stack(accelerations, dim=1)
    steering_out = torch.stack(steerings, dim=1)
    displacement = adversary_x - x0[:, None]

    ego_length, adv_length = _lengths(future_actions, ego_length, adv_length)
    ego_speed = torch.clamp(_state_speed(ego0), min=0.0)
    ego_accel = _longitudinal_acceleration(ego0)
    ego_velocity = ego_speed[:, None].expand_as(velocity_out)
    ego_acceleration = ego_accel[:, None].expand_as(velocity_out)
    ego_x = ego0[:, 0, None].expand_as(velocity_out)
    ego_y = ego0[:, 1, None].expand_as(velocity_out)
    ego_yaw = _heading_from_state(ego0)[:, None].expand_as(velocity_out)
    half_lengths = 0.5 * (ego_length + adv_length)
    gap = adversary_x - ego_x - half_lengths[:, None]
    kin = FollowingKinematics(
        jerk=jerk,
        acceleration=acceleration_out,
        velocity=velocity_out,
        displacement=displacement,
        gap=gap,
        ego_velocity=ego_velocity,
        lead_initial_acceleration=prev_accel,
        ego_acceleration=ego_acceleration,
        ego_displacement=ego_x - ego0[:, 0, None],
        adversary_x=adversary_x,
        adversary_y=adversary_y,
        adversary_yaw=adversary_yaw,
        adversary_steering=steering_out,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        ego_action_accel=ego_acceleration,
        ego_action_steering=torch.zeros_like(ego_acceleration),
    )
    return _apply_highway_idm_ego_response(
        kin,
        context_states,
        ego_length,
        adv_length,
        schema,
        config,
    )


def integrate_adversary_actions_torch(
    future_actions: torch.Tensor,
    context_states: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    schema: dict,
    config: dict | None = None,
) -> FollowingKinematics:
    """Dispatch to the configured differentiable adversary dynamics model."""
    model = str(_dynamics_config(config).get("model", "longitudinal")).lower()
    if model == "longitudinal":
        return integrate_longitudinal_actions_torch(
            future_actions,
            context_states,
            ego_length,
            adv_length,
            schema,
            config,
        )
    if model == "kinematic_bicycle":
        return integrate_kinematic_bicycle_actions_torch(
            future_actions,
            context_states,
            ego_length,
            adv_length,
            schema,
            config,
        )
    raise ValueError(f"Unknown dynamics.model: {model}")
