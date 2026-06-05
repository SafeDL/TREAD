"""Action integration helpers for natural car-following rollouts."""
from __future__ import annotations

import numpy as np

from .types import VehicleState


def integrate_following_actions(
    initial: VehicleState,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate lead-car longitudinal acceleration.

    ``actions`` must be acceleration in m/s^2. If a model is trained with
    jerk actions, decode and integrate jerk to acceleration before calling
    this helper.

    Returns states with shape ``[H + 1, 6]`` in the same local frame as
    ``initial``.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 2:
        ax_seq = a[:, 0]
    else:
        ax_seq = a.reshape(-1)
    states = np.zeros((len(ax_seq) + 1, 6), dtype=np.float32)
    states[0] = initial.as_feature()
    x = float(initial.x)
    y = float(initial.y)
    vx = max(float(initial.vx), 0.0)
    vy = float(initial.vy)
    ay = float(initial.ay)
    for i, ax in enumerate(ax_seq):
        ax_f = float(ax)
        x = x + vx * dt + 0.5 * ax_f * dt * dt
        vx = max(vx + ax_f * dt, 0.0)
        states[i + 1] = np.asarray([x, y, vx, vy, ax_f, ay], dtype=np.float32)
    return states


def integrate_cutin_acceleration_actions(
    context_states: np.ndarray,
    actions: np.ndarray,
    dt: float,
    *,
    ax_min: float = -8.0,
    ax_max: float = 4.0,
    ay_abs_max: float = 4.0,
    speed_min: float = 0.0,
    speed_max: float = 50.0,
) -> np.ndarray:
    """Integrate a maneuver-level cut-in acceleration plan.

    ``actions`` is ``[B, H, 2]`` or ``[H, 2]`` with target-car ``[ax, ay]`` in
    the same local frame as ``context_states``. The returned states are
    ``[B, H, 6]`` or ``[H, 6]`` with ``[x, y, vx, vy, ax, ay]``.
    """
    ctx = np.asarray(context_states, dtype=np.float32)
    seq = np.asarray(actions, dtype=np.float32)
    squeeze = False
    if seq.ndim == 2:
        seq = seq[None, ...]
        ctx = ctx[None, ...]
        squeeze = True
    if seq.ndim != 3 or seq.shape[-1] < 2:
        raise ValueError(f"Expected action plan shape [B, H, >=2], got {seq.shape}")
    if ctx.ndim != 4 or ctx.shape[0] != seq.shape[0]:
        raise ValueError(
            "context_states must have shape [B, history, actors, features] "
            f"matching action batch, got {ctx.shape}"
        )
    initial = ctx[:, -1, 1].astype(np.float32)
    batch, horizon = int(seq.shape[0]), int(seq.shape[1])
    states = np.zeros((batch, horizon, 6), dtype=np.float32)
    x = initial[:, 0].copy()
    y = initial[:, 1].copy()
    vx = initial[:, 2].copy()
    vy = initial[:, 3].copy()
    dt_safe = max(float(dt), 1.0e-6)
    for step in range(horizon):
        ax = np.clip(seq[:, step, 0], float(ax_min), float(ax_max)).astype(np.float32)
        ay = np.clip(
            seq[:, step, 1],
            -float(ay_abs_max),
            float(ay_abs_max),
        ).astype(np.float32)
        x = x + vx * dt_safe + 0.5 * ax * dt_safe * dt_safe
        y = y + vy * dt_safe + 0.5 * ay * dt_safe * dt_safe
        vx = np.clip(vx + ax * dt_safe, float(speed_min), float(speed_max))
        vy = vy + ay * dt_safe
        states[:, step] = np.stack([x, y, vx, vy, ax, ay], axis=-1)
    return states[0] if squeeze else states


def project_cutin_maneuver_trajectory(
    context_states: np.ndarray,
    plan: np.ndarray,
    dt: float,
    *,
    ax_min: float = -8.0,
    ax_max: float = 4.0,
    jerk_abs_max: float = 12.0,
    ay_abs_max: float = 4.0,
    lateral_jerk_abs_max: float = 8.0,
    speed_min: float = 0.0,
    speed_max: float = 50.0,
    position_gain: float = 0.5,
    limit_margin: float = 0.98,
) -> np.ndarray:
    """Project a generated cut-in maneuver plan onto simple physical bounds.

    ``plan`` is ``[B, H, 4]`` or ``[H, 4]`` with ``[dx, dy, vx, vy]`` relative
    to the current target state in ``context_states``.
    """
    ctx = np.asarray(context_states, dtype=np.float32)
    seq = np.asarray(plan, dtype=np.float32)
    squeeze = False
    if seq.ndim == 2:
        seq = seq[None, ...]
        ctx = ctx[None, ...]
        squeeze = True
    if seq.ndim != 3 or seq.shape[-1] < 4:
        raise ValueError(f"Expected maneuver plan shape [B, H, >=4], got {seq.shape}")
    if ctx.ndim != 4 or ctx.shape[0] != seq.shape[0]:
        raise ValueError(
            "context_states must have shape [B, history, actors, features] "
            f"matching plan batch, got {ctx.shape}"
        )
    initial = ctx[:, -1, 1].astype(np.float32)
    batch, horizon = int(seq.shape[0]), int(seq.shape[1])
    states = np.zeros((batch, horizon, 6), dtype=np.float32)
    x = initial[:, 0].copy()
    y = initial[:, 1].copy()
    vx = initial[:, 2].copy()
    vy = initial[:, 3].copy()
    ax = initial[:, 4].copy()
    ay = initial[:, 5].copy()
    dt_safe = max(float(dt), 1.0e-6)
    j_step = max(float(jerk_abs_max), 0.0) * max(float(limit_margin), 0.0) * dt_safe
    lj_step = (
        max(float(lateral_jerk_abs_max), 0.0)
        * max(float(limit_margin), 0.0)
        * dt_safe
    )
    for step in range(horizon):
        target_x = initial[:, 0] + seq[:, step, 0]
        target_y = initial[:, 1] + seq[:, step, 1]
        desired_vx = seq[:, step, 2] + float(position_gain) * (target_x - x)
        desired_vy = seq[:, step, 3] + float(position_gain) * (target_y - y)
        desired_ax = (desired_vx - vx) / dt_safe
        desired_ay = (desired_vy - vy) / dt_safe
        ax_lower = np.maximum(float(ax_min), ax - j_step)
        ax_upper = np.minimum(float(ax_max), ax + j_step)
        ay_lower = np.maximum(-float(ay_abs_max), ay - lj_step)
        ay_upper = np.minimum(float(ay_abs_max), ay + lj_step)
        ax = np.clip(desired_ax, ax_lower, ax_upper).astype(np.float32)
        ay = np.clip(desired_ay, ay_lower, ay_upper).astype(np.float32)
        vx = np.clip(vx + ax * dt_safe, float(speed_min), float(speed_max))
        vy = vy + ay * dt_safe
        x = x + vx * dt_safe
        y = y + vy * dt_safe
        states[:, step] = np.stack([x, y, vx, vy, ax, ay], axis=-1)
    return states[0] if squeeze else states
