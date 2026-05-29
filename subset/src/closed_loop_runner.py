"""Closed-loop highway-env rollouts for subset rolling plans."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HIGHWAY_ROOT = ROOT / "HighwayEnv"
HIGHWAY_PACKAGE = HIGHWAY_ROOT / "highway_env"
if not HIGHWAY_PACKAGE.is_dir():
    raise FileNotFoundError(
        f"Required local highway-env package not found: {HIGHWAY_PACKAGE}"
    )
if str(HIGHWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGHWAY_ROOT))

from diffusion.src.features import extract_context
from diffusion.src.scenario_frame import compute_ego_frame, world_to_ego_states
from utils.normalization import normalize_numpy
from utils.risk import apply_closed_loop_risk

from .frozen_diffusion_sampler import FrozenDiffusionSampler

try:
    from highway_env.road.road import Road, RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.kinematics import Vehicle
except ImportError as exc:
    py_version = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    raise RuntimeError(
        "Failed to import the required local highway-env package. "
        f"Package path: {HIGHWAY_PACKAGE}. "
        "Install dependencies from HighwayEnv/pyproject.toml, including "
        f"gymnasium. Current Python: {py_version}."
    ) from exc


@dataclass
class RolloutResult:
    risk_score: float
    metrics: dict[str, float]
    num_generated_plans: int
    trace: list[dict[str, float]] = field(default_factory=list)
    actions: np.ndarray | None = None
    prior_actions: np.ndarray | None = None
    plan_summaries: list[dict[str, float]] = field(default_factory=list)

    @property
    def closed_loop_risk(self) -> float:
        return self.risk_score


class ScriptedLeadVehicle(Vehicle):
    """A highway-env vehicle whose acceleration and steering are scripted."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commanded_acceleration = 0.0
        self.commanded_steering = 0.0
        self.forced_position: np.ndarray | None = None
        self.forced_heading: float | None = None
        self.forced_speed: float | None = None

    def set_control(self, acceleration: float, steering: float = 0.0) -> None:
        self.commanded_acceleration = float(acceleration)
        self.commanded_steering = float(steering)

    def set_forced_state(
        self,
        position: np.ndarray,
        heading: float,
        speed: float,
    ) -> None:
        self.forced_position = np.asarray(position, dtype=np.float64)
        self.forced_heading = float(heading)
        self.forced_speed = float(speed)

    def act(self, action: dict | str = None) -> None:
        Vehicle.act(
            self,
            {
                "steering": self.commanded_steering,
                "acceleration": self.commanded_acceleration,
            },
        )

    def step(self, dt: float) -> None:
        if self.forced_position is None:
            Vehicle.step(self, dt)
            return
        self.position = self.forced_position
        self.heading = float(self.forced_heading)
        self.speed = float(self.forced_speed)
        self.forced_position = None
        self.forced_heading = None
        self.forced_speed = None
        self.on_state_update()


def _relative_history(
    history_local: np.ndarray,
    ego_length: float,
    lead_length: float,
) -> np.ndarray:
    ego = np.asarray(history_local[:, 0], dtype=np.float32)
    lead = np.asarray(history_local[:, 1], dtype=np.float32)
    gap = lead[:, 0] - ego[:, 0] - 0.5 * (ego_length + lead_length)
    lateral = lead[:, 1] - ego[:, 1]
    delta_v = ego[:, 2] - lead[:, 2]
    delta_a = ego[:, 4] - lead[:, 4]
    eps = 1e-6
    ttc_cap = 1000.0
    thw_cap = 200.0
    ttc = np.where(delta_v > eps, gap / np.maximum(delta_v, eps), ttc_cap)
    thw = gap / np.maximum(ego[:, 2], eps)
    return np.stack(
        [
            gap,
            lateral,
            delta_v,
            delta_a,
            np.clip(ttc, 0.0, ttc_cap),
            np.clip(thw, 0.0, thw_cap),
        ],
        axis=-1,
    ).astype(np.float32)


def _localize_history(history_world: np.ndarray) -> np.ndarray:
    states = np.asarray(history_world, dtype=np.float32)
    ego_frame = compute_ego_frame(states[-1, 0])
    return world_to_ego_states(states, ego_frame).astype(np.float32)


def _speed_and_yaw(state: np.ndarray) -> tuple[float, float]:
    speed = float(np.hypot(float(state[2]), float(state[3])))
    if speed > 1e-4:
        return speed, float(np.arctan2(float(state[3]), float(state[2])))
    return max(float(state[2]), 0.0), 0.0


def _accel_bounds_for_speed(
    speed: float,
    dt: float,
    ax_min: float,
    ax_max: float,
    speed_min: float,
    speed_max: float,
) -> tuple[float, float]:
    lower = float(ax_min)
    upper = float(ax_max)
    if dt > 0.0:
        lower = max(lower, (speed_min - speed) / dt)
        upper = min(upper, (speed_max - speed) / dt)
    if lower > upper:
        return float(ax_min), float(ax_max)
    return lower, upper


def _bound_residual(value: float, lower: float, upper: float) -> float:
    return max(0.0, lower - value) ** 2 + max(0.0, value - upper) ** 2


class ClosedLoopFollowingRunner:
    """Roll a generated lead plan on a highway-env car-following road."""

    def __init__(
        self,
        sampler: FrozenDiffusionSampler,
        config: dict[str, Any],
    ) -> None:
        self.sampler = sampler
        self.config = config
        env_cfg = config.get("env", {})
        prior_cfg = sampler.prior.model.denoiser.cfg
        target_fps = float(
            sampler.prior.config.get("sampling", {}).get(
                "target_fps",
                25.0,
            )
        )
        self.dt = float(env_cfg.get("dt", 1.0 / max(target_fps, 1.0)))
        self.history_steps = int(prior_cfg.history_steps)
        self.episode_steps = int(
            env_cfg.get("episode_steps", min(25, prior_cfg.horizon_steps))
        )
        self.commit_steps_max = int(env_cfg.get("commit_steps_max", 1))
        self.lanes_count = int(env_cfg.get("lanes_count", 1))
        self.speed_limit = float(env_cfg.get("speed_limit", 40.0))
        self.ego_target_speed = float(env_cfg.get("ego_target_speed", 30.0))
        self.initial_gap_min = float(env_cfg.get("initial_gap_min", 0.1))
        self.dynamics_model = str(
            config.get("dynamics", {}).get("model", "longitudinal")
        ).lower()
        if self.dynamics_model not in {"longitudinal", "kinematic_bicycle"}:
            raise ValueError(f"Unknown dynamics.model: {self.dynamics_model}")

    def _make_road(self) -> Any:
        return Road(
            network=RoadNetwork.straight_road_network(
                self.lanes_count,
                speed_limit=self.speed_limit,
            ),
            np_random=np.random.RandomState(
                int(self.config.get("training", {}).get("seed", 42))
            ),
            record_history=False,
        )

    def _build_observation(
        self,
        history_world: deque[np.ndarray],
        ego_length: float,
        lead_length: float,
    ) -> dict[str, np.ndarray]:
        hist = np.asarray(list(history_world), dtype=np.float32)
        history_local = _localize_history(hist)
        context_features, _keys = extract_context(
            history_local,
            ego_length,
            lead_length,
            self.dt,
        )
        relative = _relative_history(history_local, ego_length, lead_length)
        stats = self.sampler.prior.stats
        return {
            "context_states": normalize_numpy(
                history_local,
                stats,
                "context_states",
            ),
            "context_features": normalize_numpy(
                context_features,
                stats,
                "context_features",
            ),
            "relative_history": normalize_numpy(
                relative,
                stats,
                "relative_history",
            ),
            "raw_context_states": history_local,
        }

    @staticmethod
    def _vehicle_state(vehicle: Vehicle) -> np.ndarray:
        if isinstance(vehicle.action, dict):
            acceleration = float(vehicle.action.get("acceleration", 0.0))
        else:
            acceleration = 0.0
        vx = float(vehicle.speed) * float(np.cos(vehicle.heading))
        vy = float(vehicle.speed) * float(np.sin(vehicle.heading))
        ax = acceleration * float(np.cos(vehicle.heading))
        ay = acceleration * float(np.sin(vehicle.heading))
        return np.asarray(
            [vehicle.position[0], vehicle.position[1], vx, vy, ax, ay],
            dtype=np.float32,
        )

    def _closed_loop_risk(
        self,
        metrics: dict[str, float],
        trace: list[dict[str, float]],
    ) -> float:
        return apply_closed_loop_risk(
            metrics,
            trace,
            self.config,
            scoring_section="closed_loop_risk_scoring",
        )

    def rollout(
        self,
        initial_context: dict[str, Any],
        *,
        seed: int | None = None,
        fixed_plan: np.ndarray | None = None,
        episode_steps: int | None = None,
        plan_callback: (
            Callable[
                [dict[str, np.ndarray], int, int],
                dict[str, Any],
            ]
            | None
        ) = None,
    ) -> RolloutResult:
        raw_context = np.asarray(
            initial_context["raw_context_states"],
            dtype=np.float32,
        ).copy()
        ego_length = float(initial_context.get("ego_length", 4.8))
        lead_length = float(
            initial_context.get(
                "adv_length",
                initial_context.get("lead_length", 4.8),
            )
        )
        ego0 = raw_context[-1, 0]
        lead0 = raw_context[-1, 1]
        initial_gap = float(lead0[0] - ego0[0] - 0.5 * (ego_length + lead_length))
        if initial_gap <= self.initial_gap_min:
            raise RuntimeError(
                "Invalid initial context: gap "
                f"{initial_gap:.3f} <= {self.initial_gap_min:.3f}"
            )
        road = self._make_road()
        ego_speed, ego_yaw = _speed_and_yaw(ego0)
        lead_speed, lead_yaw = _speed_and_yaw(lead0)
        ego = IDMVehicle(
            road,
            position=np.asarray([ego0[0], ego0[1]], dtype=np.float64),
            heading=ego_yaw,
            speed=ego_speed,
            target_speed=self.ego_target_speed,
            enable_lane_change=bool(
                self.config.get("ego_response", {}).get(
                    "enable_lane_change",
                    False,
                )
            ),
        )
        lead = ScriptedLeadVehicle(
            road,
            position=np.asarray([lead0[0], lead0[1]], dtype=np.float64),
            heading=lead_yaw,
            speed=lead_speed,
        )
        ego.LENGTH = ego_length
        lead.LENGTH = lead_length
        if hasattr(ego, "diagonal"):
            ego.diagonal = float(np.sqrt(ego.LENGTH**2 + ego.WIDTH**2))
        if hasattr(lead, "diagonal"):
            lead.diagonal = float(np.sqrt(lead.LENGTH**2 + lead.WIDTH**2))
        road.vehicles = [ego, lead]
        if hasattr(ego, "front_vehicle"):
            ego.front_vehicle = lead

        history_world: deque[np.ndarray] = deque(maxlen=self.history_steps)
        for item in raw_context[-self.history_steps :]:
            v = np.asarray(item, dtype=np.float32).copy()
            history_world.append(v)

        num_generated_plans = 0
        plan: np.ndarray | None = (
            None if fixed_plan is None else np.asarray(fixed_plan, dtype=np.float32)
        )
        plan_cursor = 0
        lead_accel = float(lead0[4])
        prev_lead_accel = lead_accel
        min_ttc = 1000.0
        min_gap = float("inf")
        min_ego_accel = 0.0
        lead_physics_penalty = 0.0
        action_clip_count = 0
        jerk_violation_count = 0
        speed_negative_count = 0
        speed_violation_count = 0
        lead_accel_values: list[float] = []
        lead_jerk_values: list[float] = []
        lead_speed_values: list[float] = []
        trace: list[dict[str, float]] = []
        action_cfg = self.config.get("physics", {})
        dyn_cfg = self.config.get("dynamics", {})
        ax_min = float(action_cfg.get("ax_min", -8.0))
        ax_max = float(action_cfg.get("ax_max", 4.0))
        jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
        steering_abs_max = float(dyn_cfg.get("steering_abs_max", 0.5))
        steering_rate_abs_max = float(dyn_cfg.get("steering_rate_abs_max", 1.0))
        wheelbase = max(float(dyn_cfg.get("wheelbase", 5.0)), 1e-6)
        speed_min = float(dyn_cfg.get("speed_min", 0.0))
        speed_max = float(dyn_cfg.get("speed_max", self.speed_limit))
        initial_accel_lower, initial_accel_upper = _accel_bounds_for_speed(
            max(float(lead.speed), 0.0),
            self.dt,
            ax_min,
            ax_max,
            speed_min,
            speed_max,
        )
        lead_accel = float(
            np.clip(lead_accel, initial_accel_lower, initial_accel_upper)
        )
        prev_lead_accel = lead_accel
        lead_steering = 0.0
        schema_rep = self.sampler.prior.schema.get("action_representation")
        config_rep = self.sampler.prior.config.get("action", {}).get(
            "representation",
            "jerk",
        )
        rep = str(schema_rep or config_rep).lower()

        total_steps = (
            self.episode_steps if episode_steps is None else int(episode_steps)
        )
        if total_steps <= 0:
            raise ValueError(f"episode_steps must be positive, got {total_steps}")
        active_prior_plan: np.ndarray | None = None
        executed_actions: list[np.ndarray] = []
        executed_prior_actions: list[np.ndarray] = []
        plan_summaries: list[dict[str, float]] = []

        for step in range(total_steps):
            needs_plan = (
                plan is None
                or (
                    fixed_plan is None
                    and (
                        plan_cursor >= len(plan) or plan_cursor >= self.commit_steps_max
                    )
                )
                or (fixed_plan is not None and plan_cursor >= len(plan))
            )
            if needs_plan:
                if fixed_plan is not None and plan_cursor >= len(plan):
                    break
                obs = self._build_observation(
                    history_world,
                    ego_length,
                    lead_length,
                )
                if plan_callback is None:
                    raise ValueError(
                        "subset rollout requires plan_callback or fixed_plan"
                    )
                else:
                    payload = plan_callback(obs, num_generated_plans, step)
                    if "plan" not in payload:
                        raise KeyError("plan_callback must return a 'plan' array")
                    plan = np.asarray(payload["plan"], dtype=np.float32)
                    prior = payload.get("prior_plan", plan)
                    active_prior_plan = np.asarray(prior, dtype=np.float32)
                    summary = payload.get("summary", {})
                    plan_summaries.append(
                        {
                            "start_step": float(step),
                            **{
                                str(key): float(value)
                                for key, value in dict(summary).items()
                                if np.isfinite(float(value))
                            },
                        }
                    )
                plan_cursor = 0
                num_generated_plans += 1
            elif fixed_plan is not None and num_generated_plans == 0:
                num_generated_plans += 1
                active_prior_plan = plan

            cursor = plan_cursor
            raw_action_row = np.asarray(plan[cursor], dtype=np.float32)
            action_row = raw_action_row.copy()
            if active_prior_plan is not None and cursor < len(active_prior_plan):
                prior_action_row = np.asarray(
                    active_prior_plan[cursor],
                    dtype=np.float32,
                )
            else:
                prior_action_row = action_row
            plan_cursor += 1
            speed_before = max(float(lead.speed), 0.0)
            if rep == "jerk":
                raw_jerk = float(raw_action_row[0])
                jerk = float(
                    np.clip(
                        raw_jerk,
                        -jerk_abs_max,
                        jerk_abs_max,
                    )
                )
                proposed_accel = prev_lead_accel + jerk * self.dt
                accel_lower, accel_upper = _accel_bounds_for_speed(
                    speed_before,
                    self.dt,
                    ax_min,
                    ax_max,
                    speed_min,
                    speed_max,
                )
                jerk_lower = prev_lead_accel - jerk_abs_max * self.dt
                jerk_upper = prev_lead_accel + jerk_abs_max * self.dt
                accel_lower = max(accel_lower, jerk_lower)
                accel_upper = min(accel_upper, jerk_upper)
                if accel_lower > accel_upper:
                    accel_lower = max(ax_min, jerk_lower)
                    accel_upper = min(ax_max, jerk_upper)
                lead_accel = float(np.clip(proposed_accel, accel_lower, accel_upper))
                jerk = (lead_accel - prev_lead_accel) / max(self.dt, 1e-6)
                action_row[0] = jerk
            else:
                raw_accel = float(raw_action_row[0])
                accel_lower, accel_upper = _accel_bounds_for_speed(
                    speed_before,
                    self.dt,
                    ax_min,
                    ax_max,
                    speed_min,
                    speed_max,
                )
                accel_lower = max(
                    accel_lower,
                    prev_lead_accel - jerk_abs_max * self.dt,
                )
                accel_upper = min(
                    accel_upper,
                    prev_lead_accel + jerk_abs_max * self.dt,
                )
                if accel_lower > accel_upper:
                    accel_lower = max(
                        ax_min,
                        prev_lead_accel - jerk_abs_max * self.dt,
                    )
                    accel_upper = min(
                        ax_max,
                        prev_lead_accel + jerk_abs_max * self.dt,
                    )
                lead_accel = float(np.clip(raw_accel, accel_lower, accel_upper))
                jerk = (lead_accel - prev_lead_accel) / max(self.dt, 1e-6)
                action_row[0] = lead_accel
            if self.dynamics_model == "kinematic_bicycle":
                previous_steering = lead_steering
                steering_rate = (
                    float(raw_action_row[1]) if raw_action_row.size > 1 else 0.0
                )
                steering_rate = float(
                    np.clip(
                        steering_rate,
                        -steering_rate_abs_max,
                        steering_rate_abs_max,
                    )
                )
                lead_steering = float(
                    np.clip(
                        lead_steering + steering_rate * self.dt,
                        -steering_abs_max,
                        steering_abs_max,
                    )
                )
                if action_row.size > 1:
                    action_row[1] = (lead_steering - previous_steering) / max(
                        self.dt, 1e-6
                    )
            else:
                steering_rate = 0.0
                lead_steering = 0.0
            lead_physics_penalty += _bound_residual(
                lead_accel,
                ax_min,
                ax_max,
            )
            lead_physics_penalty += _bound_residual(
                jerk,
                -jerk_abs_max,
                jerk_abs_max,
            )
            action_clip_count += int(np.max(np.abs(raw_action_row - action_row)) > 1e-6)
            jerk_violation_count += int(abs(jerk) > jerk_abs_max + 1e-6)
            executed_actions.append(action_row.copy())
            executed_prior_actions.append(prior_action_row.copy())
            prev_lead_accel = lead_accel
            lead_accel_values.append(float(lead_accel))
            lead_jerk_values.append(float(jerk))
            lead.set_control(lead_accel, lead_steering)
            if self.dynamics_model == "kinematic_bicycle":
                lead_speed_before = max(float(lead.speed), 0.0)
                lead_position_next = np.asarray(
                    [
                        float(lead.position[0])
                        + lead_speed_before * float(np.cos(lead.heading)) * self.dt,
                        float(lead.position[1])
                        + lead_speed_before * float(np.sin(lead.heading)) * self.dt,
                    ],
                    dtype=np.float64,
                )
                lead_heading_next = float(
                    lead.heading
                    + lead_speed_before
                    / wheelbase
                    * float(np.tan(lead_steering))
                    * self.dt
                )
                lead_speed_next = float(
                    np.clip(
                        lead_speed_before + lead_accel * self.dt,
                        speed_min,
                        speed_max,
                    )
                )
                lead.set_forced_state(
                    lead_position_next,
                    lead_heading_next,
                    lead_speed_next,
                )

            road.act()
            road.step(self.dt)
            speed_negative_count += int(
                float(lead.speed) < float(action_cfg.get("speed_min", 0.0))
            )
            speed_violation_count += int(
                float(lead.speed) < speed_min - 1e-6
                or float(lead.speed) > speed_max + 1e-6
            )
            ego_state = self._vehicle_state(ego)
            lead_state = self._vehicle_state(lead)
            history_world.append(
                np.stack([ego_state, lead_state], axis=0).astype(np.float32)
            )

            gap = float(
                lead.position[0] - ego.position[0] - 0.5 * (ego_length + lead_length)
            )
            closing = float(ego.speed - lead.speed)
            ttc = gap / max(closing, 1e-6) if closing > 1e-6 else 1000.0
            ego_accel = float(ego.action.get("acceleration", 0.0))
            min_gap = min(min_gap, gap)
            min_ttc = min(min_ttc, ttc)
            min_ego_accel = min(min_ego_accel, ego_accel)
            lead_speed_values.append(float(lead.speed))
            trace.append(
                {
                    "step": float(step),
                    "gap": gap,
                    "ttc": float(ttc),
                    "ego_accel": ego_accel,
                    "ego_speed": float(ego.speed),
                    "ego_position": float(ego.position[0]),
                    "ego_y": float(ego.position[1]),
                    "ego_yaw": float(ego.heading),
                    "ego_action_accel": ego_accel,
                    "ego_action_steering": float(ego.action.get("steering", 0.0)),
                    "lead_speed": float(lead.speed),
                    "lead_position": float(lead.position[0]),
                    "lead_y": float(lead.position[1]),
                    "lead_yaw": float(lead.heading),
                    "lead_accel": float(lead_accel),
                    "lead_jerk": float(jerk),
                    "lead_steering": float(lead_steering),
                    "lead_steering_rate": float(steering_rate),
                }
            )
            if ego.crashed:
                break

        collision = bool(ego.crashed)
        risk_cfg = self.config.get("closed_loop_risk", {})
        near_gap = float(risk_cfg.get("near_collision_gap", 2.0))
        hard_brake_threshold = float(risk_cfg.get("hard_brake_threshold", -4.0))
        physics_penalty_mean = float(lead_physics_penalty / max(len(trace), 1))
        physical_feasible = bool(
            physics_penalty_mean <= 1e-8
            and jerk_violation_count == 0
            and speed_negative_count == 0
            and speed_violation_count == 0
        )
        metrics = {
            "collision": float(collision),
            "invalid_initial_context": 0.0,
            "initial_gap": float(initial_gap),
            "min_ttc": float(min_ttc),
            "min_gap": float(min_gap),
            "final_gap": float(trace[-1]["gap"]) if trace else float(min_gap),
            "min_ego_accel": float(min_ego_accel),
            "near_collision": float(min_gap < near_gap),
            "hard_brake": float(min_ego_accel <= hard_brake_threshold),
            "lead_physics_penalty": physics_penalty_mean,
            "physical_feasible": float(physical_feasible),
            "lead_accel_mean": (
                float(np.mean(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_accel_std": (
                float(np.std(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_accel_min": (
                float(np.min(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_accel_max": (
                float(np.max(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_jerk_mean": (
                float(np.mean(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_std": (
                float(np.std(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_min": (
                float(np.min(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_max": (
                float(np.max(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_abs_mean": (
                float(np.mean(np.abs(lead_jerk_values))) if lead_jerk_values else 0.0
            ),
            "lead_jerk_abs_max": (
                float(np.max(np.abs(lead_jerk_values))) if lead_jerk_values else 0.0
            ),
            "lead_speed_mean": (
                float(np.mean(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "lead_speed_std": (
                float(np.std(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "lead_speed_min": (
                float(np.min(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "lead_speed_max": (
                float(np.max(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "action_clip_rate": float(action_clip_count / max(len(trace), 1)),
            "jerk_violation_rate": float(jerk_violation_count / max(len(trace), 1)),
            "speed_negative_rate": float(speed_negative_count / max(len(trace), 1)),
            "speed_violation_rate": float(speed_violation_count / max(len(trace), 1)),
            "num_generated_plans": float(num_generated_plans),
            "steps": float(len(trace)),
        }
        return RolloutResult(
            risk_score=self._closed_loop_risk(metrics, trace),
            metrics=metrics,
            num_generated_plans=num_generated_plans,
            trace=trace,
            actions=(
                np.asarray(executed_actions, dtype=np.float32)
                if executed_actions
                else np.zeros((0, 1), dtype=np.float32)
            ),
            prior_actions=(
                np.asarray(executed_prior_actions, dtype=np.float32)
                if executed_prior_actions
                else np.zeros((0, 1), dtype=np.float32)
            ),
            plan_summaries=plan_summaries,
        )

    def rollout_pre_sampled_plan(
        self,
        initial_context: dict[str, Any],
        plan: np.ndarray,
        *,
        episode_steps: int | None = None,
    ) -> RolloutResult:
        return self.rollout(
            initial_context,
            fixed_plan=plan,
            episode_steps=episode_steps,
        )
