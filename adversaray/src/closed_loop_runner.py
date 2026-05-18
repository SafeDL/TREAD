"""Closed-loop highway-env rollouts for prior-guided diffusion policies."""
from __future__ import annotations

import sys
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HIGHWAY_ROOT = ROOT / "HighwayEnv"
if HIGHWAY_ROOT.exists() and str(HIGHWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGHWAY_ROOT))

from diffusion.src.features import extract_context  # noqa: E402
from diffusion.src.data import _build_world_states, prepare_recording  # noqa: E402

from .normalization_adapter import normalize_numpy  # noqa: E402
from .prior_guided_sampler import PriorGuidedDiffusionSampler  # noqa: E402
from .rss import RSSConfig, rss_safe_distance  # noqa: E402

try:
    from process_highD.src.io_utils import load_config, resolve_data_path  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001
    load_config = None
    resolve_data_path = None

try:
    import pandas as pd  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001
    pd = None

logger = logging.getLogger(__name__)

try:
    from highway_env.road.road import Road, RoadNetwork  # type: ignore  # noqa: E402
    from highway_env.vehicle.behavior import IDMVehicle  # type: ignore  # noqa: E402
    from highway_env.vehicle.kinematics import Vehicle  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001
    Road = None
    RoadNetwork = None

    class Vehicle:  # type: ignore[no-redef]
        LENGTH = 5.0
        WIDTH = 2.0

        def __init__(self, road: Any, position: Any, heading: float = 0.0, speed: float = 0.0) -> None:
            self.road = road
            self.position = np.asarray(position, dtype=np.float64)
            self.heading = float(heading)
            self.speed = float(speed)
            self.action = {"steering": 0.0, "acceleration": 0.0}
            self.crashed = False

        def act(self, action: dict | str = None) -> None:
            if isinstance(action, dict):
                self.action = action

        def step(self, dt: float) -> None:
            self.position[0] += self.speed * float(dt)
            self.speed = max(0.0, self.speed + float(self.action.get("acceleration", 0.0)) * float(dt))

    class IDMVehicle(Vehicle):  # type: ignore[no-redef]
        COMFORT_ACC_MAX = 3.0
        COMFORT_ACC_MIN = -5.0
        DISTANCE_WANTED = 10.0
        TIME_WANTED = 1.5
        DELTA = 4.0

        def __init__(
            self,
            road: Any,
            position: Any,
            heading: float = 0.0,
            speed: float = 0.0,
            target_speed: float | None = None,
            enable_lane_change: bool = False,
            **_: Any,
        ) -> None:
            super().__init__(road, position, heading, speed)
            self.target_speed = float(target_speed if target_speed is not None else speed)
            self.front_vehicle: Vehicle | None = None

        def act(self, action: dict | str = None) -> None:
            front = self.front_vehicle
            target = max(self.target_speed, 1e-6)
            accel = self.COMFORT_ACC_MAX * (1.0 - (max(self.speed, 0.0) / target) ** self.DELTA)
            if front is not None:
                gap = front.position[0] - self.position[0] - 0.5 * (self.LENGTH + front.LENGTH)
                closing = self.speed - front.speed
                desired = self.DISTANCE_WANTED + max(0.0, self.speed * self.TIME_WANTED + self.speed * closing / 10.0)
                accel -= self.COMFORT_ACC_MAX * (desired / max(gap, 1e-3)) ** 2
            self.action = {"steering": 0.0, "acceleration": float(np.clip(accel, self.COMFORT_ACC_MIN, self.COMFORT_ACC_MAX))}

    class _FallbackRoad:
        def __init__(self) -> None:
            self.vehicles: list[Vehicle] = []

        def act(self) -> None:
            for vehicle in self.vehicles:
                vehicle.act()

        def step(self, dt: float) -> None:
            for vehicle in self.vehicles:
                vehicle.step(dt)
            if len(self.vehicles) >= 2:
                ego, lead = self.vehicles[0], self.vehicles[1]
                gap = lead.position[0] - ego.position[0] - 0.5 * (ego.LENGTH + lead.LENGTH)
                if gap <= 0.0:
                    ego.crashed = True
                    lead.crashed = True


@dataclass
class RolloutResult:
    reward: float
    metrics: dict[str, float]
    log_prob_sum: torch.Tensor
    prior_kl_sum: torch.Tensor
    guidance_norm_sum: torch.Tensor
    trace: list[dict[str, float]] = field(default_factory=list)


class ScriptedLeadVehicle(Vehicle):
    """A highway-env vehicle whose longitudinal action is set externally."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commanded_acceleration = 0.0

    def set_acceleration(self, acceleration: float) -> None:
        self.commanded_acceleration = float(acceleration)

    def act(self, action: dict | str = None) -> None:
        Vehicle.act(self, {"steering": 0.0, "acceleration": self.commanded_acceleration})


def _relative_history(history_local: np.ndarray, ego_length: float, lead_length: float) -> np.ndarray:
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
    current_ego = history_world[-1, 0].copy()
    out = np.asarray(history_world, dtype=np.float32).copy()
    out[:, :, 0] -= current_ego[0]
    out[:, :, 1] -= current_ego[1]
    return out.astype(np.float32)


class ClosedLoopFollowingRunner:
    """Roll a generated lead plan on a highway-env vehicle-dynamics car-following road."""

    def __init__(self, sampler: PriorGuidedDiffusionSampler, config: dict[str, Any]) -> None:
        self.sampler = sampler
        self.config = config
        env_cfg = config.get("env", {})
        prior_cfg = sampler.prior.model.denoiser.cfg
        target_fps = float(sampler.prior.config.get("sampling", {}).get("target_fps", 25.0))
        self.dt = float(env_cfg.get("dt", 1.0 / max(target_fps, 1.0)))
        self.history_steps = int(prior_cfg.history_steps)
        self.episode_steps = int(env_cfg.get("episode_steps", min(25, prior_cfg.horizon_steps)))
        self.commit_steps_max = int(env_cfg.get("commit_steps_max", 1))
        self.lanes_count = int(env_cfg.get("lanes_count", 1))
        self.speed_limit = float(env_cfg.get("speed_limit", 40.0))
        self.ego_target_speed = float(env_cfg.get("ego_target_speed", 30.0))
        self.initial_gap_min = float(env_cfg.get("initial_gap_min", 0.1))
        self.skip_invalid_initial_context = bool(env_cfg.get("skip_invalid_initial_context", True))
        self.invalid_context_reward = float(env_cfg.get("invalid_context_reward", 0.0))
        self.reconstruct_highd_events = bool(env_cfg.get("reconstruct_highd_events", False))
        self.rss_cfg = RSSConfig.from_config(config)
        runtime = config.get("_runtime", {})
        paths = config.get("paths", {})
        self.highd_events_csv = Path(
            runtime.get("highd_events_csv", paths.get("highd_events_csv", ROOT / "data/highd_events/events.csv"))
        )
        self.highd_raw_dir = Path(
            runtime.get("highd_raw_dir", paths.get("highd_raw_dir", ROOT / "highD_dataset/Matlab/data"))
        )
        self.highd_config_path = Path(
            runtime.get(
                "highd_config",
                paths.get("highd_config", ROOT / "process_highD/scripts/configs/highd_default.yaml"),
            )
        )
        self._events_table: Any | None = None
        self._recording_cache: dict[int, Any] = {}
        self._highd_config: dict[str, Any] | None = None

    def _make_road(self) -> Any:
        if Road is None or RoadNetwork is None:
            return _FallbackRoad()
        return Road(
            network=RoadNetwork.straight_road_network(self.lanes_count, speed_limit=self.speed_limit),
            np_random=np.random.RandomState(int(self.config.get("training", {}).get("seed", 42))),
            record_history=False,
        )

    def _load_events_table(self) -> Any | None:
        if not self.highd_events_csv.exists() or pd is None:
            return None
        if self._events_table is None:
            events = pd.read_csv(self.highd_events_csv)
            if "event_type" in events.columns:
                events = events[events["event_type"] == "following"]
            self._events_table = events.reset_index(drop=True)
        return self._events_table

    def _load_highd_config(self) -> dict[str, Any] | None:
        if load_config is None or not self.highd_config_path.exists():
            return None
        if self._highd_config is None:
            cfg = load_config(str(self.highd_config_path))
            if self.highd_raw_dir.exists():
                cfg.setdefault("paths", {})["raw_dir"] = str(self.highd_raw_dir)
            self._highd_config = cfg
        return self._highd_config

    def _resolved_highd_raw_dir(self, cfg: dict[str, Any]) -> Path | None:
        raw_dir = cfg.get("paths", {}).get("raw_dir")
        if raw_dir is None:
            return None
        if resolve_data_path is not None:
            return resolve_data_path(str(raw_dir), str(self.highd_config_path))
        raw_path = Path(raw_dir)
        return raw_path if raw_path.is_absolute() else (self.highd_config_path.parent / raw_path).resolve()

    def _maybe_reconstruct_highd_context(
        self,
        initial_context: dict[str, Any],
        ego_length: float,
        lead_length: float,
    ) -> tuple[np.ndarray, float, float] | None:
        if not self.reconstruct_highd_events:
            return None
        event_id = initial_context.get("event_id")
        recording_id = initial_context.get("recording_id")
        anchor_frame = initial_context.get("anchor_frame")
        if event_id is None or recording_id is None or anchor_frame is None:
            return None
        events = self._load_events_table()
        cfg = self._load_highd_config()
        if events is None or cfg is None:
            return None
        event_id = str(event_id)
        recording_id = int(recording_id)
        matches = events[(events["event_id"].astype(str) == event_id) & (events["recording_id"].astype(int) == recording_id)]
        if matches.empty:
            return None
        raw_dir = self._resolved_highd_raw_dir(cfg)
        if raw_dir is None or not raw_dir.exists():
            return None
        try:
            if recording_id not in self._recording_cache:
                self._recording_cache[recording_id] = prepare_recording(raw_dir, recording_id, cfg)
            recording = self._recording_cache[recording_id]
            event = matches.iloc[0]
            end_frame = int(anchor_frame)
            frames = np.arange(end_frame - self.history_steps + 1, end_frame + 1, dtype=np.int64)
            states = _build_world_states(recording, event, frames)
            if states is None:
                return None
            meta = recording.tracks_meta
            ego_length = float(meta.loc[int(event["ego_id"])]["width"]) if int(event["ego_id"]) in meta.index else ego_length
            lead_length = float(meta.loc[int(event["target_id"])]["width"]) if int(event["target_id"]) in meta.index else lead_length
            return _localize_history(states), ego_length, lead_length
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falling back to dataset context for event %s: %s", event_id, exc)
            return None

    def _build_observation(
        self,
        history_world: deque[np.ndarray],
        ego_length: float,
        lead_length: float,
    ) -> dict[str, np.ndarray]:
        hist = np.asarray(list(history_world), dtype=np.float32)
        history_local = _localize_history(hist)
        context_features, _keys = extract_context(history_local, ego_length, lead_length, self.dt)
        relative = _relative_history(history_local, ego_length, lead_length)
        stats = self.sampler.prior.stats
        return {
            "context_states": normalize_numpy(history_local, stats, "context_states"),
            "context_features": normalize_numpy(context_features, stats, "context_features"),
            "relative_history": normalize_numpy(relative, stats, "relative_history"),
            "raw_context_states": history_local,
        }

    @staticmethod
    def _vehicle_state(vehicle: Vehicle) -> np.ndarray:
        acceleration = float(vehicle.action.get("acceleration", 0.0)) if isinstance(vehicle.action, dict) else 0.0
        return np.asarray([vehicle.position[0], vehicle.position[1], vehicle.speed, 0.0, acceleration, 0.0], dtype=np.float32)

    def _reward(self, metrics: dict[str, float], *, prior_kl_sum: float = 0.0) -> float:
        cfg = self.config.get("reward", {})
        ttc_target = float(cfg.get("ttc_target", 3.0))
        gap_target = float(cfg.get("gap_target", 3.0))
        hard_brake_threshold = float(cfg.get("hard_brake_threshold", -4.0))
        collision = float(metrics.get("collision_valid", metrics["collision"]))
        ttc_risk = max(0.0, ttc_target - float(metrics["min_ttc"])) / max(ttc_target, 1e-6)
        gap_risk = max(0.0, gap_target - float(metrics["min_gap"])) / max(gap_target, 1e-6)
        rss_scale = max(float(cfg.get("rss_scale", 20.0)), 1e-6)
        rss_clip = float(cfg.get("rss_risk_clip", 1.0))
        rss_risk = float(np.clip(max(0.0, -float(metrics["min_rss_margin"])) / rss_scale, 0.0, rss_clip))
        hard_brake = max(0.0, hard_brake_threshold - float(metrics["min_ego_accel"])) / max(abs(hard_brake_threshold), 1e-6)
        risk_reward = float(
            float(cfg.get("collision_bonus", 20.0)) * collision
            + float(cfg.get("ttc_weight", 4.0)) * ttc_risk
            + float(cfg.get("gap_weight", 2.0)) * gap_risk
            + float(cfg.get("rss_weight", 0.25)) * rss_risk
            + float(cfg.get("hard_brake_weight", 1.0)) * hard_brake
        )
        gate = 1.0
        if bool(cfg.get("naturalness_gate_enabled", False)):
            kl_gate = float(cfg.get("prior_kl_gate", 0.0))
            phy_gate = float(cfg.get("physics_gate", 0.0))
            jerk_gate = float(cfg.get("jerk_violation_gate", 0.0))
            if kl_gate > 0.0 and prior_kl_sum > kl_gate:
                gate *= kl_gate / max(prior_kl_sum, 1e-6)
            if phy_gate > 0.0 and float(metrics["lead_physics_penalty"]) > phy_gate:
                gate *= phy_gate / max(float(metrics["lead_physics_penalty"]), 1e-6)
            if jerk_gate > 0.0 and float(metrics.get("jerk_violation_rate", 0.0)) > jerk_gate:
                gate *= jerk_gate / max(float(metrics["jerk_violation_rate"]), 1e-6)
        metrics["naturalness_gate"] = float(gate)
        return float(risk_reward * gate - float(cfg.get("lead_physics_weight", 0.1)) * float(metrics["lead_physics_penalty"]))

    def rollout(self, initial_context: dict[str, Any], *, seed: int | None = None) -> RolloutResult:
        raw_context = np.asarray(initial_context["raw_context_states"], dtype=np.float32).copy()
        raw_context[:, :, 1] = 0.0
        ego_length = float(initial_context.get("ego_length", 4.8))
        lead_length = float(initial_context.get("adv_length", initial_context.get("lead_length", 4.8)))
        rebuilt = self._maybe_reconstruct_highd_context(initial_context, ego_length, lead_length)
        if rebuilt is not None:
            raw_context, ego_length, lead_length = rebuilt
            raw_context[:, :, 1] = 0.0
        ego0 = raw_context[-1, 0]
        lead0 = raw_context[-1, 1]
        initial_gap = float(lead0[0] - ego0[0] - 0.5 * (ego_length + lead_length))
        if initial_gap <= self.initial_gap_min:
            if self.skip_invalid_initial_context:
                device = self.sampler.prior.device
                zero = torch.zeros((), dtype=torch.float32, device=device)
                metrics = {
                    "collision": 0.0,
                    "collision_valid": 0.0,
                    "invalid_collision": 0.0,
                    "invalid_initial_context": 1.0,
                    "initial_gap": initial_gap,
                    "min_ttc": 1000.0,
                    "min_gap": initial_gap,
                    "min_rss_margin": initial_gap,
                    "min_ego_accel": 0.0,
                    "near_collision": 0.0,
                    "hard_brake": 0.0,
                    "lead_physics_penalty": 0.0,
                    "lead_accel_mean": 0.0,
                    "lead_accel_std": 0.0,
                    "lead_accel_min": 0.0,
                    "lead_accel_max": 0.0,
                    "lead_jerk_mean": 0.0,
                    "lead_jerk_std": 0.0,
                    "lead_jerk_min": 0.0,
                    "lead_jerk_max": 0.0,
                    "lead_jerk_abs_mean": 0.0,
                    "lead_jerk_abs_max": 0.0,
                    "lead_speed_mean": 0.0,
                    "lead_speed_std": 0.0,
                    "lead_speed_min": 0.0,
                    "lead_speed_max": 0.0,
                    "action_clip_rate": 0.0,
                    "jerk_violation_rate": 0.0,
                    "speed_negative_rate": 0.0,
                    "naturalness_gate": 1.0,
                    "steps": 0.0,
                }
                return RolloutResult(
                    reward=self.invalid_context_reward,
                    metrics=metrics,
                    log_prob_sum=zero,
                    prior_kl_sum=zero,
                    guidance_norm_sum=zero,
                    trace=[],
                )
            raw_context[-1, 1, 0] = raw_context[-1, 0, 0] + 0.5 * (ego_length + lead_length) + self.initial_gap_min
            lead0 = raw_context[-1, 1]
            initial_gap = self.initial_gap_min
        road = self._make_road()
        ego = IDMVehicle(
            road,
            position=np.asarray([ego0[0], 0.0], dtype=np.float64),
            heading=0.0,
            speed=max(float(ego0[2]), 0.0),
            target_speed=self.ego_target_speed,
            enable_lane_change=False,
        )
        lead = ScriptedLeadVehicle(
            road,
            position=np.asarray([lead0[0], 0.0], dtype=np.float64),
            heading=0.0,
            speed=max(float(lead0[2]), 0.0),
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
            v[:, 1] = 0.0
            history_world.append(v)

        device = self.sampler.prior.device
        log_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
        prior_kl_sum = torch.zeros((), dtype=torch.float32, device=device)
        guidance_norm_sum = torch.zeros((), dtype=torch.float32, device=device)
        plan: np.ndarray | None = None
        plan_cursor = 0
        lead_accel = float(lead0[4])
        prev_lead_accel = lead_accel
        min_ttc = 1000.0
        min_gap = float("inf")
        min_rss_margin = float("inf")
        min_ego_accel = 0.0
        collision_gap = float("inf")
        collision_following_order = True
        lead_physics_penalty = 0.0
        action_clip_count = 0
        jerk_violation_count = 0
        speed_negative_count = 0
        lead_accel_values: list[float] = []
        lead_jerk_values: list[float] = []
        lead_speed_values: list[float] = []
        trace: list[dict[str, float]] = []
        action_cfg = self.config.get("physics", self.config.get("action", {}))
        ax_min = float(action_cfg.get("ax_min", -8.0))
        ax_max = float(action_cfg.get("ax_max", 4.0))
        jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
        rep = str(self.sampler.prior.schema.get("action_representation", self.sampler.prior.config.get("action", {}).get("representation", "jerk"))).lower()

        for step in range(self.episode_steps):
            if plan is None or plan_cursor >= len(plan) or plan_cursor >= self.commit_steps_max:
                obs = self._build_observation(history_world, ego_length, lead_length)
                sample = self.sampler.sample(
                    torch.from_numpy(obs["context_states"][None]).float(),
                    torch.from_numpy(obs["context_features"][None]).float(),
                    torch.from_numpy(obs["relative_history"][None]).float(),
                    ego_length=torch.tensor([ego_length], dtype=torch.float32),
                    adv_length=torch.tensor([lead_length], dtype=torch.float32),
                    num_samples=1,
                    seed=None if seed is None else int(seed) + step,
                )
                plan = sample.raw_actions[0].detach().cpu().numpy().astype(np.float32)
                plan_cursor = 0
                log_prob_sum = log_prob_sum + sample.trajectory_log_prob[0]
                prior_kl_sum = prior_kl_sum + sample.prior_kl[0]
                guidance_norm_sum = guidance_norm_sum + sample.guidance_norm[0]

            action_value = float(plan[plan_cursor, 0])
            plan_cursor += 1
            if rep == "jerk":
                lead_accel = lead_accel + action_value * self.dt
                jerk = action_value
            else:
                jerk = (action_value - prev_lead_accel) / max(self.dt, 1e-6)
                lead_accel = action_value
            commanded_accel = lead_accel
            lead_physics_penalty += max(0.0, ax_min - lead_accel) ** 2
            lead_physics_penalty += max(0.0, lead_accel - ax_max) ** 2
            lead_physics_penalty += max(0.0, abs(jerk) - jerk_abs_max) ** 2
            lead_accel = float(np.clip(lead_accel, ax_min, ax_max))
            prev_lead_accel = lead_accel
            action_clip_count += int(abs(commanded_accel - lead_accel) > 1e-6)
            jerk_violation_count += int(abs(jerk) > jerk_abs_max)
            lead_accel_values.append(float(lead_accel))
            lead_jerk_values.append(float(jerk))
            lead.set_acceleration(lead_accel)

            road.act()
            road.step(self.dt)
            speed_negative_count += int(float(lead.speed) < float(action_cfg.get("speed_min", 0.0)))
            ego_state = self._vehicle_state(ego)
            lead_state = self._vehicle_state(lead)
            history_world.append(np.stack([ego_state, lead_state], axis=0).astype(np.float32))

            gap = float(lead.position[0] - ego.position[0] - 0.5 * (ego_length + lead_length))
            closing = float(ego.speed - lead.speed)
            ttc = gap / max(closing, 1e-6) if closing > 1e-6 else 1000.0
            safe = float(rss_safe_distance(torch.tensor([ego.speed]), torch.tensor([max(lead.speed, 0.0)]), self.rss_cfg)[0])
            rss_margin = gap - safe
            ego_accel = float(ego.action.get("acceleration", 0.0))
            min_gap = min(min_gap, gap)
            min_ttc = min(min_ttc, ttc)
            min_rss_margin = min(min_rss_margin, rss_margin)
            min_ego_accel = min(min_ego_accel, ego_accel)
            lead_speed_values.append(float(lead.speed))
            trace.append(
                {
                    "step": float(step),
                    "gap": gap,
                    "ttc": float(ttc),
                    "rss_margin": float(rss_margin),
                    "ego_accel": ego_accel,
                    "ego_speed": float(ego.speed),
                    "lead_speed": float(lead.speed),
                    "lead_accel": float(lead_accel),
                    "lead_jerk": float(jerk),
                }
            )
            if ego.crashed or lead.crashed:
                collision_gap = gap
                collision_following_order = bool(float(ego.position[0]) <= float(lead.position[0]))
                break

        collision = bool(ego.crashed or lead.crashed)
        collision_valid = bool(collision and collision_gap <= 0.0 and collision_following_order)
        reward_cfg = self.config.get("reward", {})
        near_gap = float(reward_cfg.get("near_collision_gap", 2.0))
        hard_brake_threshold = float(reward_cfg.get("hard_brake_threshold", -4.0))
        metrics = {
            "collision": float(collision),
            "collision_valid": float(collision_valid),
            "invalid_collision": float(collision and not collision_valid),
            "invalid_initial_context": 0.0,
            "initial_gap": float(initial_gap),
            "min_ttc": float(min_ttc),
            "min_gap": float(min_gap),
            "min_rss_margin": float(min_rss_margin),
            "min_ego_accel": float(min_ego_accel),
            "near_collision": float(min_gap < near_gap),
            "hard_brake": float(min_ego_accel <= hard_brake_threshold),
            "lead_physics_penalty": float(lead_physics_penalty / max(len(trace), 1)),
            "lead_accel_mean": float(np.mean(lead_accel_values)) if lead_accel_values else 0.0,
            "lead_accel_std": float(np.std(lead_accel_values)) if lead_accel_values else 0.0,
            "lead_accel_min": float(np.min(lead_accel_values)) if lead_accel_values else 0.0,
            "lead_accel_max": float(np.max(lead_accel_values)) if lead_accel_values else 0.0,
            "lead_jerk_mean": float(np.mean(lead_jerk_values)) if lead_jerk_values else 0.0,
            "lead_jerk_std": float(np.std(lead_jerk_values)) if lead_jerk_values else 0.0,
            "lead_jerk_min": float(np.min(lead_jerk_values)) if lead_jerk_values else 0.0,
            "lead_jerk_max": float(np.max(lead_jerk_values)) if lead_jerk_values else 0.0,
            "lead_jerk_abs_mean": float(np.mean(np.abs(lead_jerk_values))) if lead_jerk_values else 0.0,
            "lead_jerk_abs_max": float(np.max(np.abs(lead_jerk_values))) if lead_jerk_values else 0.0,
            "lead_speed_mean": float(np.mean(lead_speed_values)) if lead_speed_values else 0.0,
            "lead_speed_std": float(np.std(lead_speed_values)) if lead_speed_values else 0.0,
            "lead_speed_min": float(np.min(lead_speed_values)) if lead_speed_values else 0.0,
            "lead_speed_max": float(np.max(lead_speed_values)) if lead_speed_values else 0.0,
            "action_clip_rate": float(action_clip_count / max(len(trace), 1)),
            "jerk_violation_rate": float(jerk_violation_count / max(len(trace), 1)),
            "speed_negative_rate": float(speed_negative_count / max(len(trace), 1)),
            "steps": float(len(trace)),
        }
        prior_kl_value = float(prior_kl_sum.detach().cpu())
        return RolloutResult(
            reward=self._reward(metrics, prior_kl_sum=prior_kl_value),
            metrics=metrics,
            log_prob_sum=log_prob_sum,
            prior_kl_sum=prior_kl_sum,
            guidance_norm_sum=guidance_norm_sum,
            trace=trace,
        )
