"""highway-env IDM ego response rollout helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

try:
    from highway_env.road.road import Road, RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.kinematics import Vehicle
except ImportError as exc:
    raise RuntimeError(
        "Failed to import the required local highway-env package. "
        f"Package path: {HIGHWAY_PACKAGE}. "
        "Install dependencies from HighwayEnv/pyproject.toml."
    ) from exc


def highway_env_error_message() -> str:
    py_version = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    return (
        "adversaray requires the real highway-env package. "
        f"Expected local package path: {HIGHWAY_PACKAGE}. "
        f"Current Python: {py_version}."
    )


def require_highway_env() -> None:
    if not HIGHWAY_PACKAGE.is_dir():
        raise RuntimeError(highway_env_error_message())


class ScriptedTraceVehicle(Vehicle):
    """A highway-env vehicle replayed from a precomputed future trace."""

    def __init__(
        self,
        *args: Any,
        trace_x: np.ndarray,
        trace_y: np.ndarray,
        trace_yaw: np.ndarray,
        trace_speed: np.ndarray,
        trace_accel: np.ndarray,
        trace_steering: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trace_x = np.asarray(trace_x, dtype=np.float64)
        self.trace_y = np.asarray(trace_y, dtype=np.float64)
        self.trace_yaw = np.asarray(trace_yaw, dtype=np.float64)
        self.trace_speed = np.asarray(trace_speed, dtype=np.float64)
        self.trace_accel = np.asarray(trace_accel, dtype=np.float64)
        self.trace_steering = np.asarray(trace_steering, dtype=np.float64)
        self.trace_index = 0

    def act(self, action: dict | str = None) -> None:
        if self.trace_index >= len(self.trace_accel):
            return
        Vehicle.act(
            self,
            {
                "steering": float(self.trace_steering[self.trace_index]),
                "acceleration": float(self.trace_accel[self.trace_index]),
            },
        )

    def step(self, dt: float) -> None:
        if self.trace_index >= len(self.trace_x):
            return
        self.position = np.asarray(
            [
                self.trace_x[self.trace_index],
                self.trace_y[self.trace_index],
            ],
            dtype=np.float64,
        )
        self.heading = float(self.trace_yaw[self.trace_index])
        self.speed = float(max(self.trace_speed[self.trace_index], 0.0))
        self.action = {
            "steering": float(self.trace_steering[self.trace_index]),
            "acceleration": float(self.trace_accel[self.trace_index]),
        }
        self.trace_index += 1
        self.on_state_update()


def _dt(schema: dict, config: dict[str, Any]) -> float:
    return float(schema.get("dt", config.get("sampling", {}).get("dt", 0.04)))


def _speed_and_yaw(state: np.ndarray) -> tuple[float, float]:
    speed = float(np.hypot(float(state[2]), float(state[3])))
    if speed > 1e-4:
        return speed, float(np.arctan2(float(state[3]), float(state[2])))
    return max(float(state[2]), 0.0), 0.0


def _make_road(config: dict[str, Any]) -> Any:
    require_highway_env()
    env_cfg = config.get("env", {})
    lanes_count = int(env_cfg.get("lanes_count", 1))
    speed_limit = float(env_cfg.get("speed_limit", 40.0))
    seed = int(config.get("training", {}).get("seed", 42))
    return Road(
        network=RoadNetwork.straight_road_network(
            lanes_count,
            speed_limit=speed_limit,
        ),
        np_random=np.random.RandomState(seed),
        record_history=False,
    )


def _configure_idm(vehicle: Any, config: dict[str, Any]) -> None:
    env_cfg = config.get("env", {})
    idm_cfg = config.get("idm", {})
    vehicle.target_speed = float(
        idm_cfg.get("desired_speed", env_cfg.get("ego_target_speed", 30.0))
    )
    if "max_accel" in idm_cfg:
        vehicle.COMFORT_ACC_MAX = float(idm_cfg["max_accel"])
    if "comfortable_brake" in idm_cfg:
        vehicle.COMFORT_ACC_MIN = -float(idm_cfg["comfortable_brake"])
    if "min_gap" in idm_cfg:
        vehicle.DISTANCE_WANTED = float(idm_cfg["min_gap"])
    if "desired_headway" in idm_cfg:
        vehicle.TIME_WANTED = float(idm_cfg["desired_headway"])
    if "delta" in idm_cfg:
        vehicle.DELTA = float(idm_cfg["delta"])


def _set_vehicle_size(vehicle: Any, length: float) -> None:
    vehicle.LENGTH = float(length)
    if hasattr(vehicle, "diagonal"):
        vehicle.diagonal = float(np.sqrt(vehicle.LENGTH**2 + vehicle.WIDTH**2))


def rollout_highway_idm_ego_trace(
    *,
    context_states: torch.Tensor,
    adversary_x: torch.Tensor,
    adversary_y: torch.Tensor,
    adversary_yaw: torch.Tensor,
    adversary_speed: torch.Tensor,
    adversary_accel: torch.Tensor,
    adversary_steering: torch.Tensor,
    ego_length: torch.Tensor,
    adv_length: torch.Tensor,
    schema: dict,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Replay an adversary trace and return detached IDM ego trace."""
    require_highway_env()
    context_np = context_states.detach().cpu().numpy()
    adv_x_np = adversary_x.detach().cpu().numpy()
    adv_y_np = adversary_y.detach().cpu().numpy()
    adv_yaw_np = adversary_yaw.detach().cpu().numpy()
    adv_speed_np = adversary_speed.detach().cpu().numpy()
    adv_accel_np = adversary_accel.detach().cpu().numpy()
    adv_steer_np = adversary_steering.detach().cpu().numpy()
    ego_len_np = ego_length.detach().cpu().numpy()
    adv_len_np = adv_length.detach().cpu().numpy()
    dt = _dt(schema, config)
    enable_lane_change = bool(
        config.get("ego_response", {}).get("enable_lane_change", False)
    )
    batch_size, horizon = adv_x_np.shape
    ego_x = np.zeros((batch_size, horizon), dtype=np.float32)
    ego_y = np.zeros((batch_size, horizon), dtype=np.float32)
    ego_speed = np.zeros((batch_size, horizon), dtype=np.float32)
    ego_accel = np.zeros((batch_size, horizon), dtype=np.float32)
    ego_yaw = np.zeros((batch_size, horizon), dtype=np.float32)
    ego_action_accel = np.zeros((batch_size, horizon), dtype=np.float32)
    ego_action_steering = np.zeros((batch_size, horizon), dtype=np.float32)

    for batch_idx in range(batch_size):
        road = _make_road(config)
        ego0 = context_np[batch_idx, -1, 0]
        adv0 = context_np[batch_idx, -1, 1]
        ego0_speed, ego0_yaw = _speed_and_yaw(ego0)
        adv0_speed, adv0_yaw = _speed_and_yaw(adv0)
        ego = IDMVehicle(
            road,
            position=np.asarray([ego0[0], ego0[1]], dtype=np.float64),
            heading=ego0_yaw,
            speed=ego0_speed,
            target_speed=float(
                config.get("env", {}).get("ego_target_speed", 30.0)
            ),
            enable_lane_change=enable_lane_change,
        )
        lead = ScriptedTraceVehicle(
            road,
            position=np.asarray([adv0[0], adv0[1]], dtype=np.float64),
            heading=adv0_yaw,
            speed=adv0_speed,
            trace_x=adv_x_np[batch_idx],
            trace_y=adv_y_np[batch_idx],
            trace_yaw=adv_yaw_np[batch_idx],
            trace_speed=adv_speed_np[batch_idx],
            trace_accel=adv_accel_np[batch_idx],
            trace_steering=adv_steer_np[batch_idx],
        )
        _configure_idm(ego, config)
        _set_vehicle_size(ego, float(ego_len_np[batch_idx]))
        _set_vehicle_size(lead, float(adv_len_np[batch_idx]))
        road.vehicles = [ego, lead]

        for step in range(horizon):
            road.act()
            road.step(dt)
            action = ego.action if isinstance(ego.action, dict) else {}
            ego_x[batch_idx, step] = float(ego.position[0])
            ego_y[batch_idx, step] = float(ego.position[1])
            ego_speed[batch_idx, step] = float(max(ego.speed, 0.0))
            ego_accel[batch_idx, step] = float(action.get("acceleration", 0.0))
            ego_yaw[batch_idx, step] = float(ego.heading)
            ego_action_accel[batch_idx, step] = float(
                action.get("acceleration", 0.0)
            )
            ego_action_steering[batch_idx, step] = float(
                action.get("steering", 0.0)
            )

    return {
        "ego_x": torch.from_numpy(ego_x).detach(),
        "ego_y": torch.from_numpy(ego_y).detach(),
        "ego_speed": torch.from_numpy(ego_speed).detach(),
        "ego_accel": torch.from_numpy(ego_accel).detach(),
        "ego_yaw": torch.from_numpy(ego_yaw).detach(),
        "ego_action_accel": torch.from_numpy(ego_action_accel).detach(),
        "ego_action_steering": torch.from_numpy(ego_action_steering).detach(),
    }
