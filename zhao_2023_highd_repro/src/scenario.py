from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import ensure_repo_imports, finite_summary, output_dir, resolve_path, save_json, set_seed


ensure_repo_imports()

from process_highD.src.idm_ego import load_idm_ego_config
from tools.risk import longitudinal_series_from_arrays


IDM_PARAMETER_KEYS = (
    "ACC_MAX",
    "COMFORT_ACC_MAX",
    "COMFORT_ACC_MIN",
    "DISTANCE_WANTED",
    "TIME_WANTED",
    "DELTA",
    "POLITENESS",
    "LANE_CHANGE_MIN_ACC_GAIN",
    "LANE_CHANGE_MAX_BRAKING_IMPOSED",
    "LANE_CHANGE_DELAY",
)


def _paper_initial_state(
    target_local: np.ndarray,
    duration: float,
    lane_width: float,
    cfg: dict,
) -> tuple[float, float, float]:
    t2 = float(cfg["scenario"].get("t2_seconds", 0.2))
    amax = float(cfg["scenario"].get("max_brake_accel", 6.0))
    length = float(cfg["scenario"].get("vehicle_length", 4.0))
    t3 = max(float(duration) - t2, 0.0)
    target_speed = float(np.mean(target_local[:, 2]))
    dmin = (
        0.5 * (amax * t3 + 0.5 * amax * t2) * t2
        + 0.5 * amax * t3 * t3
        + length
    )
    ego_speed = target_speed + amax * t3 + 0.5 * amax * t2
    lateral_safe = 0.000066 * (ego_speed * ego_speed - target_speed * target_speed) + 1.49
    lateral_safe = float(np.clip(lateral_safe, 0.5, max(float(lane_width), 1.0)))
    return float(dmin), float(ego_speed), lateral_safe


def _load_highway_env_classes():
    import sys

    root = Path(__file__).resolve().parents[2]
    highway_root = root / "HighwayEnv"
    highway_package = highway_root / "highway_env"
    if not highway_package.is_dir():
        raise FileNotFoundError(f"Required local highway-env package not found: {highway_package}")
    if str(highway_root) not in sys.path:
        sys.path.insert(0, str(highway_root))
    from highway_env.road.road import Road, RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.kinematics import Vehicle

    return Road, RoadNetwork, IDMVehicle, Vehicle


def _vehicle_state(vehicle: Any) -> np.ndarray:
    acceleration = 0.0
    if isinstance(vehicle.action, dict):
        acceleration = float(vehicle.action.get("acceleration", 0.0))
    vx = float(vehicle.speed) * float(np.cos(vehicle.heading))
    vy = float(vehicle.speed) * float(np.sin(vehicle.heading))
    ax = acceleration * float(np.cos(vehicle.heading))
    ay = acceleration * float(np.sin(vehicle.heading))
    return np.asarray(
        [vehicle.position[0], vehicle.position[1], vx, vy, ax, ay],
        dtype=np.float32,
    )


def _speed_and_heading(state: np.ndarray, *, keep_lateral_heading: bool) -> tuple[float, float]:
    vx = float(state[2])
    vy = float(state[3])
    speed = float(np.hypot(vx, vy))
    if keep_lateral_heading and speed > 1.0e-6:
        return speed, float(np.arctan2(vy, max(vx, 1.0e-6)))
    return speed, 0.0


def _rollout_fixed_lane_idm_ego_trajectory(
    initial_states: np.ndarray,
    adversary_trajectory: np.ndarray,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    *,
    dt: float,
    config: dict[str, Any],
) -> np.ndarray:
    """Roll out highway-env IDM with the ego vehicle locked to its initial lane."""
    Road, RoadNetwork, IDMVehicle, Vehicle = _load_highway_env_classes()

    class ScriptedAdversaryVehicle(Vehicle):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.forced_position: np.ndarray | None = None
            self.forced_heading: float | None = None
            self.forced_speed: float | None = None

        def set_forced_state(self, position: np.ndarray, heading: float, speed: float) -> None:
            self.forced_position = np.asarray(position, dtype=np.float64)
            self.forced_heading = float(heading)
            self.forced_speed = float(speed)

        def act(self, action: dict | str = None) -> None:
            del action
            Vehicle.act(self, {"steering": 0.0, "acceleration": 0.0})

        def step(self, dt: float) -> None:
            del dt
            if self.forced_position is None:
                return
            self.position = self.forced_position
            self.heading = float(self.forced_heading)
            self.speed = float(self.forced_speed)
            self.forced_position = None
            self.forced_heading = None
            self.forced_speed = None
            self.on_state_update()

    class FixedLaneIDMVehicle(IDMVehicle):
        def __init__(self, *args: Any, front_vehicle: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, enable_lane_change=False, **kwargs)
            self.scripted_front_vehicle = front_vehicle

        def act(self, action: dict | str = None) -> None:
            del action
            if self.crashed:
                return
            self.enable_lane_change = False
            self.target_lane_index = self.lane_index
            acceleration = self.acceleration(
                ego_vehicle=self,
                front_vehicle=self.scripted_front_vehicle,
                rear_vehicle=None,
            )
            acceleration = float(np.clip(acceleration, -self.ACC_MAX, self.ACC_MAX))
            Vehicle.act(self, {"steering": 0.0, "acceleration": acceleration})

    cfg = dict(config)
    cfg["enable_lane_change"] = False
    init = np.asarray(initial_states, dtype=np.float32)
    adv = np.asarray(adversary_trajectory, dtype=np.float32)
    if init.ndim != 3 or init.shape[1:] != (2, 6):
        raise ValueError(f"initial_states must have shape [B, 2, 6], got {init.shape}")
    if adv.ndim != 3 or adv.shape[0] != init.shape[0] or adv.shape[2] != 6:
        raise ValueError(f"adversary_trajectory shape mismatch: {adv.shape}")

    batch, horizon = int(adv.shape[0]), int(adv.shape[1])
    out = np.zeros((batch, horizon, 6), dtype=np.float32)
    lanes_count = int(cfg.get("lanes_count", 3))
    speed_limit = float(cfg.get("speed_limit", 50.0))
    keep_lateral_heading = bool(cfg.get("keep_lateral_heading", True))
    target_speed_cfg = cfg.get("target_speed", "initial")
    seed = int(cfg.get("seed", 42))
    idm_params = {key: float(cfg[key]) for key in IDM_PARAMETER_KEYS if key in cfg}

    for idx in range(batch):
        road = Road(
            network=RoadNetwork.straight_road_network(lanes_count, speed_limit=speed_limit),
            np_random=np.random.RandomState(seed + idx),
            record_history=False,
        )
        ego0 = init[idx, 0]
        adv0 = init[idx, 1]
        ego_speed, ego_heading = _speed_and_heading(ego0, keep_lateral_heading=False)
        adv_speed, adv_heading = _speed_and_heading(adv0, keep_lateral_heading=keep_lateral_heading)
        if target_speed_cfg is None or str(target_speed_cfg).lower() in {"initial", "context"}:
            target_speed = ego_speed
        else:
            target_speed = float(target_speed_cfg)

        adversary = ScriptedAdversaryVehicle(
            road,
            position=np.asarray([adv0[0], adv0[1]], dtype=np.float64),
            heading=adv_heading,
            speed=adv_speed,
        )
        ego = FixedLaneIDMVehicle(
            road,
            position=np.asarray([ego0[0], ego0[1]], dtype=np.float64),
            heading=ego_heading,
            speed=ego_speed,
            target_speed=target_speed,
            front_vehicle=adversary,
        )
        for key, value in idm_params.items():
            setattr(ego, key, value)
        ego.LENGTH = float(np.asarray(ego_length, dtype=np.float64)[idx])
        adversary.LENGTH = float(np.asarray(adv_length, dtype=np.float64)[idx])
        if hasattr(ego, "diagonal"):
            ego.diagonal = float(np.sqrt(ego.LENGTH**2 + ego.WIDTH**2))
        if hasattr(adversary, "diagonal"):
            adversary.diagonal = float(np.sqrt(adversary.LENGTH**2 + adversary.WIDTH**2))
        road.vehicles.extend([ego, adversary])

        for step in range(horizon):
            adv_state = adv[idx, step]
            adv_speed_next, adv_heading_next = _speed_and_heading(
                adv_state,
                keep_lateral_heading=keep_lateral_heading,
            )
            adversary.set_forced_state(
                np.asarray([adv_state[0], adv_state[1]], dtype=np.float64),
                adv_heading_next,
                adv_speed_next,
            )
            road.act()
            road.step(float(dt))
            state = _vehicle_state(ego)
            state[1] = float(ego0[1])
            state[3] = 0.0
            state[5] = 0.0
            out[idx, step] = state
    return out


def _build_single_scenario(
    target_local: np.ndarray,
    condition: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    duration = float(np.clip(condition[0], 0.4, cfg["data"]["max_lane_change_seconds"]))
    dt = duration / max(target_local.shape[0] - 1, 1)
    lane_width = abs(float(condition[5]))
    if lane_width < 1.0:
        lane_width = float(cfg["scenario"].get("lane_width_default", 3.75))
    dmin, ego_speed, lateral_safe = _paper_initial_state(
        target_local,
        duration,
        lane_width,
        cfg,
    )
    sign = 1.0 if float(target_local[-1, 1]) >= 0.0 else -1.0
    progress = np.clip(np.abs(target_local[:, 1]) / max(lane_width, 1.0e-6), 0.0, 1.0)
    traj = target_local.copy()
    traj[:, 0] += dmin
    traj[:, 1] = sign * (lane_width - progress * lane_width)
    traj[:, 3] = np.gradient(traj[:, 1], dt).astype(np.float32)
    traj[:, 5] = np.gradient(traj[:, 3], dt).astype(np.float32)
    ego0 = np.asarray([0.0, 0.0, ego_speed, 0.0, 0.0, 0.0], dtype=np.float32)
    init = np.stack([ego0, traj[0].astype(np.float32)], axis=0)
    return init, traj.astype(np.float32), dt, float(dmin), float(lateral_safe)


def build_dangerous_scenarios(config: dict) -> dict[str, Path]:
    set_seed(int(config["scenario"]["seed"]))
    out = output_dir(config)
    generated_path = out / "generated_trajectories.npz"
    if not generated_path.exists():
        raise FileNotFoundError(f"Generated trajectories not found: {generated_path}")
    data = np.load(generated_path, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    conditions = data["conditions"].astype(np.float32)
    count = min(int(config["scenario"]["sample_count"]), len(trajectories))
    rng = np.random.default_rng(int(config["scenario"]["seed"]))
    idx = rng.choice(len(trajectories), size=count, replace=False if count <= len(trajectories) else True)
    initial_states: list[np.ndarray] = []
    target_trajs: list[np.ndarray] = []
    dts: list[float] = []
    gaps: list[float] = []
    ego_lengths: list[float] = []
    target_lengths: list[float] = []
    lateral_safe_distances: list[float] = []
    for item in idx:
        init, traj, dt, gap, lateral_safe = _build_single_scenario(
            trajectories[int(item)],
            conditions[int(item)],
            config,
        )
        initial_states.append(init)
        target_trajs.append(traj)
        dts.append(float(dt))
        gaps.append(float(gap))
        lateral_safe_distances.append(float(lateral_safe))
        ego_lengths.append(float(config["scenario"]["vehicle_length"]))
        target_lengths.append(float(config["scenario"]["vehicle_length"]))
    path = out / "dangerous_scenarios.npz"
    np.savez_compressed(
        path,
        initial_states=np.stack(initial_states).astype(np.float32),
        target_trajectories=np.stack(target_trajs).astype(np.float32),
        dt=np.asarray(dts, dtype=np.float32),
        initial_gap=np.asarray(gaps, dtype=np.float32),
        lateral_safe_distance=np.asarray(lateral_safe_distances, dtype=np.float32),
        ego_length=np.asarray(ego_lengths, dtype=np.float32),
        target_length=np.asarray(target_lengths, dtype=np.float32),
    )
    save_json(
        out / "scenario_summary.json",
        {
            "scenario_count": int(count),
            "paper_dmin_m": finite_summary(np.asarray(gaps)),
            "paper_dL_m": finite_summary(np.asarray(lateral_safe_distances)),
            "dt_s": finite_summary(np.asarray(dts)),
        },
    )
    return {"scenarios": path}


def _metrics_for_pair(
    ego: np.ndarray,
    target: np.ndarray,
    ego_length: float,
    target_length: float,
    cfg: dict,
) -> dict[str, float]:
    gap = target[:, 0] - ego[:, 0] - 0.5 * (float(ego_length) + float(target_length))
    series = longitudinal_series_from_arrays(
        gap=gap,
        ego_speed=ego[:, 2],
        lead_speed=target[:, 2],
        ego_accel=ego[:, 4],
    )
    min_gap = float(np.min(series["gap"]))
    lateral_overlap = np.abs(target[:, 1] - ego[:, 1]) < 1.0
    collision = bool(np.any((gap <= 0.0) & lateral_overlap))
    ttc = np.asarray(series["ttc"], dtype=np.float32).copy()
    ttc[(gap <= 0.0) & lateral_overlap] = 0.0
    min_ttc = float(np.min(ttc))
    return {
        "min_ttc_s": min_ttc,
        "min_gap_m": min_gap,
        "collision": float(collision),
        "near_collision": float(np.any(min_gap <= float(cfg["simulation"]["near_gap_threshold"]))),
        "ttc_below_threshold": float(min_ttc < float(cfg["simulation"]["ttc_threshold_seconds"])),
    }


def run_idm_evaluation(config: dict) -> dict[str, Path]:
    out = output_dir(config)
    scenarios_path = out / "dangerous_scenarios.npz"
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenarios not found: {scenarios_path}")
    scenarios = np.load(scenarios_path, allow_pickle=True)
    initial = scenarios["initial_states"].astype(np.float32)
    target = scenarios["target_trajectories"].astype(np.float32)
    ego_len = scenarios["ego_length"].astype(np.float32)
    target_len = scenarios["target_length"].astype(np.float32)
    dts = scenarios["dt"].astype(np.float32)
    max_n = min(int(config["simulation"]["max_scenarios"]), len(initial))
    idm_cfg = load_idm_ego_config(resolve_path(config["paths"]["idm_config"]), event_type="cut_in")
    idm_cfg["enable_lane_change"] = False
    ego_rollouts: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    for idx in range(max_n):
        ego = _rollout_fixed_lane_idm_ego_trajectory(
            initial[idx:idx + 1],
            target[idx:idx + 1],
            ego_len[idx:idx + 1],
            target_len[idx:idx + 1],
            dt=float(dts[idx]),
            config=idm_cfg,
        )[0]
        ego_rollouts.append(ego.astype(np.float32))
        metrics = _metrics_for_pair(ego, target[idx], float(ego_len[idx]), float(target_len[idx]), config)
        metrics["scenario_index"] = float(idx)
        metrics["dt_s"] = float(dts[idx])
        metrics["ego_max_abs_lateral_m"] = float(np.max(np.abs(ego[:, 1] - ego[0, 1])))
        rows.append(metrics)
    metrics_path = out / "idm_metrics.csv"
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)
    rollouts_path = out / "idm_rollouts.npz"
    np.savez_compressed(
        rollouts_path,
        ego_trajectories=np.stack(ego_rollouts).astype(np.float32),
        target_trajectories=target[:max_n].astype(np.float32),
        initial_states=initial[:max_n].astype(np.float32),
        dt=dts[:max_n].astype(np.float32),
        ego_length=ego_len[:max_n].astype(np.float32),
        target_length=target_len[:max_n].astype(np.float32),
    )
    save_json(
        out / "idm_summary.json",
        {
            "evaluated_count": int(max_n),
            "lane_change_enabled": bool(idm_cfg.get("enable_lane_change", False)),
            "ego_max_abs_lateral_m": float(np.max(np.abs(np.stack(ego_rollouts)[:, :, 1] - initial[:max_n, 0:1, 1]))),
            "ttc_lt_1s_rate": float(df["ttc_below_threshold"].mean()) if len(df) else float("nan"),
            "collision_rate": float(df["collision"].mean()) if len(df) else float("nan"),
            "near_collision_rate": float(df["near_collision"].mean()) if len(df) else float("nan"),
            "min_ttc_s": finite_summary(df["min_ttc_s"].to_numpy(dtype=float)),
            "min_gap_m": finite_summary(df["min_gap_m"].to_numpy(dtype=float)),
            "idm_config_path": str(resolve_path(config["paths"]["idm_config"])),
        },
    )
    return {"idm_metrics": metrics_path, "idm_rollouts": rollouts_path}
