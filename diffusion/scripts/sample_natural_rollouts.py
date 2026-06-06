#!/usr/bin/env python3
"""Sample natural lead-car rollouts from a trained diffusion prior."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion.src.data import split_indices
from diffusion.src.kinematics import (
    integrate_cutin_acceleration_actions,
    integrate_following_actions,
)
from diffusion.src.model import build_model_from_schema
from diffusion.src.types import VehicleBox, VehicleState
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging
from utils.io import load_npz


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_following.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/best_noise_mse.pt"
SCRIPT_DEFAULTS = {
    "split": "val",
    "num_samples": 64,
    "log_level": "INFO",
}


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    paths = config.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    return (config_dir / paths["output_dir"]).resolve()


def _resolve_checkpoint_path(checkpoint: str | None, output_dir: Path) -> Path:
    path = Path(checkpoint or DEFAULT_CHECKPOINT_PATH)
    if path.is_absolute():
        return path
    return (output_dir / path).resolve()


def _decode_actions(x: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["actions"]["mean"], dtype=np.float32)
    std = np.asarray(stats["actions"]["std"], dtype=np.float32)
    return (x * std + mean).astype(np.float32)


def _actions_to_ax(
    actions: np.ndarray,
    initial_states: np.ndarray,
    schema: dict,
    config: dict,
) -> np.ndarray:
    rep = str(schema["action_representation"]).lower()
    dt = float(schema["dt"])
    ax_min = float(config["action"]["ax_min"])
    ax_max = float(config["action"]["ax_max"])
    if rep == "jerk":
        prev_ax = initial_states[:, 1, 4].astype(np.float32)
        ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt
    else:
        ax = actions[:, :, 0]
    return np.clip(ax, ax_min, ax_max).astype(np.float32)


def _integrate(ax: np.ndarray, initial_states: np.ndarray, adv_length: np.ndarray, schema: dict) -> np.ndarray:
    dt = float(schema["dt"])
    trajectories: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        lead0 = initial_states[i, 1]
        initial = VehicleState(
            x=float(lead0[0]),
            y=float(lead0[1]),
            vx=float(lead0[2]),
            vy=float(lead0[3]),
            ax=float(lead0[4]),
            ay=float(lead0[5]),
            box=VehicleBox(length=float(adv_length[i])),
        )
        trajectories.append(integrate_following_actions(initial, ax[i, :, None], dt)[1:])
    return np.stack(trajectories, axis=0)


def _cutin_trajectory_from_actions(
    actions: np.ndarray,
    initial_states: np.ndarray,
    schema: dict,
    config: dict,
) -> np.ndarray:
    rep = str(schema.get("generation_target", "")).lower()
    action_rep = str(schema.get("action_representation", "")).lower()
    action_cfg = config.get("action", {})
    projection_cfg = config.get("trajectory_projection", {})
    if rep == "action" and action_rep in {"ax_ay", "acceleration"}:
        return integrate_cutin_acceleration_actions(
            initial_states,
            actions,
            float(schema["dt"]),
            ax_min=float(action_cfg.get("ax_min", -8.0)),
            ax_max=float(action_cfg.get("ax_max", 4.0)),
            ay_abs_max=float(action_cfg.get("ay_abs_max", 4.0)),
            speed_min=float(projection_cfg.get("speed_min", 0.0)),
            speed_max=float(projection_cfg.get("speed_max", 50.0)),
        )
    raise ValueError(f"Unsupported cut-in generation_target: {rep}")


def _sample_actions(
    model,
    batch_size: int,
    scenario_conditions: torch.Tensor,
) -> np.ndarray:
    sample = model.sample_ddim(
        batch_size,
        scenario_conditions,
    )
    return sample.detach().cpu().numpy()


def sample_rollouts(config: dict, config_dir: Path, checkpoint: str | None, split: str, num_samples: int) -> Path:
    output_dir = _resolve_output_dir(config, config_dir)
    schema = load_json(output_dir / "feature_schema.json")
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = load_npz(output_dir / "dataset_normalized.npz")
    raw = load_npz(output_dir / "dataset.npz")
    set_seed(int(config["evaluation"]["seed"]))
    device = select_device(config["training"]["device"])
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(_resolve_checkpoint_path(checkpoint, output_dir), map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    idx = split_indices(arrays, split)[: int(num_samples)]
    if len(idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    scenario_conditions = torch.from_numpy(arrays["scenario_conditions"][idx]).float().to(device)
    with torch.no_grad():
        normalized_actions = _sample_actions(
            model,
            len(idx),
            scenario_conditions,
        )
    actions = _decode_actions(normalized_actions, stats)
    if str(schema.get("event_type", "")).lower() == "cut_in":
        trajectories = _cutin_trajectory_from_actions(
            actions,
            raw["initial_states"][idx],
            schema,
            config,
        )
        ax = trajectories[:, :, 4]
    else:
        ax = _actions_to_ax(actions, raw["initial_states"][idx], schema, config)
        trajectories = _integrate(ax, raw["initial_states"][idx], raw["adv_length"][idx], schema)

    out_path = output_dir / "natural_rollouts.npz"
    np.savez_compressed(
        out_path,
        sample_index=idx.astype(np.int64),
        actions=actions.astype(np.float32),
        acceleration=ax.astype(np.float32),
        lead_trajectory=trajectories.astype(np.float32),
    )
    save_json(
        {
            "path": str(out_path),
            "num_samples": int(len(idx)),
            "split": split,
            "sampler": "ddim",
            "action_representation": schema["action_representation"],
        },
        output_dir / "natural_rollouts_summary.json",
    )
    return out_path


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", default=str(SCRIPT_DEFAULTS["split"]))
    parser.add_argument("--num-samples", type=int, default=int(SCRIPT_DEFAULTS["num_samples"]))
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    sample_rollouts(
        load_yaml(cfg_path),
        cfg_path.parent,
        args.checkpoint,
        str(args.split).lower(),
        int(args.num_samples),
    )


if __name__ == "__main__":
    main()
