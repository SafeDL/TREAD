#!/usr/bin/env python3
"""Offline sampling evaluation for the car-following action diffusion model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import SPLIT_TO_INDEX, load_normalized_dataset
from diffusion.src.kinematics import integrate_following_actions
from diffusion.src.model import build_model_from_schema
from diffusion.src.risk import constant_velocity_rollout, score_future_risk
from diffusion.src.types import VehicleBox, VehicleState
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diffusion_following.yaml"


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    return (config_dir / config.get("paths", {}).get("output_dir", "../../../data/diffusion/following")).resolve()


def _decode_actions(x: np.ndarray, stats: dict) -> np.ndarray:
    norm = stats["actions"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return (x * std + mean).astype(np.float32)


def _encode_risk_condition(raw_values: np.ndarray, raw_train: np.ndarray, config: dict, stats: dict) -> np.ndarray:
    transform = str(config.get("risk_condition", {}).get("transform", "log1p_and_percentile")).lower()
    raw = np.asarray(raw_values, dtype=np.float32)
    log = np.log1p(np.maximum(raw, 0.0)).astype(np.float32)
    ref = np.sort(np.asarray(raw_train, dtype=np.float32))
    pct = (np.searchsorted(ref, raw, side="right") / max(len(ref), 1)).astype(np.float32)
    if transform == "raw":
        cond = raw.reshape(-1, 1)
    elif transform == "log1p":
        cond = log.reshape(-1, 1)
    elif transform == "percentile":
        cond = pct.reshape(-1, 1)
    else:
        cond = np.stack([log, np.clip(pct, 0.0, 1.0)], axis=-1)
    if "risk_condition" in stats:
        mean = np.asarray(stats["risk_condition"]["mean"], dtype=np.float32)
        std = np.asarray(stats["risk_condition"]["std"], dtype=np.float32)
        cond = (cond - mean) / std
    return cond.astype(np.float32)


def _actions_to_ax(actions: np.ndarray, context_states: np.ndarray, schema: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    action_cfg = config.get("action", {})
    rep = str(schema.get("action_representation", action_cfg.get("representation", "acceleration"))).lower()
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    dt = float(schema.get("dt", 0.04))
    if rep == "jerk":
        prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
        ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt
    else:
        ax = actions[:, :, 0]
    unclipped = ax.astype(np.float32)
    return np.clip(unclipped, ax_min, ax_max).astype(np.float32), unclipped


def _jerk_from_ax(ax: np.ndarray, context_states: np.ndarray, dt: float) -> np.ndarray:
    prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
    return np.diff(np.concatenate([prev_ax[:, None], ax], axis=1), axis=1) / max(float(dt), 1e-6)


def _summarize_actions(real_ax: np.ndarray, gen_ax: np.ndarray, gen_unclipped: np.ndarray, context_states: np.ndarray, dt: float, cfg: dict) -> dict[str, Any]:
    ax_min = float(cfg.get("action", {}).get("ax_min", -8.0))
    ax_max = float(cfg.get("action", {}).get("ax_max", 4.0))
    real_jerk = _jerk_from_ax(real_ax, context_states, dt)
    gen_jerk = _jerk_from_ax(gen_ax, context_states, dt)
    clip_rate = float(np.mean((gen_unclipped < ax_min) | (gen_unclipped > ax_max)))
    return {
        "real_ax_mean": float(np.mean(real_ax)),
        "gen_ax_mean": float(np.mean(gen_ax)),
        "real_ax_std": float(np.std(real_ax)),
        "gen_ax_std": float(np.std(gen_ax)),
        "real_jerk_std": float(np.std(real_jerk)),
        "gen_jerk_std": float(np.std(gen_jerk)),
        "real_hard_brake_ratio": float(np.mean(real_ax < -3.0)),
        "gen_hard_brake_ratio": float(np.mean(gen_ax < -3.0)),
        "gen_action_clip_rate": clip_rate,
    }


def _trajectory_risks(ax: np.ndarray, context_states: np.ndarray, meta: dict[str, np.ndarray], schema: dict, config: dict) -> dict[str, float]:
    dt = float(schema.get("dt", 0.04))
    risks: list[float] = []
    min_gaps: list[float] = []
    final_gaps: list[float] = []
    for i in range(ax.shape[0]):
        lead0 = context_states[i, -1, 1]
        ego0 = context_states[i, -1, 0]
        adv_len = float(meta["adv_length"][i])
        ego_len = float(meta["ego_length"][i])
        lane_w = float(meta["lane_width"][i])
        lead_state = VehicleState(
            x=float(lead0[0]),
            y=float(lead0[1]),
            vx=float(lead0[2]),
            vy=float(lead0[3]),
            ax=float(lead0[4]),
            ay=float(lead0[5]),
            box=VehicleBox(length=adv_len),
        )
        adv = integrate_following_actions(lead_state, ax[i, :, None], dt)[1:]
        ego = constant_velocity_rollout(ego0, ax.shape[1], dt)[1:]
        gaps = adv[:, 0] - ego[:, 0] - 0.5 * (ego_len + adv_len)
        risks.append(score_future_risk("following", ego, adv, ego_len, adv_len, lane_w, config.get("risk", {})))
        min_gaps.append(float(np.min(gaps)))
        final_gaps.append(float(gaps[-1]))
    return {
        "generated_risk_mean": float(np.mean(risks)),
        "generated_risk_std": float(np.std(risks)),
        "generated_min_gap_mean": float(np.mean(min_gaps)),
        "generated_final_gap_mean": float(np.mean(final_gaps)),
    }


def _sample_actions(model, arrays: dict, idx: np.ndarray, risk_condition: np.ndarray, device: torch.device, guidance_scale: float) -> np.ndarray:
    history = torch.from_numpy(arrays["context_states"][idx]).float().to(device)
    context = torch.from_numpy(arrays["context_features"][idx]).float().to(device)
    relative = torch.from_numpy(arrays["relative_history"][idx]).float().to(device)
    risk = torch.from_numpy(risk_condition).float().to(device)
    sample = model.sample(len(idx), history, context, relative, risk, guidance_scale=guidance_scale)
    return sample.detach().cpu().numpy()


def evaluate(config: dict, config_dir: Path) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    schema = load_json(output_dir / "feature_schema.json")
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = load_normalized_dataset(output_dir)
    raw_npz = np.load(output_dir / "dataset.npz", allow_pickle=True)
    raw = {k: raw_npz[k] for k in raw_npz.files}

    checkpoint = output_dir / "checkpoints" / "best.pt"
    device = select_device(config.get("training", {}).get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    eval_cfg = config.get("evaluation", {})
    split = str(eval_cfg.get("split", "val"))
    max_samples = int(eval_cfg.get("max_samples", 64))
    guidance_scale = float(eval_cfg.get("guidance_scale", 1.0))
    mask_idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]
    if len(mask_idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    idx = mask_idx[:max_samples]

    gen_norm = _sample_actions(model, arrays, idx, arrays["risk_condition"][idx], device, guidance_scale)
    gen_actions = _decode_actions(gen_norm, stats)
    real_actions = raw["actions"][idx]
    real_context = raw["context_states"][idx]
    real_ax, _ = _actions_to_ax(real_actions, real_context, schema, config)
    gen_ax, gen_unclipped = _actions_to_ax(gen_actions, real_context, schema, config)
    meta = {k: raw[k][idx] for k in ("ego_length", "adv_length", "lane_width")}
    dt = float(schema.get("dt", 0.04))

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "split": split,
        "num_samples": int(len(idx)),
        "action_distribution": _summarize_actions(real_ax, gen_ax, gen_unclipped, real_context, dt, config),
        "trajectory": _trajectory_risks(gen_ax, real_context, meta, schema, config),
    }

    train_raw = raw["risk_raw"][raw["split_index"] == SPLIT_TO_INDEX["train"]]
    levels = np.asarray(eval_cfg.get("risk_percentile_levels", [0.5, 0.8, 0.9, 0.95]), dtype=np.float32)
    raw_levels = np.quantile(train_raw, levels)
    mono_n = min(int(eval_cfg.get("monotonic_samples", 16)), len(idx))
    mono_idx = idx[:mono_n]
    mono_risks: list[float] = []
    for raw_level in raw_levels:
        cond = _encode_risk_condition(np.full((mono_n,), raw_level, dtype=np.float32), train_raw, config, stats)
        sample = _decode_actions(_sample_actions(model, arrays, mono_idx, cond, device, guidance_scale), stats)
        ax, _ = _actions_to_ax(sample, raw["context_states"][mono_idx], schema, config)
        mono_meta = {k: raw[k][mono_idx] for k in ("ego_length", "adv_length", "lane_width")}
        mono_risks.append(_trajectory_risks(ax, raw["context_states"][mono_idx], mono_meta, schema, config)["generated_risk_mean"])
    summary["risk_control"] = {
        "percentile_levels": [float(x) for x in levels],
        "raw_risk_levels": [float(x) for x in raw_levels],
        "generated_risk_means": mono_risks,
        "is_monotonic": bool(np.all(np.diff(mono_risks) >= -1e-6)),
    }

    zero_cond = np.zeros_like(arrays["risk_condition"][idx])
    shuffled_cond = arrays["risk_condition"][idx].copy()
    np.random.default_rng(int(eval_cfg.get("seed", 42))).shuffle(shuffled_cond)
    ablation = {}
    for name, cond in {"zero": zero_cond, "shuffle": shuffled_cond}.items():
        sample = _decode_actions(_sample_actions(model, arrays, idx, cond, device, guidance_scale), stats)
        ax, _ = _actions_to_ax(sample, real_context, schema, config)
        ablation[name] = _trajectory_risks(ax, real_context, meta, schema, config)
    summary["risk_condition_ablation"] = ablation

    save_json(summary, output_dir / "evaluation_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH
    evaluate(load_yaml(cfg_path), cfg_path.parent)


if __name__ == "__main__":
    main()
