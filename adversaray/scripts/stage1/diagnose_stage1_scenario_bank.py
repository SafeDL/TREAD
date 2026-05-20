#!/usr/bin/env python3
"""Diagnose Stage 1 shared proposal scenario bank coverage and quality."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.risk_utils import actions_to_accel_jerk, write_json
from adversaray.src.stage1_shared_utils import risk_type_summary, tensor_stats
from diffusion.src.utils import load_json, load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "prior_guided_following.yaml"
IDM_PARAM_KEYS = (
    "desired_speed",
    "desired_headway",
    "min_gap",
    "max_accel",
    "comfortable_brake",
    "response_time",
    "delta",
)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _stage1_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("stage1_shared", {}))
    cfg.setdefault("output_dir", "../../../data/adversaray/following/stage1_shared")
    cfg.setdefault("scenario_bank", {})
    cfg["scenario_bank"].setdefault("output_name", "scenario_bank.npz")
    return cfg


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=40, color="#3b82f6", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#14b8a6", alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _curve_plot(path: Path, curves: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in curves.items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size:
            ax.plot(np.arange(arr.size), arr, label=label)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if curves:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _multi_curve_plot(path: Path, values: np.ndarray, title: str, ylabel: str, *, max_curves: int = 20) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return
    arr = arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for curve in arr[:max_curves]:
        ax.plot(np.arange(curve.shape[0]), curve, color="#64748b", alpha=0.35)
    ax.plot(np.arange(arr.shape[1]), np.mean(arr, axis=0), color="#ef4444", linewidth=2.0, label="mean")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _lead_speed(actions: np.ndarray, context_states: np.ndarray, schema: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    accel, _jerk = actions_to_accel_jerk(actions, context_states, schema, config)
    dt = float(schema.get("dt", config.get("sampling", {}).get("dt", 0.04)))
    v0 = np.maximum(np.asarray(context_states[:, -1, 1, 2], dtype=np.float32), 0.0)
    return v0[:, None] + np.cumsum(accel, axis=1) * dt


def _candidate_counts(dataset_index: np.ndarray) -> np.ndarray:
    _unique, counts = np.unique(np.asarray(dataset_index, dtype=np.int64), return_counts=True)
    return counts.astype(np.float32)


def diagnose(arrays: dict[str, np.ndarray], schema: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    shared_actions = np.asarray(arrays["shared_actions"], dtype=np.float32)
    prior_actions = np.asarray(arrays["prior_actions"], dtype=np.float32)
    delta = np.asarray(arrays.get("delta_actions", shared_actions - prior_actions), dtype=np.float32)
    context = np.asarray(arrays["context_states"], dtype=np.float32)
    accel, jerk = actions_to_accel_jerk(shared_actions, context, schema, config)
    speed = _lead_speed(shared_actions, context, schema, config)
    action_l2 = np.sqrt(np.mean(np.square(delta), axis=tuple(range(1, delta.ndim))))

    out: dict[str, Any] = {
        "num_scenarios": int(shared_actions.shape[0]),
        "num_source_contexts": int(np.unique(arrays["dataset_index"]).size) if "dataset_index" in arrays else 0,
        **tensor_stats(arrays.get("proxy_risk_delta", np.asarray([])), "proxy_risk_delta"),
        **tensor_stats(action_l2, "action_l2"),
        **tensor_stats(jerk, "jerk"),
        **tensor_stats(accel, "acceleration"),
        **tensor_stats(speed, "speed"),
        **tensor_stats(arrays.get("delta_l2", np.asarray([])), "delta_l2"),
        **tensor_stats(arrays.get("delta_smoothness", np.asarray([])), "delta_smoothness"),
        **tensor_stats(arrays.get("delta_abs_max", np.asarray([])), "delta_abs_max"),
        **tensor_stats(arrays.get("naturalness_penalty", np.asarray([])), "naturalness_penalty"),
        **tensor_stats(arrays.get("physics_penalty", np.asarray([])), "physics_penalty"),
        **tensor_stats(_candidate_counts(arrays["dataset_index"]), "retained_candidates_per_context"),
    }
    if "risk_type_id" in arrays:
        out.update(risk_type_summary(arrays["risk_type_id"]))
    if "ego_surrogate_params" in arrays:
        params = np.asarray(arrays["ego_surrogate_params"], dtype=np.float32)
        out["ego_surrogate_param_coverage"] = {
            key: tensor_stats(params[:, idx], key)
            for idx, key in enumerate(IDM_PARAM_KEYS[: params.shape[1]])
        }
    if "selection_reason" in arrays:
        reasons, counts = np.unique(arrays["selection_reason"].astype(str), return_counts=True)
        out["selection_reason_count"] = {str(reason): int(count) for reason, count in zip(reasons, counts)}
    return out


def write_figures(arrays: dict[str, np.ndarray], schema: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> None:
    shared_actions = np.asarray(arrays["shared_actions"], dtype=np.float32)
    prior_actions = np.asarray(arrays["prior_actions"], dtype=np.float32)
    delta = np.asarray(arrays.get("delta_actions", shared_actions - prior_actions), dtype=np.float32)
    context = np.asarray(arrays["context_states"], dtype=np.float32)
    accel, jerk = actions_to_accel_jerk(shared_actions, context, schema, config)
    speed = _lead_speed(shared_actions, context, schema, config)
    action_l2 = np.sqrt(np.mean(np.square(delta), axis=tuple(range(1, delta.ndim))))

    _hist(figure_dir / "proxy_risk_delta.png", arrays.get("proxy_risk_delta", np.asarray([])), "Proxy Risk Delta", "risk delta")
    _hist(figure_dir / "action_l2.png", action_l2, "Action Delta L2", "L2")
    _hist(figure_dir / "jerk.png", jerk, "Shared Action Jerk", "jerk")
    _hist(figure_dir / "acceleration.png", accel, "Lead Acceleration", "m/s^2")
    _hist(figure_dir / "speed.png", speed, "Lead Speed", "m/s")
    _hist(figure_dir / "delta_l2.png", arrays.get("delta_l2", np.asarray([])), "Delta L2", "delta_l2")
    _hist(figure_dir / "delta_smoothness.png", arrays.get("delta_smoothness", np.asarray([])), "Delta Smoothness", "delta_smoothness")
    _hist(figure_dir / "delta_abs_max.png", arrays.get("delta_abs_max", np.asarray([])), "Delta Abs Max", "delta_abs_max")
    _hist(figure_dir / "naturalness_penalty.png", arrays.get("naturalness_penalty", np.asarray([])), "Naturalness Penalty", "penalty")
    _hist(figure_dir / "physics_penalty.png", arrays.get("physics_penalty", np.asarray([])), "Physics Penalty", "penalty")
    _hist(
        figure_dir / "retained_candidates_per_context.png",
        _candidate_counts(arrays["dataset_index"]),
        "Retained Candidates Per Source Context",
        "count",
    )
    if "risk_type_id" in arrays:
        summary = risk_type_summary(arrays["risk_type_id"])
        _bar(
            figure_dir / "risk_type_count.png",
            list(summary["risk_type_count"].keys()),
            [float(v) for v in summary["risk_type_count"].values()],
            "Risk Type Count",
            "count",
        )
    _curve_plot(
        figure_dir / "delta_actions_mean_curve.png",
        {"delta_mean": np.mean(delta.reshape(delta.shape[0], delta.shape[1], -1)[..., 0], axis=0)},
        "Delta Actions Mean Curve",
        "delta action",
    )
    if "proxy_risk_delta" in arrays:
        order = np.argsort(-np.asarray(arrays["proxy_risk_delta"], dtype=np.float64))
        _multi_curve_plot(
            figure_dir / "top_risk_delta_actions_curves.png",
            delta[order[: min(20, len(order))]],
            "Top-Risk Delta Action Curves",
            "delta action",
        )
    _curve_plot(
        figure_dir / "prior_vs_shared_actions_mean.png",
        {
            "prior": np.mean(prior_actions.reshape(prior_actions.shape[0], prior_actions.shape[1], -1)[..., 0], axis=0),
            "shared": np.mean(shared_actions.reshape(shared_actions.shape[0], shared_actions.shape[1], -1)[..., 0], axis=0),
        },
        "Prior vs Shared Actions Mean",
        "action",
    )
    if "ego_surrogate_params" in arrays:
        params = np.asarray(arrays["ego_surrogate_params"], dtype=np.float32)
        for idx, key in enumerate(IDM_PARAM_KEYS[: params.shape[1]]):
            _hist(figure_dir / f"ego_surrogate_{key}.png", params[:, idx], f"Ego Surrogate {key}", key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--scenario-bank", default="", help="Override the scenario bank npz path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    stage1 = _stage1_cfg(cfg)
    output_dir = _resolve(stage1["output_dir"], base)
    bank_path = (
        _resolve(args.scenario_bank, base)
        if str(args.scenario_bank or "").strip()
        else output_dir / str(stage1["scenario_bank"].get("output_name", "scenario_bank.npz"))
    )
    if not bank_path.exists():
        raise FileNotFoundError(f"Scenario bank not found: {bank_path}")
    natural_dir = _resolve(cfg.get("paths", {}).get("natural_dataset_dir", ""), base)
    schema = load_json(natural_dir / "feature_schema.json")
    arrays = _load_npz(bank_path)
    summary = {
        "scenario_bank_path": str(bank_path),
        **diagnose(arrays, schema, cfg),
    }
    write_json(output_dir / "stage1_scenario_bank_diagnostics.json", summary)
    write_figures(arrays, schema, cfg, output_dir / "figures")


if __name__ == "__main__":
    main()
