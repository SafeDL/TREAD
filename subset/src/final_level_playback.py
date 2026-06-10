#!/usr/bin/env python3
"""Shared final-level subset playback implementation."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from tools.plot_style import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import save_json
from process_highD.src.idm_ego import load_idm_ego_config
from tools.io import load_npz, resolve_path
from subset.src.closed_loop_runner import ClosedLoopCutInRunner, ClosedLoopFollowingRunner
from subset.src.context_distribution import load_tail_context_distribution
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler


DEFAULT_CONFIG_PATH = (
    ROOT
    / "subset"
    / "scripts"
    / "configs"
    / "latent_subset_following.yaml"
)
SCRIPT_DEFAULTS: dict[str, Any] = {
    "config": str(DEFAULT_CONFIG_PATH),
    "samples_path": None,
    "output_dir": None,
    "num_cases": 5,
    "level": -1,
    "unique_contexts": True,
    "view_width": 120.0,
    "vehicle_width": 2.0,
    "tail_steps": 50,
    "speed": 1.0,
    "render_gif": True,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _paths(
    config: dict[str, Any],
    base: Path,
    *,
    samples_path: str | None,
    output_dir: str | None,
) -> dict[str, Path]:
    paths = config.get("paths", {})
    subset_cfg = config.get("subset_simulation", {})
    if "tail_context_path" not in paths:
        raise KeyError("Config paths.tail_context_path is required")
    if "condition_distribution_path" not in paths:
        raise KeyError("Config paths.condition_distribution_path is required")
    if "evt_model_path" not in paths:
        raise KeyError("Config paths.evt_model_path is required")
    if "output_dir" not in subset_cfg:
        raise KeyError("Config subset_simulation.output_dir is required")
    subset_output = resolve_path(str(subset_cfg["output_dir"]), base)
    samples = (
        resolve_path(samples_path, base)
        if samples_path
        else subset_output / "latent_subset_samples.npz"
    )
    out_dir = (
        resolve_path(output_dir, base)
        if output_dir
        else subset_output / "figures" / "final_level_playbacks"
    )
    return {
        "tail_contexts": resolve_path(paths["tail_context_path"], base),
        "condition_distribution": resolve_path(
            paths["condition_distribution_path"],
            base,
        ),
        "evt_model": resolve_path(paths["evt_model_path"], base),
        "samples": samples,
        "output_dir": out_dir,
    }


def _load_contexts(
    path: Path,
    distribution_path: Path,
    config: dict[str, Any],
    *,
    event_type: str,
) -> Any:
    sampling_cfg = dict(config.get("context_sampling", {}) or {})
    target_fps = float(config.get("sampling", {}).get("target_fps", 25.0))
    return load_tail_context_distribution(
        path,
        distribution_path,
        event_type=event_type,
        seed=int(sampling_cfg.get("seed", config.get("training", {}).get("seed", 42))),
        population_size=int(sampling_cfg.get("population_size", 2_147_483_647)),
        dt=1.0 / max(target_fps, 1.0e-6),
    )


def _level_index(samples: dict[str, np.ndarray], requested: int) -> int:
    levels = int(samples["scores"].shape[0])
    idx = requested if requested >= 0 else levels + requested
    if idx < 0 or idx >= levels:
        raise IndexError(f"level {requested} is out of range for {levels} levels")
    return int(idx)


def _case_rows(
    samples: dict[str, np.ndarray],
    level_idx: int,
    *,
    num_cases: int,
    unique_contexts: bool,
) -> list[dict[str, Any]]:
    scores = np.asarray(samples["scores"][level_idx], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    rows: list[dict[str, Any]] = []
    mask = samples.get("action_mask")
    seen_contexts: set[int] = set()
    rank = 0
    for sample_idx in order:
        sample_idx = int(sample_idx)
        context_index = int(samples["context_indices"][level_idx, sample_idx])
        if unique_contexts and context_index in seen_contexts:
            continue
        seen_contexts.add(context_index)
        rank += 1
        if mask is None:
            steps = int(samples["actions"].shape[2])
        else:
            steps = int(np.sum(mask[level_idx, sample_idx] > 0.0))
        steps = max(steps, 1)
        rows.append(
            {
                "rank": int(rank),
                "level": int(level_idx),
                "sample_index": sample_idx,
                "context_index": context_index,
                "score": float(scores[sample_idx]),
                "threshold": float(samples["thresholds"][level_idx])
                if "thresholds" in samples
                else float("nan"),
                "accepted": float(samples["accepted_mask"][level_idx, sample_idx])
                if "accepted_mask" in samples
                else float("nan"),
                "steps": int(steps),
                "actions": np.asarray(
                    samples["actions"][level_idx, sample_idx, :steps],
                    dtype=np.float32,
                ),
            }
        )
        if len(rows) >= int(num_cases):
            break
    return rows


def _display_ttc_label(item: dict[str, float]) -> str:
    if float(item.get("collision", 0.0)) > 0.0:
        return "collision"
    ttc = float(item.get("ttc", np.nan))
    if not np.isfinite(ttc) or ttc >= 999.0:
        return "n/a"
    if ttc > 60.0:
        return ">60s"
    return f"{ttc:.2f}s"


def _context_kinematics(context: dict[str, Any]) -> dict[str, float]:
    raw = np.asarray(context["initial_states"], dtype=np.float32)
    ego = raw[0]
    lead = raw[1]
    ego_speed = float(np.hypot(float(ego[2]), float(ego[3])))
    lead_speed = float(np.hypot(float(lead[2]), float(lead[3])))
    gap = float(
        lead[0]
        - ego[0]
        - 0.5
        * (
            float(context.get("ego_length", 4.8))
            + float(context.get("adv_length", 4.8))
        )
    )
    return {
        "context_initial_gap": gap,
        "context_initial_ego_speed": ego_speed,
        "context_initial_lead_speed": lead_speed,
        "context_initial_closing_speed": ego_speed - lead_speed,
    }


def _event_type_from_config(config: dict[str, Any]) -> str:
    return str(config.get("event", {}).get("event_type", "following"))


def _apply_shared_idm_ego_config(
    config: dict[str, Any],
    config_dir: Path,
    *,
    event_type: str,
) -> None:
    configured = config.get("idm_ego_config_path") or config.get("paths", {}).get(
        "idm_ego_config_path"
    )
    if not configured:
        return
    shared = load_idm_ego_config(
        resolve_path(str(configured), config_dir),
        event_type=event_type,
    )
    config["idm_ego"] = {**dict(config.get("idm_ego", {}) or {}), **shared}
    env_cfg = config.setdefault("env", {})
    ego_response_cfg = config.setdefault("ego_response", {})
    if "target_speed" in shared:
        env_cfg["ego_target_speed"] = float(shared["target_speed"])
    if "speed_limit" in shared:
        env_cfg["speed_limit"] = float(shared["speed_limit"])
    if "lanes_count" in shared:
        env_cfg["lanes_count"] = int(shared["lanes_count"])
    if "enable_lane_change" in shared:
        ego_response_cfg["enable_lane_change"] = bool(shared["enable_lane_change"])


def _make_runner(
    config: dict[str, Any],
    config_dir: Path,
) -> ClosedLoopFollowingRunner:
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=config_dir).eval()
    event_type = str(
        sampler.prior.schema.get("event_type", _event_type_from_config(config))
    )
    if event_type == "cut_in":
        execution_mode = str(
            config.get("event", {}).get("execution_mode", "rolling_control")
        )
        if execution_mode != "rolling_control":
            raise ValueError(
                "Cut-in playback requires event.execution_mode to be "
                "'rolling_control'"
            )
        return ClosedLoopCutInRunner(sampler, config)
    return ClosedLoopFollowingRunner(sampler, config)


def _trace_array(trace: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray(
        [float(item.get(key, np.nan)) for item in trace],
        dtype=np.float32,
    )


def _add_vehicle(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color: str,
    label: str,
    zorder: int,
) -> None:
    rect = Rectangle(
        (-0.5 * length, -0.5 * width),
        length,
        width,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.92,
        zorder=zorder,
    )
    rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(rect)
    ax.text(
        x,
        y + 0.8 * width,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=zorder + 1,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )


def _write_overview_png(
    trace: list[dict[str, float]],
    row: dict[str, Any],
    context: dict[str, Any],
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    if not trace:
        raise RuntimeError("Cannot write overview for an empty rollout trace")
    steps = _trace_array(trace, "step")
    ego_x = _trace_array(trace, "ego_position")
    lead_x = _trace_array(trace, "lead_position")
    gap = _trace_array(trace, "gap")
    raw_ttc = _trace_array(trace, "ttc")
    ttc = np.where(raw_ttc >= 999.0, np.nan, np.clip(raw_ttc, 0.0, 60.0))
    ego_accel = _trace_array(trace, "ego_accel")
    lead_accel = _trace_array(trace, "lead_accel")
    collisions = _trace_array(trace, "collision") > 0.0

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=False)
    axes[0].plot(steps, ego_x, label="ego x", color="tab:red")
    axes[0].plot(steps, lead_x, label="lead x", color="tab:blue")
    axes[0].set_ylabel("x [m]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(steps, gap, label="gap", color="tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].axhline(2.0, color="tab:orange", linewidth=0.9, alpha=0.7, linestyle="--")
    axes[1].set_ylabel("gap [m]")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(steps, ttc, label="TTC", color="tab:green")
    axes[2].plot(steps, ego_accel, label="ego accel", color="tab:red", alpha=0.75)
    axes[2].plot(steps, lead_accel, label="lead accel", color="tab:blue", alpha=0.75)
    if np.any(collisions):
        axes[2].scatter(
            steps[collisions],
            np.zeros(int(np.sum(collisions)), dtype=np.float32),
            label="collision",
            color="black",
            marker="x",
            zorder=5,
        )
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("TTC [s] / accel [m/s^2]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    event_id = context.get("event_id", "")
    fig.suptitle(
        "subset final-level case "
        f"rank={row['rank']} level={row['level']} sample={row['sample_index']} "
        f"context={row['context_index']} event={event_id}\n"
        f"score={row['score']:.4f} replay_risk={metrics.get('risk_score', np.nan):.4f} "
        f"min_gap={metrics.get('min_gap', np.nan):.3f} "
        f"min_ttc={metrics.get('min_ttc', np.nan):.3f} "
        f"collision={metrics.get('collision', np.nan):.0f}",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _write_gif(
    trace: list[dict[str, float]],
    row: dict[str, Any],
    context: dict[str, Any],
    metrics: dict[str, float],
    output_path: Path,
    *,
    view_width: float,
    vehicle_width: float,
    tail_steps: int,
    fps: float,
) -> Path:
    if not trace:
        raise RuntimeError("Cannot write GIF for an empty rollout trace")
    actual_path = output_path.with_suffix(".gif")
    writer = PillowWriter(fps=max(min(float(fps), 15.0), 1.0))

    ego_length = float(context.get("ego_length", 4.8))
    lead_length = float(context.get("adv_length", 4.8))
    ego_y = _trace_array(trace, "ego_y")
    lead_y = _trace_array(trace, "lead_y")
    ymin = float(np.nanmin(np.concatenate([ego_y, lead_y]))) - 5.0
    ymax = float(np.nanmax(np.concatenate([ego_y, lead_y]))) + 5.0
    half_width = 0.5 * float(view_width)

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#6f7378")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title("")
    ego_line, = ax.plot([], [], color="tab:red", linewidth=1.6, alpha=0.8)
    lead_line, = ax.plot([], [], color="tab:blue", linewidth=1.6, alpha=0.8)
    for y in np.arange(np.floor(ymin / 4.0) * 4.0, ymax + 4.0, 4.0):
        ax.axhline(y, color="white", linewidth=0.8, alpha=0.28, linestyle="--")
    frame_artists: list[Any] = []

    with writer.saving(fig, str(actual_path), dpi=110):
        for frame_idx, item in enumerate(trace):
            for artist in frame_artists:
                artist.remove()
            frame_artists.clear()

            ego_x = float(item["ego_position"])
            lead_x = float(item["lead_position"])
            center_x = 0.5 * (ego_x + lead_x)
            ax.set_xlim(center_x - half_width, center_x + half_width)

            trail_start = max(0, frame_idx - int(tail_steps))
            trail = trace[trail_start:frame_idx + 1]
            ego_line.set_data(
                [float(t["ego_position"]) for t in trail],
                [float(t["ego_y"]) for t in trail],
            )
            lead_line.set_data(
                [float(t["lead_position"]) for t in trail],
                [float(t["lead_y"]) for t in trail],
            )
            before = len(ax.patches), len(ax.texts)
            _add_vehicle(
                ax,
                x=ego_x,
                y=float(item["ego_y"]),
                heading=float(item["ego_yaw"]),
                length=ego_length,
                width=float(vehicle_width),
                color="#e31a1c",
                label="ego",
                zorder=5,
            )
            frame_artists.extend(list(ax.patches)[before[0]:])
            frame_artists.extend(list(ax.texts)[before[1]:])
            before = len(ax.patches), len(ax.texts)
            _add_vehicle(
                ax,
                x=lead_x,
                y=float(item["lead_y"]),
                heading=float(item["lead_yaw"]),
                length=lead_length,
                width=float(vehicle_width),
                color="#1f78b4",
                label="lead",
                zorder=4,
            )
            frame_artists.extend(list(ax.patches)[before[0]:])
            frame_artists.extend(list(ax.texts)[before[1]:])
            title.set_text(
                f"rank={row['rank']} sample={row['sample_index']} "
                f"score={row['score']:.3f} replay_risk={metrics.get('risk_score', np.nan):.3f} | "
                f"step={int(item['step'])} gap={float(item['gap']):.2f}m "
                f"TTC={_display_ttc_label(item)}"
            )
            writer.grab_frame()
    plt.close(fig)
    return actual_path


def _manifest_row(
    row: dict[str, Any],
    context: dict[str, Any],
    metrics: dict[str, float],
    *,
    png_path: Path,
    animation_path: Path | None,
) -> dict[str, Any]:
    context_kin = _context_kinematics(context)
    return {
        "rank": int(row["rank"]),
        "level": int(row["level"]),
        "sample_index": int(row["sample_index"]),
        "context_index": int(row["context_index"]),
        "recording_id": context.get("recording_id"),
        "event_id": context.get("event_id"),
        "source_type": context.get("source_type"),
        "tail_threshold": float(context.get("tail_threshold", np.nan)),
        "context_y_long": float(context.get("y_long", np.nan)),
        "context_risk_score": float(context.get("risk_score", np.nan)),
        "recorded_min_gap": float(context.get("recorded_min_gap", np.nan)),
        "recorded_min_ttc": float(context.get("recorded_min_ttc", np.nan)),
        **context_kin,
        "subset_score": float(row["score"]),
        "replay_risk": float(metrics.get("risk_score", np.nan)),
        "risk_score": float(metrics.get("risk_score", np.nan)),
        "y_long": float(metrics.get("y_long", np.nan)),
        "y_cutin": float(metrics.get("y_cutin", np.nan)),
        "evt_tail_probability": float(metrics.get("evt_tail_probability", np.nan)),
        "collision": float(metrics.get("collision", np.nan)),
        "near_collision": float(metrics.get("near_collision", np.nan)),
        "min_gap": float(metrics.get("min_gap", np.nan)),
        "min_ttc": float(metrics.get("min_ttc", np.nan)),
        "hard_brake": float(metrics.get("hard_brake", np.nan)),
        "cutin_safety_risk_score": float(
            metrics.get("cutin_safety_risk_score", np.nan)
        ),
        "cutin_time_headway": float(metrics.get("cutin_time_headway", np.nan)),
        "cutin_lateral_time_gap": float(
            metrics.get("cutin_lateral_time_gap", np.nan)
        ),
        "safety_distance_deficit": float(
            metrics.get("safety_distance_deficit", np.nan)
        ),
        "max_post_cutin_drac": float(
            metrics.get("max_post_cutin_drac", np.nan)
        ),
        "min_abs_lateral_offset": float(
            metrics.get("min_abs_lateral_offset", np.nan)
        ),
        "final_abs_lateral_offset": float(
            metrics.get("final_abs_lateral_offset", np.nan)
        ),
        "max_lateral_approach_speed": float(
            metrics.get("max_lateral_approach_speed", np.nan)
        ),
        "lateral_overlap_fraction": float(
            metrics.get("lateral_overlap_fraction", np.nan)
        ),
        "is_cutin": float(metrics.get("is_cutin", np.nan)),
        "planned_steps": int(row["steps"]),
        "steps": int(metrics.get("steps", row["steps"])),
        "png": str(png_path),
        "animation": str(animation_path) if animation_path is not None else None,
    }


def replay_final_level(
    config: dict[str, Any],
    config_dir: Path,
    *,
    expected_event_type: str | None = None,
) -> Path:
    paths = _paths(
        config,
        config_dir,
        samples_path=SCRIPT_DEFAULTS["samples_path"],
        output_dir=SCRIPT_DEFAULTS["output_dir"],
    )
    if not paths["samples"].exists():
        raise FileNotFoundError(f"Subset samples not found: {paths['samples']}")
    evt_cfg = config.setdefault("evt", {})
    evt_cfg["model_path"] = str(paths["evt_model"])
    evt_cfg["score_space"] = str(evt_cfg.get("score_space", "evt"))
    samples = load_npz(paths["samples"])
    level_idx = _level_index(samples, int(SCRIPT_DEFAULTS["level"]))
    cases = _case_rows(
        samples,
        level_idx,
        num_cases=int(SCRIPT_DEFAULTS["num_cases"]),
        unique_contexts=bool(SCRIPT_DEFAULTS["unique_contexts"]),
    )
    if not cases:
        raise RuntimeError("No final-level subset cases found")
    event_type = _event_type_from_config(config)
    if expected_event_type is not None and event_type != expected_event_type:
        raise ValueError(f"Expected {expected_event_type} config, got {event_type}")
    contexts = _load_contexts(
        paths["tail_contexts"],
        paths["condition_distribution"],
        config,
        event_type=event_type,
    )
    _apply_shared_idm_ego_config(config, config_dir, event_type=event_type)

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _make_runner(config, config_dir)
    target_fps = float(config.get("sampling", {}).get("target_fps", 25.0))
    fps = target_fps * SCRIPT_DEFAULTS["speed"]
    manifest: list[dict[str, Any]] = []

    for row in cases:
        context_idx = int(row["context_index"])
        if context_idx < 0 or context_idx >= len(contexts):
            raise IndexError(f"context_index out of range: {context_idx}")
        context = contexts[context_idx]
        actions = np.asarray(row["actions"], dtype=np.float32)
        result = runner.rollout_pre_sampled_plan(
            context,
            actions,
            episode_steps=int(row["steps"]),
        )
        safe_event_id = str(context.get("event_id", "event")).replace("/", "_")
        stem = (
            f"level{level_idx:02d}_rank{int(row['rank']):02d}_"
            f"sample{int(row['sample_index']):04d}_ctx{context_idx:03d}_{safe_event_id}"
        )
        png_path = output_dir / f"{stem}.png"
        animation_path = None
        gif_path = output_dir / f"{stem}.gif"
        _write_overview_png(
            result.trace,
            row,
            context,
            result.metrics,
            png_path,
        )
        if bool(SCRIPT_DEFAULTS["render_gif"]):
            animation_path = _write_gif(
                result.trace,
                row,
                context,
                result.metrics,
                gif_path,
                view_width=SCRIPT_DEFAULTS["view_width"],
                vehicle_width=SCRIPT_DEFAULTS["vehicle_width"],
                tail_steps=int(SCRIPT_DEFAULTS["tail_steps"]),
                fps=fps,
            )
        manifest.append(
            _manifest_row(
                row,
                context,
                result.metrics,
                png_path=png_path,
                animation_path=animation_path
                if animation_path is not None and animation_path.exists()
                else None,
            )
        )
        logger.info(
            "Rendered rank %d sample %d score %.4f -> %s",
            row["rank"],
            row["sample_index"],
            row["score"],
            png_path,
        )

    manifest_path = output_dir / "final_level_playback_manifest.json"
    save_json(
        {
            "samples": str(paths["samples"]),
            "tail_contexts": str(paths["tail_contexts"]),
            "level": int(level_idx),
            "num_cases": int(len(manifest)),
            "unique_contexts": bool(SCRIPT_DEFAULTS["unique_contexts"]),
            "cases": manifest,
        },
        manifest_path,
    )
    return manifest_path
