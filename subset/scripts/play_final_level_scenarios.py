#!/usr/bin/env python3
"""Replay and visualize final-level subset simulation scenarios."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, save_json, setup_logging
from utils.context import context_from_npz, load_context_npz
from utils.io import load_npz, resolve_path
from subset.src.closed_loop_runner import ClosedLoopFollowingRunner
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler


DEFAULT_CONFIG_PATH = (
    ROOT
    / "subset"
    / "scripts"
    / "configs"
    / "latent_subset_simulation.yaml"
)
SCRIPT_DEFAULTS: dict[str, Any] = {
    "samples_path": None,
    "output_dir": None,
    "num_cases": 5,
    "level": -1,
    "view_width": 120.0,
    "vehicle_width": 2.0,
    "tail_steps": 50,
    "speed": 1.0,
    "render_mp4": True,
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
        "evt_model": resolve_path(paths["evt_model_path"], base),
        "samples": samples,
        "output_dir": out_dir,
    }


def _load_contexts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Tail context file not found: {path}")
    raw = load_context_npz(path)
    count = int(raw["context_states"].shape[0])
    return [context_from_npz(raw, idx) for idx in range(count)]


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
) -> list[dict[str, Any]]:
    scores = np.asarray(samples["scores"][level_idx], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    rows: list[dict[str, Any]] = []
    mask = samples.get("action_mask")
    for rank, sample_idx in enumerate(order[: int(num_cases)]):
        sample_idx = int(sample_idx)
        if mask is None:
            steps = int(samples["actions"].shape[2])
        else:
            steps = int(np.sum(mask[level_idx, sample_idx] > 0.0))
        steps = max(steps, 1)
        rows.append(
            {
                "rank": int(rank + 1),
                "level": int(level_idx),
                "sample_index": sample_idx,
                "context_index": int(samples["context_indices"][level_idx, sample_idx]),
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
    return rows


def _make_runner(config: dict[str, Any], config_dir: Path) -> ClosedLoopFollowingRunner:
    sampler = FrozenDiffusionSampler.from_config(config, config_dir=config_dir).eval()
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
    ttc = np.clip(_trace_array(trace, "ttc"), 0.0, 60.0)
    ego_accel = _trace_array(trace, "ego_accel")
    lead_accel = _trace_array(trace, "lead_accel")

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


def _write_animation(
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
) -> Path | None:
    if animation.writers.is_available("ffmpeg"):
        writer: Any = animation.FFMpegWriter(fps=max(float(fps), 1.0), bitrate=2000)
        actual_path = output_path
    elif animation.writers.is_available("pillow"):
        writer = animation.PillowWriter(fps=max(min(float(fps), 15.0), 1.0))
        actual_path = output_path.with_suffix(".gif")
        logger.warning("ffmpeg is unavailable; writing GIF animation %s", actual_path)
    else:
        logger.warning(
            "Neither ffmpeg nor pillow animation writers are available; "
            "skipping animation %s",
            output_path,
        )
        return None
    if not trace:
        raise RuntimeError("Cannot write animation for an empty rollout trace")

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
            trail = trace[trail_start : frame_idx + 1]
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
            frame_artists.extend(list(ax.patches)[before[0] :])
            frame_artists.extend(list(ax.texts)[before[1] :])
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
            frame_artists.extend(list(ax.patches)[before[0] :])
            frame_artists.extend(list(ax.texts)[before[1] :])
            title.set_text(
                f"rank={row['rank']} sample={row['sample_index']} "
                f"score={row['score']:.3f} replay_risk={metrics.get('risk_score', np.nan):.3f} | "
                f"step={int(item['step'])} gap={float(item['gap']):.2f}m "
                f"TTC={min(float(item['ttc']), 60.0):.2f}s"
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
    return {
        "rank": int(row["rank"]),
        "level": int(row["level"]),
        "sample_index": int(row["sample_index"]),
        "context_index": int(row["context_index"]),
        "recording_id": context.get("recording_id"),
        "event_id": context.get("event_id"),
        "subset_score": float(row["score"]),
        "replay_risk": float(metrics.get("risk_score", np.nan)),
        "risk_score": float(metrics.get("risk_score", np.nan)),
        "y_long": float(metrics.get("y_long", np.nan)),
        "evt_tail_probability": float(metrics.get("evt_tail_probability", np.nan)),
        "collision": float(metrics.get("collision", np.nan)),
        "near_collision": float(metrics.get("near_collision", np.nan)),
        "min_gap": float(metrics.get("min_gap", np.nan)),
        "min_ttc": float(metrics.get("min_ttc", np.nan)),
        "hard_brake": float(metrics.get("hard_brake", np.nan)),
        "steps": int(row["steps"]),
        "png": str(png_path),
        "animation": str(animation_path) if animation_path is not None else None,
    }


def replay_final_level(config: dict[str, Any], config_dir: Path, args: argparse.Namespace) -> Path:
    paths = _paths(
        config,
        config_dir,
        samples_path=args.samples_path,
        output_dir=args.output_dir,
    )
    if not paths["samples"].exists():
        raise FileNotFoundError(f"Subset samples not found: {paths['samples']}")
    contexts = _load_contexts(paths["tail_contexts"])
    evt_cfg = config.setdefault("evt", {})
    evt_cfg["model_path"] = str(paths["evt_model"])
    evt_cfg["score_space"] = str(evt_cfg.get("score_space", "evt"))
    samples = load_npz(paths["samples"])
    level_idx = _level_index(samples, int(args.level))
    cases = _case_rows(samples, level_idx, num_cases=int(args.num_cases))
    if not cases:
        raise RuntimeError("No final-level subset cases found")

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _make_runner(config, config_dir)
    target_fps = float(config.get("sampling", {}).get("target_fps", 25.0))
    fps = target_fps * float(args.speed)
    manifest: list[dict[str, Any]] = []

    for row in cases:
        context_idx = int(row["context_index"])
        if context_idx < 0 or context_idx >= len(contexts):
            raise IndexError(f"context_index out of range: {context_idx}")
        context = contexts[context_idx]
        result = runner.rollout_pre_sampled_plan(
            context,
            np.asarray(row["actions"], dtype=np.float32),
            episode_steps=int(row["steps"]),
        )
        safe_event_id = str(context.get("event_id", "event")).replace("/", "_")
        stem = (
            f"level{level_idx:02d}_rank{int(row['rank']):02d}_"
            f"sample{int(row['sample_index']):04d}_ctx{context_idx:03d}_{safe_event_id}"
        )
        png_path = output_dir / f"{stem}.png"
        animation_path = None
        requested_animation_path = output_dir / f"{stem}.mp4"
        _write_overview_png(
            result.trace,
            row,
            context,
            result.metrics,
            png_path,
        )
        if bool(args.render_mp4):
            animation_path = _write_animation(
                result.trace,
                row,
                context,
                result.metrics,
                requested_animation_path,
                view_width=float(args.view_width),
                vehicle_width=float(args.vehicle_width),
                tail_steps=int(args.tail_steps),
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
            "cases": manifest,
        },
        manifest_path,
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay top-scoring final-level subset simulation cases.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--samples-path", default=SCRIPT_DEFAULTS["samples_path"])
    parser.add_argument("--output-dir", default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--num-cases", type=int, default=SCRIPT_DEFAULTS["num_cases"])
    parser.add_argument("--level", type=int, default=SCRIPT_DEFAULTS["level"])
    parser.add_argument("--view-width", type=float, default=SCRIPT_DEFAULTS["view_width"])
    parser.add_argument("--vehicle-width", type=float, default=SCRIPT_DEFAULTS["vehicle_width"])
    parser.add_argument("--tail-steps", type=int, default=SCRIPT_DEFAULTS["tail_steps"])
    parser.add_argument("--speed", type=float, default=SCRIPT_DEFAULTS["speed"])
    parser.add_argument(
        "--no-mp4",
        dest="render_mp4",
        action="store_false",
        default=SCRIPT_DEFAULTS["render_mp4"],
    )
    parser.add_argument("--log-level", default=SCRIPT_DEFAULTS["log_level"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(str(args.log_level))
    cfg_path = Path(args.config).resolve()
    manifest = replay_final_level(
        load_yaml(cfg_path),
        cfg_path.parent,
        args,
    )
    logger.info("Wrote manifest to %s", manifest)


if __name__ == "__main__":
    main()
