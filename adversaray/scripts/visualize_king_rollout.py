#!/usr/bin/env python3
"""Replay saved prior/KING plans with highway-env road graphics."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HIGHWAY_ROOT = ROOT / "HighwayEnv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if HIGHWAY_ROOT.exists() and str(HIGHWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGHWAY_ROOT))

from adversaray.src.closed_loop_runner import (
    ClosedLoopFollowingRunner,
    IDMVehicle,
    ScriptedLeadVehicle,
    _highway_env_error_message,
)
from adversaray.src.config_utils import apply_rss_config_override
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler
from adversaray.src.prior_guided_train import _context
from adversaray.src.rss import rss_safe_distance
from diffusion.src.utils import load_yaml, setup_logging

try:
    import pygame  # type: ignore
    from highway_env.road.graphics import RoadGraphics, WorldSurface  # type: ignore
except Exception as exc:  # noqa: BLE001
    pygame = None
    RoadGraphics = None
    WorldSurface = None
    HIGHWAY_GRAPHICS_IMPORT_ERROR = exc
else:
    HIGHWAY_GRAPHICS_IMPORT_ERROR = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
SCRIPT_DEFAULTS = {
    "samples_name": "king_guided_samples.npz",
    "case_index": 0,
    "mode": "both",
    "synthetic_val": True,
    "render_human": True,
    "save_video": False,
    "save_frames": False,
    "width": 960,
    "height": 480,
    "fps": 15,
    "scaling": 7.0,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _output_dir(cfg: dict[str, Any], base: Path) -> Path:
    paths = cfg.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    return _resolve(paths["output_dir"], base)


def _make_frozen_runner(cfg: dict[str, Any], base: Path) -> ClosedLoopFollowingRunner:
    prior_cfg = copy.deepcopy(cfg)
    prior_cfg.setdefault("policy", {})["enabled"] = False
    prior_cfg.setdefault("paths", {})["policy_checkpoint"] = ""
    sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()
    sampler.set_guidance_enabled(False)
    return ClosedLoopFollowingRunner(sampler, prior_cfg)


def _check_graphics() -> None:
    if pygame is None or RoadGraphics is None or WorldSurface is None:
        raise RuntimeError(f"highway-env graphics are unavailable: {HIGHWAY_GRAPHICS_IMPORT_ERROR}")
    if IDMVehicle is None:
        raise RuntimeError(_highway_env_error_message())


class HighwayReplayRenderer:
    def __init__(
        self,
        *,
        render_human: bool,
        save_frames: bool,
        frame_dir: Path,
        width: int,
        height: int,
        fps: int,
        scaling: float,
    ) -> None:
        _check_graphics()
        self.render_human = bool(render_human)
        self.save_frames = bool(save_frames)
        self.frame_dir = frame_dir
        self.fps = int(fps)
        self.frame_count = 0
        self.frames: list[Image.Image] = []
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height)) if self.render_human else None
        if self.render_human:
            pygame.display.set_caption("KING highway-env rollout")
        self.surface = WorldSurface((width, height), 0, pygame.Surface((width, height)))
        self.surface.scaling = float(scaling)
        self.surface.centering_position = [0.35, 0.52]
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 19)

    def handle_events(self) -> bool:
        if not self.render_human:
            return True
        keep_running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False
            self.surface.handle_event(event)
        return keep_running

    def draw(self, road: Any, ego: Any, mode_label: str, metrics: dict[str, float]) -> bool:
        if not self.handle_events():
            return False
        self.surface.move_display_window_to(np.asarray(ego.position, dtype=np.float64))
        RoadGraphics.display(road, self.surface)
        RoadGraphics.display_road_objects(road, self.surface, offscreen=not self.render_human)
        RoadGraphics.display_traffic(road, self.surface, simulation_frequency=self.fps, offscreen=not self.render_human)
        self._draw_overlay(mode_label, metrics)
        if self.render_human and self.screen is not None:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)
        array = pygame.surfarray.array3d(self.surface)
        frame = Image.fromarray(np.moveaxis(array, 0, 1))
        self.frames.append(frame)
        if self.save_frames:
            path = self.frame_dir / f"frame_{self.frame_count:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(path)
        self.frame_count += 1
        return True

    def _draw_overlay(self, mode_label: str, metrics: dict[str, float]) -> None:
        lines = [
            f"{mode_label.upper()}  step={int(metrics['step'])}",
            f"gap={metrics['gap']:.2f} m   TTC={metrics['ttc']:.2f} s   RSS={metrics['rss_margin']:.2f} m",
            f"ego v={metrics['ego_speed']:.2f} m/s   lead v={metrics['lead_speed']:.2f} m/s",
            f"lead a={metrics['lead_accel']:.2f} m/s2   jerk={metrics['lead_jerk']:.2f} m/s3",
        ]
        padding = 8
        line_height = 22
        box_width = 520
        box_height = padding * 2 + line_height * len(lines)
        overlay = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        overlay.fill((20, 24, 28, 190))
        for i, text in enumerate(lines):
            font = self.font if i == 0 else self.small_font
            color = (255, 255, 255) if i == 0 else (230, 235, 240)
            overlay.blit(font.render(text, True, color), (padding, padding + i * line_height))
        self.surface.blit(overlay, (12, 12))

    def save_video(self, path: Path) -> None:
        if not self.frames:
            raise RuntimeError("No frames were rendered")
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        duration_ms = int(round(1000.0 / max(self.fps, 1)))
        if suffix in {"", ".gif"}:
            gif_path = path if suffix == ".gif" else path.with_suffix(".gif")
            self.frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            logger.info("Saved rollout GIF to %s", gif_path)
            return
        if suffix == ".mp4":
            try:
                import imageio.v2 as imageio  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "MP4 export needs imageio/ffmpeg, which is not installed in this environment. "
                    "Use .gif or --save-frames."
                ) from exc
            imageio.mimsave(path, [np.asarray(frame) for frame in self.frames], fps=max(self.fps, 1))
            logger.info("Saved rollout MP4 to %s", path)
            return
        raise ValueError(f"Unsupported video suffix {suffix!r}; use .gif or --save-frames")

    def close(self) -> None:
        if pygame is not None:
            pygame.quit()


def _init_road_vehicles(
    runner: ClosedLoopFollowingRunner,
    initial_context: dict[str, Any],
) -> tuple[Any, Any, Any, float, float, float]:
    raw_context = np.asarray(initial_context["raw_context_states"], dtype=np.float32).copy()
    raw_context[:, :, 1] = 0.0
    ego_length = float(initial_context["ego_length"])
    lead_length = float(initial_context["adv_length"])
    rebuilt = runner._maybe_reconstruct_highd_context(initial_context, ego_length, lead_length)
    if rebuilt is not None:
        raw_context, ego_length, lead_length = rebuilt
        raw_context[:, :, 1] = 0.0
    ego0 = raw_context[-1, 0]
    lead0 = raw_context[-1, 1]
    initial_gap = float(lead0[0] - ego0[0] - 0.5 * (ego_length + lead_length))
    if initial_gap <= runner.initial_gap_min:
        raise RuntimeError(f"Invalid visualization context: initial gap {initial_gap:.3f} <= {runner.initial_gap_min:.3f}")
    road = runner._make_road()
    ego = IDMVehicle(
        road,
        position=np.asarray([ego0[0], 0.0], dtype=np.float64),
        heading=0.0,
        speed=max(float(ego0[2]), 0.0),
        target_speed=runner.ego_target_speed,
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
    ego.color = (90, 180, 255)
    lead.color = (255, 208, 92)
    if hasattr(ego, "diagonal"):
        ego.diagonal = float(np.sqrt(ego.LENGTH**2 + ego.WIDTH**2))
    if hasattr(lead, "diagonal"):
        lead.diagonal = float(np.sqrt(lead.LENGTH**2 + lead.WIDTH**2))
    road.vehicles = [ego, lead]
    if hasattr(ego, "front_vehicle"):
        ego.front_vehicle = lead
    return road, ego, lead, ego_length, lead_length, float(lead0[4])


def _replay_plan(
    *,
    runner: ClosedLoopFollowingRunner,
    context: dict[str, Any],
    plan: np.ndarray,
    mode_label: str,
    renderer: HighwayReplayRenderer,
) -> list[dict[str, float]]:
    road, ego, lead, ego_length, lead_length, lead_accel = _init_road_vehicles(runner, context)
    action_cfg = runner.config.get("physics", runner.config.get("action", {}))
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    rep = str(runner.sampler.prior.schema.get("action_representation", runner.sampler.prior.config.get("action", {}).get("representation", "jerk"))).lower()
    prev_lead_accel = float(lead_accel)
    trace: list[dict[str, float]] = []
    steps = min(int(runner.episode_steps), int(plan.shape[0]))
    for step in range(steps):
        action_value = float(plan[step, 0])
        if rep == "jerk":
            lead_accel = lead_accel + action_value * runner.dt
            jerk = action_value
        elif rep == "acceleration":
            jerk = (action_value - prev_lead_accel) / max(runner.dt, 1e-6)
            lead_accel = action_value
        else:
            raise ValueError(f"Unsupported action representation: {rep}")
        lead_accel = float(np.clip(lead_accel, ax_min, ax_max))
        prev_lead_accel = lead_accel
        lead.set_acceleration(lead_accel)
        road.act()
        road.step(runner.dt)
        gap = float(lead.position[0] - ego.position[0] - 0.5 * (ego_length + lead_length))
        closing = float(ego.speed - lead.speed)
        ttc = gap / max(closing, 1e-6) if closing > 1e-6 else 1000.0
        safe = float(rss_safe_distance(torch.tensor([ego.speed]), torch.tensor([max(lead.speed, 0.0)]), runner.rss_cfg)[0])
        metrics = {
            "step": float(step),
            "gap": gap,
            "ttc": float(ttc),
            "rss_margin": gap - safe,
            "ego_speed": float(ego.speed),
            "ego_accel": float(ego.action.get("acceleration", 0.0)),
            "lead_speed": float(lead.speed),
            "lead_accel": float(lead_accel),
            "lead_jerk": float(jerk),
        }
        trace.append(metrics)
        if not renderer.draw(road, ego, mode_label, metrics):
            break
        if ego.crashed or lead.crashed:
            break
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--samples", default="")
    parser.add_argument("--synthetic-val", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["synthetic_val"])
    parser.add_argument("--case-index", type=int, default=SCRIPT_DEFAULTS["case_index"])
    parser.add_argument("--mode", choices=("prior", "king", "both"), default=SCRIPT_DEFAULTS["mode"])
    parser.add_argument("--render-human", action="store_true", default=SCRIPT_DEFAULTS["render_human"])
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["save_video"])
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["save_frames"])
    parser.add_argument("--video-path", default="")
    parser.add_argument("--frame-dir", default="")
    parser.add_argument("--width", type=int, default=SCRIPT_DEFAULTS["width"])
    parser.add_argument("--height", type=int, default=SCRIPT_DEFAULTS["height"])
    parser.add_argument("--fps", type=int, default=SCRIPT_DEFAULTS["fps"])
    parser.add_argument("--scaling", type=float, default=SCRIPT_DEFAULTS["scaling"])
    parser.add_argument("--log-level", default=SCRIPT_DEFAULTS["log_level"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not args.render_human and not args.save_video and not args.save_frames:
        raise ValueError("Enable at least one output: --render-human, --save-video, or --save-frames")
    if not args.synthetic_val:
        raise ValueError("This KING visualizer expects synthetic-val KING samples; pass a matching samples npz if extending it.")

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    apply_rss_config_override(cfg, cfg_path.parent)
    base = cfg_path.parent
    output_root = _output_dir(cfg, base)
    samples_path = _resolve(args.samples, base) if str(args.samples or "").strip() else output_root / str(SCRIPT_DEFAULTS["samples_name"])
    if not samples_path.exists():
        raise FileNotFoundError(f"KING samples not found: {samples_path}")
    samples = _load_npz(samples_path)
    required = {"context_states", "ego_length", "adv_length", "prior_actions", "king_actions"}
    missing = sorted(required - set(samples))
    if missing:
        raise KeyError(f"{samples_path} is missing required arrays: {missing}")
    case_index = int(args.case_index)
    if not 0 <= case_index < int(samples["context_states"].shape[0]):
        raise IndexError(f"case-index {case_index} outside [0, {samples['context_states'].shape[0] - 1}]")

    raw_case = {
        "context_states": samples["context_states"][case_index : case_index + 1],
        "ego_length": samples["ego_length"][case_index : case_index + 1],
        "adv_length": samples["adv_length"][case_index : case_index + 1],
    }
    if "dataset_index" in samples:
        raw_case["dataset_index"] = samples["dataset_index"][case_index : case_index + 1]
    context = _context(raw_case, 0)
    runner = _make_frozen_runner(cfg, base)

    frame_dir = _resolve(args.frame_dir, base) if str(args.frame_dir or "").strip() else output_root / "figures" / f"king_rollout_case_{case_index:04d}_frames"
    renderer = HighwayReplayRenderer(
        render_human=bool(args.render_human),
        save_frames=bool(args.save_frames),
        frame_dir=frame_dir,
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        scaling=float(args.scaling),
    )
    try:
        modes = ("prior", "king") if args.mode == "both" else (args.mode,)
        for mode in modes:
            plan_key = "prior_actions" if mode == "prior" else "king_actions"
            logger.info("Rendering case %d mode=%s", case_index, mode)
            _replay_plan(
                runner=runner,
                context=context,
                plan=np.asarray(samples[plan_key][case_index], dtype=np.float32),
                mode_label=mode,
                renderer=renderer,
            )
        if args.save_video:
            video_path = (
                _resolve(args.video_path, base)
                if str(args.video_path or "").strip()
                else output_root / "figures" / f"king_rollout_case_{case_index:04d}_{args.mode}.gif"
            )
            renderer.save_video(video_path)
        if args.save_frames:
            logger.info("Saved rollout frames to %s", frame_dir)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
