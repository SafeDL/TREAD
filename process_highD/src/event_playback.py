"""Shared highD generated-scenario GIF playback helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "Liberation Serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from process_highD.src.io_utils import ensure_dir, load_config, resolve_data_path
from process_highD.src.loader import load_recording
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)

LOGGER = logging.getLogger(__name__)


def _select_indices(
    count: int,
    selection: str | int | tuple[int, ...] | list[int],
    random_seed: int,
) -> list[int]:
    if isinstance(selection, str):
        if selection.lower() != "all":
            raise ValueError(
                "selection string must be 'all', "
                f"got {selection!r}"
            )
        return list(range(count))
    if isinstance(selection, int):
        if selection <= 0:
            raise ValueError("selection integer must be positive")
        rng = np.random.default_rng(int(random_seed))
        sample_size = min(int(selection), count)
        return sorted(
            int(idx)
            for idx in rng.choice(count, size=sample_size, replace=False)
        )
    indices = [int(idx) for idx in selection]
    if not indices:
        raise ValueError("selection cannot be empty")
    return indices


def _load_recording(config: dict, config_path: Path, recording_id: int):
    raw_dir = resolve_data_path(config["paths"]["raw_dir"], config_path)
    rec = load_recording(str(raw_dir), recording_id)
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(
        config.get("sampling", {}).get(
            "target_fps",
            rec.recording_meta.get("frameRate", 25),
        )
    )
    return resample_recording(rec, target_fps)


def _safe_int(value, default=None):
    if pd.isna(value):
        return default
    return int(value)


def render_generated_scenarios_gif(
    *,
    generated_npz_path: Path,
    output_dir: Path,
    output_name: str,
    scenario_selection: str | int | tuple[int, ...] | list[int],
    random_seed: int,
    background_config_path: Path | None = None,
    event_type: str | None = None,
    dt: float = 0.04,
    view_width: float = 160.0,
    trail_frames: int = 50,
    playback_speed: float = 1.0,
    fps: float = 25.0,
) -> list[Path]:
    """Render diffusion-generated following or cut-in scenarios to GIF files.

    Each selected scenario produces one GIF. If the generated NPZ stores an
    ego trajectory, playback uses it; otherwise ego falls back to constant
    longitudinal speed while the target/lead vehicle follows the generated
    trajectory.
    """
    from matplotlib.animation import PillowWriter
    from tqdm import tqdm

    if not generated_npz_path.exists():
        raise FileNotFoundError(f"Generated scenarios not found: {generated_npz_path}")

    data = np.load(generated_npz_path, allow_pickle=True)
    inferred_event_type = _generated_event_type(data, event_type)
    num_scenarios = int(data["initial_states"].shape[0])
    trajectory_key = (
        "target_trajectory" if inferred_event_type == "cut_in" else "lead_trajectory"
    )
    if trajectory_key not in data.files:
        raise KeyError(
            f"{generated_npz_path} is missing {trajectory_key!r} for "
            f"{inferred_event_type} playback"
        )
    horizon = int(data[trajectory_key].shape[1])

    indices = _select_indices(
        num_scenarios, scenario_selection, random_seed,
    )
    LOGGER.info(
        "Selected %d / %d generated scenarios: %s",
        len(indices),
        num_scenarios,
        indices,
    )

    initial = data["initial_states"][indices].astype(np.float64)
    target_traj = data[trajectory_key][indices].astype(np.float64)
    ego_len = data["ego_length"][indices].astype(np.float64)
    target_len = data["adv_length"][indices].astype(np.float64)
    conditions = data["scenario_conditions"][indices].astype(np.float64)
    realized_conditions = (
        data["realized_scenario_conditions"][indices].astype(np.float64)
        if "realized_scenario_conditions" in data.files
        else conditions
    )
    condition_keys: list[str] = (
        [str(item) for item in data["condition_keys"].tolist()]
        if "condition_keys" in data.files
        else []
    )
    base_event_ids = (
        [str(item) for item in data["base_event_id"][indices].tolist()]
        if "base_event_id" in data.files
        else [""] * len(indices)
    )
    backgrounds = _load_generated_background_contexts(
        base_event_ids=base_event_ids,
        conditions=conditions,
        condition_keys=condition_keys,
        background_config_path=background_config_path,
        event_type=inferred_event_type,
        dt=dt,
    )
    ego0 = initial[:, 0]
    if "ego_trajectory" in data.files:
        ego_traj = data["ego_trajectory"][indices].astype(np.float64)
        ego_x = ego_traj[:, :, 0]
        ego_y = ego_traj[:, :, 1]
    else:
        t_arr = np.arange(horizon, dtype=np.float64) * dt
        ego_x = ego0[:, 0:1] + ego0[:, 2:3] * t_arr[None, :]
        ego_y = np.full_like(ego_x, ego0[:, 1:2])

    vehicle_width = 2.0
    half_width = view_width / 2.0

    ensure_dir(output_dir)
    output_paths: list[Path] = []

    for list_idx, global_idx in enumerate(indices):
        out_path = output_dir / f"{output_name}_{global_idx:05d}.gif"
        output_paths.append(out_path)
        LOGGER.info(
            "Rendering scenario %d (global idx %d) -> %s",
            list_idx,
            global_idx,
            out_path,
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_facecolor("#707070")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("mirrored y (m)")
        title_obj = ax.set_title("")
        title_obj.set_fontfamily("Liberation Serif")

        ego_line, = ax.plot([], [], color="#e31a1c", lw=1.8, alpha=0.9, zorder=3)
        target_line, = ax.plot([], [], color="#1f78b4", lw=1.8, alpha=0.9, zorder=3)

        lane_width = 3.75
        lane_boundaries = np.array(
            [
                -1.5 * lane_width,
                -0.5 * lane_width,
                0.5 * lane_width,
                1.5 * lane_width,
            ],
            dtype=float,
        )
        for j, y_lane in enumerate(lane_boundaries):
            is_outer = j == 0 or j == len(lane_boundaries) - 1
            ax.axhline(
                y_lane,
                color="#f5f5f5",
                lw=1.4 if is_outer else 1.0,
                ls="-" if is_outer else (0, (8, 7)),
                alpha=0.92 if is_outer else 0.78,
            )
        ax.set_ylim(-6.6, 6.6)

        writer = PillowWriter(fps=fps * playback_speed)
        with writer.saving(fig, str(out_path), dpi=100):
            for fi in tqdm(range(horizon), desc="Frames", unit="frame", leave=False):
                for artist in list(ax.patches) + list(ax.texts):
                    artist.remove()

                e_x = float(ego_x[list_idx, fi])
                e_y = float(ego_y[list_idx, fi])
                e_l = float(ego_len[list_idx])

                t_x = float(target_traj[list_idx, fi, 0])
                t_y = -float(target_traj[list_idx, fi, 1])
                t_l = float(target_len[list_idx])

                center_x = (e_x + t_x) / 2.0
                ax.set_xlim(center_x - half_width, center_x + half_width)
                _draw_generated_background_traffic(
                    ax,
                    background=backgrounds[list_idx],
                    frame_offset=fi,
                    event_type=inferred_event_type,
                    generated_ego_x=e_x,
                    generated_ego_y=e_y,
                    generated_target_y=t_y,
                    target_initial_lateral_offset=float(initial[list_idx, 1, 1] - initial[list_idx, 0, 1]),
                    lane_width=lane_width,
                    xlim=(center_x - half_width, center_x + half_width),
                    neighbor_margin=20.0,
                    vehicle_width=vehicle_width,
                )

                ax.add_patch(
                    Rectangle(
                        (e_x - e_l / 2.0, e_y - vehicle_width / 2.0),
                        e_l, vehicle_width,
                        facecolor="#e31a1c", edgecolor="black", lw=0.6,
                        alpha=1.0, zorder=4,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (t_x - t_l / 2.0, t_y - vehicle_width / 2.0),
                        t_l, vehicle_width,
                        facecolor="#1f78b4", edgecolor="black", lw=0.6,
                        alpha=1.0, zorder=4,
                    )
                )

                ax.text(e_x, e_y + vehicle_width / 2.0 + 0.25, r"$\mathrm{Ego}$",
                        fontsize=8, color="black", zorder=5, fontfamily="Liberation Serif",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})
                ax.text(t_x, t_y + vehicle_width / 2.0 + 0.25, r"$\mathrm{Target}$",
                        fontsize=8, color="black", zorder=5, fontfamily="Liberation Serif",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

                trail_start = max(0, fi - trail_frames)
                ego_line.set_data(
                    ego_x[list_idx, trail_start : fi + 1],
                    ego_y[list_idx, trail_start : fi + 1],
                )
                target_line.set_data(
                    target_traj[list_idx, trail_start : fi + 1, 0],
                    -target_traj[list_idx, trail_start : fi + 1, 1],
                )

                title_obj.set_text(
                    _generated_scenario_title(
                        event_type=inferred_event_type,
                        global_idx=global_idx,
                        frame_index=fi,
                        dt=dt,
                        conditions=conditions[list_idx],
                        realized_conditions=realized_conditions[list_idx],
                        condition_keys=condition_keys,
                    )
                )

                writer.grab_frame()

        plt.close(fig)
        LOGGER.info("Saved %s", out_path)

    return output_paths


def _generated_event_type(data: Any, requested: str | None) -> str:
    if requested is not None:
        event_type = str(requested).lower()
    elif "event_type" in data.files:
        raw = data["event_type"]
        event_type = str(raw.item() if hasattr(raw, "item") else raw).lower()
    elif "target_trajectory" in data.files:
        event_type = "cut_in"
    elif "lead_trajectory" in data.files:
        event_type = "following"
    else:
        raise KeyError(
            "Generated scenario NPZ must contain event_type or one of "
            "target_trajectory/lead_trajectory"
        )
    if event_type not in {"cut_in", "following"}:
        raise ValueError(f"Unsupported generated scenario event_type: {event_type}")
    return event_type


def _load_generated_background_contexts(
    *,
    base_event_ids: list[str],
    conditions: np.ndarray,
    condition_keys: list[str],
    background_config_path: Path | None,
    event_type: str,
    dt: float,
) -> list[dict[str, Any] | None]:
    if background_config_path is None or not base_event_ids:
        return [None for _ in base_event_ids]
    try:
        cfg = load_config(background_config_path)
        events_path = resolve_data_path(cfg["paths"]["output_dir"], background_config_path) / "events.csv"
        events = pd.read_csv(events_path)
        if "is_valid" in events.columns:
            valid = events["is_valid"]
            if valid.dtype != bool:
                valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
            events = events[valid]
        by_event_id = {
            str(row["event_id"]): row
            for _, row in events[events["event_type"] == event_type].iterrows()
        }
        recording_cache = {}
        ttc_idx = (
            condition_keys.index("time_to_cross")
            if "time_to_cross" in condition_keys
            else None
        )
        out: list[dict[str, Any] | None] = []
        for row_idx, event_id in enumerate(base_event_ids):
            event = by_event_id.get(str(event_id))
            if event is None:
                out.append(None)
                continue
            rid = int(event["recording_id"])
            if rid not in recording_cache:
                recording_cache[rid] = _load_recording(cfg, background_config_path, rid)
            rec = recording_cache[rid]
            if event_type == "cut_in":
                anchor_frame = _safe_int(
                    event.get("cross_frame"),
                    _safe_int(event.get("anchor_frame"), _safe_int(event.get("start_frame"))),
                )
            else:
                anchor_frame = _safe_int(
                    event.get("anchor_frame"),
                    _safe_int(event.get("start_frame")),
                )
            if anchor_frame is None:
                out.append(None)
                continue
            pre_steps = (
                int(round(float(conditions[row_idx, ttc_idx]) / max(float(dt), 1.0e-6)))
                if event_type == "cut_in" and ttc_idx is not None
                else 0
            )
            bg_anchor_frame = int(anchor_frame) - max(pre_steps, 0)
            ego_id = int(event["ego_id"])
            target_id = int(event["target_id"])
            ego_track = rec.get_vehicle_track(ego_id)
            if ego_track.empty:
                out.append(None)
                continue
            if bg_anchor_frame not in ego_track.index:
                available = np.asarray(ego_track.index, dtype=np.int64)
                bg_anchor_frame = int(available[np.argmin(np.abs(available - bg_anchor_frame))])
            anchor_row = ego_track.loc[bg_anchor_frame]
            out.append(
                {
                    "recording": rec,
                    "anchor_frame": int(bg_anchor_frame),
                    "anchor_ego_x": float(anchor_row["x"]),
                    "anchor_ego_y": float(anchor_row["y"]),
                    "exclude_ids": {ego_id, target_id},
                }
            )
        return out
    except Exception as exc:  # pragma: no cover - visualization fallback.
        LOGGER.warning("Could not load generated-scenario background traffic: %s", exc)
        return [None for _ in base_event_ids]


def _draw_generated_background_traffic(
    ax: Any,
    *,
    background: dict[str, Any] | None,
    frame_offset: int,
    event_type: str,
    generated_ego_x: float,
    generated_ego_y: float,
    generated_target_y: float,
    target_initial_lateral_offset: float,
    lane_width: float,
    xlim: tuple[float, float],
    neighbor_margin: float,
    vehicle_width: float,
) -> None:
    if background is None:
        return
    rec = background["recording"]
    frame = int(background["anchor_frame"]) + int(frame_offset)
    frame_df = rec.get_frame(frame)
    if frame_df.empty:
        return
    exclude_ids = set(background["exclude_ids"])
    for idx, row in frame_df.iterrows():
        vid = int(idx[0]) if isinstance(idx, tuple) else int(idx)
        if vid in exclude_ids:
            continue
        x = float(row["x"]) - float(background["anchor_ego_x"]) + generated_ego_x
        y = -(float(row["y"]) - float(background["anchor_ego_y"])) + generated_ego_y
        length = float(row.get("width", 4.5))
        width = float(row.get("height", vehicle_width))
        if x < xlim[0] - neighbor_margin or x > xlim[1] + neighbor_margin:
            continue
        if y < -7.5 or y > 7.5:
            continue
        blocked_lane_centers = [float(generated_ego_y), float(generated_target_y)]
        if event_type == "cut_in":
            target_source_display_y = (
                generated_ego_y
                - np.sign(float(target_initial_lateral_offset)) * float(lane_width)
            )
            blocked_lane_centers.append(float(target_source_display_y))
        if any(
            abs(y - center_y) <= 0.5 * float(lane_width)
            for center_y in blocked_lane_centers
        ):
            continue
        ax.add_patch(
            Rectangle(
                (x - length / 2.0, y - width / 2.0),
                length,
                width,
                facecolor="#bdbdbd",
                edgecolor="#4d4d4d",
                lw=0.45,
                alpha=0.50,
                zorder=1,
            )
        )


def _generated_scenario_title(
    *,
    event_type: str,
    global_idx: int,
    frame_index: int,
    dt: float,
    conditions: np.ndarray,
    realized_conditions: np.ndarray,
    condition_keys: list[str],
) -> str:
    def value(name: str, source: np.ndarray = conditions, default: float = float("nan")) -> float:
        if name not in condition_keys:
            return default
        return float(source[condition_keys.index(name)])

    gap = value("initial_gap")
    if event_type == "following":
        ego_vx = value("ego_vx_0")
        delta_v = value("initial_delta_v")
        lead_ax = value("lead_ax_0")
        lead_min_ax = value("lead_min_ax", realized_conditions)
        brake_duration = value("lead_braking_duration", realized_conditions)
        return (
            rf"Generated following #{global_idx:05d} | $t={frame_index * dt:.2f}\,\mathrm{{s}}$"
            "\n"
            rf"$g_0={gap:.1f}\,\mathrm{{m}}$   "
            rf"$v_{{\mathrm{{ego}},0}}={ego_vx:.1f}\,\mathrm{{m/s}}$   "
            rf"$\Delta v_0={delta_v:.1f}\,\mathrm{{m/s}}$   "
            rf"$a_{{\mathrm{{lead}},0}}={lead_ax:.1f}\,\mathrm{{m/s^2}}$   "
            rf"$a_{{\min}}={lead_min_ax:.1f}\,\mathrm{{m/s^2}}$   "
            rf"$T_{{\mathrm{{brake}}}}={brake_duration:.1f}\,\mathrm{{s}}$"
        )

    dy0 = value("initial_lateral_offset")
    t_cross = value("time_to_cross")
    dvx = value("initial_delta_vx")
    final_y = value("final_lateral_offset", realized_conditions)
    speed_change = value("target_speed_change", realized_conditions)
    return (
        rf"Generated cut-in #{global_idx:05d} | $t={frame_index * dt:.2f}\,\mathrm{{s}}$"
        "\n"
        rf"$g_0={gap:.1f}\,\mathrm{{m}}$   "
        rf"$d_{{y,0}}={dy0:.1f}\,\mathrm{{m}}$   "
        rf"$t_{{\mathrm{{cross}}}}={t_cross:.1f}\,\mathrm{{s}}$   "
        rf"$\Delta v_{{x,0}}={dvx:.1f}\,\mathrm{{m/s}}$   "
        rf"$y_{{\mathrm{{final}}}}={final_y:.2f}\,\mathrm{{m}}$   "
        rf"$\Delta v_{{\mathrm{{target}}}}={speed_change:.1f}\,\mathrm{{m/s}}$"
    )
