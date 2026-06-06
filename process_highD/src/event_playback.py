"""Shared highD tail-event GIF playback helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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


def select_tail_context_indices(
    count: int,
    selection: str | int | tuple[int, ...] | list[int],
    random_seed: int,
) -> list[int]:
    if isinstance(selection, str):
        if selection.lower() != "all":
            raise ValueError(
                "tail_context_selection string must be 'all', "
                f"got {selection!r}"
            )
        return list(range(count))
    if isinstance(selection, int):
        if selection <= 0:
            raise ValueError("tail_context_selection integer must be positive")
        rng = np.random.default_rng(int(random_seed))
        sample_size = min(int(selection), count)
        return sorted(
            int(idx)
            for idx in rng.choice(count, size=sample_size, replace=False)
        )
    indices = [int(idx) for idx in selection]
    if not indices:
        raise ValueError("tail_context_selection cannot be empty")
    return indices


def load_tail_context_events(
    *,
    events_path: Path,
    tail_contexts_path: Path,
    event_type: str,
    tail_context_selection: str | int | tuple[int, ...] | list[int],
    random_seed: int,
) -> pd.DataFrame:
    raw_tail = np.load(tail_contexts_path, allow_pickle=True)
    missing = sorted({"event_id", "recording_id"} - set(raw_tail.files))
    if missing:
        raise KeyError(f"{tail_contexts_path} is missing required arrays: {missing}")

    events = pd.read_csv(events_path)
    if "is_valid" in events.columns:
        valid = events["is_valid"]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        events = events[valid]
    events = events[events["event_type"] == event_type].copy()
    by_event_id = {str(row["event_id"]): row for _, row in events.iterrows()}

    selected: list[dict[str, Any]] = []
    missing_events: list[str] = []
    for idx in select_tail_context_indices(
        int(raw_tail["event_id"].shape[0]),
        tail_context_selection,
        random_seed,
    ):
        if idx < 0 or idx >= int(raw_tail["event_id"].shape[0]):
            raise IndexError(
                f"tail_context_index {idx} is outside "
                f"[0, {int(raw_tail['event_id'].shape[0]) - 1}] in {tail_contexts_path}"
            )
        event_id = str(_scalar_at(raw_tail["event_id"], idx))
        synthetic = (
            int(_scalar_at(raw_tail["synthetic_context"], idx))
            if "synthetic_context" in raw_tail
            else 0
        )
        base_event_id = (
            str(_scalar_at(raw_tail["base_event_id"], idx))
            if "base_event_id" in raw_tail
            and str(_scalar_at(raw_tail["base_event_id"], idx))
            else event_id
        )
        lookup_event_id = base_event_id if synthetic else event_id
        source = by_event_id.get(lookup_event_id)
        if source is None:
            missing_events.append(lookup_event_id)
            continue
        out = source.to_dict()
        out.update(
            {
                "tail_context_index": idx,
                "tail_context_event_id": event_id,
                "synthetic_context": synthetic,
                "source_type": (
                    str(_scalar_at(raw_tail["source_type"], idx))
                    if "source_type" in raw_tail
                    else ""
                ),
            }
        )
        selected.append(out)

    if missing_events:
        LOGGER.warning(
            "Tail contexts referenced %d events missing from %s; first examples: %s",
            len(missing_events),
            events_path,
            missing_events[:5],
        )
    if not selected:
        raise ValueError(
            f"No {event_type} tail context events from {tail_contexts_path} "
            f"could be matched in {events_path}"
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def render_tail_event_gif(
    *,
    config_path: Path,
    tail_contexts_path: Path,
    output_dir: Path,
    output_name: str,
    event_type: str,
    tail_context_selection: str | int | tuple[int, ...] | list[int],
    random_seed: int,
    pre_frames: int = 0,
    post_frames: int = 0,
    view_width: float = 160.0,
    neighbor_margin: float = 20.0,
    trail_frames: int = 50,
    playback_speed: float = 1.0,
) -> Path:
    cfg = load_config(config_path)
    events_path = resolve_data_path(cfg["paths"]["output_dir"], config_path) / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"events.csv not found: {events_path}")
    if not tail_contexts_path.exists():
        raise FileNotFoundError(f"tail contexts npz not found: {tail_contexts_path}")

    events = load_tail_context_events(
        events_path=events_path,
        tail_contexts_path=tail_contexts_path,
        event_type=event_type,
        tail_context_selection=tail_context_selection,
        random_seed=random_seed,
    )
    LOGGER.info(
        "Loaded %d %s events from tail contexts selection=%r in %s",
        len(events),
        event_type,
        tail_context_selection,
        tail_contexts_path,
    )

    recording_cache = {}
    for rid in sorted(events["recording_id"].unique()):
        rid = int(rid)
        LOGGER.info("Loading recording %02d ...", rid)
        recording_cache[rid] = _load_recording(cfg, config_path, rid)

    frame_list = _build_frame_list(
        events,
        recording_cache,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )
    if not frame_list:
        raise RuntimeError("No frames to render")
    LOGGER.info("Total frames to render: %d", len(frame_list))

    output_path = output_dir / f"{output_name}.gif"
    ensure_dir(output_dir)
    LOGGER.info("Saving %s ...", output_path)
    _render_to_gif(
        frame_list,
        output_path,
        view_width=view_width,
        neighbor_margin=neighbor_margin,
        trail_frames=trail_frames,
        playback_speed=playback_speed,
    )
    LOGGER.info("Saved to %s", output_path)
    return output_path


def _scalar_at(array: np.ndarray, idx: int):
    value = array[idx]
    return value.item() if hasattr(value, "item") else value


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


def _frame_sequence(recording, event: pd.Series, pre: int, post: int) -> list[int]:
    start = _safe_int(event.get("start_frame")) - pre
    end = _safe_int(event.get("end_frame")) + post
    available = np.asarray(recording.frame_ids(), dtype=int)
    frames = available[(available >= start) & (available <= end)]
    if len(frames) == 0:
        raise ValueError(f"No frames in window [{start}, {end}].")
    return frames.astype(int).tolist()


def _track_centers(track: pd.DataFrame, frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows = track.loc[track.index.intersection(frames)]
    if rows.empty:
        return np.array([]), np.array([])
    return rows["x"].to_numpy(float), -rows["y"].to_numpy(float)


def _lane_groups(recording) -> list[np.ndarray]:
    groups = []
    for key in ["upperLaneMarkings", "lowerLaneMarkings"]:
        vals = np.asarray(recording.recording_meta.get(key, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            groups.append(vals)
    return groups


def _row_box(row: pd.Series) -> tuple[float, float, float, float]:
    w, h = float(row["width"]), float(row["height"])
    return float(row["x"]) - w / 2.0, -float(row["y"]) - h / 2.0, w, h


def _vehicle_style(vid: int, ego_id: int, target_id: int) -> tuple[str, float, int]:
    if vid == ego_id:
        return "#e31a1c", 1.0, 4
    if vid == target_id:
        return "#1f78b4", 1.0, 4
    return "#d9d9d9", 0.55, 1


def _frame_title(event: pd.Series, frame: int, fps: float) -> str:
    anchor = _safe_int(event.get("anchor_frame"))
    cross = _safe_int(event.get("cross_frame"))
    marks = [f"frame={frame}", f"t={(frame - int(event['start_frame'])) / fps:.2f}s"]
    if anchor is not None:
        marks.append(f"anchor={anchor}")
    if cross is not None:
        marks.append(f"cross={cross}")
    tail_parts = []
    if "tail_context_index" in event and pd.notna(event["tail_context_index"]):
        tail_parts.append(f"tail_idx={int(event['tail_context_index'])}")
        if int(event.get("synthetic_context", 0)):
            tail_parts.append(f"synthetic={event.get('tail_context_event_id', '')}")
        elif event.get("source_type", ""):
            tail_parts.append(str(event.get("source_type", "")))
    tail_text = f" | {', '.join(tail_parts)}" if tail_parts else ""
    return (
        f"{event['event_id']} ({event['event_type']}){tail_text} | "
        f"ego={int(event['ego_id'])}, target={int(event['target_id'])}\n"
        + ", ".join(marks)
    )


def _build_frame_list(
    events_df: pd.DataFrame,
    recording_cache: dict,
    *,
    pre_frames: int,
    post_frames: int,
) -> list:
    rows = []
    for _, event in events_df.iterrows():
        rec = recording_cache[int(event["recording_id"])]
        try:
            frames = _frame_sequence(rec, event, pre_frames, post_frames)
            for fi, frame in enumerate(frames):
                rows.append((rec, event, frame, fi, frames))
        except ValueError as exc:
            LOGGER.warning("Skipping event %s: %s", event["event_id"], exc)
    return rows


def _render_to_gif(
    frame_list: list,
    output_path: Path,
    *,
    view_width: float,
    neighbor_margin: float,
    trail_frames: int,
    playback_speed: float,
) -> None:
    from matplotlib.animation import PillowWriter
    from tqdm import tqdm

    track_cache: dict[tuple, pd.DataFrame] = {}
    for rec, event, *_ in frame_list:
        rid = rec.recording_id
        for vid in (int(event["ego_id"]), int(event["target_id"])):
            if (rid, vid) not in track_cache:
                track_cache[(rid, vid)] = rec.get_vehicle_track(vid)

    fps = float(frame_list[0][0].recording_meta.get("frameRate", 25))
    half_width = view_width / 2.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#707070")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("mirrored y (m)")
    title_obj = ax.set_title("")
    ego_line, = ax.plot([], [], color="#e31a1c", lw=1.8, alpha=0.9, zorder=3)
    target_line, = ax.plot([], [], color="#1f78b4", lw=1.8, alpha=0.9, zorder=3)
    lane_artists: list = []
    frame_artists: list = []
    current_rid: list[int | None] = [None]

    def _refresh_lanes(rec):
        for artist in lane_artists:
            artist.remove()
        lane_artists.clear()
        groups = _lane_groups(rec)
        markings = np.concatenate(groups) if groups else np.array([])
        ax.set_ylim(
            (-float(np.nanmax(markings)) - 2.0, -float(np.nanmin(markings)) + 2.0)
            if len(markings)
            else (-20.0, 5.0)
        )
        for group in groups:
            for j, y in enumerate(group):
                is_boundary = j == 0 or j == len(group) - 1
                lane_artists.append(
                    ax.axhline(
                        -float(y),
                        color="white",
                        lw=1.2 if is_boundary else 0.8,
                        ls="-" if is_boundary else "--",
                        alpha=0.9 if is_boundary else 0.65,
                    )
                )

    writer = PillowWriter(fps=fps * playback_speed)
    with writer.saving(fig, str(output_path), dpi=100):
        for rec, event, frame, fi, event_frames in tqdm(
            frame_list,
            desc="Rendering",
            unit="frame",
        ):
            ego_id = int(event["ego_id"])
            target_id = int(event["target_id"])
            rid = rec.recording_id
            if rid != current_rid[0]:
                _refresh_lanes(rec)
                current_rid[0] = rid

            for artist in frame_artists:
                artist.remove()
            frame_artists.clear()

            ego_track = track_cache[(rid, ego_id)]
            target_track = track_cache[(rid, target_id)]
            if frame in ego_track.index:
                center_x = float(ego_track.loc[frame, "x"])
            elif frame in target_track.index:
                center_x = float(target_track.loc[frame, "x"])
            else:
                center_x = view_width / 2.0
            xlim = (center_x - half_width, center_x + half_width)
            ax.set_xlim(*xlim)

            frame_df = rec.get_frame(frame)
            if not frame_df.empty:
                visible = frame_df[
                    (frame_df["x"] >= xlim[0] - neighbor_margin)
                    & (frame_df["x"] <= xlim[1] + neighbor_margin)
                ]
                for idx, row in visible.iterrows():
                    vid = int(idx[0]) if isinstance(idx, tuple) else int(idx)
                    color, alpha, zorder = _vehicle_style(vid, ego_id, target_id)
                    x, y, w, h = _row_box(row)
                    frame_artists.append(
                        ax.add_patch(
                            Rectangle(
                                (x, y),
                                w,
                                h,
                                facecolor=color,
                                edgecolor="black",
                                lw=0.6,
                                alpha=alpha,
                                zorder=zorder,
                            )
                        )
                    )
                    if vid in {ego_id, target_id}:
                        label = "ego" if vid == ego_id else "target"
                        frame_artists.append(
                            ax.text(
                                x,
                                y + h + 0.25,
                                f"{label} {vid}",
                                fontsize=8,
                                color="black",
                                zorder=5,
                                bbox={
                                    "facecolor": "white",
                                    "alpha": 0.75,
                                    "edgecolor": "none",
                                    "pad": 1.5,
                                },
                            )
                        )

            trail = event_frames[max(0, fi - trail_frames): fi + 1]
            ego_line.set_data(*_track_centers(ego_track, trail))
            target_line.set_data(*_track_centers(target_track, trail))
            title_obj.set_text(_frame_title(event, frame, fps))
            writer.grab_frame()

    plt.close(fig)


def render_generated_scenarios_gif(
    *,
    generated_npz_path: Path,
    output_dir: Path,
    output_name: str,
    scenario_selection: str | int | tuple[int, ...] | list[int],
    random_seed: int,
    dt: float = 0.04,
    view_width: float = 160.0,
    trail_frames: int = 50,
    playback_speed: float = 1.0,
    fps: float = 25.0,
) -> list[Path]:
    """Render diffusion-generated cut-in scenarios to GIF files.

    Each selected scenario produces one GIF. The ego vehicle moves at
    constant longitudinal speed (the scenario assumption used during
    trajectory integration) while the target vehicle follows the
    diffusion-generated trajectory.
    """
    from matplotlib.animation import PillowWriter
    from tqdm import tqdm

    if not generated_npz_path.exists():
        raise FileNotFoundError(f"Generated scenarios not found: {generated_npz_path}")

    data = np.load(generated_npz_path, allow_pickle=True)
    num_scenarios = int(data["initial_states"].shape[0])
    horizon = int(data["target_trajectory"].shape[1])

    indices = select_tail_context_indices(
        num_scenarios, scenario_selection, random_seed,
    )
    LOGGER.info("Selected %d / %d generated scenarios: %s", len(indices), num_scenarios, indices)

    initial = data["initial_states"][indices].astype(np.float64)
    target_traj = data["target_trajectory"][indices].astype(np.float64)
    ego_len = data["ego_length"][indices].astype(np.float64)
    target_len = data["adv_length"][indices].astype(np.float64)
    conditions = data["scenario_conditions"][indices].astype(np.float64)
    source_types = data["source_type"][indices]
    base_event_ids = data["base_event_id"][indices]
    condition_keys: list[str] = data["condition_keys"].tolist()

    # Ego: constant longitudinal speed (aligned with trajectory integration).
    ego0 = initial[:, 0]
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
        LOGGER.info("Rendering scenario %d (global idx %d) → %s", list_idx, global_idx, out_path)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_facecolor("#707070")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("mirrored y (m)")
        title_obj = ax.set_title("")

        ego_line, = ax.plot([], [], color="#e31a1c", lw=1.8, alpha=0.9, zorder=3)
        target_line, = ax.plot([], [], color="#1f78b4", lw=1.8, alpha=0.9, zorder=3)

        # Default lane markings for visual reference (3.75 m lane width).
        default_lanes = np.array([3.75, 0.0, -3.75, -7.5], dtype=float)
        for j, y_lane in enumerate(default_lanes):
            is_boundary = j == 0 or j == len(default_lanes) - 1
            ax.axhline(
                y_lane,
                color="white",
                lw=1.2 if is_boundary else 0.8,
                ls="-" if is_boundary else "--",
                alpha=0.9 if is_boundary else 0.65,
            )
        ax.set_ylim(-8.5, 5.5)

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

                ax.text(e_x, e_y + vehicle_width / 2.0 + 0.25, "ego",
                        fontsize=8, color="black", zorder=5,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})
                ax.text(t_x, t_y + vehicle_width / 2.0 + 0.25, "target",
                        fontsize=8, color="black", zorder=5,
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

                cond_str = ", ".join(
                    f"{key}={conditions[list_idx, k]:.2f}" for k, key in enumerate(condition_keys)
                )
                title_obj.set_text(
                    f"Scenario {global_idx} | source={source_types[list_idx]} | base={base_event_ids[list_idx]}\n"
                    f"t={fi * dt:.2f}s frame={fi} | {cond_str}"
                )

                writer.grab_frame()

        plt.close(fig)
        LOGGER.info("Saved %s", out_path)

    return output_paths
