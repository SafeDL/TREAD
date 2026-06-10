"""Shared plotting style for paper-oriented result figures."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


REAL_COLOR = "#4C78A8"
GENERATED_COLOR = "#F58518"
SAMPLED_COLOR = "#54A24B"
REFERENCE_COLOR = "#333333"
CRITICAL_COLOR = "#E45756"


def configure_matplotlib() -> Any:
    """Configure matplotlib for deterministic, serif, mathtext-ready figures."""
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "figure.titlesize": 12.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    return matplotlib


def get_pyplot() -> Any:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    return plt


def style_axes(ax: Any, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.7)


def label_for(name: str) -> str:
    labels = {
        "ego_vx_0": r"$v_{x,\mathrm{ego}}^0$ (m/s)",
        "log_initial_gap": r"$\log g_0$",
        "initial_gap": r"$g_0$ (m)",
        "initial_lateral_offset": r"$\Delta y_0$ (m)",
        "initial_delta_v": r"$\Delta v^0$ (m/s)",
        "initial_delta_vx": r"$\Delta v_x^0$ (m/s)",
        "target_ax_0": r"$a_{x,\mathrm{tar}}^0$ (m/s$^2$)",
        "target_vy_0": r"$v_{y,\mathrm{tar}}^0$ (m/s)",
        "target_ay_0": r"$a_{y,\mathrm{tar}}^0$ (m/s$^2$)",
        "lead_ax_0": r"$a_{x,\mathrm{lead}}^0$ (m/s$^2$)",
        "final_lateral_offset": r"$\Delta y_T$ (m)",
        "time_to_cross": r"$t_{\mathrm{cross}}$ (s)",
        "target_speed_change": r"$\Delta v_{\mathrm{tar}}$ (m/s)",
        "lead_speed_change": r"$\Delta v_{\mathrm{lead}}$ (m/s)",
        "lead_min_ax": r"$\min a_{x,\mathrm{lead}}$ (m/s$^2$)",
        "lead_braking_duration": r"$T_{\mathrm{brake}}$ (s)",
        "lead_final_speed": r"$v_{x,\mathrm{lead}}^T$ (m/s)",
        "lead_displacement": r"$\Delta x_{\mathrm{lead}}$ (m)",
        "lead_mean_abs_ax": r"$\overline{|a_{x,\mathrm{lead}}|}$ (m/s$^2$)",
        "lead_accel_std": r"$\sigma(a_{x,\mathrm{lead}})$ (m/s$^2$)",
        "lead_braking_impulse": r"$I_{\mathrm{brake}}$ (m/s)",
        "lane_entry_time": r"$t_{\mathrm{entry}}$ (s)",
        "longitudinal_displacement": r"$\Delta x$ (m)",
        "total_lateral_displacement": r"$\Delta y$ (m)",
        "lateral_progress_toward_ego_lane": r"$\Delta y_{\mathrm{ego}}$ (m)",
        "final_abs_lateral_offset": r"$|\Delta y_T|$ (m)",
        "max_abs_longitudinal_accel": r"$\max |a_x|$ (m/s$^2$)",
        "max_abs_lateral_velocity": r"$\max |v_y|$ (m/s)",
        "mean_abs_lateral_accel": r"$\overline{|a_y|}$ (m/s$^2$)",
        "max_abs_jerk": r"$\max |j_x|$ (m/s$^3$)",
    }
    return labels.get(str(name), str(name).replace("_", " "))

