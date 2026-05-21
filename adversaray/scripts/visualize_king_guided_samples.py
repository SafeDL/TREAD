#!/usr/bin/env python3
"""Plot diagnostics from saved prior/KING guided sample plans."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.rss import RSSConfig, rss_margin
from adversaray.src.torch_kinematics import integrate_following_actions_torch
from diffusion.src.utils import load_json, load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "king_guided_following.yaml"
SCRIPT_DEFAULTS = {
    "samples_name": "king_guided_samples.npz",
    "figure_dir": "figures",
    "max_cases": 0,
    "bins": 40,
    "dpi": 160,
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


def _schema(cfg: dict[str, Any], base: Path) -> dict[str, Any]:
    paths = cfg.get("paths", {})
    if "natural_dataset_dir" not in paths:
        raise KeyError("Config paths.natural_dataset_dir is required")
    schema_path = _resolve(paths["natural_dataset_dir"], base) / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    return load_json(schema_path)


def _series(
    actions: np.ndarray,
    context: np.ndarray,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    schema: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    action_t = torch.from_numpy(np.asarray(actions, dtype=np.float32))
    context_t = torch.from_numpy(np.asarray(context, dtype=np.float32))
    ego_t = torch.from_numpy(np.asarray(ego_length, dtype=np.float32))
    adv_t = torch.from_numpy(np.asarray(adv_length, dtype=np.float32))
    kin = integrate_following_actions_torch(action_t, context_t, ego_t, adv_t, schema, cfg)
    margin, _safe = rss_margin(kin, RSSConfig.from_config(cfg))
    closing = kin.ego_velocity - kin.velocity
    ttc = torch.where(
        closing > 1e-6,
        kin.gap / torch.clamp(closing, min=1e-6),
        torch.full_like(kin.gap, 1000.0),
    )
    lead0 = context_t[:, -1, 1, 0]
    ego0 = context_t[:, -1, 0, 0]
    lead_position = lead0[:, None] + kin.displacement
    ego_position = ego0[:, None] + kin.ego_displacement
    return {
        "jerk": kin.jerk.detach().cpu().numpy(),
        "acceleration": kin.acceleration.detach().cpu().numpy(),
        "ego_acceleration": kin.ego_acceleration.detach().cpu().numpy(),
        "lead_speed": kin.velocity.detach().cpu().numpy(),
        "ego_speed": kin.ego_velocity.detach().cpu().numpy(),
        "gap": kin.gap.detach().cpu().numpy(),
        "ttc": torch.clamp(ttc, 0.0, 1000.0).detach().cpu().numpy(),
        "rss_margin": margin.detach().cpu().numpy(),
        "lead_position": lead_position.detach().cpu().numpy(),
        "ego_position": ego_position.detach().cpu().numpy(),
    }


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _min_by_case(series: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = np.asarray(series[key], dtype=np.float32)
    if key == "ttc":
        values = np.clip(values, 0.0, 60.0)
    return np.min(values, axis=1)


def _mean_abs_by_case(series: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.mean(np.abs(np.asarray(series[key], dtype=np.float32)), axis=1)


def _hist_overlay(ax: Any, prior: np.ndarray, king: np.ndarray, title: str, xlabel: str, bins: int) -> None:
    prior_f = _finite(prior)
    king_f = _finite(king)
    if prior_f.size == 0 or king_f.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        return
    lo = float(min(np.min(prior_f), np.min(king_f)))
    hi = float(max(np.max(prior_f), np.max(king_f)))
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    ax.hist(prior_f, bins=edges, alpha=0.55, density=True, label="prior")
    ax.hist(king_f, bins=edges, alpha=0.55, density=True, label="king")
    ax.axvline(float(np.mean(prior_f)), color="C0", linewidth=1.4)
    ax.axvline(float(np.mean(king_f)), color="C1", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)


def _plot_summary_histograms(
    prior: dict[str, np.ndarray],
    king: dict[str, np.ndarray],
    out_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    panels = (
        ("min gap", _min_by_case(prior, "gap"), _min_by_case(king, "gap"), "min gap [m]"),
        ("min TTC", _min_by_case(prior, "ttc"), _min_by_case(king, "ttc"), "min TTC [s]"),
        ("min RSS margin", _min_by_case(prior, "rss_margin"), _min_by_case(king, "rss_margin"), "min RSS margin [m]"),
        ("mean |jerk|", _mean_abs_by_case(prior, "jerk"), _mean_abs_by_case(king, "jerk"), "mean |jerk| [m/s^3]"),
        (
            "mean |acceleration|",
            _mean_abs_by_case(prior, "acceleration"),
            _mean_abs_by_case(king, "acceleration"),
            "mean |lead acceleration| [m/s^2]",
        ),
        ("mean lead speed", np.mean(prior["lead_speed"], axis=1), np.mean(king["lead_speed"], axis=1), "mean lead speed [m/s]"),
        (
            "mean |ego acceleration|",
            _mean_abs_by_case(prior, "ego_acceleration"),
            _mean_abs_by_case(king, "ego_acceleration"),
            "mean |ego acceleration| [m/s^2]",
        ),
        ("mean ego speed", np.mean(prior["ego_speed"], axis=1), np.mean(king["ego_speed"], axis=1), "mean ego speed [m/s]"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 11.5))
    for ax, (title, prior_values, king_values, xlabel) in zip(axes.reshape(-1), panels, strict=True):
        _hist_overlay(ax, prior_values, king_values, title, xlabel, bins)
    axes[0, 0].legend(loc="best")
    fig.suptitle("KING-guided sample diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_case(
    case_id: int,
    dataset_index: int | None,
    prior: dict[str, np.ndarray],
    king: dict[str, np.ndarray],
    out_path: Path,
    *,
    dt: float,
    dpi: int,
) -> None:
    steps = np.arange(prior["jerk"].shape[0], dtype=np.float32)
    time = (steps + 1.0) * float(dt)
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), sharex=True)
    panels = (
        ("acceleration", "lead acceleration [m/s^2]"),
        ("jerk", "lead jerk [m/s^3]"),
        ("gap", "gap [m]"),
        ("ttc", "TTC [s]"),
        ("rss_margin", "RSS margin [m]"),
    )
    for ax, (key, ylabel) in zip(axes.reshape(-1)[:5], panels, strict=True):
        prior_y = np.asarray(prior[key], dtype=np.float32)
        king_y = np.asarray(king[key], dtype=np.float32)
        if key == "ttc":
            prior_y = np.clip(prior_y, 0.0, 60.0)
            king_y = np.clip(king_y, 0.0, 60.0)
        ax.plot(time, prior_y, label="prior", linewidth=1.8)
        ax.plot(time, king_y, label="king", linewidth=1.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if key in {"gap", "rss_margin"}:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)

    ax = axes.reshape(-1)[5]
    ax.plot(time, prior["ego_position"], color="C0", linestyle="--", linewidth=1.6, label="ego/prior")
    ax.plot(time, prior["lead_position"], color="C0", linewidth=1.8, label="lead/prior")
    ax.plot(time, king["ego_position"], color="C1", linestyle="--", linewidth=1.6, label="ego/king")
    ax.plot(time, king["lead_position"], color="C1", linewidth=1.8, label="lead/king")
    ax.set_ylabel("x position [m]")
    ax.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    axes[0, 0].legend(loc="best")
    ax.legend(loc="best")
    title = f"KING-guided case {case_id:04d}"
    if dataset_index is not None:
        title += f" / dataset_index={dataset_index}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _parse_case_indices(value: str) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _required_samples(samples: dict[str, np.ndarray]) -> None:
    required = {"context_states", "ego_length", "adv_length", "prior_actions", "king_actions"}
    missing = sorted(required - set(samples))
    if missing:
        raise KeyError(f"Samples file is missing required arrays: {missing}")


def main() -> None:
    setup_logging(SCRIPT_DEFAULTS["log_level"])

    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    output_root = _output_dir(cfg, base)
    samples_path = output_root / str(SCRIPT_DEFAULTS["samples_name"])
    figure_dir = output_root / str(SCRIPT_DEFAULTS["figure_dir"])
    if not samples_path.exists():
        raise FileNotFoundError(f"KING samples not found: {samples_path}")

    data = _load_npz(samples_path)
    _required_samples(data)
    schema = _schema(cfg, base)
    dt = float(schema.get("dt", cfg.get("sampling", {}).get("dt", 0.04)))
    total = int(data["context_states"].shape[0])
    max_cases = int(SCRIPT_DEFAULTS["max_cases"])
    num_cases = total if max_cases <= 0 else min(total, max_cases)
    if num_cases <= 0:
        raise RuntimeError("No samples selected for visualization")

    prior_series = _series(
        data["prior_actions"][:num_cases],
        data["context_states"][:num_cases],
        data["ego_length"][:num_cases],
        data["adv_length"][:num_cases],
        schema,
        cfg,
    )
    king_series = _series(
        data["king_actions"][:num_cases],
        data["context_states"][:num_cases],
        data["ego_length"][:num_cases],
        data["adv_length"][:num_cases],
        schema,
        cfg,
    )
    hist_path = figure_dir / "king_guided_sample_histograms.png"
    _plot_summary_histograms(prior_series, king_series, hist_path, bins=int(SCRIPT_DEFAULTS["bins"]), dpi=int(SCRIPT_DEFAULTS["dpi"]))

    for case_id in _parse_case_indices(str(SCRIPT_DEFAULTS.get("case_indices", ""))):
        if not 0 <= case_id < num_cases:
            raise IndexError(f"case id {case_id} outside loaded range [0, {num_cases - 1}]")
        prior_case = {key: value[case_id] for key, value in prior_series.items()}
        king_case = {key: value[case_id] for key, value in king_series.items()}
        dataset_index = int(data["dataset_index"][case_id]) if "dataset_index" in data else None
        _plot_case(
            case_id,
            dataset_index,
            prior_case,
            king_case,
            figure_dir / f"king_guided_case_{case_id:04d}.png",
            dt=dt,
            dpi=int(SCRIPT_DEFAULTS["dpi"]),
        )
    logger.info("Wrote KING-guided aggregate histogram to %s", hist_path)


if __name__ == "__main__":
    main()
