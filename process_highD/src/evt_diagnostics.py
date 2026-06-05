"""EVT diagnostic plots shared by highD following and cut-in fitting."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import genpareto

from utils.evt import RETURN_PERIODS, empirical_survival


def _mean_excess_rows(
    values: np.ndarray,
    candidates: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    u_values = np.asarray([row["u"] for row in candidates], dtype=np.float64)
    means = []
    for u_value in u_values:
        excess = values[values > u_value] - u_value
        means.append(float(np.mean(excess)) if excess.size else float("nan"))
    return u_values, np.asarray(means, dtype=np.float64)


def write_evt_diagnostic_plots(
    figure_dir: Path,
    *,
    model: Any,
    values: np.ndarray,
    collision_critical_level: float,
    risk_variable: str = "Y_long",
    histogram_filename: str = "peak_evt_y_long_histogram.png",
    histogram_key: str = "peak_y_long_histogram",
) -> dict[str, str]:
    """Write standard EVT diagnostic plots without exposure-dependent plots."""
    cache_dir = Path(tempfile.gettempdir()) / "tread_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    risk_label = str(risk_variable)
    x_limit = float(np.quantile(values, 0.999))
    x_limit = max(x_limit, float(model.u) * 1.05, 1.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.hist(values[values <= x_limit], bins=70, color="tab:blue", alpha=0.72)
    ax.axvline(float(model.u), color="black", linestyle="--", label="POT threshold u")
    for period in RETURN_PERIODS:
        key = f"z{period}"
        if key in model.return_levels:
            z_value = float(model.return_levels[key])
            if z_value <= x_limit:
                ax.axvline(z_value, linestyle=":", label=key)
    if float(collision_critical_level) <= x_limit:
        ax.axvline(
            float(collision_critical_level),
            color="tab:red",
            linestyle="-.",
            label="critical level",
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0.0, right=x_limit)
    ax.set_xlabel(f"independent peak {risk_label}")
    ax.set_ylabel("peak count (log)")
    ax.set_title(
        f"highD independent peak {risk_label}, clipped at p99.9; "
        f"max={np.max(values):.2f}"
    )
    ax.legend()
    path = figure_dir / histogram_filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths[histogram_key] = str(path)

    sorted_values = np.sort(values[values >= float(model.u)])
    if sorted_values.size:
        empirical = empirical_survival(values, sorted_values)
        model_survival = model.survival(sorted_values)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        focus_max = float(
            max(
                float(model.return_levels.get("z100", model.return_level(100))) * 1.35,
                np.quantile(values, 0.995),
            )
        )
        panels = (
            (axes[0], focus_max, "return-level region"),
            (axes[1], float(np.max(values)), "full tail"),
        )
        for ax, right_limit, title in panels:
            mask = sorted_values <= right_limit
            ax.scatter(
                sorted_values[mask],
                empirical[mask],
                label="empirical survival",
                s=7,
                alpha=0.38,
            )
            ax.plot(
                sorted_values[mask],
                model_survival[mask],
                label="GPD tail survival",
                linewidth=2.0,
                color="tab:orange",
            )
            ax.axvline(float(model.u), color="black", linestyle="--", label="u")
            for period in RETURN_PERIODS:
                key = f"z{period}"
                if key not in model.return_levels:
                    continue
                z_value = float(model.return_levels[key])
                if z_value <= right_limit:
                    survival = float(model.survival(z_value))
                    ax.scatter([z_value], [survival], s=32)
                    ax.annotate(key, (z_value, survival))
            ax.set_yscale("log")
            ax.set_xlabel(f"independent peak {risk_label}")
            ax.set_title(title)
            ax.grid(True, alpha=0.22)
        axes[0].set_ylabel(f"P({risk_label}_peak > y)")
        axes[1].legend()
        path = figure_dir / "peak_evt_survival_fit.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_survival_fit"] = str(path)

    candidates = model.threshold_candidates
    if candidates:
        u = np.asarray([row["u"] for row in candidates], dtype=np.float64)
        xi = np.asarray([row["xi"] for row in candidates], dtype=np.float64)
        modified_scale = np.asarray(
            [row["modified_scale"] for row in candidates],
            dtype=np.float64,
        )
        exceedance_rate = np.asarray(
            [row["exceedance_rate"] for row in candidates],
            dtype=np.float64,
        )
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
        for ax in axes:
            ax.axvline(float(model.u), color="black", linestyle="--", linewidth=1.3)
            ax.axvspan(
                float(model.u),
                float(np.max(u)),
                color="tab:green",
                alpha=0.08,
                label="more extreme than selected u",
            )
            ax.grid(True, alpha=0.22)
        axes[0].plot(u, xi, linewidth=1.4)
        axes[0].scatter([float(model.u)], [float(model.xi)], color="black", s=28)
        axes[0].set_ylabel("xi")
        axes[0].set_title("Peak EVT threshold stability")
        axes[1].plot(u, modified_scale, linewidth=1.4, color="tab:green")
        axes[1].set_ylabel("modified scale")
        axes[2].plot(u, exceedance_rate, linewidth=1.4, color="tab:orange")
        axes[2].set_xlabel("threshold u")
        axes[2].set_ylabel("P(peak > u)")
        fig.tight_layout()
        path = figure_dir / "peak_evt_threshold_stability.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_threshold_stability"] = str(path)

        mean_u, mean_excess = _mean_excess_rows(values, candidates)
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        ax.plot(mean_u, mean_excess, linewidth=1.5)
        ax.axvline(float(model.u), color="black", linestyle="--", label="u")
        ax.axvspan(float(model.u), float(np.max(mean_u)), color="tab:green", alpha=0.08)
        ax.set_xlabel("threshold u")
        ax.set_ylabel("mean excess E[Y-u | Y>u]")
        ax.set_title("Peak EVT mean residual life diagnostic")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = figure_dir / "peak_evt_mean_excess.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_mean_excess"] = str(path)

    excess = np.sort(values[values > float(model.u)] - float(model.u))
    if excess.size:
        empirical_cdf = np.arange(1, excess.size + 1, dtype=np.float64)
        empirical_cdf = empirical_cdf / (excess.size + 1.0)
        gpd_cdf = genpareto.cdf(
            excess,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        gpd_quantiles = genpareto.ppf(
            empirical_cdf,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
        max_q = float(max(np.max(excess), np.max(gpd_quantiles)))
        axes[0].scatter(gpd_quantiles, excess, s=8, alpha=0.40)
        axes[0].plot([0.0, max_q], [0.0, max_q], color="black", linestyle="--")
        axes[0].set_xlabel("GPD theoretical excess quantile")
        axes[0].set_ylabel("empirical excess quantile")
        axes[0].set_title("Peak tail QQ diagnostic")
        axes[0].grid(True, alpha=0.25)
        axes[1].scatter(empirical_cdf, gpd_cdf, s=8, alpha=0.40, color="tab:orange")
        axes[1].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
        axes[1].set_xlabel("empirical CDF of excess")
        axes[1].set_ylabel("GPD CDF of excess")
        axes[1].set_title("Peak tail PP diagnostic")
        axes[1].grid(True, alpha=0.25)
        path = figure_dir / "peak_evt_tail_fit_diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_tail_fit_diagnostics"] = str(path)

    return paths
