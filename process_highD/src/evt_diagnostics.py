"""EVT diagnostic plots shared by highD following and cut-in fitting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import genpareto

from tools.evt import RETURN_PERIODS, empirical_survival
from tools.plot_style import (
    CRITICAL_COLOR,
    GENERATED_COLOR,
    REAL_COLOR,
    REFERENCE_COLOR,
    get_pyplot,
    style_axes,
)


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


def _math_name(name: str) -> str:
    parts = str(name).split("_", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return rf"{parts[0]}_{{\mathrm{{{parts[1]}}}}}"
    return str(name)


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
    plt = get_pyplot()

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    risk_label = str(risk_variable)
    risk_math = _math_name(risk_label)
    risk_peak_math = rf"{risk_math},\mathrm{{peak}}"
    x_limit = float(np.quantile(values, 0.999))
    x_limit = max(x_limit, float(model.u) * 1.05, 1.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    clipped_values = values[values <= x_limit]
    _, bins, _ = ax.hist(
        clipped_values,
        bins=70,
        color=REAL_COLOR,
        alpha=0.62,
        label="Empirical peaks",
    )
    ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"Threshold $u$")
    tail_count = int(np.sum(values > float(model.u)))
    if tail_count > 0 and x_limit > float(model.u):
        bin_width = float(np.mean(np.diff(bins)))
        xs = np.linspace(float(model.u), x_limit, 320)
        tail_pdf = genpareto.pdf(
            xs - float(model.u),
            c=float(model.xi),
            scale=float(model.beta),
        )
        expected_tail_bin_count = tail_count * bin_width * tail_pdf
        ax.plot(
            xs,
            np.maximum(expected_tail_bin_count, 1.0e-12),
            color=GENERATED_COLOR,
            linewidth=2.2,
            label="GPD fit",
        )
    for period in RETURN_PERIODS:
        key = f"z{period}"
        if key in model.return_levels:
            z_value = float(model.return_levels[key])
            if z_value <= x_limit:
                ax.axvline(z_value, linestyle=":", label=rf"$z_{{{period}}}$")
    if float(collision_critical_level) <= x_limit:
        ax.axvline(
            float(collision_critical_level),
            color=CRITICAL_COLOR,
            linestyle="-.",
            label="Critical level",
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0.0, right=x_limit)
    ax.set_xlabel(fr"Peak ${risk_math}$")
    ax.set_ylabel("Count (log)")
    ax.set_title(fr"Peak ${risk_math}$ distribution")
    style_axes(ax)
    ax.legend(frameon=False)
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
                color=REAL_COLOR,
                label="Empirical",
                s=7,
                alpha=0.38,
            )
            ax.plot(
                sorted_values[mask],
                model_survival[mask],
                label="GPD",
                linewidth=2.0,
                color=GENERATED_COLOR,
            )
            ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"$u$")
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
            ax.set_xlabel(fr"Peak ${risk_math}$")
            ax.set_title(title)
            style_axes(ax)
        axes[0].set_ylabel(fr"$P({risk_peak_math} > y)$")
        axes[1].legend(frameon=False)
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
            ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", linewidth=1.3)
            ax.axvspan(
                float(model.u),
                float(np.max(u)),
                color=GENERATED_COLOR,
                alpha=0.08,
                label=r"$Y>u$",
            )
            style_axes(ax)
        axes[0].plot(u, xi, linewidth=1.4)
        axes[0].scatter([float(model.u)], [float(model.xi)], color=REFERENCE_COLOR, s=28)
        axes[0].set_ylabel(r"$\xi$")
        axes[0].set_title("Threshold stability")
        axes[1].plot(u, modified_scale, linewidth=1.4, color=REAL_COLOR)
        axes[1].set_ylabel(r"$\tilde{\sigma}$")
        axes[2].plot(u, exceedance_rate, linewidth=1.4, color=GENERATED_COLOR)
        axes[2].set_xlabel(r"Threshold $u$")
        axes[2].set_ylabel(r"$P(Y>u)$")
        fig.tight_layout()
        path = figure_dir / "peak_evt_threshold_stability.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_threshold_stability"] = str(path)

        mean_u, mean_excess = _mean_excess_rows(values, candidates)
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        ax.plot(mean_u, mean_excess, linewidth=1.5)
        ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"$u$")
        ax.axvspan(float(model.u), float(np.max(mean_u)), color=GENERATED_COLOR, alpha=0.08)
        ax.set_xlabel(r"Threshold $u$")
        ax.set_ylabel(r"$E[Y-u \mid Y>u]$")
        ax.set_title("Mean residual life")
        style_axes(ax)
        ax.legend(frameon=False)
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
        axes[0].plot([0.0, max_q], [0.0, max_q], color=REFERENCE_COLOR, linestyle="--")
        axes[0].set_xlabel("GPD quantile")
        axes[0].set_ylabel("Empirical quantile")
        axes[0].set_title("QQ plot")
        style_axes(axes[0])
        axes[1].scatter(empirical_cdf, gpd_cdf, s=8, alpha=0.40, color=GENERATED_COLOR)
        axes[1].plot([0.0, 1.0], [0.0, 1.0], color=REFERENCE_COLOR, linestyle="--")
        axes[1].set_xlabel("Empirical CDF")
        axes[1].set_ylabel("GPD CDF")
        axes[1].set_title("PP plot")
        style_axes(axes[1])
        path = figure_dir / "peak_evt_tail_fit_diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["peak_tail_fit_diagnostics"] = str(path)

    return paths
