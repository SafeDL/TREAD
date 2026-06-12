"""Build car-following paper experiment artifacts from existing results.

This is a read-only post-processing script for existing experiment outputs.
It does not retrain models, refit EVT models, or rerun subset simulation.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.paper_experiment_utils import (
    as_float,
    build_manifest,
    fget,
    fraction_true,
    gpd_survival,
    nested,
    read_csv_rows,
    read_json,
    record,
    rel_path,
    save_figure as save_figure_to,
    write_experiment_readme,
    write_json,
    write_table as write_csv_table,
)
from tools.plot_style import (
    CRITICAL_COLOR,
    GENERATED_COLOR,
    REAL_COLOR,
    REFERENCE_COLOR,
    SAMPLED_COLOR,
    get_pyplot,
    style_axes,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_experiments" / "following"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"

SOURCE_PATHS = {
    "event_scores": RESULTS / "highd_events" / "following_event_scores.csv",
    "event_cache_summary": RESULTS / "highd_events" / "following_event_cache_summary.json",
    "subset_summary": RESULTS / "subset_simulation_following" / "latent_subset_summary.json",
    "subset_level_stats": RESULTS / "subset_simulation_following" / "latent_subset_level_stats.csv",
    "subset_samples": RESULTS / "subset_simulation_following" / "latent_subset_samples.npz",
    "subset_score_histograms": RESULTS
    / "subset_simulation_following"
    / "figures"
    / "subset_score_histograms.png",
    "monte_carlo_summary": RESULTS / "monte_carlo_following" / "latent_monte_carlo_summary.json",
    "naturalness_summary": RESULTS / "diffusion_natural" / "following" / "naturalness_summary.json",
    "natural_ax_plot": RESULTS
    / "diffusion_natural"
    / "following"
    / "natural_prior_plots"
    / "ax_distribution_real_vs_generated.png",
    "natural_jerk_plot": RESULTS
    / "diffusion_natural"
    / "following"
    / "natural_prior_plots"
    / "jerk_distribution_real_vs_generated.png",
    "natural_interaction_plot": RESULTS
    / "diffusion_natural"
    / "following"
    / "natural_prior_plots"
    / "phase_space_gap_delta_v.png",
    "evt_model": RESULTS / "highd_following_tail" / "evt" / "longitudinal_peak_evt_model.json",
    "evt_summary": RESULTS / "highd_following_tail" / "evt" / "longitudinal_peak_evt_summary.json",
    "evt_return_level_distance": RESULTS
    / "highd_following_tail"
    / "exposure"
    / "figures"
    / "peak_evt_return_level_distance.png",
    "exposure_summary": RESULTS / "highd_following_tail" / "exposure" / "highd_exposure_summary.json",
}


def write_table(base: str, rows: list[dict[str, Any]], *, force: bool) -> list[str]:
    return write_csv_table(TABLES, ROOT, base, rows, force=force)


def rel(path: Path) -> str:
    return rel_path(path, ROOT)


def save_figure(fig: Any, path: Path, *, force: bool) -> list[str]:
    return save_figure_to(fig, path, ROOT, force=force)


def exp1(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    exposure = data["exposure"]
    event_cache = data["event_cache"]
    rows = read_csv_rows(SOURCE_PATHS["event_scores"])
    mileage = subset.get("mileage_return_period", {}) or {}
    xc = fget(exposure, "collision_critical_level", fget(subset, "evt_return_level_target"))
    y = as_float(rows, "y_long")
    row: dict[str, Any] = {
        "event_type": "following",
        "num_scored_following_events": len(rows) if rows else math.nan,
        "num_semantic_following_events": fget(event_cache, "num_following_contexts"),
        "num_independent_tail_peaks": fget(exposure, "num_independent_tail_peaks"),
        "primary_exposure_label": fget(mileage, "primary_exposure_label", "following ego"),
        "exposure_denominator": fget(exposure, "exposure_denominator", "following_ego_miles"),
        "primary_exposure_miles": fget(exposure, "following_ego_miles"),
        "primary_exposure_hours": fget(exposure, "following_ego_hours"),
        "all_vehicle_miles": fget(exposure, "all_vehicle_miles"),
        "all_vehicle_hours": fget(exposure, "all_vehicle_hours"),
        "following_ego_mile_fraction_of_all_vehicle_miles": fget(
            exposure, "ego_mile_fraction_of_all_vehicle"
        ),
        "tail_peak_rate_per_mile": fget(exposure, "tail_peak_rate_per_mile"),
        "tail_peak_rate_per_hour": fget(exposure, "tail_peak_rate_per_hour"),
        "collision_critical_level": xc,
        "evt_failure_threshold": fget(subset, "evt_failure_threshold"),
    }
    if y.size:
        row.update(
            {
                "y_long_mean": float(np.mean(y)),
                "y_long_std": float(np.std(y, ddof=1)),
                "y_long_p50": float(np.quantile(y, 0.50)),
                "y_long_p90": float(np.quantile(y, 0.90)),
                "y_long_p95": float(np.quantile(y, 0.95)),
                "y_long_p99": float(np.quantile(y, 0.99)),
                "y_long_max": float(np.max(y)),
                "empirical_exceedance_rate_at_xc": float(np.mean(y > float(xc))),
                "min_gap_p05": float(np.quantile(as_float(rows, "recorded_min_gap"), 0.05)),
                "min_ttc_p05": float(np.quantile(as_float(rows, "recorded_min_ttc"), 0.05)),
                "near_collision_rate": fraction_true(rows, "near_collision"),
                "collision_rate": fraction_true(rows, "collision"),
            }
        )
    outputs = write_table("exp1_following_event_exposure_stats", [row], force=force)
    if y.size:
        plt = get_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
        axes[0].hist(y, bins=60, color=REAL_COLOR, alpha=0.85)
        axes[0].axvline(float(xc), color=CRITICAL_COLOR, linestyle="--", label=r"$x_c$")
        axes[0].set_xlabel(r"$Y_{\mathrm{long}}$")
        axes[0].set_ylabel("Count")
        axes[0].legend(frameon=False)
        sorted_y = np.sort(y)
        ccdf = 1.0 - np.arange(1, sorted_y.size + 1) / sorted_y.size
        axes[1].plot(sorted_y, np.maximum(ccdf, 1.0 / sorted_y.size), color=REAL_COLOR)
        axes[1].axvline(float(xc), color=CRITICAL_COLOR, linestyle="--", label=r"$x_c$")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$Y_{\mathrm{long}}$")
        axes[1].set_ylabel("Empirical CCDF")
        axes[1].legend(frameon=False)
        for ax in axes:
            style_axes(ax)
        fig.tight_layout()
        outputs.extend(save_figure(fig, FIGURES / "exp1_following_y_long_hist_ccdf.png", force=force))
        plt.close(fig)
        record(manifest, "exp1_event_exposure_stats", status="generated", outputs=outputs)
    else:
        skip = {
            "status": "skipped",
            "reason": f"missing input file: {rel(SOURCE_PATHS['event_scores'])}",
        }
        log_path = LOGS / "exp1_skipped_y_distribution.json"
        write_json(log_path, skip, force=force)
        outputs.append(rel(log_path))
        record(manifest, "exp1_event_exposure_stats", status="generated", outputs=outputs)


def exp2(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    exposure = data["exposure"]
    evt_model = data["evt_model"]
    evt_summary = data["evt_summary"]
    mileage = subset.get("mileage_return_period", {}) or {}
    human = mileage.get("human_highd_reference", {}) or {}
    row = {
        "event_type": "following",
        "risk_variable": "Y_long",
        "pot_threshold_u": fget(evt_model, "u", fget(subset, "evt_model_u")),
        "shape_xi": fget(evt_model, "xi", fget(subset, "evt_model_xi")),
        "scale_beta": fget(evt_model, "beta", fget(subset, "evt_model_beta")),
        "exceedance_rate": fget(evt_model, "exceedance_rate", fget(subset, "evt_model_exceedance_rate")),
        "collision_critical_level_xc": fget(
            evt_summary,
            "collision_critical_level",
            fget(subset, "evt_return_level_target"),
        ),
        "evt_failure_threshold": fget(subset, "evt_failure_threshold"),
        "score_space": fget(subset, "score_space"),
        "human_safety_critical_intensity_per_mile": fget(
            human,
            "highd_safety_critical_intensity_per_mile",
            fget(exposure, "highd_safety_critical_intensity_per_mile"),
        ),
        "human_safety_critical_return_period_miles": fget(
            human,
            "highd_safety_critical_return_period_miles",
            fget(exposure, "highd_safety_critical_return_period_miles"),
        ),
        "human_safety_critical_intensity_per_hour": fget(
            human,
            "highd_safety_critical_intensity_per_hour",
            fget(exposure, "highd_safety_critical_intensity_per_hour"),
        ),
        "human_safety_critical_return_period_hours": fget(
            human,
            "highd_safety_critical_return_period_hours",
            fget(exposure, "highd_safety_critical_return_period_hours"),
        ),
    }
    outputs = write_table("exp2_following_evt_params", [row], force=force)
    u = float(row["pot_threshold_u"])
    xi = float(row["shape_xi"])
    beta = float(row["scale_beta"])
    lam = float(row["exceedance_rate"])
    xc = float(row["collision_critical_level_xc"])
    if all(np.isfinite([u, xi, beta, lam, xc])):
        y = np.linspace(u, max(xc * 1.25, u + 5.0), 400)
        surv = gpd_survival(y, u=u, xi=xi, beta=beta, exceedance_rate=lam)
        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        ax.plot(y, surv, color=REAL_COLOR, label="GPD tail survival")
        ax.axvline(u, color=REFERENCE_COLOR, linestyle=":", label="POT threshold u")
        ax.axvline(xc, color=CRITICAL_COLOR, linestyle="--", label=r"$x_c$")
        ax.set_yscale("log")
        ax.set_xlabel("Raw risk value y")
        ax.set_ylabel(r"$Pr(Y > y)$")
        ax.legend(frameon=False)
        style_axes(ax)
        fig.tight_layout()
        outputs.extend(save_figure(fig, FIGURES / "exp2_following_evt_survival_curve.png", force=force))
        plt.close(fig)
    else:
        skip = {"status": "skipped", "reason": "missing EVT model parameters"}
        path = LOGS / "exp2_skipped_survival_curve.json"
        write_json(path, skip, force=force)
        outputs.append(rel(path))
    if SOURCE_PATHS["evt_return_level_distance"].exists():
        status = "reused"
        notes = f"return-level curve reused existing artifact: {rel(SOURCE_PATHS['evt_return_level_distance'])}"
    else:
        skip = {"status": "skipped", "reason": f"missing input file: {rel(SOURCE_PATHS['evt_return_level_distance'])}"}
        path = LOGS / "exp2_skipped_return_level_curve.json"
        write_json(path, skip, force=force)
        outputs.append(rel(path))
        status = "generated"
        notes = "return-level curve skipped because existing exposure figure is unavailable"
    record(
        manifest,
        "exp2_evt_params_and_curves",
        status=status,
        outputs=outputs,
        sources=[rel(SOURCE_PATHS["evt_return_level_distance"])],
        notes=notes,
    )


def flatten_naturalness(summary: dict[str, Any]) -> dict[str, Any]:
    sections = summary.get("sections", {}) or {}
    row: dict[str, Any] = {
        "event_type": "following",
        "split": fget(summary, "split"),
        "num_samples": fget(summary, "num_samples"),
        "num_available_split_samples": fget(summary, "num_available_split_samples"),
        "sample_selection": fget(summary, "sample_selection"),
        "sampler": fget(summary, "sampler"),
        "action_representation": fget(summary, "action_representation"),
    }
    for section_name in (
        "validation",
        "action_distribution",
        "trajectory_naturalness",
        "interaction_naturalness",
        "physical_feasibility",
    ):
        section = sections.get(section_name, {}) or {}
        for key, value in section.items():
            if any(
                token in key
                for token in (
                    "val_",
                    "_wasserstein",
                    "_ks",
                    "action_clip_rate",
                    "speed_negative_rate",
                    "jerk_violation_rate",
                    "ax_violation_rate",
                    "trajectory_discontinuity_rate",
                )
            ):
                row[key] = value
    return row


def exp3(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    natural = data["natural"]
    outputs = write_table("exp3_following_naturalness_summary", [flatten_naturalness(natural)], force=force)
    sources = [
        rel(path)
        for path in (
            SOURCE_PATHS["natural_ax_plot"],
            SOURCE_PATHS["natural_jerk_plot"],
            SOURCE_PATHS["natural_interaction_plot"],
        )
        if path.exists()
    ]
    record(
        manifest,
        "exp3_naturalness",
        status="reused" if sources else "generated",
        outputs=outputs,
        sources=sources,
        notes="naturalness figures are referenced in their original folders; no duplicate images were copied",
    )


def exp4(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    counts = subset.get("simulation_counts", {}) or {}
    relobs = nested(subset, "reliability", "observed", default={})
    row = {
        "event_type": "following",
        "probability_target": fget(subset, "probability_target"),
        "failure_event": fget(subset, "failure_event"),
        "probability": fget(subset, "probability"),
        "probability_ci95_lower": fget(subset, "probability_ci95_lower"),
        "probability_ci95_upper": fget(subset, "probability_ci95_upper"),
        "relative_standard_error": fget(subset, "relative_standard_error"),
        "probability_estimate_kind": fget(subset, "probability_estimate_kind"),
        "strict_probability_interpretation": fget(subset, "strict_probability_interpretation"),
        "num_samples": fget(subset, "num_samples"),
        "num_levels": fget(subset, "num_levels"),
        "p0": fget(subset, "p0"),
        "proposal_std": fget(subset, "proposal_std"),
        "context_refresh_prob": fget(subset, "context_refresh_prob"),
        "mh_retries_per_sample": fget(subset, "mh_retries_per_sample"),
        "closed_loop_evaluations": fget(counts, "closed_loop_evaluations"),
        "proposal_evaluations": fget(counts, "proposal_evaluations"),
        "final_failure_fraction": fget(subset, "final_failure_fraction"),
        "stop_reason": fget(subset, "stop_reason"),
        "reliability_status": nested(subset, "reliability", "status"),
        "score_space": fget(subset, "score_space"),
        "transition_acceptance_rate": fget(relobs, "acceptance_rate"),
    }
    outputs = write_table("exp4_following_subset_main_results", [row], force=force)
    levels = subset.get("level_stats", []) or []
    if levels:
        plt = get_pyplot()
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        xs = [int(float(d["level"])) for d in levels]
        for key, label, color in [
            ("score_p50", "p50", REAL_COLOR),
            ("score_p90", "p90", GENERATED_COLOR),
            ("score_p95", "p95", SAMPLED_COLOR),
            ("score_max", "max", REFERENCE_COLOR),
        ]:
            ax.plot(xs, [float(d[key]) for d in levels], marker="o", label=label, color=color)
        ax.axhline(float(subset["failure_threshold"]), color=CRITICAL_COLOR, linestyle="--", label="Failure threshold")
        ax.set_xlabel("Subset level")
        ax.set_ylabel("EVT score")
        ax.set_xticks(xs)
        ax.legend(frameon=False)
        style_axes(ax)
        fig.tight_layout()
        outputs.extend(save_figure(fig, FIGURES / "exp4_following_level_score_shift.png", force=force))
        plt.close(fig)
    record(
        manifest,
        "exp4_subset_main_results",
        status="generated",
        outputs=outputs,
        sources=[rel(SOURCE_PATHS["subset_score_histograms"])],
        notes="subset score histogram is referenced in its original folder; no duplicate image was copied",
    )


def exp5(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    mileage = subset.get("mileage_return_period", {}) or {}
    global_exposure = subset.get("global_risk_exposure_comparison", {}) or {}
    row = {
        "event_type": "following",
        "risk_label": fget(mileage, "risk_label"),
        "exposure_denominator": fget(global_exposure, "exposure_denominator"),
        "total_all_vehicle_km": fget(global_exposure, "total_all_vehicle_km"),
        "tail_peak_rate_per_all_vehicle_km": fget(
            global_exposure, "tail_peak_rate_per_all_vehicle_km"
        ),
        "ads_exceedance_probability_conditional": fget(
            global_exposure, "tail_conditional_failure_probability"
        ),
        "ads_safety_critical_intensity_per_all_vehicle_km": fget(
            global_exposure, "ads_safety_critical_intensity_per_all_vehicle_km"
        ),
        "highd_safety_critical_intensity_per_all_vehicle_km": fget(
            global_exposure, "highd_safety_critical_intensity_per_all_vehicle_km"
        ),
        "ads_to_highd_intensity_ratio_per_all_vehicle_km": fget(
            global_exposure, "ads_to_highd_intensity_ratio_per_all_vehicle_km"
        ),
        "ads_safety_critical_return_period_all_vehicle_km": fget(
            global_exposure, "ads_safety_critical_return_period_all_vehicle_km"
        ),
        "highd_safety_critical_return_period_all_vehicle_km": fget(
            global_exposure, "highd_safety_critical_return_period_all_vehicle_km"
        ),
        "ads_return_period_over_highd_return_period_all_vehicle_km": fget(
            global_exposure,
            "ads_return_period_over_highd_return_period_all_vehicle_km",
        ),
    }
    outputs = write_table("exp5_following_ads_vs_highd_intensity", [row], force=force)
    plt = get_pyplot()
    for filename, ads_key, human_key, ylabel in [
        (
            "exp5_following_ads_vs_highd_intensity.png",
            "ads_safety_critical_intensity_per_all_vehicle_km",
            "highd_safety_critical_intensity_per_all_vehicle_km",
            "Safety-critical intensity per all-vehicle km",
        ),
        (
            "exp5_following_ads_vs_highd_return_period.png",
            "ads_safety_critical_return_period_all_vehicle_km",
            "highd_safety_critical_return_period_all_vehicle_km",
            "Return period (all-vehicle km)",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        vals = [float(row[ads_key]), float(row[human_key])]
        ax.bar(["ADS", "highD human baseline"], vals, color=[GENERATED_COLOR, REAL_COLOR])
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title("Car-following")
        style_axes(ax)
        fig.tight_layout()
        outputs.extend(save_figure(fig, FIGURES / filename, force=force))
        plt.close(fig)
    record(
        manifest,
        "exp5_ads_vs_highd_intensity",
        status="generated",
        outputs=outputs,
        notes="ADS vs highD comparison uses the all_vehicle_km denominator",
    )


def exp6(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    mc = data["mc"]
    levels = subset.get("level_stats", []) or []
    rows: list[dict[str, Any]] = []
    if levels:
        level0 = levels[0]
        rows.append(
            {
                "method": "naive_mc_level0_under_tail_context_distribution",
                "available": True,
                "probability_or_failure_fraction": fget(level0, "failure_fraction"),
                "relative_standard_error": math.nan,
                "closed_loop_evaluations": fget(level0, "num_samples"),
                "proposal_evaluations": 0,
                "num_samples": fget(level0, "num_samples"),
                "num_levels": 1,
                "score_mean": fget(level0, "score_mean"),
                "score_p90": fget(level0, "score_p90"),
                "score_p95": fget(level0, "score_p95"),
                "score_max": fget(level0, "score_max"),
                "notes": "direct level-0 sampling under highD tail context distribution",
            }
        )
    rows.append(
        {
            "method": "latent_subset",
            "available": True,
            "probability_or_failure_fraction": fget(subset, "probability"),
            "relative_standard_error": fget(subset, "relative_standard_error"),
            "closed_loop_evaluations": nested(subset, "simulation_counts", "closed_loop_evaluations"),
            "proposal_evaluations": nested(subset, "simulation_counts", "proposal_evaluations"),
            "num_samples": fget(subset, "num_samples"),
            "num_levels": fget(subset, "num_levels"),
            "score_mean": fget(levels[-1], "score_mean") if levels else math.nan,
            "score_p90": fget(levels[-1], "score_p90") if levels else math.nan,
            "score_p95": fget(levels[-1], "score_p95") if levels else math.nan,
            "score_max": fget(levels[-1], "score_max") if levels else math.nan,
            "notes": "standard subset estimate",
        }
    )
    if mc:
        rows.append(
            {
                "method": "independent_monte_carlo_under_tail_context_distribution",
                "available": True,
                "probability_or_failure_fraction": fget(mc, "probability"),
                "relative_standard_error": float(mc["probability_standard_error"]) / float(mc["probability"]),
                "closed_loop_evaluations": nested(mc, "simulation_counts", "closed_loop_evaluations"),
                "proposal_evaluations": 0,
                "num_samples": fget(mc, "num_samples"),
                "num_levels": 1,
                "score_mean": nested(mc, "stats", "score_mean"),
                "score_p90": nested(mc, "stats", "score_p90"),
                "score_p95": nested(mc, "stats", "score_p95"),
                "score_max": nested(mc, "stats", "score_max"),
                "notes": "existing independent Monte Carlo result; not rerun",
            }
        )
    rows.extend(
        [
            {
                "method": "risk_tilted_sampling",
                "available": False,
                "probability_or_failure_fraction": math.nan,
                "relative_standard_error": math.nan,
                "closed_loop_evaluations": math.nan,
                "proposal_evaluations": math.nan,
                "num_samples": math.nan,
                "num_levels": math.nan,
                "score_mean": math.nan,
                "score_p90": math.nan,
                "score_p95": math.nan,
                "score_max": math.nan,
                "notes": "unavailable in current following outputs",
            },
            {
                "method": "empirical_tail_contexts_only",
                "available": False,
                "probability_or_failure_fraction": math.nan,
                "relative_standard_error": math.nan,
                "closed_loop_evaluations": math.nan,
                "proposal_evaluations": math.nan,
                "num_samples": math.nan,
                "num_levels": math.nan,
                "score_mean": math.nan,
                "score_p90": math.nan,
                "score_p95": math.nan,
                "score_max": math.nan,
                "notes": "unavailable in current following outputs",
            },
        ]
    )
    outputs = write_table("exp6_following_sampling_ablation", rows, force=force)
    avail = [r for r in rows if r["available"]]
    if len(avail) >= 2:
        plt = get_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        labels = [str(r["method"]).replace("_under_tail_context_distribution", "") for r in avail]
        xs = np.arange(len(avail))
        axes[0].bar(xs, [float(r["closed_loop_evaluations"]) for r in avail], color=REAL_COLOR)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Closed-loop evaluations")
        axes[1].bar(xs, [float(r["probability_or_failure_fraction"]) for r in avail], color=GENERATED_COLOR)
        axes[1].set_ylabel("Estimated probability / failure fraction")
        for ax in axes:
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            style_axes(ax)
        fig.tight_layout()
        outputs.extend(save_figure(fig, FIGURES / "exp6_following_sampling_efficiency.png", force=force))
        plt.close(fig)
    record(manifest, "exp6_sampling_ablation", status="generated", outputs=outputs)


def exp7(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    global_exposure = subset.get("global_risk_exposure_comparison", {}) or {}
    ads_highd_ratio = fget(
        global_exposure,
        "ads_to_highd_intensity_ratio_per_all_vehicle_km",
    )
    rows = [
        {
            "target_name": "raw_y_long_threshold",
            "mathematical_definition": "Y_long_sim > x_c",
            "requires_evt_model": False,
            "can_map_to_human_return_period": True,
            "available_in_current_outputs": True,
            "estimated_probability_if_available": fget(subset, "probability"),
            "ads_highd_intensity_ratio_if_available": ads_highd_ratio,
            "notes": "raw event at the EVT-calibrated collision-critical level",
        },
        {
            "target_name": "evt_severity_threshold",
            "mathematical_definition": "S_EVT(Y_long_sim) > S_EVT(x_c)",
            "requires_evt_model": True,
            "can_map_to_human_return_period": True,
            "available_in_current_outputs": True,
            "estimated_probability_if_available": fget(subset, "probability"),
            "ads_highd_intensity_ratio_if_available": ads_highd_ratio,
            "notes": "monotone-equivalent ordering to Y_long_sim > x_c under the fitted GPD score",
        },
        {
            "target_name": "simple_surrogate_threshold",
            "mathematical_definition": (
                "min_ttc < tau_ttc or min_gap < tau_gap or hard_brake = 1 or max_drac > tau_drac"
            ),
            "requires_evt_model": False,
            "can_map_to_human_return_period": False,
            "available_in_current_outputs": False,
            "estimated_probability_if_available": math.nan,
            "ads_highd_intensity_ratio_if_available": math.nan,
            "notes": (
                "sample-level min_ttc/min_gap exist, but no calibrated following surrogate threshold "
                "is defined in current outputs"
            ),
        },
    ]
    outputs = write_table("exp7_following_risk_target_ablation", rows, force=force)
    record(
        manifest,
        "exp7_risk_target_ablation",
        status="generated",
        outputs=outputs,
        notes="TTC/THW/DRAC-style thresholds are not treated as highD-calibrated return periods",
    )


def exp8(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    main_path = SOURCE_PATHS["subset_summary"].resolve()
    candidates = {
        str(Path(p).resolve())
        for pattern in [
            "results/subset_simulation_following*/latent_subset_summary.json",
            "results/**/following*normal*/latent_subset_summary.json",
            "results/**/following*empirical*/latent_subset_summary.json",
            "results/**/following*copula*/latent_subset_summary.json",
        ]
        for p in glob.glob(str(ROOT / pattern), recursive=True)
    }
    rows = [
        {
            "context_source": "highd_tail_scenario_condition_distribution",
            "summary_path": rel(main_path),
            "available": True,
            "probability_target": fget(subset, "probability_target"),
            "probability": fget(subset, "probability"),
            "ci95_lower": fget(subset, "probability_ci95_lower"),
            "ci95_upper": fget(subset, "probability_ci95_upper"),
            "num_samples": fget(subset, "num_samples"),
            "num_levels": fget(subset, "num_levels"),
            "unique_contexts_final": nested(subset, "simulation_counts", "unique_context_indices_final_level"),
            "largest_context_share": nested(subset, "reliability", "observed", "largest_context_share"),
            "strict_probability_interpretation": fget(subset, "strict_probability_interpretation"),
            "notes": f"context_sampling_mode = {fget(subset, 'context_sampling_mode')}",
        }
    ]
    for label in ("normal_context_distribution", "empirical_independent_tail_peaks"):
        rows.append(
            {
                "context_source": label,
                "summary_path": "",
                "available": False,
                "probability_target": "",
                "probability": math.nan,
                "ci95_lower": math.nan,
                "ci95_upper": math.nan,
                "num_samples": math.nan,
                "num_levels": math.nan,
                "unique_contexts_final": math.nan,
                "largest_context_share": math.nan,
                "strict_probability_interpretation": False,
                "notes": (
                    "no existing alternative following subset summary found; "
                    "requires a separate explicit config/run"
                ),
            }
        )
    alt = sorted(p for p in candidates if p != str(main_path))
    if alt:
        rows[1]["summary_path"] = "; ".join(rel(Path(p)) for p in alt)
        rows[1]["notes"] = "candidate alternative summaries found; inspect before interpretation"
    outputs = write_table("exp8_following_context_distribution_ablation", rows, force=force)
    record(manifest, "exp8_context_distribution_ablation", status="generated", outputs=outputs)


def exp9(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    subset = data["subset"]
    natural = data["natural"]
    rel = subset.get("reliability", {}) or {}
    observed = rel.get("observed", {}) or {}
    thresholds = rel.get("thresholds", {}) or {}
    physical = nested(natural, "sections", "physical_feasibility", default={})
    row = {
        "event_type": "following",
        "reliability_status": fget(rel, "status"),
        "assessed_level": fget(rel, "assessed_level"),
        "acceptance_rate": fget(observed, "acceptance_rate"),
        "unique_contexts": fget(observed, "unique_contexts"),
        "unique_states": fget(observed, "unique_states"),
        "largest_context_share": fget(observed, "largest_context_share"),
        "largest_state_share": fget(observed, "largest_state_share"),
        "min_unique_contexts_required": fget(thresholds, "min_unique_contexts"),
        "min_unique_states_required": fget(thresholds, "min_unique_states"),
        "max_largest_context_share_allowed": fget(thresholds, "max_largest_context_share"),
        "max_largest_state_share_allowed": fget(thresholds, "max_largest_state_share"),
        "closed_loop_evaluations": nested(subset, "simulation_counts", "closed_loop_evaluations"),
        "stored_level_samples": nested(subset, "simulation_counts", "stored_level_samples"),
        "physical_feasibility_proxy": "diffusion naturalness physical_feasibility",
        "action_clip_rate": fget(physical, "action_clip_rate"),
        "speed_negative_rate": fget(physical, "speed_negative_rate"),
        "trajectory_discontinuity_rate": fget(physical, "trajectory_discontinuity_rate"),
        "jerk_violation_rate": fget(physical, "jerk_violation_rate"),
        "ax_violation_rate": fget(physical, "ax_violation_rate"),
    }
    outputs = write_table("exp9_following_reliability_diagnostics", [row], force=force)
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].bar(
        ["Contexts", "States"],
        [row["unique_contexts"], row["unique_states"]],
        color=REAL_COLOR,
        label="Observed",
    )
    axes[0].scatter(
        ["Contexts", "States"],
        [row["min_unique_contexts_required"], row["min_unique_states_required"]],
        color=CRITICAL_COLOR,
        marker="_",
        s=300,
        label="Required minimum",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Unique count")
    axes[0].legend(frameon=False)
    axes[1].bar(
        ["Context share", "State share"],
        [row["largest_context_share"], row["largest_state_share"]],
        color=GENERATED_COLOR,
        label="Observed",
    )
    axes[1].scatter(
        ["Context share", "State share"],
        [row["max_largest_context_share_allowed"], row["max_largest_state_share_allowed"]],
        color=CRITICAL_COLOR,
        marker="_",
        s=300,
        label="Allowed maximum",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Largest share")
    axes[1].legend(frameon=False)
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    outputs.extend(save_figure(fig, FIGURES / "exp9_following_reliability_diagnostics.png", force=force))
    plt.close(fig)
    record(manifest, "exp9_reliability_diagnostics", status="generated", outputs=outputs)


def write_readme(manifest: dict[str, Any], *, force: bool) -> None:
    write_experiment_readme(
        OUT / "FOLLOWING_EXPERIMENT_README.md",
        manifest,
        title="Following Paper Experiments",
        description=(
            "This directory contains post-processed car-following paper artifacts built from existing results only."
        ),
        no_rerun_note="No following diffusion training, EVT fitting, or subset simulation rerun was performed.",
        interpretation_notes=[
            "Main exposure denominator is `following_ego_miles`.",
            "All-vehicle exposure values are reported only as background.",
            "ADS intensity is `conditional exceedance probability x highD tail peak exposure rate`.",
            (
                "The probabilities are conditional on the highD following tail scenario-condition distribution, "
                "not unconditional road crash rates."
            ),
        ],
        force=force,
    )


def build(force: bool) -> dict[str, Any]:
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    data = {
        "subset": read_json(SOURCE_PATHS["subset_summary"]),
        "mc": read_json(SOURCE_PATHS["monte_carlo_summary"]),
        "natural": read_json(SOURCE_PATHS["naturalness_summary"]),
        "evt_model": read_json(SOURCE_PATHS["evt_model"]),
        "evt_summary": read_json(SOURCE_PATHS["evt_summary"]),
        "exposure": read_json(SOURCE_PATHS["exposure_summary"]),
        "event_cache": read_json(SOURCE_PATHS["event_cache_summary"]),
    }
    manifest = build_manifest(
        "following",
        "tools/build_following_paper_experiments.py",
        ROOT,
        SOURCE_PATHS,
    )
    exp1(manifest, data, force=force)
    exp2(manifest, data, force=force)
    exp3(manifest, data, force=force)
    exp4(manifest, data, force=force)
    exp5(manifest, data, force=force)
    exp6(manifest, data, force=force)
    exp7(manifest, data, force=force)
    exp8(manifest, data, force=force)
    exp9(manifest, data, force=force)
    write_json(OUT / "following_experiment_manifest.json", manifest, force=True)
    write_readme(manifest, force=force)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing paper experiment artifacts")
    args = parser.parse_args()
    manifest = build(force=args.force)
    print(f"Wrote {rel(OUT / 'following_experiment_manifest.json')}")
    print(f"Experiments: {len(manifest['experiments'])}")


if __name__ == "__main__":
    main()
