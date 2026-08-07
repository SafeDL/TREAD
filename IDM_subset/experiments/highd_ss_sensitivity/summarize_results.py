#!/usr/bin/env python3
"""Summarize the highD IDM SS OAT experiment into auditable tables and figures."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.experiments.highd_ss_sensitivity.sensitivity_spec import (
    EVENTS,
    GRID,
    RESULTS_ROOT,
)
from tools.plot_style import (
    CRITICAL_COLOR,
    GENERATED_COLOR,
    PAPER_ANNOTATION_FONTSIZE,
    PAPER_NOTE_BBOX,
    PAPER_PANEL_LABELSIZE,
    REAL_COLOR,
    REFERENCE_COLOR,
    configure_matplotlib,
    style_axes,
)


Z95 = 1.959963984540054
PARAMETER_LABELS = {
    "num_samples": "Population size N",
    "p0": r"Conditional probability $p_0$",
    "proposal_std": r"Latent proposal std. $\sigma$",
    "context_refresh_prob": r"Context refresh probability $r_c$",
}
PANEL_TITLES = {
    "num_samples": "Population size",
    "p0": "Conditional probability",
    "proposal_std": "Latent proposal scale",
    "context_refresh_prob": "Context refresh",
}
PANEL_XLABELS = {
    "num_samples": r"$N$",
    "p0": r"$p_0$",
    "proposal_std": r"$\sigma_p$",
    "context_refresh_prob": r"$q_c$",
}
PANEL_LETTERS = {
    "num_samples": "a",
    "p0": "b",
    "proposal_std": "c",
    "context_refresh_prob": "d",
}
FIGURE_FILENAMES = (
    "normalized_risk_estimation_error_vs_parameter.png",
    "closed_loop_evaluations_vs_parameter.png",
    "acceptance_and_diversity_vs_parameter.png",
)
LEGACY_FIGURE_FILENAMES = ("probability_vs_parameter.png",)
TABLE_FILENAMES = (
    "ss_sensitivity_seed_level_results.csv",
    "ss_sensitivity_setting_level_summary.csv",
    "ss_sensitivity_paper_conclusion_table.csv",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _frozen_oat_defaults(
    results_root: Path,
) -> dict[str, dict[str, float | int]]:
    """Load OAT baselines from the immutable experiment manifest."""
    manifest_path = results_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Cannot summarize frozen OAT results without the experiment manifest: "
            f"{manifest_path}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Frozen OAT experiment manifest is invalid JSON: {manifest_path}"
        ) from exc

    oat_grid = manifest.get("oat_grid")
    if not isinstance(oat_grid, dict):
        raise ValueError(
            "Frozen OAT experiment manifest is missing the 'oat_grid' section: "
            f"{manifest_path}"
        )

    defaults: dict[str, dict[str, float | int]] = {}
    for event_type in EVENTS:
        event_design = oat_grid.get(event_type)
        if not isinstance(event_design, dict):
            raise ValueError(
                "Frozen OAT experiment manifest is missing the design for "
                f"event '{event_type}': {manifest_path}"
            )
        raw_defaults = event_design.get("defaults")
        if not isinstance(raw_defaults, dict):
            raise ValueError(
                "Frozen OAT experiment manifest is missing default parameters for "
                f"event '{event_type}': {manifest_path}"
            )

        event_defaults: dict[str, float | int] = {}
        for parameter, values in GRID[event_type].items():
            if parameter not in raw_defaults:
                raise ValueError(
                    "Frozen OAT experiment manifest is missing the default for "
                    f"'{event_type}.{parameter}': {manifest_path}"
                )
            try:
                default = (
                    int(raw_defaults[parameter])
                    if parameter == "num_samples"
                    else float(raw_defaults[parameter])
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Frozen OAT experiment manifest has a non-numeric default for "
                    f"'{event_type}.{parameter}': {manifest_path}"
                ) from exc
            if not math.isfinite(float(default)):
                raise ValueError(
                    "Frozen OAT experiment manifest has a non-finite default for "
                    f"'{event_type}.{parameter}': {manifest_path}"
                )
            if default not in values:
                raise ValueError(
                    "Frozen OAT experiment manifest default is absent from its "
                    f"parameter grid for '{event_type}.{parameter}': {manifest_path}"
                )
            event_defaults[parameter] = default
        defaults[event_type] = event_defaults
    return defaults


def _stored_results_path(results_root: Path, stored_path: str) -> Path:
    """Resolve a portable path and tolerate legacy Windows CSV separators."""
    windows_path = PureWindowsPath(stored_path)
    posix_path = PurePosixPath(stored_path)
    if windows_path.drive or windows_path.root or posix_path.is_absolute():
        raise ValueError(
            "Result metadata must use a path relative to the results root: "
            f"{stored_path}"
        )
    return results_root.joinpath(*windows_path.parts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1.0 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2.0 * n)) / denominator
    half = Z95 * math.sqrt((p * (1.0 - p) + Z95 * Z95 / (4.0 * n)) / n) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _mc_reference(results_root: Path, event_type: str) -> dict[str, float | int | str]:
    path = results_root / "references" / event_type / "latent_monte_carlo_summary.json"
    if not path.exists():
        return {
            "status": "missing",
            "probability": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "num_samples": 0,
            "failure_count": 0,
        }
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    stats = dict(summary.get("stats", {}) or {})
    n = _int(summary.get("num_samples", stats.get("num_samples")))
    failures = _int(stats.get("failure_count"))
    lower, upper = _wilson_interval(failures, n)
    return {
        "status": "available",
        "probability": _float(summary.get("probability")),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "num_samples": n,
        "failure_count": failures,
        "summary_path": path.relative_to(results_root).as_posix(),
    }


def _acceptance(summary: dict[str, Any]) -> float:
    values = [_float(value) for value in summary.get("acceptance_rates", [])]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _seed_rows(
    plan_rows: list[dict[str, str]], results_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in plan_rows:
        row: dict[str, Any] = dict(plan)
        summary_path_text = plan.get("summary_path", "")
        summary_path = (
            _stored_results_path(results_root, summary_path_text)
            if summary_path_text
            else None
        )
        if (
            plan.get("execution_status") == "completed"
            and summary_path
            and summary_path.exists()
        ):
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            reliability = dict(summary.get("reliability", {}) or {})
            observed = dict(reliability.get("observed", {}) or {})
            counts = dict(summary.get("simulation_counts", {}) or {})
            row.update(
                {
                    "probability": _float(summary.get("probability")),
                    "binomial_rse": _float(summary.get("relative_standard_error")),
                    "binomial_ci95_lower": _float(
                        summary.get("probability_ci95_lower")
                    ),
                    "binomial_ci95_upper": _float(
                        summary.get("probability_ci95_upper")
                    ),
                    "num_levels": _int(summary.get("num_levels")),
                    "closed_loop_evaluations": _int(
                        counts.get("closed_loop_evaluations")
                    ),
                    "proposal_evaluations": _int(counts.get("proposal_evaluations")),
                    "mean_transition_acceptance_rate": _acceptance(summary),
                    "reliability_status_observed": str(
                        reliability.get("status", "missing")
                    ),
                    "final_unique_contexts": _float(observed.get("unique_contexts")),
                    "final_unique_states": _float(observed.get("unique_states")),
                    "final_largest_context_share": _float(
                        observed.get("largest_context_share")
                    ),
                    "final_largest_state_share": _float(
                        observed.get("largest_state_share")
                    ),
                    "failure_threshold": _float(summary.get("failure_threshold")),
                    "stop_reason": str(summary.get("stop_reason", "")),
                }
            )
        else:
            row.update(
                {
                    "probability": float("nan"),
                    "binomial_rse": float("nan"),
                    "binomial_ci95_lower": float("nan"),
                    "binomial_ci95_upper": float("nan"),
                    "num_levels": float("nan"),
                    "closed_loop_evaluations": float("nan"),
                    "proposal_evaluations": float("nan"),
                    "mean_transition_acceptance_rate": float("nan"),
                    "reliability_status_observed": "missing",
                    "final_unique_contexts": float("nan"),
                    "final_unique_states": float("nan"),
                    "final_largest_context_share": float("nan"),
                    "final_largest_state_share": float("nan"),
                    "failure_threshold": float("nan"),
                    "stop_reason": "",
                }
            )
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def _sum(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.sum(finite)) if finite else float("nan")


def _student_t_ci95(
    mean: float,
    sample_std: float,
    n: int,
) -> tuple[float, float]:
    if n < 2 or not math.isfinite(mean) or not math.isfinite(sample_std):
        return float("nan"), float("nan")
    half_width = float(student_t.ppf(0.975, df=n - 1)) * sample_std / math.sqrt(n)
    return mean - half_width, mean + half_width


def _setting_row(
    event_type: str,
    parameter: str,
    value: float | int,
    default: float | int,
    num_samples: int,
    records: list[dict[str, Any]],
    mc: dict[str, Any],
) -> dict[str, Any]:
    probabilities = [_float(record.get("probability")) for record in records]
    valid_probs = [value for value in probabilities if math.isfinite(value)]
    probability_mean = _mean(probabilities)
    probability_std = _std(probabilities)
    n_valid = len(valid_probs)
    ci_lower, ci_upper = _student_t_ci95(
        probability_mean,
        probability_std,
        n_valid,
    )
    mc_probability = _float(mc.get("probability"))
    relative_bias = (
        (probability_mean - mc_probability) / mc_probability
        if math.isfinite(probability_mean) and mc_probability > 0
        else float("nan")
    )
    expected_runs = 5 if value == default else 3
    reliability_passes = sum(
        record.get("quality_status") == "pass" for record in records
    )
    failed_records = [
        record for record in records if record.get("quality_status") != "pass"
    ]
    reasons = sorted(
        {
            str(record.get("failure_reason", "")).strip()
            for record in failed_records
            if str(record.get("failure_reason", "")).strip()
        }
    )
    return {
        "event_type": event_type,
        "varied_parameter": parameter,
        "parameter_value": value,
        "is_default_value": str(value == default).lower(),
        "num_samples": int(num_samples),
        "expected_seed_runs": expected_runs,
        "completed_seed_runs": n_valid,
        "reliability_pass_runs": reliability_passes,
        "all_completed_runs_reliability_pass": str(
            n_valid == expected_runs and reliability_passes == expected_runs
        ).lower(),
        "failure_or_missing_runs": expected_runs - reliability_passes,
        "failure_reasons": " | ".join(reasons),
        "probability_mean": probability_mean,
        "probability_std_across_seed": probability_std,
        "probability_ci95_across_seed_lower": ci_lower,
        "probability_ci95_across_seed_upper": ci_upper,
        "probability_ci95_method": "two-sided Student-t across independent seeds",
        "cross_seed_cov": (
            probability_std / probability_mean
            if probability_mean > 0 and math.isfinite(probability_std)
            else float("nan")
        ),
        "relative_bias_vs_mc_point": relative_bias,
        "mean_binomial_rse": _mean(
            [_float(record.get("binomial_rse")) for record in records]
        ),
        "mean_closed_loop_evaluations": _mean(
            [_float(record.get("closed_loop_evaluations")) for record in records]
        ),
        "total_closed_loop_evaluations": _sum(
            [_float(record.get("closed_loop_evaluations")) for record in records]
        ),
        "mean_proposal_evaluations": _mean(
            [_float(record.get("proposal_evaluations")) for record in records]
        ),
        "mean_num_levels": _mean(
            [_float(record.get("num_levels")) for record in records]
        ),
        "mean_transition_acceptance_rate": _mean(
            [
                _float(record.get("mean_transition_acceptance_rate"))
                for record in records
            ]
        ),
        "mean_final_unique_contexts": _mean(
            [_float(record.get("final_unique_contexts")) for record in records]
        ),
        "mean_final_unique_states": _mean(
            [_float(record.get("final_unique_states")) for record in records]
        ),
        "mean_final_largest_context_share": _mean(
            [_float(record.get("final_largest_context_share")) for record in records]
        ),
        "mean_final_largest_state_share": _mean(
            [_float(record.get("final_largest_state_share")) for record in records]
        ),
        "mc_probability": mc_probability,
        "mc_wilson_ci95_lower": _float(mc.get("ci95_lower")),
        "mc_wilson_ci95_upper": _float(mc.get("ci95_upper")),
        "mc_num_samples": _int(mc.get("num_samples")),
        "mc_failure_count": _int(mc.get("failure_count")),
    }


def _setting_rows(
    seed_rows: list[dict[str, Any]], results_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    defaults = _frozen_oat_defaults(results_root)
    for event_type in EVENTS:
        event_records = [
            row for row in seed_rows if row.get("event_type") == event_type
        ]
        mc = _mc_reference(results_root, event_type)
        default_records = [
            row for row in event_records if row.get("setting_id") == "default"
        ]
        for parameter, values in GRID[event_type].items():
            default = defaults[event_type][parameter]
            for value in values:
                num_samples = (
                    int(value)
                    if parameter == "num_samples"
                    else int(defaults[event_type]["num_samples"])
                )
                if value == default:
                    records = default_records
                else:
                    records = [
                        row
                        for row in event_records
                        if row.get("varied_parameter") == parameter
                        and _float(row.get("parameter_value")) == float(value)
                    ]
                rows.append(
                    _setting_row(
                        event_type,
                        parameter,
                        value,
                        default,
                        num_samples,
                        records,
                        mc,
                    )
                )
    return rows


def _default_conclusions(setting_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_type in EVENTS:
        default_rows = [
            row
            for row in setting_rows
            if row["event_type"] == event_type and row["is_default_value"] == "true"
        ]
        if not default_rows:
            continue
        default = default_rows[0]
        ss_lower = _float(default["probability_ci95_across_seed_lower"])
        ss_upper = _float(default["probability_ci95_across_seed_upper"])
        mc_lower = _float(default["mc_wilson_ci95_lower"])
        mc_upper = _float(default["mc_wilson_ci95_upper"])
        intervals_overlap = (
            math.isfinite(ss_lower)
            and math.isfinite(mc_lower)
            and max(ss_lower, mc_lower) <= min(ss_upper, mc_upper)
        )
        robust = (
            default["all_completed_runs_reliability_pass"] == "true"
            and intervals_overlap
        )
        rows.append(
            {
                "event_type": event_type,
                "default_ss_probability_mean": default["probability_mean"],
                "default_ss_ci95_across_seed_lower": ss_lower,
                "default_ss_ci95_across_seed_upper": ss_upper,
                "default_ss_reliability_all_five_pass": default[
                    "all_completed_runs_reliability_pass"
                ],
                "mc_probability": default["mc_probability"],
                "mc_wilson_ci95_lower": mc_lower,
                "mc_wilson_ci95_upper": mc_upper,
                "default_ss_and_mc_ci95_overlap": str(intervals_overlap).lower(),
                "robust_by_predeclared_rule": str(robust).lower(),
                "interpretation": "Robust only when all five default repeats pass reliability and the cross-seed SS 95% interval overlaps the MC Wilson 95% interval; MCMC dependence is assessed by independent seeds.",
            }
        )
    return rows


def _plot_series(
    ax: Any,
    rows: list[dict[str, Any]],
    event_type: str,
    parameter: str,
    *,
    metric: str,
    ylabel: str,
    mc_band: bool = False,
) -> None:
    selected = [
        row
        for row in rows
        if row["event_type"] == event_type and row["varied_parameter"] == parameter
    ]
    selected.sort(key=lambda row: _float(row["parameter_value"]))
    x = np.asarray([_float(row["parameter_value"]) for row in selected])
    y = np.asarray([_float(row[metric]) for row in selected])
    defaults = np.asarray([row["is_default_value"] == "true" for row in selected])
    ax.plot(x, y, marker="o", color="#4C78A8", linewidth=1.5, markersize=4.5)
    if np.any(defaults):
        ax.scatter(
            x[defaults],
            y[defaults],
            marker="*",
            s=92,
            color="#E45756",
            zorder=3,
            label="default",
        )
    failed_or_missing = np.asarray(
        [row["all_completed_runs_reliability_pass"] != "true" for row in selected]
    )
    if np.any(failed_or_missing):
        ax.scatter(
            x[failed_or_missing],
            np.full(np.sum(failed_or_missing), 0.03),
            marker="x",
            s=40,
            color="#E45756",
            zorder=4,
            label="failure/missing (bottom)",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    if mc_band:
        mc_lower = (
            _float(selected[0]["mc_wilson_ci95_lower"]) if selected else float("nan")
        )
        mc_upper = (
            _float(selected[0]["mc_wilson_ci95_upper"]) if selected else float("nan")
        )
        if math.isfinite(mc_lower) and math.isfinite(mc_upper):
            ax.axhspan(
                mc_lower, mc_upper, color="#54A24B", alpha=0.16, label="MC Wilson 95%"
            )
    ax.set_title(f"{event_type}: {PARAMETER_LABELS[parameter]}")
    ax.set_xlabel(PARAMETER_LABELS[parameter])
    ax.set_ylabel(ylabel)
    style_axes(ax)


def _mark_failed_or_missing(
    ax: Any,
    x: np.ndarray,
    setting_rows: list[dict[str, Any]],
) -> None:
    failed_or_missing = np.asarray(
        [row["all_completed_runs_reliability_pass"] != "true" for row in setting_rows]
    )
    if not np.any(failed_or_missing):
        return
    ax.scatter(
        x[failed_or_missing],
        np.full(np.sum(failed_or_missing), 0.03),
        marker="x",
        s=40,
        color="#E45756",
        zorder=4,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )


def _plot_normalized_risk_estimation_error(
    setting_rows: list[dict[str, Any]], figures_dir: Path
) -> None:
    """Plot comparable MC-normalized error and reliability status by factor."""
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    event_styles = {
        "following": {"color": REAL_COLOR, "marker": "o", "label": "Car-following"},
        "cutin": {"color": GENERATED_COLOR, "marker": "s", "label": "Cut-in"},
    }
    failure_color = CRITICAL_COLOR
    parameters = tuple(PARAMETER_LABELS)
    # A landscape canvas keeps each 2-by-2 panel comfortably wider than tall
    # when it is placed at the manuscript text width.
    fig, axes = plt.subplots(2, 2, figsize=(7.65, 5.95), sharey=True, squeeze=False)
    global_error_extent = 0.0

    for index, parameter in enumerate(parameters):
        ax = axes.flat[index]
        parameter_values = sorted(
            {
                _float(row["parameter_value"])
                for row in setting_rows
                if row["varied_parameter"] == parameter
                and math.isfinite(_float(row["parameter_value"]))
            }
        )
        x_positions = {value: position for position, value in enumerate(parameter_values)}
        for event_type in EVENTS:
            selected = [
                row
                for row in setting_rows
                if row["event_type"] == event_type
                and row["varied_parameter"] == parameter
            ]
            selected.sort(key=lambda row: _float(row["parameter_value"]))
            if not selected:
                continue

            parameter_x = np.asarray(
                [_float(row["parameter_value"]) for row in selected]
            )
            x = np.asarray([x_positions[value] for value in parameter_x])
            mc_probability = np.asarray(
                [_float(row["mc_probability"]) for row in selected]
            )
            ss_probability = np.asarray(
                [_float(row["probability_mean"]) for row in selected]
            )
            ss_std = np.asarray(
                [_float(row["probability_std_across_seed"]) for row in selected]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_error = np.abs(ss_probability - mc_probability) / mc_probability
                normalized_std = ss_std / mc_probability
            valid = (
                np.isfinite(x)
                & np.isfinite(normalized_error)
                & np.isfinite(normalized_std)
            )
            if not np.any(valid):
                continue
            global_error_extent = max(
                global_error_extent,
                float(np.max(normalized_error[valid] + normalized_std[valid])),
            )

            style = event_styles[event_type]
            ax.errorbar(
                x[valid],
                normalized_error[valid],
                yerr=normalized_std[valid],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.75,
                markersize=6.0,
                capsize=3.2,
                elinewidth=1.2,
                zorder=3,
            )
            default_values = np.asarray(
                [row["is_default_value"] == "true" for row in selected]
            )
            if np.any(default_values & valid):
                ax.scatter(
                    x[default_values & valid],
                    normalized_error[default_values & valid],
                    marker="*",
                    s=165,
                    color=REFERENCE_COLOR,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=5,
                )
                for default_index in np.flatnonzero(default_values & valid):
                    raw_value = parameter_x[default_index]
                    if parameter == "num_samples":
                        default_label = rf"$N$={int(raw_value):,}"
                    elif parameter == "p0":
                        default_label = rf"$p_0$={raw_value:.2f}"
                    elif parameter == "proposal_std":
                        default_label = rf"$\sigma_p$={raw_value:.2f}"
                    else:
                        default_label = rf"$q_c$={raw_value:.2f}"
                    annotation_offset = (4, 7) if event_type == "following" else (4, -16)
                    ax.annotate(
                        default_label,
                        (x[default_index], normalized_error[default_index]),
                        xytext=annotation_offset,
                        textcoords="offset points",
                        color=style["color"],
                        fontsize=0.92 * PAPER_ANNOTATION_FONTSIZE,
                        ha="left",
                        va="bottom" if event_type == "following" else "top",
                        bbox={**PAPER_NOTE_BBOX, "pad": 0.16},
                        zorder=6,
                    )
            failed = np.asarray(
                [
                    row["all_completed_runs_reliability_pass"] != "true"
                    for row in selected
                ]
            )
            for failed_x in x[failed & np.isfinite(x)]:
                ax.axvline(
                    failed_x,
                    color=failure_color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=1,
                )
        if parameter_values:
            ax.set_xticks(range(len(parameter_values)))
            ax.set_xticklabels(
                [
                    str(int(value))
                    if parameter == "num_samples"
                    else f"{value:.2f}"
                    for value in parameter_values
                ]
            )
        ax.set_title(PANEL_TITLES[parameter], pad=7)
        ax.text(
            -0.13,
            1.045,
            PANEL_LETTERS[parameter],
            transform=ax.transAxes,
            fontsize=PAPER_PANEL_LABELSIZE,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        ax.set_xlabel(PANEL_XLABELS[parameter])
        style_axes(ax)

    y_upper = max(0.1, 1.08 * global_error_extent)
    for ax in axes.flat:
        ax.set_ylim(0.0, y_upper)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=event_styles[event_type]["color"],
            marker=event_styles[event_type]["marker"],
            linewidth=1.75,
            markersize=6.0,
            label=event_styles[event_type]["label"],
        )
        for event_type in EVENTS
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLOR,
                marker="*",
                linestyle="None",
                markersize=10,
                label="OAT reference setting",
            ),
            Line2D(
                [0],
                [0],
                color=failure_color,
                linestyle="--",
                linewidth=1.2,
                label="Reliability failure",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.supylabel(
        "Normalized MC-reference deviation",
        # Keep the shared label close to the left-hand axes instead of leaving
        # a visually detached gutter along the figure edge.
        x=0.095,
        fontsize=14.5,
    )
    fig.tight_layout(
        rect=(0.065, 0.0, 1.0, 0.92),
        h_pad=0.95,
        w_pad=0.80,
    )
    fig.savefig(
        figures_dir / "normalized_risk_estimation_error_vs_parameter.png", dpi=300
    )
    plt.close(fig)


def _figures(setting_rows: list[dict[str, Any]], results_root: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_FIGURE_FILENAMES:
        legacy_path = figures_dir / filename
        if legacy_path.is_file():
            legacy_path.unlink()
    _plot_normalized_risk_estimation_error(setting_rows, figures_dir)
    parameters = tuple(PARAMETER_LABELS)
    figure_specs = (
        (
            "closed_loop_evaluations_vs_parameter.png",
            "mean_closed_loop_evaluations",
            "Mean closed-loop evaluations",
            False,
        ),
    )
    for filename, metric, ylabel, mc_band in figure_specs:
        fig, axes = plt.subplots(
            len(parameters), len(EVENTS), figsize=(13.8, 13.8), squeeze=False
        )
        for row_idx, parameter in enumerate(parameters):
            for col_idx, event_type in enumerate(EVENTS):
                _plot_series(
                    axes[row_idx, col_idx],
                    setting_rows,
                    event_type,
                    parameter,
                    metric=metric,
                    ylabel=ylabel,
                    mc_band=mc_band,
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center", ncol=len(handles), frameon=False
            )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figures_dir / filename, dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(
        len(parameters), len(EVENTS), figsize=(13.8, 13.8), squeeze=False
    )
    for row_idx, parameter in enumerate(parameters):
        for col_idx, event_type in enumerate(EVENTS):
            ax = axes[row_idx, col_idx]
            selected = [
                row
                for row in setting_rows
                if row["event_type"] == event_type
                and row["varied_parameter"] == parameter
            ]
            selected.sort(key=lambda row: _float(row["parameter_value"]))
            x = np.asarray([_float(row["parameter_value"]) for row in selected])
            acceptance = np.asarray(
                [_float(row["mean_transition_acceptance_rate"]) for row in selected]
            )
            contexts = np.asarray(
                [
                    _float(row["mean_final_unique_contexts"])
                    / max(_float(row["num_samples"]), 1.0)
                    for row in selected
                ]
            )
            states = np.asarray(
                [
                    _float(row["mean_final_unique_states"])
                    / max(_float(row["num_samples"]), 1.0)
                    for row in selected
                ]
            )
            ax.plot(x, acceptance, marker="o", color="#4C78A8", label="acceptance rate")
            ax.plot(
                x, contexts, marker="s", color="#54A24B", label="unique contexts / N"
            )
            ax.plot(x, states, marker="^", color="#F58518", label="unique states / N")
            _mark_failed_or_missing(ax, x, selected)
            ax.set_title(f"{event_type}: {PARAMETER_LABELS[parameter]}")
            ax.set_xlabel(PARAMETER_LABELS[parameter])
            ax.set_ylabel("Acceptance / diversity fraction")
            ax.set_ylim(bottom=0.0)
            style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures_dir / "acceptance_and_diversity_vs_parameter.png", dpi=300)
    plt.close(fig)


def _remove_stale_figures(results_root: Path) -> None:
    figures_dir = results_root / "figures"
    for filename in (*FIGURE_FILENAMES, *LEGACY_FIGURE_FILENAMES):
        path = figures_dir / filename
        if path.is_file():
            path.unlink()


def _write_tables(
    results_root: Path,
    seed_rows: list[dict[str, Any]],
    setting_rows: list[dict[str, Any]],
    conclusions: list[dict[str, Any]],
) -> None:
    tables = results_root / "tables"
    _write_csv(tables / "ss_sensitivity_seed_level_results.csv", seed_rows)
    _write_csv(tables / "ss_sensitivity_setting_level_summary.csv", setting_rows)
    _write_csv(tables / "ss_sensitivity_paper_conclusion_table.csv", conclusions)


def _remove_stale_tables(results_root: Path) -> None:
    tables_dir = results_root / "tables"
    for filename in TABLE_FILENAMES:
        path = tables_dir / filename
        if path.is_file():
            path.unlink()


def summarize(*, results_root: Path = RESULTS_ROOT) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    plan_rows = _read_csv(results_root / "run_plan.csv")
    seed_rows = _seed_rows(plan_rows, results_root)
    setting_rows = _setting_rows(seed_rows, results_root)
    conclusions = _default_conclusions(setting_rows)
    has_recorded_execution = any(
        row.get("execution_status") != "pending" for row in seed_rows
    )
    if has_recorded_execution:
        _write_tables(results_root, seed_rows, setting_rows, conclusions)
    else:
        _remove_stale_tables(results_root)
    has_valid_ss_result = any(
        math.isfinite(_float(row["probability_mean"])) for row in setting_rows
    )
    status_path = results_root / "summary_status.json"
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "has_valid_ss_result": has_valid_ss_result,
                "has_recorded_execution": has_recorded_execution,
                "completed_ss_seed_runs": sum(
                    math.isfinite(_float(row["probability"])) for row in seed_rows
                ),
                "tables_generated": has_recorded_execution,
                "figures_generated": has_valid_ss_result,
                "message": (
                    "Figures are generated only after at least one valid SS run "
                    "to prevent empty plan artifacts from being mistaken for results."
                ),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    if has_valid_ss_result:
        _figures(setting_rows, results_root)
    else:
        _remove_stale_figures(results_root)


if __name__ == "__main__":
    summarize()
