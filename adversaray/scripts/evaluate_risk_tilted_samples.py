#!/usr/bin/env python3
"""Evaluate prior and risk-tilted plans in highway-env rollouts."""
from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
from adversaray.src.frozen_diffusion_sampler import FrozenDiffusionSampler
from diffusion.src.utils import load_yaml, save_json, setup_logging
from utils.context import context_from_npz


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "king_guided_following.yaml"
)
PLAN_FIELDS = [
    ("prior", "prior_actions"),
    ("tilted", "tilted_actions"),
]
PLAN_LABELS = {
    "prior": "prior",
    "tilted": "risk-tilted",
}
SCRIPT_DEFAULTS = {
    "samples_name": "risk_tilted_samples.npz",
    "output_name": "risk_tilted_eval_summary.json",
    "figure_dir": "figures",
    "num_contexts": 0,
    "num_workers": 8,
    "num_case_figures": 4,
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


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    return context_from_npz(raw, idx)


def _output_dir(cfg: dict[str, Any], base: Path) -> Path:
    paths = cfg.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    return _resolve(paths["output_dir"], base)


def _attach_runtime_paths(cfg: dict[str, Any], base: Path) -> None:
    paths = cfg.get("paths", {})
    runtime: dict[str, str] = {"config_dir": str(base)}
    for key in ("natural_dataset_dir", "output_dir"):
        if key in paths:
            runtime[key] = str(_resolve(paths[key], base))
    cfg["_runtime"] = runtime


def _make_frozen_runner(
    cfg: dict[str, Any],
    base: Path,
) -> ClosedLoopFollowingRunner:
    sampler = FrozenDiffusionSampler.from_config(cfg, config_dir=base).eval()
    return ClosedLoopFollowingRunner(sampler, cfg)


def _make_expert_runner(
    target_runner: ClosedLoopFollowingRunner,
    cfg: dict[str, Any],
) -> ClosedLoopFollowingRunner:
    expert_cfg = deepcopy(cfg)
    idm_cfg = dict(expert_cfg.get("idm", {}))
    idm_cfg["comfortable_brake"] = float(
        idm_cfg.get("expert_comfortable_brake", 6.0)
    )
    idm_cfg["min_gap"] = float(idm_cfg.get("expert_min_gap", 3.0))
    idm_cfg["desired_headway"] = float(
        idm_cfg.get("expert_desired_headway", 1.5)
    )
    expert_cfg["idm"] = idm_cfg
    return ClosedLoopFollowingRunner(target_runner.sampler, expert_cfg)


def _numeric_row(result: Any) -> dict[str, float]:
    row = {"closed_loop_risk": float(result.closed_loop_risk)}
    for key, value in result.metrics.items():
        if isinstance(value, (int, float, np.floating)):
            row[key] = float(value)
    return row


def _available_plan_fields(
    samples: dict[str, np.ndarray],
) -> list[tuple[str, str]]:
    required = {
        "context_states",
        "ego_length",
        "adv_length",
        "prior_actions",
        "tilted_actions",
    }
    missing = sorted(required - set(samples))
    if missing:
        raise KeyError(f"Samples file is missing required arrays: {missing}")
    fields = [(name, key) for name, key in PLAN_FIELDS if key in samples]
    if len(fields) < 2:
        raise RuntimeError(
            "At least prior_actions and tilted_actions are required"
        )
    return fields


def _evaluate_case(
    *,
    runner: ClosedLoopFollowingRunner,
    expert_runner: ClosedLoopFollowingRunner,
    samples: dict[str, np.ndarray],
    plan_fields: list[tuple[str, str]],
    case_id: int,
) -> tuple[
    int,
    dict[str, dict[str, float]],
    dict[str, list[dict[str, float]]],
]:
    ctx = _context(samples, case_id)
    rows: dict[str, dict[str, float]] = {}
    traces: dict[str, list[dict[str, float]]] = {}
    results: dict[str, Any] = {}
    default_steps = int(samples["prior_actions"].shape[1])
    event_steps = int(samples.get("event_steps", [default_steps])[case_id])
    for name, action_key in plan_fields:
        mask_key = f"{name}_action_mask"
        if mask_key in samples:
            plan_steps = int(np.sum(samples[mask_key][case_id]))
        elif "action_mask" in samples and name == "tilted":
            plan_steps = int(np.sum(samples["action_mask"][case_id]))
        else:
            plan_steps = event_steps
        plan_steps = max(plan_steps, 1)
        plan = np.asarray(
            samples[action_key][case_id, :plan_steps],
            dtype=np.float32,
        )
        result = runner.rollout_pre_sampled_plan(
            ctx,
            plan,
            episode_steps=plan_steps,
        )
        results[name] = result
    if "prior" not in results:
        raise RuntimeError("Risk-tilted evaluation requires a prior plan")
    for name, result in results.items():
        if name != "prior":
            action_key = dict(plan_fields)[name]
            mask_key = f"{name}_action_mask"
            if mask_key in samples:
                plan_steps = int(np.sum(samples[mask_key][case_id]))
            elif "action_mask" in samples:
                plan_steps = int(np.sum(samples["action_mask"][case_id]))
            else:
                plan_steps = event_steps
            plan_steps = max(plan_steps, 1)
            plan = np.asarray(
                samples[action_key][case_id, :plan_steps],
                dtype=np.float32,
            )
            expert_result = expert_runner.rollout_pre_sampled_plan(
                ctx,
                plan,
                episode_steps=plan_steps,
            )
            result.metrics["expert_closed_loop_risk"] = float(
                expert_result.closed_loop_risk
            )
            result.metrics["useful_failure_score"] = float(
                result.closed_loop_risk
                / (
                    1.0
                    + np.exp(float(expert_result.closed_loop_risk) - 1.0)
                )
            )
        rows[name] = _numeric_row(result)
        traces[name] = result.trace
    return case_id, rows, traces


def _log_case(
    case_id: int,
    selected: int,
    rows: dict[str, dict[str, float]],
) -> None:
    parts = []
    for name in rows:
        row = rows[name]
        parts.append(
            f"{PLAN_LABELS[name]} risk {row['closed_loop_risk']:.4f} "
            f"gap {row['min_gap']:.3f} TTC {row['min_ttc']:.3f}"
        )
    logger.info(
        "Evaluated case %d/%d | %s",
        case_id + 1,
        selected,
        " | ".join(parts),
    )


def evaluate_samples(
    *,
    runner: ClosedLoopFollowingRunner,
    expert_runner: ClosedLoopFollowingRunner,
    samples: dict[str, np.ndarray],
    num_contexts: int,
) -> tuple[
    dict[str, list[dict[str, float]]],
    dict[str, list[list[dict[str, float]]]],
]:
    plan_fields = _available_plan_fields(samples)
    total = int(samples["context_states"].shape[0])
    if int(num_contexts) <= 0:
        selected = total
    else:
        selected = min(total, int(num_contexts))
    if selected <= 0:
        raise RuntimeError("No samples selected for evaluation")
    rows_by_plan: dict[str, list[dict[str, float] | None]] = {
        name: [None] * selected for name, _key in plan_fields
    }
    traces_by_plan: dict[str, list[list[dict[str, float]] | None]] = {
        name: [None] * selected for name, _key in plan_fields
    }
    cpu_count = max(int(os.cpu_count() or 1), 1)
    num_workers = min(
        max(int(SCRIPT_DEFAULTS["num_workers"]), 1),
        selected,
        cpu_count,
    )
    if num_workers == 1:
        for case_id in range(selected):
            out_id, rows, traces = _evaluate_case(
                runner=runner,
                expert_runner=expert_runner,
                samples=samples,
                plan_fields=plan_fields,
                case_id=case_id,
            )
            for name in rows:
                rows_by_plan[name][out_id] = rows[name]
                traces_by_plan[name][out_id] = traces[name]
            _log_case(out_id, selected, rows)
    else:
        logger.info(
            "Evaluating %d cases with %d highway-env worker threads",
            selected,
            num_workers,
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_case,
                    runner=runner,
                    expert_runner=expert_runner,
                    samples=samples,
                    plan_fields=plan_fields,
                    case_id=case_id,
                )
                for case_id in range(selected)
            ]
            for future in as_completed(futures):
                out_id, rows, traces = future.result()
                for name in rows:
                    rows_by_plan[name][out_id] = rows[name]
                    traces_by_plan[name][out_id] = traces[name]
                _log_case(out_id, selected, rows)
    final_rows = {
        name: [row for row in rows if row is not None]
        for name, rows in rows_by_plan.items()
    }
    final_traces = {
        name: [trace for trace in traces if trace is not None]
        for name, traces in traces_by_plan.items()
    }
    return final_rows, final_traces


def _summarize(
    rows: list[dict[str, float]],
    prefix: str = "",
) -> dict[str, float]:
    if not rows:
        if prefix:
            key = f"{prefix}_closed_loop_risk_mean"
        else:
            key = "closed_loop_risk_mean"
        return {key: float("nan")}
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray(
            [row.get(key, np.nan) for row in rows],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out_key = f"{prefix}_{key}" if prefix else key
        out[f"{out_key}_mean"] = float(np.mean(values))
        out[f"{out_key}_p05"] = float(np.percentile(values, 5.0))
        out[f"{out_key}_p95"] = float(np.percentile(values, 95.0))
    for key in (
        "collision",
        "invalid_collision",
        "near_collision",
        "hard_brake",
        "invalid_initial_context",
    ):
        mean_key = f"{prefix}_{key}_mean" if prefix else f"{key}_mean"
        rate_key = f"{prefix}_{key}_rate" if prefix else f"{key}_rate"
        if mean_key in out:
            out[rate_key] = out[mean_key]
    return out


def _delta_summary(
    lhs_rows: list[dict[str, float]],
    rhs_rows: list[dict[str, float]],
) -> dict[str, float]:
    keys = (
        "closed_loop_risk",
        "collision",
        "invalid_collision",
        "near_collision",
        "min_gap",
        "final_gap",
        "min_ttc",
        "min_rss_margin",
        "relative_rss_objective",
        "raw_rss_objective",
        "expert_closed_loop_risk",
        "useful_failure_score",
        "min_ego_accel",
        "hard_brake",
        "lead_physics_penalty",
        "physical_feasible",
        "action_clip_rate",
        "jerk_violation_rate",
        "speed_negative_rate",
        "speed_violation_rate",
        "lead_accel_mean",
        "lead_jerk_abs_mean",
        "lead_speed_mean",
    )
    rows: list[dict[str, float]] = []
    for lhs, rhs in zip(lhs_rows, rhs_rows, strict=True):
        row = {
            f"{key}_delta": float(lhs.get(key, np.nan) - rhs.get(key, np.nan))
            for key in keys
        }
        rows.append(row)
    return _summarize(rows)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _trace_series(
    traces: list[list[dict[str, float]]],
    key: str,
    reducer: str,
) -> np.ndarray:
    values: list[float] = []
    for trace in traces:
        arr = np.asarray(
            [float(item.get(key, np.nan)) for item in trace],
            dtype=np.float64,
        )
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            values.append(float("nan"))
        elif reducer == "min":
            values.append(float(np.min(arr)))
        elif reducer == "mean":
            values.append(float(np.mean(arr)))
        elif reducer == "mean_abs":
            values.append(float(np.mean(np.abs(arr))))
        else:
            raise ValueError(f"Unknown trace reducer: {reducer}")
    return np.asarray(values, dtype=np.float32)


def _hist_overlay_many(
    ax: Any,
    values_by_plan: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    bins: int,
) -> None:
    finite_items = {
        name: _finite(values)
        for name, values in values_by_plan.items()
        if _finite(values).size > 0
    }
    if len(finite_items) < 2:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "not enough finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    lo = min(float(np.min(values)) for values in finite_items.values())
    hi = max(float(np.max(values)) for values in finite_items.values())
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    for name, values in finite_items.items():
        ax.hist(
            values,
            bins=edges,
            alpha=0.45,
            density=True,
            label=PLAN_LABELS[name],
        )
        ax.axvline(float(np.mean(values)), linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)


def _plot_closed_loop_histograms(
    rows_by_plan: dict[str, list[dict[str, float]]],
    traces_by_plan: dict[str, list[list[dict[str, float]]]],
    out_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    def row_values(name: str, key: str) -> np.ndarray:
        rows = rows_by_plan.get(name, [])
        return np.asarray(
            [row.get(key, np.nan) for row in rows],
            dtype=np.float32,
        )

    def values_for(
        key: str,
        reducer: str | None = None,
    ) -> dict[str, np.ndarray]:
        if reducer is None:
            return {name: row_values(name, key) for name in rows_by_plan}
        return {
            name: _trace_series(traces_by_plan.get(name, []), key, reducer)
            for name in rows_by_plan
        }

    panels = (
        ("closed-loop risk", values_for("closed_loop_risk"), "risk score"),
        ("min gap", values_for("min_gap"), "min gap [m]"),
        ("min TTC", values_for("min_ttc"), "min TTC [s]"),
        (
            "mean |lead acceleration|",
            values_for("lead_accel", "mean_abs"),
            "mean |lead acceleration| [m/s^2]",
        ),
        (
            "mean |lead jerk|",
            values_for("lead_jerk", "mean_abs"),
            "mean |lead jerk| [m/s^3]",
        ),
        (
            "min ego acceleration",
            values_for("ego_accel", "min"),
            "min ego acceleration [m/s^2]",
        ),
        (
            "mean ego speed",
            values_for("ego_speed", "mean"),
            "mean ego speed [m/s]",
        ),
    )
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 11.5))
    for ax, (title, values, xlabel) in zip(
        axes.reshape(-1),
        panels,
        strict=True,
    ):
        _hist_overlay_many(ax, values, title, xlabel, bins)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Risk-tilted diffusion closed-loop highway-env diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _trace_array(trace: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray(
        [float(item.get(key, np.nan)) for item in trace],
        dtype=np.float32,
    )


def _plot_closed_loop_case(
    case_id: int,
    dataset_index: int | None,
    traces_by_plan: dict[str, list[list[dict[str, float]]]],
    out_path: Path,
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), sharex=False)
    panels = (
        ("lead_accel", "lead acceleration [m/s^2]"),
        ("lead_jerk", "lead jerk [m/s^3]"),
        ("gap", "gap [m]"),
        ("ttc", "TTC [s]"),
        ("lead_speed", "lead speed [m/s]"),
    )
    for ax, (key, ylabel) in zip(axes.reshape(-1)[:5], panels, strict=True):
        for name in traces_by_plan:
            trace = traces_by_plan[name][case_id]
            steps = _trace_array(trace, "step")
            values = _trace_array(trace, key)
            if key == "ttc":
                values = np.clip(values, 0.0, 60.0)
            ax.plot(steps, values, label=PLAN_LABELS[name], linewidth=1.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if key == "gap":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)

    ax = axes.reshape(-1)[5]
    for name in traces_by_plan:
        trace = traces_by_plan[name][case_id]
        steps = _trace_array(trace, "step")
        ax.plot(
            steps,
            _trace_array(trace, "lead_position"),
            linewidth=1.7,
            label=f"lead/{PLAN_LABELS[name]}",
        )
    ax.set_ylabel("lead x position [m]")
    ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("highway-env step")
    axes[0, 0].legend(loc="best")
    axes.reshape(-1)[5].legend(loc="best")
    title = f"Risk-tilted closed-loop case {case_id:04d}"
    if dataset_index is not None:
        title += f" / dataset_index={dataset_index}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _case_indices_for_figures(
    rows_by_plan: dict[str, list[dict[str, float]]],
    num_cases: int,
) -> list[int]:
    count = max(int(SCRIPT_DEFAULTS["num_case_figures"]), 0)
    if count <= 0:
        return []
    if "prior" not in rows_by_plan or "tilted" not in rows_by_plan:
        return list(range(min(count, num_cases)))
    pairs = zip(
        rows_by_plan["prior"],
        rows_by_plan["tilted"],
        strict=True,
    )
    delta = np.asarray(
        [
            tilted.get("min_gap", np.nan) - prior.get("min_gap", np.nan)
            for prior, tilted in pairs
        ],
        dtype=np.float64,
    )
    finite = np.where(np.isfinite(delta))[0]
    if finite.size == 0:
        return list(range(min(count, num_cases)))
    ordered = finite[np.argsort(delta[finite])]
    return [int(item) for item in ordered[: min(count, len(ordered))]]


def _summary(
    rows_by_plan: dict[str, list[dict[str, float]]],
    samples_path: Path,
    samples: dict[str, np.ndarray],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples_path": str(samples_path),
        "num_contexts": int(len(next(iter(rows_by_plan.values())))),
    }
    for name in ("prior", "tilted"):
        if name in rows_by_plan:
            summary[name] = _summarize(rows_by_plan[name])
        else:
            summary[name] = {}
    if "prior" in rows_by_plan and "tilted" in rows_by_plan:
        summary["tilted_minus_prior"] = _delta_summary(
            rows_by_plan["tilted"],
            rows_by_plan["prior"],
        )
        action_l2 = _masked_action_l2(samples, int(summary["num_contexts"]))
        summary["tilted_minus_prior"]["action_l2_mean"] = (
            float(np.mean(action_l2)) if action_l2.size else float("nan")
        )
        summary["tilted_minus_prior"]["action_l2_p95"] = (
            float(np.percentile(action_l2, 95.0))
            if action_l2.size
            else float("nan")
        )
    else:
        summary["tilted_minus_prior"] = {}
    return summary


def _masked_action_l2(
    samples: dict[str, np.ndarray],
    count: int,
) -> np.ndarray:
    reference_key = (
        "tilted_reference_actions"
        if "tilted_reference_actions" in samples
        else "prior_actions"
    )
    diff = (
        np.asarray(samples["tilted_actions"][:count], dtype=np.float32)
        - np.asarray(samples[reference_key][:count], dtype=np.float32)
    )
    per_step = np.mean(np.square(diff), axis=-1)
    if "tilted_action_mask" in samples:
        mask = np.asarray(
            samples["tilted_action_mask"][:count],
            dtype=np.float32,
        )
    elif "action_mask" in samples:
        mask = np.asarray(samples["action_mask"][:count], dtype=np.float32)
    else:
        return np.sqrt(np.mean(per_step, axis=1))
    denom = np.maximum(np.sum(mask, axis=1), 1.0)
    return np.sqrt(np.sum(per_step * mask, axis=1) / denom)


def main() -> None:
    setup_logging(SCRIPT_DEFAULTS["log_level"])

    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    _attach_runtime_paths(cfg, base)
    output_root = _output_dir(cfg, base)
    samples_path = output_root / str(SCRIPT_DEFAULTS["samples_name"])
    output_path = output_root / str(SCRIPT_DEFAULTS["output_name"])
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Risk-tilted samples not found: {samples_path}"
        )

    samples = _load_npz(samples_path)
    runner = _make_frozen_runner(cfg, base)
    expert_runner = _make_expert_runner(runner, cfg)
    rows_by_plan, traces_by_plan = evaluate_samples(
        runner=runner,
        expert_runner=expert_runner,
        samples=samples,
        num_contexts=int(SCRIPT_DEFAULTS["num_contexts"]),
    )
    summary = _summary(rows_by_plan, samples_path, samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_path)

    figure_dir = output_root / str(SCRIPT_DEFAULTS["figure_dir"])
    hist_path = figure_dir / "risk_tilted_closed_loop_histograms.png"
    _plot_closed_loop_histograms(
        rows_by_plan,
        traces_by_plan,
        hist_path,
        bins=int(SCRIPT_DEFAULTS["bins"]),
        dpi=int(SCRIPT_DEFAULTS["dpi"]),
    )
    for case_id in _case_indices_for_figures(
        rows_by_plan,
        int(summary["num_contexts"]),
    ):
        if "dataset_index" in samples:
            dataset_index = int(samples["dataset_index"][case_id])
        else:
            dataset_index = None
        _plot_closed_loop_case(
            case_id,
            dataset_index,
            traces_by_plan,
            figure_dir / f"risk_tilted_case_{case_id:04d}.png",
            dpi=int(SCRIPT_DEFAULTS["dpi"]),
        )
    logger.info(
        "Saved risk-tilted highway-env evaluation summary to %s",
        output_path,
    )
    logger.info("Saved risk-tilted closed-loop diagnostics to %s", hist_path)


if __name__ == "__main__":
    main()
