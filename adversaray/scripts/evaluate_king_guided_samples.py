#!/usr/bin/env python3
"""Evaluate saved prior/KING plans in closed-loop highway-env rollouts."""
from __future__ import annotations

import logging
import os
import sys
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
from adversaray.src.context_utils import _context
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "king_guided_following.yaml"
SCRIPT_DEFAULTS = {
    "samples_name": "king_guided_samples.npz",
    "output_name": "king_guided_eval_summary.json",
    "figure_dir": "figures",
    "num_contexts": 0,
    "num_workers": 10,
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


def _make_frozen_runner(cfg: dict[str, Any], base: Path) -> ClosedLoopFollowingRunner:
    sampler = FrozenDiffusionSampler.from_config(cfg, config_dir=base).eval()
    return ClosedLoopFollowingRunner(sampler, cfg)


def _numeric_row(result: Any) -> dict[str, float]:
    row = {"closed_loop_risk": float(result.closed_loop_risk)}
    for key, value in result.metrics.items():
        if isinstance(value, (int, float, np.floating)):
            row[key] = float(value)
    return row


def _trace_series(traces: list[list[dict[str, float]]], key: str, reducer: str) -> np.ndarray:
    values: list[float] = []
    for trace in traces:
        arr = np.asarray([float(item.get(key, np.nan)) for item in trace], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            values.append(float("nan"))
        elif reducer == "min":
            values.append(float(np.min(arr)))
        elif reducer == "max":
            values.append(float(np.max(arr)))
        elif reducer == "mean":
            values.append(float(np.mean(arr)))
        elif reducer == "mean_abs":
            values.append(float(np.mean(np.abs(arr))))
        else:
            raise ValueError(f"Unknown trace reducer: {reducer}")
    return np.asarray(values, dtype=np.float32)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


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


def _plot_closed_loop_histograms(
    prior_rows: list[dict[str, float]],
    king_rows: list[dict[str, float]],
    prior_traces: list[list[dict[str, float]]],
    king_traces: list[list[dict[str, float]]],
    out_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    def row_values(rows: list[dict[str, float]], key: str) -> np.ndarray:
        return np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float32)

    panels = (
        ("closed-loop risk", row_values(prior_rows, "closed_loop_risk"), row_values(king_rows, "closed_loop_risk"), "risk score"),
        ("min gap", row_values(prior_rows, "min_gap"), row_values(king_rows, "min_gap"), "min gap [m]"),
        ("min TTC", row_values(prior_rows, "min_ttc"), row_values(king_rows, "min_ttc"), "min TTC [s]"),
        ("min RSS margin", row_values(prior_rows, "min_rss_margin"), row_values(king_rows, "min_rss_margin"), "min RSS margin [m]"),
        ("mean |lead acceleration|", _trace_series(prior_traces, "lead_accel", "mean_abs"), _trace_series(king_traces, "lead_accel", "mean_abs"), "mean |lead acceleration| [m/s^2]"),
        ("mean |lead jerk|", _trace_series(prior_traces, "lead_jerk", "mean_abs"), _trace_series(king_traces, "lead_jerk", "mean_abs"), "mean |lead jerk| [m/s^3]"),
        ("min ego acceleration", _trace_series(prior_traces, "ego_accel", "min"), _trace_series(king_traces, "ego_accel", "min"), "min ego acceleration [m/s^2]"),
        ("mean ego speed", _trace_series(prior_traces, "ego_speed", "mean"), _trace_series(king_traces, "ego_speed", "mean"), "mean ego speed [m/s]"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 11.5))
    for ax, (title, prior, king, xlabel) in zip(axes.reshape(-1), panels, strict=True):
        _hist_overlay(ax, prior, king, title, xlabel, bins)
    axes[0, 0].legend(loc="best")
    fig.suptitle("KING-guided closed-loop highway-env diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _trace_array(trace: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(item.get(key, np.nan)) for item in trace], dtype=np.float32)


def _plot_closed_loop_case(
    case_id: int,
    dataset_index: int | None,
    prior_trace: list[dict[str, float]],
    king_trace: list[dict[str, float]],
    out_path: Path,
    *,
    dpi: int,
) -> None:
    prior_steps = _trace_array(prior_trace, "step")
    king_steps = _trace_array(king_trace, "step")
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), sharex=False)
    panels = (
        ("lead_accel", "lead acceleration [m/s^2]"),
        ("lead_jerk", "lead jerk [m/s^3]"),
        ("gap", "gap [m]"),
        ("ttc", "TTC [s]"),
        ("rss_margin", "RSS margin [m]"),
    )
    for ax, (key, ylabel) in zip(axes.reshape(-1)[:5], panels, strict=True):
        prior_y = _trace_array(prior_trace, key)
        king_y = _trace_array(king_trace, key)
        if key == "ttc":
            prior_y = np.clip(prior_y, 0.0, 60.0)
            king_y = np.clip(king_y, 0.0, 60.0)
        ax.plot(prior_steps, prior_y, label="prior", linewidth=1.8)
        ax.plot(king_steps, king_y, label="king", linewidth=1.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if key in {"gap", "rss_margin"}:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)

    ax = axes.reshape(-1)[5]
    ax.plot(prior_steps, _trace_array(prior_trace, "ego_position"), color="C0", linestyle="--", linewidth=1.6, label="ego/prior")
    ax.plot(prior_steps, _trace_array(prior_trace, "lead_position"), color="C0", linewidth=1.8, label="lead/prior")
    ax.plot(king_steps, _trace_array(king_trace, "ego_position"), color="C1", linestyle="--", linewidth=1.6, label="ego/king")
    ax.plot(king_steps, _trace_array(king_trace, "lead_position"), color="C1", linewidth=1.8, label="lead/king")
    ax.set_ylabel("x position [m]")
    ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("highway-env step")
    axes[0, 0].legend(loc="best")
    axes.reshape(-1)[5].legend(loc="best")
    title = f"Closed-loop KING case {case_id:04d}"
    if dataset_index is not None:
        title += f" / dataset_index={dataset_index}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _summarize(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not rows:
        return {f"{prefix}_closed_loop_risk_mean": float("nan")}
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[f"{prefix}_{key}_mean"] = float(np.mean(values))
        out[f"{prefix}_{key}_p05"] = float(np.percentile(values, 5.0))
        out[f"{prefix}_{key}_p95"] = float(np.percentile(values, 95.0))
    for key in ("collision", "collision_valid", "invalid_collision", "near_collision", "hard_brake", "invalid_initial_context"):
        mean_key = f"{prefix}_{key}_mean"
        if mean_key in out:
            out[f"{prefix}_{key}_rate"] = out[mean_key]
    return out


def _delta_summary(prior_rows: list[dict[str, float]], king_rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "closed_loop_risk",
        "collision",
        "collision_valid",
        "invalid_collision",
        "near_collision",
        "min_gap",
        "final_gap",
        "min_ttc",
        "min_rss_margin",
        "min_ego_accel",
        "hard_brake",
        "lead_physics_penalty",
        "action_clip_rate",
        "jerk_violation_rate",
        "speed_negative_rate",
        "lead_accel_mean",
        "lead_jerk_abs_mean",
        "lead_speed_mean",
    )
    rows: list[dict[str, float]] = []
    for prior, king in zip(prior_rows, king_rows, strict=True):
        rows.append({f"{key}_delta": float(king.get(key, np.nan) - prior.get(key, np.nan)) for key in keys})
    return _summarize(rows, "king_minus_prior")


def _closed_loop_consistency_summary(
    samples: dict[str, np.ndarray],
    prior_rows: list[dict[str, float]],
    king_rows: list[dict[str, float]],
) -> dict[str, float]:
    if "risk_after" not in samples or "risk_before" not in samples:
        return {
            "p3_available": 0.0,
            "direction_consistency_rate": float("nan"),
            "risk_increase_agreement_rate": float("nan"),
            "delta_correlation": float("nan"),
        }
    count = min(
        len(prior_rows),
        len(king_rows),
        int(samples["risk_before"].shape[0]),
        int(samples["risk_after"].shape[0]),
    )
    sample_delta = (
        np.asarray(samples["risk_after"][:count], dtype=np.float64)
        - np.asarray(samples["risk_before"][:count], dtype=np.float64)
    )
    closed_delta = np.asarray(
        [
            king.get("closed_loop_risk", np.nan)
            - prior.get("closed_loop_risk", np.nan)
            for prior, king in zip(
                prior_rows[:count],
                king_rows[:count],
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(sample_delta) & np.isfinite(closed_delta)
    if not np.any(finite):
        return {
            "p3_available": 1.0,
            "direction_consistency_rate": float("nan"),
            "risk_increase_agreement_rate": float("nan"),
            "delta_correlation": float("nan"),
        }
    sample_f = sample_delta[finite]
    closed_f = closed_delta[finite]
    nonzero = (np.abs(sample_f) > 1e-12) & (np.abs(closed_f) > 1e-12)
    if np.any(nonzero):
        direction_rate = float(
            np.mean(np.sign(sample_f[nonzero]) == np.sign(closed_f[nonzero]))
        )
    else:
        direction_rate = float("nan")
    agreement_rate = float(np.mean((sample_f > 0.0) == (closed_f > 0.0)))
    if sample_f.size > 1 and np.std(sample_f) > 1e-12 and np.std(closed_f) > 1e-12:
        correlation = float(np.corrcoef(sample_f, closed_f)[0, 1])
    else:
        correlation = float("nan")
    return {
        "p3_available": 1.0,
        "direction_consistency_rate": direction_rate,
        "risk_increase_agreement_rate": agreement_rate,
        "delta_correlation": correlation,
        "sample_risk_delta_mean": float(np.mean(sample_f)),
        "closed_loop_risk_delta_mean": float(np.mean(closed_f)),
    }


def _required_samples(samples: dict[str, np.ndarray]) -> None:
    required = {"context_states", "ego_length", "adv_length", "prior_actions", "king_actions"}
    missing = sorted(required - set(samples))
    if missing:
        raise KeyError(f"Samples file is missing required arrays: {missing}")


def _evaluate_case(
    *,
    runner: ClosedLoopFollowingRunner,
    samples: dict[str, np.ndarray],
    case_id: int,
) -> tuple[int, dict[str, float], dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    ctx = _context(samples, case_id)
    default_steps = int(samples["prior_actions"].shape[1])
    event_steps = int(samples.get("event_steps", [default_steps])[case_id])
    prior_steps = event_steps
    king_steps = event_steps
    if "prior_action_mask" in samples:
        prior_steps = int(np.sum(samples["prior_action_mask"][case_id]))
    if "king_action_mask" in samples:
        king_steps = int(np.sum(samples["king_action_mask"][case_id]))
    prior_steps = max(prior_steps, 1)
    king_steps = max(king_steps, 1)
    prior_plan = np.asarray(
        samples["prior_actions"][case_id, :prior_steps],
        dtype=np.float32,
    )
    king_plan = np.asarray(
        samples["king_actions"][case_id, :king_steps],
        dtype=np.float32,
    )
    prior_result = runner.rollout_pre_sampled_plan(
        ctx,
        prior_plan,
        episode_steps=event_steps,
    )
    king_result = runner.rollout_pre_sampled_plan(
        ctx,
        king_plan,
        episode_steps=event_steps,
    )
    return (
        case_id,
        _numeric_row(prior_result),
        _numeric_row(king_result),
        prior_result.trace,
        king_result.trace,
    )


def _log_case(case_id: int, selected: int, prior: dict[str, float], king: dict[str, float]) -> None:
    logger.info(
        "Evaluated case %d/%d closed-loop risk %.4f -> %.4f | gap %.3f -> %.3f | TTC %.3f -> %.3f | RSS %.3f -> %.3f",
        case_id + 1,
        selected,
        float(prior["closed_loop_risk"]),
        float(king["closed_loop_risk"]),
        float(prior["min_gap"]),
        float(king["min_gap"]),
        float(prior["min_ttc"]),
        float(king["min_ttc"]),
        float(prior["min_rss_margin"]),
        float(king["min_rss_margin"]),
    )


def evaluate_samples(
    *,
    runner: ClosedLoopFollowingRunner,
    samples: dict[str, np.ndarray],
    num_contexts: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[list[dict[str, float]]], list[list[dict[str, float]]]]:
    _required_samples(samples)
    total = int(samples["context_states"].shape[0])
    selected = total if int(num_contexts) <= 0 else min(total, int(num_contexts))
    if selected <= 0:
        raise RuntimeError("No samples selected for evaluation")
    prior_rows: list[dict[str, float] | None] = [None] * selected
    king_rows: list[dict[str, float] | None] = [None] * selected
    prior_traces: list[list[dict[str, float]] | None] = [None] * selected
    king_traces: list[list[dict[str, float]] | None] = [None] * selected
    num_workers = min(max(int(SCRIPT_DEFAULTS["num_workers"]), 1), selected, max(int(os.cpu_count() or 1), 1))
    if num_workers == 1:
        for case_id in range(selected):
            out_id, prior, king, prior_trace, king_trace = _evaluate_case(runner=runner, samples=samples, case_id=case_id)
            prior_rows[out_id] = prior
            king_rows[out_id] = king
            prior_traces[out_id] = prior_trace
            king_traces[out_id] = king_trace
            _log_case(out_id, selected, prior, king)
    else:
        logger.info("Evaluating %d cases with %d highway-env worker threads", selected, num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_evaluate_case, runner=runner, samples=samples, case_id=case_id)
                for case_id in range(selected)
            ]
            for future in as_completed(futures):
                out_id, prior, king, prior_trace, king_trace = future.result()
                prior_rows[out_id] = prior
                king_rows[out_id] = king
                prior_traces[out_id] = prior_trace
                king_traces[out_id] = king_trace
                _log_case(out_id, selected, prior, king)
    return (
        [row for row in prior_rows if row is not None],
        [row for row in king_rows if row is not None],
        [trace for trace in prior_traces if trace is not None],
        [trace for trace in king_traces if trace is not None],
    )


def _case_indices_for_figures(
    prior_rows: list[dict[str, float]],
    king_rows: list[dict[str, float]],
    num_cases: int,
) -> list[int]:
    count = max(int(SCRIPT_DEFAULTS["num_case_figures"]), 0)
    if count <= 0:
        return []
    delta = np.asarray(
        [king.get("min_gap", np.nan) - prior.get("min_gap", np.nan) for prior, king in zip(prior_rows, king_rows, strict=True)],
        dtype=np.float64,
    )
    finite = np.where(np.isfinite(delta))[0]
    if finite.size == 0:
        return list(range(min(count, num_cases)))
    ordered = finite[np.argsort(delta[finite])]
    return [int(item) for item in ordered[: min(count, len(ordered))]]


def _masked_action_l2(samples: dict[str, np.ndarray], count: int) -> np.ndarray:
    reference_key = (
        "king_reference_actions"
        if "king_reference_actions" in samples
        else "prior_actions"
    )
    diff = (
        np.asarray(samples["king_actions"][:count], dtype=np.float32)
        - np.asarray(samples[reference_key][:count], dtype=np.float32)
    )
    per_step = np.mean(np.square(diff), axis=-1)
    if "action_mask" not in samples:
        return np.sqrt(np.mean(per_step, axis=1))
    mask = np.asarray(samples["action_mask"][:count], dtype=np.float32)
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
        raise FileNotFoundError(f"KING samples not found: {samples_path}")

    samples = _load_npz(samples_path)
    runner = _make_frozen_runner(cfg, base)
    prior_rows, king_rows, prior_traces, king_traces = evaluate_samples(
        runner=runner,
        samples=samples,
        num_contexts=int(SCRIPT_DEFAULTS["num_contexts"]),
    )
    action_l2 = _masked_action_l2(samples, len(king_rows))
    summary = {
        "samples_path": str(samples_path),
        "num_contexts": int(len(prior_rows)),
        "prior": _summarize(prior_rows, "prior"),
        "king": _summarize(king_rows, "king"),
        "delta": {
            **_delta_summary(prior_rows, king_rows),
            "action_l2_mean": float(np.mean(action_l2)) if action_l2.size else float("nan"),
            "action_l2_p95": float(np.percentile(action_l2, 95.0)) if action_l2.size else float("nan"),
        },
        "closed_loop_consistency": _closed_loop_consistency_summary(
            samples,
            prior_rows,
            king_rows,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_path)
    figure_dir = output_root / str(SCRIPT_DEFAULTS["figure_dir"])
    hist_path = figure_dir / "king_guided_closed_loop_histograms.png"
    _plot_closed_loop_histograms(
        prior_rows,
        king_rows,
        prior_traces,
        king_traces,
        hist_path,
        bins=int(SCRIPT_DEFAULTS["bins"]),
        dpi=int(SCRIPT_DEFAULTS["dpi"]),
    )
    for case_id in _case_indices_for_figures(prior_rows, king_rows, len(prior_rows)):
        dataset_index = int(samples["dataset_index"][case_id]) if "dataset_index" in samples else None
        _plot_closed_loop_case(
            case_id,
            dataset_index,
            prior_traces[case_id],
            king_traces[case_id],
            figure_dir / f"king_guided_closed_loop_case_{case_id:04d}.png",
            dpi=int(SCRIPT_DEFAULTS["dpi"]),
        )
    logger.info("Saved KING-guided highway-env evaluation summary to %s", output_path)
    logger.info("Saved KING-guided closed-loop diagnostics to %s", hist_path)


if __name__ == "__main__":
    main()
