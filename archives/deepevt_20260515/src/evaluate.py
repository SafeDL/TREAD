"""
evaluate.py — direct DeepEVT quantile evaluation.

The report is aligned with the current testing goal:

* raw direct q85/q90/q95 for scenario ranking;
* global empirical quantiles as calibration references;
* split-local ranking / enrichment diagnostics;
* one compact calibration figure for q85/q90/q95.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from process_highd.src.io_utils import ensure_dir, load_json, save_json

from .data import DatasetArrays, apply_normalization, load_dataset, subset
from .inference import DeepEVTPredictions, load_model, predict
from .metrics import exceedance_calibration_error

logger = logging.getLogger(__name__)


def _distribution_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q05": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q95": float(qs[4]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denom) if denom > 0.0 else float("nan")


def _auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[labels]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def _topk_enrichment(
    labels: np.ndarray,
    scores: np.ndarray,
    fractions: Iterable[float] = (0.01, 0.05, 0.10, 0.20),
) -> List[Dict[str, float]]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    baseline_rate = float(np.mean(labels))
    order = np.argsort(-scores)
    out: List[Dict[str, float]] = []
    for frac in fractions:
        k = max(1, min(len(labels), int(round(float(frac) * len(labels)))))
        top_rate = float(np.mean(labels[order[:k]]))
        out.append({
            "top_fraction": float(frac),
            "k": int(k),
            "tail_label_rate": top_rate,
            "baseline_tail_label_rate": baseline_rate,
            "enrichment": float(top_rate / baseline_rate) if baseline_rate > 0.0 else float("nan"),
        })
    return out


def _quantile_crossing_rate(q_by_level: Mapping[float, np.ndarray]) -> float:
    ordered = sorted(float(x) for x in q_by_level)
    if len(ordered) < 2:
        return 0.0
    stacked = np.stack([q_by_level[tau] for tau in ordered], axis=1)
    return float(np.mean(np.any(np.diff(stacked, axis=1) < -1e-7, axis=1)))


def _q_by_level(preds: DeepEVTPredictions, levels: Iterable[float]) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {}
    pred_levels = tuple(float(x) for x in preds.quantile_levels)
    for tau in levels:
        tau_f = float(tau)
        idx = min(range(len(pred_levels)), key=lambda i: abs(pred_levels[i] - tau_f))
        if abs(pred_levels[idx] - tau_f) > 1e-6:
            raise ValueError(f"Model does not provide q{int(tau_f * 100)}")
        out[tau_f] = preds.quantiles[:, idx]
    return out


def _quantile_report(
    arrays: DatasetArrays,
    q_by_level: Mapping[float, np.ndarray],
    levels: Iterable[float],
    *,
    source: str,
) -> Dict[str, object]:
    levels_report: Dict[str, object] = {}
    for tau in levels:
        tau_f = float(tau)
        q_tau = np.asarray(q_by_level[tau_f], dtype=np.float64)
        levels_report[f"tau_{tau_f}"] = {
            **exceedance_calibration_error(arrays.risk_score, q_tau, tau_f),
            "q_distribution": _distribution_summary(q_tau),
            "invalid_rate": 0.0,
        }
    return {
        "n": int(len(arrays.risk_score)),
        "source": source,
        "crossing_rate": _quantile_crossing_rate(q_by_level),
        "tail_levels": levels_report,
    }


def _global_empirical_baseline(
    train_risk: np.ndarray,
    arrays: DatasetArrays,
    levels: Iterable[float],
) -> Dict[str, object]:
    levels_report: Dict[str, object] = {}
    for tau in levels:
        tau_f = float(tau)
        q = float(np.quantile(train_risk, tau_f))
        q_pred = np.full_like(arrays.risk_score, q, dtype=np.float64)
        levels_report[f"tau_{tau_f}"] = {
            "q": q,
            **exceedance_calibration_error(arrays.risk_score, q_pred, tau_f),
        }
    return {
        "n": int(len(arrays.risk_score)),
        "source": "train_split_global_empirical_quantile",
        "tail_levels": levels_report,
    }


def _ranking_diagnostics(
    arrays: DatasetArrays,
    q_by_level: Mapping[float, np.ndarray],
    levels: Iterable[float],
) -> Dict[str, object]:
    diag: Dict[str, object] = {
        "n": int(len(arrays.risk_score)),
        "score_source": "raw_direct_quantile",
        "tail_label_definition": "risk_score > split empirical q_tau",
        "tail_levels": {},
    }
    for tau in levels:
        tau_f = float(tau)
        q_pred = np.asarray(q_by_level[tau_f], dtype=np.float64)
        split_q = float(np.quantile(arrays.risk_score, tau_f))
        labels = arrays.risk_score > split_q
        diag["tail_levels"][f"tau_{tau_f}"] = {
            "split_empirical_tail_threshold": split_q,
            "tail_label_rate": float(np.mean(labels)),
            "spearman_predicted_q_vs_risk": _spearman(q_pred, arrays.risk_score),
            "spearman_predicted_q_vs_tail_label": _spearman(q_pred, labels.astype(np.float64)),
            "tail_label_auc": _auc_from_scores(labels, q_pred),
            "topk_enrichment": _topk_enrichment(labels, q_pred),
        }
    return diag


def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _write_quantile_calibration_summary_figure(
    primary: Mapping[str, object],
    levels: Iterable[float],
    figures_dir: Path,
) -> None:
    plt = _lazy_plt()
    methods = [
        ("raw", primary["test_raw_direct"]),
        ("global", primary["global_empirical_quantile_baseline"]["test"]),
    ]
    taus = [float(x) for x in levels]
    x = np.arange(len(taus), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    for i, (name, report) in enumerate(methods):
        level_report = report["tail_levels"]
        offset = (i - 0.5) * width
        empirical = [
            float(level_report[f"tau_{tau}"]["empirical_exceed_rate"])
            for tau in taus
        ]
        ece = [
            float(level_report[f"tau_{tau}"]["ece"])
            for tau in taus
        ]
        axes[0].bar(x + offset, empirical, width=width, label=name)
        axes[1].bar(x + offset, ece, width=width, label=name)

    expected = [1.0 - tau for tau in taus]
    axes[0].plot(x, expected, "k--", linewidth=1.0, label="target")
    axes[0].set_ylabel("empirical exceedance")
    axes[0].set_title("Test quantile calibration across q85/q90/q95")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_ylabel("ECE")
    axes[1].set_xlabel("quantile level")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"q{int(round(tau * 100))}" for tau in taus])
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(figures_dir / "quantile_calibration_comparison.png", dpi=140)
    plt.close(fig)


def evaluate_deepevt(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    config: dict,
    run_quantile_baseline: bool = True,
    tail_levels: Iterable[float] = (0.85, 0.90, 0.95),
    report_name: Optional[str] = None,
) -> Dict[str, dict]:
    del run_quantile_baseline
    out = Path(output_dir)
    figures_dir = out / "figures"
    ensure_dir(figures_dir)
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "tread_deepevt_matplotlib"
    ensure_dir(cache_root)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    schema = load_json(out / "feature_schema.json")
    norm_stats = load_json(out / "normalization_stats.json")
    arrays = load_dataset(out)
    norm_arrays = apply_normalization(arrays, norm_stats)
    train_arrays = subset(arrays, "train")
    val_arrays = subset(arrays, "val")
    test_arrays = subset(arrays, "test")
    val_norm = subset(norm_arrays, "val")
    test_norm = subset(norm_arrays, "test")
    levels = tuple(float(x) for x in tail_levels)
    if not levels:
        raise ValueError("tail_levels must not be empty.")
    if len(test_arrays.risk_score) == 0:
        raise RuntimeError("Test split is empty; cannot evaluate DeepEVT.")

    model = load_model(checkpoint_path)
    val_preds = predict(model, val_norm)
    test_preds = predict(model, test_norm)
    val_raw_q = _q_by_level(val_preds, levels)
    test_raw_q = _q_by_level(test_preds, levels)

    primary = {
        "objective": "direct_prefix_conditional_quantile_prediction",
        "prefix_context": "prefix_states[0:12]",
        "ranking_score_source": "raw_direct_quantile",
        "export_threshold_source": "direct_quantile",
        "quantile_levels": list(levels),
        "validation_raw_direct": _quantile_report(
            val_arrays, val_raw_q, levels,
            source="raw_direct_quantile",
        ),
        "test_raw_direct": _quantile_report(
            test_arrays, test_raw_q, levels,
            source="raw_direct_quantile",
        ),
        "global_empirical_quantile_baseline": {
            "validation": _global_empirical_baseline(
                train_arrays.risk_score, val_arrays, levels,
            ),
            "test": _global_empirical_baseline(
                train_arrays.risk_score, test_arrays, levels,
            ),
        },
        "ranking_diagnostics": {
            "validation": _ranking_diagnostics(
                val_arrays, val_raw_q, levels,
            ),
            "test": _ranking_diagnostics(
                test_arrays, test_raw_q, levels,
            ),
        },
    }

    report: Dict[str, dict] = {
        "event_type": schema["event_type"],
        "n_train": int(len(train_arrays.risk_score)),
        "n_val": int(len(val_arrays.risk_score)),
        "n_test": int(len(test_arrays.risk_score)),
        "primary_task": primary,
        "deepevt_dataset_tail": {
            "train": _distribution_summary(train_arrays.risk_score),
            "validation": _distribution_summary(val_arrays.risk_score),
            "test": _distribution_summary(test_arrays.risk_score),
            "train_global_empirical_quantiles": {
                f"tau_{float(tau)}": float(np.quantile(train_arrays.risk_score, float(tau)))
                for tau in levels
            },
        },
    }

    _write_quantile_calibration_summary_figure(primary, levels, figures_dir)

    report_filename = report_name or "eval_report.json"
    save_json(report, out / report_filename)
    logger.info("Saved %s and figures under %s", report_filename, figures_dir)
    return report
