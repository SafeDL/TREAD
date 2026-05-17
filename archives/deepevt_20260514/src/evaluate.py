"""
evaluate.py — DeepEVT 测试集评估
==================================

比较:
  * DeepEVT   (条件 GPD)
  * GlobalPOT (固定阈值 GPD)
  * QuantileOnly neural baseline (可选)

输出:
  eval_report.json
  figures/calibration_q95.png
  figures/gpd_qq_plot.png
  figures/tail_quantile_error.png
  figures/predicted_vs_empirical_exceedance.png
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from process_highd.src.io_utils import ensure_dir, load_json, save_json

from .baselines import (
    GlobalPOTGPDParams,
    fit_global_pot_gpd,
    predict_quantile_only,
    train_quantile_only,
)
from .data import DatasetArrays, apply_normalization, load_dataset, subset
from .inference import load_model, predict
from .losses import expected_shortfall_np, tail_quantile_invalid_mask, tail_quantile_np
from .metrics import (
    exceedance_calibration_error,
    expected_shortfall_error,
    gpd_tail_nll,
    tail_quantile_error_by_bin,
)

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


def _reliability_bins(
    risk: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    num_bins: int = 5,
) -> List[Dict[str, float]]:
    edges = np.quantile(p, np.linspace(0.0, 1.0, num_bins + 1))
    out: List[Dict[str, float]] = []
    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() < 5:
            continue
        out.append({
            "bin_index": i,
            "lower": float(edges[i]),
            "upper": float(edges[i + 1]),
            "n": int(mask.sum()),
            "p_mean": float(np.mean(p[mask])),
            "empirical_exceed_u": float(np.mean(risk[mask] > u[mask])),
        })
    return out


def _spearman_from_small_arrays(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _reliability_summary(bins: List[Dict[str, float]]) -> Dict[str, float]:
    if not bins:
        return {
            "mean_abs_error": float("nan"),
            "max_abs_error": float("nan"),
            "signed_mean_error": float("nan"),
            "spearman_p_vs_empirical": float("nan"),
            "monotonic_violations": 0,
        }
    p_mean = np.asarray([b["p_mean"] for b in bins], dtype=np.float64)
    empirical = np.asarray([b["empirical_exceed_u"] for b in bins], dtype=np.float64)
    diff = p_mean - empirical
    violations = int(np.sum(np.diff(empirical) < 0.0))
    return {
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "signed_mean_error": float(np.mean(diff)),
        "spearman_p_vs_empirical": _spearman_from_small_arrays(p_mean, empirical),
        "monotonic_violations": violations,
    }


def _mean_abs_error_from_bins(bins_info: List[Dict[str, float]]) -> float:
    if not bins_info:
        return float("nan")
    return float(np.mean([float(b["abs_error"]) for b in bins_info]))


def _fit_rate_scale_calibration(
    risk: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
) -> Dict[str, float]:
    raw_mean = float(np.mean(p))
    empirical = float(np.mean(risk > u))
    scale = empirical / max(raw_mean, 1e-6)
    return {
        "method": "rate_scale",
        "scale": float(scale),
        "raw_p_mean": raw_mean,
        "empirical_exceed_u": empirical,
    }


def _apply_rate_scale_calibration(p: np.ndarray, calibration: Dict[str, float]) -> np.ndarray:
    return np.clip(p * float(calibration.get("scale", 1.0)), 0.0, 1.0)


def _pava_fit(y: np.ndarray, *, increasing: bool = True) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64)
    if not increasing:
        return -_pava_fit(-target, increasing=True)
    blocks: List[Dict[str, float]] = []
    for idx, val in enumerate(target):
        blocks.append({
            "start": float(idx),
            "end": float(idx),
            "weight": 1.0,
            "sum": float(val),
            "level": float(val),
        })
        while len(blocks) >= 2 and blocks[-2]["level"] > blocks[-1]["level"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            total = left["sum"] + right["sum"]
            blocks.append({
                "start": left["start"],
                "end": right["end"],
                "weight": weight,
                "sum": total,
                "level": total / max(weight, 1e-12),
            })
    fitted = np.empty_like(target, dtype=np.float64)
    for block in blocks:
        start = int(block["start"])
        end = int(block["end"])
        fitted[start:end + 1] = block["level"]
    return fitted


def _fit_isotonic_calibration(
    risk: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    *,
    direction: str,
) -> Dict[str, object]:
    labels = (risk > u).astype(np.float64)
    order = np.argsort(p)
    x_sorted = np.asarray(p[order], dtype=np.float64)
    y_sorted = labels[order]
    increasing = direction == "increasing"
    fitted_sorted = np.clip(_pava_fit(y_sorted, increasing=increasing), 0.0, 1.0)
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    unique_y = np.zeros_like(unique_x, dtype=np.float64)
    counts = np.zeros_like(unique_x, dtype=np.float64)
    for idx, group in enumerate(inverse):
        unique_y[group] += fitted_sorted[idx]
        counts[group] += 1.0
    unique_y = unique_y / np.maximum(counts, 1.0)
    change = np.r_[True, np.abs(np.diff(unique_y)) > 1e-12]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:] - 1, len(unique_x) - 1]
    thresholds = unique_x[ends]
    values = unique_y[starts]
    return {
        "method": "isotonic",
        "direction": direction,
        "thresholds": thresholds.tolist(),
        "values": values.tolist(),
        "num_input_knots": int(len(unique_x)),
        "num_output_steps": int(len(values)),
        "raw_p_mean": float(np.mean(p)),
        "empirical_exceed_u": float(np.mean(labels)),
    }


def _apply_isotonic_calibration(p: np.ndarray, calibration: Dict[str, object]) -> np.ndarray:
    thresholds = np.asarray(calibration.get("thresholds", []), dtype=np.float64)
    values = np.asarray(calibration.get("values", []), dtype=np.float64)
    if len(thresholds) == 0 or len(values) == 0:
        return np.asarray(p, dtype=np.float64)
    idx = np.searchsorted(thresholds, p, side="left")
    idx = np.clip(idx, 0, len(values) - 1)
    return np.clip(values[idx], 0.0, 1.0)


def _p_calibration_score(
    risk: np.ndarray,
    u: np.ndarray,
    p_cal: np.ndarray,
) -> float:
    labels = (risk > u).astype(np.float64)
    bins = _reliability_bins(risk, u, p_cal)
    summary = _reliability_summary(bins)
    brier = float(np.mean((p_cal - labels) ** 2))
    monotonic_penalty = 0.02 * float(summary.get("monotonic_violations", 0))
    spearman = float(summary.get("spearman_p_vs_empirical", 0.0))
    if not np.isfinite(spearman):
        spearman = 0.0
    return brier + monotonic_penalty - 0.01 * spearman


def _fit_p_calibration(
    risk: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    method: str,
) -> Dict[str, object]:
    method = method.lower()
    if method == "rate_scale":
        return _fit_rate_scale_calibration(risk, u, p)
    if method in {"isotonic_increasing", "isotonic_decreasing"}:
        direction = method.removeprefix("isotonic_")
        calibration = _fit_isotonic_calibration(risk, u, p, direction=direction)
        calibration["selection_score"] = _p_calibration_score(
            risk, u, _apply_isotonic_calibration(p, calibration)
        )
        return calibration
    if method == "isotonic_auto":
        candidates: List[Dict[str, object]] = []
        rate_scale = _fit_rate_scale_calibration(risk, u, p)
        rate_scale["selection_score"] = _p_calibration_score(
            risk, u, _apply_rate_scale_calibration(p, rate_scale)
        )
        candidates.append(rate_scale)
        for direction in ("increasing", "decreasing"):
            cal = _fit_isotonic_calibration(risk, u, p, direction=direction)
            cal["selection_score"] = _p_calibration_score(
                risk, u, _apply_isotonic_calibration(p, cal)
            )
            candidates.append(cal)
        best = min(candidates, key=lambda c: float(c.get("selection_score", float("inf"))))
        best = dict(best)
        best["method"] = (
            "isotonic_auto:"
            + str(best.get("direction", "rate_scale"))
            if best.get("method") == "isotonic"
            else "isotonic_auto:rate_scale"
        )
        return best
    raise ValueError(f"Unsupported p calibration method: {method}")


def _apply_p_calibration(p: np.ndarray, calibration: Dict[str, object]) -> np.ndarray:
    method = str(calibration.get("method", "rate_scale"))
    if method == "rate_scale" or method.endswith(":rate_scale"):
        return _apply_rate_scale_calibration(p, calibration)
    if method == "isotonic" or method.startswith("isotonic_auto:"):
        return _apply_isotonic_calibration(p, calibration)
    raise ValueError(f"Unsupported p calibration method in report: {method}")


def _fit_gap_bin_shrink_calibration(
    risk: np.ndarray,
    q_pred: np.ndarray,
    feature: np.ndarray,
    tau: float,
    *,
    num_bins: int,
    shrink_gamma: float,
) -> Dict[str, object]:
    edges = np.quantile(feature, np.linspace(0.0, 1.0, num_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bins: List[Dict[str, float]] = []
    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (feature >= edges[i]) & (feature <= edges[i + 1])
        else:
            mask = (feature >= edges[i]) & (feature < edges[i + 1])
        if mask.sum() < 5:
            continue
        bins.append({
            "bin_index": i,
            "lower": float(edges[i]),
            "upper": float(edges[i + 1]),
            "n": int(mask.sum()),
            "q_pred_mean": float(np.mean(q_pred[mask])),
            "empirical_quantile": float(np.quantile(risk[mask], tau)),
        })
    return {
        "method": "gap_bin_shrink",
        "tau": float(tau),
        "num_bins": int(num_bins),
        "shrink_gamma": float(shrink_gamma),
        "bins": bins,
    }


def _apply_gap_bin_shrink_calibration(
    q_pred: np.ndarray,
    feature: np.ndarray,
    calibration: Dict[str, object],
) -> np.ndarray:
    q_cal = np.array(q_pred, copy=True)
    gamma = float(calibration.get("shrink_gamma", 0.0))
    for b in calibration.get("bins", []):
        lower = float(b["lower"])
        upper = float(b["upper"])
        if int(b["bin_index"]) == int(calibration.get("num_bins", 1)) - 1:
            mask = (feature >= lower) & (feature <= upper)
        else:
            mask = (feature >= lower) & (feature < upper)
        q_cal[mask] = float(b["empirical_quantile"]) + gamma * (
            q_pred[mask] - float(b["q_pred_mean"])
        )
    return q_cal


def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _calibration_plot(
    risk: np.ndarray, q_pred: np.ndarray, tau: float, path: Path,
) -> None:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    order = np.argsort(q_pred)
    q_sorted = q_pred[order]
    r_sorted = risk[order]
    ax.scatter(q_sorted, r_sorted, s=4, alpha=0.3, label="samples")
    lims = [float(min(q_sorted.min(), r_sorted.min())),
            float(max(q_sorted.max(), r_sorted.max()))]
    ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
    ax.set_xlabel(f"predicted q{int(tau * 100)}")
    ax.set_ylabel("observed risk")
    ax.set_title(f"Calibration at tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _gpd_qq_plot(risk: np.ndarray, u: np.ndarray, xi: np.ndarray,
                 beta: np.ndarray, path: Path) -> None:
    plt = _lazy_plt()
    y = risk - u
    mask = y > 0
    if mask.sum() < 10:
        logger.warning("Too few exceedances for QQ plot (%d).", mask.sum())
        return
    y_pos = y[mask]
    empirical = np.sort(y_pos)
    n = len(empirical)
    probs = (np.arange(1, n + 1) - 0.5) / n
    # theoretical quantile from per-sample GPD
    xi_pos = xi[mask]; beta_pos = beta[mask]
    theo = np.empty(n)
    for i, q in enumerate(probs):
        th = np.where(
            np.abs(xi_pos) < 1e-4,
            -beta_pos * np.log(1.0 - q),
            beta_pos / np.maximum(xi_pos, 1e-6) * (np.power(1.0 - q, -xi_pos) - 1.0),
        )
        theo[i] = float(np.median(th))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(theo, empirical, s=6)
    lims = [0, float(max(theo.max(), empirical.max()))]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("theoretical GPD quantile")
    ax.set_ylabel("empirical exceedance")
    ax.set_title("GPD QQ plot")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _tail_quantile_error_plot(
    bins_info: List[dict], tau: float, path: Path, feature_name: str,
) -> None:
    if not bins_info:
        return
    plt = _lazy_plt()
    idxs = [b["bin_index"] for b in bins_info]
    emp = [b["empirical_quantile"] for b in bins_info]
    pred = [b["predicted_quantile_mean"] for b in bins_info]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(idxs, emp, "o-", label="empirical")
    ax.plot(idxs, pred, "s--", label="predicted")
    ax.set_xlabel(f"{feature_name} bin")
    ax.set_ylabel(f"q{int(tau * 100)}")
    ax.set_title(f"Tail quantile error @ tau={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _predicted_vs_empirical_exceedance(
    risk: np.ndarray, q_pred: np.ndarray, tau: float, path: Path,
) -> None:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(q_pred.min(), q_pred.max(), 11)
    idx = np.clip(np.digitize(q_pred, bins) - 1, 0, len(bins) - 2)
    xs, ys = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(float(bins[b]))
        ys.append(float((risk[m] > q_pred[m]).mean()))
    ax.plot(xs, ys, "o-")
    ax.axhline(1.0 - tau, color="red", linestyle="--", label=f"target {1-tau:.2f}")
    ax.set_xlabel(f"predicted q{int(tau * 100)}")
    ax.set_ylabel("empirical exceedance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_deepevt(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    config: dict,
    run_quantile_baseline: bool = True,
    tail_levels: Iterable[float] = (0.90, 0.95),
    report_name: Optional[str] = None,
) -> Dict[str, dict]:
    out = Path(output_dir)
    figures_dir = out / "figures"
    ensure_dir(figures_dir)

    schema = load_json(out / "feature_schema.json")
    norm_stats = load_json(out / "normalization_stats.json")
    arrays = load_dataset(out)
    norm_arrays = apply_normalization(arrays, norm_stats)
    train_arrays = subset(arrays, "train")
    val_arrays = subset(arrays, "val")
    val_norm = subset(norm_arrays, "val")
    test_arrays = subset(arrays, "test")
    test_norm = subset(norm_arrays, "test")
    alpha_u = float(config.get("training", {}).get("alpha_u", 0.9))
    if len(test_arrays.risk_score) == 0:
        raise RuntimeError("Test split is empty; cannot evaluate DeepEVT.")

    # ---- DeepEVT predictions on test split ----
    model = load_model(checkpoint_path)
    preds = predict(model, test_norm)
    val_preds = predict(model, val_norm)
    eval_cfg = config.get("evaluation", {})

    ctx_keys = schema["context_keys"]
    if "gap_current" in ctx_keys:
        default_bin_feature = "gap_current"
    elif "initial_gap" in ctx_keys:
        default_bin_feature = "initial_gap"
    else:
        default_bin_feature = ctx_keys[0]
    default_bin_feature_index = ctx_keys.index(default_bin_feature)

    p_report = preds.p
    p_report_val = val_preds.p
    p_calibration_report = None
    p_cal_cfg = eval_cfg.get("p_calibration", {})
    if bool(p_cal_cfg.get("enabled", False)):
        p_calibration_report = _fit_p_calibration(
            val_arrays.risk_score,
            val_preds.u,
            val_preds.p,
            str(p_cal_cfg.get("method", "rate_scale")),
        )
        p_report = _apply_p_calibration(preds.p, p_calibration_report)
        p_report_val = _apply_p_calibration(val_preds.p, p_calibration_report)

    # 计算 test 样本的 q_tau 与 es
    deepevt_q: Dict[float, np.ndarray] = {}
    deepevt_q_gpd: Dict[float, np.ndarray] = {}
    val_deepevt_q: Dict[float, np.ndarray] = {}
    val_deepevt_q_gpd: Dict[float, np.ndarray] = {}
    deepevt_es: Dict[float, np.ndarray] = {}
    q_calibrations: Dict[float, Dict[str, object]] = {}
    q_cal_cfg = eval_cfg.get("tail_quantile_calibration", {})
    q_cal_enabled = bool(q_cal_cfg.get("enabled", False))
    q_cal_levels = {float(tau) for tau in q_cal_cfg.get("levels", [])}
    q_cal_feature = str(q_cal_cfg.get("feature", default_bin_feature))
    q_cal_feature_index = ctx_keys.index(q_cal_feature) if q_cal_feature in ctx_keys else default_bin_feature_index
    for tau in tail_levels:
        tau_f = float(tau)
        q_gpd = tail_quantile_np(
            preds.u, preds.p, preds.xi, preds.beta, float(tau)
        )
        q_val_gpd = tail_quantile_np(
            val_preds.u, val_preds.p, val_preds.xi, val_preds.beta, float(tau)
        )
        deepevt_q_gpd[tau_f] = q_gpd
        val_deepevt_q_gpd[tau_f] = q_val_gpd
        deepevt_q[tau_f] = q_gpd
        val_deepevt_q[tau_f] = q_val_gpd
        if q_cal_enabled and tau_f in q_cal_levels:
            calibration = _fit_gap_bin_shrink_calibration(
                val_arrays.risk_score,
                q_val_gpd,
                val_arrays.context_features[:, q_cal_feature_index],
                tau_f,
                num_bins=int(q_cal_cfg.get("num_bins", 4)),
                shrink_gamma=float(q_cal_cfg.get("shrink_gamma", 0.1)),
            )
            q_calibrations[tau_f] = calibration
            deepevt_q[tau_f] = _apply_gap_bin_shrink_calibration(
                q_gpd,
                test_arrays.context_features[:, q_cal_feature_index],
                calibration,
            )
            val_deepevt_q[tau_f] = _apply_gap_bin_shrink_calibration(
                q_val_gpd,
                val_arrays.context_features[:, q_cal_feature_index],
                calibration,
            )
        deepevt_es[tau_f] = expected_shortfall_np(
            preds.u, preds.p, preds.xi, preds.beta, float(tau)
        )

    # ---- Global POT-GPD baseline ----
    global_params: GlobalPOTGPDParams = fit_global_pot_gpd(
        train_arrays.risk_score, alpha_u=alpha_u
    )
    logger.info("Global POT-GPD: u=%.3f  xi=%.3f  beta=%.3f  p=%.3f",
                global_params.u, global_params.xi, global_params.beta,
                global_params.p)

    # ---- QuantileOnly baseline (optional) ----
    quantile_only_preds: Dict[float, np.ndarray] = {}
    if run_quantile_baseline:
        train_norm = subset(norm_arrays, "train")
        val_norm = subset(norm_arrays, "val")
        try:
            qmodel = train_quantile_only(train_norm, val_norm, config)
            for tau in tail_levels:
                # 用同一网络在不同 alpha 上微调的成本较高；第一版只报告 alpha_u 下的预测
                if abs(tau - alpha_u) < 1e-3:
                    quantile_only_preds[float(tau)] = predict_quantile_only(qmodel, test_norm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QuantileOnly baseline failed: %s", exc)

    test_p_bins = _reliability_bins(
        test_arrays.risk_score, preds.u, p_report,
    )

    def _split_diagnostics(
        arrays_for_split: DatasetArrays,
        split_preds,
        split_p_report: np.ndarray,
        split_q: Dict[float, np.ndarray],
        split_q_gpd: Dict[float, np.ndarray],
    ) -> Dict[str, object]:
        p_bins = _reliability_bins(
            arrays_for_split.risk_score, split_preds.u, split_p_report,
        )
        diag: Dict[str, object] = {
            "n": int(len(arrays_for_split.risk_score)),
            "empirical_exceed_u": float(np.mean(arrays_for_split.risk_score > split_preds.u)),
            "raw_p_mean": float(np.mean(split_preds.p)),
            "calibrated_p_mean": float(np.mean(split_p_report)),
            "p_reliability_bins": p_bins,
            "p_reliability_summary": _reliability_summary(p_bins),
            "tail_levels": {},
        }
        for tau in tail_levels:
            tau_f = float(tau)
            q_cal = split_q[tau_f]
            q_raw = split_q_gpd[tau_f]
            bins_cal = tail_quantile_error_by_bin(
                arrays_for_split.risk_score,
                q_cal,
                arrays_for_split.context_features[:, default_bin_feature_index],
                tau_f,
                num_bins=4,
            )
            bins_raw = tail_quantile_error_by_bin(
                arrays_for_split.risk_score,
                q_raw,
                arrays_for_split.context_features[:, default_bin_feature_index],
                tau_f,
                num_bins=4,
            )
            invalid = tail_quantile_invalid_mask(split_preds.p, tau_f)
            diag["tail_levels"][f"tau_{tau_f}"] = {
                "calibrated": {
                    "ece": exceedance_calibration_error(
                        arrays_for_split.risk_score, q_cal, tau_f,
                    ),
                    "q_distribution": _distribution_summary(q_cal),
                    "bin_mean_abs_error": _mean_abs_error_from_bins(bins_cal),
                },
                "raw_gpd": {
                    "ece": exceedance_calibration_error(
                        arrays_for_split.risk_score, q_raw, tau_f,
                    ),
                    "q_distribution": _distribution_summary(q_raw),
                    "bin_mean_abs_error": _mean_abs_error_from_bins(bins_raw),
                },
                "invalid_rate": float(np.mean(invalid)),
            }
        return diag

    # ---- Metrics ----
    report: Dict[str, dict] = {
        "event_type": schema["event_type"],
        "n_test": int(len(test_arrays.risk_score)),
        "alpha_u": alpha_u,
        "tail_quantile_source": "gpd",
        "tail_quantile_calibration": q_calibrations,
        "p_calibration": p_calibration_report,
        "deepevt": {
            "u_mean": float(np.mean(preds.u)),
            "xi_mean": float(np.mean(preds.xi)),
            "beta_mean": float(np.mean(preds.beta)),
            "p_mean": float(np.mean(p_report)),
            "raw_p_mean": float(np.mean(preds.p)),
            "u_distribution": _distribution_summary(preds.u),
            "xi_distribution": _distribution_summary(preds.xi),
            "beta_distribution": _distribution_summary(preds.beta),
            "p_distribution": _distribution_summary(p_report),
            "raw_p_distribution": _distribution_summary(preds.p),
            "empirical_exceed_u": float(np.mean(test_arrays.risk_score > preds.u)),
            "p_reliability_bins": test_p_bins,
            "p_reliability_summary": _reliability_summary(test_p_bins),
            "u_scale_mean": float(np.mean(preds.u_scale)),
            "xi_scale_mean": float(np.mean(preds.xi_scale)),
            "beta_scale_mean": float(np.mean(preds.beta_scale)),
            "gpd_tail_nll": gpd_tail_nll(
                test_arrays.risk_score, preds.u, preds.xi, preds.beta,
            ),
        },
        "global_pot_gpd": {
            "u": global_params.u, "xi": global_params.xi,
            "beta": global_params.beta, "p": global_params.p,
        },
        "ece": {},
        "tail_quantile_diagnostics": {},
        "tail_quantile_bins": {},
        "split_diagnostics": {
            "validation": _split_diagnostics(
                val_arrays, val_preds, p_report_val, val_deepevt_q, val_deepevt_q_gpd,
            ),
            "test": _split_diagnostics(
                test_arrays, preds, p_report, deepevt_q, deepevt_q_gpd,
            ),
        },
        "es_error": {},
    }

    for tau in tail_levels:
        tau_f = float(tau)
        q_deep = deepevt_q[tau_f]
        q_gpd = deepevt_q_gpd[tau_f]
        q_global = np.full_like(q_deep, global_params.tail_quantile(tau_f))
        invalid = tail_quantile_invalid_mask(preds.p, tau_f)
        report["tail_quantile_diagnostics"][f"tau_{tau_f}"] = {
            "deepevt": {
                "source": (
                    "gap_bin_shrink_calibrated_gpd"
                    if tau_f in q_calibrations
                    else "gpd"
                ),
                "q_distribution": _distribution_summary(q_deep),
                "invalid_rate": float(np.mean(invalid)),
                "valid_rate": float(1.0 - np.mean(invalid)),
                "invalid_count": int(np.sum(invalid)),
                "valid_count": int(len(invalid) - np.sum(invalid)),
                "mean_q_minus_u": float(np.mean(q_deep - preds.u)),
            },
            "deepevt_gpd": {
                "q_distribution": _distribution_summary(q_gpd),
                "ece": exceedance_calibration_error(test_arrays.risk_score, q_gpd, tau_f),
                "mean_q_minus_u": float(np.mean(q_gpd - preds.u)),
            },
            "global_pot_gpd": {
                "q": float(global_params.tail_quantile(tau_f)),
            },
        }
        report["ece"][f"tau_{tau_f}"] = {
            "deepevt": exceedance_calibration_error(
                test_arrays.risk_score, q_deep, tau_f
            ),
            "deepevt_gpd": exceedance_calibration_error(
                test_arrays.risk_score, q_gpd, tau_f
            ),
            "global_pot_gpd": exceedance_calibration_error(
                test_arrays.risk_score, q_global, tau_f
            ),
        }
        if tau_f in quantile_only_preds:
            report["ece"][f"tau_{tau_f}"]["quantile_only"] = (
                exceedance_calibration_error(
                    test_arrays.risk_score, quantile_only_preds[tau_f], tau_f,
                )
            )

        # ES error for reported extreme levels.
        if tau_f >= 0.95:
            report["es_error"][f"tau_{tau_f}"] = {
                "deepevt": expected_shortfall_error(
                    test_arrays.risk_score, q_deep, deepevt_es[tau_f],
                ),
            }

        # bin analysis — choose feature per event type
        feature_name = default_bin_feature
        fi = default_bin_feature_index
        bins_info = tail_quantile_error_by_bin(
            test_arrays.risk_score, q_deep,
            test_arrays.context_features[:, fi], tau_f, num_bins=4,
        )
        report["tail_quantile_bins"][f"tau_{tau_f}"] = {
            "feature": feature_name,
            "deepevt": bins_info,
        }

        # figures for reported extreme levels.
        if tau_f >= 0.95:
            _calibration_plot(
                test_arrays.risk_score, q_deep, tau_f,
                figures_dir / f"calibration_q{int(tau_f * 100)}.png",
            )
            _predicted_vs_empirical_exceedance(
                test_arrays.risk_score, q_deep, tau_f,
                figures_dir / f"predicted_vs_empirical_exceedance_q{int(tau_f * 100)}.png",
            )

    # additional figures
    _gpd_qq_plot(
        test_arrays.risk_score, preds.u, preds.xi, preds.beta,
        figures_dir / "gpd_qq_plot.png",
    )
    bins_info_95 = report["tail_quantile_bins"].get("tau_0.95", {}).get("deepevt", [])
    _tail_quantile_error_plot(
        bins_info_95, 0.95,
        figures_dir / "tail_quantile_error.png",
        feature_name=report["tail_quantile_bins"].get("tau_0.95", {}).get("feature", "feature"),
    )

    report_filename = report_name or "eval_report.json"
    save_json(report, out / report_filename)
    logger.info("Saved %s and figures under %s", report_filename, figures_dir)
    return report
