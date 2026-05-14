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
from typing import Dict, Iterable, List

import numpy as np

from tread_highd.src.io_utils import ensure_dir, load_json, save_json

from .baselines import (
    GlobalPOTGPDParams,
    fit_global_pot_gpd,
    predict_quantile_only,
    train_quantile_only,
)
from .data import DatasetArrays, apply_normalization, load_dataset, subset
from .inference import load_model, predict
from .losses import expected_shortfall_np, tail_quantile_np
from .metrics import (
    exceedance_calibration_error,
    expected_shortfall_error,
    gpd_tail_nll,
    tail_quantile_error_by_bin,
)

logger = logging.getLogger(__name__)


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
) -> Dict[str, dict]:
    out = Path(output_dir)
    figures_dir = out / "figures"
    ensure_dir(figures_dir)

    schema = load_json(out / "feature_schema.json")
    norm_stats = load_json(out / "normalization_stats.json")
    arrays = load_dataset(out)
    norm_arrays = apply_normalization(arrays, norm_stats)
    train_arrays = subset(arrays, "train")
    test_arrays = subset(arrays, "test")
    test_norm = subset(norm_arrays, "test")
    alpha_u = float(config.get("training", {}).get("alpha_u", 0.9))
    if len(test_arrays.risk_score) == 0:
        raise RuntimeError("Test split is empty; cannot evaluate DeepEVT.")

    # ---- DeepEVT predictions on test split ----
    model = load_model(checkpoint_path)
    preds = predict(model, test_norm)

    # 计算 test 样本的 q_tau 与 es
    deepevt_q: Dict[float, np.ndarray] = {}
    deepevt_es: Dict[float, np.ndarray] = {}
    for tau in tail_levels:
        deepevt_q[float(tau)] = tail_quantile_np(
            preds.u, preds.p, preds.xi, preds.beta, float(tau)
        )
        deepevt_es[float(tau)] = expected_shortfall_np(
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

    # ---- Metrics ----
    report: Dict[str, dict] = {
        "event_type": schema["event_type"],
        "n_test": int(len(test_arrays.risk_score)),
        "alpha_u": alpha_u,
        "deepevt": {
            "u_mean": float(np.mean(preds.u)),
            "xi_mean": float(np.mean(preds.xi)),
            "beta_mean": float(np.mean(preds.beta)),
            "p_mean": float(np.mean(preds.p)),
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
        "tail_quantile_bins": {},
        "es_error": {},
    }

    for tau in tail_levels:
        tau_f = float(tau)
        q_deep = deepevt_q[tau_f]
        q_global = np.full_like(q_deep, global_params.tail_quantile(tau_f))
        report["ece"][f"tau_{tau_f}"] = {
            "deepevt": exceedance_calibration_error(
                test_arrays.risk_score, q_deep, tau_f
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
        ctx_keys = schema["context_keys"]
        if "gap_current" in ctx_keys:
            feature_name = "gap_current"
        elif "initial_gap" in ctx_keys:
            feature_name = "initial_gap"
        else:
            feature_name = ctx_keys[0]
        fi = ctx_keys.index(feature_name)
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

    save_json(report, out / "eval_report.json")
    logger.info("Saved eval_report.json and figures under %s", figures_dir)
    return report
