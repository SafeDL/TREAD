"""
metrics.py — DeepEVT 评估指标
=============================

实现:
  * ECE_tau: |empirical_exceed - (1-tau)|
  * tail quantile error (按 context bin)
  * GPD tail NLL on exceedances
  * Expected shortfall error
  * Reliability 曲线数据
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


_EPS = 1e-6
_XI_SMALL = 1e-4


def exceedance_calibration_error(
    risk: np.ndarray, q_pred: np.ndarray, tau: float,
) -> Dict[str, float]:
    empirical = float(np.mean(risk > q_pred))
    expected = 1.0 - tau
    return {
        "empirical_exceed_rate": empirical,
        "expected_exceed_rate": expected,
        "ece": float(abs(empirical - expected)),
    }


def tail_quantile_error_by_bin(
    risk: np.ndarray, q_pred: np.ndarray, feature: np.ndarray, tau: float,
    num_bins: int = 4,
) -> List[Dict[str, float]]:
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(feature, qs)
    out: List[Dict[str, float]] = []
    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (feature >= edges[i]) & (feature <= edges[i + 1])
        else:
            mask = (feature >= edges[i]) & (feature < edges[i + 1])
        if mask.sum() < 5:
            continue
        emp_q = float(np.quantile(risk[mask], tau))
        pred_q = float(np.mean(q_pred[mask]))
        out.append({
            "bin_index": i,
            "lower": float(edges[i]),
            "upper": float(edges[i + 1]),
            "n": int(mask.sum()),
            "empirical_quantile": emp_q,
            "predicted_quantile_mean": pred_q,
            "abs_error": float(abs(emp_q - pred_q)),
        })
    return out


def gpd_tail_nll(
    risk: np.ndarray, u: np.ndarray, xi: np.ndarray, beta: np.ndarray,
) -> float:
    """只在 risk > u 的样本上计算 GPD NLL (per-sample mean)."""
    y = risk - u
    mask = y > 0
    if mask.sum() == 0:
        return float("nan")
    y_pos = y[mask]
    xi_pos = xi[mask]
    beta_pos = np.maximum(beta[mask], _EPS)
    is_small = np.abs(xi_pos) < _XI_SMALL
    nll_exp = np.log(beta_pos) + y_pos / beta_pos
    xi_safe = np.where(is_small, _XI_SMALL, xi_pos)
    support = np.maximum(1.0 + xi_safe * y_pos / beta_pos, _EPS)
    nll_gen = np.log(beta_pos) + (1.0 + 1.0 / xi_safe) * np.log(support)
    nll = np.where(is_small, nll_exp, nll_gen)
    return float(np.mean(nll))


def expected_shortfall_error(
    risk: np.ndarray, q_pred: np.ndarray, es_pred: np.ndarray,
) -> float:
    mask = risk > q_pred
    if mask.sum() < 5:
        return float("nan")
    empirical_es = float(np.mean(risk[mask]))
    predicted_es = float(np.mean(es_pred[mask]))
    return float(abs(empirical_es - predicted_es))



