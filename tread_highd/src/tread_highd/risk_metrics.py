"""
risk_metrics.py — 风险指标计算
===============================
计算 TTC、THW、DRAC 以及轨迹级综合风险分数。
参考: 需求文档 §5.5, Matlab SafetyIndicator(), 熵权法论文
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.special import logsumexp as _logsumexp

logger = logging.getLogger(__name__)


def compute_gap(ego_x, target_x, ego_length, target_length):
    """纵向净间距: gap = x_target - x_ego - 0.5*(len_target+len_ego)"""
    return target_x - ego_x - 0.5 * (target_length + ego_length)


def compute_ttc(gap, ego_vx, target_vx, max_ttc=20.0, eps=1e-6):
    """TTC = gap / closing_speed (仅 closing>0 且 gap>0)"""
    closing = ego_vx - target_vx
    ttc = np.full_like(gap, max_ttc, dtype=float)
    v = (closing > eps) & (gap > eps)
    ttc[v] = gap[v] / closing[v]
    return np.clip(ttc, 0.0, max_ttc)


def compute_thw(gap, ego_vx, max_thw=10.0, eps=1e-6):
    """THW = gap / max(ego_vx, eps)"""
    thw = gap / np.maximum(ego_vx, eps)
    return np.clip(thw, 0.0, max_thw)


def compute_drac(gap, ego_vx, target_vx, eps=1e-6):
    """DRAC = closing^2 / (2*gap)  (仅 closing>0 且 gap>0)"""
    closing = ego_vx - target_vx
    drac = np.zeros_like(gap, dtype=float)
    v = (closing > eps) & (gap > eps)
    drac[v] = closing[v] ** 2 / (2.0 * gap[v])
    return drac


def compute_instant_risk(ttc, thw, drac, weights=None, eps=1e-6):
    """S(t) = w_ttc/(TTC+eps) + w_thw/(THW+eps) + w_drac*DRAC"""
    w = weights or {}
    return (w.get("ttc_weight", 1.0) / (ttc + eps)
            + w.get("thw_weight", 0.5) / (thw + eps)
            + w.get("drac_weight", 1.0) * drac)


def compute_trajectory_risk(instant_risk, softmax_lambda=10.0):
    """R = logsumexp(λ*S) / λ"""
    if len(instant_risk) == 0:
        return 0.0
    return float(_logsumexp(softmax_lambda * instant_risk) / softmax_lambda)


def entropy_weight_method(data, eps=1e-12):
    """熵权法计算各指标客观权重 (参考 Efficient and Unbiased Safety Test)"""
    n, m = data.shape
    if n == 0 or m == 0:
        return np.ones(m) / max(m, 1)
    col_sums = np.maximum(data.sum(axis=0), eps)
    p = data / col_sums
    p_safe = np.where(p < eps, eps, p)
    k = 1.0 / np.log(max(n, 2))
    entropy = -k * np.sum(p_safe * np.log(p_safe), axis=0)
    d = 1.0 - entropy
    d_sum = d.sum()
    if d_sum < eps:
        return np.ones(m) / m
    return d / d_sum
