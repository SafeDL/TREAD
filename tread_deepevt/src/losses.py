"""
losses.py — DeepEVT 损失与尾部分位公式
======================================

训练损失使用 torch.Tensor；尾部分位和 Expected Shortfall 的 numpy 版本供评估与推理使用。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_EPS_SMALL = 1e-6
_XI_SMALL = 1e-4


# ---------------------------------------------------------------------------
# Quantile (pinball) loss
# ---------------------------------------------------------------------------

def pinball_loss(target: torch.Tensor, u: torch.Tensor, alpha: float) -> torch.Tensor:
    """Pinball / quantile loss at level ``alpha``。"""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    e = target - u
    return torch.mean(torch.maximum(alpha * e, (alpha - 1.0) * e))


def pinball_loss_per_sample(target: torch.Tensor, u: torch.Tensor, alpha: float) -> torch.Tensor:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    e = target - u
    return torch.maximum(alpha * e, (alpha - 1.0) * e)


# ---------------------------------------------------------------------------
# Exceedance head loss
# ---------------------------------------------------------------------------

def exceedance_bce(target: torch.Tensor, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """给定阈值 u 的超阈值概率预测 p 的 BCE；u detach 以免梯度穿透。"""
    exceed = (target > u.detach()).float()
    p_clamped = p.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(p_clamped, exceed)


# ---------------------------------------------------------------------------
# GPD negative log-likelihood
# ---------------------------------------------------------------------------

def gpd_nll(
    y: torch.Tensor,
    xi: torch.Tensor,
    beta: torch.Tensor,
    eps: float = _EPS_SMALL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised Pareto NLL。同时返回 support penalty (未加权)。

    支持 ``|xi| < _XI_SMALL`` 使用指数极限以避免数值问题。
    仅对 ``y > 0`` 的样本计算 NLL，若没有任何超额样本返回 0 张量。
    """
    mask = y > 0
    if mask.sum() == 0:
        zero = torch.zeros((), device=y.device, dtype=y.dtype)
        return zero, zero

    y_pos = y[mask]
    xi_pos = xi[mask]
    beta_pos = beta[mask].clamp_min(eps)

    support = 1.0 + xi_pos * y_pos / beta_pos
    # penalty 对 support <= eps 的样本施加二次惩罚（与 relu(eps - support)**2 等价）
    penalty = F.relu(eps - support).pow(2).mean()

    safe_support = support.clamp_min(eps)
    log_support = torch.log(safe_support)

    # |xi| 非常小的时候使用指数极限: log(beta) + y/beta
    is_small = xi_pos.abs() < _XI_SMALL
    nll_exp = torch.log(beta_pos) + y_pos / beta_pos
    # 通常 GPD:  log(beta) + (1 + 1/xi) * log(1 + xi*y/beta)
    xi_safe = torch.where(is_small, torch.full_like(xi_pos, _XI_SMALL), xi_pos)
    nll_gen = torch.log(beta_pos) + (1.0 + 1.0 / xi_safe) * log_support
    nll = torch.where(is_small, nll_exp, nll_gen)
    return nll.mean(), penalty


# ---------------------------------------------------------------------------
# Calibration loss (soft exceedance rate)
# ---------------------------------------------------------------------------

def calibration_loss(
    target: torch.Tensor, u: torch.Tensor, alpha: float, delta: float = 0.05,
) -> torch.Tensor:
    """软约束：经过 sigmoid 后的平均超阈值比例应接近 ``1 - alpha``。"""
    soft_exceed = torch.sigmoid((target - u) / max(delta, 1e-6))
    return (soft_exceed.mean() - (1.0 - alpha)).pow(2)


# ---------------------------------------------------------------------------
# Combined loss helper
# ---------------------------------------------------------------------------

def deepevt_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    alpha: float,
    weights: Dict[str, float],
    use_exceedance_head: bool = True,
    include_gpd: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """按权重组合 quantile / exceedance / GPD / calibration / support。"""
    u = outputs["u"]
    q_per = pinball_loss_per_sample(target, u, alpha)
    loss_q = q_per.mean()
    if "u_log_scale" in outputs and weights.get("lambda_u_unc", 0.0) > 0:
        u_scale = torch.exp(outputs["u_log_scale"]).clamp_min(_EPS_SMALL)
        loss_u_unc = (q_per / u_scale + outputs["u_log_scale"]).mean()
    else:
        loss_u_unc = torch.zeros((), device=target.device, dtype=target.dtype)
    loss_cal = calibration_loss(
        target, u, alpha, delta=weights.get("cal_delta", 0.05),
    )

    log: Dict[str, float] = {
        "loss_q": float(loss_q.detach().item()),
        "loss_u_unc": float(loss_u_unc.detach().item()),
        "loss_cal": float(loss_cal.detach().item()),
    }

    total = weights.get("lambda_q", 1.0) * loss_q + weights.get("lambda_cal", 0.5) * loss_cal
    total = total + weights.get("lambda_u_unc", 0.0) * loss_u_unc

    if use_exceedance_head and "p" in outputs:
        loss_exc = exceedance_bce(target, u, outputs["p"])
        total = total + weights.get("lambda_exc", 0.2) * loss_exc
        log["loss_exc"] = float(loss_exc.detach().item())

    if include_gpd and "xi" in outputs and "beta" in outputs:
        y = target - u.detach()
        nll, support_penalty = gpd_nll(y, outputs["xi"], outputs["beta"])
        total = total + weights.get("lambda_gpd", 1.0) * nll
        total = total + weights.get("lambda_support", 10.0) * support_penalty
        log["loss_gpd"] = float(nll.detach().item())
        log["loss_support"] = float(support_penalty.detach().item())

        xi_prior = float(weights.get("xi_prior", 0.0))
        loss_xi_reg = (outputs["xi"] - xi_prior).pow(2).mean()
        beta_ref = float(weights.get("beta_ref", 1.0))
        beta_ref = max(beta_ref, _EPS_SMALL)
        loss_beta_reg = torch.log(outputs["beta"].clamp_min(_EPS_SMALL) / beta_ref).pow(2).mean()
        total = total + weights.get("lambda_xi_reg", 0.0) * loss_xi_reg
        total = total + weights.get("lambda_beta_reg", 0.0) * loss_beta_reg
        log["loss_xi_reg"] = float(loss_xi_reg.detach().item())
        log["loss_beta_reg"] = float(loss_beta_reg.detach().item())

        if "xi_log_scale" in outputs and "beta_log_scale" in outputs:
            min_log_scale = float(weights.get("min_tail_log_scale", -2.0))
            xi_conf = F.relu(min_log_scale - outputs["xi_log_scale"]).pow(2).mean()
            beta_conf = F.relu(min_log_scale - outputs["beta_log_scale"]).pow(2).mean()
            loss_tail_unc = xi_conf + beta_conf
            total = total + weights.get("lambda_tail_unc", 0.0) * loss_tail_unc
            log["loss_tail_unc"] = float(loss_tail_unc.detach().item())

    log["loss_total"] = float(total.detach().item())
    return total, log


# ---------------------------------------------------------------------------
# Tail quantile (closed-form GPD extrapolation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Numpy variants — used during eval / inference without torch tensors
# ---------------------------------------------------------------------------

def tail_quantile_np(
    u: np.ndarray, p: np.ndarray, xi: np.ndarray, beta: np.ndarray,
    tau: float, eps: float = _EPS_SMALL,
) -> np.ndarray:
    """GPD 外推分位 numpy 版。

    ``p <= 1 - tau`` 时 GPD 外推到 ``tau`` 不再有意义,会得到 ``q < u``。
    此处把 ``p`` clamp 到 ``1 - tau + eps`` 以保证数值稳定;调用方若关心
    "无效样本"比例,应配合 :func:`tail_quantile_invalid_mask` 单独标记。
    """
    p_min = 1.0 - tau + eps
    p_safe = np.maximum(p, p_min)
    frac = p_safe / (1.0 - tau + eps)
    is_small = np.abs(xi) < _XI_SMALL
    xi_safe = np.where(is_small, _XI_SMALL, xi)
    q_gen = u + beta / xi_safe * (np.power(frac, xi_safe) - 1.0)
    q_exp = u + beta * np.log(frac)
    return np.where(is_small, q_exp, q_gen)


def tail_quantile_invalid_mask(p: np.ndarray, tau: float) -> np.ndarray:
    """``p <= 1 - tau`` 的样本对应 tail extrapolation 无效,返回布尔掩码。"""
    return p <= (1.0 - tau)


def expected_shortfall_np(
    u: np.ndarray, p: np.ndarray, xi: np.ndarray, beta: np.ndarray,
    tau: float, eps: float = _EPS_SMALL,
) -> np.ndarray:
    q = tail_quantile_np(u, p, xi, beta, tau, eps=eps)
    denom = np.maximum(1.0 - xi, eps)
    es = q + (beta + xi * (q - u)) / denom
    es = np.where(xi >= 1.0 - _XI_SMALL, np.nan, es)
    return es
