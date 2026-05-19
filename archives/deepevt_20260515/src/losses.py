"""
losses.py — direct conditional quantile losses.

The current DeepEVT objective is deliberately narrow:

* direct q85/q90/q95 pinball loss;
* soft exceedance calibration for each quantile;
* optional pairwise ranking loss on the primary tail score.

No GPD/POT/exceedance-probability losses are part of the training path.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def pinball_loss(target: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball / quantile loss at level ``tau``."""
    if not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    err = target - q
    return torch.mean(torch.maximum(float(tau) * err, (float(tau) - 1.0) * err))


def pinball_loss_per_sample(target: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    if not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    err = target - q
    return torch.maximum(float(tau) * err, (float(tau) - 1.0) * err)


def multi_quantile_loss(
    target: torch.Tensor,
    quantiles: torch.Tensor,
    levels: Tuple[float, ...],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Average pinball loss over directly predicted conditional quantiles."""
    if quantiles.ndim != 2:
        raise ValueError(f"quantiles must be [batch, levels], got {tuple(quantiles.shape)}")
    if quantiles.shape[1] != len(levels):
        raise ValueError(
            f"quantile level count mismatch: tensor has {quantiles.shape[1]}, "
            f"levels={levels}"
        )
    losses = []
    logs: Dict[str, float] = {}
    for i, tau in enumerate(levels):
        loss_i = pinball_loss(target, quantiles[:, i], float(tau))
        losses.append(loss_i)
        logs[f"loss_q{int(round(float(tau) * 100))}"] = float(loss_i.detach().item())
    return torch.stack(losses).mean(), logs


def calibration_loss(
    target: torch.Tensor,
    q: torch.Tensor,
    tau: float,
    *,
    delta: float = 0.05,
) -> torch.Tensor:
    """Softly match the exceedance rate of q_tau to ``1 - tau``."""
    soft_exceed = torch.sigmoid((target - q) / max(float(delta), 1e-6))
    return (soft_exceed.mean() - (1.0 - float(tau))).pow(2)


def multi_quantile_calibration_loss(
    target: torch.Tensor,
    quantiles: torch.Tensor,
    levels: Tuple[float, ...],
    *,
    delta: float = 0.05,
) -> torch.Tensor:
    losses = [
        calibration_loss(target, quantiles[:, i], float(tau), delta=delta)
        for i, tau in enumerate(levels)
    ]
    return torch.stack(losses).mean()


def pairwise_ranking_loss(
    target: torch.Tensor,
    score: torch.Tensor,
    *,
    temperature: float = 0.2,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Pairwise logistic ranking loss for high-risk scenario prioritization."""
    n = int(target.shape[0])
    if n < 2:
        return torch.zeros((), device=target.device, dtype=target.dtype)
    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=target.device)
    if i_idx.numel() > int(max_pairs):
        perm = torch.randperm(i_idx.numel(), device=target.device)[: int(max_pairs)]
        i_idx = i_idx[perm]
        j_idx = j_idx[perm]
    risk_diff = target[i_idx] - target[j_idx]
    valid = risk_diff.abs() > 1e-8
    if valid.sum() == 0:
        return torch.zeros((), device=target.device, dtype=target.dtype)
    direction = torch.sign(risk_diff[valid])
    score_diff = score[i_idx[valid]] - score[j_idx[valid]]
    return torch.nn.functional.softplus(
        -direction * score_diff / max(float(temperature), 1e-6)
    ).mean()


def deepevt_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    alpha: float,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Direct q85/q90/q95 training objective."""
    if "quantiles" not in outputs:
        raise ValueError("Direct quantile training requires outputs['quantiles'].")
    levels = tuple(float(x) for x in weights.get("direct_quantile_levels", (0.85, 0.90, 0.95)))
    quantiles = outputs["quantiles"]
    loss_q, q_logs = multi_quantile_loss(target, quantiles, levels)
    loss_cal = multi_quantile_calibration_loss(
        target,
        quantiles,
        levels,
        delta=float(weights.get("cal_delta", 0.05)),
    )
    primary_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - float(alpha)))
    loss_rank = pairwise_ranking_loss(
        target,
        quantiles[:, primary_idx],
        temperature=float(weights.get("rank_temperature", 0.2)),
        max_pairs=int(weights.get("rank_max_pairs", 4096)),
    )
    total = (
        float(weights.get("lambda_q", 1.0)) * loss_q
        + float(weights.get("lambda_cal", 0.0)) * loss_cal
        + float(weights.get("lambda_rank", 0.0)) * loss_rank
    )
    logs: Dict[str, float] = {
        "loss_q": float(loss_q.detach().item()),
        "loss_cal": float(loss_cal.detach().item()),
        "loss_rank": float(loss_rank.detach().item()),
        "loss_total": float(total.detach().item()),
    }
    logs.update(q_logs)
    return total, logs
