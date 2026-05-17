"""
baselines.py — DeepEVT 基线模型
================================

1. GlobalPOTGPD    — 固定阈值 GPD (per event_type)
2. QuantileOnlyNet — 仅预测条件分位 u_theta(z)，用于验证 DeepEVT 必要性
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import DatasetArrays
from .losses import pinball_loss, tail_quantile_np

logger = logging.getLogger(__name__)

_EPS = 1e-6
_XI_SMALL = 1e-4


# ---------------------------------------------------------------------------
# POT-GPD MLE (numpy implementation)
# ---------------------------------------------------------------------------

def _gpd_nll_np(params: np.ndarray, y: np.ndarray) -> float:
    xi, log_beta = params
    beta = np.exp(log_beta)
    if beta <= 0:
        return 1e18
    if abs(xi) < _XI_SMALL:
        return float(np.sum(np.log(beta) + y / beta))
    support = 1.0 + xi * y / beta
    if np.any(support <= 0):
        return 1e18
    return float(np.sum(np.log(beta) + (1.0 + 1.0 / xi) * np.log(support)))


def fit_gpd_mle(y: np.ndarray) -> Tuple[float, float]:
    """简单网格搜索 + 局部优化拟合 GPD 的 (xi, beta)。"""
    if len(y) < 2:
        return 0.0, max(float(np.mean(y)) if len(y) else 1.0, _EPS)
    from scipy.optimize import minimize
    y = np.asarray(y, dtype=np.float64)
    y = y[y > 0]
    if len(y) == 0:
        return 0.0, _EPS
    x0 = np.array([0.1, np.log(max(np.mean(y), _EPS))])
    try:
        res = minimize(
            _gpd_nll_np, x0, args=(y,), method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
        )
        xi, log_beta = res.x
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPD MLE failed, fallback to MoM: %s", exc)
        xi = 0.0
        log_beta = np.log(max(np.mean(y), _EPS))
    xi = float(np.clip(xi, -0.4, 0.9))
    beta = float(max(np.exp(log_beta), _EPS))
    return xi, beta


@dataclass
class GlobalPOTGPDParams:
    alpha_u: float
    u: float
    xi: float
    beta: float
    p: float

    def tail_quantile(self, tau: float) -> float:
        u = np.array([self.u])
        p = np.array([self.p])
        xi = np.array([self.xi])
        beta = np.array([self.beta])
        return float(tail_quantile_np(u, p, xi, beta, tau)[0])


def fit_global_pot_gpd(train_risk: np.ndarray, alpha_u: float) -> GlobalPOTGPDParams:
    u = float(np.quantile(train_risk, alpha_u))
    y = train_risk[train_risk > u] - u
    xi, beta = fit_gpd_mle(y)
    p = float(len(y) / max(len(train_risk), 1))
    return GlobalPOTGPDParams(alpha_u=alpha_u, u=u, xi=xi, beta=beta, p=p)





# ---------------------------------------------------------------------------
# Quantile-only neural baseline
# ---------------------------------------------------------------------------

class QuantileOnlyNet(nn.Module):
    def __init__(self, prefix_steps: int, num_actors: int, state_features: int,
                 context_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.prefix_steps = prefix_steps
        self.num_actors = num_actors
        self.state_features = state_features
        in_dim = prefix_steps * num_actors * state_features + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prefix: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B = prefix.shape[0]
        x = torch.cat([prefix.reshape(B, -1), ctx], dim=-1)
        return self.net(x).squeeze(-1)


def train_quantile_only(
    train_arrays: DatasetArrays, val_arrays: DatasetArrays, config: dict,
) -> QuantileOnlyNet:
    from torch.utils.data import DataLoader, TensorDataset

    training_cfg = config.get("training", {})
    alpha = float(training_cfg.get("alpha_u", 0.9))
    batch_size = int(training_cfg.get("batch_size", 256))
    lr = float(training_cfg.get("lr", 1e-3))
    epochs = int(training_cfg.get("pretrain_quantile_epochs", 50))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix_steps = train_arrays.prefix_states.shape[1]
    num_actors = train_arrays.prefix_states.shape[2]
    state_features = train_arrays.prefix_states.shape[3]
    context_dim = train_arrays.context_features.shape[1]
    model = QuantileOnlyNet(
        prefix_steps, num_actors, state_features, context_dim,
        hidden_dim=int(config.get("model", {}).get("hidden_dim", 128)),
    ).to(device)

    ds = TensorDataset(
        torch.from_numpy(train_arrays.prefix_states).float(),
        torch.from_numpy(train_arrays.context_features).float(),
        torch.from_numpy(train_arrays.risk_score).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for p_b, c_b, r_b in loader:
            p_b = p_b.to(device); c_b = c_b.to(device); r_b = r_b.to(device)
            pred = model(p_b, c_b)
            loss = pinball_loss(r_b, pred, alpha)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            logger.info("QuantileOnly ep%03d  mean_loss=%.4f",
                        ep, float(np.mean(losses)))
    _ = val_arrays
    return model


def predict_quantile_only(model: QuantileOnlyNet, arrays: DatasetArrays) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        p = torch.from_numpy(arrays.prefix_states).float().to(device)
        c = torch.from_numpy(arrays.context_features).float().to(device)
        return model(p, c).cpu().numpy()
