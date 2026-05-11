"""Unit tests for GPD NLL & support handling."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_deepevt.src.losses import gpd_nll  # noqa: E402


def test_gpd_nll_is_finite_for_positive_support():
    y = torch.tensor([0.1, 0.5, 1.0])
    xi = torch.tensor([0.1, 0.1, 0.1])
    beta = torch.tensor([0.5, 0.5, 0.5])
    nll, penalty = gpd_nll(y, xi, beta)
    assert torch.isfinite(nll)
    assert penalty.item() == 0.0


def test_gpd_nll_small_xi_uses_exponential_limit():
    y = torch.tensor([0.2, 0.5, 1.0, 2.0])
    xi_zero = torch.tensor([0.0, 0.0, 0.0, 0.0])
    beta = torch.tensor([1.0, 1.0, 1.0, 1.0])
    nll, _ = gpd_nll(y, xi_zero, beta)
    # E[-log f_exp(y; beta=1)] = 1 + mean(y); we check closeness within tolerance
    expected = (torch.log(beta) + y / beta).mean()
    assert torch.isfinite(nll)
    assert torch.allclose(nll, expected, atol=1e-3)


def test_gpd_nll_penalises_negative_support():
    # xi*y/beta = -2 => support = -1, should be penalised
    y = torch.tensor([1.0])
    xi = torch.tensor([-1.0])
    beta = torch.tensor([0.5])
    _, penalty = gpd_nll(y, xi, beta)
    assert penalty.item() > 0


def test_gpd_nll_zero_exceedance_returns_zero():
    y = torch.tensor([-0.1, 0.0, -2.0])
    xi = torch.tensor([0.1, 0.1, 0.1])
    beta = torch.tensor([0.5, 0.5, 0.5])
    nll, penalty = gpd_nll(y, xi, beta)
    assert nll.item() == 0.0
    assert penalty.item() == 0.0
