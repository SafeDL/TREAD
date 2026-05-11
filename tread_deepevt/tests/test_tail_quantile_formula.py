"""Unit tests for tail quantile and expected shortfall formulae."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_deepevt.src.losses import expected_shortfall_np, tail_quantile_np  # noqa: E402


def _vec(x):
    return np.array([x], dtype=np.float64)


def test_tail_quantile_increases_with_tau():
    u, p, xi, beta = _vec(0.5), _vec(0.1), _vec(0.1), _vec(0.3)
    q95 = tail_quantile_np(u, p, xi, beta, 0.95)[0]
    q99 = tail_quantile_np(u, p, xi, beta, 0.99)[0]
    assert q99 > q95 > 0.5


def test_tail_quantile_increases_with_beta():
    u, p, xi = _vec(0.5), _vec(0.1), _vec(0.1)
    q_low = tail_quantile_np(u, p, xi, _vec(0.1), 0.95)[0]
    q_high = tail_quantile_np(u, p, xi, _vec(1.0), 0.95)[0]
    assert q_high > q_low


def test_tail_quantile_continuous_near_zero_xi():
    u, p, beta = _vec(0.5), _vec(0.1), _vec(0.3)
    q_small = tail_quantile_np(u, p, _vec(1e-5), beta, 0.95)[0]
    q_zero = tail_quantile_np(u, p, _vec(0.0), beta, 0.95)[0]
    assert abs(q_small - q_zero) < 1e-2


def test_expected_shortfall_greater_than_quantile():
    u, p, xi, beta = _vec(1.0), _vec(0.1), _vec(0.2), _vec(0.5)
    q = tail_quantile_np(u, p, xi, beta, 0.95)[0]
    es = expected_shortfall_np(u, p, xi, beta, 0.95)[0]
    assert es > q


def test_expected_shortfall_nan_when_xi_ge_one():
    u, p, beta = _vec(0.5), _vec(0.1), _vec(0.3)
    es = expected_shortfall_np(u, p, _vec(1.0), beta, 0.95)[0]
    assert np.isnan(es)
