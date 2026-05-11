"""Unit tests for pinball loss behaviour."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_deepevt.src.losses import pinball_loss  # noqa: E402


def test_pinball_positive_when_under_predict_high_alpha():
    target = torch.tensor([1.0, 2.0, 3.0, 10.0])
    under = torch.full_like(target, 2.0)
    over = torch.full_like(target, 8.0)
    loss_under = pinball_loss(target, under, alpha=0.9).item()
    loss_over = pinball_loss(target, over, alpha=0.9).item()
    assert loss_under > loss_over


def test_pinball_symmetric_at_median():
    target = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    u = torch.tensor([0.5] * 4)
    loss_05 = pinball_loss(target, u, alpha=0.5).item()
    expected = torch.mean(torch.abs(target - u)).item() * 0.5
    assert math.isclose(loss_05, expected, rel_tol=1e-5)


def test_pinball_invalid_alpha_raises():
    t = torch.tensor([0.0])
    u = torch.tensor([0.0])
    with pytest.raises(ValueError):
        pinball_loss(t, u, alpha=0.0)
    with pytest.raises(ValueError):
        pinball_loss(t, u, alpha=1.0)
