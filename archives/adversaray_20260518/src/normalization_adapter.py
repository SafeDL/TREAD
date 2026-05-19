"""Torch normalization helpers used by adversaray adapters."""
from __future__ import annotations

from typing import Any, Sequence

import torch


def _pair(stats: dict[str, Any], key: str) -> tuple[Sequence[float], Sequence[float]]:
    if key not in stats:
        raise KeyError(f"Missing normalization stats for {key}")
    item = stats[key]
    return item["mean"], item["std"]


def denormalize_torch(x: torch.Tensor, stats: dict[str, Any], key: str) -> torch.Tensor:
    mean, std = _pair(stats, key)
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device)
    return x * std_t + mean_t
