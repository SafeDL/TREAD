"""Naturalness constraints for adversary action plans."""
from __future__ import annotations

import torch


def _mean_flat(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).mean(dim=1)


def action_residual_penalty(
    actions: torch.Tensor,
    prior_actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    penalty = _mean_flat((actions - prior_actions).square())
    return penalty, {"n1_action_residual": penalty}
