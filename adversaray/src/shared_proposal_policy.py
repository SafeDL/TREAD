"""Shared low-dimensional proposal policy for Stage 1 scenario banks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ego_surrogate import IDMSurrogateParams


TEMPLATE_KEYS = (
    "brake_start",
    "brake_duration",
    "brake_intensity",
    "recovery_intensity",
    "oscillation_amplitude",
)


@dataclass(frozen=True)
class SharedProposalPolicyConfig:
    context_dim: int
    relative_dim: int
    horizon_steps: int
    action_dim: int = 1
    hidden_dim: int = 128
    latent_dim: int = 8
    output_residual_scale: float = 1.0

    @classmethod
    def from_prior(cls, prior_cfg: Any, config: dict[str, Any]) -> "SharedProposalPolicyConfig":
        cfg = config.get("stage1_shared", {}).get("policy", {})
        return cls(
            context_dim=int(prior_cfg.context_dim),
            relative_dim=int(prior_cfg.relative_dim),
            horizon_steps=int(prior_cfg.horizon_steps),
            action_dim=int(prior_cfg.action_dim),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            latent_dim=int(cfg.get("latent_dim", 8)),
            output_residual_scale=float(cfg.get("output_residual_scale", 1.0)),
        )


def prior_action_summary(prior_actions: torch.Tensor) -> torch.Tensor:
    if prior_actions.ndim != 3 or prior_actions.shape[-1] < 1:
        raise ValueError(f"Expected prior_actions shape [B,H,1+], got {tuple(prior_actions.shape)}")
    jerk = prior_actions[:, :, 0]
    first = jerk[:, : max(1, jerk.shape[1] // 4)]
    middle = jerk[:, jerk.shape[1] // 4 : max(jerk.shape[1] // 4 + 1, 3 * jerk.shape[1] // 4)]
    last = jerk[:, 3 * jerk.shape[1] // 4 :]
    return torch.stack(
        [
            jerk.mean(dim=1),
            jerk.std(dim=1, unbiased=False),
            jerk.amin(dim=1),
            jerk.amax(dim=1),
            first.mean(dim=1),
            middle.mean(dim=1),
            last.mean(dim=1),
            jerk[:, 0],
        ],
        dim=-1,
    )


class SharedProposalPolicy(nn.Module):
    def __init__(self, cfg: SharedProposalPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(cfg.context_dim),
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.relative_proj = nn.Linear(cfg.relative_dim, cfg.hidden_dim)
        self.relative_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.prior_encoder = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.surrogate_encoder = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.latent_encoder = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 5),
            nn.Linear(cfg.hidden_dim * 5, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, len(TEMPLATE_KEYS)),
        )

    def forward(
        self,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        prior_action_summary: torch.Tensor,
        ego_surrogate_params: IDMSurrogateParams,
        latent_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context_features.ndim != 2 or context_features.shape[-1] != self.cfg.context_dim:
            raise ValueError(f"Unexpected context_features shape {tuple(context_features.shape)}")
        if relative_history.ndim != 3 or relative_history.shape[-1] != self.cfg.relative_dim:
            raise ValueError(f"Unexpected relative_history shape {tuple(relative_history.shape)}")
        if prior_action_summary.shape[-1] != 8:
            raise ValueError(f"Expected prior_action_summary last dim 8, got {tuple(prior_action_summary.shape)}")
        if latent_z.shape[-1] != self.cfg.latent_dim:
            raise ValueError(f"Expected latent_z last dim {self.cfg.latent_dim}, got {tuple(latent_z.shape)}")

        rel_tokens = self.relative_proj(relative_history)
        _, rel_hidden = self.relative_gru(rel_tokens)
        rel_token = rel_hidden[-1]
        surrogate_features = ego_surrogate_params.to_feature_tensor()
        token = torch.cat(
            [
                self.context_encoder(context_features),
                rel_token,
                self.prior_encoder(prior_action_summary),
                self.surrogate_encoder(surrogate_features),
                self.latent_encoder(latent_z),
            ],
            dim=-1,
        )
        raw = self.head(token)
        scale = float(self.cfg.output_residual_scale)
        return {
            "brake_start": torch.sigmoid(raw[:, 0]) * 0.85,
            "brake_duration": 0.06 + torch.sigmoid(raw[:, 1]) * 0.54,
            "brake_intensity": F.softplus(raw[:, 2]) * scale,
            "recovery_intensity": torch.tanh(raw[:, 3]) * scale,
            "oscillation_amplitude": torch.tanh(raw[:, 4]) * scale,
        }


def template_to_jerk_delta(params: dict[str, torch.Tensor], *, horizon: int, action_dim: int = 1) -> torch.Tensor:
    missing = [key for key in TEMPLATE_KEYS if key not in params]
    if missing:
        raise KeyError(f"Template params missing keys: {missing}")
    sample = params["brake_start"]
    device = sample.device
    dtype = sample.dtype
    t = torch.arange(int(horizon), device=device, dtype=dtype)[None, :]
    horizon_t = torch.as_tensor(max(int(horizon) - 1, 1), device=device, dtype=dtype)
    start = params["brake_start"][:, None] * horizon_t
    duration = torch.clamp(params["brake_duration"][:, None] * horizon_t, min=1.0)
    end = start + duration
    edge = torch.clamp(0.08 * duration, min=1.0)
    brake_window = torch.sigmoid((t - start) / edge) * torch.sigmoid((end - t) / edge)
    recovery_start = end
    recovery_end = recovery_start + torch.clamp(0.5 * duration, min=1.0)
    recovery_window = torch.sigmoid((t - recovery_start) / edge) * torch.sigmoid((recovery_end - t) / edge)
    phase = (t - start) / torch.clamp(duration, min=1.0)
    oscillation = torch.sin(2.0 * torch.pi * phase)
    delta = (
        -params["brake_intensity"][:, None] * brake_window
        + params["recovery_intensity"][:, None] * recovery_window
        + params["oscillation_amplitude"][:, None] * oscillation * brake_window
    )
    if int(action_dim) <= 1:
        return delta[:, :, None]
    pad = torch.zeros((delta.shape[0], delta.shape[1], int(action_dim) - 1), dtype=dtype, device=device)
    return torch.cat([delta[:, :, None], pad], dim=-1)


def template_params_to_tensor(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([params[key] for key in TEMPLATE_KEYS], dim=-1)
