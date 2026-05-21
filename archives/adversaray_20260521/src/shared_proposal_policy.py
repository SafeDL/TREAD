"""Direct residual sequence proposal policy for Stage 1 scenario banks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .ego_surrogate import IDMSurrogateParams


@dataclass(frozen=True)
class SharedProposalPolicyConfig:
    context_dim: int
    relative_dim: int
    horizon_steps: int
    action_dim: int = 1
    hidden_dim: int = 128
    latent_dim: int = 8
    output_residual_scale: float = 6.0
    max_delta_jerk: float = 8.0
    prior_action_hidden_dim: int = 128
    decoder_layers: int = 1
    zero_init_output: bool = True

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
            output_residual_scale=float(cfg.get("output_residual_scale", 6.0)),
            max_delta_jerk=float(cfg.get("max_delta_jerk", 8.0)),
            prior_action_hidden_dim=int(cfg.get("prior_action_hidden_dim", 128)),
            decoder_layers=int(cfg.get("decoder_layers", 1)),
            zero_init_output=bool(cfg.get("zero_init_output", True)),
        )


class DirectResidualSequencePolicy(nn.Module):
    """Direct residual sequence policy for Stage 1."""

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
        action_norm: nn.Module = nn.LayerNorm(cfg.action_dim) if cfg.action_dim > 1 else nn.Identity()
        self.prior_action_proj = nn.Sequential(
            action_norm,
            nn.Linear(cfg.action_dim, cfg.prior_action_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.prior_action_hidden_dim, cfg.hidden_dim),
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
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 4),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.time_embedding = nn.Parameter(torch.randn(cfg.horizon_steps, cfg.hidden_dim) * 0.02)
        self.decoder = nn.GRU(
            input_size=cfg.hidden_dim * 2,
            hidden_size=cfg.hidden_dim,
            num_layers=max(int(cfg.decoder_layers), 1),
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        if cfg.zero_init_output:
            last = self.output_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        prior_actions: torch.Tensor,
        ego_surrogate_params: IDMSurrogateParams,
        latent_z: torch.Tensor,
    ) -> torch.Tensor:
        if context_features.ndim != 2 or context_features.shape[-1] != self.cfg.context_dim:
            raise ValueError(f"Unexpected context_features shape {tuple(context_features.shape)}")
        if relative_history.ndim != 3 or relative_history.shape[-1] != self.cfg.relative_dim:
            raise ValueError(f"Unexpected relative_history shape {tuple(relative_history.shape)}")
        if prior_actions.ndim != 3:
            raise ValueError(f"Expected prior_actions [B,H,A], got {tuple(prior_actions.shape)}")
        if latent_z.shape[-1] != self.cfg.latent_dim:
            raise ValueError(f"Expected latent_z last dim {self.cfg.latent_dim}, got {tuple(latent_z.shape)}")

        batch, horizon, action_dim = prior_actions.shape
        if action_dim != self.cfg.action_dim:
            raise ValueError(f"Expected action_dim={self.cfg.action_dim}, got {action_dim}")
        if horizon > self.time_embedding.shape[0]:
            raise ValueError(f"Policy horizon {self.time_embedding.shape[0]} is shorter than action horizon {horizon}")

        rel_tokens = self.relative_proj(relative_history)
        _, rel_hidden = self.relative_gru(rel_tokens)
        rel_token = rel_hidden[-1]
        surrogate_features = ego_surrogate_params.to_feature_tensor()
        global_token = torch.cat(
            [
                self.context_encoder(context_features),
                rel_token,
                self.surrogate_encoder(surrogate_features),
                self.latent_encoder(latent_z),
            ],
            dim=-1,
        )
        global_token = self.global_fusion(global_token)

        prior_tokens = self.prior_action_proj(prior_actions)
        time_tokens = self.time_embedding[:horizon][None, :, :].expand(batch, horizon, -1)
        seq_token = prior_tokens + time_tokens
        global_seq = global_token[:, None, :].expand(batch, horizon, -1)
        decoder_input = torch.cat([seq_token, global_seq], dim=-1)
        hidden, _ = self.decoder(decoder_input)
        raw_delta = self.output_head(hidden)
        scale = float(self.cfg.output_residual_scale)
        bound = float(self.cfg.max_delta_jerk)
        return torch.tanh(raw_delta * scale / max(bound, 1e-6)) * bound


SharedProposalPolicy = DirectResidualSequencePolicy
