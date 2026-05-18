"""Learnable residual guidance network for frozen action diffusion priors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from diffusion.src.model import ActionDiffusionConfig, sinusoidal_embedding


@dataclass(frozen=True)
class GuidancePolicyConfig:
    history_steps: int
    num_actors: int
    state_features: int
    context_dim: int
    relative_dim: int
    horizon_steps: int
    action_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.05
    zero_init_output: bool = True

    @classmethod
    def from_prior(cls, prior_cfg: ActionDiffusionConfig, config: dict[str, Any]) -> "GuidancePolicyConfig":
        cfg = config.get("policy", config)
        return cls(
            history_steps=int(prior_cfg.history_steps),
            num_actors=int(prior_cfg.num_actors),
            state_features=int(prior_cfg.state_features),
            context_dim=int(prior_cfg.context_dim),
            relative_dim=int(prior_cfg.relative_dim),
            horizon_steps=int(prior_cfg.horizon_steps),
            action_dim=int(prior_cfg.action_dim),
            hidden_dim=int(cfg.get("hidden_dim", prior_cfg.hidden_dim)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_heads=int(cfg.get("num_heads", prior_cfg.num_heads)),
            dropout=float(cfg.get("dropout", 0.05)),
            zero_init_output=bool(cfg.get("zero_init_output", True)),
        )


class GuidancePolicy(nn.Module):
    """Predict a small score-like residual ``g_phi(x_t, t, c)``.

    The network intentionally mirrors the prior's conditioning inputs while
    staying much smaller than the diffusion denoiser. A zero-initialized output
    head makes the initial guided sampler identical to the frozen prior.
    """

    def __init__(self, cfg: GuidancePolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        flat_context_dim = (
            cfg.history_steps * cfg.num_actors * cfg.state_features
            + cfg.context_dim
            + cfg.history_steps * cfg.relative_dim
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(flat_context_dim),
            nn.Linear(flat_context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, cfg.horizon_steps, cfg.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=max(1, cfg.num_heads),
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=max(1, cfg.num_layers))
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        if cfg.zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
    ) -> torch.Tensor:
        b, horizon, action_dim = x_t.shape
        if horizon != self.cfg.horizon_steps or action_dim != self.cfg.action_dim:
            raise ValueError(f"Unexpected x_t shape {tuple(x_t.shape)}")
        context_flat = torch.cat(
            [
                context_states.reshape(b, -1),
                context_features.reshape(b, -1),
                relative_history.reshape(b, -1),
            ],
            dim=-1,
        )
        cond = self.context_encoder(context_flat)
        cond = cond + self.timestep_mlp(sinusoidal_embedding(timesteps, self.cfg.hidden_dim))
        tokens = self.action_proj(x_t) + self.action_pos + cond.unsqueeze(1)
        tokens = self.layers(tokens)
        return self.out(self.out_norm(tokens))
