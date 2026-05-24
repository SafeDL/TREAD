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
        self.state_proj = nn.Linear(cfg.state_features, cfg.hidden_dim)
        self.actor_type = nn.Parameter(torch.zeros(1, cfg.num_actors, cfg.hidden_dim))
        self.history_pos = nn.Parameter(torch.zeros(1, cfg.history_steps, cfg.hidden_dim))
        self.history_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.relative_proj = nn.Linear(cfg.relative_dim, cfg.hidden_dim)
        self.relative_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(cfg.context_dim),
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 3),
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
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
        if context_states.shape[1:] != (
            self.cfg.history_steps,
            self.cfg.num_actors,
            self.cfg.state_features,
        ):
            raise ValueError(f"Unexpected context_states shape {tuple(context_states.shape)}")
        if relative_history.shape[1:] != (self.cfg.history_steps, self.cfg.relative_dim):
            raise ValueError(f"Unexpected relative_history shape {tuple(relative_history.shape)}")
        history = context_states.permute(0, 2, 1, 3).reshape(
            b * self.cfg.num_actors,
            self.cfg.history_steps,
            self.cfg.state_features,
        )
        history_tokens = self.state_proj(history) + self.history_pos
        _, history_hidden = self.history_gru(history_tokens)
        actor_tokens = history_hidden[-1].reshape(b, self.cfg.num_actors, self.cfg.hidden_dim) + self.actor_type
        scene_token = actor_tokens.mean(dim=1)

        relative_tokens = self.relative_proj(relative_history) + self.history_pos
        _, relative_hidden = self.relative_gru(relative_tokens)
        relative_token = relative_hidden[-1]

        feature_token = self.feature_encoder(context_features)
        cond = self.context_fuse(torch.cat([scene_token, relative_token, feature_token], dim=-1))
        cond = cond + self.timestep_mlp(sinusoidal_embedding(timesteps, self.cfg.hidden_dim))
        tokens = self.action_proj(x_t) + self.action_pos + cond.unsqueeze(1)
        tokens = self.layers(tokens)
        return self.out(self.out_norm(tokens))
