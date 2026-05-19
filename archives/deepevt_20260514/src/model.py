"""
model.py — DeepEVT 模型
========================

结构: ShortHistorySceneTransformer + ego-target interaction token + EVT heads。

输出 heads:
    u    — 条件阈值 (unconstrained scalar)
    p    — 超阈值概率 (sigmoid)
    xi   — GPD shape，通过 sigmoid 限制在 [xi_min, xi_max]
    beta — GPD scale，softplus + beta_min 保证为正
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepEVTConfig:
    prefix_steps: int
    num_actors: int
    state_features: int
    context_dim: int
    hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.1
    xi_min: float = -0.3
    xi_max: float = 0.5
    beta_min: float = 1e-4
    use_exceedance_head: bool = True
    use_interaction_token: bool = True


# ---------------------------------------------------------------------------
# Scene token encoder
# ---------------------------------------------------------------------------

class ShortHistorySceneTransformer(nn.Module):
    """Temporal actor encoder followed by scene-level token self-attention.

    Each actor's ``prefix_steps`` states are first encoded into one temporal
    token. When enabled, an explicit ego-target interaction sequence is encoded
    into an additional token. The scene Transformer then models interactions
    among CLS, actor temporal tokens, the interaction token, and scalar context
    tokens. ``prefix_steps=1`` naturally degenerates to a single-frame
    current-scene encoder with the same token schema.
    """

    def __init__(
        self,
        prefix_steps: int,
        num_actors: int,
        state_features: int,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_interaction_token: bool = True,
    ) -> None:
        super().__init__()
        self.prefix_steps = int(prefix_steps)
        self.num_actors = int(num_actors)
        self.state_features = int(state_features)
        self.context_dim = int(context_dim)
        self.use_interaction_token = bool(use_interaction_token)
        self.state_proj = nn.Linear(state_features, hidden_dim)
        self.time_pos = nn.Parameter(torch.zeros(1, self.prefix_steps, hidden_dim))
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.interaction_proj = nn.Linear(4, hidden_dim)
        self.interaction_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.context_value_proj = nn.Linear(1, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.actor_type = nn.Parameter(torch.zeros(1, num_actors, hidden_dim))
        self.interaction_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.context_type = nn.Parameter(torch.zeros(1, context_dim, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_dim = hidden_dim
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, num_layers))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, prefix_states: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        if prefix_states.ndim != 4:
            raise ValueError(
                "prefix_states must be [batch, prefix_steps, actors, state_features], "
                f"got shape={tuple(prefix_states.shape)}"
            )
        if context_features.ndim != 2:
            raise ValueError(
                "context_features must be [batch, context_features], "
                f"got shape={tuple(context_features.shape)}"
            )
        if prefix_states.shape[1] != self.prefix_steps:
            raise ValueError(
                f"Expected prefix_steps={self.prefix_steps}, got {prefix_states.shape[1]}"
            )
        if prefix_states.shape[2] != self.num_actors:
            raise ValueError(
                f"Expected num_actors={self.num_actors}, got {prefix_states.shape[2]}"
            )
        if prefix_states.shape[3] != self.state_features:
            raise ValueError(
                f"Expected state_features={self.state_features}, got {prefix_states.shape[3]}"
            )
        if context_features.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context_dim={self.context_dim}, got {context_features.shape[1]}"
            )
        batch_size, prefix_steps, num_actors, state_features = prefix_states.shape
        actor_sequences = prefix_states.permute(0, 2, 1, 3).reshape(
            batch_size * num_actors, prefix_steps, state_features,
        )
        temporal_in = self.state_proj(actor_sequences) + self.time_pos
        _, h_n = self.temporal_encoder(temporal_in)
        actor_tokens = h_n[-1].reshape(batch_size, num_actors, -1) + self.actor_type
        cls = self.cls_token.expand(batch_size, -1, -1)
        token_parts = [cls, actor_tokens]
        if self.use_interaction_token and num_actors >= 2 and state_features >= 5:
            interaction_tokens = self._build_interaction_token(prefix_states)
            token_parts.append(interaction_tokens)
        ctx_tokens = self.context_value_proj(context_features.unsqueeze(-1)) + self.context_type
        token_parts.append(ctx_tokens)
        tokens = torch.cat(token_parts, dim=1)
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])

    def _build_interaction_token(self, prefix_states: torch.Tensor) -> torch.Tensor:
        ego = prefix_states[:, :, 0, :]
        target = prefix_states[:, :, 1, :]
        # The prefix tensor does not carry vehicle lengths, so longitudinal
        # center distance is used as the temporal gap proxy.
        longitudinal_delta = target[..., 0] - ego[..., 0]
        lateral_offset = target[..., 1] - ego[..., 1]
        relative_speed = ego[..., 2] - target[..., 2]
        relative_acceleration = ego[..., 4] - target[..., 4]
        interaction_seq = torch.stack(
            [longitudinal_delta, relative_speed, relative_acceleration, lateral_offset],
            dim=-1,
        )
        interaction_in = self.interaction_proj(interaction_seq) + self.time_pos
        _, h_n = self.interaction_encoder(interaction_in)
        return h_n[-1].unsqueeze(1) + self.interaction_type

# ---------------------------------------------------------------------------
# DeepEVT model
# ---------------------------------------------------------------------------

class DeepEVTModel(nn.Module):
    def __init__(self, cfg: DeepEVTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ShortHistorySceneTransformer(
            prefix_steps=cfg.prefix_steps,
            num_actors=cfg.num_actors,
            state_features=cfg.state_features,
            context_dim=cfg.context_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_transformer_layers,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            use_interaction_token=cfg.use_interaction_token,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder.output_dim, cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        # heads
        self.u_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.p_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.xi_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.beta_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.u_log_scale_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.xi_log_scale_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.beta_log_scale_head = nn.Linear(cfg.fusion_hidden_dim, 1)

    # ---- helpers ----------------------------------------------------------
    def encoder_parameters(self):
        return list(self.encoder.parameters()) + list(self.fusion.parameters())

    def threshold_head_parameters(self):
        return list(self.u_head.parameters())

    def tail_head_parameters(self):
        params = list(self.xi_head.parameters()) + list(self.beta_head.parameters())
        params += list(self.u_log_scale_head.parameters())
        params += list(self.xi_log_scale_head.parameters())
        params += list(self.beta_log_scale_head.parameters())
        if self.cfg.use_exceedance_head:
            params += list(self.p_head.parameters())
        return params

    # ---- forward ----------------------------------------------------------
    def forward(
        self, prefix_states: torch.Tensor, context_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_scene = self.encoder(prefix_states, context_features)
        z = self.fusion(z_scene)

        u = self.u_head(z).squeeze(-1)
        xi_raw = self.xi_head(z).squeeze(-1)
        beta_raw = self.beta_head(z).squeeze(-1)
        xi = self.cfg.xi_min + (self.cfg.xi_max - self.cfg.xi_min) * torch.sigmoid(xi_raw)
        beta = F.softplus(beta_raw) + self.cfg.beta_min
        outputs = {
            "u": u,
            "xi": xi,
            "beta": beta,
            "u_log_scale": self.u_log_scale_head(z).squeeze(-1).clamp(-5.0, 5.0),
            "xi_log_scale": self.xi_log_scale_head(z).squeeze(-1).clamp(-5.0, 5.0),
            "beta_log_scale": self.beta_log_scale_head(z).squeeze(-1).clamp(-5.0, 5.0),
        }

        if self.cfg.use_exceedance_head:
            p_raw = self.p_head(z).squeeze(-1)
            outputs["p"] = torch.sigmoid(p_raw)
        return outputs


def build_model_from_schema(schema: dict, config: dict) -> DeepEVTModel:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    cfg = DeepEVTConfig(
        prefix_steps=int(schema["prefix_steps"]),
        num_actors=int(schema["num_actors"]),
        state_features=len(schema["prefix_state_features"]),
        context_dim=int(schema["context_dim"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 128)),
        num_transformer_layers=int(model_cfg.get("num_transformer_layers", 2)),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        xi_min=float(model_cfg.get("xi_min", -0.3)),
        xi_max=float(model_cfg.get("xi_max", 0.5)),
        beta_min=float(model_cfg.get("beta_min", 1e-4)),
        use_exceedance_head=bool(training_cfg.get("use_exceedance_head", True)),
        use_interaction_token=bool(model_cfg.get("use_interaction_token", True)),
    )
    return DeepEVTModel(cfg)
