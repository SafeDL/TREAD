"""
model.py — DeepEVT 模型
========================

结构: InitialSceneTransformer + EVT heads。

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


# ---------------------------------------------------------------------------
# Scene token encoder
# ---------------------------------------------------------------------------

class InitialSceneTransformer(nn.Module):
    """Token Transformer over initial actors and scalar context features.

    With prefix_steps=1 this is not a temporal Transformer. It is a compact
    set encoder over ego/target initial states plus physically named context
    scalars, which fits the current closed-loop initial-condition contract.
    """

    def __init__(
        self,
        num_actors: int,
        state_features: int,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.actor_proj = nn.Linear(state_features, hidden_dim)
        self.context_value_proj = nn.Linear(1, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.actor_type = nn.Parameter(torch.zeros(1, num_actors, hidden_dim))
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
        x0 = prefix_states[:, 0]  # [B, A, F], current config keeps prefix_steps=1.
        actor_tokens = self.actor_proj(x0) + self.actor_type
        ctx_tokens = self.context_value_proj(context_features.unsqueeze(-1)) + self.context_type
        cls = self.cls_token.expand(prefix_states.shape[0], -1, -1)
        tokens = torch.cat([cls, actor_tokens, ctx_tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])


# ---------------------------------------------------------------------------
# DeepEVT model
# ---------------------------------------------------------------------------

class DeepEVTModel(nn.Module):
    def __init__(self, cfg: DeepEVTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = InitialSceneTransformer(
            num_actors=cfg.num_actors,
            state_features=cfg.state_features,
            context_dim=cfg.context_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_transformer_layers,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
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
    )
    return DeepEVTModel(cfg)
