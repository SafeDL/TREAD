"""
model.py — DeepEVT 模型
========================

结构: PrefixEncoder (GRU/TCN/MLP) + ContextMLP + Fusion MLP + EVT heads。

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
    encoder_type: str = "gru"          # gru / tcn / mlp
    hidden_dim: int = 128
    context_hidden_dim: int = 64
    fusion_hidden_dim: int = 128
    dropout: float = 0.1
    xi_min: float = -0.3
    xi_max: float = 0.5
    beta_min: float = 1e-4
    use_exceedance_head: bool = True


# ---------------------------------------------------------------------------
# Prefix encoders
# ---------------------------------------------------------------------------

class GRUPrefixEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, input_dim]
        out, h = self.gru(x)
        return self.dropout(h[-1])


class TCNPrefixEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, input_dim] -> [B, input_dim, K]
        h = self.net(x.transpose(1, 2))
        return h.mean(dim=-1)  # global average pool


class MLPPrefixEncoder(nn.Module):
    def __init__(self, input_dim: int, prefix_steps: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * prefix_steps, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_prefix_encoder(cfg: DeepEVTConfig) -> nn.Module:
    input_dim = cfg.num_actors * cfg.state_features
    enc_type = cfg.encoder_type.lower()
    if enc_type == "gru":
        return GRUPrefixEncoder(input_dim, cfg.hidden_dim, cfg.dropout)
    if enc_type == "tcn":
        return TCNPrefixEncoder(input_dim, cfg.hidden_dim, cfg.dropout)
    if enc_type == "mlp":
        return MLPPrefixEncoder(input_dim, cfg.prefix_steps, cfg.hidden_dim, cfg.dropout)
    raise ValueError(f"Unsupported encoder_type: {cfg.encoder_type}")


# ---------------------------------------------------------------------------
# DeepEVT model
# ---------------------------------------------------------------------------

class DeepEVTModel(nn.Module):
    def __init__(self, cfg: DeepEVTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.prefix_encoder = _build_prefix_encoder(cfg)
        self.context_mlp = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.context_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden_dim, cfg.context_hidden_dim),
            nn.ReLU(inplace=True),
        )
        fusion_in = self.prefix_encoder.output_dim + cfg.context_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden_dim),
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

    # ---- helpers ----------------------------------------------------------
    def encoder_parameters(self):
        return list(self.prefix_encoder.parameters()) \
            + list(self.context_mlp.parameters()) \
            + list(self.fusion.parameters())

    def threshold_head_parameters(self):
        return list(self.u_head.parameters())

    def tail_head_parameters(self):
        params = list(self.xi_head.parameters()) + list(self.beta_head.parameters())
        if self.cfg.use_exceedance_head:
            params += list(self.p_head.parameters())
        return params

    # ---- forward ----------------------------------------------------------
    def forward(
        self, prefix_states: torch.Tensor, context_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K, A, F_ = prefix_states.shape
        x = prefix_states.reshape(B, K, A * F_)
        z_prefix = self.prefix_encoder(x)
        z_ctx = self.context_mlp(context_features)
        z = self.fusion(torch.cat([z_prefix, z_ctx], dim=-1))

        u = self.u_head(z).squeeze(-1)
        xi_raw = self.xi_head(z).squeeze(-1)
        beta_raw = self.beta_head(z).squeeze(-1)
        xi = self.cfg.xi_min + (self.cfg.xi_max - self.cfg.xi_min) * torch.sigmoid(xi_raw)
        beta = F.softplus(beta_raw) + self.cfg.beta_min
        outputs = {"u": u, "xi": xi, "beta": beta}

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
        encoder_type=str(model_cfg.get("encoder_type", "gru")),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        context_hidden_dim=int(model_cfg.get("context_hidden_dim", 64)),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        xi_min=float(model_cfg.get("xi_min", -0.3)),
        xi_max=float(model_cfg.get("xi_max", 0.5)),
        beta_min=float(model_cfg.get("beta_min", 1e-4)),
        use_exceedance_head=bool(training_cfg.get("use_exceedance_head", True)),
    )
    return DeepEVTModel(cfg)
