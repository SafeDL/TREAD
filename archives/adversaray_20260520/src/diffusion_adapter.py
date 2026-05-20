"""Adapter around the frozen Stage 1 GaussianActionDiffusion prior."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from diffusion.src.model import GaussianActionDiffusion, build_model_from_schema, extract_coeff
from diffusion.src.utils import load_json, select_device

from .normalization_adapter import denormalize_torch


@dataclass
class DiffusionPriorAdapter:
    model: GaussianActionDiffusion
    schema: dict[str, Any]
    config: dict[str, Any]
    stats: dict[str, Any]
    device: torch.device

    @classmethod
    def load(
        cls,
        natural_dataset_dir: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
    ) -> "DiffusionPriorAdapter":
        natural_dir = Path(natural_dataset_dir).resolve()
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = (natural_dir / ckpt).resolve()
        schema = load_json(natural_dir / "feature_schema.json")
        stats = load_json(natural_dir / "normalization_stats.json")
        resolved_device = select_device(device) if isinstance(device, str) else device
        state = torch.load(ckpt, map_location=resolved_device)
        config = state.get("config", {})
        model = build_model_from_schema(state.get("schema", schema), config).to(resolved_device)
        model.load_state_dict(state["model_state"])
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model=model, schema=state.get("schema", schema), config=config, stats=stats, device=resolved_device)

    @property
    def num_steps(self) -> int:
        return int(self.model.num_steps)

    def decode_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        return denormalize_torch(normalized_actions, self.stats, "actions")

    def decode_context_states(self, normalized_context_states: torch.Tensor) -> torch.Tensor:
        return denormalize_torch(normalized_context_states, self.stats, "context_states")

    def predict_eps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.denoiser(x_t, timesteps, context_states, context_features, relative_history)

    def predict_x0(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.predict_start_from_noise(x_t, timesteps, eps)

    def posterior_mean_variance(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            extract_coeff(self.model.posterior_mean_coef1, timesteps, x_t.shape) * x0
            + extract_coeff(self.model.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        variance = extract_coeff(self.model.posterior_variance, timesteps, x_t.shape)
        log_variance = extract_coeff(self.model.posterior_log_variance_clipped, timesteps, x_t.shape)
        return mean, variance, log_variance
