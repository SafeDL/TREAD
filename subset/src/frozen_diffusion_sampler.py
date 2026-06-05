"""Frozen DDIM prior sampler for latent-space subset simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.diffusion_adapter import DiffusionPriorAdapter


@dataclass
class FrozenDiffusionSampleResult:
    normalized_actions: torch.Tensor
    raw_actions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class FrozenDiffusionSampler:
    """Decode externally supplied diffusion latents with deterministic DDIM."""

    def __init__(
        self,
        prior: DiffusionPriorAdapter,
        config: dict[str, Any],
    ) -> None:
        self.prior = prior
        self.config = config
        self.training = False
        for param in self.prior.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        config_dir: str | Path | None = None,
    ) -> "FrozenDiffusionSampler":
        base = Path(config_dir).resolve() if config_dir else Path.cwd()
        paths = config.get("paths", {})
        required_paths = ("natural_dataset_dir", "diffusion_checkpoint")
        missing = [key for key in required_paths if key not in paths]
        if missing:
            raise KeyError(f"Config paths is missing required keys: {missing}")

        natural_dir = (base / paths["natural_dataset_dir"]).resolve()
        diffusion_ckpt = Path(paths["diffusion_checkpoint"])
        if not diffusion_ckpt.is_absolute():
            diffusion_ckpt = (base / diffusion_ckpt).resolve()
        if not natural_dir.exists():
            raise FileNotFoundError(
                f"Natural diffusion dataset directory not found: {natural_dir}"
            )
        if not diffusion_ckpt.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {diffusion_ckpt}"
            )

        device = config.get("training", {}).get(
            "device",
            config.get("device", "auto"),
        )
        prior = DiffusionPriorAdapter.load(
            natural_dir,
            diffusion_ckpt,
            device=device,
        )
        clip = float(
            config.get("diffusion", {}).get(
                "x0_clip_abs",
                prior.config.get("diffusion", {}).get("x0_clip_abs", 0.0),
            )
        )
        if clip > 0.0:
            prior.model.denoiser.cfg.x0_clip_abs = clip
        return cls(prior, config)

    def train(self, mode: bool = True) -> "FrozenDiffusionSampler":
        self.training = bool(mode)
        self.prior.model.eval()
        return self

    def eval(self) -> "FrozenDiffusionSampler":
        return self.train(False)

    def _ddim_timesteps(self, inference_steps: int | None) -> list[int]:
        if inference_steps is not None and int(inference_steps) > 0:
            steps = min(int(inference_steps), self.prior.num_steps)
        else:
            sampling_cfg = self.config.get("sampling", {})
            steps = int(
                sampling_cfg.get(
                    "eval_diffusion_steps",
                    sampling_cfg.get("diffusion_steps", self.prior.num_steps),
                )
            )
            steps = min(max(steps, 1), self.prior.num_steps)
        raw_steps = np.linspace(0, self.prior.num_steps - 1, steps)
        timesteps = sorted({int(round(step)) for step in raw_steps})
        return list(reversed(timesteps))

    def sample_from_noise(
        self,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        init_noise: torch.Tensor,
        *,
        inference_steps: int | None = None,
    ) -> FrozenDiffusionSampleResult:
        device = self.prior.device
        cfg = self.prior.model.denoiser.cfg
        expected_shape = (int(cfg.horizon_steps), int(cfg.action_dim))
        init_noise = init_noise.to(device).float()
        if init_noise.ndim != 3:
            raise ValueError(
                "init_noise must have shape [batch, horizon, action_dim], "
                f"got {tuple(init_noise.shape)}"
            )
        if tuple(init_noise.shape[1:]) != expected_shape:
            raise ValueError(
                "init_noise trailing shape must be "
                f"{expected_shape}, got {tuple(init_noise.shape[1:])}"
            )
        batch_size = int(init_noise.shape[0])

        context_states = self._align_batch(
            context_states,
            batch_size,
            "context_states",
        )
        context_features = self._align_batch(
            context_features,
            batch_size,
            "context_features",
        )
        relative_history = self._align_batch(
            relative_history,
            batch_size,
            "relative_history",
        )

        x_t = init_noise
        timesteps = self._ddim_timesteps(inference_steps)
        with torch.no_grad():
            for loop_idx, step in enumerate(timesteps):
                t = torch.full(
                    (batch_size,),
                    int(step),
                    dtype=torch.long,
                    device=device,
                )
                eps = self.prior.predict_eps(
                    x_t,
                    t,
                    context_states,
                    context_features,
                    relative_history,
                )
                prev_step = (
                    timesteps[loop_idx + 1]
                    if loop_idx + 1 < len(timesteps)
                    else -1
                )
                prev_t = torch.full_like(t, int(prev_step))
                x_t = self.prior.ddim_step(x_t, t, prev_t, eps).detach()

        raw_actions = self.prior.decode_actions(x_t)
        diagnostics = {
            "guidance_steps": torch.zeros(
                batch_size,
                dtype=x_t.dtype,
                device=x_t.device,
            )
        }
        return FrozenDiffusionSampleResult(
            normalized_actions=x_t.detach(),
            raw_actions=raw_actions.detach(),
            diagnostics=diagnostics,
        )

    def _align_batch(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        tensor = tensor.to(self.prior.device).float()
        if int(tensor.shape[0]) == batch_size:
            return tensor
        if int(tensor.shape[0]) == 1:
            return tensor.repeat_interleave(batch_size, dim=0)
        raise ValueError(
            f"{name} batch must be 1 or {batch_size}, got {tensor.shape[0]}"
        )
