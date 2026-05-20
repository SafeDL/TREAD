"""Frozen natural diffusion sampler used by Stage 1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .diffusion_adapter import DiffusionPriorAdapter
from .guidance_losses import physical_violation_penalty
from .rss import RSSConfig, rss_criticality_objective
from .torch_kinematics import integrate_following_actions_torch


@dataclass
class PriorGuidedSampleResult:
    normalized_actions: torch.Tensor
    raw_actions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    guidance_trace: list[dict[str, float]]
    trajectory_log_prob: torch.Tensor
    prior_kl: torch.Tensor
    guidance_norm: torch.Tensor


def _repeat_context(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    if num_samples <= 1:
        return tensor
    return tensor.repeat_interleave(int(num_samples), dim=0)


class PriorGuidedDiffusionSampler:
    """DDPM sampler for the frozen natural-action prior."""

    def __init__(self, prior: DiffusionPriorAdapter, config: dict[str, Any]) -> None:
        self.prior = prior
        self.config = config
        self.rss_cfg = RSSConfig.from_config(config)
        self.training = False

    @classmethod
    def from_config(cls, config: dict[str, Any], *, config_dir: str | Path | None = None) -> "PriorGuidedDiffusionSampler":
        base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
        paths = config.get("paths", {})
        missing = [key for key in ("natural_dataset_dir", "diffusion_checkpoint") if key not in paths]
        if missing:
            raise KeyError(f"Config paths is missing required keys: {missing}")
        natural_dir = (base / paths["natural_dataset_dir"]).resolve()
        diffusion_ckpt = Path(paths["diffusion_checkpoint"])
        if not diffusion_ckpt.is_absolute():
            diffusion_ckpt = (base / diffusion_ckpt).resolve()
        if not natural_dir.exists():
            raise FileNotFoundError(f"Natural diffusion dataset directory not found: {natural_dir}")
        if not diffusion_ckpt.exists():
            raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt}")
        device = config.get("training", {}).get("device", config.get("device", "auto"))
        prior = DiffusionPriorAdapter.load(natural_dir, diffusion_ckpt, device=device)
        return cls(prior, config)

    def train(self, mode: bool = True) -> "PriorGuidedDiffusionSampler":
        self.training = bool(mode)
        self.prior.model.eval()
        return self

    def eval(self) -> "PriorGuidedDiffusionSampler":
        return self.train(False)

    def _make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        device = self.prior.device
        generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _make_generators(
        self,
        batch_size: int,
        *,
        seed: int | Sequence[int] | np.ndarray | None,
    ) -> torch.Generator | list[torch.Generator] | None:
        if seed is None:
            return None
        if isinstance(seed, (list, tuple, np.ndarray)):
            seeds = [int(item) for item in seed]
            if len(seeds) != int(batch_size):
                raise ValueError(f"Expected {batch_size} seeds for batch sampling, got {len(seeds)}")
            return [self._make_generator(item) for item in seeds]  # type: ignore[list-item]
        return self._make_generator(int(seed))

    def _randn(
        self,
        shape: tuple[int, ...],
        *,
        generators: torch.Generator | list[torch.Generator] | None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"device": self.prior.device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if generators is None:
            return torch.randn(*shape, **kwargs)
        if isinstance(generators, list):
            if len(generators) != int(shape[0]):
                raise ValueError(f"Expected {shape[0]} generators, got {len(generators)}")
            return torch.cat(
                [torch.randn(1, *shape[1:], **kwargs, generator=generator) for generator in generators],
                dim=0,
            )
        return torch.randn(*shape, **kwargs, generator=generators)

    def _initial_noise(
        self,
        batch_size: int,
        *,
        generators: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        cfg = self.prior.model.denoiser.cfg
        return self._randn((batch_size, cfg.horizon_steps, cfg.action_dim), generators=generators)

    def _configured_inference_steps(self, inference_steps: int | None) -> int:
        if inference_steps is not None and int(inference_steps) > 0:
            return min(int(inference_steps), self.prior.num_steps)
        sampling_cfg = self.config.get("sampling", {})
        steps = int(sampling_cfg.get("eval_diffusion_steps", sampling_cfg.get("diffusion_steps", self.prior.num_steps)))
        return min(max(steps, 1), self.prior.num_steps)

    def _sampling_timesteps(self, inference_steps: int | None) -> list[int]:
        return list(reversed(range(self._configured_inference_steps(inference_steps))))

    def sample(
        self,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        *,
        ego_length: torch.Tensor | None = None,
        adv_length: torch.Tensor | None = None,
        num_samples: int = 1,
        seed: int | Sequence[int] | np.ndarray | None = None,
        inference_steps: int | None = None,
    ) -> PriorGuidedSampleResult:
        device = self.prior.device
        context_states = _repeat_context(context_states.to(device).float(), num_samples)
        context_features = _repeat_context(context_features.to(device).float(), num_samples)
        relative_history = _repeat_context(relative_history.to(device).float(), num_samples)
        if ego_length is not None:
            ego_length = _repeat_context(ego_length.to(device).float(), num_samples)
        if adv_length is not None:
            adv_length = _repeat_context(adv_length.to(device).float(), num_samples)

        generators = self._make_generators(context_states.shape[0], seed=seed)
        x_t = self._initial_noise(context_states.shape[0], generators=generators)
        timesteps = self._sampling_timesteps(inference_steps)
        for step in timesteps:
            t = torch.full((x_t.shape[0],), step, dtype=torch.long, device=device)
            with torch.no_grad():
                eps = self.prior.predict_eps(x_t, t, context_states, context_features, relative_history)
                x0_hat = self.prior.predict_x0(x_t, t, eps)
                mean, _variance, log_variance = self.prior.posterior_mean_variance(x_t, t, x0_hat)
            noise = self._randn(tuple(x_t.shape), generators=generators, dtype=x_t.dtype)
            mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
            x_t = (mean + mask * torch.exp(0.5 * log_variance) * noise).detach()

        raw_context = self.prior.decode_context_states(context_states)
        raw_actions = self.prior.decode_actions(x_t)
        kin = integrate_following_actions_torch(raw_actions, raw_context, ego_length, adv_length, self.prior.schema, self.prior.config)
        rss_obj, rss_diag = rss_criticality_objective(kin, self.rss_cfg)
        phy, phy_diag = physical_violation_penalty(kin, self.config)
        zeros = torch.zeros((raw_actions.shape[0],), dtype=raw_actions.dtype, device=device)
        diagnostics = {
            "rss_objective": rss_obj.detach(),
            "physics_penalty": phy.detach(),
            "trajectory_log_prob": zeros.detach(),
            "prior_kl": zeros.detach(),
            "guidance_norm": zeros.detach(),
            **{key: value.detach() for key, value in rss_diag.items()},
            **{key: value.detach() for key, value in phy_diag.items()},
        }
        return PriorGuidedSampleResult(
            normalized_actions=x_t.detach(),
            raw_actions=raw_actions.detach(),
            diagnostics=diagnostics,
            guidance_trace=[],
            trajectory_log_prob=zeros,
            prior_kl=zeros,
            guidance_norm=zeros,
        )

    def sample_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        seed: int | Sequence[int] | np.ndarray | None = None,
        inference_steps: int | None = None,
    ) -> PriorGuidedSampleResult:
        return self.sample(
            batch["context_states"],
            batch["context_features"],
            batch["relative_history"],
            ego_length=batch.get("ego_length"),
            adv_length=batch.get("adv_length"),
            num_samples=num_samples,
            seed=seed,
            inference_steps=inference_steps,
        )
