"""Prior-regularized learnable guidance sampler for Stage 1 diffusion."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .diffusion_adapter import DiffusionPriorAdapter
from .guidance_losses import physical_violation_penalty
from .guidance_policy import GuidancePolicy, GuidancePolicyConfig
from .rss import RSSConfig, rss_criticality_objective
from .torch_kinematics import integrate_following_actions_torch

logger = logging.getLogger(__name__)


@dataclass
class PriorGuidedSampleResult:
    normalized_actions: torch.Tensor
    raw_actions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    guidance_trace: list[dict[str, float]]
    trajectory_log_prob: torch.Tensor
    prior_kl: torch.Tensor
    guidance_norm: torch.Tensor


@dataclass(frozen=True)
class PriorGuidanceSchedule:
    enabled: bool = True
    guidance_start_ratio: float = 0.2
    guidance_end_ratio: float = 0.8
    guidance_clip_norm: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PriorGuidanceSchedule":
        cfg = config.get("policy", config.get("guided_denoising", config))
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            guidance_start_ratio=float(cfg.get("guidance_start_ratio", 0.2)),
            guidance_end_ratio=float(cfg.get("guidance_end_ratio", 0.8)),
            guidance_clip_norm=float(cfg.get("guidance_clip_norm", cfg.get("grad_clip_norm", 1.0))),
        )

    def active(self, step_index: int, num_steps: int) -> bool:
        if not self.enabled or num_steps <= 1:
            return False
        ratio = float(step_index) / float(num_steps - 1)
        return self.guidance_start_ratio <= ratio <= self.guidance_end_ratio


def _repeat_context(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    if num_samples <= 1:
        return tensor
    return tensor.repeat_interleave(int(num_samples), dim=0)


class PriorGuidedDiffusionSampler:
    """DDPM sampler with learnable residual guidance and KL accounting."""

    def __init__(
        self,
        prior: DiffusionPriorAdapter,
        policy: GuidancePolicy,
        config: dict[str, Any],
    ) -> None:
        self.prior = prior
        self.policy = policy.to(prior.device)
        self.config = config
        self.schedule = PriorGuidanceSchedule.from_config(config)
        self.rss_cfg = RSSConfig.from_config(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, config_dir: str | Path | None = None) -> "PriorGuidedDiffusionSampler":
        base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
        paths = config.get("paths", {})
        natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
        diffusion_ckpt = Path(paths.get("diffusion_checkpoint", "checkpoints/best_noise_mse.pt"))
        if not diffusion_ckpt.is_absolute():
            diffusion_ckpt = (base / diffusion_ckpt).resolve()
            if not diffusion_ckpt.exists():
                diffusion_ckpt = (natural_dir / paths.get("diffusion_checkpoint", "checkpoints/best_noise_mse.pt")).resolve()
        device = config.get("training", {}).get("device", config.get("device", "auto"))
        prior = DiffusionPriorAdapter.load(natural_dir, diffusion_ckpt, device=device)
        policy_cfg = GuidancePolicyConfig.from_prior(prior.model.denoiser.cfg, config)
        policy = GuidancePolicy(policy_cfg)
        policy_ckpt = str(paths.get("policy_checkpoint", "") or "")
        if policy_ckpt:
            ckpt = Path(policy_ckpt)
            if not ckpt.is_absolute():
                ckpt = (base / ckpt).resolve()
            state = torch.load(ckpt, map_location=prior.device)
            policy.load_state_dict(state["policy_state"])
        return cls(prior, policy, config)

    def train(self, mode: bool = True) -> "PriorGuidedDiffusionSampler":
        self.policy.train(mode)
        self.prior.model.eval()
        return self

    def eval(self) -> "PriorGuidedDiffusionSampler":
        return self.train(False)

    def set_guidance_enabled(self, enabled: bool) -> None:
        self.schedule = PriorGuidanceSchedule(
            enabled=bool(enabled),
            guidance_start_ratio=self.schedule.guidance_start_ratio,
            guidance_end_ratio=self.schedule.guidance_end_ratio,
            guidance_clip_norm=self.schedule.guidance_clip_norm,
        )

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
        device = self.prior.device
        kwargs: dict[str, Any] = {"device": device}
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
        shape = (batch_size, cfg.horizon_steps, cfg.action_dim)
        return self._randn(shape, generators=generators)

    def _configured_inference_steps(self, inference_steps: int | None) -> int:
        if inference_steps is not None and int(inference_steps) > 0:
            steps = min(int(inference_steps), self.prior.num_steps)
        else:
            sampling_cfg = self.config.get("sampling", {})
            key = "train_diffusion_steps" if self.policy.training else "eval_diffusion_steps"
            steps = int(sampling_cfg.get(key, sampling_cfg.get("diffusion_steps", self.prior.num_steps)))
            steps = min(max(steps, 1), self.prior.num_steps)
        sampling_cfg = self.config.get("sampling", {})
        if self.policy.training and steps < self.prior.num_steps:
            if not bool(sampling_cfg.get("allow_truncated_train_ddpm", False)):
                raise ValueError(
                    "Training with train_diffusion_steps < the Stage 1 prior diffusion steps is disabled. "
                    "Subsampled DDPM/DDIM transitions are not implemented, so use the full DDPM chain "
                    f"({self.prior.num_steps} steps) or set sampling.allow_truncated_train_ddpm=true only for debugging."
                )
            logger.warning(
                "Using truncated training DDPM chain (%d/%d steps). This is a debugging mode, not a valid "
                "subsampled DDPM/DDIM sampler.",
                steps,
                self.prior.num_steps,
            )
        return steps

    def _sampling_timesteps(self, inference_steps: int | None) -> list[int]:
        steps = self._configured_inference_steps(inference_steps)
        return list(reversed(range(steps)))

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
        log_prob_sum = torch.zeros((x_t.shape[0],), dtype=x_t.dtype, device=device)
        prior_kl_sum = torch.zeros_like(log_prob_sum)
        guidance_norm_sum = torch.zeros_like(log_prob_sum)
        trace: list[dict[str, float]] = []

        timesteps = self._sampling_timesteps(inference_steps)
        for step_index, step in enumerate(timesteps):
            t = torch.full((x_t.shape[0],), step, dtype=torch.long, device=device)
            with torch.no_grad():
                eps = self.prior.predict_eps(x_t, t, context_states, context_features, relative_history)
                x0_hat = self.prior.predict_x0(x_t, t, eps)
                posterior_mean, posterior_var, posterior_log_var = self.prior.posterior_mean_variance(x_t, t, x0_hat)

            active = self.schedule.active(len(timesteps) - 1 - step_index, len(timesteps))
            guidance = torch.zeros_like(x_t)
            mean = posterior_mean
            if active:
                guidance = self.policy(x_t.detach(), t, context_states, context_features, relative_history)
                clip = float(self.schedule.guidance_clip_norm)
                guidance_norm = torch.clamp(guidance.flatten(1).norm(dim=1), min=1e-12)
                if clip > 0:
                    scale = torch.clamp(clip / guidance_norm, max=1.0).view(-1, 1, 1)
                    guidance = guidance * scale
                    guidance_norm = guidance.flatten(1).norm(dim=1)
                mean = posterior_mean + posterior_var * guidance
                prior_kl = 0.5 * (posterior_var * guidance.square()).flatten(1).sum(dim=1)
                prior_kl_sum = prior_kl_sum + prior_kl
                guidance_norm_sum = guidance_norm_sum + guidance_norm

            noise = self._randn(tuple(x_t.shape), generators=generators, dtype=x_t.dtype)
            mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
            std = torch.exp(0.5 * posterior_log_var)
            x_next = (mean + mask * std * noise).detach()

            if active and step > 0:
                diff = x_next - mean
                log_prob = -0.5 * (
                    diff.square() / torch.clamp(posterior_var, min=1e-20)
                    + posterior_log_var
                    + float(np.log(2.0 * np.pi))
                )
                log_prob_sum = log_prob_sum + log_prob.flatten(1).sum(dim=1)

            if step % max(1, self.prior.num_steps // 10) == 0 or step == timesteps[-1]:
                trace.append(
                    {
                        "timestep": float(step),
                        "active": float(active),
                        "prior_kl": float(prior_kl_sum.detach().mean().cpu()),
                        "guidance_norm": float(guidance.flatten(1).norm(dim=1).detach().mean().cpu()),
                    }
                )
            x_t = x_next

        raw_context = self.prior.decode_context_states(context_states)
        raw_actions = self.prior.decode_actions(x_t)
        kin = integrate_following_actions_torch(raw_actions, raw_context, ego_length, adv_length, self.prior.schema, self.prior.config)
        rss_obj, rss_diag = rss_criticality_objective(kin, self.rss_cfg)
        phy, phy_diag = physical_violation_penalty(kin, self.config)
        diagnostics = {
            "rss_objective": rss_obj.detach(),
            "physics_penalty": phy.detach(),
            "trajectory_log_prob": log_prob_sum.detach(),
            "prior_kl": prior_kl_sum.detach(),
            "guidance_norm": guidance_norm_sum.detach(),
            **{key: value.detach() for key, value in rss_diag.items()},
            **{key: value.detach() for key, value in phy_diag.items()},
        }
        return PriorGuidedSampleResult(
            normalized_actions=x_t.detach(),
            raw_actions=raw_actions.detach(),
            diagnostics=diagnostics,
            guidance_trace=trace,
            trajectory_log_prob=log_prob_sum,
            prior_kl=prior_kl_sum,
            guidance_norm=guidance_norm_sum.detach(),
        )

    def sample_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        seed: int | Sequence[int] | np.ndarray | None = None,
        inference_steps: int | None = None,
    ) -> PriorGuidedSampleResult:
        """Sample a batch of plans from tensors keyed like runner observations.

        This is a thin hook for faster training paths: DDPM sampling is already
        vectorized in ``sample()``, while highway-env rollout can remain scalar.
        """
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


def result_to_numpy(result: PriorGuidedSampleResult) -> dict[str, np.ndarray]:
    out = {
        "normalized_actions": result.normalized_actions.detach().cpu().numpy().astype(np.float32),
        "actions": result.raw_actions.detach().cpu().numpy().astype(np.float32),
        "trajectory_log_prob": result.trajectory_log_prob.detach().cpu().numpy().astype(np.float32),
        "prior_kl": result.prior_kl.detach().cpu().numpy().astype(np.float32),
        "guidance_norm": result.guidance_norm.detach().cpu().numpy().astype(np.float32),
    }
    for key, value in result.diagnostics.items():
        out[key] = value.detach().cpu().numpy().astype(np.float32)
    return out
