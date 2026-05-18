"""RSS + naturalness + physics guided diffusion sampler."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diffusion.src.utils import load_json

from .diffusion_adapter import DiffusionPriorAdapter
from .discriminator import NaturalnessDiscriminator, build_discriminator_from_schema, score_naturalness
from .guidance_losses import physical_violation_penalty
from .guidance_schedule import GuidanceSchedule
from .rss import RSSConfig, rss_criticality_objective
from .torch_kinematics import integrate_following_actions_torch


@dataclass
class GuidedSampleResult:
    normalized_actions: torch.Tensor
    raw_actions: torch.Tensor
    acceleration: torch.Tensor
    velocity: torch.Tensor
    displacement: torch.Tensor
    gap: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    guidance_trace: list[dict[str, float]]


def _load_discriminator(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[NaturalnessDiscriminator, dict[str, Any], dict[str, Any], dict[str, Any]]:
    ckpt = Path(checkpoint_path).resolve()
    state = torch.load(ckpt, map_location=device)
    schema = state["schema"]
    config = state.get("config", {})
    model = build_discriminator_from_schema(schema, config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    stats_path = ckpt.parent.parent / "discriminator_stats.json"
    stats = load_json(stats_path)
    return model, schema, config, stats


class GuidedDiffusionSampler:
    def __init__(
        self,
        prior: DiffusionPriorAdapter,
        discriminator: NaturalnessDiscriminator,
        discriminator_schema: dict[str, Any],
        discriminator_config: dict[str, Any],
        discriminator_stats: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        self.prior = prior
        self.discriminator = discriminator
        self.discriminator_schema = discriminator_schema
        self.discriminator_config = discriminator_config
        self.discriminator_stats = discriminator_stats
        self.config = config
        self.schedule = GuidanceSchedule.from_config(config)
        self.rss_cfg = RSSConfig.from_config(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, config_dir: str | Path | None = None) -> "GuidedDiffusionSampler":
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
        disc_ckpt = Path(paths.get("discriminator_checkpoint", "../../../data/adversaray/following/discriminator/checkpoints/best_auc.pt"))
        if not disc_ckpt.is_absolute():
            disc_ckpt = (base / disc_ckpt).resolve()
        discriminator, disc_schema, disc_config, disc_stats = _load_discriminator(disc_ckpt, prior.device)
        return cls(prior, discriminator, disc_schema, disc_config, disc_stats, config)

    def _repeat_context(
        self,
        tensor: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples <= 1:
            return tensor
        return tensor.repeat_interleave(int(num_samples), dim=0)

    def sample(
        self,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        *,
        ego_length: torch.Tensor | None = None,
        adv_length: torch.Tensor | None = None,
        num_samples: int = 1,
        seed: int | None = None,
    ) -> GuidedSampleResult:
        device = self.prior.device
        context_states = self._repeat_context(context_states.to(device).float(), num_samples)
        context_features = self._repeat_context(context_features.to(device).float(), num_samples)
        relative_history = self._repeat_context(relative_history.to(device).float(), num_samples)
        if ego_length is not None:
            ego_length = self._repeat_context(ego_length.to(device).float(), num_samples)
        if adv_length is not None:
            adv_length = self._repeat_context(adv_length.to(device).float(), num_samples)
        if seed is not None:
            generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
            generator.manual_seed(int(seed))
            x_t = torch.randn(
                context_states.shape[0],
                self.prior.model.denoiser.cfg.horizon_steps,
                self.prior.model.denoiser.cfg.action_dim,
                device=device,
                generator=generator,
            )
        else:
            x_t = torch.randn(
                context_states.shape[0],
                self.prior.model.denoiser.cfg.horizon_steps,
                self.prior.model.denoiser.cfg.action_dim,
                device=device,
            )
        trace: list[dict[str, float]] = []
        raw_context = self.prior.decode_context_states(context_states)

        for step in reversed(range(self.prior.num_steps)):
            t = torch.full((x_t.shape[0],), step, dtype=torch.long, device=device)
            weights = self.schedule.weights_for_timestep(step, self.prior.num_steps)
            with torch.enable_grad():
                x_work = x_t.detach().requires_grad_(weights.active)
                eps = self.prior.predict_eps(x_work, t, context_states, context_features, relative_history)
                x0_hat = self.prior.predict_x0(x_work, t, eps)
                posterior_mean, posterior_var, posterior_log_var = self.prior.posterior_mean_variance(x_work, t, x0_hat)
                if weights.active:
                    raw_actions = self.prior.decode_actions(x0_hat)
                    kin = integrate_following_actions_torch(raw_actions, raw_context, ego_length, adv_length, self.prior.schema, self.prior.config)
                    rss_obj, rss_diag = rss_criticality_objective(kin, self.rss_cfg)
                    logits, _scores = score_naturalness(
                        self.discriminator,
                        context_states,
                        context_features,
                        relative_history,
                        x0_hat,
                        ego_length=ego_length,
                        adv_length=adv_length,
                        schema=self.prior.schema,
                        config=self.discriminator_config,
                        discriminator_stats=self.discriminator_stats,
                        stage1_stats=self.prior.stats,
                        inputs_normalized=True,
                        actions_normalized=True,
                    )
                    log_nat = F.logsigmoid(logits)
                    phy, phy_diag = physical_violation_penalty(kin, self.config)
                    objective = weights.lambda_rss * rss_obj + weights.lambda_nat * log_nat - weights.lambda_phy * phy
                    grad = torch.autograd.grad(objective.mean(), x_work, retain_graph=False, create_graph=False)[0]
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                    clip = float(self.schedule.grad_clip_norm)
                    if clip > 0:
                        norm = torch.clamp(grad.flatten(1).norm(dim=1), min=1e-12)
                        scale = torch.clamp(clip / norm, max=1.0).view(-1, 1, 1)
                        grad = grad * scale
                    posterior_mean = posterior_mean + posterior_var * grad
                    if step % max(1, self.prior.num_steps // 10) == 0:
                        trace.append(
                            {
                                "timestep": float(step),
                                "rss": float(rss_obj.detach().mean().cpu()),
                                "log_nat": float(log_nat.detach().mean().cpu()),
                                "physics": float(phy.detach().mean().cpu()),
                                "objective": float(objective.detach().mean().cpu()),
                                "min_rss_margin": float(rss_diag["min_rss_margin"].detach().mean().cpu()),
                                "negative_speed_rate": float(phy_diag["negative_speed_rate"].detach().mean().cpu()),
                            }
                        )
            noise = torch.randn_like(x_t)
            mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
            x_t = (posterior_mean + mask * torch.exp(0.5 * posterior_log_var) * noise).detach()

        raw_actions = self.prior.decode_actions(x_t)
        kin = integrate_following_actions_torch(raw_actions, raw_context, ego_length, adv_length, self.prior.schema, self.prior.config)
        rss_obj, rss_diag = rss_criticality_objective(kin, self.rss_cfg)
        logits, scores = score_naturalness(
            self.discriminator,
            context_states,
            context_features,
            relative_history,
            x_t,
            ego_length=ego_length,
            adv_length=adv_length,
            schema=self.prior.schema,
            config=self.discriminator_config,
            discriminator_stats=self.discriminator_stats,
            stage1_stats=self.prior.stats,
            inputs_normalized=True,
            actions_normalized=True,
        )
        phy, phy_diag = physical_violation_penalty(kin, self.config)
        diagnostics = {
            "rss_objective": rss_obj.detach(),
            "naturalness_logit": logits.detach(),
            "naturalness_score": scores.detach(),
            "physics_penalty": phy.detach(),
            **{key: value.detach() for key, value in rss_diag.items()},
            **{key: value.detach() for key, value in phy_diag.items()},
        }
        return GuidedSampleResult(
            normalized_actions=x_t.detach(),
            raw_actions=raw_actions.detach(),
            acceleration=kin.acceleration.detach(),
            velocity=kin.velocity.detach(),
            displacement=kin.displacement.detach(),
            gap=kin.gap.detach(),
            diagnostics=diagnostics,
            guidance_trace=trace,
        )


def result_to_numpy(result: GuidedSampleResult) -> dict[str, np.ndarray]:
    out = {
        "normalized_actions": result.normalized_actions.cpu().numpy().astype(np.float32),
        "actions": result.raw_actions.cpu().numpy().astype(np.float32),
        "acceleration": result.acceleration.cpu().numpy().astype(np.float32),
        "velocity": result.velocity.cpu().numpy().astype(np.float32),
        "displacement": result.displacement.cpu().numpy().astype(np.float32),
        "gap": result.gap.cpu().numpy().astype(np.float32),
    }
    for key, value in result.diagnostics.items():
        out[key] = value.cpu().numpy().astype(np.float32)
    return out

