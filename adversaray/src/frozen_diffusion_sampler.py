"""Frozen diffusion-prior sampler used by KING-guided experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .diffusion_adapter import DiffusionPriorAdapter
from .king_gradient_guidance import compute_king_risk
from .physics_losses import physical_violation_penalty
from .rss import RSSConfig, rss_criticality_objective
from .adversary_dynamics import integrate_adversary_actions_torch


@dataclass
class FrozenDiffusionSampleResult:
    normalized_actions: torch.Tensor
    raw_actions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _repeat_context(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    if num_samples <= 1:
        return tensor
    return tensor.repeat_interleave(int(num_samples), dim=0)


class FrozenDiffusionSampler:
    """Thin DDPM sampler around the frozen Stage 1 diffusion prior."""

    def __init__(
        self,
        prior: DiffusionPriorAdapter,
        config: dict[str, Any],
    ) -> None:
        self.prior = prior
        self.config = config
        self.rss_cfg = RSSConfig.from_config(config)
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
        if config_dir is not None:
            base = Path(config_dir).resolve()
        else:
            base = Path.cwd()
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
        return cls(prior, config)

    def train(self, mode: bool = True) -> "FrozenDiffusionSampler":
        self.training = bool(mode)
        self.prior.model.eval()
        return self

    def eval(self) -> "FrozenDiffusionSampler":
        return self.train(False)

    def _make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        device = self.prior.device
        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
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
                raise ValueError(
                    f"Expected {batch_size} seeds for batch sampling, "
                    f"got {len(seeds)}"
                )
            return [self._make_generator(item) for item in seeds]
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
                raise ValueError(
                    f"Expected {shape[0]} generators, got {len(generators)}"
                )
            chunks = [
                torch.randn(1, *shape[1:], **kwargs, generator=generator)
                for generator in generators
            ]
            return torch.cat(chunks, dim=0)
        return torch.randn(*shape, **kwargs, generator=generators)

    def _initial_noise(
        self,
        batch_size: int,
        *,
        generators: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        cfg = self.prior.model.denoiser.cfg
        return self._randn(
            (batch_size, cfg.horizon_steps, cfg.action_dim),
            generators=generators,
        )

    def _sampling_timesteps(self, inference_steps: int | None) -> list[int]:
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
        return list(reversed(range(steps)))

    def _risk_tilted_config(
        self,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        defaults = {
            "enabled": False,
            "late_fraction": 0.60,
            "num_late_steps": 0,
            "guidance_scale": 20.0,
            "scale_schedule": "constant",
            "guidance_variance_mode": "posterior_std",
            "max_grad_norm": 1.0,
            "normalize_grad": True,
            "scale_by_sqrt_dim": False,
            "apply_at_t0": False,
            "lambda_action_l2": 0.0,
            "min_grad_norm": 1e-12,
            "nan_to_num": True,
            "save_guidance_diagnostics": True,
        }
        cfg = dict(self.config.get("risk_tilted_diffusion", {}))
        defaults.update(cfg)
        if override:
            defaults.update(override)
        return defaults

    def _guided_loop_indices(
        self,
        timesteps: list[int],
        cfg: dict[str, Any],
    ) -> set[int]:
        if not bool(cfg.get("enabled", False)):
            return set()
        if int(cfg.get("num_late_steps", 0)) > 0:
            late_count = min(len(timesteps), int(cfg["num_late_steps"]))
        else:
            fraction = float(cfg.get("late_fraction", 0.30))
            late_count = max(1, int(round(len(timesteps) * fraction)))
        return set(range(len(timesteps) - late_count, len(timesteps)))

    def _normalize_guidance_grad(
        self,
        grad: torch.Tensor,
        cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(cfg.get("nan_to_num", True)):
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        raw_grad_norm = grad.flatten(1).norm(dim=1)
        min_norm = max(float(cfg.get("min_grad_norm", 1e-12)), 1e-30)
        safe_norm = torch.clamp(raw_grad_norm, min=min_norm)
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

        if bool(cfg.get("normalize_grad", True)):
            grad = grad / safe_norm.view(-1, 1, 1)
            if max_grad_norm > 0.0:
                grad = grad * max_grad_norm
        elif max_grad_norm > 0.0:
            scale = torch.clamp(max_grad_norm / safe_norm, max=1.0)
            grad = grad * scale.view(-1, 1, 1)

        if bool(cfg.get("scale_by_sqrt_dim", False)):
            dim_scale = float(grad[0].numel()) ** 0.5
            grad = grad * dim_scale
        return grad, raw_grad_norm

    def _guidance_scale(
        self,
        *,
        loop_idx: int,
        guided_loop_indices: set[int],
        cfg: dict[str, Any],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        base = float(cfg.get("guidance_scale", 20.0))
        schedule = str(cfg.get("scale_schedule", "linear_ramp")).lower()
        if schedule == "constant":
            scale = base
        elif schedule == "linear_ramp":
            ordered = sorted(guided_loop_indices)
            if loop_idx in guided_loop_indices:
                position = ordered.index(loop_idx) + 1
            else:
                position = 0
            scale = base * float(position) / max(float(len(ordered)), 1.0)
        else:
            raise ValueError(f"Unknown risk-tilted scale_schedule: {schedule}")
        return torch.tensor(scale, dtype=dtype, device=device)

    def _guidance_variance_multiplier(
        self,
        posterior_variance: torch.Tensor,
        cfg: dict[str, Any],
    ) -> torch.Tensor:
        mode = str(
            cfg.get("guidance_variance_mode", "posterior_variance")
        ).lower()
        if mode in {"posterior_variance", "variance"}:
            return posterior_variance
        if mode in {"posterior_std", "sqrt_variance", "std"}:
            return torch.sqrt(torch.clamp(posterior_variance, min=0.0))
        if mode in {"none", "constant"}:
            return torch.ones_like(posterior_variance)
        raise ValueError(f"Unknown guidance_variance_mode: {mode}")

    def _risk_guided_reverse_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        raw_context: torch.Tensor,
        ego_length: torch.Tensor | None,
        adv_length: torch.Tensor | None,
        generators: torch.Generator | list[torch.Generator] | None,
        loop_idx: int,
        guided_loop_indices: set[int],
        cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        apply_at_t0 = bool(cfg.get("apply_at_t0", False))
        if bool(torch.any(t == 0).item()) and not apply_at_t0:
            raise RuntimeError(
                "risk-guided reverse step received t == 0 while "
                "apply_at_t0 is false"
            )

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            eps = self.prior.predict_eps(
                x_in,
                t,
                context_states,
                context_features,
                relative_history,
            )
            x0_hat = self.prior.predict_x0(x_in, t, eps)
            raw_actions_hat = self.prior.decode_actions(x0_hat)
            kin = integrate_adversary_actions_torch(
                raw_actions_hat,
                raw_context,
                ego_length,
                adv_length,
                self.prior.schema,
                self.config,
            )
            risk, _risk_diag = compute_king_risk(kin, self.config)
            physics, _physics_diag = physical_violation_penalty(
                kin,
                self.config,
            )
            action_l2 = raw_actions_hat.square().flatten(1).mean(dim=1)
            objective = (
                risk
                - float(cfg.get("lambda_action_l2", 0.0)) * action_l2
            )
            grad = torch.autograd.grad(
                objective.sum(),
                x_in,
                retain_graph=False,
                create_graph=False,
            )[0]

        grad, raw_grad_norm = self._normalize_guidance_grad(grad, cfg)

        scale = self._guidance_scale(
            loop_idx=loop_idx,
            guided_loop_indices=guided_loop_indices,
            cfg=cfg,
            dtype=x_t.dtype,
            device=x_t.device,
        )
        posterior_mean, posterior_var, posterior_log_var = (
            self.prior.posterior_mean_variance(x_in, t, x0_hat)
        )
        variance_multiplier = self._guidance_variance_multiplier(
            posterior_var,
            cfg,
        )
        effective_scale = scale * variance_multiplier
        posterior_mean = posterior_mean + effective_scale * grad.detach()
        noise = self._randn(
            tuple(x_t.shape),
            generators=generators,
            dtype=x_t.dtype,
        )
        mask = (t != 0).float().reshape(
            x_t.shape[0],
            *((1,) * (x_t.ndim - 1)),
        )
        diffusion_noise = mask * torch.exp(0.5 * posterior_log_var) * noise
        x_next = posterior_mean + diffusion_noise
        diagnostics = {
            "guidance_objective": objective.detach(),
            "guidance_risk": risk.detach(),
            "guidance_physics": physics.detach(),
            "guidance_action_l2": action_l2.detach(),
            "guidance_grad_norm": raw_grad_norm.detach(),
            "guidance_scale": torch.full_like(
                risk.detach(),
                float(scale.detach().cpu()),
            ),
            "guidance_variance_multiplier": (
                variance_multiplier.flatten(1).mean(dim=1).detach()
            ),
            "guidance_effective_scale": (
                effective_scale.flatten(1).mean(dim=1).detach()
            ),
        }
        return x_next.detach(), diagnostics

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
        risk_tilted: bool | None = None,
        risk_tilted_config: dict[str, Any] | None = None,
    ) -> FrozenDiffusionSampleResult:
        device = self.prior.device
        context_states = _repeat_context(
            context_states.to(device).float(),
            num_samples,
        )
        context_features = _repeat_context(
            context_features.to(device).float(),
            num_samples,
        )
        relative_history = _repeat_context(
            relative_history.to(device).float(),
            num_samples,
        )
        if ego_length is not None:
            ego_length = _repeat_context(
                ego_length.to(device).float(),
                num_samples,
            )
        if adv_length is not None:
            adv_length = _repeat_context(
                adv_length.to(device).float(),
                num_samples,
            )

        generators = self._make_generators(context_states.shape[0], seed=seed)
        x_t = self._initial_noise(
            context_states.shape[0],
            generators=generators,
        )

        timesteps = self._sampling_timesteps(inference_steps)
        tilted_cfg = self._risk_tilted_config(risk_tilted_config)
        if risk_tilted is None:
            risk_tilted = bool(tilted_cfg.get("enabled", False))
        tilted_cfg["enabled"] = bool(risk_tilted)
        guided_loop_indices = self._guided_loop_indices(timesteps, tilted_cfg)
        if not bool(tilted_cfg.get("apply_at_t0", False)):
            guided_loop_indices = {
                idx for idx in guided_loop_indices if int(timesteps[idx]) > 0
            }
            if bool(risk_tilted) and not guided_loop_indices:
                positive_indices = [
                    idx for idx, step in enumerate(timesteps) if int(step) > 0
                ]
                if positive_indices:
                    guided_loop_indices = {positive_indices[-1]}
        raw_context = self.prior.decode_context_states(context_states).detach()
        guidance_diagnostics: dict[str, list[torch.Tensor]] = {}
        num_guided_steps = 0

        for loop_idx, step in enumerate(timesteps):
            t = torch.full(
                (x_t.shape[0],),
                step,
                dtype=torch.long,
                device=device,
            )
            is_guided = bool(risk_tilted) and loop_idx in guided_loop_indices
            if is_guided:
                num_guided_steps += 1
                x_t, guided_diag = self._risk_guided_reverse_step(
                    x_t,
                    t,
                    context_states=context_states,
                    context_features=context_features,
                    relative_history=relative_history,
                    raw_context=raw_context,
                    ego_length=ego_length,
                    adv_length=adv_length,
                    generators=generators,
                    loop_idx=loop_idx,
                    guided_loop_indices=guided_loop_indices,
                    cfg=tilted_cfg,
                )
                if bool(tilted_cfg.get("save_guidance_diagnostics", True)):
                    for key, value in guided_diag.items():
                        chunks = guidance_diagnostics.setdefault(key, [])
                        chunks.append(value.detach())
            else:
                with torch.no_grad():
                    eps = self.prior.predict_eps(
                        x_t,
                        t,
                        context_states,
                        context_features,
                        relative_history,
                    )
                    x0_hat = self.prior.predict_x0(x_t, t, eps)
                    posterior_mean, _posterior_var, posterior_log_var = (
                        self.prior.posterior_mean_variance(x_t, t, x0_hat)
                    )
                    noise = self._randn(
                        tuple(x_t.shape),
                        generators=generators,
                        dtype=x_t.dtype,
                    )
                    mask = (t != 0).float().reshape(
                        x_t.shape[0],
                        *((1,) * (x_t.ndim - 1)),
                    )
                    x_t = (
                        posterior_mean
                        + mask * torch.exp(0.5 * posterior_log_var) * noise
                    ).detach()

        raw_actions = self.prior.decode_actions(x_t)
        kin = integrate_adversary_actions_torch(
            raw_actions,
            raw_context,
            ego_length,
            adv_length,
            self.prior.schema,
            self.config,
        )
        rss_obj, rss_diag = rss_criticality_objective(kin, self.rss_cfg)
        phy, phy_diag = physical_violation_penalty(kin, self.config)
        diagnostics = {
            "rss_objective": rss_obj.detach(),
            "physics_penalty": phy.detach(),
            **{key: value.detach() for key, value in rss_diag.items()},
            **{key: value.detach() for key, value in phy_diag.items()},
        }
        for key, chunks in guidance_diagnostics.items():
            diagnostics[key] = torch.stack(chunks, dim=0).detach()
        diagnostics["guidance_steps"] = torch.full(
            (x_t.shape[0],),
            float(num_guided_steps),
            dtype=x_t.dtype,
            device=x_t.device,
        )
        return FrozenDiffusionSampleResult(
            normalized_actions=x_t.detach(),
            raw_actions=raw_actions.detach(),
            diagnostics=diagnostics,
        )

    def sample_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        seed: int | Sequence[int] | np.ndarray | None = None,
        inference_steps: int | None = None,
        risk_tilted: bool | None = None,
        risk_tilted_config: dict[str, Any] | None = None,
    ) -> FrozenDiffusionSampleResult:
        return self.sample(
            batch["context_states"],
            batch["context_features"],
            batch["relative_history"],
            ego_length=batch.get("ego_length"),
            adv_length=batch.get("adv_length"),
            num_samples=num_samples,
            seed=seed,
            inference_steps=inference_steps,
            risk_tilted=risk_tilted,
            risk_tilted_config=risk_tilted_config,
        )
