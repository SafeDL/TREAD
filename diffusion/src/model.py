"""History-conditioned action diffusion prior."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionDiffusionConfig:
    history_steps: int
    num_actors: int
    state_features: int
    context_dim: int
    relative_dim: int
    horizon_steps: int
    action_dim: int
    dt: float = 0.04
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 100
    x0_clip_abs: float = 0.0
    x0_weight: float = 0.0
    smooth_weight: float = 0.0
    trajectory_y_weight: float = 0.0
    trajectory_x_weight: float = 0.0
    trajectory_vy_weight: float = 0.0
    trajectory_vx_weight: float = 0.0
    endpoint_y_weight: float = 0.0
    endpoint_x_weight: float = 0.0
    cross_y_weight: float = 0.0
    end_y_weight: float = 0.0
    kinematic_consistency_weight: float = 0.0
    action_representation: str = "acceleration"
    generation_target: str = "action"


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SceneConditionEncoder(nn.Module):
    def __init__(self, cfg: ActionDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_proj = nn.Linear(cfg.state_features, cfg.hidden_dim)
        self.actor_type = nn.Parameter(torch.zeros(1, cfg.num_actors, cfg.hidden_dim))
        self.time_pos = nn.Parameter(torch.zeros(1, cfg.history_steps, cfg.hidden_dim))
        self.temporal = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.rel_proj = nn.Linear(cfg.relative_dim, cfg.hidden_dim)
        self.rel_temporal = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.context_proj = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 3),
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        history: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
    ) -> torch.Tensor:
        b, steps, actors, features = history.shape
        if steps != self.cfg.history_steps or actors != self.cfg.num_actors or features != self.cfg.state_features:
            raise ValueError(f"Unexpected history shape {tuple(history.shape)}")
        seq = history.permute(0, 2, 1, 3).reshape(b * actors, steps, features)
        x = self.state_proj(seq) + self.time_pos
        _, h = self.temporal(x)
        actor_tokens = h[-1].reshape(b, actors, -1) + self.actor_type
        scene_token = actor_tokens.mean(dim=1)
        rel_x = self.rel_proj(relative_history) + self.time_pos
        _, rel_h = self.rel_temporal(rel_x)
        rel_token = rel_h[-1]
        context_token = self.context_proj(context_features)
        return self.fuse(torch.cat([scene_token, rel_token, context_token], dim=-1))


class FiLMTransformerBlock(nn.Module):
    def __init__(self, cfg: ActionDiffusionConfig) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=max(1, cfg.num_heads),
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
        )

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        tokens = self.layer(tokens)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ActionDenoiser(nn.Module):
    """Predict diffusion noise for an action sequence."""

    def __init__(self, cfg: ActionDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_encoder = SceneConditionEncoder(cfg)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, cfg.horizon_steps, cfg.hidden_dim))
        self.timestep_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.layers = nn.ModuleList([FiLMTransformerBlock(cfg) for _ in range(max(1, cfg.num_layers))])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        history: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.shape[1] != self.cfg.horizon_steps:
            raise ValueError(f"Expected horizon={self.cfg.horizon_steps}, got {noisy_actions.shape[1]}")
        cond = self.cond_encoder(history, context_features, relative_history)
        t_emb = self.timestep_mlp(sinusoidal_embedding(timesteps, self.cfg.hidden_dim))
        tokens = self.action_proj(noisy_actions) + self.action_pos
        cond = cond + t_emb
        tokens = tokens + cond.unsqueeze(1)
        for layer in self.layers:
            tokens = layer(tokens, cond)
        return self.out(tokens)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cosine beta schedule for DDPM noise training."""
    steps = int(timesteps) + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 1e-5, 0.999), dtype=dtype)


def extract_coeff(coeff: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    b = timesteps.shape[0]
    out = coeff.gather(0, timesteps)
    return out.reshape(b, *((1,) * (len(shape) - 1)))


def _alpha_at(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    safe_t = torch.clamp(timesteps, min=0)
    value = extract_coeff(alphas_cumprod, safe_t, shape)
    one = torch.ones_like(value)
    mask = (timesteps < 0).view(-1, *((1,) * (len(shape) - 1)))
    return torch.where(mask, one, value)


class GaussianActionDiffusion(nn.Module):
    def __init__(self, denoiser: ActionDenoiser, diffusion_steps: int) -> None:
        super().__init__()
        self.denoiser = denoiser
        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.num_steps = int(diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_coeff(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract_coeff(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_coeff(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract_coeff(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def maybe_clip_x0(self, x0: torch.Tensor) -> torch.Tensor:
        clip = float(getattr(self.denoiser.cfg, "x0_clip_abs", 0.0))
        if clip <= 0.0:
            return x0
        return torch.clamp(x0, -clip, clip)

    @staticmethod
    def _integrate_cutin_accel_torch(
        initial_target: torch.Tensor,
        accel: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        dt_safe = max(float(dt), 1.0e-6)
        x0 = initial_target[:, 0]
        y0 = initial_target[:, 1]
        x = initial_target[:, 0]
        y = initial_target[:, 1]
        vx = initial_target[:, 2]
        vy = initial_target[:, 3]
        rows: list[torch.Tensor] = []
        for step in range(accel.shape[1]):
            ax = accel[:, step, 0]
            ay = accel[:, step, 1]
            x = x + vx * dt_safe + 0.5 * ax * dt_safe * dt_safe
            y = y + vy * dt_safe + 0.5 * ay * dt_safe * dt_safe
            vx = vx + ax * dt_safe
            vy = vy + ay * dt_safe
            rows.append(torch.stack([x - x0, y - y0, vx, vy], dim=-1))
        return torch.stack(rows, dim=1)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        prev_timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.maybe_clip_x0(self.predict_start_from_noise(x_t, timesteps, eps))
        alpha_prev = _alpha_at(
            self.alphas_cumprod,
            prev_timesteps,
            x_t.shape,
        )
        return (
            torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0
            + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps
        )

    def p_losses(
        self,
        actions: torch.Tensor,
        history: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        trajectory_meta: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        b = actions.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=actions.device, dtype=torch.long)
        noise = torch.randn_like(actions)
        noisy = self.q_sample(actions, t, noise)
        pred = self.denoiser(noisy, t, history, context_features, relative_history)
        noise_mse = F.mse_loss(pred, noise)
        x0 = self.predict_start_from_noise(noisy, t, pred)
        x0_l1 = F.l1_loss(x0, actions)
        if x0.shape[1] > 1:
            smooth = torch.mean(torch.abs(x0[:, 1:] - x0[:, :-1]))
        else:
            smooth = torch.zeros((), device=actions.device, dtype=actions.dtype)
        loss = noise_mse + self.denoiser.cfg.x0_weight * x0_l1 + self.denoiser.cfg.smooth_weight * smooth
        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "noise_mse": noise_mse.detach(),
            "x0_l1": x0_l1.detach(),
            "smooth": smooth.detach(),
        }
        cfg = self.denoiser.cfg
        if (
            cfg.generation_target
            in {"maneuver_trajectory", "maneuver_acceleration"}
            and trajectory_meta is not None
        ):
            action_stats = trajectory_meta.get("action_stats")
            if action_stats is None:
                mean = torch.zeros(actions.shape[-1], dtype=actions.dtype, device=actions.device)
                std = torch.ones(actions.shape[-1], dtype=actions.dtype, device=actions.device)
            else:
                mean = torch.as_tensor(
                    action_stats["mean"],
                    dtype=actions.dtype,
                    device=actions.device,
                )
                std = torch.as_tensor(
                    action_stats["std"],
                    dtype=actions.dtype,
                    device=actions.device,
                )
            x0_raw = x0 * std + mean
            actions_raw = actions * std + mean

            kinematic_l1 = torch.zeros((), dtype=actions.dtype, device=actions.device)
            context_stats = trajectory_meta.get("context_stats")
            pred_traj = x0_raw
            target_traj = actions_raw
            if cfg.generation_target == "maneuver_acceleration":
                target_traj = trajectory_meta.get("trajectory_targets")
                if target_traj is None:
                    raise KeyError(
                        "maneuver_acceleration loss requires trajectory_targets"
                    )
                target_traj = target_traj.to(actions.device).float()
                if context_stats is None:
                    raise KeyError(
                        "maneuver_acceleration loss requires context_stats"
                    )
                state_mean = torch.as_tensor(
                    context_stats["mean"],
                    dtype=history.dtype,
                    device=history.device,
                )
                state_std = torch.as_tensor(
                    context_stats["std"],
                    dtype=history.dtype,
                    device=history.device,
                )
                history_raw = history * state_std + state_mean
                initial = history_raw[:, -1, 1]
                pred_traj = self._integrate_cutin_accel_torch(
                    initial,
                    x0_raw,
                    float(cfg.dt),
                )
            elif context_stats is not None:
                state_mean = torch.as_tensor(
                    context_stats["mean"],
                    dtype=history.dtype,
                    device=history.device,
                )
                state_std = torch.as_tensor(
                    context_stats["std"],
                    dtype=history.dtype,
                    device=history.device,
                )
                history_raw = history * state_std + state_mean
                initial = history_raw[:, -1, 1]
                x_pos = initial[:, None, 0] + x0_raw[:, :, 0]
                y_pos = initial[:, None, 1] + x0_raw[:, :, 1]
                vx_from_pos = (
                    torch.diff(
                        torch.cat([initial[:, None, 0], x_pos], dim=1),
                        dim=1,
                    )
                    / max(float(cfg.dt), 1.0e-6)
                )
                vy_from_pos = (
                    torch.diff(
                        torch.cat([initial[:, None, 1], y_pos], dim=1),
                        dim=1,
                    )
                    / max(float(cfg.dt), 1.0e-6)
                )
                kinematic_l1 = F.l1_loss(vx_from_pos, x0_raw[:, :, 2])
                kinematic_l1 = kinematic_l1 + F.l1_loss(
                    vy_from_pos,
                    x0_raw[:, :, 3],
                )

            y_l1 = F.l1_loss(pred_traj[:, :, 1], target_traj[:, :, 1])
            x_l1 = F.l1_loss(pred_traj[:, :, 0], target_traj[:, :, 0])
            vy_l1 = F.l1_loss(pred_traj[:, :, 3], target_traj[:, :, 3])
            vx_l1 = F.l1_loss(pred_traj[:, :, 2], target_traj[:, :, 2])
            endpoint_y_l1 = F.l1_loss(pred_traj[:, -1, 1], target_traj[:, -1, 1])
            endpoint_x_l1 = F.l1_loss(pred_traj[:, -1, 0], target_traj[:, -1, 0])

            def _masked_index_l1(index_key: str, mask_key: str) -> torch.Tensor:
                idx = trajectory_meta[index_key].long()
                mask = trajectory_meta[mask_key].to(actions.device).float()
                valid = mask > 0.5
                if not torch.any(valid):
                    return torch.zeros((), dtype=actions.dtype, device=actions.device)
                safe_idx = torch.clamp(idx[valid], 0, actions.shape[1] - 1)
                rows = torch.arange(actions.shape[0], device=actions.device)[valid]
                err = torch.abs(
                    pred_traj[rows, safe_idx, 1]
                    - target_traj[rows, safe_idx, 1]
                )
                return torch.mean(err)

            cross_y_l1 = _masked_index_l1("future_cross_index", "cross_mask")
            end_y_l1 = _masked_index_l1("future_cutin_end_index", "cutin_end_mask")
            loss = (
                loss
                + cfg.trajectory_x_weight * x_l1
                + cfg.trajectory_y_weight * y_l1
                + cfg.trajectory_vx_weight * vx_l1
                + cfg.trajectory_vy_weight * vy_l1
                + cfg.endpoint_x_weight * endpoint_x_l1
                + cfg.endpoint_y_weight * endpoint_y_l1
                + cfg.cross_y_weight * cross_y_l1
                + cfg.end_y_weight * end_y_l1
                + cfg.kinematic_consistency_weight * kinematic_l1
            )
            out.update(
                {
                    "loss": loss,
                    "trajectory_x_l1": x_l1.detach(),
                    "trajectory_y_l1": y_l1.detach(),
                    "trajectory_vx_l1": vx_l1.detach(),
                    "trajectory_vy_l1": vy_l1.detach(),
                    "endpoint_x_l1": endpoint_x_l1.detach(),
                    "endpoint_y_l1": endpoint_y_l1.detach(),
                    "cross_y_l1": cross_y_l1.detach(),
                    "end_y_l1": end_y_l1.detach(),
                    "kinematic_consistency_l1": kinematic_l1.detach(),
                }
            )
        return out

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        history: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.denoiser(
            x_t,
            timesteps,
            history,
            context_features,
            relative_history,
        )
        x0 = self.maybe_clip_x0(self.predict_start_from_noise(x_t, timesteps, eps))
        mean = (
            extract_coeff(self.posterior_mean_coef1, timesteps, x_t.shape) * x0
            + extract_coeff(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        log_var = extract_coeff(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t.shape,
        )
        noise = torch.randn_like(x_t)
        mask = (timesteps != 0).float().reshape(
            x_t.shape[0],
            *((1,) * (x_t.ndim - 1)),
        )
        return mean + mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        history: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.denoiser.cfg
        x = torch.randn(
            batch_size,
            cfg.horizon_steps,
            cfg.action_dim,
            device=history.device,
        )
        for i in reversed(range(self.num_steps)):
            t = torch.full(
                (batch_size,),
                i,
                device=history.device,
                dtype=torch.long,
            )
            x = self.p_sample(x, t, history, context_features, relative_history)
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        history: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        *,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        cfg = self.denoiser.cfg
        steps = (
            self.num_steps
            if inference_steps is None
            else int(inference_steps)
        )
        steps = min(max(steps, 1), self.num_steps)
        raw_steps = np.linspace(0, self.num_steps - 1, steps)
        timesteps = sorted({int(round(step)) for step in raw_steps})
        x = torch.randn(
            batch_size,
            cfg.horizon_steps,
            cfg.action_dim,
            device=history.device,
        )
        for idx in reversed(range(len(timesteps))):
            step = int(timesteps[idx])
            prev_step = int(timesteps[idx - 1]) if idx > 0 else -1
            t = torch.full(
                (batch_size,),
                step,
                device=history.device,
                dtype=torch.long,
            )
            prev_t = torch.full_like(t, prev_step)
            eps = self.denoiser(
                x,
                t,
                history,
                context_features,
                relative_history,
            )
            x = self.ddim_step(x, t, prev_t, eps)
        return x


def build_model_from_schema(schema: dict, config: dict) -> GaussianActionDiffusion:
    model_cfg = config.get("model", {})
    diffusion_cfg = config.get("diffusion", {})
    cfg = ActionDiffusionConfig(
        history_steps=int(schema["history_steps"]),
        num_actors=int(schema["num_actors"]),
        state_features=len(schema["state_features"]),
        context_dim=len(schema["context_keys"]),
        relative_dim=len(schema["relative_history_keys"]),
        horizon_steps=int(schema["horizon_steps"]),
        action_dim=len(schema["action_keys"]),
        dt=float(schema.get("dt", 0.04)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        diffusion_steps=int(diffusion_cfg.get("steps", 100)),
        x0_clip_abs=float(diffusion_cfg.get("x0_clip_abs", 0.0)),
        x0_weight=float(config.get("loss", {}).get("x0_weight", 0.0)),
        smooth_weight=float(config.get("loss", {}).get("smooth_weight", 0.0)),
        trajectory_x_weight=float(config.get("loss", {}).get("trajectory_x_weight", 0.0)),
        trajectory_y_weight=float(config.get("loss", {}).get("trajectory_y_weight", 0.0)),
        trajectory_vx_weight=float(config.get("loss", {}).get("trajectory_vx_weight", 0.0)),
        trajectory_vy_weight=float(config.get("loss", {}).get("trajectory_vy_weight", 0.0)),
        endpoint_x_weight=float(config.get("loss", {}).get("endpoint_x_weight", 0.0)),
        endpoint_y_weight=float(config.get("loss", {}).get("endpoint_y_weight", 0.0)),
        cross_y_weight=float(config.get("loss", {}).get("cross_y_weight", 0.0)),
        end_y_weight=float(config.get("loss", {}).get("end_y_weight", 0.0)),
        kinematic_consistency_weight=float(config.get("loss", {}).get("kinematic_consistency_weight", 0.0)),
        action_representation=str(schema["action_representation"]),
        generation_target=str(schema.get("generation_target", "action")),
    )
    return GaussianActionDiffusion(ActionDenoiser(cfg), diffusion_steps=cfg.diffusion_steps)
