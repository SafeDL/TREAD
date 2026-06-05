"""Evaluate latent MPC episodes with highway-env closed-loop rollouts."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import torch

from subset.src.closed_loop_runner import ClosedLoopFollowingRunner
from subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler


@dataclass
class LatentEvaluation:
    score: float
    actions: np.ndarray
    metrics: dict[str, float]
    trace: list[dict[str, float]]
    context_index: int


class LatentMpcEpisodeEvaluator:
    """Evaluate one context index and one latent sequence as an MPC episode."""

    def __init__(
        self,
        sampler: FrozenDiffusionSampler,
        runner: ClosedLoopFollowingRunner,
        contexts: list[dict[str, Any]],
        config: dict[str, Any],
        *,
        inference_steps: int | None = None,
    ) -> None:
        if not contexts:
            raise ValueError("contexts must not be empty")
        self.sampler = sampler
        self.runner = runner
        self.contexts = list(contexts)
        self.config = config
        self.inference_steps = inference_steps
        env_cfg = config.get("env", {})
        self.episode_steps = int(env_cfg.get("episode_steps", 200))
        if self.episode_steps <= 0:
            raise ValueError("env.episode_steps must be positive")
        self.commit_steps = int(env_cfg.get("commit_steps_max", 10))
        if self.commit_steps <= 0:
            raise ValueError("env.commit_steps_max must be positive")
        self.num_plans = int(ceil(self.episode_steps / self.commit_steps))

    @property
    def context_count(self) -> int:
        return len(self.contexts)

    @property
    def plan_latent_shape(self) -> tuple[int, int]:
        cfg = self.sampler.prior.model.denoiser.cfg
        return int(cfg.horizon_steps), int(cfg.action_dim)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        horizon, action_dim = self.plan_latent_shape
        return self.num_plans, horizon, action_dim

    def decode_plan(
        self,
        obs: dict[str, np.ndarray],
        latent: np.ndarray,
    ) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32)
        if latent.shape != self.plan_latent_shape:
            raise ValueError(
                f"Expected plan latent shape {self.plan_latent_shape}, "
                f"got {latent.shape}"
            )
        with torch.no_grad():
            sample = self.sampler.sample_from_noise(
                torch.from_numpy(obs["context_states"][None]).float(),
                torch.from_numpy(obs["context_features"][None]).float(),
                torch.from_numpy(obs["relative_history"][None]).float(),
                torch.from_numpy(latent[None]).float(),
                inference_steps=self.inference_steps,
            )
        return sample.raw_actions[0].detach().cpu().numpy().astype(np.float32)

    def evaluate(
        self,
        context_index: int,
        z: np.ndarray,
    ) -> LatentEvaluation:
        if context_index < 0 or context_index >= len(self.contexts):
            raise IndexError(f"context_index out of range: {context_index}")
        latent = np.asarray(z, dtype=np.float32)
        if latent.shape != self.latent_shape:
            raise ValueError(
                f"Expected latent shape {self.latent_shape}, "
                f"got {latent.shape}"
            )
        context = self.contexts[int(context_index)]
        def plan_callback(
            obs: dict[str, np.ndarray],
            plan_idx: int,
            step: int,
        ) -> dict[str, Any]:
            if plan_idx >= latent.shape[0]:
                raise RuntimeError(
                    "Latent sequence exhausted before episode finished"
                )
            plan = self.decode_plan(obs, latent[plan_idx])
            return {
                "plan": plan,
                "prior_plan": plan,
                "summary": {
                    "plan_idx": float(plan_idx),
                    "step": float(step),
                    "latent_l2": float(np.linalg.norm(latent[plan_idx])),
                    "plan_mean": float(np.mean(plan)),
                    "plan_std": float(np.std(plan)),
                },
            }

        result = self.runner.rollout(
            context,
            plan_callback=plan_callback,
            episode_steps=self.episode_steps,
        )
        if result.actions is None:
            raise RuntimeError("Rolling rollout did not return actions")
        metrics = dict(result.metrics)
        metrics.update(
            {
                "context_index": float(context_index),
                "recording_id": float(context.get("recording_id", -1)),
                "event_steps_config": float(self.episode_steps),
                "executed_steps": float(len(result.trace)),
            }
        )
        return LatentEvaluation(
            score=float(result.risk_score),
            actions=result.actions.astype(np.float32),
            metrics=metrics,
            trace=list(result.trace),
            context_index=int(context_index),
        )
