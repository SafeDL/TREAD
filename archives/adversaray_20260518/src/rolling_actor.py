"""Closed-loop per-frame rolling guided diffusion actor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from .guided_sampler import GuidedDiffusionSampler


class Observation(Protocol):
    context_states: np.ndarray
    context_features: np.ndarray
    relative_history: np.ndarray
    ego_length: float
    adv_length: float


@dataclass
class RollingActorState:
    current_plan_actions: np.ndarray | None = None
    plan_cursor: int = 0
    steps_since_replan: int = 0
    replan_count: int = 0
    reuse_lengths: list[int] = field(default_factory=list)


class RollingGuidedDiffusionActor:
    """Receding-horizon actor that executes one lead action per simulation tick."""

    def __init__(self, sampler: GuidedDiffusionSampler, config: dict[str, Any]) -> None:
        self.sampler = sampler
        cfg = config.get("rolling_actor", config)
        self.plan_horizon_steps = int(cfg.get("plan_horizon_steps", 50))
        self.commit_steps_max = int(cfg.get("commit_steps_max", 5))
        self.step_mode = str(cfg.get("step_mode", "closed_loop_per_frame"))
        if self.step_mode != "closed_loop_per_frame":
            raise ValueError(f"Unsupported rolling_actor.step_mode={self.step_mode!r}")
        self.replan_on_invalid_plan = bool(cfg.get("replan_on_invalid_plan", True))
        self.state = RollingActorState()
        self.trace: list[dict[str, float]] = []

    def reset(self) -> None:
        self.state = RollingActorState()
        self.trace.clear()

    def _needs_replan(self, observation: Observation | dict[str, Any]) -> bool:
        plan = self.state.current_plan_actions
        if plan is None:
            return True
        if self.state.plan_cursor >= len(plan):
            return True
        if self.state.steps_since_replan >= self.commit_steps_max:
            return True
        if self.replan_on_invalid_plan and not np.isfinite(plan).all():
            return True
        return False

    @staticmethod
    def _get(observation: Observation | dict[str, Any], key: str, default: Any = None) -> Any:
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    def _make_plan(self, observation: Observation | dict[str, Any]) -> None:
        context_states = torch.from_numpy(np.asarray(self._get(observation, "context_states"), dtype=np.float32)[None])
        context_features = torch.from_numpy(np.asarray(self._get(observation, "context_features"), dtype=np.float32)[None])
        relative_history = torch.from_numpy(np.asarray(self._get(observation, "relative_history"), dtype=np.float32)[None])
        ego_length = torch.tensor([float(self._get(observation, "ego_length", 4.8))], dtype=torch.float32)
        adv_length = torch.tensor([float(self._get(observation, "adv_length", 4.8))], dtype=torch.float32)
        result = self.sampler.sample(
            context_states,
            context_features,
            relative_history,
            ego_length=ego_length,
            adv_length=adv_length,
            num_samples=1,
        )
        self.state.current_plan_actions = result.raw_actions[0].detach().cpu().numpy().astype(np.float32)
        self.state.plan_cursor = 0
        self.state.steps_since_replan = 0
        self.state.replan_count += 1
        self.trace.append(
            {
                "event": 1.0,
                "replan_count": float(self.state.replan_count),
                "plan_min_rss_margin": float(result.diagnostics["min_rss_margin"][0].detach().cpu()),
                "plan_naturalness_score": float(result.diagnostics["naturalness_score"][0].detach().cpu()),
            }
        )

    def step(self, observation: Observation | dict[str, Any]) -> np.ndarray:
        """Return the current frame's lead action; caller then steps ego/ADS in the same tick."""
        if self._needs_replan(observation):
            if self.state.steps_since_replan > 0:
                self.state.reuse_lengths.append(int(self.state.steps_since_replan))
            self._make_plan(observation)
        assert self.state.current_plan_actions is not None
        action = self.state.current_plan_actions[self.state.plan_cursor].copy()
        self.state.plan_cursor += 1
        self.state.steps_since_replan += 1
        if self.state.steps_since_replan > self.commit_steps_max:
            raise RuntimeError("Rolling actor reused a plan beyond commit_steps_max")
        self.trace.append(
            {
                "event": 0.0,
                "plan_cursor": float(self.state.plan_cursor),
                "steps_since_replan": float(self.state.steps_since_replan),
            }
        )
        return action

    def summary(self) -> dict[str, Any]:
        reuse = list(self.state.reuse_lengths)
        if self.state.steps_since_replan > 0:
            reuse.append(int(self.state.steps_since_replan))
        return {
            "step_mode": self.step_mode,
            "commit_steps_max": self.commit_steps_max,
            "replan_count": int(self.state.replan_count),
            "reuse_lengths": reuse,
            "max_reuse_length": int(max(reuse) if reuse else 0),
            "trace": self.trace,
        }
