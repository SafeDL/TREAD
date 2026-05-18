"""Fixed guidance schedule for adversarial denoising."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceWeights:
    lambda_rss: float
    lambda_nat: float
    lambda_phy: float
    active: bool


@dataclass(frozen=True)
class GuidanceSchedule:
    enabled: bool = True
    guidance_start_ratio: float = 0.2
    guidance_end_ratio: float = 0.8
    lambda_rss: float = 1.0
    lambda_nat: float = 0.5
    lambda_phy: float = 0.5
    grad_clip_norm: float = 1.0
    update_target: str = "x_t"

    @classmethod
    def from_config(cls, config: dict) -> "GuidanceSchedule":
        cfg = config.get("guided_denoising", config)
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            guidance_start_ratio=float(cfg.get("guidance_start_ratio", 0.2)),
            guidance_end_ratio=float(cfg.get("guidance_end_ratio", 0.8)),
            lambda_rss=float(cfg.get("lambda_rss", 1.0)),
            lambda_nat=float(cfg.get("lambda_nat", 0.5)),
            lambda_phy=float(cfg.get("lambda_phy", 0.5)),
            grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0)),
            update_target=str(cfg.get("update_target", "x_t")),
        )

    def weights_for_timestep(self, timestep: int, num_steps: int) -> GuidanceWeights:
        if not self.enabled or num_steps <= 1:
            return GuidanceWeights(0.0, 0.0, 0.0, False)
        ratio = float(timestep) / float(num_steps - 1)
        active = self.guidance_start_ratio <= ratio <= self.guidance_end_ratio
        if not active:
            return GuidanceWeights(0.0, 0.0, 0.0, False)
        return GuidanceWeights(self.lambda_rss, self.lambda_nat, self.lambda_phy, True)

