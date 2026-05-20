"""Ego-response surrogate utilities for Stage 1 proposal generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class IDMSurrogateParams:
    desired_speed: torch.Tensor
    desired_headway: torch.Tensor
    min_gap: torch.Tensor
    max_accel: torch.Tensor
    comfortable_brake: torch.Tensor
    response_time: torch.Tensor
    delta: torch.Tensor

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "IDMSurrogateParams":
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return IDMSurrogateParams(**{key: value.to(**kwargs) for key, value in self.as_dict().items()})

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "desired_speed": self.desired_speed,
            "desired_headway": self.desired_headway,
            "min_gap": self.min_gap,
            "max_accel": self.max_accel,
            "comfortable_brake": self.comfortable_brake,
            "response_time": self.response_time,
            "delta": self.delta,
        }

    def flatten(self) -> "IDMSurrogateParams":
        return IDMSurrogateParams(**{key: value.reshape(-1) for key, value in self.as_dict().items()})

    def repeat_interleave(self, repeats: int, dim: int = 0) -> "IDMSurrogateParams":
        return IDMSurrogateParams(
            **{key: value.repeat_interleave(int(repeats), dim=dim) for key, value in self.as_dict().items()}
        )

    def to_feature_tensor(self) -> torch.Tensor:
        return torch.stack(
            [
                self.desired_speed,
                self.desired_headway,
                self.min_gap,
                self.max_accel,
                self.comfortable_brake,
                self.response_time,
                self.delta,
            ],
            dim=-1,
        )


def _range(cfg: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = cfg.get(key, default)
    if isinstance(value, (int, float)):
        item = float(value)
        return item, item
    if len(value) != 2:
        raise ValueError(f"{key} must be a scalar or [lo, hi] range")
    lo = float(value[0])
    hi = float(value[1])
    if hi < lo:
        raise ValueError(f"{key} has hi < lo: {value}")
    return lo, hi


def _uniform(
    shape: tuple[int, ...],
    value_range: tuple[float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    lo, hi = value_range
    if hi == lo:
        return torch.full(shape, lo, device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lo, hi, generator=generator)


def ego_surrogate_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if "ego_surrogate" in config:
        return dict(config.get("ego_surrogate", {}))
    return dict(config.get("stage1_shared", {}).get("ego_surrogate", {}))


def sample_idm_surrogate_params(
    config: dict[str, Any] | None,
    *,
    batch_size: int,
    num_samples: int = 1,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    flatten: bool = True,
) -> IDMSurrogateParams:
    cfg = ego_surrogate_config(config)
    shape = (int(batch_size), int(num_samples))
    params = IDMSurrogateParams(
        desired_speed=_uniform(shape, _range(cfg, "desired_speed_range", (20.0, 35.0)), device=device, dtype=dtype, generator=generator),
        desired_headway=_uniform(shape, _range(cfg, "desired_headway_range", (0.6, 2.5)), device=device, dtype=dtype, generator=generator),
        min_gap=_uniform(shape, _range(cfg, "min_gap_range", (1.0, 8.0)), device=device, dtype=dtype, generator=generator),
        max_accel=_uniform(shape, _range(cfg, "max_accel_range", (0.8, 3.0)), device=device, dtype=dtype, generator=generator),
        comfortable_brake=_uniform(
            shape,
            _range(cfg, "comfortable_brake_range", (2.0, 8.0)),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        response_time=_uniform(shape, _range(cfg, "response_time_range", (0.0, 1.2)), device=device, dtype=dtype, generator=generator),
        delta=_uniform(shape, _range(cfg, "delta_range", (4.0, 4.0)), device=device, dtype=dtype, generator=generator),
    )
    return params.flatten() if flatten else params


def idm_params_from_tensor(values: torch.Tensor) -> IDMSurrogateParams:
    if values.shape[-1] != 7:
        raise ValueError(f"Expected IDM feature tensor last dim 7, got {tuple(values.shape)}")
    return IDMSurrogateParams(
        desired_speed=values[..., 0],
        desired_headway=values[..., 1],
        min_gap=values[..., 2],
        max_accel=values[..., 3],
        comfortable_brake=values[..., 4],
        response_time=values[..., 5],
        delta=values[..., 6],
    )
