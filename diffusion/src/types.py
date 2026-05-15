"""Shared dataclasses and constants for action diffusion."""
from __future__ import annotations

from dataclasses import dataclass
from enum import strEnum
from typing import Dict, Tuple

import numpy as np


class EventType(strEnum):
    FOLLOWING = "following"
    CUT_IN = "cut_in"


STATE_FEATURES: Tuple[str, ...] = ("x", "y", "vx", "vy", "ax", "ay")
NUM_ACTORS = 2
NUM_STATE_FEATURES = len(STATE_FEATURES)

FOLLOWING_ACTION_KEYS: Tuple[str, ...] = ("ax",)
CUTIN_ACTION_KEYS: Tuple[str, ...] = ("ax", "yaw_rate")


@dataclass(frozen=True)
class VehicleBox:
    length: float = 4.8
    width: float = 1.8


@dataclass
class VehicleState:
    """Minimal closed-loop vehicle state in the local simulation frame."""

    x: float
    y: float
    vx: float
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    yaw: float = 0.0
    box: VehicleBox = VehicleBox()
    lane_id: int | None = None

    def as_feature(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.vx, self.vy, self.ax, self.ay], dtype=np.float32)


@dataclass
class ClosedLoopContext:
    """Current ADS-centred interaction state for rolling generation.

    ``history`` is in ego-current coordinates with shape
    ``[history_steps, actors, state_features]`` and actor order
    ``0=ego/ADS, 1=adversarial vehicle``.
    """

    history: np.ndarray
    ego: VehicleState
    adversary: VehicleState
    event_type: EventType | str
    lane_width: float = 3.75
    target_final_y: float | None = None
    extra: Dict[str, float] | None = None


@dataclass
class CandidatePlan:
    actions: np.ndarray
    trajectory: np.ndarray
    risk: float
    naturalness_cost: float
    violation_cost: float
    score: float
    is_valid: bool

