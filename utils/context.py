"""Shared context helpers for rollout scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_npz


CONTEXT_META_KEYS = (
    "recording_id",
    "event_id",
    "anchor_frame",
    "source_type",
    "event_steps",
    "y_long",
    "risk_score",
    "evt_tail_probability",
    "evt_return_level_target",
    "evt_failure_threshold",
)


def load_context_npz(path: str | Path) -> dict[str, np.ndarray]:
    return load_npz(path)


def context_from_npz(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    required = ("context_states", "ego_length", "adv_length")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(
            f"Context dataset is missing required arrays: {missing}"
        )
    context: dict[str, Any] = {
        "raw_context_states": raw["context_states"][idx],
        "ego_length": float(raw["ego_length"][idx]),
        "adv_length": float(raw["adv_length"][idx]),
    }
    for key in CONTEXT_META_KEYS:
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    if "risk_score" not in context and "criticality_score" in raw:
        value = raw["criticality_score"][idx]
        context["risk_score"] = value.item() if hasattr(value, "item") else value
    return context
