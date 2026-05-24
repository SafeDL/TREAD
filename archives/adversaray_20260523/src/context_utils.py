"""Shared context helpers for frozen-prior/KING rollouts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
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
    for key in (
        "recording_id",
        "event_id",
        "anchor_frame",
        "source_type",
        "anchor_dataset_index",
        "dataset_index",
        "event_steps",
        "target_gap",
        "target_ttc",
        "target_rss_margin",
        "criticality_score",
    ):
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    return context
