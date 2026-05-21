"""Context helpers for Stage 1 prior sampling."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .closed_loop_runner import ClosedLoopFollowingRunner


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    required = ("context_states", "ego_length", "adv_length")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Context dataset is missing required arrays: {missing}")
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
        "target_gap",
        "target_ttc",
        "target_rss_margin",
        "criticality_score",
    ):
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    return context


def _batch_observation_for_contexts(
    runner: ClosedLoopFollowingRunner,
    contexts: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    observations: list[dict[str, np.ndarray]] = []
    prepared_contexts: list[dict[str, Any]] = []
    ego_lengths: list[float] = []
    adv_lengths: list[float] = []
    for ctx in contexts:
        raw_context = np.asarray(ctx["raw_context_states"], dtype=np.float32).copy()
        raw_context[:, :, 1] = 0.0
        if "ego_length" not in ctx or "adv_length" not in ctx:
            raise KeyError("Prepared context must contain ego_length and adv_length")
        ego_length = float(ctx["ego_length"])
        lead_length = float(ctx["adv_length"])
        rebuilt = runner._maybe_reconstruct_highd_context(ctx, ego_length, lead_length)
        if rebuilt is not None:
            raw_context, ego_length, lead_length = rebuilt
            raw_context[:, :, 1] = 0.0
        initial_gap = float(raw_context[-1, 1, 0] - raw_context[-1, 0, 0] - 0.5 * (ego_length + lead_length))
        if initial_gap <= runner.initial_gap_min and not runner.skip_invalid_initial_context:
            raw_context[-1, 1, 0] = raw_context[-1, 0, 0] + 0.5 * (ego_length + lead_length) + runner.initial_gap_min
        history_world: deque[np.ndarray] = deque(maxlen=runner.history_steps)
        for item in raw_context[-runner.history_steps :]:
            v = np.asarray(item, dtype=np.float32).copy()
            v[:, 1] = 0.0
            history_world.append(v)
        observations.append(runner._build_observation(history_world, ego_length, lead_length))
        prepared = dict(ctx)
        prepared["raw_context_states"] = raw_context
        prepared["ego_length"] = ego_length
        prepared["adv_length"] = lead_length
        prepared_contexts.append(prepared)
        ego_lengths.append(ego_length)
        adv_lengths.append(lead_length)
    batch = {
        "context_states": torch.from_numpy(np.stack([obs["context_states"] for obs in observations], axis=0)).float(),
        "context_features": torch.from_numpy(np.stack([obs["context_features"] for obs in observations], axis=0)).float(),
        "relative_history": torch.from_numpy(np.stack([obs["relative_history"] for obs in observations], axis=0)).float(),
        "ego_length": torch.tensor(ego_lengths, dtype=torch.float32),
        "adv_length": torch.tensor(adv_lengths, dtype=torch.float32),
    }
    return batch, prepared_contexts
