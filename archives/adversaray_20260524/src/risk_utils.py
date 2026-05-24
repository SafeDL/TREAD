"""Shared risk, tail-scoring, and Stage 1 action utilities."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .rss import RSSConfig


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def rss_safe_distance_np(
    ego_velocity: np.ndarray,
    lead_velocity: np.ndarray,
    cfg: RSSConfig,
) -> np.ndarray:
    ego_velocity = np.asarray(ego_velocity, dtype=np.float64)
    lead_velocity = np.asarray(lead_velocity, dtype=np.float64)
    rho = float(cfg.response_time)
    ego_after_response = ego_velocity + rho * float(cfg.ego_max_accel)
    ego_distance = (
        ego_velocity * rho
        + 0.5 * float(cfg.ego_max_accel) * rho * rho
    )
    ego_brake_distance = np.square(ego_after_response) / max(
        2.0 * float(cfg.ego_min_brake),
        1e-6,
    )
    lead_brake_distance = np.square(lead_velocity) / max(
        2.0 * float(cfg.lead_max_brake),
        1e-6,
    )
    return np.maximum(
        ego_distance + ego_brake_distance - lead_brake_distance,
        0.0,
    )


def interaction_metrics_from_states(
    context_states: np.ndarray,
    future_states: np.ndarray,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    rss_cfg: RSSConfig,
) -> dict[str, np.ndarray]:
    future = np.asarray(future_states, dtype=np.float32)
    ego = future[:, :, 0]
    lead = future[:, :, 1]
    ego_length = np.asarray(ego_length, dtype=np.float32).reshape(-1)
    adv_length = np.asarray(adv_length, dtype=np.float32).reshape(-1)
    gap = (
        lead[:, :, 0]
        - ego[:, :, 0]
        - 0.5 * (ego_length[:, None] + adv_length[:, None])
    )
    closing = ego[:, :, 2] - lead[:, :, 2]
    ttc = np.where(closing > 1e-6, gap / np.maximum(closing, 1e-6), 1000.0)
    safe = rss_safe_distance_np(
        np.maximum(ego[:, :, 2], 0.0),
        np.maximum(lead[:, :, 2], 0.0),
        rss_cfg,
    )
    rss_margin = gap - safe
    initial_context = np.asarray(context_states, dtype=np.float32)
    initial_ego = initial_context[:, -1, 0]
    initial_lead = initial_context[:, -1, 1]
    initial_gap = (
        initial_lead[:, 0]
        - initial_ego[:, 0]
        - 0.5 * (ego_length + adv_length)
    )
    initial_closing_speed = initial_ego[:, 2] - initial_lead[:, 2]
    initial_safe = rss_safe_distance_np(
        np.maximum(initial_ego[:, 2], 0.0),
        np.maximum(initial_lead[:, 2], 0.0),
        rss_cfg,
    )
    return {
        "gap": gap.astype(np.float32),
        "ttc": np.clip(ttc, 0.0, 1000.0).astype(np.float32),
        "rss_margin": rss_margin.astype(np.float32),
        "min_gap": np.min(gap, axis=1).astype(np.float32),
        "min_ttc": np.min(
            np.clip(ttc, 0.0, 1000.0),
            axis=1,
        ).astype(np.float32),
        "min_rss_margin": np.min(rss_margin, axis=1).astype(np.float32),
        "initial_gap": initial_gap.astype(np.float32),
        "initial_closing_speed": initial_closing_speed.astype(np.float32),
        "initial_rss_margin": (initial_gap - initial_safe).astype(np.float32),
    }
