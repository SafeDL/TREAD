"""Evaluation metrics for guided samples and rolling traces."""
from __future__ import annotations

import numpy as np


def violation_rate(x: np.ndarray, low: float | None = None, high: float | None = None) -> float:
    mask = np.zeros_like(x, dtype=bool)
    if low is not None:
        mask |= x < low
    if high is not None:
        mask |= x > high
    return float(np.mean(mask)) if x.size else 0.0


def summarize_guided_arrays(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "min_rss_margin" in arrays:
        out["min_rss_margin_mean"] = float(np.mean(arrays["min_rss_margin"]))
        out["rss_violation_rate"] = float(np.mean(arrays["min_rss_margin"] < 0.0))
    if "naturalness_score" in arrays:
        out["naturalness_score_mean"] = float(np.mean(arrays["naturalness_score"]))
    if "velocity" in arrays:
        out["negative_speed_rate"] = violation_rate(arrays["velocity"], low=0.0)
    if "acceleration" in arrays:
        out["ax_violation_rate"] = violation_rate(arrays["acceleration"], low=-8.0, high=4.0)
    if "actions" in arrays:
        out["jerk_violation_rate"] = violation_rate(arrays["actions"], low=-12.0, high=12.0)
        if arrays["actions"].shape[0] > 1:
            flat = arrays["actions"].reshape(arrays["actions"].shape[0], -1)
            out["sample_diversity_l2"] = float(np.mean(np.std(flat, axis=0)))
    return out

