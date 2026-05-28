"""Array normalization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, axis=None, eps: float = 1e-6) -> "Normalizer":
        mean = np.mean(x, axis=axis, keepdims=False).astype(np.float32)
        std = np.std(x, axis=axis, keepdims=False).astype(np.float32)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "Normalizer":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )


def fit_dataset_normalizers(
    context_states: np.ndarray,
    context_features: np.ndarray,
    actions: np.ndarray,
    train_mask: np.ndarray,
    relative_history: np.ndarray,
) -> Dict[str, Dict[str, list]]:
    idx = np.asarray(train_mask, dtype=bool)
    if not np.any(idx):
        raise RuntimeError("Cannot fit diffusion normalizers without train split samples")
    state_norm = Normalizer.fit(context_states[idx], axis=(0, 1, 2))
    context_norm = Normalizer.fit(context_features[idx], axis=0)
    action_norm = Normalizer.fit(actions[idx], axis=(0, 1))
    return {
        "context_states": state_norm.to_dict(),
        "context_features": context_norm.to_dict(),
        "actions": action_norm.to_dict(),
        "relative_history": Normalizer.fit(relative_history[idx], axis=(0, 1)).to_dict(),
    }


def apply_normalizers(
    arrays: dict[str, np.ndarray],
    stats: Dict[str, Dict[str, list]],
) -> dict[str, np.ndarray]:
    required_arrays = ("context_states", "context_features", "actions", "relative_history")
    missing_arrays = [key for key in required_arrays if key not in arrays]
    if missing_arrays:
        raise KeyError(f"Diffusion dataset is missing arrays: {missing_arrays}")
    missing_stats = [key for key in required_arrays if key not in stats]
    if missing_stats:
        raise KeyError(f"Diffusion normalization stats are missing keys: {missing_stats}")
    out = dict(arrays)
    out["context_states"] = Normalizer.from_dict(stats["context_states"]).encode(out["context_states"])
    out["context_features"] = Normalizer.from_dict(stats["context_features"]).encode(out["context_features"])
    out["actions"] = Normalizer.from_dict(stats["actions"]).encode(out["actions"])
    out["relative_history"] = Normalizer.from_dict(stats["relative_history"]).encode(out["relative_history"])
    return out
