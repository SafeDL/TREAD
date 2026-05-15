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
    risk: np.ndarray,
    train_mask: np.ndarray,
) -> Dict[str, Dict[str, list]]:
    idx = np.asarray(train_mask, dtype=bool)
    if not np.any(idx):
        idx = np.ones((context_states.shape[0],), dtype=bool)
    state_norm = Normalizer.fit(context_states[idx], axis=(0, 1, 2))
    context_norm = Normalizer.fit(context_features[idx], axis=0)
    action_norm = Normalizer.fit(actions[idx], axis=(0, 1))
    risk_norm = Normalizer.fit(risk[idx].reshape(-1, 1), axis=0)
    return {
        "context_states": state_norm.to_dict(),
        "context_features": context_norm.to_dict(),
        "actions": action_norm.to_dict(),
        "risk": risk_norm.to_dict(),
    }


def apply_normalizers(
    arrays: dict[str, np.ndarray],
    stats: Dict[str, Dict[str, list]],
) -> dict[str, np.ndarray]:
    out = dict(arrays)
    out["context_states"] = Normalizer.from_dict(stats["context_states"]).encode(out["context_states"])
    out["context_features"] = Normalizer.from_dict(stats["context_features"]).encode(out["context_features"])
    out["actions"] = Normalizer.from_dict(stats["actions"]).encode(out["actions"])
    risk_norm = Normalizer.from_dict(stats["risk"])
    out["risk"] = risk_norm.encode(out["risk"].reshape(-1, 1)).reshape(-1).astype(np.float32)
    return out

