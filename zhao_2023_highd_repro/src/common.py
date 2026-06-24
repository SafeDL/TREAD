from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPRO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REPRO_ROOT.parent


def ensure_repo_imports() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_path(value: str | Path, *, base: Path = REPRO_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_path(path or "config/default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def output_dir(config: dict[str, Any]) -> Path:
    path = resolve_path(config.get("paths", {}).get("output_dir", "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def finite_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    p05, p50, p95 = np.quantile(arr, [0.05, 0.5, 0.95])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }
