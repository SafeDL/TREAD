"""
inference.py — DeepEVT 推理与 tail_conditions 导出
===================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from tread_highd.src.io_utils import load_json

from .data import DatasetArrays, apply_normalization, load_dataset
from .losses import expected_shortfall_np, tail_quantile_np
from .model import DeepEVTConfig, DeepEVTModel

logger = logging.getLogger(__name__)


@dataclass
class DeepEVTPredictions:
    u: np.ndarray
    p: np.ndarray
    xi: np.ndarray
    beta: np.ndarray


def load_model(checkpoint_path: str | Path, device: Optional[str] = None) -> DeepEVTModel:
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg_dict = ckpt["model_cfg"]
    cfg = DeepEVTConfig(**cfg_dict)
    model = DeepEVTModel(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model._alpha_u = float(ckpt.get("alpha_u", 0.9))  # stash for convenience
    return model


def predict(
    model: DeepEVTModel,
    arrays: DatasetArrays,
    batch_size: int = 512,
) -> DeepEVTPredictions:
    device = next(model.parameters()).device
    prefix = torch.from_numpy(arrays.prefix_states).float()
    ctx = torch.from_numpy(arrays.context_features).float()

    us, ps, xis, betas = [], [], [], []
    alpha_u = float(getattr(model, "_alpha_u", 0.9))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(arrays.risk_score), batch_size):
            p_b = prefix[i:i + batch_size].to(device)
            c_b = ctx[i:i + batch_size].to(device)
            out = model(p_b, c_b)
            us.append(out["u"].cpu().numpy())
            if "p" in out:
                ps.append(out["p"].cpu().numpy())
            else:
                ps.append(np.full(len(p_b), 1.0 - alpha_u, dtype=np.float32))
            xis.append(out["xi"].cpu().numpy())
            betas.append(out["beta"].cpu().numpy())

    return DeepEVTPredictions(
        u=np.concatenate(us), p=np.concatenate(ps),
        xi=np.concatenate(xis), beta=np.concatenate(betas),
    )


def export_tail_conditions(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    tail_levels=(0.90, 0.95, 0.99),
    include_context_features: bool = True,
) -> pd.DataFrame:
    """导出 tail_conditions.csv 给 diffusion 使用。"""
    out = Path(output_dir)
    schema = load_json(out / "feature_schema.json")
    norm_stats = load_json(out / "normalization_stats.json")
    arrays = load_dataset(out)

    model = load_model(checkpoint_path)
    preds = predict(model, apply_normalization(arrays, norm_stats))

    split_map = {0: "train", 1: "val", 2: "test"}
    split_names = np.array([split_map[int(s)] for s in arrays.split_index])

    data = {
        "event_id": arrays.event_id,
        "event_type": np.array([schema["event_type"]] * len(arrays.event_id)),
        "recording_id": arrays.recording_id,
        "split": split_names,
        "risk_score": arrays.risk_score,
        "u_pred": preds.u,
        "p_exceed_pred": preds.p,
        "xi_pred": preds.xi,
        "beta_pred": preds.beta,
    }

    for tau in tail_levels:
        q = tail_quantile_np(preds.u, preds.p, preds.xi, preds.beta, float(tau))
        data[f"q{int(tau * 100)}_pred"] = q

    # ES at 0.95 and 0.99 by default
    for tau in (0.95, 0.99):
        es = expected_shortfall_np(preds.u, preds.p, preds.xi, preds.beta, float(tau))
        data[f"es{int(tau * 100)}_pred"] = es

    # tail labels from empirical per-split quantile (train only)
    train_mask = arrays.split_index == 0
    train_risk = arrays.risk_score[train_mask]
    for tau in tail_levels:
        thr = float(np.quantile(train_risk, tau)) if train_risk.size else float("nan")
        data[f"tail_label_{int(tau * 100)}"] = (arrays.risk_score > thr).astype(np.int8)
        data[f"empirical_risk_percentile_vs_train_{int(tau * 100)}"] = (
            (arrays.risk_score > thr).astype(np.int8)
        )

    if include_context_features:
        for i, key in enumerate(schema["context_keys"]):
            data[f"context_{key}"] = arrays.context_features[:, i]

    df = pd.DataFrame(data)
    df.to_csv(out / "tail_conditions.csv", index=False)
    logger.info("Wrote %s  (N=%d)", out / "tail_conditions.csv", len(df))
    return df
