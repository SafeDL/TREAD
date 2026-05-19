"""
inference.py — direct q85/q90/q95 inference and tail-condition export.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from process_highd.src.io_utils import load_json

from .data import DatasetArrays, apply_normalization, load_dataset
from .model import DeepEVTConfig, DeepEVTModel

logger = logging.getLogger(__name__)


@dataclass
class DeepEVTPredictions:
    u: np.ndarray
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]


def load_model(checkpoint_path: str | Path, device: Optional[str] = None) -> DeepEVTModel:
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg_dict = dict(ckpt["model_cfg"])
    for stale_key in (
        "xi_min", "xi_max", "beta_min", "use_exceedance_head",
        "use_tail_distribution_heads", "use_direct_quantile_head",
        "encoder_type", "context_hidden_dim", "use_tail_quantile_heads",
        "tail_quantile_levels", "tail_quantile_min_increment",
        "tail_quantile_initial_increment",
    ):
        cfg_dict.pop(stale_key, None)
    cfg = DeepEVTConfig(**cfg_dict)
    model = DeepEVTModel(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    model._alpha_u = float(ckpt.get("alpha_u", 0.85))
    return model


def predict(
    model: DeepEVTModel,
    arrays: DatasetArrays,
    batch_size: int = 512,
) -> DeepEVTPredictions:
    if len(arrays.risk_score) == 0:
        return DeepEVTPredictions(
            u=np.empty(0, dtype=np.float32),
            quantiles=np.empty((0, 0), dtype=np.float32),
            quantile_levels=tuple(float(x) for x in getattr(model.cfg, "quantile_levels", ())),
        )

    device = next(model.parameters()).device
    prefix = torch.from_numpy(arrays.prefix_states).float()
    ctx = torch.from_numpy(arrays.context_features).float()
    us, quantiles = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(arrays.risk_score), batch_size):
            out = model(prefix[i:i + batch_size].to(device), ctx[i:i + batch_size].to(device))
            us.append(out["u"].cpu().numpy())
            quantiles.append(out["quantiles"].cpu().numpy())
    return DeepEVTPredictions(
        u=np.concatenate(us),
        quantiles=np.concatenate(quantiles, axis=0),
        quantile_levels=tuple(float(x) for x in getattr(model.cfg, "quantile_levels", ())),
    )


def _q_by_level(preds: DeepEVTPredictions, levels: Iterable[float]) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {}
    pred_levels = tuple(float(x) for x in preds.quantile_levels)
    for tau in levels:
        tau_f = float(tau)
        idx = min(range(len(pred_levels)), key=lambda i: abs(pred_levels[i] - tau_f))
        if abs(pred_levels[idx] - tau_f) > 1e-6:
            raise ValueError(f"Model does not provide q{int(tau_f * 100)}")
        out[tau_f] = preds.quantiles[:, idx]
    return out


def export_tail_conditions(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    tail_levels=(0.85, 0.90, 0.95),
    include_context_features: bool = False,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Export raw direct quantiles for diffusion / MATLAB / RoadRunner."""
    del config
    out = Path(output_dir)
    schema = load_json(out / "feature_schema.json")
    norm_stats = load_json(out / "normalization_stats.json")
    arrays = load_dataset(out)
    norm_arrays = apply_normalization(arrays, norm_stats)

    model = load_model(checkpoint_path)
    preds = predict(model, norm_arrays)
    levels = tuple(float(x) for x in tail_levels)
    q_raw = _q_by_level(preds, levels)
    q_export = q_raw
    source = "direct_quantile"

    split_map = {0: "train", 1: "val", 2: "test"}
    split_names = np.array([split_map[int(s)] for s in arrays.split_index])
    data = {
        "event_id": arrays.event_id,
        "event_type": np.array([schema["event_type"]] * len(arrays.event_id)),
        "recording_id": arrays.recording_id,
        "split": split_names,
        "prefix_start_frame": arrays.prefix_start_frame,
        "prefix_end_frame": arrays.prefix_end_frame,
        "risk_window_start_frame": arrays.risk_window_start_frame,
        "risk_window_end_frame": arrays.risk_window_end_frame,
        "risk_score": arrays.risk_score,
        "u_pred": q_export[levels[0]],
        "scenario_frame": np.array([schema.get("scenario_frame", "ego_current")] * len(arrays.event_id)),
        "ego_origin_x": arrays.ego_origin_x,
        "ego_origin_y": arrays.ego_origin_y,
        "ego_rot_cos": arrays.ego_rot_cos,
        "ego_rot_sin": arrays.ego_rot_sin,
        "ego_length": arrays.ego_length,
        "target_length": arrays.target_length,
    }

    for tau in levels:
        label = int(round(tau * 100))
        data[f"q{label}_raw_pred"] = q_raw[tau]
        data[f"q{label}_pred"] = q_export[tau]
        data[f"q{label}_invalid_mask"] = np.zeros(len(arrays.risk_score), dtype=np.int8)
        data[f"q{label}_source"] = np.array([source] * len(arrays.risk_score))

    if include_context_features:
        for i, key in enumerate(schema["context_keys"]):
            data[f"context_{key}"] = arrays.context_features[:, i]

    df = pd.DataFrame(data)
    csv_path = out / "tail_conditions.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s (%d rows)", csv_path, len(df))
    return df
