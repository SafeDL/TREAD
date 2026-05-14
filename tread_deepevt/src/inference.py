"""
inference.py — DeepEVT 推理与 tail_conditions 导出
===================================================

``tail_conditions.csv`` 是给 diffusion / MATLAB 共用的契约文件。除
DeepEVT 预测的尾部分位/ES 外，每行还携带:

* ego-current frame metadata (origin_x/origin_y/rot_cos/rot_sin)
* CanonicalScenarioContext 全部字段 (canonical_*)
* DeepEVT context_features (context_*)

只要下游模块解析这个 CSV，就能保证 DeepEVT.context、Diffusion.condition
与 MATLAB.scenario_init 完全对齐。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from tread_highd.src.io_utils import load_json

from .data import DatasetArrays, apply_normalization, load_dataset
from .losses import (
    expected_shortfall_np,
    tail_quantile_invalid_mask,
    tail_quantile_np,
)
from .model import DeepEVTConfig, DeepEVTModel
from .scenario_frame import SCENARIO_CONTEXT_SCHEMA_VERSION

logger = logging.getLogger(__name__)


@dataclass
class DeepEVTPredictions:
    u: np.ndarray
    p: np.ndarray
    xi: np.ndarray
    beta: np.ndarray
    u_scale: np.ndarray
    xi_scale: np.ndarray
    beta_scale: np.ndarray


def load_model(checkpoint_path: str | Path, device: Optional[str] = None) -> DeepEVTModel:
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg_dict = ckpt["model_cfg"]
    cfg_dict.pop("encoder_type", None)
    cfg_dict.pop("context_hidden_dim", None)
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
    if len(arrays.risk_score) == 0:
        empty = np.empty(0, dtype=np.float32)
        return DeepEVTPredictions(
            u=empty, p=empty, xi=empty, beta=empty,
            u_scale=empty, xi_scale=empty, beta_scale=empty,
        )

    device = next(model.parameters()).device
    prefix = torch.from_numpy(arrays.prefix_states).float()
    ctx = torch.from_numpy(arrays.context_features).float()

    us, ps, xis, betas = [], [], [], []
    u_scales, xi_scales, beta_scales = [], [], []
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
            u_scales.append(torch.exp(out["u_log_scale"]).cpu().numpy())
            xi_scales.append(torch.exp(out["xi_log_scale"]).cpu().numpy())
            beta_scales.append(torch.exp(out["beta_log_scale"]).cpu().numpy())

    return DeepEVTPredictions(
        u=np.concatenate(us), p=np.concatenate(ps),
        xi=np.concatenate(xis), beta=np.concatenate(betas),
        u_scale=np.concatenate(u_scales),
        xi_scale=np.concatenate(xi_scales),
        beta_scale=np.concatenate(beta_scales),
    )


def export_tail_conditions(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    tail_levels=(0.90, 0.95),
    include_context_features: bool = True,
) -> pd.DataFrame:
    """导出 tail_conditions.csv 给 diffusion / MATLAB 使用。"""
    out = Path(output_dir)
    schema = load_json(out / "feature_schema.json")
    norm_stats = load_json(out / "normalization_stats.json")
    canonical_payload = load_json(out / "canonical_contexts.json")
    arrays = load_dataset(out)

    if canonical_payload.get("schema_version") != SCENARIO_CONTEXT_SCHEMA_VERSION:
        logger.warning(
            "canonical_contexts.json schema version %s != current %s; "
            "downstream consumers should re-check field compatibility.",
            canonical_payload.get("schema_version"), SCENARIO_CONTEXT_SCHEMA_VERSION,
        )
    canonical_records = {c["event_id"]: c for c in canonical_payload.get("contexts", [])}

    model = load_model(checkpoint_path)
    preds = predict(model, apply_normalization(arrays, norm_stats))

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
        "u_pred": preds.u,
        "p_exceed_pred": preds.p,
        "xi_pred": preds.xi,
        "beta_pred": preds.beta,
        "u_scale_pred": preds.u_scale,
        "xi_scale_pred": preds.xi_scale,
        "beta_scale_pred": preds.beta_scale,
        # --- ego-current frame metadata (供 diffusion 反投 / MATLAB 实例化) ---
        "scenario_frame": np.array(
            [schema.get("scenario_frame", "ego_current")] * len(arrays.event_id)
        ),
        "scenario_context_schema_version": np.array(
            [SCENARIO_CONTEXT_SCHEMA_VERSION] * len(arrays.event_id)
        ),
        "ego_origin_x": arrays.ego_origin_x,
        "ego_origin_y": arrays.ego_origin_y,
        "ego_rot_cos": arrays.ego_rot_cos,
        "ego_rot_sin": arrays.ego_rot_sin,
        "ego_length": arrays.ego_length,
        "target_length": arrays.target_length,
    }

    for tau in tail_levels:
        q = tail_quantile_np(preds.u, preds.p, preds.xi, preds.beta, float(tau))
        invalid = tail_quantile_invalid_mask(preds.p, float(tau))
        data[f"q{int(tau * 100)}_pred"] = q
        data[f"q{int(tau * 100)}_invalid_mask"] = invalid.astype(np.int8)

    for tau in [float(t) for t in tail_levels if float(t) >= 0.95]:
        es = expected_shortfall_np(preds.u, preds.p, preds.xi, preds.beta, float(tau))
        data[f"es{int(tau * 100)}_pred"] = es

    train_mask = arrays.split_index == 0
    train_risk = arrays.risk_score[train_mask]
    # 真正的连续经验风险分位 (相对于训练集)
    if train_risk.size:
        train_sorted = np.sort(train_risk)
        data["empirical_risk_percentile_vs_train"] = (
            np.searchsorted(train_sorted, arrays.risk_score, side="right").astype(np.float32)
            / len(train_sorted)
        )
    else:
        data["empirical_risk_percentile_vs_train"] = np.full(len(arrays.event_id), np.nan, dtype=np.float32)
    for tau in tail_levels:
        thr = float(np.quantile(train_risk, tau)) if train_risk.size else float("nan")
        data[f"tail_label_{int(tau * 100)}"] = (arrays.risk_score > thr).astype(np.int8)

    if include_context_features:
        for i, key in enumerate(schema["context_keys"]):
            data[f"context_{key}"] = arrays.context_features[:, i]

    # --- canonical scenario context fields (扁平化为 canonical_*) ---
    canonical_field_order = [
        "ego_x0", "ego_y0", "ego_v0", "ego_vy0", "ego_ax0", "ego_ay0",
        "ego_length", "ego_width",
        "target_center_x0", "target_center_y0",
        "initial_gap", "initial_lateral_offset",
        "target_dx0", "target_dy0", "target_v0", "target_vy0",
        "target_ax0", "target_ay0", "target_length", "target_width",
        "relative_speed_0",
        "source_lane_id", "target_lane_id", "same_lane_initial",
        "time_horizon_s", "prefix_horizon_s", "planned_cutin_duration",
    ]
    for fld in canonical_field_order:
        col = np.empty(len(arrays.event_id), dtype=object)
        for i, eid in enumerate(arrays.event_id):
            rec = canonical_records.get(str(eid), {})
            col[i] = rec.get(fld)
        data[f"canonical_{fld}"] = col

    # extras keys may differ per event_type — encode them as canonical_extras_*
    extras_keys: set = set()
    for c in canonical_records.values():
        extras_keys.update((c.get("extras") or {}).keys())
    for ek in sorted(extras_keys):
        col = np.empty(len(arrays.event_id), dtype=object)
        for i, eid in enumerate(arrays.event_id):
            rec = canonical_records.get(str(eid), {})
            col[i] = (rec.get("extras") or {}).get(ek)
        data[f"canonical_extras_{ek}"] = col

    df = pd.DataFrame(data)
    df.to_csv(out / "tail_conditions.csv", index=False)
    logger.info("Wrote %s  (N=%d)", out / "tail_conditions.csv", len(df))
    return df
