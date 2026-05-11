"""
data.py — DeepEVT 数据集构建与加载
==================================

输出：
  dataset.npz                     # 所有样本张量
  feature_schema.json             # 特征 key 顺序与维度
  normalization_stats.json        # 仅使用 train split 计算的均值/方差
  train_val_test_split.json       # 以 recording 为粒度的切分记录
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tread_highd.src.io_utils import ensure_dir, save_json

from .features import extract_context, feature_keys_for
from .window_rebuild import (
    NUM_ACTORS,
    NUM_STATE_FEATURES,
    STATE_FEATURES,
    WindowSample,
    filter_events_by_type,
    prepare_recording,
    rebuild_event_window,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetArrays:
    event_id: np.ndarray            # [N] object
    recording_id: np.ndarray        # [N] int
    prefix_states: np.ndarray       # [N, K, 2, F] float32
    context_features: np.ndarray    # [N, C] float32
    risk_score: np.ndarray          # [N] float32
    min_ttc: np.ndarray             # [N]
    min_thw: np.ndarray             # [N]
    max_drac: np.ndarray            # [N]
    split_index: np.ndarray         # [N] int8  0/1/2


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def _split_by_recording(
    recording_ids: List[int], config: dict,
) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """按 recording 级别切分，返回 rid->split 以及 split->rids."""
    splits_cfg = config.get("splits", {})
    seed = int(splits_cfg.get("random_seed", 42))
    train_r = float(splits_cfg.get("train_ratio", 0.70))
    val_r = float(splits_cfg.get("val_ratio", 0.15))
    test_r = float(splits_cfg.get("test_ratio", 1.0 - train_r - val_r))
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-6:
        logger.warning("Split ratios sum to %.3f, renormalising.", total)
        train_r, val_r, test_r = (r / total for r in (train_r, val_r, test_r))

    rng = np.random.default_rng(seed)
    rids = sorted({int(r) for r in recording_ids})
    rng.shuffle(rids)
    n = len(rids)
    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    n_val = min(n_val, n - n_train)
    train_ids = rids[:n_train]
    val_ids = rids[n_train:n_train + n_val]
    test_ids = rids[n_train + n_val:]

    rid_to_split: Dict[int, str] = {}
    for rid in train_ids:
        rid_to_split[int(rid)] = "train"
    for rid in val_ids:
        rid_to_split[int(rid)] = "val"
    for rid in test_ids:
        rid_to_split[int(rid)] = "test"

    split_to_rids = {
        "train": [int(r) for r in train_ids],
        "val": [int(r) for r in val_ids],
        "test": [int(r) for r in test_ids],
    }
    return rid_to_split, split_to_rids


SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}
INDEX_TO_SPLIT = {v: k for k, v in SPLIT_TO_INDEX.items()}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _collect_samples(
    events_csv: str | Path,
    raw_dir: str | Path,
    config: dict,
    event_type: str,
) -> Tuple[List[WindowSample], List[np.ndarray], List[List[str]]]:
    """重建窗口，同时按每个 recording 内的 meta 提取 context 特征。"""
    events_df = pd.read_csv(events_csv)
    events_df = filter_events_by_type(events_df, event_type)
    logger.info("事件类型 %s: %d 条候选事件", event_type, len(events_df))

    samples: List[WindowSample] = []
    contexts: List[np.ndarray] = []
    keys_order: Optional[List[str]] = None

    recording_ids = sorted(events_df["recording_id"].unique().tolist())
    for rid in recording_ids:
        rid_int = int(rid)
        sub = events_df[events_df["recording_id"] == rid_int]
        if len(sub) == 0:
            continue
        try:
            rec = prepare_recording(str(raw_dir), rid_int, config)
        except Exception as exc:  # noqa: BLE001
            logger.error("Recording %02d load failed: %s", rid_int, exc)
            continue

        # 直接调用 iter_window_samples 会重复加载；此处复用预处理后的 recording
        for _, row in sub.iterrows():
            sample = rebuild_event_window(rec, row, config)
            if sample is None:
                continue
            ego_len = float(rec.tracks_meta.loc[int(sample.ego_id)]["width"])
            tgt_len = float(rec.tracks_meta.loc[int(sample.target_id)]["width"])
            ctx_vec, keys = extract_context(
                event_type=event_type,
                states=sample.states,
                event_row=row,
                config=config,
                ego_length=ego_len,
                target_length=tgt_len,
            )
            if keys_order is None:
                keys_order = keys
            elif keys != keys_order:
                raise RuntimeError("context feature keys changed across samples")
            samples.append(sample)
            contexts.append(ctx_vec)

    if keys_order is None:
        keys_order = list(feature_keys_for(event_type))
    return samples, contexts, keys_order


def _stack_samples(
    samples: List[WindowSample],
    contexts: List[np.ndarray],
    rid_to_split: Dict[int, str],
    prefix_steps: int,
) -> DatasetArrays:
    if not samples:
        raise RuntimeError("No window samples were built — nothing to save.")

    n = len(samples)
    # prefix_steps 可能 > window_length 时裁剪
    K = min(int(prefix_steps), samples[0].states.shape[0])
    prefix_states = np.zeros((n, K, NUM_ACTORS, NUM_STATE_FEATURES), dtype=np.float32)
    context = np.stack(contexts, axis=0).astype(np.float32)
    risk = np.zeros(n, dtype=np.float32)
    min_ttc = np.zeros(n, dtype=np.float32)
    min_thw = np.zeros(n, dtype=np.float32)
    max_drac = np.zeros(n, dtype=np.float32)
    event_id = np.empty(n, dtype=object)
    rid_arr = np.zeros(n, dtype=np.int64)
    split_idx = np.zeros(n, dtype=np.int8)

    for i, s in enumerate(samples):
        prefix_states[i] = s.states[:K]
        risk[i] = s.risk_score
        min_ttc[i] = s.min_ttc
        min_thw[i] = s.min_thw
        max_drac[i] = s.max_drac
        event_id[i] = s.event_id
        rid_arr[i] = s.recording_id
        split_idx[i] = SPLIT_TO_INDEX[rid_to_split.get(int(s.recording_id), "train")]

    return DatasetArrays(
        event_id=event_id,
        recording_id=rid_arr,
        prefix_states=prefix_states,
        context_features=context,
        risk_score=risk,
        min_ttc=min_ttc,
        min_thw=min_thw,
        max_drac=max_drac,
        split_index=split_idx,
    )


def _compute_normalization(arrays: DatasetArrays, feature_keys: List[str]) -> Dict[str, dict]:
    """只使用 train split 计算 mean/std。"""
    mask = arrays.split_index == SPLIT_TO_INDEX["train"]
    if mask.sum() == 0:
        raise RuntimeError("Train split is empty; cannot compute normalization.")

    ctx = arrays.context_features[mask]
    ctx_mean = ctx.mean(axis=0).astype(np.float32)
    ctx_std = ctx.std(axis=0).astype(np.float32)
    ctx_std[ctx_std < 1e-6] = 1.0

    state = arrays.prefix_states[mask]                 # [Nt, K, A, F]
    state_flat = state.reshape(-1, NUM_STATE_FEATURES)
    state_mean = state_flat.mean(axis=0).astype(np.float32)
    state_std = state_flat.std(axis=0).astype(np.float32)
    state_std[state_std < 1e-6] = 1.0

    return {
        "context": {
            "keys": feature_keys,
            "mean": ctx_mean.tolist(),
            "std": ctx_std.tolist(),
        },
        "prefix_states": {
            "features": list(STATE_FEATURES),
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
        },
    }


def build_and_save_dataset(
    events_csv: str | Path,
    raw_dir: str | Path,
    config: dict,
    output_dir: str | Path,
    event_type: str,
) -> DatasetArrays:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    samples, contexts, feature_keys = _collect_samples(events_csv, raw_dir, config, event_type)

    recording_ids = sorted({int(s.recording_id) for s in samples})
    rid_to_split, split_to_rids = _split_by_recording(recording_ids, config)

    prefix_steps = int(config.get("prefix", {}).get("prefix_steps", 25))
    arrays = _stack_samples(samples, contexts, rid_to_split, prefix_steps)

    norm_stats = _compute_normalization(arrays, feature_keys)

    npz_path = out_dir / "dataset.npz"
    np.savez(
        npz_path,
        event_id=arrays.event_id,
        recording_id=arrays.recording_id,
        prefix_states=arrays.prefix_states,
        context_features=arrays.context_features,
        risk_score=arrays.risk_score,
        min_ttc=arrays.min_ttc,
        min_thw=arrays.min_thw,
        max_drac=arrays.max_drac,
        split_index=arrays.split_index,
    )
    logger.info("已写出 %s  (N=%d, K=%d, C=%d)",
                npz_path, len(samples), arrays.prefix_states.shape[1],
                arrays.context_features.shape[1])

    feature_schema = {
        "event_type": event_type,
        "context_keys": feature_keys,
        "context_dim": len(feature_keys),
        "prefix_state_features": list(STATE_FEATURES),
        "num_actors": NUM_ACTORS,
        "prefix_steps": int(arrays.prefix_states.shape[1]),
        "window_length": int(samples[0].states.shape[0]),
    }
    save_json(feature_schema, out_dir / "feature_schema.json")
    save_json(norm_stats, out_dir / "normalization_stats.json")
    save_json(
        {
            "strategy": config.get("splits", {}).get("strategy", "recording"),
            "random_seed": int(config.get("splits", {}).get("random_seed", 42)),
            "train_recording_ids": split_to_rids["train"],
            "val_recording_ids": split_to_rids["val"],
            "test_recording_ids": split_to_rids["test"],
            "n_train": int((arrays.split_index == 0).sum()),
            "n_val": int((arrays.split_index == 1).sum()),
            "n_test": int((arrays.split_index == 2).sum()),
        },
        out_dir / "train_val_test_split.json",
    )
    return arrays


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_dataset(output_dir: str | Path) -> DatasetArrays:
    out = Path(output_dir)
    with np.load(out / "dataset.npz", allow_pickle=True) as npz:
        return DatasetArrays(
            event_id=npz["event_id"],
            recording_id=npz["recording_id"],
            prefix_states=npz["prefix_states"],
            context_features=npz["context_features"],
            risk_score=npz["risk_score"],
            min_ttc=npz["min_ttc"],
            min_thw=npz["min_thw"],
            max_drac=npz["max_drac"],
            split_index=npz["split_index"],
        )


def apply_normalization(arrays: DatasetArrays, norm_stats: dict) -> DatasetArrays:
    ctx_mean = np.array(norm_stats["context"]["mean"], dtype=np.float32)
    ctx_std = np.array(norm_stats["context"]["std"], dtype=np.float32)
    state_mean = np.array(norm_stats["prefix_states"]["mean"], dtype=np.float32)
    state_std = np.array(norm_stats["prefix_states"]["std"], dtype=np.float32)

    ctx = (arrays.context_features - ctx_mean) / ctx_std
    state = (arrays.prefix_states - state_mean) / state_std
    return DatasetArrays(
        event_id=arrays.event_id,
        recording_id=arrays.recording_id,
        prefix_states=state.astype(np.float32),
        context_features=ctx.astype(np.float32),
        risk_score=arrays.risk_score,
        min_ttc=arrays.min_ttc,
        min_thw=arrays.min_thw,
        max_drac=arrays.max_drac,
        split_index=arrays.split_index,
    )


def subset(arrays: DatasetArrays, split_name: str) -> DatasetArrays:
    idx = SPLIT_TO_INDEX[split_name]
    mask = arrays.split_index == idx
    return DatasetArrays(
        event_id=arrays.event_id[mask],
        recording_id=arrays.recording_id[mask],
        prefix_states=arrays.prefix_states[mask],
        context_features=arrays.context_features[mask],
        risk_score=arrays.risk_score[mask],
        min_ttc=arrays.min_ttc[mask],
        min_thw=arrays.min_thw[mask],
        max_drac=arrays.max_drac[mask],
        split_index=arrays.split_index[mask],
    )

