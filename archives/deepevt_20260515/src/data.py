"""
data.py — DeepEVT 数据集构建与加载
==================================

输出：
  dataset.npz                     # 所有样本张量 (含 ego-frame metadata)
  feature_schema.json             # 特征 key 顺序与维度
  normalization_stats.json        # 仅使用 train split 计算的均值/方差
  train_val_test_split.json       # 以 recording 为粒度的切分记录
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from process_highd.src.io_utils import ensure_dir, save_json

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
    prefix_states: np.ndarray       # [samples, prefix_steps, actors, state_features]
    context_features: np.ndarray    # [samples, context_features]
    risk_score: np.ndarray          # [N] float32
    split_index: np.ndarray         # [N] int8  0/1/2
    prefix_start_frame: np.ndarray  # [N] int64
    prefix_end_frame: np.ndarray    # [N] int64
    risk_window_start_frame: np.ndarray  # [N] int64
    risk_window_end_frame: np.ndarray    # [N] int64
    # ego-current frame metadata (供 diffusion 反投回 highD 世界坐标)
    ego_origin_x: np.ndarray        # [N] float32
    ego_origin_y: np.ndarray        # [N] float32
    ego_rot_cos: np.ndarray         # [N] float32
    ego_rot_sin: np.ndarray         # [N] float32
    ego_length: np.ndarray          # [N] float32
    target_length: np.ndarray       # [N] float32


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def _list_ints(values: Iterable[object]) -> List[int]:
    return [int(v) for v in values]


def _load_recording_locations(
    raw_dir: str | Path,
    recording_ids: Iterable[int],
) -> Dict[int, int]:
    """Read highD locationId for each recording when available."""
    out: Dict[int, int] = {}
    raw = Path(raw_dir)
    for rid in sorted({int(r) for r in recording_ids}):
        path = raw / f"{rid:02d}_recordingMeta.csv"
        if not path.exists():
            path = raw / f"{rid}_recordingMeta.csv"
        if not path.exists():
            continue
        try:
            meta = pd.read_csv(path, nrows=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read recording meta for %s: %s", rid, exc)
            continue
        if "locationId" in meta.columns and len(meta) > 0:
            out[rid] = int(meta.loc[0, "locationId"])
    return out


def _random_split_remaining(
    train_val_ids: List[int],
    config: dict,
    *,
    seed: int,
    train_val_total_ratio: float,
) -> Tuple[List[int], List[int]]:
    splits_cfg = config.get("splits", {})
    val_r = float(splits_cfg.get("val_ratio", 0.15))
    val_fraction = val_r / max(float(train_val_total_ratio), 1e-6)
    val_fraction = min(max(val_fraction, 0.0), 1.0)

    rng = np.random.default_rng(seed)
    ids = [int(r) for r in sorted(set(train_val_ids))]
    rng.shuffle(ids)
    n_val = int(round(val_fraction * len(ids)))
    n_val = min(max(n_val, 1 if len(ids) > 1 and val_fraction > 0.0 else 0), max(len(ids) - 1, 0))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def _split_by_recording(
    recording_ids: List[int],
    config: dict,
    raw_dir: str | Path | None = None,
) -> Tuple[Dict[int, str], Dict[str, List[int]], Dict[str, object]]:
    """Split at recording/location level.

    Supported strategies:
    - ``recording``: random recording-level train/val/test split.
    - ``leave_location_out``: explicit ``test_location_ids`` held out.
    """
    splits_cfg = config.get("splits", {})
    strategy = str(splits_cfg.get("strategy", "recording"))
    seed = int(splits_cfg.get("random_seed", 42))
    train_r = float(splits_cfg.get("train_ratio", 0.70))
    val_r = float(splits_cfg.get("val_ratio", 0.15))
    test_r = float(splits_cfg.get("test_ratio", 1.0 - train_r - val_r))
    rids = sorted({int(r) for r in recording_ids})
    location_map: Dict[int, int] = {}
    split_meta: Dict[str, object] = {"strategy": strategy, "random_seed": seed}

    if strategy == "leave_location_out":
        if raw_dir is None:
            raise ValueError("leave_location_out requires raw_dir to read recordingMeta locationId.")
        location_map = _load_recording_locations(raw_dir, rids)
        missing = [r for r in rids if r not in location_map]
        if missing:
            raise RuntimeError(f"Missing locationId for recordings: {missing}")
        test_locations = set(_list_ints(splits_cfg.get("test_location_ids", [])))
        if not test_locations:
            raise ValueError("splits.test_location_ids must contain at least one locationId.")
        test_ids = [r for r in rids if location_map[r] in test_locations]
        if not test_ids:
            raise ValueError(f"No recordings found for test_location_ids={sorted(test_locations)}.")
        remaining = [r for r in rids if r not in set(test_ids)]
        train_ids, val_ids = _random_split_remaining(
            remaining,
            config,
            seed=seed,
            train_val_total_ratio=train_r + val_r,
        )
        split_meta["test_location_ids"] = sorted(int(x) for x in test_locations)
        split_meta["recording_location_ids"] = {str(k): int(v) for k, v in sorted(location_map.items())}
    else:
        if strategy != "recording":
            raise ValueError(f"Unknown split strategy: {strategy}")
        total = train_r + val_r + test_r
        if abs(total - 1.0) > 1e-6:
            logger.warning("Split ratios sum to %.3f, renormalising.", total)
            train_r, val_r, test_r = (r / total for r in (train_r, val_r, test_r))

        rng = np.random.default_rng(seed)
        shuffled = list(rids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(train_r * n))
        n_val = int(round(val_r * n))
        n_val = min(n_val, n - n_train)
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train:n_train + n_val]
        test_ids = shuffled[n_train + n_val:]

    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            "Invalid split: train/val/test must all be non-empty. "
            f"train={train_ids}, val={val_ids}, test={test_ids}"
        )

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
    split_meta.update({
        "train_recording_ids": split_to_rids["train"],
        "val_recording_ids": split_to_rids["val"],
        "test_recording_ids": split_to_rids["test"],
    })
    return rid_to_split, split_to_rids, split_meta


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
) -> Tuple[List[WindowSample], List[np.ndarray], List[str]]:
    """重建窗口并提取 DeepEVT context。"""
    events_df = pd.read_csv(events_csv)
    events_df = filter_events_by_type(events_df, event_type)
    logger.info("事件类型 %s: %d 条候选事件", event_type, len(events_df))

    samples: List[WindowSample] = []
    contexts: List[np.ndarray] = []
    keys_order: List[str] | None = None

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

        for _, row in sub.iterrows():
            sample = rebuild_event_window(rec, row, config)
            if sample is None:
                continue
            ctx_vec, keys = extract_context(
                event_type=event_type,
                states=sample.states,
                event_row=row,
                config=config,
                ego_length=sample.ego_length,
                target_length=sample.target_length,
                lane_width=sample.lane_width,
                target_final_y=sample.target_final_y,
            )
            if not bool(config.get("features", {}).get("use_context_features", True)):
                keys = []
                ctx_vec = np.zeros((0,), dtype=np.float32)
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
) -> DatasetArrays:
    if not samples:
        raise RuntimeError("No window samples were built — nothing to save.")

    n = len(samples)
    prefix_step_count = int(samples[0].states.shape[0])
    if any(int(s.states.shape[0]) != prefix_step_count for s in samples):
        raise RuntimeError("prefix state length changed across samples")
    prefix_states = np.zeros(
        (n, prefix_step_count, NUM_ACTORS, NUM_STATE_FEATURES),
        dtype=np.float32,
    )
    context = np.stack(contexts, axis=0).astype(np.float32)
    risk = np.zeros(n, dtype=np.float32)
    event_id = np.empty(n, dtype=object)
    rid_arr = np.zeros(n, dtype=np.int64)
    split_idx = np.zeros(n, dtype=np.int8)
    prefix_start_frame = np.zeros(n, dtype=np.int64)
    prefix_end_frame = np.zeros(n, dtype=np.int64)
    risk_window_start_frame = np.zeros(n, dtype=np.int64)
    risk_window_end_frame = np.zeros(n, dtype=np.int64)
    ego_origin_x = np.zeros(n, dtype=np.float32)
    ego_origin_y = np.zeros(n, dtype=np.float32)
    ego_rot_cos = np.zeros(n, dtype=np.float32)
    ego_rot_sin = np.zeros(n, dtype=np.float32)
    ego_length = np.zeros(n, dtype=np.float32)
    target_length = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(samples):
        prefix_states[i] = s.states[:prefix_step_count]
        risk[i] = s.risk_score
        event_id[i] = s.event_id
        rid_arr[i] = s.recording_id
        split_idx[i] = SPLIT_TO_INDEX[rid_to_split.get(int(s.recording_id), "train")]
        prefix_start_frame[i] = s.prefix_start_frame
        prefix_end_frame[i] = s.prefix_end_frame
        risk_window_start_frame[i] = s.risk_window_start_frame
        risk_window_end_frame[i] = s.risk_window_end_frame
        ego_origin_x[i] = s.ego_frame["origin_x"]
        ego_origin_y[i] = s.ego_frame["origin_y"]
        ego_rot_cos[i] = s.ego_frame["rot_cos"]
        ego_rot_sin[i] = s.ego_frame["rot_sin"]
        ego_length[i] = s.ego_length
        target_length[i] = s.target_length

    return DatasetArrays(
        event_id=event_id,
        recording_id=rid_arr,
        prefix_states=prefix_states,
        context_features=context,
        risk_score=risk,
        split_index=split_idx,
        prefix_start_frame=prefix_start_frame,
        prefix_end_frame=prefix_end_frame,
        risk_window_start_frame=risk_window_start_frame,
        risk_window_end_frame=risk_window_end_frame,
        ego_origin_x=ego_origin_x,
        ego_origin_y=ego_origin_y,
        ego_rot_cos=ego_rot_cos,
        ego_rot_sin=ego_rot_sin,
        ego_length=ego_length,
        target_length=target_length,
    )


def _compute_normalization(arrays: DatasetArrays, feature_keys: List[str]) -> Dict[str, dict]:
    """只使用 train split 计算 mean/std。"""
    mask = arrays.split_index == SPLIT_TO_INDEX["train"]
    if mask.sum() == 0:
        raise RuntimeError("Train split is empty; cannot compute normalization.")

    ctx = arrays.context_features[mask]
    if ctx.shape[1] == 0:
        ctx_mean = np.zeros((0,), dtype=np.float32)
        ctx_std = np.ones((0,), dtype=np.float32)
    else:
        ctx_mean = ctx.mean(axis=0).astype(np.float32)
        ctx_std = ctx.std(axis=0).astype(np.float32)
        ctx_std[ctx_std < 1e-6] = 1.0

    state = arrays.prefix_states[mask]
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

    samples, contexts, feature_keys = _collect_samples(
        events_csv, raw_dir, config, event_type,
    )

    recording_ids = sorted({int(s.recording_id) for s in samples})
    rid_to_split, split_to_rids, split_meta = _split_by_recording(
        recording_ids, config, raw_dir=raw_dir,
    )

    arrays = _stack_samples(samples, contexts, rid_to_split)

    norm_stats = _compute_normalization(arrays, feature_keys)

    npz_path = out_dir / "dataset.npz"
    np.savez(
        npz_path,
        event_id=arrays.event_id,
        recording_id=arrays.recording_id,
        prefix_states=arrays.prefix_states,
        context_features=arrays.context_features,
        risk_score=arrays.risk_score,
        split_index=arrays.split_index,
        prefix_start_frame=arrays.prefix_start_frame,
        prefix_end_frame=arrays.prefix_end_frame,
        risk_window_start_frame=arrays.risk_window_start_frame,
        risk_window_end_frame=arrays.risk_window_end_frame,
        ego_origin_x=arrays.ego_origin_x,
        ego_origin_y=arrays.ego_origin_y,
        ego_rot_cos=arrays.ego_rot_cos,
        ego_rot_sin=arrays.ego_rot_sin,
        ego_length=arrays.ego_length,
        target_length=arrays.target_length,
    )
    logger.info("已写出 %s  (samples=%d, prefix_steps=%d, context_features=%d)",
                npz_path, len(samples), arrays.prefix_states.shape[1],
                arrays.context_features.shape[1])

    feature_schema = {
        "event_type": event_type,
        "context_keys": feature_keys,
        "context_dim": len(feature_keys),
        "prefix_state_features": list(STATE_FEATURES),
        "num_actors": NUM_ACTORS,
        "prefix_steps": int(arrays.prefix_states.shape[1]),
        "window_length": int(samples[0].risk_window_frames.shape[0]),
        "risk_window_length": (
            int(samples[0].risk_window_frames.shape[0])
            if all(len(s.risk_window_frames) == len(samples[0].risk_window_frames) for s in samples)
            else None
        ),
        "risk_window_steps_min": int(min(len(s.risk_window_frames) for s in samples)),
        "risk_window_steps_max": int(max(len(s.risk_window_frames) for s in samples)),
        "scenario_frame": "ego_current",
        "time_semantics": {
            "prefix_window": "prefix_start_frame..prefix_end_frame, inclusive",
            "risk_window": "risk_window_start_frame..risk_window_end_frame, inclusive",
            "prefix_end_equals_risk_start": bool(
                samples[0].prefix_end_frame == samples[0].risk_window_start_frame
            ),
            "risk_score_window": (
                "future frames after the input prefix through event end"
                if config.get("risk_target", {}).get("mode", "future_after_prefix") == "future_after_prefix"
                else "risk_window_frames only"
            ),
            "target_mode": config.get("risk_target", {}).get("mode", "future_after_prefix"),
        },
    }
    save_json(feature_schema, out_dir / "feature_schema.json")
    save_json(norm_stats, out_dir / "normalization_stats.json")
    save_json(
        {
            **split_meta,
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
        n = len(npz["event_id"])

        def optional_array(name: str) -> np.ndarray:
            if name in npz:
                return npz[name]
            return np.full(n, -1, dtype=np.int64)

        return DatasetArrays(
            event_id=npz["event_id"],
            recording_id=npz["recording_id"],
            prefix_states=npz["prefix_states"],
            context_features=npz["context_features"],
            risk_score=npz["risk_score"],
            split_index=npz["split_index"],
            prefix_start_frame=optional_array("prefix_start_frame"),
            prefix_end_frame=optional_array("prefix_end_frame"),
            risk_window_start_frame=optional_array("risk_window_start_frame"),
            risk_window_end_frame=optional_array("risk_window_end_frame"),
            ego_origin_x=npz["ego_origin_x"],
            ego_origin_y=npz["ego_origin_y"],
            ego_rot_cos=npz["ego_rot_cos"],
            ego_rot_sin=npz["ego_rot_sin"],
            ego_length=npz["ego_length"],
            target_length=npz["target_length"],
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
        split_index=arrays.split_index,
        prefix_start_frame=arrays.prefix_start_frame,
        prefix_end_frame=arrays.prefix_end_frame,
        risk_window_start_frame=arrays.risk_window_start_frame,
        risk_window_end_frame=arrays.risk_window_end_frame,
        ego_origin_x=arrays.ego_origin_x,
        ego_origin_y=arrays.ego_origin_y,
        ego_rot_cos=arrays.ego_rot_cos,
        ego_rot_sin=arrays.ego_rot_sin,
        ego_length=arrays.ego_length,
        target_length=arrays.target_length,
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
        split_index=arrays.split_index[mask],
        prefix_start_frame=arrays.prefix_start_frame[mask],
        prefix_end_frame=arrays.prefix_end_frame[mask],
        risk_window_start_frame=arrays.risk_window_start_frame[mask],
        risk_window_end_frame=arrays.risk_window_end_frame[mask],
        ego_origin_x=arrays.ego_origin_x[mask],
        ego_origin_y=arrays.ego_origin_y[mask],
        ego_rot_cos=arrays.ego_rot_cos[mask],
        ego_rot_sin=arrays.ego_rot_sin[mask],
        ego_length=arrays.ego_length[mask],
        target_length=arrays.target_length[mask],
    )
