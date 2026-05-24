#!/usr/bin/env python3
"""Prepare tail-scored natural highD contexts for KING sampling."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.risk_utils import (
    interaction_metrics_from_states,
    write_csv,
    write_json,
)
from adversaray.src.rss import RSSConfig
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "king_guided_following.yaml"
)
SCRIPT_DEFAULTS = {
    "split": "val",
    "max_score_contexts": 0,
    "num_tail_contexts": 32,
    "seed": 42,
    "source": "recorded_future,initial_context",
    "tail_quantile": 0.90,
    "w_rss": 1.0,
    "w_ttc": 1.0,
    "w_gap": 1.0,
    "w_dv": 1.0,
    "eps": 1e-3,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _paths(cfg: dict[str, Any], base: Path) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    missing_paths = [
        key
        for key in ("natural_dataset_dir", "output_dir")
        if key not in paths
    ]
    if missing_paths:
        raise KeyError(
            f"Config paths is missing required keys: {missing_paths}"
        )
    missing_training = [
        key
        for key in ("tail_score_path", "tail_context_path")
        if key not in training
    ]
    if missing_training:
        raise KeyError(
            "Config training is missing required keys: "
            f"{missing_training}"
        )
    natural_dir = _resolve(paths["natural_dataset_dir"], base)
    return {
        "dataset": natural_dir / "dataset.npz",
        "output_dir": _resolve(paths["output_dir"], base),
        "tail_scores": _resolve(training["tail_score_path"], base),
        "tail_contexts": _resolve(training["tail_context_path"], base),
    }


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out.astype(np.float32)
    order = np.argsort(arr[finite], kind="mergesort")
    ranks = np.empty(int(finite.sum()), dtype=np.float64)
    ranks[order] = np.arange(1, int(finite.sum()) + 1, dtype=np.float64)
    out[finite] = ranks / max(int(finite.sum()), 1)
    return out.astype(np.float32)


def _raw_components(
    min_rss: np.ndarray,
    min_ttc: np.ndarray,
    min_gap: np.ndarray,
    closing_speed: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.maximum(0.0, -np.asarray(min_rss, dtype=np.float32)),
        1.0 / (np.asarray(min_ttc, dtype=np.float32) + eps),
        1.0 / (np.asarray(min_gap, dtype=np.float32) + eps),
        np.maximum(0.0, np.asarray(closing_speed, dtype=np.float32)),
    )


def _add_components(
    accum: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for dst, src in zip(accum, components, strict=True):
        dst += np.asarray(src, dtype=np.float32)


def _selected_split_indices(
    raw: dict[str, np.ndarray],
    split: str,
    max_contexts: int,
    seed: int,
) -> np.ndarray:
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[str(split)])[0]
    idx = idx.astype(np.int64)
    if idx.size == 0:
        raise RuntimeError(f"No contexts found for split '{split}'")
    if int(max_contexts) > 0 and idx.size > int(max_contexts):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(
            rng.choice(idx, size=int(max_contexts), replace=False)
        ).astype(np.int64)
    return idx


def build_tail_scores(
    *,
    cfg: dict[str, Any],
    dataset_path: Path,
    output_path: Path,
    output_dir: Path,
    split: str,
    max_contexts: int,
    seed: int,
    source: str,
    tail_quantile: float,
    weights: tuple[float, float, float, float],
    eps: float,
) -> Path:
    raw = _load_npz(dataset_path)
    required = {"context_states", "ego_length", "adv_length", "split_index"}
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"{dataset_path} is missing required arrays: {missing}")
    sources = {item.strip() for item in str(source).split(",") if item.strip()}
    allowed = {"recorded_future", "initial_context"}
    unknown = sorted(sources - allowed)
    if unknown:
        raise ValueError(f"Unknown tail score source(s): {unknown}")
    if not sources:
        raise ValueError(
            "SCRIPT_DEFAULTS['source'] must include at least one source"
        )
    if "recorded_future" in sources and "future_states" not in raw:
        raise RuntimeError(
            "dataset.npz is missing future_states; recorded_future tail "
            "scoring needs recorded futures."
        )

    idx = _selected_split_indices(raw, split, max_contexts, seed)
    rss_cfg = RSSConfig.from_config(cfg)
    context = raw["context_states"][idx]
    ego_len = raw["ego_length"][idx]
    adv_len = raw["adv_length"][idx]
    initial_metrics = interaction_metrics_from_states(
        context,
        np.repeat(context[:, -1:, :, :], repeats=1, axis=1),
        ego_len,
        adv_len,
        rss_cfg,
    )
    last_ego = context[:, -1, 0]
    last_lead = context[:, -1, 1]
    initial_gap = last_lead[:, 0] - last_ego[:, 0] - 0.5 * (ego_len + adv_len)
    initial_closing = last_ego[:, 2] - last_lead[:, 2]
    initial_ttc = np.where(
        initial_closing > 1e-6,
        initial_gap / np.maximum(initial_closing, 1e-6),
        1000.0,
    )
    initial_min_gap = initial_gap.astype(np.float32)
    initial_min_ttc = np.clip(initial_ttc, 0.0, 1000.0).astype(np.float32)
    initial_min_rss = initial_metrics["initial_rss_margin"].astype(np.float32)

    rss_raw = np.zeros(len(idx), dtype=np.float32)
    ttc_raw = np.zeros(len(idx), dtype=np.float32)
    gap_raw = np.zeros(len(idx), dtype=np.float32)
    dv_raw = np.zeros(len(idx), dtype=np.float32)
    raw_accum = (rss_raw, ttc_raw, gap_raw, dv_raw)
    min_gap = np.full(len(idx), np.inf, dtype=np.float32)
    min_ttc = np.full(len(idx), np.inf, dtype=np.float32)
    min_rss = np.full(len(idx), np.inf, dtype=np.float32)

    if "recorded_future" in sources:
        recorded_metrics = interaction_metrics_from_states(
            context,
            raw["future_states"][idx],
            ego_len,
            adv_len,
            rss_cfg,
        )
        _add_components(
            raw_accum,
            _raw_components(
                recorded_metrics["min_rss_margin"],
                recorded_metrics["min_ttc"],
                recorded_metrics["min_gap"],
                initial_metrics["initial_closing_speed"],
                eps=float(eps),
            ),
        )
        min_gap = np.minimum(
            min_gap,
            recorded_metrics["min_gap"].astype(np.float32),
        )
        min_ttc = np.minimum(
            min_ttc,
            recorded_metrics["min_ttc"].astype(np.float32),
        )
        min_rss = np.minimum(
            min_rss,
            recorded_metrics["min_rss_margin"].astype(np.float32),
        )
    else:
        recorded_metrics = initial_metrics

    if "initial_context" in sources:
        _add_components(
            raw_accum,
            _raw_components(
                initial_min_rss,
                initial_min_ttc,
                initial_min_gap,
                initial_metrics["initial_closing_speed"],
                eps=float(eps),
            ),
        )
        min_gap = np.minimum(min_gap, initial_min_gap)
        min_ttc = np.minimum(min_ttc, initial_min_ttc)
        min_rss = np.minimum(min_rss, initial_min_rss)

    min_gap = np.where(
        np.isfinite(min_gap),
        min_gap,
        initial_min_gap,
    ).astype(np.float32)
    min_ttc = np.where(
        np.isfinite(min_ttc),
        min_ttc,
        initial_min_ttc,
    ).astype(np.float32)
    min_rss = np.where(
        np.isfinite(min_rss),
        min_rss,
        initial_min_rss,
    ).astype(np.float32)
    rss_component = _percentile_rank(rss_raw)
    ttc_component = _percentile_rank(ttc_raw)
    gap_component = _percentile_rank(gap_raw)
    dv_component = _percentile_rank(dv_raw)
    w_rss, w_ttc, w_gap, w_dv = weights
    score = (
        w_rss * rss_component
        + w_ttc * ttc_component
        + w_gap * gap_component
        + w_dv * dv_component
    ).astype(np.float32)
    threshold = float(
        np.quantile(score[np.isfinite(score)], float(tail_quantile))
    )
    exceedance = np.maximum(score - threshold, 0.0).astype(np.float32)

    recording = (
        raw["recording_id"][idx]
        if "recording_id" in raw
        else np.full(len(idx), -1)
    )
    event = (
        raw["event_id"][idx]
        if "event_id" in raw
        else np.full(len(idx), "")
    )
    anchor = (
        raw["anchor_frame"][idx]
        if "anchor_frame" in raw
        else np.full(len(idx), -1)
    )
    rows: list[dict[str, Any]] = []
    for pos, dataset_idx in enumerate(idx):
        rows.append(
            {
                "dataset_index": int(dataset_idx),
                "recorded_min_gap": float(recorded_metrics["min_gap"][pos]),
                "recorded_min_ttc": float(recorded_metrics["min_ttc"][pos]),
                "recorded_min_rss_margin": float(
                    recorded_metrics["min_rss_margin"][pos]
                ),
                "min_gap": float(min_gap[pos]),
                "min_ttc": float(min_ttc[pos]),
                "min_rss_margin": float(min_rss[pos]),
                "criticality_score": float(score[pos]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        dataset_index=idx.astype(np.int64),
        split_index=raw["split_index"][idx].astype(np.int64),
        recording_id=np.asarray(recording),
        event_id=np.asarray(event),
        anchor_frame=np.asarray(anchor),
        initial_gap=initial_metrics["initial_gap"].astype(np.float32),
        initial_closing_speed=initial_metrics[
            "initial_closing_speed"
        ].astype(np.float32),
        recorded_min_gap=recorded_metrics["min_gap"].astype(np.float32),
        recorded_min_ttc=recorded_metrics["min_ttc"].astype(np.float32),
        recorded_min_rss_margin=recorded_metrics[
            "min_rss_margin"
        ].astype(np.float32),
        min_gap=min_gap.astype(np.float32),
        min_ttc=min_ttc.astype(np.float32),
        min_rss_margin=min_rss.astype(np.float32),
        rss_component=rss_component.astype(np.float32),
        ttc_component=ttc_component.astype(np.float32),
        gap_component=gap_component.astype(np.float32),
        dv_component=dv_component.astype(np.float32),
        criticality_score=score.astype(np.float32),
        tail_threshold=np.full(len(idx), threshold, dtype=np.float32),
        tail_exceedance=exceedance.astype(np.float32),
    )
    write_csv(output_path.with_suffix(".csv"), rows)
    write_json(
        output_dir / "tail_score_summary.json",
        {
            "dataset": str(dataset_path),
            "output": str(output_path),
            "split": str(split),
            "sources": sorted(sources),
            "num_contexts": int(len(idx)),
            "tail_quantile": float(tail_quantile),
            "tail_threshold": float(threshold),
            "tail_fraction": float(np.mean(score >= threshold)),
            "score_mean": float(np.mean(score)),
            "score_p95": float(np.percentile(score, 95.0)),
        },
    )
    logger.info("Wrote %d tail scores to %s", len(idx), output_path)
    return output_path


def export_tail_contexts(
    *,
    dataset_path: Path,
    tail_path: Path,
    output_path: Path,
    output_dir: Path,
    split: str,
    tail_quantile: float,
    num_contexts: int,
) -> Path:
    raw = _load_npz(dataset_path)
    tail = _load_npz(tail_path)
    required_raw = {
        "context_states",
        "ego_length",
        "adv_length",
        "split_index",
        "recording_id",
        "event_id",
        "anchor_frame",
        "future_states",
    }
    required_tail = {"dataset_index", "criticality_score", "tail_threshold"}
    missing_raw = sorted(required_raw - set(raw))
    missing_tail = sorted(required_tail - set(tail))
    if missing_raw:
        raise KeyError(
            f"{dataset_path} is missing required arrays: {missing_raw}"
        )
    if missing_tail:
        raise KeyError(
            f"{tail_path} is missing required arrays: {missing_tail}"
        )

    dataset_index = np.asarray(tail["dataset_index"], dtype=np.int64)
    score = np.asarray(tail["criticality_score"], dtype=np.float32)
    threshold = float(
        np.quantile(score[np.isfinite(score)], float(tail_quantile))
    )
    mask = (
        (dataset_index >= 0)
        & (dataset_index < raw["context_states"].shape[0])
        & (raw["split_index"][dataset_index] == SPLIT_TO_INDEX[str(split)])
        & np.isfinite(score)
        & (score >= threshold)
    )
    if not np.any(mask):
        raise RuntimeError(
            "No natural tail contexts found for split "
            f"'{split}' at quantile {tail_quantile}"
        )
    tail_pos = np.where(mask)[0]
    tail_pos = tail_pos[np.argsort(score[tail_pos])[::-1]]
    if int(num_contexts) > 0:
        tail_pos = tail_pos[: int(num_contexts)]
    selected = dataset_index[tail_pos]
    horizon_steps = int(raw["future_states"].shape[1])
    event_end_by_key: dict[tuple[int, str], int] = {}
    for rec, event, anchor in zip(
        raw["recording_id"],
        raw["event_id"],
        raw["anchor_frame"],
        strict=True,
    ):
        key = (int(rec), str(event))
        end_frame = int(anchor) + horizon_steps
        event_end_by_key[key] = max(
            event_end_by_key.get(key, end_frame),
            end_frame,
        )
    event_steps = np.zeros(len(selected), dtype=np.int64)
    for pos, raw_idx in enumerate(selected):
        key = (
            int(raw["recording_id"][raw_idx]),
            str(raw["event_id"][raw_idx]),
        )
        end_frame = event_end_by_key[key]
        remaining = int(end_frame - raw["anchor_frame"][raw_idx])
        event_steps[pos] = max(remaining, 1)

    payload: dict[str, np.ndarray] = {
        "context_states": raw["context_states"][selected].astype(np.float32),
        "ego_length": raw["ego_length"][selected].astype(np.float32),
        "adv_length": raw["adv_length"][selected].astype(np.float32),
        "split_index": raw["split_index"][selected].astype(np.int64),
        "dataset_index": selected.astype(np.int64),
        "event_steps": event_steps.astype(np.int64),
        "source_type": np.asarray(["highd_tail_natural"] * len(selected)),
        "criticality_score": score[tail_pos].astype(np.float32),
        "tail_threshold": np.full(len(selected), threshold, dtype=np.float32),
    }
    for key in (
        "recording_id",
        "event_id",
        "anchor_frame",
        "initial_gap",
        "initial_closing_speed",
        "recorded_min_gap",
        "recorded_min_ttc",
        "recorded_min_rss_margin",
        "min_gap",
        "min_ttc",
        "min_rss_margin",
        "rss_component",
        "ttc_component",
        "gap_component",
        "dv_component",
        "tail_exceedance",
    ):
        if key in tail:
            payload[key] = np.asarray(tail[key][tail_pos])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    write_json(
        output_dir / "tail_context_summary.json",
        {
            "dataset": str(dataset_path),
            "tail_scores": str(tail_path),
            "output": str(output_path),
            "split": str(split),
            "num_contexts": int(len(selected)),
            "event_steps_min": int(np.min(event_steps)),
            "event_steps_max": int(np.max(event_steps)),
            "event_steps_mean": float(np.mean(event_steps)),
            "tail_quantile": float(tail_quantile),
            "tail_threshold": float(threshold),
            "score_min": float(np.min(score[tail_pos])),
            "score_max": float(np.max(score[tail_pos])),
        },
    )
    logger.info(
        "Wrote %d natural tail contexts to %s",
        len(selected),
        output_path,
    )
    return output_path


def main() -> None:
    setup_logging(SCRIPT_DEFAULTS["log_level"])

    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    cfg = load_yaml(cfg_path)
    paths = _paths(cfg, cfg_path.parent)
    split = str(SCRIPT_DEFAULTS["split"])
    tail_quantile = float(SCRIPT_DEFAULTS["tail_quantile"])
    tail_path = build_tail_scores(
        cfg=cfg,
        dataset_path=paths["dataset"],
        output_path=paths["tail_scores"],
        output_dir=paths["output_dir"],
        split=split,
        max_contexts=int(SCRIPT_DEFAULTS["max_score_contexts"]),
        seed=int(SCRIPT_DEFAULTS["seed"]),
        source=str(SCRIPT_DEFAULTS["source"]),
        tail_quantile=tail_quantile,
        weights=(
            float(SCRIPT_DEFAULTS["w_rss"]),
            float(SCRIPT_DEFAULTS["w_ttc"]),
            float(SCRIPT_DEFAULTS["w_gap"]),
            float(SCRIPT_DEFAULTS["w_dv"]),
        ),
        eps=float(SCRIPT_DEFAULTS["eps"]),
    )
    export_tail_contexts(
        dataset_path=paths["dataset"],
        tail_path=tail_path,
        output_path=paths["tail_contexts"],
        output_dir=paths["output_dir"],
        split=split,
        tail_quantile=tail_quantile,
        num_contexts=int(SCRIPT_DEFAULTS["num_tail_contexts"]),
    )


if __name__ == "__main__":
    main()
