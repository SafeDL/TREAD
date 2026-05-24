#!/usr/bin/env python3
"""Build EVT-conditioned synthetic near-boundary car-following contexts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.features import extract_context
from diffusion.src.utils import load_json, load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
SCRIPT_DEFAULTS = {
    "split": "val",
    "num_contexts": 10000,
    "max_attempts": 0,
    "seed": 42,
    "tail_anchor_quantile": 0.90,
    "zscore_limit": 4.0,
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RSSConfig:
    response_time: float = 1.0
    ego_max_accel: float = 2.0
    ego_min_brake: float = 4.0
    lead_max_brake: float = 6.0
    temperature: float = 1.0
    pool_beta: float = 8.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RSSConfig":
        cfg = config.get("rss", config)
        return cls(
            response_time=float(cfg.get("response_time", 1.0)),
            ego_max_accel=float(cfg.get("ego_max_accel", 2.0)),
            ego_min_brake=float(cfg.get("ego_min_brake", 4.0)),
            lead_max_brake=float(cfg.get("lead_max_brake", 6.0)),
            temperature=float(cfg.get("temperature", 1.0)),
            pool_beta=float(cfg.get("pool_beta", 8.0)),
        )


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def rss_safe_distance_np(ego_velocity: np.ndarray, lead_velocity: np.ndarray, cfg: RSSConfig) -> np.ndarray:
    ego_velocity = np.asarray(ego_velocity, dtype=np.float64)
    lead_velocity = np.asarray(lead_velocity, dtype=np.float64)
    rho = float(cfg.response_time)
    ego_after_response = ego_velocity + rho * float(cfg.ego_max_accel)
    ego_distance = ego_velocity * rho + 0.5 * float(cfg.ego_max_accel) * rho * rho
    ego_brake_distance = np.square(ego_after_response) / max(2.0 * float(cfg.ego_min_brake), 1e-6)
    lead_brake_distance = np.square(lead_velocity) / max(2.0 * float(cfg.lead_max_brake), 1e-6)
    return np.maximum(ego_distance + ego_brake_distance - lead_brake_distance, 0.0)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _paths(cfg: dict[str, Any], config_dir: Path) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    required_paths = ("natural_dataset_dir", "output_dir", "rss_config")
    missing_paths = [key for key in required_paths if key not in paths]
    if missing_paths:
        raise KeyError(f"Config paths is missing required keys: {missing_paths}")
    required_training = ("tail_score_path", "synthetic_context_path")
    missing_training = [key for key in required_training if key not in training]
    if missing_training:
        raise KeyError(f"Config training is missing required keys: {missing_training}")
    natural_dir = _resolve(paths["natural_dataset_dir"], config_dir)
    return {
        "dataset": natural_dir / "dataset.npz",
        "feature_schema": natural_dir / "feature_schema.json",
        "normalization_stats": natural_dir / "normalization_stats.json",
        "tail_scores": _resolve(training["tail_score_path"], config_dir),
        "rss_config": _resolve(paths["rss_config"], config_dir),
        "output": _resolve(training["synthetic_context_path"], config_dir),
    }


def _relative_history(history: np.ndarray, ego_length: float, lead_length: float) -> np.ndarray:
    ego = np.asarray(history[:, 0], dtype=np.float32)
    lead = np.asarray(history[:, 1], dtype=np.float32)
    gap = lead[:, 0] - ego[:, 0] - 0.5 * (ego_length + lead_length)
    lateral = lead[:, 1] - ego[:, 1]
    delta_v = ego[:, 2] - lead[:, 2]
    delta_a = ego[:, 4] - lead[:, 4]
    ttc = np.where(delta_v > 1e-6, gap / np.maximum(delta_v, 1e-6), 1000.0)
    thw = gap / np.maximum(ego[:, 2], 1e-6)
    return np.stack(
        [gap, lateral, delta_v, delta_a, np.clip(ttc, 0.0, 1000.0), np.clip(thw, 0.0, 200.0)],
        axis=-1,
    ).astype(np.float32)


def _zmax(value: np.ndarray, stats: dict[str, Any], key: str) -> float:
    item = stats[key]
    mean = np.asarray(item["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(item["std"], dtype=np.float32), 1e-6)
    z = (np.asarray(value, dtype=np.float32) - mean) / std
    return float(np.max(np.abs(z)))


def _support(raw: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, tuple[float, float]]:
    context = np.asarray(raw["context_states"][idx, -1], dtype=np.float32)
    ego_length = np.asarray(raw["ego_length"][idx], dtype=np.float32)
    adv_length = np.asarray(raw["adv_length"][idx], dtype=np.float32)
    gap = context[:, 1, 0] - context[:, 0, 0] - 0.5 * (ego_length + adv_length)
    rel_speed = context[:, 0, 2] - context[:, 1, 2]
    speed = context[:, :, 2].reshape(-1)
    accel = context[:, :, 4].reshape(-1)
    out: dict[str, tuple[float, float]] = {}
    for key, values in {"speed": speed, "accel": accel, "gap": gap, "rel_speed": rel_speed}.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        out[key] = (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0)))
    return out


def _sample_anchors(
    raw: dict[str, np.ndarray],
    tail: dict[str, np.ndarray],
    *,
    split: str,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if "dataset_index" not in tail or "criticality_score" not in tail:
        raise KeyError("Tail score file must contain dataset_index and criticality_score")
    dataset_index = np.asarray(tail["dataset_index"], dtype=np.int64)
    score = np.asarray(tail["criticality_score"], dtype=np.float64)
    valid = (dataset_index >= 0) & (dataset_index < raw["context_states"].shape[0]) & np.isfinite(score)
    if split != "all":
        if "split_index" not in raw:
            raise KeyError("dataset.npz is missing split_index")
        valid &= raw["split_index"][dataset_index] == SPLIT_TO_INDEX[split]
    if not np.any(valid):
        raise RuntimeError(f"No tail scores cover split '{split}'")
    threshold = float(np.quantile(score[valid], float(quantile)))
    mask = valid & (score >= threshold)
    if not np.any(mask):
        raise RuntimeError(f"No tail anchors found at quantile {quantile}")
    weight_key = "tail_sampling_weight" if "tail_sampling_weight" in tail else "tail_weight"
    weights = np.asarray(tail[weight_key], dtype=np.float64) if weight_key in tail else score
    weights = np.where(np.isfinite(weights[mask]) & (weights[mask] > 0.0), weights[mask], 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(int(mask.sum()), dtype=np.float64)
    return dataset_index[mask], score[mask], weights / float(weights.sum()), threshold


def _build_history(
    *,
    history_steps: int,
    dt: float,
    ego_length: float,
    adv_length: float,
    target_gap: float,
    v_ego: float,
    v_lead: float,
    a_ego: float,
    a_lead: float,
) -> np.ndarray:
    history = np.zeros((history_steps, 2, 6), dtype=np.float32)
    lead_x0 = float(target_gap + 0.5 * (ego_length + adv_length))
    for i in range(history_steps):
        tau = float(history_steps - 1 - i) * float(dt)
        ego_v = max(v_ego - a_ego * tau, 0.0)
        lead_v = max(v_lead - a_lead * tau, 0.0)
        history[i, 0, 0] = -v_ego * tau + 0.5 * a_ego * tau * tau
        history[i, 1, 0] = lead_x0 - v_lead * tau + 0.5 * a_lead * tau * tau
        history[i, 0, 2] = ego_v
        history[i, 1, 2] = lead_v
        history[i, 0, 4] = a_ego
        history[i, 1, 4] = a_lead
    history[-1, 0, 0] = 0.0
    history[-1, 1, 0] = lead_x0
    history[-1, 0, 2] = v_ego
    history[-1, 1, 2] = v_lead
    return history


def _passes_filter(
    history: np.ndarray,
    *,
    ego_length: float,
    adv_length: float,
    dt: float,
    target_gap: float,
    target_ttc: float,
    target_rss_margin: float,
    initial_gap_min: float,
    support: dict[str, tuple[float, float]],
    stats: dict[str, Any],
    zscore_limit: float,
) -> tuple[bool, float]:
    v_ego = float(history[-1, 0, 2])
    v_lead = float(history[-1, 1, 2])
    rel_speed = v_ego - v_lead
    if not (target_gap > initial_gap_min and target_gap > 0.0 and target_ttc > 2.0 and v_ego > 0.0 and v_lead > 0.0):
        return False, float("inf")
    if not (3.0 <= target_ttc <= 8.0 and 8.0 <= target_gap <= 25.0 and -2.0 <= target_rss_margin <= 8.0):
        return False, float("inf")
    checks = {
        "speed": np.asarray([v_ego, v_lead], dtype=np.float32),
        "accel": history[:, :, 4].reshape(-1),
        "gap": np.asarray([target_gap], dtype=np.float32),
        "rel_speed": np.asarray([rel_speed], dtype=np.float32),
    }
    for key, values in checks.items():
        lo, hi = support[key]
        values = np.asarray(values, dtype=np.float32)
        if np.any(values < lo) or np.any(values > hi):
            return False, float("inf")
    features, _ = extract_context(history, ego_length, adv_length, dt)
    relative = _relative_history(history, ego_length, adv_length)
    max_z = max(
        _zmax(history, stats, "context_states"),
        _zmax(features, stats, "context_features"),
        _zmax(relative, stats, "relative_history"),
    )
    return max_z < float(zscore_limit), max_z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--tail-score-path", default="")
    parser.add_argument("--rss-config", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default=SCRIPT_DEFAULTS["split"])
    parser.add_argument("--num-contexts", type=int, default=SCRIPT_DEFAULTS["num_contexts"])
    parser.add_argument("--max-attempts", type=int, default=SCRIPT_DEFAULTS["max_attempts"])
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--tail-anchor-quantile", type=float, default=SCRIPT_DEFAULTS["tail_anchor_quantile"])
    parser.add_argument("--zscore-limit", type=float, default=SCRIPT_DEFAULTS["zscore_limit"])
    parser.add_argument("--log-level", default=SCRIPT_DEFAULTS["log_level"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    paths = _paths(cfg, cfg_path.parent)
    dataset_path = Path(args.dataset).resolve() if args.dataset else paths["dataset"]
    tail_path = Path(args.tail_score_path).resolve() if args.tail_score_path else paths["tail_scores"]
    rss_path = Path(args.rss_config).resolve() if args.rss_config else paths["rss_config"]
    output_path = Path(args.output).resolve() if args.output else paths["output"]
    if not rss_path.exists():
        raise FileNotFoundError(f"RSS calibration config not found: {rss_path}")
    recommended = load_yaml(rss_path)
    cfg.setdefault("rss", {}).update(recommended.get("rss", recommended))

    raw = _load_npz(dataset_path)
    required_raw = {"context_states", "ego_length", "adv_length", "split_index"}
    missing_raw = sorted(required_raw - set(raw))
    if missing_raw:
        raise KeyError(f"{dataset_path} is missing required arrays: {missing_raw}")
    tail = _load_npz(tail_path)
    schema = load_json(paths["feature_schema"])
    stats = load_json(paths["normalization_stats"])
    history_steps = int(schema.get("history_steps", raw["context_states"].shape[1]))
    dt = float(schema.get("dt", cfg.get("sampling", {}).get("dt", 0.04)))
    initial_gap_min = float(cfg.get("env", {}).get("initial_gap_min", 0.1))
    rss_cfg = RSSConfig.from_config(cfg)
    rng = np.random.default_rng(int(args.seed))

    split_idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[str(args.split)])[0].astype(np.int64)
    if split_idx.size == 0:
        raise RuntimeError(f"No natural contexts found for split '{args.split}'")
    support = _support(raw, split_idx)
    anchors, anchor_scores, anchor_weights, threshold = _sample_anchors(
        raw,
        tail,
        split=str(args.split),
        quantile=float(args.tail_anchor_quantile),
    )
    speed_lo, speed_hi = support["speed"]
    accel_lo, accel_hi = support["accel"]
    gap_lo, gap_hi = support["gap"]
    rel_lo, rel_hi = support["rel_speed"]

    contexts: list[np.ndarray] = []
    ego_lengths: list[float] = []
    adv_lengths: list[float] = []
    anchor_indices: list[int] = []
    target_gaps: list[float] = []
    target_ttcs: list[float] = []
    target_rss_margins: list[float] = []
    criticality_scores: list[float] = []
    anchor_criticality_scores: list[float] = []
    max_zscores: list[float] = []
    max_attempts = int(args.max_attempts) if int(args.max_attempts) > 0 else max(int(args.num_contexts) * 50, 1000)

    for attempt in range(max_attempts):
        if len(contexts) >= int(args.num_contexts):
            break
        pos = int(rng.choice(np.arange(len(anchors)), p=anchor_weights))
        dataset_idx = int(anchors[pos])
        anchor = np.asarray(raw["context_states"][dataset_idx], dtype=np.float32)
        ego_length = float(raw["ego_length"][dataset_idx])
        adv_length = float(raw["adv_length"][dataset_idx])
        last = anchor[-1]
        v_ego = float(np.clip(last[0, 2] + rng.normal(0.0, 0.35), speed_lo, speed_hi))
        lead_seed = float(np.clip(last[1, 2] + rng.normal(0.0, 0.35), speed_lo, speed_hi))
        sampled_margin = float(rng.uniform(-2.0, 8.0))
        sampled_ttc = float(rng.uniform(3.0, 8.0))
        rss_gap = float(rss_safe_distance_np(np.asarray([v_ego]), np.asarray([lead_seed]), rss_cfg)[0] + sampled_margin)
        target_gap = float(np.clip(rss_gap, max(8.0, gap_lo, initial_gap_min + 1e-3), min(25.0, gap_hi)))
        v_lead = float(v_ego - target_gap / sampled_ttc)
        if v_lead <= 0.0:
            continue
        v_lead = float(np.clip(v_lead, speed_lo, speed_hi))
        closing = v_ego - v_lead
        if closing <= 1e-6:
            continue
        target_ttc = float(target_gap / closing)
        target_rss_margin = float(
            target_gap - rss_safe_distance_np(np.asarray([v_ego]), np.asarray([v_lead]), rss_cfg)[0]
        )
        a_ego = float(np.clip(last[0, 4] + rng.normal(0.0, 0.05), accel_lo, accel_hi))
        a_lead = float(np.clip(last[1, 4] + rng.normal(0.0, 0.05), accel_lo, accel_hi))
        history = _build_history(
            history_steps=history_steps,
            dt=dt,
            ego_length=ego_length,
            adv_length=adv_length,
            target_gap=target_gap,
            v_ego=v_ego,
            v_lead=v_lead,
            a_ego=a_ego,
            a_lead=a_lead,
        )
        ok, max_z = _passes_filter(
            history,
            ego_length=ego_length,
            adv_length=adv_length,
            dt=dt,
            target_gap=target_gap,
            target_ttc=target_ttc,
            target_rss_margin=target_rss_margin,
            initial_gap_min=initial_gap_min,
            support=support,
            stats=stats,
            zscore_limit=float(args.zscore_limit),
        )
        if not ok:
            continue
        synthetic_score = max(0.0, -target_rss_margin) + 1.0 / max(target_ttc, 1e-3) + 1.0 / max(target_gap, 1e-3) + max(
            0.0, v_ego - v_lead
        )
        contexts.append(history)
        ego_lengths.append(ego_length)
        adv_lengths.append(adv_length)
        anchor_indices.append(dataset_idx)
        target_gaps.append(target_gap)
        target_ttcs.append(target_ttc)
        target_rss_margins.append(target_rss_margin)
        criticality_scores.append(float(synthetic_score))
        anchor_criticality_scores.append(float(anchor_scores[pos]))
        max_zscores.append(float(max_z))
        if len(contexts) % 5000 == 0:
            logger.info("accepted %d synthetic contexts after %d attempts", len(contexts), attempt + 1)

    if len(contexts) < int(args.num_contexts):
        logger.warning("Generated %d/%d contexts after %d attempts", len(contexts), int(args.num_contexts), max_attempts)
    if not contexts:
        raise RuntimeError("No valid synthetic contexts were generated; relax filters or inspect tail anchors")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        context_states=np.stack(contexts, axis=0).astype(np.float32),
        ego_length=np.asarray(ego_lengths, dtype=np.float32),
        adv_length=np.asarray(adv_lengths, dtype=np.float32),
        source_type=np.asarray(["evt_synthetic"] * len(contexts)),
        anchor_dataset_index=np.asarray(anchor_indices, dtype=np.int64),
        target_gap=np.asarray(target_gaps, dtype=np.float32),
        target_ttc=np.asarray(target_ttcs, dtype=np.float32),
        target_rss_margin=np.asarray(target_rss_margins, dtype=np.float32),
        criticality_score=np.asarray(criticality_scores, dtype=np.float32),
        anchor_criticality_score=np.asarray(anchor_criticality_scores, dtype=np.float32),
        max_context_zscore=np.asarray(max_zscores, dtype=np.float32),
        split_index=np.full(len(contexts), SPLIT_TO_INDEX[str(args.split)], dtype=np.int64),
    )
    _write_json(
        output_path.with_suffix(".summary.json"),
        {
            "dataset": str(dataset_path),
            "tail_score_path": str(tail_path),
            "rss_config": str(rss_path),
            "output": str(output_path),
            "split": str(args.split),
            "num_contexts": int(len(contexts)),
            "num_requested": int(args.num_contexts),
            "tail_anchor_quantile": float(args.tail_anchor_quantile),
            "tail_anchor_threshold": float(threshold),
            "attempts_budget": int(max_attempts),
            "support_p01_p99": {key: [float(v[0]), float(v[1])] for key, v in support.items()},
            "target_gap_mean": float(np.mean(target_gaps)),
            "target_ttc_mean": float(np.mean(target_ttcs)),
            "target_rss_margin_mean": float(np.mean(target_rss_margins)),
            "max_zscore_p95": float(np.percentile(max_zscores, 95.0)),
        },
    )
    logger.info("Wrote %d EVT synthetic contexts to %s", len(contexts), output_path)


if __name__ == "__main__":
    main()
