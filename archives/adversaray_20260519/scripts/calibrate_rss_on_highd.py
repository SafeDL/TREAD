#!/usr/bin/env python3
"""Calibrate RSS parameters on highD recorded futures and optional frozen-prior rollouts."""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.risk_utils import (  # noqa: E402
    interaction_metrics_from_states,
    rss_config_dict,
    safe_corr,
    write_csv,
    write_json,
    write_simple_yaml,
)
from adversaray.src.rss import RSSConfig  # noqa: E402
from diffusion.src.data import SPLIT_TO_INDEX  # noqa: E402
from diffusion.src.utils import load_json, load_yaml, setup_logging  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    context: dict[str, Any] = {
        "raw_context_states": raw["context_states"][idx],
        "ego_length": float(raw["ego_length"][idx]) if "ego_length" in raw else 4.8,
        "adv_length": float(raw["adv_length"][idx]) if "adv_length" in raw else 4.8,
    }
    for key in ("recording_id", "event_id", "anchor_frame"):
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    return context


def _parse_grid(value: str, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _select_split_indices(raw: dict[str, np.ndarray], split_arg: str) -> np.ndarray:
    split_arg = str(split_arg).strip().lower()
    total = raw["future_states"].shape[0]
    if split_arg == "all":
        return np.arange(total, dtype=np.int64)
    if "split_index" not in raw:
        raise RuntimeError("dataset.npz is missing split_index; use --split all only if split leakage is acceptable.")
    names = [item.strip() for item in split_arg.split(",") if item.strip()]
    if not names:
        raise ValueError("--split must be one of train, val, test, a comma-separated combination, or all")
    unknown = [name for name in names if name not in SPLIT_TO_INDEX]
    if unknown:
        raise ValueError(f"Unknown split name(s): {unknown}")
    wanted = np.asarray([SPLIT_TO_INDEX[name] for name in names], dtype=np.int64)
    return np.where(np.isin(raw["split_index"], wanted))[0].astype(np.int64)


def _runtime_paths(cfg: dict[str, Any], base: Path) -> tuple[Path, Path]:
    paths = cfg.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / "../../../data/adversaray/following/rss_calibration").resolve()
    return natural_dir, output_dir


def _rows_for_recorded(raw: dict[str, np.ndarray], idx: np.ndarray, cfg: RSSConfig) -> dict[str, np.ndarray]:
    metrics = interaction_metrics_from_states(
        raw["context_states"][idx],
        raw["future_states"][idx],
        raw["ego_length"][idx] if "ego_length" in raw else np.full(len(idx), 4.8, dtype=np.float32),
        raw["adv_length"][idx] if "adv_length" in raw else np.full(len(idx), 4.8, dtype=np.float32),
        cfg,
    )
    return metrics


def _rollout_frozen_prior(
    config: dict[str, Any],
    config_dir: Path,
    raw: dict[str, np.ndarray],
    idx: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, float]]:
    from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
    from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler

    prior_cfg = copy.deepcopy(config)
    prior_cfg.setdefault("policy", {})["enabled"] = False
    sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=config_dir).eval()
    sampler.set_guidance_enabled(False)
    runner = ClosedLoopFollowingRunner(sampler, prior_cfg)
    rows: list[dict[str, float]] = []
    for offset, dataset_idx in enumerate(idx):
        result = runner.rollout(_context(raw, int(dataset_idx)), seed=int(seed) + offset)
        rows.append(result.metrics)
    return rows


def _summarize(
    cfg: RSSConfig,
    recorded: dict[str, np.ndarray],
    *,
    prior_rows: list[dict[str, float]] | None = None,
    severe_margin: float,
) -> dict[str, float]:
    min_rss = np.asarray(recorded["min_rss_margin"], dtype=np.float64)
    min_ttc = np.asarray(recorded["min_ttc"], dtype=np.float64)
    min_gap = np.asarray(recorded["min_gap"], dtype=np.float64)
    row: dict[str, float] = {
        **rss_config_dict(cfg),
        "rss_violation_rate": float(np.mean(min_rss < 0.0)),
        "severe_rss_violation_rate": float(np.mean(min_rss < severe_margin)),
        "min_rss_margin_mean": float(np.mean(min_rss)),
        "min_rss_margin_p05": float(np.percentile(min_rss, 5.0)),
        "min_rss_margin_p50": float(np.percentile(min_rss, 50.0)),
        "min_rss_margin_p95": float(np.percentile(min_rss, 95.0)),
        "corr_min_rss_margin_min_ttc": safe_corr(min_rss, min_ttc),
        "corr_min_rss_margin_min_gap": safe_corr(min_rss, min_gap),
        "recorded_highd_false_positive_rate": float(np.mean(min_rss < severe_margin)),
    }
    if prior_rows:
        prior_min_rss = np.asarray([r.get("min_rss_margin", np.nan) for r in prior_rows], dtype=np.float64)
        prior_min_ttc = np.asarray([r.get("min_ttc", np.nan) for r in prior_rows], dtype=np.float64)
        prior_min_gap = np.asarray([r.get("min_gap", np.nan) for r in prior_rows], dtype=np.float64)
        row.update(
            {
                "prior_rollout_rss_violation_rate": float(np.nanmean(prior_min_rss < 0.0)),
                "prior_rollout_severe_rss_violation_rate": float(np.nanmean(prior_min_rss < severe_margin)),
                "prior_rollout_min_rss_margin_mean": float(np.nanmean(prior_min_rss)),
                "prior_rollout_corr_min_rss_margin_min_ttc": safe_corr(prior_min_rss, prior_min_ttc),
                "prior_rollout_corr_min_rss_margin_min_gap": safe_corr(prior_min_rss, prior_min_gap),
            }
        )
    return row


def _recommend(rows: list[dict[str, float]]) -> dict[str, float]:
    def score(row: dict[str, float]) -> tuple[float, float, float, float]:
        severe = float(row.get("severe_rss_violation_rate", 1.0))
        violation = float(row.get("rss_violation_rate", 1.0))
        corr_gap = abs(float(row.get("corr_min_rss_margin_min_gap", 0.0) or 0.0))
        corr_ttc = abs(float(row.get("corr_min_rss_margin_min_ttc", 0.0) or 0.0))
        return (severe, abs(violation - 0.35), -(corr_gap + corr_ttc), -float(row.get("min_rss_margin_p05", -1e9)))

    return min(rows, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", default="train,val", help="Split(s) used for calibration recommendation: train, val, test, train,val, or all.")
    parser.add_argument("--max-contexts", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severe-margin", type=float, default=-10.0)
    parser.add_argument("--prior-rollouts", type=int, default=0)
    parser.add_argument("--response-time", default="")
    parser.add_argument("--ego-max-accel", default="")
    parser.add_argument("--ego-min-brake", default="")
    parser.add_argument("--lead-max-brake", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    natural_dir, default_output_dir = _runtime_paths(cfg, cfg_path.parent)
    dataset_path = Path(args.dataset).resolve() if args.dataset else natural_dir / "dataset.npz"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    raw = _load_npz(dataset_path)
    _ = load_json(natural_dir / "feature_schema.json")
    if "future_states" not in raw:
        raise RuntimeError("dataset.npz is missing future_states; RSS calibration needs recorded futures.")
    rng = np.random.default_rng(int(args.seed))
    all_idx = _select_split_indices(raw, str(args.split))
    if len(all_idx) == 0:
        raise RuntimeError(f"No contexts found for split selection {args.split!r}")
    if int(args.max_contexts) > 0 and len(all_idx) > int(args.max_contexts):
        all_idx = np.sort(rng.choice(all_idx, size=int(args.max_contexts), replace=False)).astype(np.int64)

    grid = {
        "response_time": _parse_grid(args.response_time, [0.3, 0.5, 0.7, 1.0]),
        "ego_max_accel": _parse_grid(args.ego_max_accel, [0.5, 1.0, 2.0]),
        "ego_min_brake": _parse_grid(args.ego_min_brake, [3.0, 4.0, 6.0, 8.0]),
        "lead_max_brake": _parse_grid(args.lead_max_brake, [3.0, 4.0, 6.0, 8.0]),
    }
    prior_idx = all_idx[: max(0, int(args.prior_rollouts))]
    rows: list[dict[str, float]] = []
    for values in itertools.product(*grid.values()):
        rss_values = dict(zip(grid.keys(), values))
        rss_cfg = RSSConfig(**rss_values, temperature=float(cfg.get("rss", {}).get("temperature", 1.0)), pool_beta=float(cfg.get("rss", {}).get("pool_beta", 8.0)))
        recorded = _rows_for_recorded(raw, all_idx, rss_cfg)
        prior_rows = None
        if len(prior_idx) > 0:
            rollout_cfg = copy.deepcopy(cfg)
            rollout_cfg["rss"] = rss_config_dict(rss_cfg)
            prior_rows = _rollout_frozen_prior(rollout_cfg, cfg_path.parent, raw, prior_idx, seed=int(args.seed) + 10000)
        rows.append(_summarize(rss_cfg, recorded, prior_rows=prior_rows, severe_margin=float(args.severe_margin)))

    recommended = _recommend(rows)
    recommended_cfg = {key: recommended[key] for key in ("response_time", "ego_max_accel", "ego_min_brake", "lead_max_brake")}
    recommended_cfg["temperature"] = float(cfg.get("rss", {}).get("temperature", 1.0))
    recommended_cfg["pool_beta"] = float(cfg.get("rss", {}).get("pool_beta", 8.0))
    write_csv(output_dir / "rss_parameter_grid.csv", rows)
    write_json(
        output_dir / "rss_calibration_summary.json",
        {
            "dataset": str(dataset_path),
            "split": str(args.split),
            "num_contexts": int(len(all_idx)),
            "severe_margin": float(args.severe_margin),
            "recommended_rss_config": recommended_cfg,
            "recommended_row": recommended,
            "grid_size": int(len(rows)),
        },
    )
    write_simple_yaml(output_dir / "recommended_rss_config.yaml", {"rss": recommended_cfg})


if __name__ == "__main__":
    main()
