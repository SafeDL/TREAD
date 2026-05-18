#!/usr/bin/env python3
"""Evaluate a prior-guided policy in closed-loop highway-env rollouts."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler
from adversaray.src.prior_guided_train import evaluate_prior_guided_policy
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _attach_runtime_paths(cfg: dict, base: Path) -> None:
    paths = cfg.get("paths", {})
    cfg["_runtime"] = {
        "config_dir": str(base),
        "natural_dataset_dir": str((base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()),
        "output_dir": str((base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()),
        "highd_events_csv": str((base / paths.get("highd_events_csv", "../../../data/highd_events/events.csv")).resolve()),
        "highd_raw_dir": str((base / paths.get("highd_raw_dir", "../../../highD_dataset/Matlab/data")).resolve()),
        "highd_config": str(
            (base / paths.get("highd_config", "../../../process_highD/scripts/configs/highd_default.yaml")).resolve()
        ),
    }


def _comparison_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    mapping = {
        "collision_rate": "collision_mean",
        "min_ttc_mean": "min_ttc_mean",
        "min_gap_mean": "min_gap_mean",
        "min_rss_margin_mean": "min_rss_margin_mean",
    }
    return {f"{prefix}_{out_key}": float(metrics.get(in_key, float("nan"))) for out_key, in_key in mapping.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--policy-checkpoint", default="", help="Optional policy checkpoint override.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-contexts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-guidance", action="store_true", help="Evaluate the frozen diffusion prior only.")
    parser.add_argument("--compare-frozen-prior", action="store_true", help="Evaluate frozen prior and guided policy on the same contexts.")
    parser.add_argument("--commit-steps", type=int, default=1, help="Evaluation replan cadence override.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    if args.policy_checkpoint:
        cfg.setdefault("paths", {})["policy_checkpoint"] = args.policy_checkpoint
    cfg.setdefault("env", {})["commit_steps_max"] = int(args.commit_steps)
    base = cfg_path.parent
    _attach_runtime_paths(cfg, base)
    paths = cfg.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_npz(natural_dir / "dataset.npz")
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[args.split])[0]
    if args.compare_frozen_prior:
        prior_cfg = copy.deepcopy(cfg)
        prior_cfg.setdefault("policy", {})["enabled"] = False
        prior_sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()
        prior_metrics = evaluate_prior_guided_policy(
            prior_sampler,
            prior_cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
        )
        guided_sampler = PriorGuidedDiffusionSampler.from_config(cfg, config_dir=base).eval()
        guided_metrics = evaluate_prior_guided_policy(
            guided_sampler,
            cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
        )
        metrics = {
            **_comparison_metrics("prior", prior_metrics),
            **_comparison_metrics("guided", guided_metrics),
            "prior_kl_mean": float(guided_metrics.get("prior_kl_mean", float("nan"))),
            "guidance_norm_mean": float(guided_metrics.get("guidance_norm_mean", float("nan"))),
            "prior_raw": prior_metrics,
            "guided_raw": guided_metrics,
        }
    else:
        if args.disable_guidance:
            cfg.setdefault("policy", {})["enabled"] = False
        sampler = PriorGuidedDiffusionSampler.from_config(cfg, config_dir=base).eval()
        metrics = evaluate_prior_guided_policy(
            sampler,
            cfg,
            raw,
            idx,
            max_contexts=int(args.num_contexts),
            seed=int(args.seed),
        )
    save_json(
        {
            "split": args.split,
            "mode": "compare" if args.compare_frozen_prior else ("frozen_prior" if args.disable_guidance else "guided"),
            "num_contexts": int(min(len(idx), int(args.num_contexts))),
            "metrics": metrics,
        },
        output_dir / "prior_guided_eval_summary.json",
    )


if __name__ == "__main__":
    main()
