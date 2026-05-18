#!/usr/bin/env python3
"""Evaluate a prior-guided policy in closed-loop highway-env rollouts."""
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--policy-checkpoint", default="", help="Optional policy checkpoint override.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-contexts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    if args.policy_checkpoint:
        cfg.setdefault("paths", {})["policy_checkpoint"] = args.policy_checkpoint
    base = cfg_path.parent
    paths = cfg.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_npz(natural_dir / "dataset.npz")
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[args.split])[0]
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
            "num_contexts": int(min(len(idx), int(args.num_contexts))),
            "metrics": metrics,
        },
        output_dir / "prior_guided_eval_summary.json",
    )


if __name__ == "__main__":
    main()
