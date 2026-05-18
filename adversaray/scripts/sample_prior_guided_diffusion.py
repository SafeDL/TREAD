#!/usr/bin/env python3
"""Open-loop sample futures from the prior-guided diffusion policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler, result_to_numpy
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_json, load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--policy-checkpoint", default="", help="Optional policy checkpoint override.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-contexts", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=1)
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
    norm = _load_npz(natural_dir / "dataset_normalized.npz")
    raw = _load_npz(natural_dir / "dataset.npz")
    idx_all = np.where(norm["split_index"] == SPLIT_TO_INDEX[args.split])[0]
    idx = idx_all[: int(args.num_contexts)]
    if len(idx) == 0:
        raise RuntimeError(f"No contexts found for split={args.split}")
    sampler = PriorGuidedDiffusionSampler.from_config(cfg, config_dir=base).eval()
    result = sampler.sample(
        torch.from_numpy(norm["context_states"][idx]).float(),
        torch.from_numpy(norm["context_features"][idx]).float(),
        torch.from_numpy(norm["relative_history"][idx]).float(),
        ego_length=torch.from_numpy(raw["ego_length"][idx]).float(),
        adv_length=torch.from_numpy(raw["adv_length"][idx]).float(),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
    )
    arrays = result_to_numpy(result)
    repeated_idx = np.repeat(idx.astype(np.int64), int(args.num_samples))
    out_path = output_dir / "prior_guided_samples.npz"
    np.savez_compressed(out_path, sample_index=repeated_idx, **arrays)
    save_json(
        {
            "path": str(out_path),
            "split": args.split,
            "num_contexts": int(len(idx)),
            "num_samples_per_context": int(args.num_samples),
            "schema": load_json(natural_dir / "feature_schema.json"),
            "guidance_trace": result.guidance_trace,
        },
        output_dir / "prior_guided_samples_summary.json",
    )


if __name__ == "__main__":
    main()
