#!/usr/bin/env python3
"""Run an offline rolling-actor smoke test on Stage 1 contexts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.guided_sampler import GuidedDiffusionSampler
from adversaray.src.rolling_actor import RollingGuidedDiffusionActor
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "guided_sampling_following.yaml"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    natural_dir = (base / cfg.get("paths", {}).get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / cfg.get("paths", {}).get("output_dir", "../../../data/adversaray/following/guided_sampling")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    norm = _load_npz(natural_dir / "dataset_normalized.npz")
    raw = _load_npz(natural_dir / "dataset.npz")
    idx = np.where(norm["split_index"] == SPLIT_TO_INDEX[args.split])[0]
    if len(idx) == 0:
        raise RuntimeError(f"No contexts found for split={args.split}")
    sampler = GuidedDiffusionSampler.from_config(cfg, config_dir=base)
    actor = RollingGuidedDiffusionActor(sampler, cfg.get("rolling_actor", {}))
    actions: list[np.ndarray] = []
    for i in range(int(args.steps)):
        sample_idx = int(idx[min(i, len(idx) - 1)])
        obs = {
            "context_states": norm["context_states"][sample_idx],
            "context_features": norm["context_features"][sample_idx],
            "relative_history": norm["relative_history"][sample_idx],
            "ego_length": float(raw["ego_length"][sample_idx]),
            "adv_length": float(raw["adv_length"][sample_idx]),
        }
        actions.append(actor.step(obs))
    summary = actor.summary()
    summary["num_actions"] = int(len(actions))
    summary["actions_shape"] = list(np.asarray(actions, dtype=np.float32).shape)
    save_json(summary, output_dir / "rolling_actor_summary.json")


if __name__ == "__main__":
    main()

