#!/usr/bin/env python3
"""Evaluate guided diffusion sample files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.metrics import summarize_guided_arrays
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "guided_sampling_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--samples", default="guided_samples.npz", help="Guided samples path, relative to output_dir if needed.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    output_dir = (cfg_path.parent / cfg.get("paths", {}).get("output_dir", "../../../data/adversaray/following/guided_sampling")).resolve()
    samples = Path(args.samples)
    if not samples.is_absolute():
        samples = output_dir / samples
    data = np.load(samples, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    summary = summarize_guided_arrays(arrays)
    summary["samples"] = str(samples)
    summary["num_samples"] = int(arrays["actions"].shape[0]) if "actions" in arrays else 0
    save_json(summary, output_dir / "guided_samples_eval_summary.json")


if __name__ == "__main__":
    main()

