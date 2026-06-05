#!/usr/bin/env python3
"""Replay final-level car-following subset scenarios."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from subset.src import final_level_playback as playback


DEFAULT_CONFIG_PATH = (
    ROOT / "subset" / "scripts" / "configs" / "latent_subset_following.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--samples-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--level", type=int, default=-1)
    args = parser.parse_args()
    playback.SCRIPT_DEFAULTS.update(
        {
            "samples_path": args.samples_path,
            "output_dir": args.output_dir,
            "num_cases": int(args.num_cases),
            "level": int(args.level),
        }
    )
    setup_logging(str(playback.SCRIPT_DEFAULTS["log_level"]))
    config_path = Path(args.config).resolve()
    playback.replay_final_level(
        load_yaml(config_path),
        config_path.parent,
        expected_event_type="following",
    )


if __name__ == "__main__":
    main()
