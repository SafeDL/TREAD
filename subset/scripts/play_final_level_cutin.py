#!/usr/bin/env python3
"""Replay final-level cut-in subset scenarios."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from subset.src import final_level_playback as playback


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = (
    ROOT / "subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--samples-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Maximum final-level failure test scenarios to render.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random final-level failure case selection.",
    )
    parser.add_argument("--level", type=int, default=-1)
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Render overview PNGs only.",
    )
    parser.add_argument(
        "--background-config",
        default=None,
        help="Path to process_highD config used to replay background traffic.",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Do not overlay highD background traffic in GIF playback.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    playback.SCRIPT_DEFAULTS.update(
        {
            "samples_path": args.samples_path,
            "output_dir": args.output_dir,
            "num_cases": int(args.num_cases),
            "random_seed": int(args.random_seed),
            "level": int(args.level),
            "unique_test_scenarios": True,
            "render_gif": not bool(args.no_gif),
            "render_background": not bool(args.no_background),
            "log_level": str(args.log_level),
        }
    )
    if args.background_config:
        playback.SCRIPT_DEFAULTS["background_config_path"] = args.background_config
    setup_logging(str(playback.SCRIPT_DEFAULTS["log_level"]))
    config_path = Path(args.config).resolve()
    manifest_path = playback.replay_final_level(
        load_yaml(config_path),
        config_path.parent,
        expected_event_type="cut_in",
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info(
        "Cut-in final-level playback manifest: %s cases=%s level=%s threshold=%.6g",
        manifest_path,
        manifest.get("num_cases"),
        manifest.get("level"),
        float(manifest.get("failure_threshold", float("nan"))),
    )


if __name__ == "__main__":
    main()
