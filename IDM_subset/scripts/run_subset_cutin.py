#!/usr/bin/env python3
"""Run latent subset simulation for cut-in events."""
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
from IDM_subset.src.latent_subset_runner import run_subset_from_config


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = (
    ROOT / "IDM_subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to cut-in latent subset config.",
    )
    args = parser.parse_args()
    setup_logging("INFO")
    config_path = Path(args.config).resolve()
    summary_path = run_subset_from_config(
        load_yaml(config_path),
        config_path.parent,
        expected_event_type="cut_in",
    )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    counts = dict(summary.get("simulation_counts", {}) or {})
    input_space = dict(summary.get("input_space", {}) or {})
    reliability = dict(summary.get("reliability", {}) or {})
    logger.info(
        (
            "Cut-in subset summary: probability=%.8g se=%.3g "
            "levels=%s stop_reason=%s reliability=%s"
        ),
        float(summary.get("probability", float("nan"))),
        float(summary.get("probability_standard_error", float("nan"))),
        summary.get("num_levels"),
        summary.get("stop_reason"),
        reliability.get("status"),
    )
    logger.info(
        (
            "Cut-in subset actual simulated scenario count: "
            "closed_loop_evaluations=%s stored_level_samples=%s "
            "unique_context_indices_all_levels=%s "
            "unique_context_indices_final_level=%s"
        ),
        counts.get("closed_loop_evaluations"),
        counts.get("stored_level_samples"),
        counts.get("unique_context_indices_all_levels"),
        counts.get("unique_context_indices_final_level"),
    )
    logger.info(
        (
            "Cut-in subset input space: scenario_condition_dim=%s "
            "diffusion_noise_shape=%s diffusion_noise_dim=%s "
            "joint_dim=%s"
        ),
        input_space.get("scenario_condition_dimension"),
        input_space.get("diffusion_noise_shape"),
        input_space.get("diffusion_noise_dimension"),
        input_space.get("joint_condition_noise_dimension"),
    )


if __name__ == "__main__":
    main()
