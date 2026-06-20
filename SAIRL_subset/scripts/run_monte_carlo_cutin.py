#!/usr/bin/env python3
"""Run independent latent Monte Carlo baseline for cut-in events."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from SAIRL_subset.src.latent_subset_runner import run_monte_carlo_from_config


DEFAULT_CONFIG_PATH = (
    ROOT / "SAIRL_subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to cut-in latent Monte Carlo config.",
    )
    args = parser.parse_args()
    setup_logging("INFO")
    config_path = Path(args.config).resolve()
    run_monte_carlo_from_config(
        load_yaml(config_path),
        config_path.parent,
        expected_event_type="cut_in",
    )


if __name__ == "__main__":
    main()
