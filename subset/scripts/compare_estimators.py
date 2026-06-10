#!/usr/bin/env python3
"""Compare Monte Carlo and latent subset probability estimates."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from subset.src.latent_subset_runner import compare_monte_carlo_subset_from_config


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = (
    ROOT / "subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def _expected_event_type(config: dict) -> str | None:
    configured = str(config.get("event", {}).get("event_type", "")).strip()
    if configured:
        return configured
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Path to latent subset config. Defaults to latent_subset_cutin.yaml; "
            "pass latent_subset_following.yaml for car-following."
        ),
    )
    args = parser.parse_args()
    setup_logging("INFO")
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    expected_event_type = _expected_event_type(config)
    logger.info(
        "Comparing Monte Carlo and subset estimates config=%s event_type=%s",
        config_path,
        expected_event_type or "not-enforced",
    )
    compare_monte_carlo_subset_from_config(
        config,
        config_path.parent,
        expected_event_type=expected_event_type,
    )


if __name__ == "__main__":
    main()
