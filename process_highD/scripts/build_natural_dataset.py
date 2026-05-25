#!/usr/bin/env python3
"""Build the highD car-following natural-prior dataset."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import build_action_dataset
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = (
    ROOT / "diffusion" / "scripts" / "configs" / "natural_following.yaml"
)
SCRIPT_DEFAULTS = {"log_level": "INFO"}


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    build_action_dataset(load_yaml(cfg_path), config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
