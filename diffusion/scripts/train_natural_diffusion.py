#!/usr/bin/env python3
"""Train the naturalistic car-following action diffusion prior."""
from __future__ import annotations

from pathlib import Path

from diffusion.src.train import train_action_diffusion
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_following.yaml"
SCRIPT_DEFAULTS = {
    "log_level": "INFO",
    "rebuild_dataset": False,
}


def main() -> None:
    setup_logging(str(SCRIPT_DEFAULTS["log_level"]))
    cfg_path = DEFAULT_CONFIG_PATH.resolve()
    config = load_yaml(cfg_path)
    if bool(SCRIPT_DEFAULTS["rebuild_dataset"]):
        config.setdefault("dataset", {})["rebuild"] = True
    train_action_diffusion(config, config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
