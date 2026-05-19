#!/usr/bin/env python3
"""Train the prior-regularized guided diffusion policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.prior_guided_train import train_prior_guided_policy
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--epochs", type=int, default=0, help="Optional epoch override for smoke tests.")
    parser.add_argument("--max-train-contexts", type=int, default=0, help="Optional context cap for smoke tests.")
    parser.add_argument("--episode-steps", type=int, default=0, help="Optional rollout horizon override.")
    parser.add_argument("--commit-steps", type=int, default=0, help="Optional plan commit horizon override.")
    parser.add_argument(
        "--rss-config",
        default="",
        help="Optional recommended_rss_config.yaml override. If omitted, paths.rss_config is used when it exists.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    paths = cfg.get("paths", {})
    rss_config_arg = str(args.rss_config).strip()
    rss_config_default = str(paths.get("rss_config", "")).strip()
    rss_path: Path | None = None
    if rss_config_arg:
        rss_path = Path(rss_config_arg)
        rss_path = rss_path if rss_path.is_absolute() else (cfg_path.parent / rss_path)
    elif rss_config_default:
        candidate = Path(rss_config_default)
        candidate = candidate if candidate.is_absolute() else (cfg_path.parent / candidate)
        if candidate.exists():
            rss_path = candidate
    if rss_path is not None:
        rss_path = rss_path.resolve()
        recommended = load_yaml(rss_path)
        if "rss" not in recommended:
            raise KeyError(f"{rss_path} does not contain an 'rss' mapping")
        cfg.setdefault("rss", {}).update(recommended["rss"])
    if args.epochs > 0:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.max_train_contexts > 0:
        cfg.setdefault("training", {})["max_train_contexts"] = int(args.max_train_contexts)
    if args.episode_steps > 0:
        cfg.setdefault("env", {})["episode_steps"] = int(args.episode_steps)
    if args.commit_steps > 0:
        cfg.setdefault("env", {})["commit_steps_max"] = int(args.commit_steps)
    train_prior_guided_policy(cfg, config_dir=cfg_path.parent)


if __name__ == "__main__":
    main()
