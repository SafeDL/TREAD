#!/usr/bin/env python3
"""Sample frozen-prior plans and optimize them with KING-style gradients."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner  # noqa: E402
from adversaray.src.config_utils import apply_rss_config_override  # noqa: E402
from adversaray.src.king_gradient_guidance import optimize_action_plan_king  # noqa: E402
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler  # noqa: E402
from adversaray.src.prior_guided_train import _batch_observation_for_contexts, _context, _load_npz  # noqa: E402
from diffusion.src.data import SPLIT_TO_INDEX  # noqa: E402
from diffusion.src.utils import load_yaml, setup_logging  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _append_tensor(out: dict[str, list[np.ndarray]], key: str, value: torch.Tensor) -> None:
    out.setdefault(key, []).append(_tensor_to_numpy(value))


def _select_raw_contexts(args: argparse.Namespace, cfg: dict[str, Any], base: Path) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    if str(args.synthetic_context_path or "").strip():
        path = _resolve(args.synthetic_context_path, base)
        raw = _load_npz(path)
        if "context_states" not in raw:
            raise KeyError(f"{path} must contain context_states")
        return raw, np.arange(raw["context_states"].shape[0], dtype=np.int64), "synthetic"

    paths = cfg.get("paths", {})
    natural_dir = _resolve(paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following"), base)
    raw = _load_npz(natural_dir / "dataset.npz")
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[args.split])[0].astype(np.int64)
    return raw, idx, args.split


def _king_proxy_config(sampler: PriorGuidedDiffusionSampler, config: dict[str, Any]) -> dict[str, Any]:
    proxy_config = copy.deepcopy(sampler.prior.config)
    proxy_config["king_gradient"] = copy.deepcopy(config.get("king_gradient", {}))
    proxy_config["rss"] = copy.deepcopy(config.get("rss", {}))
    proxy_config["physics"] = copy.deepcopy(config.get("physics", {}))
    return proxy_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-contexts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-context-path", default="", help="Optional synthetic_tail_contexts.npz input.")
    parser.add_argument("--output", default="", help="Optional output npz path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    apply_rss_config_override(cfg, cfg_path.parent)
    base = cfg_path.parent
    paths = cfg.get("paths", {})
    output_dir = _resolve(paths.get("output_dir", "../../../data/adversaray/following/prior_guided"), base)
    output_path = _resolve(args.output, base) if str(args.output or "").strip() else output_dir / "king_guided_samples.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw, idx, source_name = _select_raw_contexts(args, cfg, base)
    max_contexts = min(int(args.num_contexts), int(idx.size))
    selected = idx[:max_contexts]
    if max_contexts <= 0:
        raise ValueError("No contexts selected for KING sampling")

    prior_cfg = copy.deepcopy(cfg)
    prior_cfg.setdefault("policy", {})["enabled"] = False
    prior_cfg.setdefault("paths", {})["policy_checkpoint"] = ""
    sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()
    runner = ClosedLoopFollowingRunner(sampler, prior_cfg)
    device = sampler.prior.device
    proxy_config = _king_proxy_config(sampler, cfg)

    output: dict[str, list[np.ndarray]] = {
        "context_states": [],
        "ego_length": [],
        "adv_length": [],
        "prior_actions": [],
        "king_actions": [],
    }
    scalar_keys = (
        "risk_before",
        "risk_after",
        "rss_before",
        "rss_after",
        "ttc_before",
        "ttc_after",
        "gap_before",
        "gap_after",
        "naturalness_penalty",
        "physics_penalty",
        "rss_objective",
        "ttc_objective",
        "drac_objective",
        "gap_objective",
        "min_rss_margin",
        "min_ttc",
        "min_gap",
        "prior_rss_objective",
        "prior_ttc_objective",
        "prior_drac_objective",
        "prior_gap_objective",
        "prior_min_rss_margin",
        "prior_min_ttc",
        "prior_min_gap",
    )

    batch_size = max(int(args.batch_size), 1)
    for start in range(0, max_contexts, batch_size):
        batch_indices = selected[start : start + batch_size]
        contexts = [_context(raw, int(item)) for item in batch_indices]
        batch, prepared_contexts = _batch_observation_for_contexts(runner, contexts)
        seeds = [int(args.seed) + start + pos for pos in range(len(prepared_contexts))]
        with torch.no_grad():
            prior_sample = sampler.sample_batch(batch, seed=seeds)
            raw_context = sampler.prior.decode_context_states(batch["context_states"].to(device).float())

        result = optimize_action_plan_king(
            prior_sample.raw_actions.to(device),
            raw_context,
            batch.get("ego_length"),
            batch.get("adv_length"),
            sampler.prior.schema,
            proxy_config,
        )

        output["context_states"].append(np.stack([ctx["raw_context_states"] for ctx in prepared_contexts], axis=0).astype(np.float32))
        output["ego_length"].append(np.asarray([ctx.get("ego_length", 4.8) for ctx in prepared_contexts], dtype=np.float32))
        output["adv_length"].append(np.asarray([ctx.get("adv_length", 4.8) for ctx in prepared_contexts], dtype=np.float32))
        output["prior_actions"].append(_tensor_to_numpy(result["prior_actions"]))
        output["king_actions"].append(_tensor_to_numpy(result["adv_actions"]))
        for key in scalar_keys:
            if key in result:
                _append_tensor(output, key, result[key])

        logger.info(
            "KING batch %d-%d/%d risk %.4f -> %.4f",
            start + 1,
            start + len(prepared_contexts),
            max_contexts,
            float(result["risk_before"].detach().mean().cpu()),
            float(result["risk_after"].detach().mean().cpu()),
        )

    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in output.items()}
    rename = {
        "rss_objective": "rss_objective_after",
        "ttc_objective": "ttc_objective_after",
        "drac_objective": "drac_objective_after",
        "gap_objective": "gap_objective_after",
        "min_rss_margin": "min_rss_margin_after",
        "min_ttc": "min_ttc_after",
        "min_gap": "min_gap_after",
        "prior_rss_objective": "rss_objective_before",
        "prior_ttc_objective": "ttc_objective_before",
        "prior_drac_objective": "drac_objective_before",
        "prior_gap_objective": "gap_objective_before",
        "prior_min_rss_margin": "min_rss_margin_before",
        "prior_min_ttc": "min_ttc_before",
        "prior_min_gap": "min_gap_before",
    }
    arrays = {rename.get(key, key): value for key, value in arrays.items()}
    arrays["dataset_index"] = selected.astype(np.int64)
    arrays["source_name"] = np.asarray([source_name] * max_contexts)
    np.savez_compressed(output_path, **arrays)
    logger.info("Saved KING-guided samples to %s", output_path)


if __name__ == "__main__":
    main()
