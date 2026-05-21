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

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
from adversaray.src.config_utils import apply_rss_config_override
from adversaray.src.king_gradient_guidance import optimize_action_plan_king
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler
from adversaray.src.prior_guided_train import _batch_observation_for_contexts, _context, _load_npz
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
SCRIPT_DEFAULTS = {
    "split": "val",
    "num_contexts": 256,
    "batch_size": 16,
    "seed": 42,
    "output_name": "king_guided_samples.npz",
    "log_level": "INFO",
}
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _append_tensor(out: dict[str, list[np.ndarray]], key: str, value: torch.Tensor) -> None:
    out.setdefault(key, []).append(_tensor_to_numpy(value))


def _split_indices(raw: dict[str, np.ndarray], split: str) -> np.ndarray:
    if "split_index" not in raw:
        raise KeyError("Synthetic contexts must contain split_index; rebuild them with build_evt_synthetic_contexts.py.")
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[split])[0].astype(np.int64)
    if idx.size == 0:
        raise RuntimeError(f"No synthetic contexts found for split '{split}'")
    return idx


def _select_raw_contexts(args: argparse.Namespace, cfg: dict[str, Any], base: Path) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    training = cfg.get("training", {})
    synthetic_value = str(args.synthetic_context_path or training.get("synthetic_context_path", "") or "").strip()
    if not synthetic_value:
        raise ValueError("training.synthetic_context_path must be set for KING open-loop sampling")
    path = _resolve(synthetic_value, base)
    raw = _load_npz(path)
    required = {"context_states", "split_index"}
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")
    return raw, _split_indices(raw, args.split), "synthetic"


def _king_proxy_config(sampler: PriorGuidedDiffusionSampler, config: dict[str, Any]) -> dict[str, Any]:
    proxy_config = copy.deepcopy(sampler.prior.config)
    proxy_config["king_gradient"] = copy.deepcopy(config.get("king_gradient", {}))
    proxy_config["rss"] = copy.deepcopy(config.get("rss", {}))
    proxy_config["physics"] = copy.deepcopy(config.get("physics", {}))
    return proxy_config


def _array_mean(arrays: dict[str, np.ndarray], key: str) -> float:
    value = arrays.get(key)
    if value is None or value.size == 0:
        return float("nan")
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _sample_summary(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    prior_actions = np.asarray(arrays["prior_actions"], dtype=np.float32)
    king_actions = np.asarray(arrays["king_actions"], dtype=np.float32)
    action_l2 = np.sqrt(np.mean(np.square(king_actions - prior_actions), axis=tuple(range(1, king_actions.ndim))))
    risk_before = _array_mean(arrays, "risk_before")
    risk_after = _array_mean(arrays, "risk_after")
    return {
        "risk_before_mean": risk_before,
        "risk_after_mean": risk_after,
        "risk_delta_mean": risk_after - risk_before,
        "gap_before_mean": _array_mean(arrays, "gap_before"),
        "gap_after_mean": _array_mean(arrays, "gap_after"),
        "ttc_before_mean": _array_mean(arrays, "ttc_before"),
        "ttc_after_mean": _array_mean(arrays, "ttc_after"),
        "rss_before_mean": _array_mean(arrays, "rss_before"),
        "rss_after_mean": _array_mean(arrays, "rss_after"),
        "naturalness_penalty_mean": _array_mean(arrays, "naturalness_penalty"),
        "physics_penalty_mean": _array_mean(arrays, "physics_penalty"),
        "action_l2_mean": float(np.mean(action_l2)) if action_l2.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--split", choices=("train", "val", "test"), default=SCRIPT_DEFAULTS["split"])
    parser.add_argument("--num-contexts", type=int, default=SCRIPT_DEFAULTS["num_contexts"])
    parser.add_argument("--batch-size", type=int, default=SCRIPT_DEFAULTS["batch_size"])
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--synthetic-context-path", default="", help="Override training.synthetic_context_path.")
    parser.add_argument("--output", default="", help="Override the default KING-guided samples npz path.")
    parser.add_argument("--log-level", default=SCRIPT_DEFAULTS["log_level"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    apply_rss_config_override(cfg, cfg_path.parent)
    base = cfg_path.parent
    paths = cfg.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    output_dir = _resolve(paths["output_dir"], base)
    output_path = _resolve(args.output, base) if str(args.output or "").strip() else output_dir / str(SCRIPT_DEFAULTS["output_name"])
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
        output["ego_length"].append(np.asarray([ctx["ego_length"] for ctx in prepared_contexts], dtype=np.float32))
        output["adv_length"].append(np.asarray([ctx["adv_length"] for ctx in prepared_contexts], dtype=np.float32))
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
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    save_json(
        {
            "split": args.split,
            "source": source_name,
            "num_contexts": max_contexts,
            "output_path": str(output_path),
            **_sample_summary(arrays),
        },
        summary_path,
    )
    logger.info("Saved KING-guided samples to %s", output_path)
    logger.info("Saved KING-guided sample summary to %s", summary_path)


if __name__ == "__main__":
    main()
