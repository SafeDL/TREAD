#!/usr/bin/env python3
"""Build the reusable Stage 1 shared proposal scenario bank."""
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
from adversaray.src.ego_surrogate import sample_idm_surrogate_params
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler
from adversaray.src.prior_guided_train import _batch_observation_for_contexts, _context, _load_npz
from adversaray.src.risk_utils import write_json
from adversaray.src.shared_proposal_policy import (
    SharedProposalPolicy,
    SharedProposalPolicyConfig,
    prior_action_summary,
    template_params_to_tensor,
    template_to_jerk_delta,
)
from adversaray.src.stage1_shared_utils import (
    classify_risk_types,
    risk_type_names,
    risk_type_summary,
    rollout_proxy_diagnostics,
    template_diversity_summary,
    tensor_stats,
)
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _stage1_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config.get("stage1_shared", {}))
    cfg.setdefault("output_dir", "../../../data/adversaray/following/stage1_shared")
    cfg.setdefault("num_prior_samples_per_context", 2)
    cfg.setdefault("num_latents_per_context", 4)
    cfg.setdefault("num_surrogate_samples_per_context", 8)
    cfg.setdefault("ego_surrogate", {})
    cfg["ego_surrogate"].setdefault("type", "idm")
    cfg.setdefault("scenario_bank", {})
    bank = cfg["scenario_bank"]
    bank.setdefault("split", "val")
    bank.setdefault("num_contexts", 256)
    bank.setdefault("batch_size", 16)
    bank.setdefault("output_name", "scenario_bank.npz")
    return cfg


def _proxy_config(prior_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(prior_config)
    out["king_gradient"] = copy.deepcopy(config.get("king_gradient", {}))
    out["rss"] = copy.deepcopy(config.get("rss", {}))
    out["physics"] = copy.deepcopy(config.get("physics", {}))
    out["stage1_shared"] = copy.deepcopy(config.get("stage1_shared", {}))
    return out


def _split_indices(raw: dict[str, np.ndarray], split: str) -> np.ndarray:
    if "split_index" not in raw:
        raise KeyError("Synthetic contexts must contain split_index")
    idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[split])[0].astype(np.int64)
    if idx.size == 0:
        raise RuntimeError(f"No synthetic contexts found for split '{split}'")
    return idx


def _prepare_prior_sampler(config: dict[str, Any], base: Path) -> PriorGuidedDiffusionSampler:
    prior_cfg = copy.deepcopy(config)
    prior_cfg.setdefault("policy", {})["enabled"] = False
    prior_cfg.setdefault("paths", {})["policy_checkpoint"] = ""
    return PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()


def _sample_latents(count: int, latent_dim: int, *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randn((int(count), int(latent_dim)), device=device, generator=generator)


def _tensor(value: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def _append(output: dict[str, list[np.ndarray]], key: str, value: np.ndarray) -> None:
    output.setdefault(key, []).append(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    apply_rss_config_override(cfg, cfg_path.parent)
    base = cfg_path.parent
    stage1 = _stage1_cfg(cfg)
    cfg["stage1_shared"] = stage1
    bank_cfg = stage1["scenario_bank"]
    seed = int(cfg.get("training", {}).get("seed", 42))
    synthetic_path = _resolve(cfg.get("training", {}).get("synthetic_context_path", ""), base)
    raw = _load_npz(synthetic_path)
    split = str(bank_cfg.get("split", "val"))
    idx = _split_indices(raw, split)[: max(int(bank_cfg.get("num_contexts", 256)), 1)]

    sampler = _prepare_prior_sampler(cfg, base)
    runner = ClosedLoopFollowingRunner(sampler, cfg)
    proxy_config = _proxy_config(sampler.prior.config, cfg)
    output_dir = _resolve(stage1["output_dir"], base)
    ckpt_path = output_dir / "checkpoints" / "best_shared_proposal.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Shared proposal checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=sampler.prior.device)
    policy_cfg = SharedProposalPolicyConfig(**state["policy_config"])
    policy = SharedProposalPolicy(policy_cfg).to(sampler.prior.device)
    policy.load_state_dict(state["policy_state"])
    policy.eval()

    num_prior = max(int(stage1.get("num_prior_samples_per_context", 2)), 1)
    num_surrogate = max(int(stage1.get("num_surrogate_samples_per_context", 8)), 1)
    num_latent = max(int(stage1.get("num_latents_per_context", 4)), 1)
    candidate_repeat = num_surrogate * num_latent
    batch_size = max(int(bank_cfg.get("batch_size", 16)), 1)
    arrays: dict[str, list[np.ndarray]] = {}

    for batch_id, start in enumerate(range(0, len(idx), batch_size)):
        batch_indices = idx[start : start + batch_size]
        contexts = [_context(raw, int(item)) for item in batch_indices]
        batch, prepared_contexts = _batch_observation_for_contexts(runner, contexts)
        device = sampler.prior.device
        seeds = [
            seed + start * num_prior + context_pos * num_prior + prior_pos
            for context_pos in range(len(prepared_contexts))
            for prior_pos in range(num_prior)
        ]
        with torch.no_grad():
            prior_sample = sampler.sample_batch(batch, num_samples=num_prior, seed=seeds)
            context_states = batch["context_states"].to(device).float().repeat_interleave(num_prior, dim=0)
            raw_context = sampler.prior.decode_context_states(context_states)
            prior_actions = prior_sample.raw_actions.to(device)
            base_count = prior_actions.shape[0]
            total = base_count * candidate_repeat
            idm_params = sample_idm_surrogate_params(
                stage1,
                batch_size=base_count,
                num_samples=candidate_repeat,
                device=device,
                dtype=prior_actions.dtype,
                flatten=True,
            )
            latents = _sample_latents(total, policy_cfg.latent_dim, device=device, seed=seed + 500000 + batch_id)
            prior_exp = prior_actions.repeat_interleave(candidate_repeat, dim=0)
            raw_context_exp = raw_context.repeat_interleave(candidate_repeat, dim=0)
            ego_length_exp = batch["ego_length"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0)
            adv_length_exp = batch["adv_length"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0)
            params = policy(
                batch["context_features"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0),
                batch["relative_history"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0),
                prior_action_summary(prior_exp),
                idm_params,
                latents,
            )
            delta = template_to_jerk_delta(params, horizon=prior_exp.shape[1], action_dim=prior_exp.shape[2])
            shared = prior_exp + delta
            prior_diag = rollout_proxy_diagnostics(
                prior_exp,
                raw_context_exp,
                ego_length_exp,
                adv_length_exp,
                sampler.prior.schema,
                proxy_config,
                ego_surrogate_params=idm_params,
            )
            shared_diag = rollout_proxy_diagnostics(
                shared,
                raw_context_exp,
                ego_length_exp,
                adv_length_exp,
                sampler.prior.schema,
                proxy_config,
                ego_surrogate_params=idm_params,
            )
            risk_type = classify_risk_types(shared_diag)
            naturalness = delta.square().flatten(1).mean(dim=1)

        context_np = np.stack([ctx["raw_context_states"] for ctx in prepared_contexts], axis=0).astype(np.float32)
        context_np = np.repeat(context_np, num_prior * candidate_repeat, axis=0)
        dataset_index = np.repeat(batch_indices, num_prior * candidate_repeat).astype(np.int64)
        split_index = np.full(total, SPLIT_TO_INDEX[split], dtype=np.int64)
        _append(arrays, "context_states", context_np)
        _append(arrays, "ego_length", _tensor(ego_length_exp))
        _append(arrays, "adv_length", _tensor(adv_length_exp))
        _append(arrays, "dataset_index", dataset_index)
        _append(arrays, "split_index", split_index)
        _append(arrays, "prior_actions", _tensor(prior_exp))
        _append(arrays, "shared_actions", _tensor(shared))
        _append(arrays, "delta_actions", _tensor(delta))
        _append(arrays, "ego_surrogate_params", _tensor(idm_params.to_feature_tensor()))
        _append(arrays, "latent_z", _tensor(latents))
        template_np = _tensor(template_params_to_tensor(params))
        _append(arrays, "template_params", template_np)
        for key, value in {
            "proxy_risk_before": prior_diag["risk_objective"],
            "proxy_risk_after": shared_diag["risk_objective"],
            "proxy_risk_delta": shared_diag["risk_objective"] - prior_diag["risk_objective"],
            "min_gap_before": prior_diag["min_gap"],
            "min_gap_after": shared_diag["min_gap"],
            "min_ttc_before": prior_diag["min_ttc"],
            "min_ttc_after": shared_diag["min_ttc"],
            "min_rss_margin_before": prior_diag["min_rss_margin"],
            "min_rss_margin_after": shared_diag["min_rss_margin"],
            "drac_before": prior_diag["drac"],
            "drac_after": shared_diag["drac"],
            "naturalness_penalty": naturalness,
            "physics_penalty": shared_diag["physics_penalty"],
            "risk_type_id": risk_type,
        }.items():
            _append(arrays, key, _tensor(value, np.int64 if key == "risk_type_id" else np.float32))
        _append(arrays, "risk_type", risk_type_names(risk_type))
        logger.info("bank batch %d-%d/%d", start + 1, start + len(batch_indices), len(idx))

    final_arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in arrays.items()}
    output_path = output_dir / str(bank_cfg.get("output_name", "scenario_bank.npz"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **final_arrays)
    summary = {
        "split": split,
        "num_source_contexts": int(len(idx)),
        "num_scenarios": int(final_arrays["shared_actions"].shape[0]),
        "output_path": str(output_path),
        **tensor_stats(final_arrays["proxy_risk_delta"], "proxy_risk_delta"),
        **tensor_stats(final_arrays["min_gap_after"], "min_gap_after"),
        **tensor_stats(final_arrays["min_ttc_after"], "min_ttc_after"),
        **tensor_stats(final_arrays["min_rss_margin_after"], "min_rss_margin_after"),
        **risk_type_summary(final_arrays["risk_type_id"]),
        **template_diversity_summary(final_arrays["template_params"]),
    }
    write_json(output_dir / "scenario_bank_summary.json", summary)
    logger.info("Saved Stage 1 scenario bank to %s", output_path)


if __name__ == "__main__":
    main()
