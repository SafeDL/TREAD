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

ROOT = Path(__file__).resolve().parents[3]
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


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "prior_guided_following.yaml"
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
    bank.setdefault("top_k_per_context", 8)
    bank.setdefault("min_proxy_risk_delta", 0.0)
    bank.setdefault("max_physics_penalty", 0.05)
    bank.setdefault("max_naturalness_penalty", 1.0)
    bank.setdefault("require_diverse_risk_types", True)
    return cfg


def _proxy_config(prior_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(prior_config)
    out["proxy_risk"] = copy.deepcopy(config.get("proxy_risk", {}))
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
    return PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=base).eval()


def _sample_latents(count: int, latent_dim: int, *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randn((int(count), int(latent_dim)), device=device, generator=generator)


def _tensor(value: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def _append(output: dict[str, list[np.ndarray]], key: str, value: np.ndarray) -> None:
    output.setdefault(key, []).append(value)


def _select_candidates(
    batch_arrays: dict[str, np.ndarray],
    bank_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_index = np.asarray(batch_arrays["dataset_index"], dtype=np.int64)
    risk_delta = np.asarray(batch_arrays["proxy_risk_delta"], dtype=np.float64)
    physics = np.asarray(batch_arrays["physics_penalty"], dtype=np.float64)
    naturalness = np.asarray(batch_arrays["naturalness_penalty"], dtype=np.float64)
    risk_type = np.asarray(batch_arrays["risk_type_id"], dtype=np.int64)
    valid = (
        (risk_delta >= float(bank_cfg.get("min_proxy_risk_delta", 0.0)))
        & (physics <= float(bank_cfg.get("max_physics_penalty", 0.05)))
        & (naturalness <= float(bank_cfg.get("max_naturalness_penalty", 1.0)))
        & (risk_type > 0)
        & np.isfinite(risk_delta)
        & np.isfinite(physics)
        & np.isfinite(naturalness)
    )
    top_k = max(int(bank_cfg.get("top_k_per_context", 8)), 1)
    require_diverse = bool(bank_cfg.get("require_diverse_risk_types", True))
    selected: list[int] = []
    reasons: dict[int, str] = {}
    ranks: dict[int, int] = {}
    for source_idx in np.unique(dataset_index):
        group = np.where((dataset_index == source_idx) & valid)[0]
        if group.size == 0:
            continue
        ordered = group[np.argsort(-risk_delta[group], kind="stable")]
        chosen: list[int] = []
        if require_diverse:
            for risk_id in sorted(np.unique(risk_type[ordered])):
                type_candidates = ordered[risk_type[ordered] == risk_id]
                if type_candidates.size == 0:
                    continue
                item = int(type_candidates[0])
                chosen.append(item)
                reasons[item] = "diverse_risk_type"
                if len(chosen) >= top_k:
                    break
        for item in ordered:
            if len(chosen) >= top_k:
                break
            item_i = int(item)
            if item_i in chosen:
                continue
            chosen.append(item_i)
            reasons[item_i] = "score_rank"
        chosen = sorted(chosen, key=lambda item: float(-risk_delta[item]))
        for rank, item in enumerate(chosen, start=1):
            ranks[int(item)] = rank
        selected.extend(chosen)
    selected_arr = np.asarray(selected, dtype=np.int64)
    rank_arr = np.asarray([ranks[int(item)] for item in selected_arr], dtype=np.int64)
    reason_arr = np.asarray([reasons[int(item)] for item in selected_arr])
    return selected_arr, rank_arr, reason_arr


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
    generated_total = 0
    accepted_total = 0

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
            idm_base = sample_idm_surrogate_params(
                stage1,
                batch_size=base_count,
                num_samples=num_surrogate,
                device=device,
                dtype=prior_actions.dtype,
                flatten=True,
            )
            idm_params = idm_base.repeat_interleave(num_latent, dim=0)
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
        batch_arrays: dict[str, np.ndarray] = {
            "context_states": context_np,
            "ego_length": _tensor(ego_length_exp),
            "adv_length": _tensor(adv_length_exp),
            "dataset_index": dataset_index,
            "split_index": split_index,
            "prior_actions": _tensor(prior_exp),
            "shared_actions": _tensor(shared),
            "delta_actions": _tensor(delta),
            "ego_surrogate_params": _tensor(idm_params.to_feature_tensor()),
            "latent_z": _tensor(latents),
        }
        template_np = _tensor(template_params_to_tensor(params))
        batch_arrays["template_params"] = template_np
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
            batch_arrays[key] = _tensor(value, np.int64 if key == "risk_type_id" else np.float32)
        batch_arrays["risk_type"] = risk_type_names(risk_type)
        selected, selection_rank, selection_reason = _select_candidates(batch_arrays, bank_cfg)
        generated_total += int(total)
        accepted_total += int(selected.size)
        if selected.size:
            batch_arrays["accepted_mask"] = np.ones(selected.size, dtype=bool)
            batch_arrays["selection_rank"] = selection_rank
            batch_arrays["selection_reason"] = selection_reason
            for key, value in batch_arrays.items():
                if key in {"accepted_mask", "selection_rank", "selection_reason"}:
                    _append(arrays, key, value)
                else:
                    _append(arrays, key, value[selected])
        logger.info(
            "bank batch %d-%d/%d accepted %d/%d",
            start + 1,
            start + len(batch_indices),
            len(idx),
            int(selected.size),
            int(total),
        )

    if not arrays:
        raise RuntimeError("Scenario bank filtering rejected all candidates; relax stage1_shared.scenario_bank thresholds")
    final_arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in arrays.items()}
    output_path = output_dir / str(bank_cfg.get("output_name", "scenario_bank.npz"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **final_arrays)
    summary = {
        "split": split,
        "num_source_contexts": int(len(idx)),
        "num_generated_candidates": int(generated_total),
        "num_accepted_candidates": int(accepted_total),
        "num_scenarios": int(final_arrays["shared_actions"].shape[0]),
        "acceptance_rate": float(accepted_total / max(generated_total, 1)),
        "scenario_bank_filter": {
            "top_k_per_context": int(bank_cfg.get("top_k_per_context", 8)),
            "min_proxy_risk_delta": float(bank_cfg.get("min_proxy_risk_delta", 0.0)),
            "max_physics_penalty": float(bank_cfg.get("max_physics_penalty", 0.05)),
            "max_naturalness_penalty": float(bank_cfg.get("max_naturalness_penalty", 1.0)),
            "require_diverse_risk_types": bool(bank_cfg.get("require_diverse_risk_types", True)),
        },
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
