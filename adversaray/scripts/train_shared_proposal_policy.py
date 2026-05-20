#!/usr/bin/env python3
"""Train the Stage 1 shared proposal policy from synthetic following contexts."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import asdict
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
from adversaray.src.risk_utils import write_csv, write_json
from adversaray.src.shared_proposal_policy import (
    SharedProposalPolicy,
    SharedProposalPolicyConfig,
    prior_action_summary,
    template_params_to_tensor,
    template_to_jerk_delta,
)
from adversaray.src.stage1_shared_utils import (
    action_template_diversity_reward,
    classify_risk_types,
    latent_diversity_reward,
    risk_coverage_reward,
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
    cfg.setdefault("policy", {})
    cfg["policy"].setdefault("hidden_dim", 128)
    cfg["policy"].setdefault("latent_dim", 8)
    cfg["policy"].setdefault("output_residual_scale", 1.0)
    cfg.setdefault("optimization", {})
    opt = cfg["optimization"]
    opt.setdefault("epochs", 50)
    opt.setdefault("batch_size", 64)
    opt.setdefault("lr", 1.0e-4)
    opt.setdefault("lambda_nat", 0.05)
    opt.setdefault("lambda_phys", 1.0)
    opt.setdefault("lambda_div", 0.1)
    opt.setdefault("max_train_contexts", 0)
    opt.setdefault("max_val_contexts", 512)
    opt.setdefault("val_batches", 4)
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


def _candidate_batch(
    sampler: PriorGuidedDiffusionSampler,
    runner: ClosedLoopFollowingRunner,
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    stage1: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    contexts = [_context(raw, int(item)) for item in indices]
    batch, prepared_contexts = _batch_observation_for_contexts(runner, contexts)
    device = sampler.prior.device
    num_prior = max(int(stage1.get("num_prior_samples_per_context", 2)), 1)
    num_surrogate = max(int(stage1.get("num_surrogate_samples_per_context", 8)), 1)
    num_latent = max(int(stage1.get("num_latents_per_context", 4)), 1)
    base_count = len(prepared_contexts) * num_prior
    candidate_repeat = num_surrogate * num_latent
    seeds = [
        int(seed) + context_pos * num_prior + prior_pos
        for context_pos in range(len(prepared_contexts))
        for prior_pos in range(num_prior)
    ]
    with torch.no_grad():
        prior_sample = sampler.sample_batch(batch, num_samples=num_prior, seed=seeds)
        context_states = batch["context_states"].to(device).float().repeat_interleave(num_prior, dim=0)
        raw_context = sampler.prior.decode_context_states(context_states)
    prior_actions = prior_sample.raw_actions.to(device)
    expanded = {
        "raw_context": raw_context.repeat_interleave(candidate_repeat, dim=0),
        "context_features": batch["context_features"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0),
        "relative_history": batch["relative_history"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0),
        "ego_length": batch["ego_length"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0),
        "adv_length": batch["adv_length"].to(device).float().repeat_interleave(num_prior, dim=0).repeat_interleave(candidate_repeat, dim=0),
        "prior_actions": prior_actions.repeat_interleave(candidate_repeat, dim=0),
    }
    idm_params = sample_idm_surrogate_params(
        stage1,
        batch_size=base_count,
        num_samples=candidate_repeat,
        device=device,
        dtype=prior_actions.dtype,
        flatten=True,
    )
    latents = _sample_latents(
        base_count * candidate_repeat,
        int(stage1.get("policy", {}).get("latent_dim", 8)),
        device=device,
        seed=int(seed) + 7919,
    )
    expanded["ego_surrogate_params"] = idm_params
    expanded["latent_z"] = latents
    expanded["base_count"] = base_count
    expanded["candidate_repeat"] = candidate_repeat
    return expanded


def _forward_objective(
    policy: SharedProposalPolicy,
    candidate: dict[str, Any],
    schema: dict[str, Any],
    proxy_config: dict[str, Any],
    stage1: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    prior_actions = candidate["prior_actions"]
    params = policy(
        candidate["context_features"],
        candidate["relative_history"],
        prior_action_summary(prior_actions),
        candidate["ego_surrogate_params"],
        candidate["latent_z"],
    )
    delta = template_to_jerk_delta(params, horizon=prior_actions.shape[1], action_dim=prior_actions.shape[2])
    shared_actions = prior_actions + delta
    after = rollout_proxy_diagnostics(
        shared_actions,
        candidate["raw_context"],
        candidate["ego_length"],
        candidate["adv_length"],
        schema,
        proxy_config,
        ego_surrogate_params=candidate["ego_surrogate_params"],
    )
    opt = stage1["optimization"]
    naturalness = delta.square().flatten(1).mean(dim=1)
    template_tensor = template_params_to_tensor(params)
    latent_diversity = latent_diversity_reward(
        delta,
        contexts=int(candidate["base_count"]),
        candidates_per_context=int(candidate["candidate_repeat"]),
    )
    risk_diversity = risk_coverage_reward(after)
    template_diversity = action_template_diversity_reward(template_tensor)
    diversity = latent_diversity + 0.1 * risk_diversity + 0.1 * template_diversity
    objective = (
        after["risk_objective"].mean()
        - float(opt.get("lambda_nat", 0.05)) * naturalness.mean()
        - float(opt.get("lambda_phys", 1.0)) * after["physics_penalty"].mean()
        + float(opt.get("lambda_div", 0.1)) * diversity
    )
    diag = {
        "template_params": template_tensor,
        "delta_actions": delta,
        "shared_actions": shared_actions,
        "risk_after": after["risk_objective"],
        "risk_type": classify_risk_types(after),
        "naturalness_penalty": naturalness,
        "physics_penalty": after["physics_penalty"],
        "diversity_reward": diversity,
        "latent_diversity_reward": latent_diversity,
        "risk_coverage_reward": risk_diversity,
        "template_diversity_reward": template_diversity,
    }
    return objective, diag


@torch.no_grad()
def _evaluate(
    policy: SharedProposalPolicy,
    sampler: PriorGuidedDiffusionSampler,
    runner: ClosedLoopFollowingRunner,
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    stage1: dict[str, Any],
    schema: dict[str, Any],
    proxy_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    policy.eval()
    opt = stage1["optimization"]
    batch_size = max(int(opt.get("batch_size", 64)), 1)
    val_batches = max(int(opt.get("val_batches", 4)), 1)
    rows: list[dict[str, float]] = []
    risk_types: list[np.ndarray] = []
    template_chunks: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    selected = indices[: max(1, min(len(indices), batch_size * val_batches))]
    for batch_id, start in enumerate(range(0, len(selected), batch_size)):
        if batch_id >= val_batches:
            break
        candidate = _candidate_batch(
            sampler,
            runner,
            raw,
            selected[start : start + batch_size],
            stage1=stage1,
            seed=int(seed) + 100000 + batch_id * batch_size,
        )
        prior_diag = rollout_proxy_diagnostics(
            candidate["prior_actions"],
            candidate["raw_context"],
            candidate["ego_length"],
            candidate["adv_length"],
            schema,
            proxy_config,
            ego_surrogate_params=candidate["ego_surrogate_params"],
        )
        objective, diag = _forward_objective(policy, candidate, schema, proxy_config, stage1)
        risk_delta = diag["risk_after"] - prior_diag["risk_objective"]
        rows.append(
            {
                "objective": float(objective.detach().cpu()),
                "risk_after": float(diag["risk_after"].mean().detach().cpu()),
                "risk_delta": float(risk_delta.mean().detach().cpu()),
                "naturalness_penalty": float(diag["naturalness_penalty"].mean().detach().cpu()),
                "physics_penalty": float(diag["physics_penalty"].mean().detach().cpu()),
                "diversity_reward": float(diag["diversity_reward"].detach().cpu()),
                "latent_diversity_reward": float(diag["latent_diversity_reward"].detach().cpu()),
                "risk_coverage_reward": float(diag["risk_coverage_reward"].detach().cpu()),
                "template_diversity_reward": float(diag["template_diversity_reward"].detach().cpu()),
            }
        )
        risk_types.append(diag["risk_type"].detach().cpu().numpy())
        template_chunks.append(diag["template_params"].detach().cpu().numpy())
        deltas.append(diag["delta_actions"].detach().cpu().numpy())
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
    if risk_types:
        summary.update(risk_type_summary(np.concatenate(risk_types, axis=0)))
    if template_chunks:
        summary.update(template_diversity_summary(np.concatenate(template_chunks, axis=0)))
    if deltas:
        summary.update(tensor_stats(np.concatenate(deltas, axis=0), "delta_action"))
    policy.train()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    apply_rss_config_override(cfg, cfg_path.parent)
    stage1 = _stage1_cfg(cfg)
    cfg["stage1_shared"] = stage1
    base = cfg_path.parent
    synthetic_path = _resolve(cfg.get("training", {}).get("synthetic_context_path", ""), base)
    raw = _load_npz(synthetic_path)
    train_idx = _split_indices(raw, "train")
    val_idx = _split_indices(raw, "val")
    opt = stage1["optimization"]
    max_train = int(opt.get("max_train_contexts", 0))
    max_val = int(opt.get("max_val_contexts", 512))
    if max_train > 0:
        train_idx = train_idx[:max_train]
    if max_val > 0:
        val_idx = val_idx[:max_val]

    sampler = _prepare_prior_sampler(cfg, base)
    runner = ClosedLoopFollowingRunner(sampler, cfg)
    proxy_config = _proxy_config(sampler.prior.config, cfg)
    policy_cfg = SharedProposalPolicyConfig.from_prior(sampler.prior.model.denoiser.cfg, cfg)
    policy = SharedProposalPolicy(policy_cfg).to(sampler.prior.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(opt.get("lr", 1.0e-4)))
    output_dir = _resolve(stage1["output_dir"], base)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    batch_size = max(int(opt.get("batch_size", 64)), 1)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_path = checkpoint_dir / "best_shared_proposal.pt"

    for epoch in range(1, max(int(opt.get("epochs", 50)), 1) + 1):
        policy.train()
        shuffled = rng.permutation(train_idx)
        epoch_rows: list[dict[str, float]] = []
        for batch_id, start in enumerate(range(0, len(shuffled), batch_size)):
            candidate = _candidate_batch(
                sampler,
                runner,
                raw,
                shuffled[start : start + batch_size],
                stage1=stage1,
                seed=seed + epoch * 100000 + batch_id * batch_size,
            )
            optimizer.zero_grad(set_to_none=True)
            objective, diag = _forward_objective(policy, candidate, sampler.prior.schema, proxy_config, stage1)
            loss = -objective
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(opt.get("grad_clip_norm", 10.0)))
            optimizer.step()
            epoch_rows.append(
                {
                    "objective": float(objective.detach().cpu()),
                    "risk_after": float(diag["risk_after"].mean().detach().cpu()),
                    "naturalness_penalty": float(diag["naturalness_penalty"].mean().detach().cpu()),
                    "physics_penalty": float(diag["physics_penalty"].mean().detach().cpu()),
                    "diversity_reward": float(diag["diversity_reward"].detach().cpu()),
                    "latent_diversity_reward": float(diag["latent_diversity_reward"].detach().cpu()),
                    "risk_coverage_reward": float(diag["risk_coverage_reward"].detach().cpu()),
                    "template_diversity_reward": float(diag["template_diversity_reward"].detach().cpu()),
                }
            )
        train_summary = {f"train_{key}": float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        val_summary = _evaluate(
            policy,
            sampler,
            runner,
            raw,
            val_idx,
            stage1=stage1,
            schema=sampler.prior.schema,
            proxy_config=proxy_config,
            seed=seed + epoch * 1000,
        )
        score = float(val_summary.get("risk_delta", 0.0)) + 0.05 * float(val_summary.get("risk_type_entropy", 0.0)) + 0.01 * float(
            val_summary.get("brake_start_std", 0.0)
        )
        row = {"epoch": epoch, "checkpoint_score": score, **train_summary, **{f"val_{k}": v for k, v in val_summary.items() if isinstance(v, (int, float))}}
        history.append(row)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "policy_state": policy.state_dict(),
                    "policy_config": asdict(policy_cfg),
                    "stage1_config": stage1,
                    "schema": sampler.prior.schema,
                    "epoch": epoch,
                    "checkpoint_score": best_score,
                },
                best_path,
            )
        logger.info("epoch %d score %.4f val risk delta %.4f", epoch, score, float(val_summary.get("risk_delta", float("nan"))))

    write_csv(output_dir / "training_history.csv", history)
    final_val = _evaluate(
        policy,
        sampler,
        runner,
        raw,
        val_idx,
        stage1=stage1,
        schema=sampler.prior.schema,
        proxy_config=proxy_config,
        seed=seed + 999999,
    )
    write_json(
        output_dir / "training_summary.json",
        {
            "synthetic_context_path": str(synthetic_path),
            "num_train_contexts": int(len(train_idx)),
            "num_val_contexts": int(len(val_idx)),
            "best_checkpoint": str(best_path),
            "best_checkpoint_score": float(best_score),
            "final_val": final_val,
        },
    )
    logger.info("Saved shared proposal checkpoint to %s", best_path)


if __name__ == "__main__":
    main()
