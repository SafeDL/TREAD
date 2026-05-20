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

ROOT = Path(__file__).resolve().parents[3]
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
)
from adversaray.src.stage1_shared_utils import (
    classify_risk_types,
    latent_diversity_reward,
    risk_coverage_reward,
    risk_type_summary,
    rollout_proxy_diagnostics,
    tensor_stats,
)
from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_yaml, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "prior_guided_following.yaml"
logger = logging.getLogger(__name__)

TENSORBOARD_KEYS = {
    "score/checkpoint": "checkpoint_score",
    "objective/train": "train_objective",
    "objective/val": "val_objective",
    "risk/train_before": "train_risk_before",
    "risk/train_after": "train_risk_after",
    "risk/train_delta": "train_risk_delta",
    "risk/val_before": "val_risk_before",
    "risk/val_after": "val_risk_after",
    "risk/val_delta": "val_risk_delta",
    "penalty/train_naturalness": "train_naturalness_penalty",
    "penalty/train_physics": "train_physics_penalty",
    "penalty/val_naturalness": "val_naturalness_penalty",
    "penalty/val_physics": "val_physics_penalty",
    "diversity/train_total": "train_diversity_reward",
    "diversity/train_latent": "train_latent_diversity_reward",
    "diversity/train_risk_coverage": "train_risk_coverage_reward",
    "diversity/val_total": "val_diversity_reward",
    "diversity/val_latent": "val_latent_diversity_reward",
    "diversity/val_risk_coverage": "val_risk_coverage_reward",
    "coverage/val_risk_type_entropy": "val_risk_type_entropy",
    "residual/train_l2": "train_delta_l2",
    "residual/train_smoothness": "train_delta_smoothness",
    "residual/train_abs_max": "train_delta_abs_max",
    "residual/val_l2": "val_delta_l2",
    "residual/val_smoothness": "val_delta_smoothness",
    "residual/val_abs_max": "val_delta_abs_max",
    "risk/train_term": "train_risk_term",
    "risk/val_term": "val_risk_term",
    "risk/train_delta_p95": "train_risk_delta_p95",
    "risk/val_delta_p95": "val_risk_delta_p95",
    "risk/train_delta_positive_rate": "train_risk_delta_positive_rate",
    "risk/val_delta_positive_rate": "val_risk_delta_positive_rate",
}


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
    cfg["policy"].setdefault("mode", "direct_residual_sequence")
    cfg["policy"].setdefault("hidden_dim", 128)
    cfg["policy"].setdefault("latent_dim", 8)
    cfg["policy"].setdefault("prior_action_hidden_dim", 128)
    cfg["policy"].setdefault("decoder_layers", 1)
    cfg["policy"].setdefault("output_residual_scale", 6.0)
    cfg["policy"].setdefault("max_delta_jerk", 8.0)
    cfg["policy"].setdefault("zero_init_output", True)
    cfg.setdefault("optimization", {})
    opt = cfg["optimization"]
    opt.setdefault("epochs", 50)
    opt.setdefault("batch_size", 64)
    opt.setdefault("lr", 3.0e-4)
    opt.setdefault("risk_pool_beta", 8.0)
    opt.setdefault("lambda_nat", 0.01)
    opt.setdefault("lambda_delta_smooth", 0.1)
    opt.setdefault("lambda_phys", 0.1)
    opt.setdefault("lambda_div", 0.01)
    opt.setdefault("max_train_contexts", 0)
    opt.setdefault("max_val_contexts", 512)
    opt.setdefault("val_batches", 4)
    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("tensorboard", True)
    cfg["logging"].setdefault("tensorboard_dir", "runs")
    return cfg


def _tensorboard_writer(stage1: dict[str, Any], output_dir: Path) -> Any | None:
    log_cfg = stage1.get("logging", {})
    if not bool(log_cfg.get("tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard logging disabled because SummaryWriter is unavailable: %s", exc)
        return None
    log_dir = Path(str(log_cfg.get("tensorboard_dir", "runs")))
    if not log_dir.is_absolute():
        log_dir = output_dir / log_dir
    return SummaryWriter(log_dir=str(log_dir))


def _log_tensorboard(writer: Any | None, row: dict[str, Any]) -> None:
    if writer is None:
        return
    step = int(row["epoch"])
    for tag, key in TENSORBOARD_KEYS.items():
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            writer.add_scalar(tag, float(value), step)


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
    idm_base = sample_idm_surrogate_params(
        stage1,
        batch_size=base_count,
        num_samples=num_surrogate,
        device=device,
        dtype=prior_actions.dtype,
        flatten=True,
    )
    idm_params = idm_base.repeat_interleave(num_latent, dim=0)
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
    expanded["num_surrogate"] = num_surrogate
    expanded["num_latent"] = num_latent
    return expanded


def _forward_objective(
    policy: SharedProposalPolicy,
    candidate: dict[str, Any],
    schema: dict[str, Any],
    proxy_config: dict[str, Any],
    stage1: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    prior_actions = candidate["prior_actions"]
    delta = policy(
        candidate["context_features"],
        candidate["relative_history"],
        prior_actions,
        candidate["ego_surrogate_params"],
        candidate["latent_z"],
    )
    shared_actions = prior_actions + delta
    before = rollout_proxy_diagnostics(
        prior_actions,
        candidate["raw_context"],
        candidate["ego_length"],
        candidate["adv_length"],
        schema,
        proxy_config,
        ego_surrogate_params=candidate["ego_surrogate_params"],
    )
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
    risk_delta = after["risk_objective"] - before["risk_objective"].detach()
    delta_l2 = delta.square().flatten(1).mean(dim=1)
    if delta.shape[1] > 1:
        delta_smoothness = (delta[:, 1:] - delta[:, :-1]).square().flatten(1).mean(dim=1)
    else:
        delta_smoothness = torch.zeros_like(delta_l2)
    delta_abs_max = delta.abs().amax(dim=(1, 2))
    naturalness = delta_l2 + float(opt.get("lambda_delta_smooth", 0.1)) * delta_smoothness
    base_count = int(candidate["base_count"])
    candidate_repeat = int(candidate["candidate_repeat"])
    risk_delta_grouped = risk_delta.reshape(base_count, candidate_repeat)
    risk_pool_beta = float(opt.get("risk_pool_beta", 8.0))
    weights = torch.softmax(risk_pool_beta * risk_delta_grouped, dim=1)
    risk_term = (weights * risk_delta_grouped).sum(dim=1).mean()
    latent_diversity = latent_diversity_reward(
        delta,
        groups=base_count * int(candidate["num_surrogate"]),
        num_latents=int(candidate["num_latent"]),
    )
    risk_diversity = risk_coverage_reward(after)
    diversity = latent_diversity + 0.1 * risk_diversity
    objective = (
        risk_term
        - float(opt.get("lambda_nat", 0.01)) * naturalness.mean()
        - float(opt.get("lambda_phys", 0.1)) * after["physics_penalty"].mean()
        + float(opt.get("lambda_div", 0.01)) * diversity
    )
    diag = {
        "delta_actions": delta,
        "shared_actions": shared_actions,
        "risk_before": before["risk_objective"],
        "risk_after": after["risk_objective"],
        "risk_delta": risk_delta,
        "risk_term": risk_term,
        "risk_type": classify_risk_types(after),
        "naturalness_penalty": naturalness,
        "delta_l2": delta_l2,
        "delta_smoothness": delta_smoothness,
        "delta_abs_max": delta_abs_max,
        "physics_penalty": after["physics_penalty"],
        "diversity_reward": diversity,
        "latent_diversity_reward": latent_diversity,
        "risk_coverage_reward": risk_diversity,
        "risk_delta_p95": torch.quantile(risk_delta.detach(), 0.95),
        "risk_delta_positive_rate": (risk_delta.detach() > 0.0).float().mean(),
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
        objective, diag = _forward_objective(policy, candidate, schema, proxy_config, stage1)
        rows.append(
            {
                "objective": float(objective.detach().cpu()),
                "risk_term": float(diag["risk_term"].detach().cpu()),
                "risk_before": float(diag["risk_before"].mean().detach().cpu()),
                "risk_after": float(diag["risk_after"].mean().detach().cpu()),
                "risk_delta": float(diag["risk_delta"].mean().detach().cpu()),
                "risk_delta_p95": float(diag["risk_delta_p95"].detach().cpu()),
                "risk_delta_positive_rate": float(diag["risk_delta_positive_rate"].detach().cpu()),
                "naturalness_penalty": float(diag["naturalness_penalty"].mean().detach().cpu()),
                "delta_l2": float(diag["delta_l2"].mean().detach().cpu()),
                "delta_smoothness": float(diag["delta_smoothness"].mean().detach().cpu()),
                "delta_abs_max": float(diag["delta_abs_max"].mean().detach().cpu()),
                "physics_penalty": float(diag["physics_penalty"].mean().detach().cpu()),
                "diversity_reward": float(diag["diversity_reward"].detach().cpu()),
                "latent_diversity_reward": float(diag["latent_diversity_reward"].detach().cpu()),
                "risk_coverage_reward": float(diag["risk_coverage_reward"].detach().cpu()),
            }
        )
        risk_types.append(diag["risk_type"].detach().cpu().numpy())
        deltas.append(diag["delta_actions"].detach().cpu().numpy())
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
    if risk_types:
        summary.update(risk_type_summary(np.concatenate(risk_types, axis=0)))
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
    tb_writer = _tensorboard_writer(stage1, output_dir)

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
                    "risk_term": float(diag["risk_term"].detach().cpu()),
                    "risk_before": float(diag["risk_before"].mean().detach().cpu()),
                    "risk_after": float(diag["risk_after"].mean().detach().cpu()),
                    "risk_delta": float(diag["risk_delta"].mean().detach().cpu()),
                    "risk_delta_p95": float(diag["risk_delta_p95"].detach().cpu()),
                    "risk_delta_positive_rate": float(diag["risk_delta_positive_rate"].detach().cpu()),
                    "naturalness_penalty": float(diag["naturalness_penalty"].mean().detach().cpu()),
                    "delta_l2": float(diag["delta_l2"].mean().detach().cpu()),
                    "delta_smoothness": float(diag["delta_smoothness"].mean().detach().cpu()),
                    "delta_abs_max": float(diag["delta_abs_max"].mean().detach().cpu()),
                    "physics_penalty": float(diag["physics_penalty"].mean().detach().cpu()),
                    "diversity_reward": float(diag["diversity_reward"].detach().cpu()),
                    "latent_diversity_reward": float(diag["latent_diversity_reward"].detach().cpu()),
                    "risk_coverage_reward": float(diag["risk_coverage_reward"].detach().cpu()),
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
        score = float(val_summary.get("risk_term", val_summary.get("risk_delta", 0.0))) + 0.05 * float(
            val_summary.get("risk_type_entropy", 0.0)
        )
        row = {"epoch": epoch, "checkpoint_score": score, **train_summary, **{f"val_{k}": v for k, v in val_summary.items() if isinstance(v, (int, float))}}
        history.append(row)
        _log_tensorboard(tb_writer, row)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "policy_state": policy.state_dict(),
                    "policy_config": asdict(policy_cfg),
                    "policy_mode": str(stage1.get("policy", {}).get("mode", "direct_residual_sequence")),
                    "stage1_config": stage1,
                    "schema": sampler.prior.schema,
                    "epoch": epoch,
                    "checkpoint_score": best_score,
                },
                best_path,
            )
        logger.info("epoch %d score %.4f val risk delta %.4f", epoch, score, float(val_summary.get("risk_delta", float("nan"))))

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
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
