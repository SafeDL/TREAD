#!/usr/bin/env python3
"""Distill searched expert-plan residuals into GuidancePolicy."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner  # noqa: E402
from adversaray.src.config_utils import apply_rss_config_override  # noqa: E402
from adversaray.src.diffusion_adapter import DiffusionPriorAdapter  # noqa: E402
from adversaray.src.guidance_policy import GuidancePolicy, GuidancePolicyConfig  # noqa: E402
from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler  # noqa: E402
from adversaray.src.prior_guided_train import _batch_observation_for_contexts, _load_npz, _resolve_paths  # noqa: E402
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"
logger = logging.getLogger(__name__)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _dataset_context(data: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    return {
        "raw_context_states": np.asarray(data["context_states"][idx], dtype=np.float32),
        "ego_length": float(data["ego_length"][idx]) if "ego_length" in data else 4.8,
        "adv_length": float(data["adv_length"][idx]) if "adv_length" in data else 4.8,
        "source_type": str(data["source_type"][idx]) if "source_type" in data else "expert_dataset",
    }


def _checkpoint_path(config: dict[str, Any], config_dir: Path, output_dir: Path) -> Path:
    distill = config.get("distillation", {})
    value = str(distill.get("checkpoint_path", "") or "").strip()
    if not value:
        return output_dir / "checkpoints" / "distilled_guidance.pt"
    return _resolve(value, config_dir)


def _save_checkpoint(
    path: Path,
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    schema: dict[str, Any],
    epoch: int,
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": sampler.policy.state_dict(),
            "config": config,
            "schema": schema,
            "epoch": int(epoch),
            "summary": summary,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--dataset", default="", help="Override distillation.dataset_path.")
    parser.add_argument("--epochs", type=int, default=0, help="Override distillation.epochs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override distillation.batch_size.")
    parser.add_argument("--max-contexts", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--output-checkpoint", default="", help="Optional checkpoint path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    config = load_yaml(cfg_path)
    apply_rss_config_override(config, cfg_path.parent)
    training = config.get("training", {})
    distill = config.setdefault("distillation", {})
    set_seed(int(training.get("seed", 42)))

    natural_dir, diffusion_ckpt, output_dir = _resolve_paths(config, cfg_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_value = str(args.dataset or distill.get("dataset_path", "") or "").strip()
    if not dataset_value:
        dataset_path = output_dir / "adversarial_plan_dataset.npz"
    else:
        dataset_path = _resolve(dataset_value, cfg_path.parent)
    data = _load_npz(dataset_path)
    required = ("context_states", "prior_plan", "expert_plan")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{dataset_path} is missing required fields: {missing}")

    successful = np.asarray(data.get("successful", np.ones(data["context_states"].shape[0])), dtype=np.float32) > 0.5
    reward_delta = np.asarray(data.get("reward_delta", np.zeros(data["context_states"].shape[0])), dtype=np.float32)
    min_delta = float(distill.get("min_reward_delta", 0.0))
    idx = np.where(successful & (reward_delta > min_delta))[0].astype(np.int64)
    if int(args.max_contexts) > 0:
        idx = idx[: int(args.max_contexts)]
    if idx.size == 0:
        raise RuntimeError(
            f"No successful expert rows found in {dataset_path}; run plan search first or lower distillation.min_reward_delta."
        )

    device = select_device(training.get("device", "auto"))
    prior = DiffusionPriorAdapter.load(natural_dir, diffusion_ckpt, device=device)
    policy = GuidancePolicy(GuidancePolicyConfig.from_prior(prior.model.denoiser.cfg, config))
    sampler = PriorGuidedDiffusionSampler(prior, policy, config).train(True)
    runner = ClosedLoopFollowingRunner(sampler, config)
    optimizer = torch.optim.AdamW(
        sampler.policy.parameters(),
        lr=float(distill.get("lr", 1e-4)),
        weight_decay=float(distill.get("weight_decay", training.get("weight_decay", 1e-5))),
    )

    epochs = int(args.epochs or distill.get("epochs", 20))
    batch_size = int(args.batch_size or distill.get("batch_size", 8))
    lambda_kl = float(distill.get("lambda_kl", 0.01))
    residual_weight = float(distill.get("residual_weight", 1.0))
    grad_clip = float(distill.get("grad_clip", training.get("grad_clip", 1.0)))
    rng = np.random.default_rng(int(training.get("seed", 42)))
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    checkpoint_path = _resolve(args.output_checkpoint, cfg_path.parent) if args.output_checkpoint else _checkpoint_path(config, cfg_path.parent, output_dir)
    schema = load_json(natural_dir / "feature_schema.json")
    prior_seeds = np.asarray(data.get("prior_seed", np.arange(data["context_states"].shape[0]) + int(training.get("seed", 42))), dtype=np.int64)

    logger.info("Distilling guidance from %d expert rows on %s", int(idx.size), device)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(idx)
        rows: list[dict[str, float]] = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            contexts = [_dataset_context(data, int(item)) for item in batch_idx]
            batch, _prepared = _batch_observation_for_contexts(runner, contexts)
            seeds = [int(prior_seeds[int(item)]) for item in batch_idx]
            expert = torch.as_tensor(data["expert_plan"][batch_idx], dtype=torch.float32, device=device)
            prior_plan = torch.as_tensor(data["prior_plan"][batch_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            sample = sampler.sample_batch_differentiable(batch, seed=seeds)
            guided = sample.raw_actions
            expert_residual = expert - prior_plan
            guided_residual = guided - prior_plan
            residual_l1 = F.l1_loss(guided_residual, expert_residual)
            residual_smooth_l1 = F.smooth_l1_loss(guided_residual, expert_residual)
            plan_l1 = F.l1_loss(guided, expert)
            distill_loss = residual_l1 + residual_weight * residual_smooth_l1
            kl_loss = sample.prior_kl.mean()
            loss = distill_loss + lambda_kl * kl_loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sampler.policy.parameters(), grad_clip)
            optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "distill_l1": float(distill_loss.detach().cpu()),
                    "residual_l1": float(residual_l1.detach().cpu()),
                    "residual_smooth_l1": float(residual_smooth_l1.detach().cpu()),
                    "plan_l1": float(plan_l1.detach().cpu()),
                    "prior_kl": float(kl_loss.detach().cpu()),
                    "guidance_norm": float(sample.guidance_norm.detach().mean().cpu()),
                }
            )

        summary = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("loss", "distill_l1", "residual_l1", "residual_smooth_l1", "plan_l1", "prior_kl", "guidance_norm")
        }
        summary["epoch"] = float(epoch)
        history.append(summary)
        if summary["loss"] < best_loss:
            best_loss = summary["loss"]
            _save_checkpoint(checkpoint_path, sampler, config, schema, epoch, {"train": summary, "dataset_path": str(dataset_path)})
        _save_checkpoint(checkpoint_path.with_name("distilled_guidance_last.pt"), sampler, config, schema, epoch, {"train": summary, "dataset_path": str(dataset_path)})
        logger.info(
            "epoch=%03d loss=%.5f l1=%.5f prior_kl=%.5f",
            epoch,
            summary["loss"],
            summary["distill_l1"],
            summary["prior_kl"],
        )

    final_summary = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "num_training_rows": int(idx.size),
        "best_loss": float(best_loss),
        "history": history,
    }
    save_json(final_summary, output_dir / "guidance_distill_summary.json")


if __name__ == "__main__":
    main()
