"""REINFORCE training loop for prior-regularized guided diffusion."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_json, save_json, select_device, set_seed

from .closed_loop_runner import ClosedLoopFollowingRunner
from .diffusion_adapter import DiffusionPriorAdapter
from .guidance_policy import GuidancePolicy, GuidancePolicyConfig
from .prior_guided_sampler import PriorGuidedDiffusionSampler

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _resolve_paths(config: dict[str, Any], config_dir: str | Path | None) -> tuple[Path, Path, Path]:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths = config.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    diffusion_ckpt = Path(paths.get("diffusion_checkpoint", "checkpoints/best_noise_mse.pt"))
    if not diffusion_ckpt.is_absolute():
        diffusion_ckpt = (base / diffusion_ckpt).resolve()
        if not diffusion_ckpt.exists():
            diffusion_ckpt = (natural_dir / paths.get("diffusion_checkpoint", "checkpoints/best_noise_mse.pt")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()
    return natural_dir, diffusion_ckpt, output_dir


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _make_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard unavailable: %s", exc)
        return None
    return SummaryWriter(str(output_dir / "runs"))


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    ego_lengths = raw.get("ego_length")
    adv_lengths = raw.get("adv_length")
    return {
        "raw_context_states": raw["context_states"][idx],
        "ego_length": float(ego_lengths[idx]) if ego_lengths is not None else 4.8,
        "adv_length": float(adv_lengths[idx]) if adv_lengths is not None else 4.8,
    }


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


@torch.no_grad()
def evaluate_prior_guided_policy(
    sampler: PriorGuidedDiffusionSampler,
    config: dict[str, Any],
    raw: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    max_contexts: int,
    seed: int,
) -> dict[str, float]:
    was_training = sampler.policy.training
    sampler.eval()
    runner = ClosedLoopFollowingRunner(sampler, config)
    rows: list[dict[str, float]] = []
    for offset, idx in enumerate(indices[:max_contexts]):
        result = runner.rollout(_context(raw, int(idx)), seed=int(seed) + offset)
        rows.append({"reward": result.reward, **result.metrics})
    sampler.train(was_training)
    if not rows:
        return {"reward_mean": float("nan")}
    keys = rows[0].keys()
    return {f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in keys}


def train_prior_guided_policy(config: dict[str, Any], *, config_dir: str | Path | None = None) -> dict[str, Any]:
    training = config.get("training", {})
    set_seed(int(training.get("seed", 42)))
    natural_dir, diffusion_ckpt, output_dir = _resolve_paths(config, config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_npz(natural_dir / "dataset.npz")
    schema = load_json(natural_dir / "feature_schema.json")
    split_index = raw["split_index"]
    train_idx = np.where(split_index == SPLIT_TO_INDEX[str(training.get("split", "train"))])[0]
    val_idx = np.where(split_index == SPLIT_TO_INDEX[str(training.get("val_split", "val"))])[0]
    max_train_contexts = int(training.get("max_train_contexts", 0))
    if max_train_contexts > 0:
        train_idx = train_idx[:max_train_contexts]
    if len(train_idx) == 0:
        raise RuntimeError("No training contexts found for prior-guided policy")

    device = select_device(training.get("device", "auto"))
    prior = DiffusionPriorAdapter.load(natural_dir, diffusion_ckpt, device=device)
    policy = GuidancePolicy(GuidancePolicyConfig.from_prior(prior.model.denoiser.cfg, config))
    sampler = PriorGuidedDiffusionSampler(prior, policy, config).train(True)
    runner = ClosedLoopFollowingRunner(sampler, config)
    optimizer = torch.optim.AdamW(
        sampler.policy.parameters(),
        lr=float(training.get("lr", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = int(training.get("epochs", 20))
    batch_size = int(training.get("batch_size", 4))
    grad_clip = float(training.get("grad_clip", 1.0))
    lambda_prior = float(training.get("lambda_prior", config.get("reward", {}).get("prior_kl_weight", 0.01)))
    baseline_beta = float(training.get("baseline_ema_beta", 0.9))
    eval_contexts = int(training.get("eval_contexts", 16))
    rng = np.random.default_rng(int(training.get("seed", 42)))
    writer = _make_writer(output_dir, bool(training.get("tensorboard", True)))
    history: list[dict[str, Any]] = []
    baseline: float | None = None
    best_reward = float("-inf")
    global_step = 0

    logger.info("Training prior-guided policy on %s with %d contexts", device, len(train_idx))
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_idx)
        epoch_rows: list[dict[str, float]] = []
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_prior: list[float] = []
            batch_metrics: list[dict[str, float]] = []
            for local_i, idx in enumerate(batch):
                result = runner.rollout(_context(raw, int(idx)), seed=None)
                reward = float(result.reward)
                baseline = reward if baseline is None else baseline_beta * baseline + (1.0 - baseline_beta) * reward
                advantage = reward - float(baseline)
                loss = -float(advantage) * result.log_prob_sum + lambda_prior * result.prior_kl_sum
                losses.append(loss)
                batch_rewards.append(reward)
                batch_prior.append(float(result.prior_kl_sum.detach().cpu()))
                batch_metrics.append(result.metrics)
                global_step += 1
                if writer is not None:
                    writer.add_scalar("rollout/reward", reward, global_step)
                    writer.add_scalar("rollout/prior_kl", batch_prior[-1], global_step)
                    writer.add_scalar("rollout/advantage", advantage, global_step)
            if losses:
                batch_loss = torch.stack(losses).mean()
                batch_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(sampler.policy.parameters(), grad_clip)
                optimizer.step()
            row = {
                "epoch": float(epoch),
                "reward_mean": float(np.mean(batch_rewards)),
                "prior_kl_mean": float(np.mean(batch_prior)),
                "loss": float(batch_loss.detach().cpu()) if losses else 0.0,
                "collision_rate": float(np.mean([m["collision"] for m in batch_metrics])),
                "min_ttc_mean": float(np.mean([m["min_ttc"] for m in batch_metrics])),
                "min_gap_mean": float(np.mean([m["min_gap"] for m in batch_metrics])),
                "min_rss_margin_mean": float(np.mean([m["min_rss_margin"] for m in batch_metrics])),
            }
            epoch_rows.append(row)
        epoch_summary = {key: float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}
        val_metrics: dict[str, float] = {}
        if len(val_idx) > 0 and (epoch == 1 or epoch % int(training.get("eval_every_epochs", 5)) == 0 or epoch == epochs):
            val_metrics = evaluate_prior_guided_policy(
                sampler,
                config,
                raw,
                val_idx,
                max_contexts=eval_contexts,
                seed=int(training.get("seed", 42)) + 1000 + epoch,
            )
        history_row = {"epoch": epoch, **epoch_summary, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(history_row)
        if writer is not None:
            writer.add_scalar("epoch/reward_mean", epoch_summary["reward_mean"], epoch)
            writer.add_scalar("epoch/prior_kl_mean", epoch_summary["prior_kl_mean"], epoch)
            if "reward_mean_mean" in val_metrics:
                writer.add_scalar("eval/reward_mean", val_metrics["reward_mean_mean"], epoch)
        score = float(val_metrics.get("reward_mean_mean", epoch_summary["reward_mean"]))
        checkpoint_summary = {"train": epoch_summary, "val": val_metrics}
        if score > best_reward:
            best_reward = score
            _save_checkpoint(output_dir / "checkpoints" / "best_reward.pt", sampler, config, schema, epoch, checkpoint_summary)
        _save_checkpoint(output_dir / "checkpoints" / "last.pt", sampler, config, schema, epoch, checkpoint_summary)
        if epoch == 1 or epoch % int(training.get("log_every_epochs", 1)) == 0 or epoch == epochs:
            logger.info(
                "epoch=%03d reward=%.4f prior_kl=%.4f collision=%.3f score=%.4f",
                epoch,
                epoch_summary["reward_mean"],
                epoch_summary["prior_kl_mean"],
                epoch_summary["collision_rate"],
                score,
            )

    _write_history_csv(output_dir / "training_history.csv", history)
    summary = {
        "best_reward": best_reward,
        "epochs_completed": epochs,
        "history": history,
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "training_summary.json")
    if writer is not None:
        writer.close()
    return summary
