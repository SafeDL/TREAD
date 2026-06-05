"""Training loop for naturalistic action diffusion priors."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .data import SPLIT_TO_INDEX, build_action_dataset, load_normalized_dataset
from .model import GaussianActionDiffusion, build_model_from_schema
from .utils import load_json, save_json, select_device, set_seed

logger = logging.getLogger(__name__)


def _make_loader(
    arrays: dict,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int = 0,
) -> DataLoader:
    mask = arrays["split_index"] == SPLIT_TO_INDEX[split]
    if not np.any(mask):
        raise RuntimeError(f"No samples for split={split}")
    idx = np.where(mask)[0]
    if max_samples and max_samples > 0:
        idx = idx[: int(max_samples)]
    if "relative_history" not in arrays:
        raise KeyError("Diffusion dataset is missing required relative_history")
    tensor_items = [
        torch.from_numpy(arrays["context_states"][idx]).float(),
        torch.from_numpy(arrays["context_features"][idx]).float(),
        torch.from_numpy(arrays["relative_history"][idx]).float(),
        torch.from_numpy(arrays["actions"][idx]).float(),
    ]
    for key, dtype in (
        ("future_cross_index", "long"),
        ("future_cutin_end_index", "long"),
        ("cross_mask", "float"),
        ("cutin_end_mask", "float"),
        ("trajectory_targets", "float"),
    ):
        if key not in arrays:
            continue
        tensor = torch.from_numpy(arrays[key][idx])
        tensor_items.append(tensor.long() if dtype == "long" else tensor.float())
    return DataLoader(
        TensorDataset(*tensor_items),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def _epoch(
    model: GaussianActionDiffusion,
    loader: DataLoader,
    device: torch.device,
    action_stats: dict | None = None,
    context_stats: dict | None = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: Dict[str, float] = {}
    total_n = 0
    for batch in loader:
        history, context, relative, actions = batch[:4]
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        trajectory_meta = None
        if len(batch) >= 8:
            trajectory_meta = {
                "future_cross_index": batch[4].to(device, non_blocking=True),
                "future_cutin_end_index": batch[5].to(device, non_blocking=True),
                "cross_mask": batch[6].to(device, non_blocking=True),
                "cutin_end_mask": batch[7].to(device, non_blocking=True),
                "action_stats": action_stats,
                "context_stats": context_stats,
            }
            if len(batch) >= 9:
                trajectory_meta["trajectory_targets"] = batch[8].to(
                    device,
                    non_blocking=True,
                )
        with torch.set_grad_enabled(train):
            losses = model.p_losses(
                actions,
                history,
                context,
                relative,
                trajectory_meta=trajectory_meta,
            )
            loss = losses["loss"]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        n = int(actions.shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * n
        total_n += n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def _torch_generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _fixed_noise_losses(
    model: GaussianActionDiffusion,
    actions: torch.Tensor,
    history: torch.Tensor,
    context: torch.Tensor,
    relative: torch.Tensor,
    timestep: int,
    noise_seed: int,
) -> Dict[str, torch.Tensor]:
    t = torch.full((actions.shape[0],), int(timestep), device=actions.device, dtype=torch.long)
    generator = _torch_generator_for(actions.device, int(noise_seed))
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    noisy = model.q_sample(actions, t, noise)
    pred = model.denoiser(noisy, t, history, context, relative)
    noise_mse = F.mse_loss(pred, noise)
    x0 = model.predict_start_from_noise(noisy, t, pred)
    x0_l1 = F.l1_loss(x0, actions)
    if x0.shape[1] > 1:
        smooth = torch.mean(torch.abs(x0[:, 1:] - x0[:, :-1]))
    else:
        smooth = torch.zeros((), device=actions.device, dtype=actions.dtype)
    loss = noise_mse + model.denoiser.cfg.x0_weight * x0_l1 + model.denoiser.cfg.smooth_weight * smooth
    return {
        "loss": loss,
        "noise_mse": noise_mse.detach(),
        "x0_l1": x0_l1.detach(),
        "smooth": smooth.detach(),
    }


@torch.no_grad()
def _deterministic_epoch(
    model: GaussianActionDiffusion,
    loader: DataLoader,
    device: torch.device,
    timesteps: list[int],
    noise_seed: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_n = 0
    for batch_idx, batch in enumerate(loader):
        history, context, relative, actions = batch[:4]
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        n = int(actions.shape[0])
        for offset, timestep in enumerate(timesteps):
            losses = _fixed_noise_losses(
                model,
                actions,
                history,
                context,
                relative,
                timestep,
                int(noise_seed) + batch_idx * 1009 + offset * 9173,
            )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * n
            total_n += n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def _fixed_timesteps_from_config(training: dict, model: GaussianActionDiffusion) -> list[int]:
    raw = training.get("fixed_val_timesteps", [0, 25, 50, 75, 99])
    out = sorted({max(0, min(int(t), model.num_steps - 1)) for t in raw})
    if not out:
        raise ValueError("training.fixed_val_timesteps must contain at least one timestep")
    return out


def _validate_schema_matches_config(schema: dict, config: dict, output_dir: Path) -> None:
    expected = {
        "event_type": str(config.get("event", {}).get("event_type", "")),
        "history_steps": int(config.get("context", {}).get("history_steps", -1)),
        "horizon_steps": int(config.get("generation", {}).get("horizon_steps", -1)),
        "action_representation": str(config.get("action", {}).get("representation", "")),
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if expected_value in {"", -1}:
            continue
        actual = schema.get(key)
        if isinstance(expected_value, int):
            actual = int(actual)
        else:
            actual = str(actual)
        if actual != expected_value:
            mismatches.append(f"{key}: schema={actual!r}, config={expected_value!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(
            "Existing diffusion dataset schema does not match the training "
            f"config in {output_dir}: {joined}. Rebuild the dataset first with "
            "process_highD/scripts/build_natural_dataset.py or set "
            "dataset.rebuild=true for one run."
        )


def train_action_diffusion(config: dict, *, config_dir: str | Path | None = None) -> dict:
    paths = config.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    output_dir = (base / paths["output_dir"]).resolve()
    dataset_path = output_dir / "dataset_normalized.npz"
    if bool(config.get("dataset", {}).get("rebuild", False)):
        build_action_dataset(config, config_dir=base)
    elif not dataset_path.exists():
        raise FileNotFoundError(f"Diffusion dataset not found: {dataset_path}")

    schema = load_json(output_dir / "feature_schema.json")
    _validate_schema_matches_config(schema, config, output_dir)
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = load_normalized_dataset(output_dir)
    training = config.get("training", {})
    set_seed(int(training.get("seed", 42)))
    device = select_device(training.get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    train_loader = _make_loader(arrays, "train", batch_size, True, num_workers)
    val_loader = _make_loader(arrays, "val", batch_size, False, num_workers)
    fixed_val_max_samples = int(training.get("fixed_val_max_samples", 512))
    fixed_val_loader = _make_loader(arrays, "val", batch_size, False, num_workers, fixed_val_max_samples)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 160))
    grad_clip = float(training.get("grad_clip", 1.0))
    min_lr = float(training.get("min_lr", 5e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1), eta_min=min_lr)
    fixed_val_timesteps = _fixed_timesteps_from_config(training, model)
    fixed_val_seed = int(training.get("fixed_val_seed", 12345))
    best_noise_mse = float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    final_metrics: dict[str, float] = {}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard"

    logger.info("Training on %s for %d epochs; samples=%d", device, epochs, int(arrays["actions"].shape[0]))
    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(
                model,
                train_loader,
                device,
                stats.get("actions"),
                stats.get("context_states"),
                optimizer,
                grad_clip,
            )
            with torch.no_grad():
                val_metrics = _epoch(
                    model,
                    val_loader,
                    device,
                    stats.get("actions"),
                    stats.get("context_states"),
                    None,
                )
            fixed_val_metrics = _deterministic_epoch(model, fixed_val_loader, device, fixed_val_timesteps, fixed_val_seed)
            final_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "fixed_val_loss": fixed_val_metrics["loss"],
                "train_noise_mse": train_metrics["noise_mse"],
                "val_noise_mse": val_metrics["noise_mse"],
                "fixed_val_noise_mse": fixed_val_metrics["noise_mse"],
                "train_x0_l1": train_metrics["x0_l1"],
                "val_x0_l1": val_metrics["x0_l1"],
                "fixed_val_x0_l1": fixed_val_metrics["x0_l1"],
                "train_smooth": train_metrics["smooth"],
                "val_smooth": val_metrics["smooth"],
                "fixed_val_smooth": fixed_val_metrics["smooth"],
            }
            for key in (
                "trajectory_x_l1",
                "trajectory_y_l1",
                "trajectory_vx_l1",
                "trajectory_vy_l1",
                "endpoint_x_l1",
                "endpoint_y_l1",
                "cross_y_l1",
                "end_y_l1",
                "kinematic_consistency_l1",
            ):
                if key in train_metrics:
                    final_metrics[f"train_{key}"] = train_metrics[key]
                if key in val_metrics:
                    final_metrics[f"val_{key}"] = val_metrics[key]
            best_val_loss = min(best_val_loss, float(val_metrics["loss"]))
            val_noise_mse = float(val_metrics["noise_mse"])
            if val_noise_mse < best_noise_mse:
                best_noise_mse = val_noise_mse
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "schema": schema,
                        "config": config,
                        "epoch": epoch,
                        "val_noise_mse": best_noise_mse,
                        "val_loss": val_metrics["loss"],
                    },
                    checkpoint_dir / "best_noise_mse.pt",
                )
            writer.add_scalar("loss/train", float(train_metrics["loss"]), epoch)
            writer.add_scalar("loss/val", float(val_metrics["loss"]), epoch)
            writer.add_scalar("loss/fixed_val", float(fixed_val_metrics["loss"]), epoch)
            writer.add_scalar(
                "noise_mse/train",
                float(train_metrics["noise_mse"]),
                epoch,
            )
            writer.add_scalar("noise_mse/val", val_noise_mse, epoch)
            writer.add_scalar(
                "noise_mse/fixed_val",
                float(fixed_val_metrics["noise_mse"]),
                epoch,
            )
            writer.add_scalar("learning_rate", float(scheduler.get_last_lr()[0]), epoch)
            writer.add_scalar("best/val_noise_mse", float(best_noise_mse), epoch)
            for key in (
                "trajectory_x_l1",
                "trajectory_y_l1",
                "trajectory_vx_l1",
                "trajectory_vy_l1",
                "endpoint_x_l1",
                "endpoint_y_l1",
                "cross_y_l1",
                "end_y_l1",
                "kinematic_consistency_l1",
            ):
                if key in train_metrics:
                    writer.add_scalar(f"{key}/train", float(train_metrics[key]), epoch)
                if key in val_metrics:
                    writer.add_scalar(f"{key}/val", float(val_metrics[key]), epoch)
            if epoch == 1 or epoch % int(training.get("log_every_epochs", 10)) == 0 or epoch == epochs:
                logger.info(
                    "epoch=%03d train_noise_mse=%.6f val_noise_mse=%.6f",
                    epoch,
                    train_metrics["noise_mse"],
                    val_metrics["noise_mse"],
                )
            scheduler.step()

    save_json(
        {
            "checkpoint": str(checkpoint_dir / "best_noise_mse.pt"),
            "best_epoch": int(best_epoch),
            "best_val_loss": best_val_loss,
            "best_val_noise_mse": best_noise_mse,
            "final_metrics": final_metrics,
            "epochs": epochs,
            "lr_schedule": "cosine",
            "min_lr": min_lr,
            "fixed_val_timesteps": fixed_val_timesteps,
            "fixed_val_seed": fixed_val_seed,
            "fixed_val_max_samples": fixed_val_max_samples,
            "tensorboard_dir": str(tensorboard_dir),
        },
        output_dir / "training_summary.json",
    )
    return {"output_dir": output_dir, "best_val_loss": best_val_loss, "epochs": epochs}
