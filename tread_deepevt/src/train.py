"""
train.py — DeepEVT 三阶段训练
============================

Stage 1: threshold pretrain  (u_head + encoder + context_mlp + fusion)
         损失 = pinball + calibration
Stage 2: tail train  (全部 heads + 低 lr encoder)
         损失 = pinball + exceedance + GPD NLL + calibration + support
Stage 3: end-to-end finetune
         损失同 Stage 2，但所有模块共用 finetune_lr
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tread_highd.src.io_utils import ensure_dir, load_json, save_json

from .data import DatasetArrays, apply_normalization, load_dataset, subset
from .losses import deepevt_loss
from .model import DeepEVTModel, build_model_from_schema

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(arrays: DatasetArrays, batch_size: int, shuffle: bool) -> DataLoader:
    prefix = torch.from_numpy(arrays.prefix_states).float()
    ctx = torch.from_numpy(arrays.context_features).float()
    risk = torch.from_numpy(arrays.risk_score).float()
    ds = TensorDataset(prefix, ctx, risk)
    # num_workers=0 保持确定性；不强制 pin_memory 以兼容 CPU
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _run_epoch(
    model: DeepEVTModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    alpha: float,
    weights: Dict[str, float],
    use_exc: bool,
    include_gpd: bool,
    grad_clip: float,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    totals: Dict[str, float] = {}
    n_batches = 0
    for prefix, ctx, risk in loader:
        prefix = prefix.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True)
        risk = risk.to(device, non_blocking=True)

        outputs = model(prefix, ctx)
        loss, logs = deepevt_loss(
            outputs, risk, alpha=alpha, weights=weights,
            use_exceedance_head=use_exc and "p" in outputs,
            include_gpd=include_gpd,
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train_deepevt(output_dir: str | Path, config: dict) -> Dict[str, dict]:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    training_cfg = config.get("training", {})
    weights = dict(config.get("loss_weights", {}))
    alpha = float(training_cfg.get("alpha_u", 0.90))
    use_exc = bool(training_cfg.get("use_exceedance_head", True))
    batch_size = int(training_cfg.get("batch_size", 256))
    lr = float(training_cfg.get("lr", 1e-3))
    finetune_lr = float(training_cfg.get("finetune_lr", 2e-4))
    wd = float(training_cfg.get("weight_decay", 1e-5))
    grad_clip = float(training_cfg.get("grad_clip", 5.0))
    seed = int(config.get("splits", {}).get("random_seed", 42))

    _set_seed(seed)
    device = _select_device(training_cfg.get("device", "auto"))
    logger.info("Device: %s", device)

    schema = load_json(out_dir / "feature_schema.json")
    norm_stats = load_json(out_dir / "normalization_stats.json")

    arrays = load_dataset(out_dir)
    arrays = apply_normalization(arrays, norm_stats)
    train_arr = subset(arrays, "train")
    val_arr = subset(arrays, "val")
    logger.info("Train=%d  Val=%d", len(train_arr.risk_score), len(val_arr.risk_score))

    train_loader = _make_loader(train_arr, batch_size, shuffle=True)
    val_loader = _make_loader(val_arr, batch_size, shuffle=False)

    model = build_model_from_schema(schema, config).to(device)

    history: Dict[str, list] = {"train": [], "val": []}

    # ------------------------------------------------------------------
    # Stage 1 — pinball + calibration
    # ------------------------------------------------------------------
    epochs_s1 = int(training_cfg.get("pretrain_quantile_epochs", 50))
    if epochs_s1 > 0:
        logger.info("Stage 1: threshold pretrain for %d epochs", epochs_s1)
        params_s1 = model.encoder_parameters() + model.threshold_head_parameters()
        opt = torch.optim.Adam(params_s1, lr=lr, weight_decay=wd)
        for ep in range(1, epochs_s1 + 1):
            tr = _run_epoch(model, train_loader, opt, device,
                            alpha=alpha, weights=weights,
                            use_exc=False, include_gpd=False,
                            grad_clip=grad_clip, train=True)
            va = _run_epoch(model, val_loader, None, device,
                            alpha=alpha, weights=weights,
                            use_exc=False, include_gpd=False,
                            grad_clip=grad_clip, train=False)
            history["train"].append({"stage": 1, "epoch": ep, **tr})
            history["val"].append({"stage": 1, "epoch": ep, **va})
            if ep == 1 or ep % 10 == 0 or ep == epochs_s1:
                logger.info("S1 ep%03d  train_q=%.4f  val_q=%.4f  val_cal=%.4f",
                            ep, tr.get("loss_q", 0.0), va.get("loss_q", 0.0),
                            va.get("loss_cal", 0.0))

    # ------------------------------------------------------------------
    # Stage 2 — tail training
    # ------------------------------------------------------------------
    epochs_s2 = int(training_cfg.get("tail_train_epochs", 100))
    if epochs_s2 > 0:
        logger.info("Stage 2: tail training for %d epochs", epochs_s2)
        # encoder 低 lr, tail heads 常规 lr
        opt = torch.optim.Adam([
            {"params": model.encoder_parameters(), "lr": lr * 0.1},
            {"params": model.threshold_head_parameters(), "lr": lr * 0.1},
            {"params": model.tail_head_parameters(), "lr": lr},
        ], weight_decay=wd)
        for ep in range(1, epochs_s2 + 1):
            tr = _run_epoch(model, train_loader, opt, device,
                            alpha=alpha, weights=weights,
                            use_exc=use_exc, include_gpd=True,
                            grad_clip=grad_clip, train=True)
            va = _run_epoch(model, val_loader, None, device,
                            alpha=alpha, weights=weights,
                            use_exc=use_exc, include_gpd=True,
                            grad_clip=grad_clip, train=False)
            history["train"].append({"stage": 2, "epoch": ep, **tr})
            history["val"].append({"stage": 2, "epoch": ep, **va})
            if ep == 1 or ep % 10 == 0 or ep == epochs_s2:
                logger.info("S2 ep%03d  val_q=%.4f  val_gpd=%.4f  val_cal=%.4f",
                            ep, va.get("loss_q", 0.0),
                            va.get("loss_gpd", 0.0), va.get("loss_cal", 0.0))

    # ------------------------------------------------------------------
    # Stage 3 — end-to-end finetune
    # ------------------------------------------------------------------
    epochs_s3 = int(training_cfg.get("finetune_epochs", 30))
    if epochs_s3 > 0:
        logger.info("Stage 3: finetune for %d epochs", epochs_s3)
        opt = torch.optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=wd)
        for ep in range(1, epochs_s3 + 1):
            tr = _run_epoch(model, train_loader, opt, device,
                            alpha=alpha, weights=weights,
                            use_exc=use_exc, include_gpd=True,
                            grad_clip=grad_clip, train=True)
            va = _run_epoch(model, val_loader, None, device,
                            alpha=alpha, weights=weights,
                            use_exc=use_exc, include_gpd=True,
                            grad_clip=grad_clip, train=False)
            history["train"].append({"stage": 3, "epoch": ep, **tr})
            history["val"].append({"stage": 3, "epoch": ep, **va})
            if ep == 1 or ep % 5 == 0 or ep == epochs_s3:
                logger.info("S3 ep%03d  val_total=%.4f", ep, va.get("loss_total", 0.0))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": model.cfg.__dict__,
        "schema": schema,
        "alpha_u": alpha,
    }, ckpt_path)
    save_json(history, out_dir / "training_history.json")
    logger.info("Saved checkpoint: %s", ckpt_path)
    return {"history": history, "device": str(device)}
