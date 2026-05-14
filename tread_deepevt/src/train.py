"""
train.py — DeepEVT 三阶段训练
============================

Stage 1: threshold pretrain  (u_head + ShortHistorySceneTransformer + fusion)
         损失 = pinball + calibration
Stage 2: tail train  (全部 heads + 低 lr encoder)
         损失 = pinball + exceedance + GPD NLL + calibration + support
Stage 3: end-to-end finetune
         损失同 Stage 2，但所有模块共用 finetune_lr
"""
from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from tread_highd.src.io_utils import ensure_dir, load_json, save_json

from .data import DatasetArrays, apply_normalization, load_dataset, subset
from .features import feature_keys_for
from .losses import deepevt_loss
from .model import DeepEVTModel, build_model_from_schema

logger = logging.getLogger(__name__)

TENSORBOARD_METRICS_BY_STAGE = {
    1: (
        "selection_score", "loss_q", "exceed_rate_error",
        "empirical_exceed_rate", "u_mean",
    ),
    2: (
        "selection_score", "loss_q", "loss_exc", "loss_gpd",
        "exceed_rate_error", "empirical_exceed_rate", "u_mean",
        "p_mean", "xi_mean", "beta_mean",
    ),
    3: (
        "selection_score", "loss_q", "loss_exc", "loss_gpd",
        "exceed_rate_error", "empirical_exceed_rate", "u_mean",
        "p_mean", "xi_mean", "beta_mean",
    ),
}


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


def _make_loader(
    arrays: DatasetArrays,
    batch_size: int,
    shuffle: bool,
    training_cfg: Optional[dict] = None,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    prefix = torch.from_numpy(arrays.prefix_states).float()
    ctx = torch.from_numpy(arrays.context_features).float()
    risk = torch.from_numpy(arrays.risk_score).float()
    ds = TensorDataset(prefix, ctx, risk)
    sampler = None
    if shuffle and training_cfg and bool(training_cfg.get("tail_balanced_sampling", False)):
        q = float(training_cfg.get("tail_sampling_quantile", 0.85))
        tail_weight = float(training_cfg.get("tail_sample_weight", 4.0))
        thr = float(np.quantile(arrays.risk_score, q))
        weights = np.ones(len(arrays.risk_score), dtype=np.float64)
        weights[arrays.risk_score >= thr] = tail_weight
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=max(0, int(num_workers)) > 0,
    )


def _make_tensorboard_writer(out_dir: Path, training_cfg: dict) -> Optional[Any]:
    if not bool(training_cfg.get("tensorboard", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard requested but unavailable: %s", exc)
        return None

    log_dir = Path(training_cfg.get("tensorboard_log_dir") or (out_dir / "runs"))
    ensure_dir(log_dir)
    logger.info("TensorBoard logs: %s", log_dir)
    return SummaryWriter(log_dir=str(log_dir))


def _validate_feature_schema(schema: dict) -> None:
    event_type = schema.get("event_type")
    if not event_type:
        return
    expected_keys = list(feature_keys_for(str(event_type)))
    actual_keys = list(schema.get("context_keys", []))
    if actual_keys != expected_keys:
        raise RuntimeError(
            "feature_schema.json context_keys do not match the current code. "
            "Please rebuild dataset.npz/feature_schema.json before training. "
            f"expected={expected_keys}, actual={actual_keys}"
        )


def _write_epoch_scalars(
    writer: Optional[Any],
    split: str,
    stage: int,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    if writer is None:
        return
    for metric_name in TENSORBOARD_METRICS_BY_STAGE.get(stage, ()):
        if metric_name in metrics:
            writer.add_scalar(
                f"stage_{stage}/{metric_name}/{split}",
                float(metrics[metric_name]),
                epoch,
            )


def _selection_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
    *,
    alpha: float,
    hard_cal_weight: float,
) -> float:
    """Validation score for checkpoint selection.

    ``loss_total`` contains the GPD NLL, which can become more negative while
    calibration/support deteriorate. This score deliberately excludes GPD NLL,
    and uses the hard exceedance rate to avoid selecting a threshold whose
    empirical tail mass is far from ``1 - alpha``.
    """
    score = float(metrics.get("loss_q", 0.0))
    score += float(weights.get("lambda_cal", 0.5)) * float(metrics.get("loss_cal", 0.0))
    score += float(weights.get("lambda_exc", 0.2)) * float(metrics.get("loss_exc", 0.0))
    score += float(weights.get("lambda_support", 10.0)) * float(metrics.get("loss_support", 0.0))
    target_exceed = 1.0 - float(alpha)
    hard_error = metrics.get("exceed_rate_error")
    if hard_error is None:
        hard_error = abs(float(metrics.get("empirical_exceed_rate", target_exceed)) - target_exceed)
    score += float(hard_cal_weight) * float(hard_error)
    return score


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
    n_samples = 0
    output_sums: Dict[str, float] = {}
    exceed_sum = 0.0
    for prefix, ctx, risk in loader:
        prefix = prefix.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True)
        risk = risk.to(device, non_blocking=True)

        outputs = model(prefix, ctx)
        batch_size = int(risk.shape[0])
        n_samples += batch_size
        with torch.no_grad():
            exceed_sum += float((risk > outputs["u"].detach()).float().sum().item())
            for name in ("u", "p", "xi", "beta"):
                if name in outputs:
                    output_sums[name] = output_sums.get(name, 0.0) + (
                        float(outputs[name].detach().sum().item())
                    )
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

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    denom = max(n_samples, 1)
    metrics["empirical_exceed_rate"] = exceed_sum / denom
    metrics["exceed_rate_error"] = abs(metrics["empirical_exceed_rate"] - (1.0 - alpha))
    for name, total in output_sums.items():
        metrics[f"{name}_mean"] = total / denom
    return metrics


def train_deepevt(output_dir: str | Path, config: dict) -> Dict[str, dict]:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    training_cfg = config.get("training", {})
    weights = dict(config.get("loss_weights", {}))
    alpha = float(training_cfg.get("alpha_u", 0.90))
    use_exc = bool(training_cfg.get("use_exceedance_head", True))
    batch_size = int(training_cfg.get("batch_size", 256))
    num_workers = int(training_cfg.get("num_workers", 0))
    lr = float(training_cfg.get("lr", 1e-3))
    finetune_lr = float(training_cfg.get("finetune_lr", 2e-4))
    encoder_lr_multiplier = float(training_cfg.get("encoder_lr_multiplier", 0.05))
    threshold_lr_multiplier = float(
        training_cfg.get("threshold_lr_multiplier", encoder_lr_multiplier)
    )
    tail_lr_multiplier = float(training_cfg.get("tail_lr_multiplier", 0.5))
    selection_hard_cal_weight = float(training_cfg.get("selection_hard_cal_weight", 1.0))
    early_patience = int(training_cfg.get("early_stopping_patience", 0))
    wd = float(training_cfg.get("weight_decay", 1e-5))
    grad_clip = float(training_cfg.get("grad_clip", 5.0))
    seed = int(config.get("splits", {}).get("random_seed", 42))

    _set_seed(seed)
    device = _select_device(training_cfg.get("device", "auto"))
    logger.info("Device: %s", device)
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    writer = _make_tensorboard_writer(out_dir, training_cfg)

    schema = load_json(out_dir / "feature_schema.json")
    _validate_feature_schema(schema)
    norm_stats = load_json(out_dir / "normalization_stats.json")

    arrays = load_dataset(out_dir)
    arrays = apply_normalization(arrays, norm_stats)
    train_arr = subset(arrays, "train")
    val_arr = subset(arrays, "val")
    logger.info("Train=%d  Val=%d", len(train_arr.risk_score), len(val_arr.risk_score))

    train_loader_plain = _make_loader(
        train_arr,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader_tail = _make_loader(
        train_arr,
        batch_size,
        shuffle=True,
        training_cfg=training_cfg,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        val_arr,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model_from_schema(schema, config).to(device)

    history: Dict[str, list] = {"train": [], "val": []}
    best_tail_state: Optional[dict] = None
    best_tail_info: Optional[dict] = None
    best_tail_score = float("inf")
    best_pretrain_state: Optional[dict] = None
    best_pretrain_info: Optional[dict] = None
    best_pretrain_score = float("inf")

    try:
        # ------------------------------------------------------------------
        # Stage 1 — pinball + calibration
        # ------------------------------------------------------------------
        epochs_s1 = int(training_cfg.get("pretrain_quantile_epochs", 50))
        if epochs_s1 > 0:
            logger.info("Stage 1: threshold pretrain for %d epochs", epochs_s1)
            params_s1 = model.encoder_parameters() + model.threshold_head_parameters()
            opt = torch.optim.Adam(params_s1, lr=lr, weight_decay=wd)
            for ep in range(1, epochs_s1 + 1):
                tr = _run_epoch(
                    model, train_loader_plain, opt, device,
                    alpha=alpha, weights=weights,
                    use_exc=False, include_gpd=False,
                    grad_clip=grad_clip, train=True,
                )
                va = _run_epoch(
                    model, val_loader, None, device,
                    alpha=alpha, weights=weights,
                    use_exc=False, include_gpd=False,
                    grad_clip=grad_clip, train=False,
                )
                pretrain_score = _selection_score(
                    va, weights,
                    alpha=alpha,
                    hard_cal_weight=selection_hard_cal_weight,
                )
                va["selection_score"] = pretrain_score
                if pretrain_score < best_pretrain_score:
                    best_pretrain_score = pretrain_score
                    best_pretrain_state = copy.deepcopy(model.state_dict())
                    best_pretrain_info = {
                        "stage": 1,
                        "epoch": ep,
                        "selection_score": pretrain_score,
                        "empirical_exceed_rate": va.get("empirical_exceed_rate"),
                        "exceed_rate_error": va.get("exceed_rate_error"),
                    }
                train_record = {"stage": 1, "epoch": ep, **tr}
                val_record = {"stage": 1, "epoch": ep, **va}
                history["train"].append(train_record)
                history["val"].append(val_record)
                _write_epoch_scalars(writer, "train", 1, ep, tr)
                _write_epoch_scalars(writer, "val", 1, ep, va)
                if ep == 1 or ep % 10 == 0 or ep == epochs_s1:
                    logger.info("S1 ep%03d  train_q=%.4f  val_q=%.4f  val_hard=%.4f",
                                ep, tr.get("loss_q", 0.0), va.get("loss_q", 0.0),
                                va.get("exceed_rate_error", 0.0))

        # ------------------------------------------------------------------
        # Stage 2 — tail training
        # ------------------------------------------------------------------
        epochs_s2 = int(training_cfg.get("tail_train_epochs", 100))
        if epochs_s2 > 0:
            if best_pretrain_state is not None:
                model.load_state_dict(best_pretrain_state)
                logger.info(
                    "Restored best Stage 1 checkpoint before tail training: %s",
                    best_pretrain_info,
                )
            logger.info("Stage 2: tail training for %d epochs", epochs_s2)
            # encoder/threshold use lower lr; tail heads are intentionally
            # conservative to reduce validation support/exceedance drift.
            opt = torch.optim.Adam([
                {"params": model.encoder_parameters(), "lr": lr * encoder_lr_multiplier},
                {"params": model.threshold_head_parameters(), "lr": lr * threshold_lr_multiplier},
                {"params": model.tail_head_parameters(), "lr": lr * tail_lr_multiplier},
            ], weight_decay=wd)
            stale_epochs = 0
            for ep in range(1, epochs_s2 + 1):
                tr = _run_epoch(
                    model, train_loader_tail, opt, device,
                    alpha=alpha, weights=weights,
                    use_exc=use_exc, include_gpd=True,
                    grad_clip=grad_clip, train=True,
                )
                va = _run_epoch(
                    model, val_loader, None, device,
                    alpha=alpha, weights=weights,
                    use_exc=use_exc, include_gpd=True,
                    grad_clip=grad_clip, train=False,
                )
                score = _selection_score(
                    va, weights,
                    alpha=alpha,
                    hard_cal_weight=selection_hard_cal_weight,
                )
                va["selection_score"] = score
                if score < best_tail_score:
                    best_tail_score = score
                    best_tail_state = copy.deepcopy(model.state_dict())
                    best_tail_info = {
                        "stage": 2,
                        "epoch": ep,
                        "selection_score": score,
                        "empirical_exceed_rate": va.get("empirical_exceed_rate"),
                        "exceed_rate_error": va.get("exceed_rate_error"),
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                train_record = {"stage": 2, "epoch": ep, **tr}
                val_record = {"stage": 2, "epoch": ep, **va}
                history["train"].append(train_record)
                history["val"].append(val_record)
                _write_epoch_scalars(writer, "train", 2, ep, tr)
                _write_epoch_scalars(writer, "val", 2, ep, va)
                if ep == 1 or ep % 10 == 0 or ep == epochs_s2:
                    logger.info("S2 ep%03d  val_q=%.4f  val_gpd=%.4f  val_hard=%.4f  sel=%.4f",
                                ep, va.get("loss_q", 0.0),
                                va.get("loss_gpd", 0.0), va.get("exceed_rate_error", 0.0),
                                score)
                if early_patience > 0 and stale_epochs >= early_patience:
                    logger.info(
                        "S2 early stopping at epoch %d; best=%s",
                        ep, best_tail_info,
                    )
                    break

        # ------------------------------------------------------------------
        # Stage 3 — end-to-end finetune
        # ------------------------------------------------------------------
        epochs_s3 = int(training_cfg.get("finetune_epochs", 30))
        if epochs_s3 > 0:
            logger.info("Stage 3: finetune for %d epochs", epochs_s3)
            opt = torch.optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=wd)
            stale_epochs = 0
            for ep in range(1, epochs_s3 + 1):
                tr = _run_epoch(
                    model, train_loader_tail, opt, device,
                    alpha=alpha, weights=weights,
                    use_exc=use_exc, include_gpd=True,
                    grad_clip=grad_clip, train=True,
                )
                va = _run_epoch(
                    model, val_loader, None, device,
                    alpha=alpha, weights=weights,
                    use_exc=use_exc, include_gpd=True,
                    grad_clip=grad_clip, train=False,
                )
                score = _selection_score(
                    va, weights,
                    alpha=alpha,
                    hard_cal_weight=selection_hard_cal_weight,
                )
                va["selection_score"] = score
                if score < best_tail_score:
                    best_tail_score = score
                    best_tail_state = copy.deepcopy(model.state_dict())
                    best_tail_info = {
                        "stage": 3,
                        "epoch": ep,
                        "selection_score": score,
                        "empirical_exceed_rate": va.get("empirical_exceed_rate"),
                        "exceed_rate_error": va.get("exceed_rate_error"),
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                train_record = {"stage": 3, "epoch": ep, **tr}
                val_record = {"stage": 3, "epoch": ep, **va}
                history["train"].append(train_record)
                history["val"].append(val_record)
                _write_epoch_scalars(writer, "train", 3, ep, tr)
                _write_epoch_scalars(writer, "val", 3, ep, va)
                if ep == 1 or ep % 5 == 0 or ep == epochs_s3:
                    logger.info(
                        "S3 ep%03d  val_total=%.4f  sel=%.4f",
                        ep, va.get("loss_total", 0.0), score,
                    )
                if early_patience > 0 and stale_epochs >= early_patience:
                    logger.info(
                        "S3 early stopping at epoch %d; best=%s",
                        ep, best_tail_info,
                    )
                    break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    best_info = best_tail_info or best_pretrain_info
    best_state = best_tail_state or best_pretrain_state
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best validation checkpoint: %s", best_info)
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": model.cfg.__dict__,
        "schema": schema,
        "alpha_u": alpha,
        "best_validation": best_info,
    }, ckpt_path)
    save_json(history, out_dir / "training_history.json")
    logger.info("Saved checkpoint: %s", ckpt_path)
    return {"history": history, "device": str(device)}
