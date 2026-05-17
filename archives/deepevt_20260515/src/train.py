"""
train.py — DeepEVT / conditional tail-quantile training
=======================================================

The current objective is direct conditional tail-quantile prediction: the model
observes a short prefix and predicts q85/q90/q95.  Training always runs the full
configured epoch count; best/final checkpoints are both saved for analysis.
"""
from __future__ import annotations

import copy
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from process_highd.src.io_utils import ensure_dir, load_json, save_json

from .data import DatasetArrays, apply_normalization, load_dataset, subset
from .features import feature_keys_for
from .losses import deepevt_loss
from .model import DeepEVTModel, build_model_from_schema

logger = logging.getLogger(__name__)

COMPACT_HISTORY_BASE_KEYS = (
    "loss_q",
    "loss_cal",
    "loss_rank",
    "selection_score",
    "prefix_bin_mae_mean",
    "prefix_bin_count",
)


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
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    prefix = torch.from_numpy(arrays.prefix_states).float()
    ctx = torch.from_numpy(arrays.context_features).float()
    risk = torch.from_numpy(arrays.risk_score).float()
    ds = TensorDataset(prefix, ctx, risk)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=max(0, int(num_workers)) > 0,
    )


def _validate_feature_schema(schema: dict) -> None:
    event_type = schema.get("event_type")
    if not event_type:
        return
    expected_keys = list(feature_keys_for(str(event_type)))
    actual_keys = list(schema.get("context_keys", []))
    if not actual_keys:
        return
    if actual_keys != expected_keys:
        raise RuntimeError(
            "feature_schema.json context_keys do not match the current code. "
            "Please rebuild dataset.npz/feature_schema.json before training. "
            f"expected={expected_keys}, actual={actual_keys}"
        )


def _make_tensorboard_writer(out_dir: Path, training_cfg: dict) -> Optional[Any]:
    if not bool(training_cfg.get("tensorboard", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard requested but unavailable: %s", exc)
        return None
    log_dir = out_dir / "runs"
    ensure_dir(log_dir)
    logger.info("TensorBoard logs: %s", log_dir)
    return SummaryWriter(log_dir=str(log_dir))


def _write_tensorboard_minimal(
    writer: Optional[Any],
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    tail_levels: tuple[float, ...],
) -> None:
    if writer is None:
        return
    writer.add_scalar("loss_q/train", float(train_metrics.get("loss_q", 0.0)), epoch)
    writer.add_scalar("loss_q/val", float(val_metrics.get("loss_q", 0.0)), epoch)
    writer.add_scalar(
        "selection_score/val",
        float(val_metrics.get("selection_score", 0.0)),
        epoch,
    )
    for tau in tail_levels:
        label = f"q{int(round(float(tau) * 100))}_ece"
        writer.add_scalar(label, float(val_metrics.get(label, 0.0)), epoch)
        smooth_label = f"{label}_smooth"
        if smooth_label in val_metrics:
            writer.add_scalar(smooth_label, float(val_metrics.get(smooth_label, 0.0)), epoch)
        bin_label = f"q{int(round(float(tau) * 100))}_prefix_bin_mae"
        if bin_label in val_metrics:
            writer.add_scalar(bin_label, float(val_metrics.get(bin_label, 0.0)), epoch)
    if "prefix_bin_mae_mean" in val_metrics:
        writer.add_scalar(
            "prefix_bin_mae_mean/val",
            float(val_metrics.get("prefix_bin_mae_mean", 0.0)),
            epoch,
        )


def _q_label(tau: float) -> str:
    return f"q{int(round(float(tau) * 100))}"


def _selection_config(training_cfg: dict, weights: Dict[str, float]) -> Dict[str, float]:
    """Checkpoint-selection weights, separated from the backprop loss weights."""
    cfg = dict(training_cfg.get("checkpoint_selection", {}))
    cfg.setdefault("w_loss", 1.0)
    cfg.setdefault("w_ece85", 1.0)
    cfg.setdefault("w_ece90", 2.0)
    cfg.setdefault("w_ece95", 4.0)
    cfg.setdefault("w_bin", 0.5)
    cfg.setdefault("w_rank", 0.0)
    cfg.setdefault("ece_smoothing_window", 5)
    cfg.setdefault("prefix_bin_feature", "auto")
    cfg.setdefault("prefix_bin_count", 5)
    cfg.setdefault("prefix_bin_min_samples", 10)
    cfg.setdefault("ranking_metric", "ranking_score")
    # Legacy fallback: if a project pins the old scalar weight, honor it only
    # when no explicit per-quantile weights were provided.
    if (
        "checkpoint_selection" not in training_cfg
        and "selection_q_ece_weight" in weights
    ):
        legacy = float(weights.get("selection_q_ece_weight", 0.0))
        if legacy > 0:
            cfg["w_ece85"] = legacy
            cfg["w_ece90"] = legacy
            cfg["w_ece95"] = legacy
    return cfg


def _add_smoothed_ece_metrics(
    metrics: Dict[str, float],
    val_history: list,
    tail_levels: tuple[float, ...],
    window: int,
) -> None:
    """Add rolling-median ECE fields while keeping raw ECE untouched."""
    k = max(1, int(window))
    for tau in tail_levels:
        label = _q_label(tau)
        raw_key = f"{label}_ece"
        if raw_key not in metrics:
            continue
        recent = [
            float(r[raw_key])
            for r in val_history[-max(k - 1, 0):]
            if raw_key in r and np.isfinite(float(r[raw_key]))
        ]
        recent.append(float(metrics[raw_key]))
        metrics[f"{raw_key}_smooth"] = float(np.median(np.asarray(recent, dtype=np.float64)))


def _context_feature_values(
    arrays: DatasetArrays,
    schema: dict,
    preferred: str,
) -> Optional[np.ndarray]:
    keys = list(schema.get("context_keys", []))
    candidates = [preferred] if preferred and preferred != "auto" else []
    candidates.extend(["gap_current", "initial_gap", "min_gap_in_prefix"])
    for name in candidates:
        if name in keys:
            idx = keys.index(name)
            return arrays.context_features[:, idx].astype(np.float64)
    return None


def _prefix_longitudinal_delta(arrays: DatasetArrays) -> np.ndarray:
    prefix = arrays.prefix_states.astype(np.float64)
    current = prefix[:, -1]
    return current[:, 1, 0] - current[:, 0, 0]


def _validation_quantile_predictions(
    model: DeepEVTModel,
    loader: DataLoader,
    device: torch.device,
    tail_levels: tuple[float, ...],
) -> tuple[np.ndarray, Dict[float, np.ndarray]]:
    model.eval()
    risks: list[np.ndarray] = []
    preds: Dict[float, list[np.ndarray]] = {float(t): [] for t in tail_levels}
    with torch.no_grad():
        for prefix, ctx, risk in loader:
            prefix = prefix.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True)
            outputs = model(prefix, ctx)
            risks.append(risk.detach().cpu().numpy().astype(np.float64))
            if "quantiles" not in outputs:
                continue
            levels = tuple(float(x) for x in getattr(model.cfg, "quantile_levels", ()))
            quantiles = outputs["quantiles"].detach().cpu().numpy().astype(np.float64)
            for tau in tail_levels:
                tau_f = float(tau)
                if not levels:
                    continue
                q_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - tau_f))
                if abs(levels[q_idx] - tau_f) <= 1e-6:
                    preds[tau_f].append(quantiles[:, q_idx])
    risk_arr = np.concatenate(risks) if risks else np.asarray([], dtype=np.float64)
    pred_arrs = {
        tau: np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        for tau, parts in preds.items()
    }
    return risk_arr, pred_arrs


def _add_prefix_bin_mae_metrics(
    metrics: Dict[str, float],
    risk: np.ndarray,
    q_preds: Dict[float, np.ndarray],
    bin_values: np.ndarray,
    tail_levels: tuple[float, ...],
    *,
    n_bins: int,
    min_samples: int,
) -> None:
    risk = np.asarray(risk, dtype=np.float64)
    bin_values = np.asarray(bin_values, dtype=np.float64)
    finite = np.isfinite(risk) & np.isfinite(bin_values)
    if not np.any(finite):
        return
    values = bin_values[finite]
    quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        return
    bin_ids = np.digitize(values, edges[1:-1], right=True)
    min_n = max(1, int(min_samples))
    tau_maes: list[float] = []
    used_bins = 0
    for tau in tail_levels:
        tau_f = float(tau)
        pred = np.asarray(q_preds.get(tau_f, []), dtype=np.float64)
        if pred.shape[0] != risk.shape[0]:
            continue
        pred_f = pred[finite]
        errs: list[float] = []
        for b in range(len(edges) - 1):
            mask = bin_ids == b
            if int(mask.sum()) < min_n:
                continue
            empirical_q = float(np.quantile(risk[finite][mask], tau_f))
            mean_pred_q = float(np.mean(pred_f[mask]))
            errs.append(abs(empirical_q - mean_pred_q))
        if errs:
            mae = float(np.mean(np.asarray(errs, dtype=np.float64)))
            metrics[f"{_q_label(tau_f)}_prefix_bin_mae"] = mae
            tau_maes.append(mae)
            used_bins = max(used_bins, len(errs))
    if tau_maes:
        metrics["prefix_bin_mae_mean"] = float(np.mean(np.asarray(tau_maes, dtype=np.float64)))
        metrics["prefix_bin_count"] = float(used_bins)


def _selection_score(
    metrics: Dict[str, float],
    selection_cfg: Dict[str, float],
    tail_levels: tuple[float, ...],
) -> float:
    """Validation score for checkpoint selection.

    The score is used only to save a best-validation checkpoint.  It does not
    stop training; the final checkpoint is always written after the full epoch
    count.
    """
    score = float(selection_cfg.get("w_loss", 1.0)) * float(metrics.get("loss_q", 0.0))
    for tau in tail_levels:
        label = _q_label(tau)
        ece_key = f"{label}_ece_smooth"
        weight_key = f"w_ece{int(round(float(tau) * 100))}"
        score += float(selection_cfg.get(weight_key, 0.0)) * float(
            metrics.get(ece_key, metrics.get(f"{label}_ece", 0.0))
        )
    score += float(selection_cfg.get("w_bin", 0.0)) * float(
        metrics.get("prefix_bin_mae_mean", 0.0)
    )
    rank_key = str(selection_cfg.get("ranking_metric", "ranking_score"))
    score -= float(selection_cfg.get("w_rank", 0.0)) * float(metrics.get(rank_key, 0.0))
    return score


def _checkpoint_info(stage: int, epoch: int, score: float, metrics: Dict[str, float]) -> Dict[str, float]:
    info: Dict[str, float] = {
        "stage": stage,
        "epoch": epoch,
        "selection_score": score,
    }
    keys = [
        k for k in metrics
        if k.startswith("q") and (
            k.endswith("_ece")
            or k.endswith("_ece_smooth")
            or k.endswith("_empirical_exceed_rate")
            or k.endswith("_prefix_bin_mae")
        )
    ]
    keys.extend(k for k in metrics if k in ("prefix_bin_mae_mean", "prefix_bin_count"))
    for key in keys:
        if key in metrics:
            info[key] = metrics.get(key)
    return info


def _compact_epoch_record(
    stage: int,
    epoch: int,
    metrics: Dict[str, float],
    tail_levels: tuple[float, ...],
) -> Dict[str, float]:
    """Keep only the training signals needed for monitoring and plots."""
    record: Dict[str, float] = {"stage": stage, "epoch": epoch}
    for key in COMPACT_HISTORY_BASE_KEYS:
        if key in metrics:
            record[key] = float(metrics[key])
    for tau in tail_levels:
        label = f"q{int(round(float(tau) * 100))}"
        for suffix in ("ece", "empirical_exceed_rate"):
            key = f"{label}_{suffix}"
            if key in metrics:
                record[key] = float(metrics[key])
        for suffix in ("ece_smooth", "prefix_bin_mae"):
            key = f"{label}_{suffix}"
            if key in metrics:
                record[key] = float(metrics[key])
    return record


def _write_training_key_figure(
    history: Dict[str, list],
    tail_levels: tuple[float, ...],
    figures_dir: Path,
) -> None:
    """Write one compact figure for the full training run."""
    if not history.get("val"):
        return
    ensure_dir(figures_dir)
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "tread_deepevt_matplotlib"
    ensure_dir(cache_root)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("Training figure skipped because matplotlib is unavailable: %s", exc)
        return

    train_records = list(history.get("train", []))
    val_records = list(history.get("val", []))
    val_epochs = [int(r["epoch"]) for r in val_records]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    train_epochs = [int(r["epoch"]) for r in train_records]
    if train_records and "loss_q" in train_records[0]:
        axes[0].plot(
            train_epochs,
            [float(r.get("loss_q", np.nan)) for r in train_records],
            label="train loss_q",
            linewidth=1.4,
        )
    axes[0].plot(
        val_epochs,
        [float(r.get("loss_q", np.nan)) for r in val_records],
        label="val loss_q",
        linewidth=1.4,
    )
    axes[0].set_ylabel("pinball loss")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    if any("selection_score" in r for r in val_records):
        axes[1].plot(
            val_epochs,
            [float(r.get("selection_score", np.nan)) for r in val_records],
            color="tab:purple",
            linewidth=1.4,
        )
    axes[1].set_ylabel("selection score")
    axes[1].grid(True, alpha=0.25)

    for tau in tail_levels:
        label = f"q{int(round(float(tau) * 100))}"
        ece_key = f"{label}_ece"
        if any(ece_key in r for r in val_records):
            axes[2].plot(
                val_epochs,
                [float(r.get(ece_key, np.nan)) for r in val_records],
                label=f"{label} ECE",
                linewidth=1.4,
            )
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("validation ECE")
    axes[2].legend(frameon=False, ncol=min(3, max(1, len(tail_levels))))
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("DeepEVT key training metrics")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_key_metrics.png", dpi=150)
    plt.close(fig)


def _run_epoch(
    model: DeepEVTModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    alpha: float,
    weights: Dict[str, float],
    grad_clip: float,
    train: bool,
    tail_levels: Optional[tuple[float, ...]] = None,
) -> Dict[str, float]:
    model.train(mode=train)
    totals: Dict[str, float] = {}
    n_batches = 0
    n_samples = 0
    tail_exceed_sums: Dict[float, float] = {}
    for prefix, ctx, risk in loader:
        prefix = prefix.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True)
        risk = risk.to(device, non_blocking=True)

        outputs = model(prefix, ctx)
        batch_size = int(risk.shape[0])
        n_samples += batch_size
        with torch.no_grad():
            if tail_levels and "quantiles" in outputs:
                levels = tuple(float(x) for x in getattr(model.cfg, "quantile_levels", ()))
                for tau in tail_levels:
                    tau_f = float(tau)
                    if not levels:
                        continue
                    q_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - tau_f))
                    if abs(levels[q_idx] - tau_f) > 1e-6:
                        continue
                    q_tau = outputs["quantiles"].detach()[:, q_idx]
                    tail_exceed_sums[tau_f] = tail_exceed_sums.get(tau_f, 0.0) + (
                        float((risk > q_tau).float().sum().item())
                    )
        loss, logs = deepevt_loss(
            outputs, risk, alpha=alpha, weights=weights,
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
    for tau, total in tail_exceed_sums.items():
        label = f"q{int(round(tau * 100))}"
        empirical = total / denom
        metrics[f"{label}_empirical_exceed_rate"] = empirical
        metrics[f"{label}_ece"] = abs(empirical - (1.0 - tau))
    return metrics


def train_deepevt(output_dir: str | Path, config: dict) -> Dict[str, dict]:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    training_cfg = config.get("training", {})
    weights = dict(config.get("loss_weights", {}))
    alpha = float(training_cfg.get("alpha_u", 0.90))
    batch_size = int(training_cfg.get("batch_size", 256))
    num_workers = int(training_cfg.get("num_workers", 0))
    lr = float(training_cfg.get("lr", 1e-3))
    tail_levels = tuple(
        float(x)
        for x in training_cfg.get(
            "quantile_levels",
            training_cfg.get("eval_tail_levels", [0.85, 0.90, 0.95]),
        )
    )
    weights["direct_quantile_levels"] = tail_levels
    selection_cfg = _selection_config(training_cfg, weights)
    wd = float(training_cfg.get("weight_decay", 1e-5))
    grad_clip = float(training_cfg.get("grad_clip", 5.0))
    log_every = max(1, int(training_cfg.get("log_every_epochs", 20)))
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

    train_loader = _make_loader(
        train_arr,
        batch_size,
        shuffle=True,
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
    prefix_bin_values = _context_feature_values(
        val_arr,
        schema,
        str(selection_cfg.get("prefix_bin_feature", "auto")),
    )
    prefix_bin_source = str(selection_cfg.get("prefix_bin_feature", "auto"))
    if prefix_bin_values is None:
        prefix_bin_values = _prefix_longitudinal_delta(val_arr)
        prefix_bin_source = "prefix_longitudinal_delta"
    elif prefix_bin_source == "auto":
        keys = list(schema.get("context_keys", []))
        for candidate in ("gap_current", "initial_gap", "min_gap_in_prefix"):
            if candidate in keys:
                prefix_bin_source = candidate
                break
    logger.info(
        "Checkpoint selection: smooth_ece_window=%d prefix_bin_feature=%s",
        int(selection_cfg.get("ece_smoothing_window", 5)),
        prefix_bin_source,
    )

    model = build_model_from_schema(schema, config).to(device)

    history: Dict[str, list] = {"train": [], "val": []}
    best_state: Optional[dict] = None
    best_info: Optional[dict] = None
    best_score = float("inf")
    epochs = int(training_cfg.get("epochs", 50))

    logger.info("Direct quantile training for %d epochs", epochs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(1, epochs + 1):
        tr = _run_epoch(
            model, train_loader, opt, device,
            alpha=alpha, weights=weights,
            grad_clip=grad_clip, train=True,
            tail_levels=tail_levels,
        )
        va = _run_epoch(
            model, val_loader, None, device,
            alpha=alpha, weights=weights,
            grad_clip=grad_clip, train=False,
            tail_levels=tail_levels,
        )
        _add_smoothed_ece_metrics(
            va,
            history["val"],
            tail_levels,
            int(selection_cfg.get("ece_smoothing_window", 5)),
        )
        val_risk, val_q_preds = _validation_quantile_predictions(
            model, val_loader, device, tail_levels,
        )
        _add_prefix_bin_mae_metrics(
            va,
            val_risk,
            val_q_preds,
            prefix_bin_values,
            tail_levels,
            n_bins=int(selection_cfg.get("prefix_bin_count", 5)),
            min_samples=int(selection_cfg.get("prefix_bin_min_samples", 10)),
        )
        score = _selection_score(va, selection_cfg, tail_levels)
        va["selection_score"] = score
        _write_tensorboard_minimal(writer, ep, tr, va, tail_levels)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_info = _checkpoint_info(1, ep, score, va)

        history["train"].append(_compact_epoch_record(1, ep, tr, tail_levels))
        history["val"].append(_compact_epoch_record(1, ep, va, tail_levels))
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            q_summary = " ".join(
                f"q{int(round(t * 100))}_ece={va.get(f'q{int(round(t * 100))}_ece', 0.0):.4f}"
                for t in tail_levels
            )
            logger.info(
                "ep%03d  train_q=%.4f  val_q=%.4f  sel=%.4f  %s",
                ep,
                tr.get("loss_q", 0.0),
                va.get("loss_q", 0.0),
                score,
                q_summary,
            )

    if writer is not None:
        writer.flush()
        writer.close()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    final_state = copy.deepcopy(model.state_dict())
    final_ckpt_path = out_dir / "final_model.pt"
    final_payload = {
        "model_state_dict": final_state,
        "model_cfg": model.cfg.__dict__,
        "schema": schema,
        "alpha_u": alpha,
        "best_validation": best_info,
        "checkpoint_role": "final",
    }
    torch.save(final_payload, final_ckpt_path)

    if best_state is None:
        best_state = final_state
        best_info = {"stage": None, "epoch": None, "selection_score": None}
    best_ckpt_path = out_dir / "best_model.pt"
    best_payload = {
        "model_state_dict": best_state,
        "model_cfg": model.cfg.__dict__,
        "schema": schema,
        "alpha_u": alpha,
        "best_validation": best_info,
        "checkpoint_role": "best_validation",
    }
    torch.save(best_payload, best_ckpt_path)

    legacy_ckpt_path = out_dir / "model.pt"
    if legacy_ckpt_path.exists():
        legacy_ckpt_path.unlink()

    save_json({
        "best_checkpoint": str(best_ckpt_path),
        "final_checkpoint": str(final_ckpt_path),
        "best_validation": best_info,
        "training_policy": "full_epochs_no_early_stopping",
        "epochs_completed": {
            "train_records": len(history.get("train", [])),
            "val_records": len(history.get("val", [])),
        },
    }, out_dir / "checkpoint_summary.json")
    _write_training_key_figure(history, tail_levels, out_dir / "figures")
    save_json(history, out_dir / "training_history.json")
    logger.info("Saved checkpoints: best=%s final=%s", best_ckpt_path, final_ckpt_path)
    return {"history": history, "device": str(device)}
