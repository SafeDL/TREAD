from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import finite_summary, output_dir, save_json


def _fig_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_real_generated(config: dict) -> tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    out = output_dir(config)
    return (
        np.load(out / "emergency_cutin_dataset.npz", allow_pickle=True),
        np.load(out / "generated_trajectories.npz", allow_pickle=True),
    )


def _nearest_rmse(real: np.ndarray, generated: np.ndarray) -> np.ndarray:
    flat_real = real.reshape(real.shape[0], -1)
    flat_generated = generated.reshape(generated.shape[0], -1)
    chunks = []
    for start in range(0, flat_generated.shape[0], 512):
        diff = flat_generated[start:start + 512, None, :] - flat_real[None, :, :]
        chunks.append(np.sqrt(np.mean(np.square(diff), axis=-1)).min(axis=1))
    return np.concatenate(chunks).astype(np.float32)


def _duration_table(real: np.ndarray, generated: np.ndarray, bins: np.ndarray) -> list[dict[str, float]]:
    def count(values: np.ndarray) -> np.ndarray:
        out = []
        for idx in range(len(bins) - 1):
            lo = bins[idx]
            hi = bins[idx + 1]
            if idx == 0:
                mask = (values >= lo) & (values <= hi)
            else:
                mask = (values > lo) & (values <= hi)
            out.append(int(np.sum(mask)))
        return np.asarray(out, dtype=np.int64)

    real_hist = count(real)
    gen_hist = count(generated)
    real_pct = real_hist / max(real_hist.sum(), 1) * 100.0
    gen_pct = gen_hist / max(gen_hist.sum(), 1) * 100.0
    rows = []
    for idx in range(len(bins) - 1):
        rows.append(
            {
                "duration_bin_s": f"({bins[idx]:.1f}, {bins[idx + 1]:.1f}]",
                "real_percent": float(real_pct[idx]),
                "generated_percent": float(gen_pct[idx]),
            }
        )
    return rows


def plot_trajectory_buffer(config: dict) -> Path:
    real, generated = _load_real_generated(config)
    real_traj = real["trajectories"].astype(np.float32)
    gen_traj = generated["trajectories"].astype(np.float32)
    x_grid = np.linspace(0.0, np.percentile(real_traj[:, -1, 0], 95), 80, dtype=np.float32)

    def interp_y(traj: np.ndarray) -> np.ndarray:
        out = []
        for item in traj:
            x = item[:, 0] - item[0, 0]
            y = np.abs(item[:, 1] - item[0, 1])
            out.append(np.interp(x_grid, x, y))
        return np.stack(out)

    real_y = interp_y(real_traj)
    gen_y = interp_y(gen_traj[: min(500, len(gen_traj))])
    lower = np.quantile(real_y, 0.05, axis=0)
    upper = np.quantile(real_y, 0.95, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.fill_between(x_grid, lower, upper, color="0.85", label="real trajectory buffer")
    ax.plot(x_grid, lower, color="0.35", linewidth=1.2, label="lower bound")
    ax.plot(x_grid, upper, color="0.35", linewidth=1.2, label="upper bound")
    for item in real_y[:30]:
        ax.plot(x_grid, item, color="#1f77b4", alpha=0.25, linewidth=0.9)
    for item in gen_y[:80]:
        ax.plot(x_grid, item, color="#ff7f0e", alpha=0.22, linewidth=0.9)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("Paper-like Fig. 8: lane-change trajectory buffer")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    path = _fig_dir() / "paper_like_fig08_trajectory_buffer.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_speed_distribution(config: dict, step: int, name: str) -> Path:
    real, generated = _load_real_generated(config)
    rv = real["paper_sequences"][:, step, 1].astype(np.float32)
    gv = generated["paper_sequences"][:, step, 1].astype(np.float32)
    lo = float(min(rv.min(), gv.min()))
    hi = float(max(rv.max(), gv.max()))
    xs = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.hist(rv, bins=24, density=True, alpha=0.35, label="real speed histogram")
    ax.hist(gv, bins=24, density=True, alpha=0.35, label="generated speed histogram")
    for values, label, color in [(rv, "real speed density", "#1f77b4"), (gv, "generated speed density", "#ff7f0e")]:
        mu = float(np.mean(values))
        sigma = max(float(np.std(values)), 1.0e-6)
        pdf = np.exp(-0.5 * np.square((xs - mu) / sigma)) / (sigma * np.sqrt(2.0 * np.pi))
        ax.plot(xs, pdf, color=color, linewidth=2.0, label=f"{label} N({mu:.2f}, {sigma:.2f})")
    ax.set_xlabel("speed / (m/s)")
    ax.set_ylabel("statistical frequency")
    ax.set_title(f"Paper-like {name}: speed distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = _fig_dir() / f"paper_like_{name.lower()}_speed_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_duration_distribution(config: dict) -> Path:
    real, generated = _load_real_generated(config)
    rv = real["conditions"][:, 0].astype(np.float32)
    gv = generated["conditions"][:, 0].astype(np.float32)
    lo = np.floor(min(rv.min(), gv.min()) * 5.0) / 5.0
    hi = np.ceil(max(rv.max(), gv.max()) * 5.0) / 5.0
    bins = np.round(np.arange(lo, hi + 0.001, 0.2, dtype=np.float64), 6)
    rows = _duration_table(rv, gv, bins)
    pd.DataFrame(rows).to_csv(output_dir(config) / "duration_distribution_table.csv", index=False)
    centers = (bins[:-1] + bins[1:]) * 0.5
    width = 0.08
    real_pct = np.asarray([row["real_percent"] for row in rows])
    gen_pct = np.asarray([row["generated_percent"] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(centers - width / 2, real_pct, width=width, label="real")
    ax.bar(centers + width / 2, gen_pct, width=width, label="BN-AM-SeqGAN")
    ax.set_xlabel("lane-change completion time / s")
    ax.set_ylabel("percentage / %")
    ax.set_title("Paper-like Table 3: completion time distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    path = _fig_dir() / "paper_like_table03_duration_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_rmse_distribution(config: dict) -> Path:
    real, generated = _load_real_generated(config)
    real_seq = real["paper_sequences"].astype(np.float32)
    gen_seq = generated["paper_sequences"].astype(np.float32)
    sample = gen_seq[: min(len(gen_seq), 5000)]
    lateral_rmse = _nearest_rmse(real_seq[:, :, 0:1], sample[:, :, 0:1])
    speed_rmse = _nearest_rmse(real_seq[:, :, 1:2], sample[:, :, 1:2])
    np.savez_compressed(
        output_dir(config) / "paper_like_rmse_distribution.npz",
        lateral_rmse=lateral_rmse,
        speed_rmse=speed_rmse,
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), constrained_layout=True)
    axes[0].hist(lateral_rmse, bins=30, density=True, color="#1f77b4", alpha=0.75)
    axes[0].set_xlabel("RMSE / m")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("Paper-like Fig. 11(a): lateral position RMSE")
    axes[1].hist(speed_rmse, bins=30, density=True, color="#ff7f0e", alpha=0.75)
    axes[1].set_xlabel("RMSE / (m/s)")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Paper-like Fig. 11(b): longitudinal speed RMSE")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    path = _fig_dir() / "paper_like_fig11_rmse_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_loss_curve(config: dict) -> Path:
    out = output_dir(config)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for filename, label in [
        ("rankgan_baseline.pt", "RankGAN loss"),
        ("seqgan_baseline.pt", "SeqGAN loss"),
        ("bn_am_seqgan.pt", "BN-AM-SeqGAN loss"),
    ]:
        path = out / filename
        if not path.exists():
            continue
        stats = torch.load(path, map_location="cpu")
        pre = np.asarray(stats["pretrain_losses"], dtype=np.float32)
        adv = np.asarray(stats["adv_generator_losses"], dtype=np.float32)
        curve = np.concatenate([pre, adv]) if len(adv) else pre
        ax.plot(np.arange(len(curve)), curve, linewidth=2.0, label=label)
    stats = torch.load(out / "bn_am_seqgan.pt", map_location="cpu")
    ax.axvline(len(stats["pretrain_losses"]), color="0.4", linestyle="--", linewidth=1.0, label="adversarial start")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("loss R")
    ax.set_title("Paper-like Fig. 12: generator loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = _fig_dir() / "paper_like_fig12_loss_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _valid_generated_count(npz_path: Path, max_duration: float) -> tuple[int, int, float]:
    data = np.load(npz_path, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)
    duration = data["conditions"][:, 0].astype(np.float32)
    lateral_step = np.max(np.abs(np.diff(traj[:, :, 1], axis=1)), axis=1)
    speed_step = np.max(np.abs(np.diff(traj[:, :, 2], axis=1)), axis=1)
    valid = (duration <= float(max_duration) + 1.0e-6) & (lateral_step <= 1.2) & (speed_step <= 8.0)
    count = int(np.sum(valid))
    total = int(len(valid))
    return count, total, float(count / max(total, 1) * 100.0)


def write_output_effectiveness_table(config: dict) -> Path:
    out = output_dir(config)
    table_path = out / "output_effectiveness_table.csv"
    old_rows = {}
    if table_path.exists():
        old = pd.read_csv(table_path)
        old_rows = {str(row["model"]): row.to_dict() for _, row in old.iterrows()}
    rows = []
    for label, filename in [
        ("SeqGAN", "seqgan_generated_trajectories.npz"),
        ("RankGAN", "rankgan_generated_trajectories.npz"),
        ("BN-AM-SeqGAN", "generated_trajectories.npz"),
    ]:
        path = out / filename
        if path.exists():
            valid_count, total, effectiveness = _valid_generated_count(
                path,
                float(config["data"]["max_lane_change_seconds"]),
            )
        elif label in old_rows:
            row = old_rows[label]
            valid_count = int(row["valid_count"])
            total = int(row["generated_count"])
            effectiveness = float(row["effectiveness_percent"])
        else:
            continue
        rows.append(
            {
                "model": label,
                "generated_count": total,
                "valid_count": valid_count,
                "effectiveness_percent": effectiveness,
                "criteria": "duration<=max_lane_change_seconds, max |dy|<=1.2m, max |dvx|<=8m/s",
            }
        )
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return table_path


def plot_dangerous_scenarios(config: dict) -> Path:
    out = output_dir(config)
    rollouts = np.load(out / "idm_rollouts.npz", allow_pickle=True)
    ego = rollouts["ego_trajectories"].astype(np.float32)
    target = rollouts["target_trajectories"].astype(np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for ax, indexes, title in [
        (axes[0], range(0, min(5, len(ego))), "Paper-like Fig. 13: critical scenarios"),
        (axes[1], range(5, min(15, len(ego))), "Paper-like Fig. 14: diverse scenarios"),
    ]:
        for idx in indexes:
            ax.plot(ego[idx, :, 0], ego[idx, :, 1], linewidth=1.6)
            ax.plot(target[idx, :, 0], target[idx, :, 1], linewidth=1.6, linestyle="--")
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    path = _fig_dir() / "paper_like_fig13_14_dangerous_scenarios.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_ttc_distribution(config: dict) -> Path:
    df = pd.read_csv(output_dir(config) / "idm_metrics.csv")
    ttc = df["min_ttc_s"].to_numpy(dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.hist(ttc, bins=30, color="#1f77b4", alpha=0.78)
    ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="TTC = 1 s")
    ax.set_xlabel("minimum TTC / s")
    ax.set_ylabel("scenario count")
    ax.set_title("Paper-like Fig. 15: TTC distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    path = _fig_dir() / "paper_like_fig15_ttc_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_all_paper_figures(config: dict) -> dict[str, Path]:
    paths = {
        "fig08": plot_trajectory_buffer(config),
        "fig09": plot_speed_distribution(config, 0, "fig09_start"),
        "fig10": plot_speed_distribution(config, -1, "fig10_end"),
        "table03": plot_duration_distribution(config),
        "fig11": plot_rmse_distribution(config),
        "fig12": plot_loss_curve(config),
        "fig13_14": plot_dangerous_scenarios(config),
        "fig15": plot_ttc_distribution(config),
        "table04": write_output_effectiveness_table(config),
    }
    real, generated = _load_real_generated(config)
    rmse = np.load(output_dir(config) / "paper_like_rmse_distribution.npz")
    metrics = {
        "real_speed_start": finite_summary(real["paper_sequences"][:, 0, 1]),
        "generated_speed_start": finite_summary(generated["paper_sequences"][:, 0, 1]),
        "real_speed_end": finite_summary(real["paper_sequences"][:, -1, 1]),
        "generated_speed_end": finite_summary(generated["paper_sequences"][:, -1, 1]),
        "lateral_rmse": finite_summary(rmse["lateral_rmse"]),
        "speed_rmse": finite_summary(rmse["speed_rmse"]),
    }
    save_json(output_dir(config) / "paper_like_figure_metrics.json", metrics)
    return paths
