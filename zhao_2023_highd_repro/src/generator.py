from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import finite_summary, output_dir, save_json, set_seed


def _duration_rmse(real: np.ndarray, generated: np.ndarray, bins: int = 8) -> float:
    lo = float(min(np.min(real), np.min(generated)))
    hi = float(max(np.max(real), np.max(generated)))
    if hi <= lo:
        return 0.0
    real_hist, edges = np.histogram(real, bins=bins, range=(lo, hi), density=False)
    gen_hist, _ = np.histogram(generated, bins=edges, density=False)
    real_pct = real_hist / max(real_hist.sum(), 1)
    gen_pct = gen_hist / max(gen_hist.sum(), 1)
    return float(np.sqrt(np.mean(np.square(real_pct - gen_pct))))


def _moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if window <= 1 or arr.shape[0] < 3:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _fit_pca(values: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm, mean, std = _standardize(values)
    norm_center = norm.mean(axis=0, keepdims=True)
    centered = norm - norm_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    k = min(int(components), vh.shape[0], values.shape[0] - 1)
    basis = vh[:k].astype(np.float32)
    latent = centered @ basis.T
    return basis, latent.astype(np.float32), mean.astype(np.float32), std.astype(np.float32), norm_center.astype(np.float32)


def _safe_cov(latent: np.ndarray) -> np.ndarray:
    if latent.shape[0] <= 1:
        return np.eye(latent.shape[1], dtype=np.float32) * 0.05
    cov = np.cov(latent, rowvar=False).astype(np.float32)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float32)
    return cov + np.eye(cov.shape[0], dtype=np.float32) * 1.0e-4


def _canonical_sequences(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final_y = sequences[:, -1, 0]
    signs = np.where(final_y >= 0.0, 1.0, -1.0).astype(np.float32)
    canon = sequences.copy().astype(np.float32)
    canon[:, :, 0] *= signs[:, None]
    lateral = np.abs(final_y).astype(np.float32)
    return canon, signs, lateral


def train_pca_gaussian(config: dict) -> dict[str, Path]:
    out = output_dir(config)
    gen_cfg = config["generator"]
    data_path = out / "emergency_cutin_dataset.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    if "paper_sequences" not in data.files:
        raise KeyError(f"{data_path} is missing 'paper_sequences'; rerun extract_data.py")
    sequences = data["paper_sequences"].astype(np.float32)
    conditions = data["conditions"].astype(np.float32)
    canonical, signs, lateral = _canonical_sequences(sequences)
    flat = canonical.reshape(canonical.shape[0], -1)
    basis, latent, phys_mean, phys_std, norm_center = _fit_pca(
        flat,
        int(gen_cfg.get("pca_components", 12)),
    )
    model_path = out / "pca_trajectory_sequence_model.npz"
    np.savez_compressed(
        model_path,
        basis=basis,
        latent_mean=latent.mean(axis=0).astype(np.float32),
        latent_cov=_safe_cov(latent),
        physical_mean=phys_mean.reshape(-1).astype(np.float32),
        physical_std=phys_std.reshape(-1).astype(np.float32),
        normalized_center=norm_center.reshape(-1).astype(np.float32),
        real_conditions=conditions.astype(np.float32),
        real_signs=signs.astype(np.float32),
        real_lateral=lateral.astype(np.float32),
        real_sequences=canonical.astype(np.float32),
    )
    save_json(
        out / "pca_generator_stats.json",
        {
            "model": "paper_y_vx_pca_gaussian",
            "samples": int(sequences.shape[0]),
            "sequence_points": int(sequences.shape[1]),
            "state": "(lateral_y, longitudinal_vx)",
            "pca_components": int(basis.shape[0]),
            "latent_std": finite_summary(np.std(latent, axis=0)),
        },
    )
    sample_paths = sample_pca_trajectories(config)
    return {"pca_model": model_path, "pca_stats": out / "pca_generator_stats.json", **sample_paths}


def _sequence_from_latent(model: dict[str, np.ndarray], latent: np.ndarray) -> np.ndarray:
    mean = model["physical_mean"]
    std = model["physical_std"]
    center = model["normalized_center"]
    basis = model["basis"]
    seq = mean + (center + latent @ basis) * std
    return seq.reshape(-1, 2).astype(np.float32)


def _regularize_sequence(
    sequence: np.ndarray,
    duration: float,
    lateral_offset: float,
    speed_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(sequence, dtype=np.float32).copy()
    points = out.shape[0]
    target = max(float(lateral_offset), 1.0)
    y = out[:, 0]
    y = y - float(y[0])
    y = _moving_average(y, 5)
    y = y - float(y[0])
    y = np.maximum(y, 0.0)
    y = np.maximum.accumulate(y)
    if float(y[-1]) <= 1.0e-3:
        y = np.linspace(0.0, target, points, dtype=np.float32)
    else:
        y = y / float(y[-1]) * target
    progress = np.linspace(0.0, 1.0, points, dtype=np.float32)
    smooth_progress = 3.0 * progress**2 - 2.0 * progress**3
    y = 0.25 * y + 0.75 * smooth_progress * target
    y = np.maximum.accumulate(y)
    y[0] = 0.0
    if float(y[-1]) <= 1.0e-3:
        y = smooth_progress * target
    else:
        y = y / float(y[-1]) * target
    max_step = max(target / max(points - 1, 1) * 3.0, 0.75)
    if np.max(np.diff(y)) > max_step:
        y = smooth_progress * target
    y[-1] = target

    vx = _moving_average(out[:, 1], 5)
    vx += rng.normal(0.0, float(speed_noise), size=points).astype(np.float32)
    vx = np.clip(vx, 1.0, 45.0)
    vx = _moving_average(vx, 3)
    return np.stack([y, vx], axis=-1).astype(np.float32)


def _states_from_sequence(sequence: np.ndarray, duration: float, sign: float) -> np.ndarray:
    points = sequence.shape[0]
    dt = float(duration) / max(points - 1, 1)
    y = sequence[:, 0] * float(sign)
    vx = sequence[:, 1]
    x = np.zeros(points, dtype=np.float32)
    for idx in range(1, points):
        x[idx] = x[idx - 1] + 0.5 * (vx[idx - 1] + vx[idx]) * dt
    vy = np.gradient(y, dt).astype(np.float32)
    ax = np.gradient(vx, dt).astype(np.float32)
    ay = np.gradient(vy, dt).astype(np.float32)
    return np.stack([x, y, vx, vy, ax, ay], axis=-1).astype(np.float32)


def _monotonic_rate(trajectories: np.ndarray) -> float:
    y = np.abs(trajectories[:, :, 1])
    return float(np.mean(np.all(np.diff(y, axis=1) >= -1.0e-3, axis=1)))


def sample_pca_trajectories(config: dict) -> dict[str, Path]:
    gen_cfg = config["generator"]
    set_seed(int(gen_cfg["seed"]) + 100)
    out = output_dir(config)
    model_path = out / "pca_trajectory_sequence_model.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Sequence model not found: {model_path}")
    model_npz = np.load(model_path, allow_pickle=True)
    model = {key: model_npz[key] for key in model_npz.files}
    rng = np.random.default_rng(int(gen_cfg["seed"]) + 200)
    count = int(gen_cfg["sample_count"])
    real_conditions = model["real_conditions"].astype(np.float32)
    real_lateral = model["real_lateral"].astype(np.float32)
    real_signs = model["real_signs"].astype(np.float32)
    latent = rng.multivariate_normal(
        mean=model["latent_mean"],
        cov=model["latent_cov"],
        size=count,
    ).astype(np.float32)
    template_idx = rng.integers(0, len(real_conditions), size=count)
    durations = real_conditions[template_idx, 0].copy()
    durations += rng.normal(0.0, 0.05, size=count).astype(np.float32)
    durations = np.clip(
        durations,
        float(config["data"]["min_lane_change_seconds"]),
        float(config["data"]["max_lane_change_seconds"]),
    )
    lateral = real_lateral[template_idx].copy()
    lateral += rng.normal(0.0, float(gen_cfg.get("lateral_noise", 0.04)), size=count).astype(np.float32)
    lateral = np.clip(lateral, 2.8, 4.5)
    signs = real_signs[template_idx]
    trajectories: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    for idx in range(count):
        seq = _sequence_from_latent(model, latent[idx])
        seq = _regularize_sequence(
            seq,
            float(durations[idx]),
            float(lateral[idx]),
            float(gen_cfg.get("speed_noise", 0.25)),
            rng,
        )
        sequences.append(seq)
        trajectories.append(_states_from_sequence(seq, float(durations[idx]), float(signs[idx])))
    trajectories_arr = np.stack(trajectories).astype(np.float32)
    sequences_arr = np.stack(sequences).astype(np.float32)
    conditions = real_conditions[template_idx].copy()
    conditions[:, 0] = durations
    conditions[:, 5] = lateral * signs
    sample_path = out / "pca_generated_trajectories.npz"
    metrics_path = out / "pca_generation_metrics.json"
    np.savez_compressed(
        sample_path,
        trajectories=trajectories_arr,
        paper_sequences=sequences_arr,
        conditions=conditions.astype(np.float32),
    )
    save_json(
        metrics_path,
        {
            "generated_count": int(count),
            "duration_distribution_rmse": _duration_rmse(real_conditions[:, 0], durations),
            "real_duration_s": finite_summary(real_conditions[:, 0]),
            "generated_duration_s": finite_summary(durations),
            "real_final_lateral_m": finite_summary(real_conditions[:, 5]),
            "generated_final_lateral_m": finite_summary(conditions[:, 5]),
            "monotonic_lateral_rate": _monotonic_rate(trajectories_arr),
            "x_monotonic_rate": float(np.mean(np.all(np.diff(trajectories_arr[:, :, 0], axis=1) >= -1.0e-3, axis=1))),
        },
    )
    return {"pca_generated": sample_path, "pca_generation_metrics": metrics_path}


def train_generator(config: dict) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if bool(config["generator"].get("keep_pca_baseline", True)):
        paths.update(train_pca_gaussian(config))
    from .bn_am_seqgan import train_bn_am_seqgan

    paths.update(train_bn_am_seqgan(config))
    return paths
