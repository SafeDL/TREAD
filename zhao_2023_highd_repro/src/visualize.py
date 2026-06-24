from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import REPO_ROOT, ensure_repo_imports, output_dir


ensure_repo_imports()


def assert_highway_env_available() -> str:
    import sys

    highway_root = REPO_ROOT / "HighwayEnv"
    if str(highway_root) not in sys.path:
        sys.path.insert(0, str(highway_root))
    from highway_env.road.road import RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle

    _ = RoadNetwork, IDMVehicle
    return str(highway_root)


def plot_case(config: dict, case_index: int = 0) -> Path:
    assert_highway_env_available()
    out = output_dir(config)
    rollouts = np.load(out / "idm_rollouts.npz", allow_pickle=True)
    ego = rollouts["ego_trajectories"][int(case_index)]
    target = rollouts["target_trajectories"][int(case_index)]
    dt = float(rollouts["dt"][int(case_index)])
    time = np.arange(len(ego), dtype=float) * dt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    ax = axes[0]
    ax.plot(ego[:, 0], ego[:, 1], label="IDM ego", linewidth=2.0)
    ax.plot(target[:, 0], target[:, 1], label="generated cut-in", linewidth=2.0)
    ax.scatter([ego[0, 0]], [ego[0, 1]], marker="o", s=36)
    ax.scatter([target[0, 0]], [target[0, 1]], marker="o", s=36)
    ax.axhline(0.0, color="0.3", linewidth=1.0, linestyle="--")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"Highway-env IDM replay, case {case_index}")
    ax.legend()
    ax.grid(True, alpha=0.25)

    gap = target[:, 0] - ego[:, 0] - 4.8
    axes[1].plot(time, gap, label="net gap")
    axes[1].axhline(0.0, color="crimson", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("time / s")
    axes[1].set_ylabel("gap / m")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"highway_env_case_{int(case_index):03d}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
