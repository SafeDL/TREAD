"""Frozen highD OAT configuration for IDM subset-simulation sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "IDM_subset" / "results" / "ss_sensitivity"
DEFAULT_REPEAT_ROOTS = {
    "following": REPO_ROOT / "IDM_subset" / "results" / "following_default_repeats",
    "cutin": REPO_ROOT / "IDM_subset" / "results" / "cutin_current_default_repeats",
}
CURRENT_MC_REFERENCE_SUMMARIES = {
    "following": REPO_ROOT
    / "IDM_subset"
    / "results"
    / "monte_carlo_following"
    / "latent_monte_carlo_summary.json",
    "cutin": REPO_ROOT
    / "IDM_subset"
    / "results"
    / "monte_carlo_cutin"
    / "latent_monte_carlo_summary.json",
}

DEFAULT_SEEDS = (101, 202, 303, 404, 505)
SETTING_SEEDS = (101, 202, 303)
EVENTS = ("following", "cutin")

BASE_CONFIGS = {
    "following": REPO_ROOT
    / "IDM_subset"
    / "scripts"
    / "configs"
    / "latent_subset_following.yaml",
    "cutin": REPO_ROOT
    / "IDM_subset"
    / "scripts"
    / "configs"
    / "latent_subset_cutin.yaml",
}
MC_REFERENCE_SIZES = {"following": 200_000, "cutin": 20_000}

# Execution policy for the sensitivity experiment.  These are operational
# defaults, not OAT factors, and are recorded with each newly created run.
FORMAL_EXECUTION_DEFAULTS: dict[str, int] = {
    "scheduler_workers": 4,
    "rollout_workers": 2,
    "rollout_prefetch_batches": 2,
    "population_batch_size": 64,
    "mcmc_batch_size": 64,
}

# GPU execution may introduce harmless floating-point variation.  A parallel
# result is acceptable only when it remains within these numerical bounds and
# preserves all SS-relevant discrete decisions (chain state, elite membership,
# final failure labels, and reliability status).
PARALLEL_EQUIVALENCE_TOLERANCES: dict[str, float] = {
    "score_atol": 5.0e-6,
    "subset_threshold_atol": 5.0e-6,
    "action_atol": 2.0e-2,
}

GRID: dict[str, dict[str, tuple[float | int, ...]]] = {
    "following": {
        "num_samples": (1000, 3000, 5000),
        "p0": (0.10, 0.20, 0.30),
        "proposal_std": (0.06, 0.12, 0.24),
        "context_refresh_prob": (0.30, 0.50, 0.70, 0.90),
    },
    "cutin": {
        "num_samples": (500, 1000, 2000),
        "p0": (0.05, 0.10, 0.20),
        "proposal_std": (0.05, 0.10, 0.20),
        "context_refresh_prob": (0.25, 0.50, 0.75, 0.90),
    },
}


@dataclass(frozen=True)
class RunSpec:
    """One independent SS repeat from the frozen OAT design."""

    event_type: str
    setting_id: str
    varied_parameter: str
    parameter_value: float | int | None
    is_default_setting: bool
    seed: int
    output_root: Path | None = None

    @property
    def run_dir(self) -> Path:
        if self.output_root is not None:
            return self.output_root / f"seed_{self.seed}"
        return (
            RESULTS_ROOT
            / "runs"
            / self.event_type
            / self.setting_id
            / f"seed_{self.seed}"
        )


def defaults_from_config(config: dict[str, Any]) -> dict[str, float | int]:
    subset = dict(config.get("subset_simulation", {}) or {})
    return {
        "num_samples": int(subset["num_samples"]),
        "p0": float(subset["p0"]),
        "proposal_std": float(subset["proposal_std"]),
        "context_refresh_prob": float(subset["context_refresh_prob"]),
        "mh_retries_per_sample": int(subset["mh_retries_per_sample"]),
        "max_levels": int(subset["max_levels"]),
    }


def value_token(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".12g").replace(".", "p").replace("-", "m")


def build_run_specs(event_type: str, config: dict[str, Any]) -> list[RunSpec]:
    """Return one default plus all non-default OAT setting repeats."""
    if event_type not in EVENTS:
        raise ValueError(f"Unknown event type: {event_type}")
    defaults = defaults_from_config(config)
    specs: list[RunSpec] = [
        RunSpec(event_type, "default", "default", None, True, int(seed))
        for seed in DEFAULT_SEEDS
    ]
    for parameter, values in GRID[event_type].items():
        default = defaults[parameter]
        for value in values:
            if value == default:
                continue
            setting_id = f"{parameter}_{value_token(value)}"
            specs.extend(
                RunSpec(
                    event_type,
                    setting_id,
                    parameter,
                    value,
                    False,
                    int(seed),
                )
                for seed in SETTING_SEEDS
            )
    return specs


def build_default_repeat_specs(event_type: str) -> list[RunSpec]:
    """Return the five current-default repeat runs for one event type.

    These runs are intentionally separate from the immutable OAT layout: the
    current configuration may evolve after the sensitivity experiment has been
    frozen, while each event still receives the same predeclared seed set.
    """
    if event_type not in EVENTS:
        raise ValueError(f"Unknown event type: {event_type}")
    output_root = DEFAULT_REPEAT_ROOTS[event_type]
    return [
        RunSpec(
            event_type,
            "default",
            "default",
            None,
            True,
            int(seed),
            output_root,
        )
        for seed in DEFAULT_SEEDS
    ]
