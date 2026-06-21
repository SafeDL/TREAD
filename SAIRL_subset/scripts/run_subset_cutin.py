#!/usr/bin/env python3
"""Run latent subset simulation for cut-in events."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from SAIRL_subset.src.latent_subset_runner import run_subset_from_config
from SAIRL_subset.src.result_payload import compact_sairl_result


logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = (
    ROOT / "SAIRL_subset" / "scripts" / "configs" / "latent_subset_cutin.yaml"
)


def _override_if_set(
    config: dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    if value is None:
        return
    config.setdefault(section, {})[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to cut-in latent subset config.",
    )
    parser.add_argument("--checkpoint_path", help="SAIRL TensorFlow checkpoint prefix.")
    parser.add_argument(
        "--converted_weights_path",
        help="Optional converted PyTorch/NPZ policy weights path.",
    )
    parser.add_argument("--seed", type=int, help="Shared simulation/policy seed.")
    parser.add_argument("--num_samples", type=int, help="Subset simulation N.")
    parser.add_argument("--p0", type=float, help="Subset simulation conditional level probability.")
    parser.add_argument("--max_levels", type=int, help="Maximum subset levels.")
    parser.add_argument("--proposal_std", type=float, help="Latent random-walk proposal std.")
    parser.add_argument(
        "--context_refresh_prob",
        type=float,
        help="Probability of refreshing scenario context during MH proposal.",
    )
    parser.add_argument(
        "--mh_retries_per_sample",
        type=int,
        help="MH proposal retries per generated sample.",
    )
    args = parser.parse_args()
    setup_logging("INFO")
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if args.checkpoint_path:
        config.setdefault("sairl_policy", {})["checkpoint_path"] = args.checkpoint_path
    if args.converted_weights_path:
        config.setdefault("sairl_policy", {})[
            "converted_weights_path"
        ] = args.converted_weights_path
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = int(args.seed)
        config.setdefault("context_sampling", {})["seed"] = int(args.seed)
        config.setdefault("sairl_policy", {})["seed"] = int(args.seed)
    _override_if_set(config, "subset_simulation", "num_samples", args.num_samples)
    _override_if_set(config, "subset_simulation", "p0", args.p0)
    _override_if_set(config, "subset_simulation", "max_levels", args.max_levels)
    _override_if_set(config, "subset_simulation", "proposal_std", args.proposal_std)
    _override_if_set(
        config,
        "subset_simulation",
        "context_refresh_prob",
        args.context_refresh_prob,
    )
    _override_if_set(
        config,
        "subset_simulation",
        "mh_retries_per_sample",
        args.mh_retries_per_sample,
    )
    summary_path = run_subset_from_config(
        config,
        config_path.parent,
        expected_event_type="cut_in",
    )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    result_payload = compact_sairl_result(
        summary,
        summary_path=summary_path,
        config=config,
        config_dir=config_path.parent,
    )
    result_path = summary_path.with_name("sairl_cutin_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, sort_keys=True)
        f.write("\n")
    counts = dict(summary.get("simulation_counts", {}) or {})
    input_space = dict(summary.get("input_space", {}) or {})
    reliability = dict(summary.get("reliability", {}) or {})
    logger.info(
        (
            "Cut-in subset summary: probability=%.8g se=%.3g "
            "levels=%s stop_reason=%s reliability=%s"
        ),
        float(summary.get("probability", float("nan"))),
        float(summary.get("probability_standard_error", float("nan"))),
        summary.get("num_levels"),
        summary.get("stop_reason"),
        reliability.get("status"),
    )
    logger.info("Cut-in SAIRL result output: %s", result_path)
    logger.info(
        (
            "Cut-in subset actual simulated scenario count: "
            "closed_loop_evaluations=%s stored_level_samples=%s "
            "unique_context_indices_all_levels=%s "
            "unique_context_indices_final_level=%s"
        ),
        counts.get("closed_loop_evaluations"),
        counts.get("stored_level_samples"),
        counts.get("unique_context_indices_all_levels"),
        counts.get("unique_context_indices_final_level"),
    )
    logger.info(
        (
            "Cut-in subset input space: scenario_condition_dim=%s "
            "diffusion_noise_shape=%s diffusion_noise_dim=%s "
            "joint_dim=%s"
        ),
        input_space.get("scenario_condition_dimension"),
        input_space.get("diffusion_noise_shape"),
        input_space.get("diffusion_noise_dimension"),
        input_space.get("joint_condition_noise_dimension"),
    )


if __name__ == "__main__":
    main()
