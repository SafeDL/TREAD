#!/usr/bin/env python3
"""Build EVT/POT-style tail-risk context scores for prior-guided training."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversaray.src.risk_utils import (  # noqa: E402
    criticality_score,
    interaction_metrics_from_states,
    write_csv,
    write_json,
)
from adversaray.src.rss import RSSConfig  # noqa: E402
from diffusion.src.data import SPLIT_TO_INDEX  # noqa: E402
from diffusion.src.utils import load_json, load_yaml, setup_logging  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prior_guided_following.yaml"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _context(raw: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    context: dict[str, Any] = {
        "raw_context_states": raw["context_states"][idx],
        "ego_length": float(raw["ego_length"][idx]) if "ego_length" in raw else 4.8,
        "adv_length": float(raw["adv_length"][idx]) if "adv_length" in raw else 4.8,
    }
    for key in ("recording_id", "event_id", "anchor_frame"):
        if key in raw:
            value = raw[key][idx]
            context[key] = value.item() if hasattr(value, "item") else value
    return context


def _paths(cfg: dict[str, Any], base: Path) -> tuple[Path, Path]:
    paths = cfg.get("paths", {})
    natural_dir = (base / paths.get("natural_dataset_dir", "../../../data/diffusion_natural/following")).resolve()
    output_dir = (base / paths.get("output_dir", "../../../data/adversaray/following/prior_guided")).resolve()
    return natural_dir, output_dir


def _fit_tail_probability(score: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    score64 = np.asarray(score, dtype=np.float64)
    exceedance = np.maximum(score64 - float(threshold), 0.0)
    mask = exceedance > 0.0
    info: dict[str, Any] = {"method": "empirical", "threshold": float(threshold), "num_exceedances": int(mask.sum())}
    probability = np.zeros(score64.shape, dtype=np.float64)
    if int(mask.sum()) >= 20:
        try:
            from scipy.stats import genpareto  # type: ignore

            xi, loc, beta = genpareto.fit(exceedance[mask], floc=0.0)
            tail_fraction = float(mask.mean())
            survival = genpareto.sf(exceedance[mask], c=xi, loc=loc, scale=max(beta, 1e-6))
            probability[mask] = np.clip(tail_fraction * survival, 0.0, 1.0)
            info.update({"method": "gpd", "xi": float(xi), "loc": float(loc), "beta": float(beta)})
            return probability.astype(np.float32), info
        except Exception as exc:  # noqa: BLE001
            info["gpd_error"] = str(exc)
    order = np.argsort(score64)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score64) + 1, dtype=np.float64)
    probability = 1.0 - ranks / max(len(score64), 1)
    probability[~mask] = 0.0
    return np.clip(probability, 0.0, 1.0).astype(np.float32), info


def _frozen_prior_metrics(
    cfg: dict[str, Any],
    config_dir: Path,
    raw: dict[str, np.ndarray],
    idx: np.ndarray,
    *,
    seed: int,
) -> dict[int, dict[str, float]]:
    from adversaray.src.closed_loop_runner import ClosedLoopFollowingRunner
    from adversaray.src.prior_guided_sampler import PriorGuidedDiffusionSampler

    prior_cfg = copy.deepcopy(cfg)
    prior_cfg.setdefault("policy", {})["enabled"] = False
    sampler = PriorGuidedDiffusionSampler.from_config(prior_cfg, config_dir=config_dir).eval()
    sampler.set_guidance_enabled(False)
    runner = ClosedLoopFollowingRunner(sampler, prior_cfg)
    out: dict[int, dict[str, float]] = {}
    for offset, dataset_idx in enumerate(idx):
        result = runner.rollout(_context(raw, int(dataset_idx)), seed=int(seed) + offset)
        out[int(dataset_idx)] = result.metrics
    return out


def _score(
    min_rss: np.ndarray,
    min_ttc: np.ndarray,
    min_gap: np.ndarray,
    closing_speed: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return criticality_score(
        min_rss,
        min_ttc,
        min_gap,
        closing_speed,
        w_rss=float(args.w_rss),
        w_ttc=float(args.w_ttc),
        w_gap=float(args.w_gap),
        w_dv=float(args.w_dv),
        eps=float(args.eps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="recorded_future,initial_context")
    parser.add_argument("--frozen-prior-rollouts", type=int, default=0)
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--w-rss", type=float, default=1.0)
    parser.add_argument("--w-ttc", type=float, default=1.0)
    parser.add_argument("--w-gap", type=float, default=1.0)
    parser.add_argument("--w-dv", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    natural_dir, default_output_dir = _paths(cfg, cfg_path.parent)
    dataset_path = Path(args.dataset).resolve() if args.dataset else natural_dir / "dataset.npz"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    raw = _load_npz(dataset_path)
    schema = load_json(natural_dir / "feature_schema.json")
    sources = {item.strip() for item in str(args.source).split(",") if item.strip()}
    if "recorded_future" in sources and "future_states" not in raw:
        raise RuntimeError("dataset.npz is missing future_states; recorded_future tail scoring needs recorded futures.")
    allowed_sources = {"recorded_future", "initial_context", "frozen_prior_rollout"}
    unknown_sources = sorted(sources - allowed_sources)
    if unknown_sources:
        raise ValueError(f"Unknown source(s): {unknown_sources}")
    if not sources:
        raise ValueError("--source must include at least one source")
    if args.split == "all":
        idx = np.arange(raw["context_states"].shape[0], dtype=np.int64)
    else:
        idx = np.where(raw["split_index"] == SPLIT_TO_INDEX[str(args.split)])[0].astype(np.int64)
    rng = np.random.default_rng(int(args.seed))
    if int(args.max_contexts) > 0 and len(idx) > int(args.max_contexts):
        idx = np.sort(rng.choice(idx, size=int(args.max_contexts), replace=False)).astype(np.int64)

    rss_cfg = RSSConfig.from_config(cfg)
    context = raw["context_states"][idx]
    ego_len = raw["ego_length"][idx] if "ego_length" in raw else np.full(len(idx), 4.8, dtype=np.float32)
    adv_len = raw["adv_length"][idx] if "adv_length" in raw else np.full(len(idx), 4.8, dtype=np.float32)
    initial_metrics = interaction_metrics_from_states(
        context,
        np.repeat(context[:, -1:, :, :], repeats=1, axis=1),
        ego_len,
        adv_len,
        rss_cfg,
    )
    last_ego = context[:, -1, 0]
    last_lead = context[:, -1, 1]
    initial_gap = last_lead[:, 0] - last_ego[:, 0] - 0.5 * (ego_len + adv_len)
    initial_closing = last_ego[:, 2] - last_lead[:, 2]
    initial_ttc = np.where(initial_closing > 1e-6, initial_gap / np.maximum(initial_closing, 1e-6), 1000.0)
    initial_min_gap = initial_gap.astype(np.float32)
    initial_min_ttc = np.clip(initial_ttc, 0.0, 1000.0).astype(np.float32)
    initial_min_rss = initial_metrics["initial_rss_margin"].astype(np.float32)
    score = np.zeros(len(idx), dtype=np.float32)
    min_gap = np.full(len(idx), np.inf, dtype=np.float32)
    min_ttc = np.full(len(idx), np.inf, dtype=np.float32)
    min_rss = np.full(len(idx), np.inf, dtype=np.float32)
    if "recorded_future" in sources:
        recorded_metrics = interaction_metrics_from_states(context, raw["future_states"][idx], ego_len, adv_len, rss_cfg)
        score += _score(
            recorded_metrics["min_rss_margin"],
            recorded_metrics["min_ttc"],
            recorded_metrics["min_gap"],
            initial_metrics["initial_closing_speed"],
            args,
        )
        min_gap = np.minimum(min_gap, recorded_metrics["min_gap"].astype(np.float32))
        min_ttc = np.minimum(min_ttc, recorded_metrics["min_ttc"].astype(np.float32))
        min_rss = np.minimum(min_rss, recorded_metrics["min_rss_margin"].astype(np.float32))
    else:
        recorded_metrics = initial_metrics
    if "initial_context" in sources:
        score += _score(
            initial_min_rss,
            initial_min_ttc,
            initial_min_gap,
            initial_metrics["initial_closing_speed"],
            args,
        )
        min_gap = np.minimum(min_gap, initial_min_gap)
        min_ttc = np.minimum(min_ttc, initial_min_ttc)
        min_rss = np.minimum(min_rss, initial_min_rss)
    prior_metric_map: dict[int, dict[str, float]] = {}
    if "frozen_prior_rollout" in sources and int(args.frozen_prior_rollouts) > 0:
        prior_idx = idx[: int(args.frozen_prior_rollouts)]
        prior_metric_map = _frozen_prior_metrics(cfg, cfg_path.parent, raw, prior_idx, seed=int(args.seed) + 10000)
        for pos, dataset_idx in enumerate(idx):
            prior_metrics = prior_metric_map.get(int(dataset_idx))
            if prior_metrics is None:
                continue
            prior_gap = float(prior_metrics.get("min_gap", min_gap[pos]))
            prior_ttc = float(prior_metrics.get("min_ttc", min_ttc[pos]))
            prior_rss = float(prior_metrics.get("min_rss_margin", min_rss[pos]))
            score[pos] += float(
                _score(
                    np.asarray([prior_rss], dtype=np.float32),
                    np.asarray([prior_ttc], dtype=np.float32),
                    np.asarray([prior_gap], dtype=np.float32),
                    np.asarray([initial_metrics["initial_closing_speed"][pos]], dtype=np.float32),
                    args,
                )[0]
            )
            min_gap[pos] = min(float(min_gap[pos]), prior_gap)
            min_ttc[pos] = min(float(min_ttc[pos]), prior_ttc)
            min_rss[pos] = min(float(min_rss[pos]), prior_rss)
    min_gap = np.where(np.isfinite(min_gap), min_gap, initial_min_gap).astype(np.float32)
    min_ttc = np.where(np.isfinite(min_ttc), min_ttc, initial_min_ttc).astype(np.float32)
    min_rss = np.where(np.isfinite(min_rss), min_rss, initial_min_rss).astype(np.float32)
    threshold = float(np.quantile(score[np.isfinite(score)], float(args.tail_quantile)))
    tail_survival_probability, tail_info = _fit_tail_probability(score, threshold)
    exceedance = np.maximum(score - threshold, 0.0).astype(np.float32)
    max_exceedance = float(np.max(exceedance)) if exceedance.size else 0.0
    tail_extremeness = exceedance / max(max_exceedance, 1e-6)
    raw_weight = 1.0 + tail_extremeness + exceedance / max(float(np.mean(exceedance[exceedance > 0.0])) if np.any(exceedance > 0.0) else 1.0, 1e-6)
    raw_weight[score < threshold] = 1e-3
    tail_weight = raw_weight / max(float(np.mean(raw_weight)), 1e-6)

    recording = raw["recording_id"][idx] if "recording_id" in raw else np.full(len(idx), -1)
    event = raw["event_id"][idx] if "event_id" in raw else np.full(len(idx), "")
    anchor = raw["anchor_frame"][idx] if "anchor_frame" in raw else np.full(len(idx), -1)
    rows: list[dict[str, Any]] = []
    for pos, dataset_idx in enumerate(idx):
        rows.append(
            {
                "dataset_index": int(dataset_idx),
                "recording_id": int(recording[pos]) if np.issubdtype(np.asarray(recording).dtype, np.number) else str(recording[pos]),
                "event_id": str(event[pos]),
                "anchor_frame": int(anchor[pos]) if np.issubdtype(np.asarray(anchor).dtype, np.number) else str(anchor[pos]),
                "initial_gap": float(initial_metrics["initial_gap"][pos]),
                "initial_closing_speed": float(initial_metrics["initial_closing_speed"][pos]),
                "recorded_min_gap": float(recorded_metrics["min_gap"][pos]),
                "recorded_min_ttc": float(recorded_metrics["min_ttc"][pos]),
                "recorded_min_rss_margin": float(recorded_metrics["min_rss_margin"][pos]),
                "min_gap": float(min_gap[pos]),
                "min_ttc": float(min_ttc[pos]),
                "min_rss_margin": float(min_rss[pos]),
                "criticality_score": float(score[pos]),
                "tail_threshold": float(threshold),
                "tail_exceedance": float(exceedance[pos]),
                "tail_probability": float(tail_survival_probability[pos]),
                "tail_survival_probability": float(tail_survival_probability[pos]),
                "tail_extremeness": float(tail_extremeness[pos]),
                "tail_weight": float(tail_weight[pos]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "context_tail_scores.npz",
        dataset_index=idx.astype(np.int64),
        recording_id=np.asarray(recording),
        event_id=np.asarray(event),
        anchor_frame=np.asarray(anchor),
        initial_gap=initial_metrics["initial_gap"].astype(np.float32),
        initial_closing_speed=initial_metrics["initial_closing_speed"].astype(np.float32),
        recorded_min_gap=recorded_metrics["min_gap"].astype(np.float32),
        recorded_min_ttc=recorded_metrics["min_ttc"].astype(np.float32),
        recorded_min_rss_margin=recorded_metrics["min_rss_margin"].astype(np.float32),
        min_gap=min_gap.astype(np.float32),
        min_ttc=min_ttc.astype(np.float32),
        min_rss_margin=min_rss.astype(np.float32),
        criticality_score=score.astype(np.float32),
        tail_threshold=np.full(len(idx), threshold, dtype=np.float32),
        tail_exceedance=exceedance.astype(np.float32),
        tail_probability=tail_survival_probability.astype(np.float32),
        tail_survival_probability=tail_survival_probability.astype(np.float32),
        tail_extremeness=tail_extremeness.astype(np.float32),
        tail_weight=tail_weight.astype(np.float32),
    )
    write_csv(output_dir / "context_tail_scores.csv", rows)
    write_json(
        output_dir / "tail_score_summary.json",
        {
            "dataset": str(dataset_path),
            "schema_action_representation": schema.get("action_representation"),
            "split": str(args.split),
            "sources": sorted(sources),
            "num_contexts": int(len(idx)),
            "tail_quantile": float(args.tail_quantile),
            "tail_threshold": float(threshold),
            "tail_fraction": float(np.mean(score >= threshold)),
            "score_mean": float(np.mean(score)),
            "score_p95": float(np.percentile(score, 95.0)),
            "tail_fit": tail_info,
        },
    )


if __name__ == "__main__":
    main()
