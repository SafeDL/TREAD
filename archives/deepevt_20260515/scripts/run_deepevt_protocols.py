#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run DeepEVT robustness protocols:

1. repeated recording-level train/val/test splits;
2. leave-location-out evaluation folds.

Each protocol writes an independent output directory so normalization,
checkpoint selection, and test evaluation are isolated per split.
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from process_highd.src.io_utils import ensure_dir, load_config, load_json, resolve_data_path, save_json
from tread_deepevt.src.data import build_and_save_dataset
from tread_deepevt.src.evaluate import evaluate_deepevt
from tread_deepevt.src.inference import export_tail_conditions
from tread_deepevt.src.train import train_deepevt

logger = logging.getLogger(__name__)


def _as_list_of_int_lists(value: Any) -> List[List[int]]:
    if not value:
        return []
    out: List[List[int]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.append([int(x) for x in item])
        else:
            out.append([int(item)])
    return out


def _available_recordings(events_csv: Path, event_type: str) -> List[int]:
    df = pd.read_csv(events_csv)
    if "event_type" in df.columns:
        df = df[df["event_type"] == event_type]
    return sorted(int(x) for x in df["recording_id"].dropna().unique())


def _recording_locations(raw_dir: Path, recording_ids: Iterable[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for rid in sorted({int(x) for x in recording_ids}):
        path = raw_dir / f"{rid:02d}_recordingMeta.csv"
        if not path.exists():
            path = raw_dir / f"{rid}_recordingMeta.csv"
        if not path.exists():
            continue
        meta = pd.read_csv(path, nrows=1)
        if "locationId" in meta.columns and len(meta) > 0:
            out[rid] = int(meta.loc[0, "locationId"])
    return out


def _default_output_root(base_output: Path) -> Path:
    return base_output.parent / f"{base_output.name}_protocols"


def _run_configs(
    base_cfg: dict,
    config_path: str,
    *,
    protocols: set[str],
) -> List[dict]:
    raw_dir = Path(resolve_data_path(base_cfg["paths"]["raw_dir"], config_path))
    events_csv = Path(resolve_data_path(base_cfg["paths"]["events_csv"], config_path))
    base_output = Path(resolve_data_path(base_cfg["paths"]["output_dir"], config_path))
    robust_cfg = base_cfg.get("robustness", {})
    output_root = Path(
        resolve_data_path(str(robust_cfg.get("output_root", _default_output_root(base_output))), config_path)
    )
    event_type = str(base_cfg.get("event", {}).get("event_type", "following"))
    recording_ids = _available_recordings(events_csv, event_type)
    runs: List[dict] = []

    if "repeated" in protocols:
        repeated_cfg = robust_cfg.get("repeated_recording", {})
        seeds = [int(x) for x in repeated_cfg.get("seeds", [11, 22, 33, 44, 55])]
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["splits"] = dict(cfg.get("splits", {}), strategy="recording", random_seed=seed)
            cfg["paths"] = dict(cfg["paths"], output_dir=str(output_root / f"repeated_seed_{seed}"))
            runs.append({"name": f"repeated_seed_{seed}", "protocol": "repeated_recording", "config": cfg})

    if "leave-location" in protocols:
        leave_loc_cfg = robust_cfg.get("leave_location_out", {})
        folds = _as_list_of_int_lists(leave_loc_cfg.get("test_location_ids", []))
        if bool(leave_loc_cfg.get("auto", True)) and not folds:
            locs = sorted(set(_recording_locations(raw_dir, recording_ids).values()))
            folds = [[loc] for loc in locs]
        for i, test_locs in enumerate(folds, start=1):
            cfg = copy.deepcopy(base_cfg)
            cfg["splits"] = dict(
                cfg.get("splits", {}),
                strategy="leave_location_out",
                test_location_ids=test_locs,
            )
            label = "_".join(str(x) for x in test_locs)
            cfg["paths"] = dict(cfg["paths"], output_dir=str(output_root / f"leave_location_{i:02d}_{label}"))
            runs.append({
                "name": f"leave_location_{i:02d}_{label}",
                "protocol": "leave_location_out",
                "test_location_ids": test_locs,
                "config": cfg,
            })

    return runs


def _levels_from_cfg(cfg: dict) -> tuple[float, ...]:
    training_cfg = cfg.get("training", {})
    return tuple(
        float(x)
        for x in training_cfg.get(
            "eval_tail_levels",
            training_cfg.get("quantile_levels", [0.85, 0.90, 0.95]),
        )
    )


def _extract_scalar_metrics(report: dict, split_payload: dict, run: dict) -> dict:
    primary = report["primary_task"]
    out: Dict[str, Any] = {
        "name": run["name"],
        "protocol": run["protocol"],
        "n_train": int(report.get("n_train", 0)),
        "n_val": int(report.get("n_val", 0)),
        "n_test": int(report.get("n_test", 0)),
        "train_recording_ids": split_payload.get("train_recording_ids", []),
        "val_recording_ids": split_payload.get("val_recording_ids", []),
        "test_recording_ids": split_payload.get("test_recording_ids", []),
    }
    if "test_location_ids" in split_payload:
        out["test_location_ids"] = split_payload["test_location_ids"]
    for tau_key, metrics in primary["test_raw_direct"]["tail_levels"].items():
        suffix = tau_key.replace("tau_", "q").replace(".", "")
        out[f"{suffix}_ece"] = float(metrics["ece"])
        out[f"{suffix}_exceed"] = float(metrics["empirical_exceed_rate"])
    for tau_key, metrics in primary["ranking_diagnostics"]["test"]["tail_levels"].items():
        suffix = tau_key.replace("tau_", "q").replace(".", "")
        out[f"{suffix}_spearman"] = float(metrics["spearman_predicted_q_vs_risk"])
        out[f"{suffix}_auc"] = float(metrics["tail_label_auc"])
        enrich = metrics.get("topk_enrichment", [])
        top5 = next((x for x in enrich if abs(float(x["top_fraction"]) - 0.05) < 1e-9), None)
        if top5:
            out[f"{suffix}_top5_enrichment"] = float(top5["enrichment"])
            out[f"{suffix}_top5_tail_rate"] = float(top5["tail_label_rate"])
    return out


def _summarize_runs(rows: List[dict]) -> dict:
    numeric_keys = sorted(
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    summary: Dict[str, dict] = {}
    for key in numeric_keys:
        vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(vals.size),
        }
    return summary


def run_protocol_suite(
    config_path: str,
    *,
    protocols: set[str],
    stages: set[str],
    dry_run: bool = False,
) -> dict:
    base_cfg = load_config(config_path)
    event_type = base_cfg.get("event", {}).get("event_type")
    if event_type not in {"following", "cut_in"}:
        raise ValueError(f"config event.event_type must be following/cut_in, got {event_type}")

    runs = _run_configs(base_cfg, config_path, protocols=protocols)
    if not runs:
        raise RuntimeError("No protocol runs were generated.")

    if dry_run:
        return {"runs": [{k: v for k, v in run.items() if k != "config"} for run in runs]}

    raw_dir = resolve_data_path(base_cfg["paths"]["raw_dir"], config_path)
    events_csv = resolve_data_path(base_cfg["paths"]["events_csv"], config_path)
    rows: List[dict] = []
    for run in runs:
        cfg = run["config"]
        out_dir = Path(cfg["paths"]["output_dir"])
        ensure_dir(out_dir)
        logger.info("[%s] output_dir=%s", run["name"], out_dir)

        if "build" in stages:
            build_and_save_dataset(
                events_csv=events_csv,
                raw_dir=raw_dir,
                config=cfg,
                output_dir=out_dir,
                event_type=event_type,
            )
        if "train" in stages:
            train_deepevt(output_dir=out_dir, config=cfg)
        if "evaluate" in stages:
            evaluate_deepevt(
                output_dir=out_dir,
                checkpoint_path=out_dir / "best_model.pt",
                config=cfg,
                tail_levels=_levels_from_cfg(cfg),
            )
            report = load_json(out_dir / "eval_report.json")
            split_payload = load_json(out_dir / "train_val_test_split.json")
            rows.append(_extract_scalar_metrics(report, split_payload, run))
        if "export" in stages:
            export_tail_conditions(
                output_dir=out_dir,
                checkpoint_path=out_dir / "best_model.pt",
                tail_levels=_levels_from_cfg(cfg),
                include_context_features=True,
                config=cfg,
            )

    root = Path(runs[0]["config"]["paths"]["output_dir"]).parent
    payload = {
        "protocols": sorted(protocols),
        "stages": sorted(stages),
        "runs": rows,
        "summary": _summarize_runs(rows),
    }
    save_json(payload, root / "protocol_summary.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEVT split robustness protocols")
    default_cfg = Path(__file__).resolve().parent / "configs" / "deepevt_following.yaml"
    parser.add_argument("--config", default=str(default_cfg))
    parser.add_argument(
        "--protocols",
        default="repeated",
        help="Comma-separated: repeated,leave-location",
    )
    parser.add_argument(
        "--stages",
        default="build,train,evaluate",
        help="Comma-separated: build,train,evaluate,export",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    protocols = {x.strip() for x in args.protocols.split(",") if x.strip()}
    stages = {x.strip() for x in args.stages.split(",") if x.strip()}
    payload = run_protocol_suite(args.config, protocols=protocols, stages=stages, dry_run=args.dry_run)
    if args.dry_run:
        for run in payload["runs"]:
            logger.info("dry-run: %s", run)


if __name__ == "__main__":
    main()
