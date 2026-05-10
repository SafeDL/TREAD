#!/usr/bin/env python
"""
highd_events.py — 手动分阶段整理 highD 事件数据
================================================
用法:
  python scripts/highd_events.py finalize-events
  python scripts/highd_events.py build-artifacts
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running either from the repository root or from tread_highd/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_highd.src.filtering import assign_risk_percentiles, filter_events
from tread_highd.src.io_utils import load_config, resolve_data_path
from tread_highd.src.schema import EventRecord


def _default_config() -> str:
    return str(Path(__file__).resolve().parent / "configs" / "highd_default.yaml")


def _load_intermediate_events(out_dir: Path) -> pd.DataFrame:
    parts = []
    for name in ["candidate_events.csv", "invalid_events.csv"]:
        path = out_dir / "intermediate" / name
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(
            f"No intermediate events found under {out_dir / 'intermediate'}. "
            "Run 01_extract_highd_events.py first."
        )
    return pd.concat(parts, ignore_index=True)


def finalize_events(config_path: str) -> Path:
    cfg = load_config(config_path)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["processed_dir"], config_path)))

    events_df = _load_intermediate_events(out_dir)
    events_df = filter_events(events_df, cfg)
    events_df = assign_risk_percentiles(
        events_df,
        cfg.get("risk", {}).get("tail_quantiles", [0.90, 0.95, 0.99]),
    )

    out_path = out_dir / "events.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out_path, index=False)
    return out_path


def _clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _events_from_dataframe(events_df: pd.DataFrame) -> list[EventRecord]:
    field_names = {f.name for f in fields(EventRecord)}
    events = []
    for _, row in events_df.iterrows():
        kwargs = {
            key: _clean_value(row[key])
            for key in events_df.columns
            if key in field_names
        }
        events.append(EventRecord(**kwargs))
    return events


def build_artifacts(config_path: str) -> None:
    from tread_highd.src.dataset_builder import HighDTailRiskDatasetBuilder

    cfg = load_config(config_path)
    out_dir = Path(str(resolve_data_path(cfg["paths"]["processed_dir"], config_path)))
    events_path = out_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"{events_path} not found. Run finalize-events first.")

    events_df = pd.read_csv(events_path)
    events = _events_from_dataframe(events_df)

    builder = HighDTailRiskDatasetBuilder(config_path)
    arrays = builder.build_trajectory_arrays(events, events_df)
    splits = builder.build_splits(events_df)
    builder.export(events_df, arrays, splits)


def main():
    parser = argparse.ArgumentParser(description="TREAD: staged highD event utilities")
    parser.add_argument(
        "command",
        choices=["finalize-events", "build-artifacts"],
        help="Stage to run.",
    )
    parser.add_argument("--config", default=_default_config(), help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "finalize-events":
        out_path = finalize_events(args.config)
        print(f"Saved {out_path}")
    elif args.command == "build-artifacts":
        build_artifacts(args.config)
        print("Saved trajectories.h5, splits.json, and normalization_stats.json")


if __name__ == "__main__":
    main()
