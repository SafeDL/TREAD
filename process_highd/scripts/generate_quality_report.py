#!/usr/bin/env python
"""
generate_quality_report.py — 生成质量报告
=============================================
用法:
  conda activate jzm
  python scripts/generate_quality_report.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running either from the repository root or from tread_highd/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from process_highd.src.io_utils import load_config, resolve_data_path
from process_highd.src.quality_check import generate_quality_report


def main():
    parser = argparse.ArgumentParser(description="TREAD: Generate quality report")
    default_config = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    out_dir = str(resolve_data_path(cfg["paths"]["output_dir"], args.config))

    events_path = Path(out_dir) / "events.csv"
    if not events_path.exists():
        print("events.csv not found. Run extract_highd_events.py first.")
        return

    df = pd.read_csv(events_path)
    report = generate_quality_report(df, out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
