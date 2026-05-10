#!/usr/bin/env python
"""
04_generate_quality_report.py — 生成质量报告
=============================================
用法:
  conda activate jzm
  python scripts/04_generate_quality_report.py --config configs/highd_default.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tread_highd.io_utils import load_config, resolve_data_path
from tread_highd.quality_check import generate_quality_report


def main():
    parser = argparse.ArgumentParser(description="TREAD: Generate quality report")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    out_dir = str(resolve_data_path(cfg["paths"]["processed_dir"], args.config))

    events_path = Path(out_dir) / "events.csv"
    if not events_path.exists():
        print(f"events.csv not found. Run 02_build first.")
        return

    df = pd.read_csv(events_path)
    report = generate_quality_report(df, out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
