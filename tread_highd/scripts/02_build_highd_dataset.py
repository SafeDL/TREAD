#!/usr/bin/env python
"""
02_build_highd_dataset.py — 构建完整训练数据集
===============================================
用法:
  conda activate jzm
  python scripts/02_build_highd_dataset.py --config configs/highd_default.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tread_highd.dataset_builder import HighDTailRiskDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="TREAD: Build highD dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    builder = HighDTailRiskDatasetBuilder(args.config)
    builder.run()


if __name__ == "__main__":
    main()
