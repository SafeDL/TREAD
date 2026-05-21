#!/usr/bin/env python3
"""Plot Stage 1 shared proposal scenario bank diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR = Path(__file__).resolve().parents[1] / "stage1"
for item in (ROOT, STAGE1_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from diagnose_stage1_scenario_bank import DEFAULT_CONFIG_PATH, _load_npz, _resolve, _stage1_cfg, write_figures
from diffusion.src.utils import load_json, load_yaml, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument("--scenario-bank", default="", help="Override the scenario bank npz path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    base = cfg_path.parent
    stage1 = _stage1_cfg(cfg)
    output_dir = _resolve(stage1["output_dir"], base)
    bank_path = (
        _resolve(args.scenario_bank, base)
        if str(args.scenario_bank or "").strip()
        else output_dir / str(stage1["scenario_bank"].get("output_name", "scenario_bank.npz"))
    )
    if not bank_path.exists():
        raise FileNotFoundError(f"Scenario bank not found: {bank_path}")
    natural_dir = _resolve(cfg.get("paths", {}).get("natural_dataset_dir", ""), base)
    schema = load_json(natural_dir / "feature_schema.json")
    write_figures(_load_npz(bank_path), schema, cfg, output_dir / "figures")


if __name__ == "__main__":
    main()
