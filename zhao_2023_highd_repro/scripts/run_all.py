#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_config
from src.data import build_emergency_cutin_dataset
from src.generator import train_generator
from src.paper_figures import plot_all_paper_figures
from src.scenario import build_dangerous_scenarios, run_idm_evaluation
from src.visualize import plot_case


def main() -> None:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    outputs = {}
    outputs.update(build_emergency_cutin_dataset(config))
    outputs.update(train_generator(config))
    outputs.update(build_dangerous_scenarios(config))
    outputs.update(run_idm_evaluation(config))
    outputs["figure"] = plot_case(config, 0)
    outputs.update(plot_all_paper_figures(config))
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
