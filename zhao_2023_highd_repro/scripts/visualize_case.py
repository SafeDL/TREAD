#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_config
from src.visualize import assert_highway_env_available, plot_case


def main() -> None:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    case_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"highway_env: {assert_highway_env_available()}")
    print(f"figure: {plot_case(config, case_index)}")


if __name__ == "__main__":
    main()
