#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_config
from src.generator import train_generator


def main() -> None:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    paths = train_generator(config)
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
