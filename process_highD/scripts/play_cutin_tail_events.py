#!/usr/bin/env python3
"""Render selected highD cut-in tail events to GIF."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_highD.src.event_playback import render_tail_event_gif


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_default.yaml"
TAIL_CONTEXTS_PATH = (
    ROOT / "results" / "highd_cutin_tail" / "contexts" / "tail_contexts.npz"
)
OUTPUT_DIR = (
    ROOT / "results" / "highd_cutin_tail" / "figures" / "event_playbacks"
)
OUTPUT_NAME = "tail_cutin_context_00000"

# "all": every tail context; int: random sample count; tuple/list: exact indices.
TAIL_CONTEXT_SELECTION: str | int | tuple[int, ...] = (0,)
RANDOM_SEED = 42

PRE_FRAMES = 0
POST_FRAMES = 0
VIEW_WIDTH = 160.0
NEIGHBOR_MARGIN = 20.0
TRAIL_FRAMES = 50
PLAYBACK_SPEED = 1.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    render_tail_event_gif(
        config_path=CONFIG_PATH,
        tail_contexts_path=TAIL_CONTEXTS_PATH,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME,
        event_type="cut_in",
        tail_context_selection=TAIL_CONTEXT_SELECTION,
        random_seed=RANDOM_SEED,
        pre_frames=PRE_FRAMES,
        post_frames=POST_FRAMES,
        view_width=VIEW_WIDTH,
        neighbor_margin=NEIGHBOR_MARGIN,
        trail_frames=TRAIL_FRAMES,
        playback_speed=PLAYBACK_SPEED,
    )


if __name__ == "__main__":
    main()
