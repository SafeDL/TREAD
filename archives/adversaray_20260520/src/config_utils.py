"""Small config helpers for adversaray scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from diffusion.src.utils import load_yaml


def apply_rss_config_override(cfg: dict[str, Any], config_dir: Path) -> None:
    rss_config = str(cfg.get("paths", {}).get("rss_config", "") or "")
    if not rss_config:
        return
    path = Path(rss_config)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if path.exists():
        recommended = load_yaml(path)
        cfg.setdefault("rss", {}).update(recommended.get("rss", recommended))
