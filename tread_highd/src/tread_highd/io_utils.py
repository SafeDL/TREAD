"""
io_utils.py — I/O 工具
=======================
YAML 配置加载、路径管理等通用 I/O 辅助函数。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件并返回字典。

    Parameters
    ----------
    config_path : str
        YAML 文件的路径。

    Returns
    -------
    dict
        解析后的配置字典。
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("已加载配置: %s", config_path)
    return cfg


# ──────────────────────────────────────────────────────────
# 路径辅助
# ──────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """若目录不存在则创建并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_data_path(raw_dir: str, config_path: str | None = None) -> Path:
    """将 raw_dir 解析为绝对路径。

    如果 raw_dir 是相对路径，则基于 config_path 所在目录进行解析。
    """
    raw = Path(raw_dir)
    if raw.is_absolute():
        return raw
    if config_path is not None:
        base = Path(config_path).resolve().parent
    else:
        base = Path.cwd()
    return (base / raw).resolve()


# ──────────────────────────────────────────────────────────
# JSON 存取
# ──────────────────────────────────────────────────────────

def save_json(data: Any, path: str | Path) -> None:
    """将 Python 对象保存为 JSON 文件。"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("已保存 JSON: %s", path)


def load_json(path: str | Path) -> Any:
    """从 JSON 文件加载 Python 对象。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
