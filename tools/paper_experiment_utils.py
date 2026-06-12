"""Shared helpers for paper experiment post-processing scripts."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")
    return True


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_table(table_dir: Path, root: Path, base: str, rows: list[dict[str, Any]], *, force: bool) -> list[str]:
    csv_path = table_dir / f"{base}.csv"
    if rows and (force or not csv_path.exists()):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return [rel_path(csv_path, root)]


def fget(mapping: dict[str, Any], key: str, default: Any = math.nan) -> Any:
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def nested(mapping: dict[str, Any], *keys: str, default: Any = math.nan) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(values: list[dict[str, str]], key: str) -> np.ndarray:
    out: list[float] = []
    for row in values:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    arr = np.asarray(out, dtype=float)
    return arr[np.isfinite(arr)]


def fraction_true(values: list[dict[str, str]], key: str) -> float:
    arr = as_float(values, key)
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr > 0.0))


def save_figure(fig: Any, path: Path, root: Path, *, force: bool) -> list[str]:
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return [rel_path(path, root)]


def gpd_survival(y: np.ndarray, *, u: float, xi: float, beta: float, exceedance_rate: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    z = np.maximum(y - u, 0.0)
    if abs(xi) < 1e-10:
        tail = np.exp(-z / beta)
    else:
        tail = np.power(np.maximum(1.0 + xi * z / beta, 1e-300), -1.0 / xi)
    return exceedance_rate * tail


def record(
    manifest: dict[str, Any],
    key: str,
    *,
    status: str,
    outputs: list[str] | None = None,
    sources: list[str] | None = None,
    skipped_reason: str | None = None,
    notes: str | None = None,
) -> None:
    manifest["experiments"][key] = {
        "status": status,
        "outputs": outputs or [],
        "source_artifacts": sources or [],
        "skipped_reason": skipped_reason,
        "notes": notes,
    }


def build_manifest(scope: str, created_by: str, root: Path, source_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "experiment_scope": scope,
        "created_by": created_by,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_training_or_rerun_statement": (
            "Only existing JSON/CSV/NPZ/PNG results were read; "
            "no training, EVT fitting, or subset simulation was rerun."
        ),
        "source_files": {key: rel_path(path, root) for key, path in source_paths.items() if path.exists()},
        "missing_source_files": {key: rel_path(path, root) for key, path in source_paths.items() if not path.exists()},
        "experiments": {},
    }


def write_experiment_readme(
    path: Path,
    manifest: dict[str, Any],
    *,
    title: str,
    description: str,
    no_rerun_note: str,
    interpretation_notes: list[str],
    force: bool,
) -> None:
    if path.exists() and not force:
        return

    generated: list[str] = []
    reused: list[str] = []
    skipped: list[str] = []
    for name, exp in manifest["experiments"].items():
        generated.extend(exp.get("outputs", []))
        if exp.get("source_artifacts"):
            reused.extend(f"{name}: {src}" for src in exp["source_artifacts"] if src)
        if exp.get("skipped_reason"):
            skipped.append(f"{name}: {exp['skipped_reason']}")

    lines = [
        f"# {title}",
        "",
        description,
        no_rerun_note,
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in manifest["source_files"].items())
    lines.extend(["", "## Generated Artifacts", ""])
    lines.extend(f"- `{artifact}`" for artifact in generated)
    lines.extend(["", "## Reused Existing Artifacts", ""])
    lines.extend(f"- reused existing artifact: `{item}`" for item in reused)
    if not reused:
        lines.append("- None")
    lines.extend(["", "## Skipped Artifacts", ""])
    lines.extend(f"- {item}" for item in skipped)
    if not skipped:
        lines.append("- None")
    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in interpretation_notes)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
