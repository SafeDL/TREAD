"""Publish validated multi-policy highD results into policy-specific result trees.

The canonical experiment remains under ``results/multi_policy_validation``.  This
publisher creates a versioned, policy-scoped projection and never changes a
legacy result folder or overwrites an existing publication directory.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "multi_policy_validation"
POLICIES = ("IDM", "A2C", "PPO", "SAIRL")
EXPERIMENT_ID = "multi_policy_validation"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite published artifact: {target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _link_or_copy_tree(source: Path, target: Path) -> dict[str, int]:
    if not source.is_dir():
        raise NotADirectoryError(source)
    linked = copied = 0
    for item in sorted(path for path in source.rglob("*") if path.is_file()):
        mode = _link_or_copy(item, target / item.relative_to(source))
        if mode == "hardlink":
            linked += 1
        else:
            copied += 1
    return {"hardlinked_files": linked, "copied_files": copied}


def _validate_source() -> tuple[dict[str, Any], dict[str, tuple[tuple[str, ...], list[dict[str, str]]]]]:
    status = _read_json(SOURCE / "summary_status.json")
    if not status.get("has_valid_multi_ads_result") or status.get("fairness_status") != "pass":
        raise RuntimeError("Canonical source result is not approved for publication.")
    playback_fields, playback_rows = _read_csv(SOURCE / "tables" / "multi_ads_playback_validation.csv")
    if len(playback_rows) != 4 or any(row.get("status") != "pass" for row in playback_rows):
        raise RuntimeError("Archived-config playback validation is incomplete or failed.")
    table_names = ("multi_ads_summary", "multi_ads_seed_level", "multi_ads_relative_risk")
    tables = {name: _read_csv(SOURCE / "tables" / f"{name}.csv") for name in table_names}
    summary_rows = tables["multi_ads_summary"][1]
    seed_rows = tables["multi_ads_seed_level"][1]
    for policy in POLICIES:
        policy_summary = [row for row in summary_rows if row.get("policy") == policy]
        policy_seeds = [row for row in seed_rows if row.get("policy") == policy]
        if len(policy_summary) != 2 or any(row.get("status") != "pass" for row in policy_summary):
            raise RuntimeError(f"Canonical summary is incomplete for {policy}.")
        if len(policy_seeds) != 10 or any(
            row.get("execution_status") != "completed" or row.get("quality_status") != "pass"
            for row in policy_seeds
        ):
            raise RuntimeError(f"Canonical SS seed results are incomplete for {policy}.")
        if not (SOURCE / "ads" / policy.lower()).is_dir() or not (SOURCE / "references" / policy.lower()).is_dir():
            raise RuntimeError(f"Canonical artifacts are missing for {policy}.")
    tables["multi_ads_playback_validation"] = (playback_fields, playback_rows)
    return status, tables


def _publication_readme(policy: str) -> str:
    return (
        f"# {policy} multi-policy validation results\n\n"
        f"This is the policy-scoped publication of `results/{EXPERIMENT_ID}`. It contains both event "
        "types, all five subset-simulation seeds, and independent high-budget Monte Carlo "
        "references.\n\n"
        "The experiment passed frozen-protocol fairness, all seed-level reliability checks, "
        "and cross-seed/MC interval-overlap checks. The two highest-risk policies in each "
        "event were replayed from archived effective configurations; all 20 replayed cases "
        "passed with a maximum risk-score drift no greater than `1e-6`.\n\n"
        "Legacy result folders are deliberately preserved. This versioned directory is the "
        "replacement target for the full multi-policy result set.\n"
    )


def _publish_policy(policy: str, status: dict[str, Any], tables: dict[str, tuple[tuple[str, ...], list[dict[str, str]]]]) -> Path:
    target = ROOT / f"{policy}_subset" / "results" / EXPERIMENT_ID
    if target.exists():
        raise FileExistsError(f"Publication target already exists: {target}. It is intentionally not overwritten.")
    target.mkdir(parents=True)
    artifact_stats = {
        "ads": _link_or_copy_tree(SOURCE / "ads" / policy.lower(), target / "ads"),
        "references": _link_or_copy_tree(SOURCE / "references" / policy.lower(), target / "references"),
    }
    for table_name in ("multi_ads_summary", "multi_ads_seed_level"):
        fields, rows = tables[table_name]
        _write_csv(target / "tables" / f"{table_name}.csv", fields, [row for row in rows if row.get("policy") == policy])
    for table_name in ("multi_ads_relative_risk", "multi_ads_playback_validation"):
        fields, rows = tables[table_name]
        _write_csv(target / "tables" / f"{table_name}.csv", fields, rows)
    for figure in ("multi_ads_probability.png", "multi_ads_return_mileage.png"):
        _link_or_copy(SOURCE / "figures" / figure, target / "figures" / figure)
    for audit_name in ("experiment_manifest.json", "frozen_inputs.json", "fairness_check.json", "summary_status.json", "run_plan.csv", "reference_plan.csv"):
        _link_or_copy(SOURCE / audit_name, target / "audit" / audit_name)
    _link_or_copy(SOURCE / "playbacks" / "playback_provenance.json", target / "audit" / "playback_provenance.json")
    (target / "README.md").write_text(_publication_readme(policy), encoding="utf-8", newline="\n")
    _write_json(
        target / "publication_status.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "policy": policy,
            "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "canonical_source": SOURCE.relative_to(ROOT).as_posix(),
            "canonical_status_valid": bool(status.get("has_valid_multi_ads_result")),
            "fairness_status": status.get("fairness_status"),
            "artifact_transfer": artifact_stats,
            "baseline_results_preserved": True,
        },
    )
    return target


def main() -> None:
    status, tables = _validate_source()
    for policy in POLICIES:
        print(_publish_policy(policy, status, tables).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
