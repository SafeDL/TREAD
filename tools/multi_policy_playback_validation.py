"""Replay the highest-risk completed cases using each run's archived config.

This is intentionally separate from ``multi_policy_validation.py``.  The
multi-ADS driver freezes live source inputs before sampling and must reject a
new invocation if an upstream YAML changes.  This helper instead replays the
saved sample plans with the ``effective_config.json`` stored alongside the
completed run, so the validation remains tied to the exact experiment rather
than to subsequently edited source configuration files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "results" / "multi_policy_validation"
POLICY_PACKAGES = {
    "IDM": "IDM_subset",
    "A2C": "A2C_subset",
    "PPO": "PPO_subset",
    "SAIRL": "SAIRL_subset",
}
SCORE_DRIFT_TOLERANCE = 1e-6
LOGGER = logging.getLogger(__name__)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=base.resolve())).as_posix()


def _number(value: str | None, default: float = -math.inf) -> float:
    try:
        return float(value) if value not in {None, ""} else default
    except ValueError:
        return default


def _expected_event(event: str) -> str:
    return "cut_in" if event == "cutin" else "following"


def _select_cases() -> list[dict[str, str]]:
    summary_rows = _read_csv(RESULTS_ROOT / "tables" / "multi_ads_summary.csv")
    seed_rows = _read_csv(RESULTS_ROOT / "tables" / "multi_ads_seed_level.csv")
    selections: list[dict[str, str]] = []
    for event in ("following", "cutin"):
        candidates = [
            row
            for row in summary_rows
            if row.get("event_type") == event
            and row.get("completed_seeds") == "5"
            and row.get("status") == "pass"
        ]
        candidates.sort(key=lambda row: _number(row.get("mean_probability")), reverse=True)
        if len(candidates) < 2:
            raise RuntimeError(f"Need two completed policy results for {event}; found {len(candidates)}")
        for candidate in candidates[:2]:
            policy = candidate["policy"]
            policy_seeds = [
                row
                for row in seed_rows
                if row.get("policy") == policy
                and row.get("event_type") == event
                and row.get("execution_status") == "completed"
                and row.get("quality_status") == "pass"
            ]
            if not policy_seeds:
                raise RuntimeError(f"No completed seed found for {policy}/{event}")
            selections.append(max(policy_seeds, key=lambda row: _number(row.get("probability"))))
    return selections


def _replay(selection: dict[str, str]) -> dict[str, Any]:
    policy = selection["policy"]
    event = selection["event_type"]
    seed = int(selection["seed"])
    package = POLICY_PACKAGES[policy]
    config_path = ROOT / selection["effective_config_path"]
    sample_path = config_path.parent / "latent_subset_samples.npz"
    if not config_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError(f"Missing archived replay input for {policy}/{event}/seed_{seed}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    observed_hash = _canonical_hash(config)
    file_hash = _sha256(config_path)
    expected_hash = selection.get("effective_config_sha256", "")
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Archived effective config hash mismatch for {policy}/{event}/seed_{seed}: "
            f"expected {expected_hash}, got {observed_hash}"
        )

    config_dir = ROOT / package / "scripts" / "configs"
    output_dir = RESULTS_ROOT / "playbacks" / event / policy.lower() / f"seed_{seed}"
    from tools import final_level_playback as module

    original_defaults = dict(module.SCRIPT_DEFAULTS)
    try:
        module.SCRIPT_DEFAULTS.update(
            {
                "samples_path": _relative(sample_path, config_dir),
                "output_dir": _relative(output_dir, config_dir),
                "num_cases": 5,
                "random_seed": 20260804,
                "level": -1,
                "render_gif": False,
                "unique_test_scenarios": True,
            }
        )
        manifest_path = module.replay_final_level(
            config,
            config_dir,
            subset_name=package,
            expected_event_type=_expected_event(event),
        )
    finally:
        module.SCRIPT_DEFAULTS.clear()
        module.SCRIPT_DEFAULTS.update(original_defaults)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    drifts = [
        abs(float(case["subset_score"]) - float(case["replay_risk"]))
        for case in manifest.get("cases", [])
        if math.isfinite(float(case.get("subset_score", math.nan)))
        and math.isfinite(float(case.get("replay_risk", math.nan)))
    ]
    if len(drifts) != int(manifest.get("num_cases", 0)) or not drifts:
        raise RuntimeError(f"Playback for {policy}/{event}/seed_{seed} lacks finite score comparisons")
    max_drift = max(drifts)
    return {
        "policy": policy,
        "event_type": event,
        "seed": seed,
        "status": "pass" if max_drift <= SCORE_DRIFT_TOLERANCE else "fail",
        "num_cases": len(drifts),
        "max_score_drift": max_drift,
        "score_drift_tolerance": SCORE_DRIFT_TOLERANCE,
        "archived_effective_config": config_path.relative_to(ROOT).as_posix(),
        "archived_effective_config_canonical_sha256": observed_hash,
        "archived_effective_config_file_sha256": file_hash,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    status_path = RESULTS_ROOT / "summary_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if not status.get("has_valid_multi_ads_result"):
        raise RuntimeError("The source multi-ADS result is not marked valid; refusing playback publication.")

    rows: list[dict[str, Any]] = []
    for selection in _select_cases():
        try:
            rows.append(_replay(selection))
        except Exception as exc:
            LOGGER.exception("Playback failed for %s/%s seed=%s", selection["policy"], selection["event_type"], selection["seed"])
            rows.append(
                {
                    "policy": selection["policy"],
                    "event_type": selection["event_type"],
                    "seed": selection["seed"],
                    "status": "fail",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    fields = (
        "policy",
        "event_type",
        "seed",
        "status",
        "num_cases",
        "max_score_drift",
        "score_drift_tolerance",
        "archived_effective_config",
        "archived_effective_config_canonical_sha256",
        "archived_effective_config_file_sha256",
        "manifest_path",
        "failure_reason",
    )
    _write_csv(RESULTS_ROOT / "tables" / "multi_ads_playback_validation.csv", fields, rows)
    _write_json(
        RESULTS_ROOT / "playbacks" / "playback_provenance.json",
        {
            "source_result_status": "valid" if status.get("has_valid_multi_ads_result") else "invalid",
            "configuration_source": "archived effective_config.json from each completed run",
            "selection_rule": "two highest mean-risk policies per event, then highest-risk completed seed",
            "score_drift_tolerance": SCORE_DRIFT_TOLERANCE,
            "all_playbacks_pass": len(rows) == 4 and all(row.get("status") == "pass" for row in rows),
            "rows": rows,
        },
    )
    if len(rows) != 4 or any(row.get("status") != "pass" for row in rows):
        raise RuntimeError("At least one archived-config playback failed validation.")
    LOGGER.info("Validated %d archived-config playbacks.", len(rows))


if __name__ == "__main__":
    main()
