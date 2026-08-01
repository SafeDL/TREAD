#!/usr/bin/env python3
"""Run the frozen highD IDM subset-simulation OAT sensitivity experiment.

The script deliberately calls the shared runner API instead of modifying the
default YAMLs or using the legacy entrypoints, so every repeat gets an isolated
output directory and an immutable effective-config snapshot.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from IDM_subset.experiments.highd_ss_sensitivity.sensitivity_spec import (
    BASE_CONFIGS,
    EVENTS,
    GRID,
    MC_REFERENCE_SIZES,
    RESULTS_ROOT,
    RunSpec,
    build_run_specs,
    defaults_from_config,
)
from IDM_subset.src.latent_subset_runner import (
    run_monte_carlo_from_config,
    run_subset_from_config,
)


LOGGER = logging.getLogger("highd_ss_sensitivity")
RUN_PLAN_PATH = RESULTS_ROOT / "run_plan.csv"
MANIFEST_PATH = RESULTS_ROOT / "experiment_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    content = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def _run_file_log(path: Path):
    """Mirror the shared runner's console logs into a per-run audit log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def _resolve_configured_path(value: str, base: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _input_fingerprints(
    configs: dict[str, dict[str, Any]]
) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for event_type, config in configs.items():
        config_path = BASE_CONFIGS[event_type]
        config_dir = config_path.parent
        paths = dict(config.get("paths", {}) or {})
        listed: dict[str, Path] = {"base_config": config_path}
        for key in (
            "diffusion_checkpoint",
            "tail_context_path",
            "condition_distribution_path",
            "evt_model_path",
            "exposure_summary_path",
            "idm_ego_config_path",
        ):
            if key in paths:
                listed[key] = _resolve_configured_path(str(paths[key]), config_dir)
        event_payload: dict[str, str] = {}
        for key, path in listed.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required sensitivity input is missing: {key}={path}"
                )
            event_payload[key] = _sha256(path)
        payload[event_type] = event_payload
    return payload


def _spec_row(spec: RunSpec, config: dict[str, Any]) -> dict[str, Any]:
    defaults = defaults_from_config(config)
    return {
        "event_type": spec.event_type,
        "setting_id": spec.setting_id,
        "varied_parameter": spec.varied_parameter,
        "parameter_value": "" if spec.parameter_value is None else spec.parameter_value,
        "is_default_setting": str(spec.is_default_setting).lower(),
        "seed": spec.seed,
        "num_samples": (
            defaults["num_samples"]
            if spec.parameter_value is None or spec.varied_parameter != "num_samples"
            else spec.parameter_value
        ),
        "p0": (
            defaults["p0"]
            if spec.parameter_value is None or spec.varied_parameter != "p0"
            else spec.parameter_value
        ),
        "proposal_std": (
            defaults["proposal_std"]
            if spec.parameter_value is None or spec.varied_parameter != "proposal_std"
            else spec.parameter_value
        ),
        "context_refresh_prob": (
            defaults["context_refresh_prob"]
            if spec.parameter_value is None
            or spec.varied_parameter != "context_refresh_prob"
            else spec.parameter_value
        ),
        "mh_retries_per_sample": defaults["mh_retries_per_sample"],
        "max_levels": defaults["max_levels"],
        "run_dir": str(spec.run_dir.relative_to(RESULTS_ROOT)),
        "execution_status": "pending",
        "quality_status": "pending",
        "failure_reason": "",
        "runtime_seconds": "",
        "summary_path": "",
        "effective_config_sha256": "",
        "updated_utc": _utc_now(),
    }


def _load_plan(
    specs: list[RunSpec], configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    expected = {(spec.event_type, spec.setting_id, str(spec.seed)) for spec in specs}
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if RUN_PLAN_PATH.exists():
        with RUN_PLAN_PATH.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["event_type"], row["setting_id"], row["seed"])
                if key in expected:
                    existing[key] = dict(row)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        key = (spec.event_type, spec.setting_id, str(spec.seed))
        rows.append(existing.get(key, _spec_row(spec, configs[spec.event_type])))
    return rows


def _write_plan(rows: list[dict[str, Any]]) -> None:
    _csv_write(RUN_PLAN_PATH, rows)


def _frozen_design(
    configs: dict[str, dict[str, Any]],
    specs: list[RunSpec],
) -> dict[str, Any]:
    """Return the immutable inputs and OAT design that define this experiment."""
    return {
        "base_configs": {
            event: str(path.relative_to(ROOT)) for event, path in BASE_CONFIGS.items()
        },
        "base_config_sha256": {
            event: _sha256(path) for event, path in BASE_CONFIGS.items()
        },
        "input_sha256": _input_fingerprints(configs),
        "mc_reference_num_samples": MC_REFERENCE_SIZES,
        "oat_grid": {
            event: {
                "defaults": defaults_from_config(configs[event]),
                "parameter_values": {
                    parameter: list(values) for parameter, values in GRID[event].items()
                },
                "planned_ss_runs": sum(1 for spec in specs if spec.event_type == event),
            }
            for event in EVENTS
        },
    }


def _manifest(
    configs: dict[str, dict[str, Any]], specs: list[RunSpec]
) -> dict[str, Any]:
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        cuda_name = str(torch.cuda.get_device_name(0)) if cuda_available else None
    except Exception:  # pragma: no cover - only for diagnostics
        torch_version, cuda_available, cuda_name = None, False, None
    return {
        "experiment": "IDM_highD_subset_simulation_OAT_sensitivity",
        "scope": {
            "included": ["IDM", "highD", "following", "cutin", "subset_simulation_OAT"],
            "excluded": [
                "multi_ADS",
                "module_ablation",
                "importance_sampling",
                "urban_data",
                "risk_metric_replacement",
            ],
        },
        "created_utc": _utc_now(),
        "repository_root": str(ROOT),
        "runner": "IDM_subset.src.latent_subset_runner.run_subset_from_config",
        "seed_policy": {
            "default": [101, 202, 303, 404, 505],
            "non_default": [101, 202, 303],
            "early_stop": "Only non-default settings; after two consecutive execution or reliability failures, mark remaining planned seeds skipped.",
        },
        **_frozen_design(configs, specs),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda_available": cuda_available,
            "cuda_device": cuda_name,
        },
    }


def _verify_manifest(configs: dict[str, dict[str, Any]], specs: list[RunSpec]) -> None:
    """Reject a resume if the frozen inputs or OAT design have changed."""
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    current = _frozen_design(configs, specs)
    mismatches = [key for key, value in current.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "Frozen sensitivity experiment does not match the existing manifest "
            f"for keys: {', '.join(mismatches)}. Start a new results directory "
            "instead of mixing incompatible inputs or OAT settings."
        )


def _effective_config(base_config: dict[str, Any], spec: RunSpec) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    subset = config.setdefault("subset_simulation", {})
    if spec.parameter_value is not None:
        subset[spec.varied_parameter] = spec.parameter_value
    subset["output_dir"] = str(spec.run_dir)
    config.setdefault("training", {})["seed"] = int(spec.seed)
    config.setdefault("context_sampling", {})["seed"] = int(spec.seed)
    config["sensitivity_experiment"] = {
        "experiment": "IDM_highD_subset_simulation_OAT_sensitivity",
        "event_type": spec.event_type,
        "setting_id": spec.setting_id,
        "varied_parameter": spec.varied_parameter,
        "parameter_value": spec.parameter_value,
        "is_default_setting": spec.is_default_setting,
        "seed": int(spec.seed),
    }
    return config


def _summary_quality(summary: dict[str, Any]) -> tuple[str, str]:
    reliability = dict(summary.get("reliability", {}) or {})
    if reliability.get("status") == "pass":
        return "pass", ""
    reasons = list(reliability.get("failures", []) or []) + list(
        reliability.get("warnings", []) or []
    )
    if not reasons:
        reasons = [f"reliability_status={reliability.get('status', 'missing')}"]
    return "fail", "; ".join(str(item) for item in reasons)


def _quality_failure_from_row(row: dict[str, Any]) -> bool | None:
    """Classify an already-planned run, or return None when it has not run."""
    execution_status = row.get("execution_status")
    if execution_status == "completed":
        return row.get("quality_status") != "pass"
    if execution_status in {"failed", "skipped_after_two_quality_failures"}:
        return True
    return None


def _consecutive_prior_quality_failures(
    spec: RunSpec,
    all_setting_specs: list[RunSpec],
    index: dict[tuple[str, str, str], dict[str, Any]],
) -> int:
    """Count directly preceding failures across interrupted invocations."""
    failures = 0
    for candidate in reversed(all_setting_specs):
        if candidate.seed >= spec.seed:
            continue
        row = index[(candidate.event_type, candidate.setting_id, str(candidate.seed))]
        outcome = _quality_failure_from_row(row)
        if outcome is True:
            failures += 1
            continue
        break
    return failures


def _run_one(
    spec: RunSpec,
    base_config: dict[str, Any],
    row: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], bool]:
    run_dir = spec.run_dir
    summary_path = run_dir / "latent_subset_summary.json"
    status_path = run_dir / "run_status.json"
    if (
        not overwrite
        and summary_path.is_file()
        and row.get("execution_status") == "completed"
    ):
        LOGGER.info(
            "Skip completed run %s/%s seed=%s",
            spec.event_type,
            spec.setting_id,
            spec.seed,
        )
        return row, row.get("quality_status") != "pass"

    config = _effective_config(base_config, spec)
    config_hash = _canonical_hash(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _json_write(run_dir / "effective_config.json", config)
    row.update(
        {
            "execution_status": "running",
            "quality_status": "pending",
            "failure_reason": "",
            "runtime_seconds": "",
            "summary_path": str(summary_path.relative_to(RESULTS_ROOT)),
            "effective_config_sha256": config_hash,
            "updated_utc": _utc_now(),
        }
    )
    start = time.monotonic()
    try:
        produced = run_subset_from_config(
            config,
            BASE_CONFIGS[spec.event_type].parent,
            expected_event_type="cut_in" if spec.event_type == "cutin" else "following",
        )
        with Path(produced).open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        quality_status, failure_reason = _summary_quality(summary)
        elapsed = time.monotonic() - start
        row.update(
            {
                "execution_status": "completed",
                "quality_status": quality_status,
                "failure_reason": failure_reason,
                "runtime_seconds": f"{elapsed:.6f}",
                "summary_path": str(Path(produced).relative_to(RESULTS_ROOT)),
                "effective_config_sha256": config_hash,
                "updated_utc": _utc_now(),
            }
        )
        _json_write(
            status_path,
            {
                "execution_status": "completed",
                "quality_status": quality_status,
                "failure_reason": failure_reason,
                "runtime_seconds": elapsed,
                "summary_path": str(Path(produced)),
                "updated_utc": row["updated_utc"],
            },
        )
        return row, quality_status != "pass"
    except Exception as exc:  # required: retain failed settings and traceback
        elapsed = time.monotonic() - start
        reason = f"{type(exc).__name__}: {exc}"
        row.update(
            {
                "execution_status": "failed",
                "quality_status": "fail",
                "failure_reason": reason,
                "runtime_seconds": f"{elapsed:.6f}",
                "effective_config_sha256": config_hash,
                "updated_utc": _utc_now(),
            }
        )
        _json_write(
            status_path,
            {
                "execution_status": "failed",
                "quality_status": "fail",
                "failure_reason": reason,
                "runtime_seconds": elapsed,
                "traceback": traceback.format_exc(),
                "updated_utc": row["updated_utc"],
            },
        )
        LOGGER.exception(
            "Sensitivity run failed: %s/%s seed=%s",
            spec.event_type,
            spec.setting_id,
            spec.seed,
        )
        return row, True


def _run_reference_mc(
    event_type: str, base_config: dict[str, Any], *, overwrite: bool
) -> None:
    output_dir = RESULTS_ROOT / "references" / event_type
    summary_path = output_dir / "latent_monte_carlo_summary.json"
    if summary_path.exists() and not overwrite:
        LOGGER.info("MC reference already exists for %s: %s", event_type, summary_path)
        return
    config = copy.deepcopy(base_config)
    config.setdefault("monte_carlo", {})["num_samples"] = MC_REFERENCE_SIZES[event_type]
    config.setdefault("monte_carlo", {})["seed"] = 42
    config["monte_carlo"]["output_dir"] = str(output_dir)
    config["sensitivity_experiment"] = {
        "experiment": "IDM_highD_subset_simulation_OAT_sensitivity_MC_reference",
        "event_type": event_type,
        "num_samples": MC_REFERENCE_SIZES[event_type],
        "seed": 42,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(output_dir / "effective_config.json", config)
    LOGGER.info(
        "Run MC reference %s with N=%d", event_type, MC_REFERENCE_SIZES[event_type]
    )
    try:
        produced = run_monte_carlo_from_config(
            config,
            BASE_CONFIGS[event_type].parent,
            expected_event_type="cut_in" if event_type == "cutin" else "following",
        )
        _json_write(
            output_dir / "run_status.json",
            {
                "execution_status": "completed",
                "summary_path": str(produced),
                "updated_utc": _utc_now(),
            },
        )
    except Exception as exc:
        _json_write(
            output_dir / "run_status.json",
            {
                "execution_status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated_utc": _utc_now(),
            },
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=(*EVENTS, "all"), default="all")
    parser.add_argument(
        "--setting", help="Run only one setting id, e.g. p0_0p1 or default."
    )
    parser.add_argument(
        "--seed", type=int, help="Run only one seed from the frozen design."
    )
    parser.add_argument(
        "--run-reference-mc",
        action="store_true",
        help="Run the plan-required MC reference before SS runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create manifest and run plan without simulation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run selected cells in this experiment-specific result directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    event_types = EVENTS if args.event == "all" else (args.event,)
    configs = {event: load_yaml(BASE_CONFIGS[event]) for event in EVENTS}
    all_specs = [
        spec for event in EVENTS for spec in build_run_specs(event, configs[event])
    ]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        _json_write(MANIFEST_PATH, _manifest(configs, all_specs))
    else:
        _verify_manifest(configs, all_specs)
    rows = _load_plan(all_specs, configs)
    _write_plan(rows)
    selected = [spec for spec in all_specs if spec.event_type in event_types]
    if args.setting:
        selected = [spec for spec in selected if spec.setting_id == args.setting]
    if args.seed is not None:
        selected = [spec for spec in selected if spec.seed == args.seed]
    if not selected:
        raise SystemExit("No frozen OAT run matches the requested event/setting/seed.")
    LOGGER.info(
        "Prepared %d selected SS run(s); total frozen design=%d",
        len(selected),
        len(all_specs),
    )
    if args.dry_run:
        return
    if args.run_reference_mc:
        for event_type in event_types:
            _run_reference_mc(event_type, configs[event_type], overwrite=args.overwrite)

    index = {
        (row["event_type"], row["setting_id"], str(row["seed"])): row for row in rows
    }
    all_grouped: dict[tuple[str, str], list[RunSpec]] = {}
    for spec in all_specs:
        all_grouped.setdefault((spec.event_type, spec.setting_id), []).append(spec)
    for specs in all_grouped.values():
        specs.sort(key=lambda item: item.seed)
    grouped: dict[tuple[str, str], list[RunSpec]] = {}
    for spec in selected:
        grouped.setdefault((spec.event_type, spec.setting_id), []).append(spec)
    for (event_type, setting_id), setting_specs in grouped.items():
        setting_specs.sort(key=lambda item: item.seed)
        all_setting_specs = all_grouped[(event_type, setting_id)]
        for spec in setting_specs:
            row = index[(spec.event_type, spec.setting_id, str(spec.seed))]
            prior_failures = _consecutive_prior_quality_failures(
                spec,
                all_setting_specs,
                index,
            )
            if (
                not spec.is_default_setting
                and prior_failures >= 2
                and row.get("execution_status") != "completed"
            ):
                row.update(
                    {
                        "execution_status": "skipped_after_two_quality_failures",
                        "quality_status": "fail",
                        "failure_reason": "Two consecutive earlier seeds had execution or reliability failures.",
                        "updated_utc": _utc_now(),
                    }
                )
                _json_write(
                    spec.run_dir / "run_status.json",
                    {
                        "execution_status": row["execution_status"],
                        "failure_reason": row["failure_reason"],
                        "updated_utc": row["updated_utc"],
                    },
                )
                _write_plan(rows)
                continue
            with _run_file_log(spec.run_dir / "run.log"):
                row, quality_failed = _run_one(
                    spec,
                    configs[event_type],
                    row,
                    overwrite=args.overwrite,
                )
            index[(spec.event_type, spec.setting_id, str(spec.seed))] = row
            _write_plan(rows)

    from IDM_subset.experiments.highd_ss_sensitivity.summarize_results import summarize

    summarize(results_root=RESULTS_ROOT)


if __name__ == "__main__":
    main()
