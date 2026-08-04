#!/usr/bin/env python3
"""Run highD IDM SS OAT sensitivity or current-default repeat experiments.

Both workflows call the shared runner API instead of modifying default YAMLs or
using event-specific duplicate runners.  The frozen OAT workflow preserves its
immutable plan, whereas the default-repeat workflow isolates the five current
configuration seeds in an event-specific result directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import load_yaml, setup_logging
from IDM_subset.experiments.highd_ss_sensitivity.sensitivity_spec import (
    BASE_CONFIGS,
    CURRENT_MC_REFERENCE_SUMMARIES,
    DEFAULT_REPEAT_ROOTS,
    EVENTS,
    FORMAL_EXECUTION_DEFAULTS,
    GRID,
    MC_REFERENCE_SIZES,
    PARALLEL_EQUIVALENCE_TOLERANCES,
    RESULTS_ROOT,
    RunSpec,
    build_default_repeat_specs,
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


def _relative_to_config(path: Path, config_dir: Path) -> str:
    """Return an executable POSIX path relative to a base YAML directory."""
    return Path(
        os.path.relpath(path.resolve(), start=config_dir.resolve())
    ).as_posix()


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
                key = (
                    row.get("event_type", ""),
                    row.get("setting_id", ""),
                    row.get("seed", ""),
                )
                if key in expected:
                    existing[key] = dict(row)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        key = (spec.event_type, spec.setting_id, str(spec.seed))
        row = existing.get(key, _spec_row(spec, configs[spec.event_type]))
        summary_path = spec.run_dir / "latent_subset_summary.json"
        status_path = spec.run_dir / "run_status.json"
        if summary_path.is_file() and status_path.is_file() and row.get(
            "execution_status"
        ) != "completed":
            # Recover a completed isolated artifact if a previous scheduler
            # interruption occurred between producing the run and updating the
            # shared plan.  This is also safe for manual recovery.
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            with status_path.open("r", encoding="utf-8") as handle:
                status = json.load(handle)
            quality_status, failure_reason = _summary_quality(summary)
            config_path = spec.run_dir / "effective_config.json"
            config_hash = ""
            if config_path.is_file():
                with config_path.open("r", encoding="utf-8") as handle:
                    config_hash = _canonical_hash(json.load(handle))
            row.update(
                {
                    "execution_status": "completed",
                    "quality_status": quality_status,
                    "failure_reason": failure_reason,
                    "runtime_seconds": status.get("runtime_seconds", ""),
                    "summary_path": str(summary_path.relative_to(RESULTS_ROOT)),
                    "effective_config_sha256": config_hash,
                    "updated_utc": status.get("updated_utc", _utc_now()),
                }
            )
        rows.append(row)
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
            # Keep the frozen design portable across Linux/Windows workers.  The
            # asset bytes are audited separately by SHA-256; this field is only
            # a repository-relative identifier and must not vary by path
            # separator when a run is resumed on another platform.
            event: path.relative_to(ROOT).as_posix()
            for event, path in BASE_CONFIGS.items()
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
        # Manifest paths are deliberately repository-relative so its contents
        # remain portable across Windows and Linux checkouts.
        "repository_root": ".",
        "runner": "IDM_subset.src.latent_subset_runner.run_subset_from_config",
        "seed_policy": {
            "default": [101, 202, 303, 404, 505],
            "non_default": [101, 202, 303],
            "early_stop": "Only non-default settings; after two consecutive execution or reliability failures, mark remaining planned seeds skipped.",
        },
        "execution_implementation": {
            "formal_execution_defaults": dict(FORMAL_EXECUTION_DEFAULTS),
            "parallel_equivalence_tolerances": dict(PARALLEL_EQUIVALENCE_TOLERANCES),
            "mcmc": "Formal runs use validated batch-invariant independent-chain MCMC (mcmc_batch_size=64).",
            "population": "GPU diffusion decoding with optional spawn-safe CPU closed-loop rollout workers.",
            "scheduler": "1/2/4 independent OAT-setting workers; one active seed per setting preserves the early-stop rule.",
        },
        **_frozen_design(configs, specs),
        "environment": {
            "python": sys.version,
            "executable": Path(sys.executable).name,
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


def _effective_config(
    base_config: dict[str, Any],
    spec: RunSpec,
    *,
    execution_profile: dict[str, int],
    default_repeat: bool = False,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    subset = config.setdefault("subset_simulation", {})
    if spec.parameter_value is not None:
        subset[spec.varied_parameter] = spec.parameter_value
    subset["output_dir"] = _relative_to_config(
        spec.run_dir, BASE_CONFIGS[spec.event_type].parent
    )
    parallel_cfg = config.setdefault("parallel", {})
    parallel_cfg.update(execution_profile)
    config.setdefault("training", {})["seed"] = int(spec.seed)
    config.setdefault("context_sampling", {})["seed"] = int(spec.seed)
    if default_repeat:
        config["default_repeat_experiment"] = {
            "experiment": "IDM_current_default_configuration_repeat",
            "event_type": spec.event_type,
            "seed": int(spec.seed),
            "execution_profile": dict(execution_profile),
        }
    else:
        config["sensitivity_experiment"] = {
            "experiment": "IDM_highD_subset_simulation_OAT_sensitivity",
            "event_type": spec.event_type,
            "setting_id": spec.setting_id,
            "varied_parameter": spec.varied_parameter,
            "parameter_value": spec.parameter_value,
            "is_default_setting": spec.is_default_setting,
            "seed": int(spec.seed),
            "execution_profile": dict(execution_profile),
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
    execution_profile: dict[str, int],
    result_root: Path = RESULTS_ROOT,
    default_repeat: bool = False,
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

    config = _effective_config(
        base_config,
        spec,
        execution_profile=execution_profile,
        default_repeat=default_repeat,
    )
    config_hash = _canonical_hash(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _json_write(run_dir / "effective_config.json", config)
    row.update(
        {
            "execution_status": "running",
            "quality_status": "pending",
            "failure_reason": "",
            "runtime_seconds": "",
            "summary_path": str(summary_path.relative_to(result_root)),
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
                "summary_path": str(Path(produced).relative_to(result_root)),
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
                "summary_path": Path(produced).name,
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


def _run_one_worker(
    spec: RunSpec,
    base_config: dict[str, Any],
    row: dict[str, Any],
    overwrite: bool,
    execution_profile: dict[str, int],
) -> tuple[dict[str, Any], bool]:
    """Spawn-safe independent SS work unit used by the 2/4-task scheduler."""
    setup_logging("INFO")
    with _run_file_log(spec.run_dir / "run.log"):
        return _run_one(
            spec,
            base_config,
            row,
            overwrite=overwrite,
            execution_profile=execution_profile,
        )


def _run_reference_mc(
    event_type: str,
    base_config: dict[str, Any],
    *,
    overwrite: bool,
    execution_profile: dict[str, int],
) -> None:
    output_dir = RESULTS_ROOT / "references" / event_type
    summary_path = output_dir / "latent_monte_carlo_summary.json"
    if summary_path.exists() and not overwrite:
        LOGGER.info("MC reference already exists for %s: %s", event_type, summary_path)
        return
    config = copy.deepcopy(base_config)
    config.setdefault("monte_carlo", {})["num_samples"] = MC_REFERENCE_SIZES[event_type]
    config.setdefault("monte_carlo", {})["seed"] = 42
    config["monte_carlo"]["output_dir"] = _relative_to_config(
        output_dir, BASE_CONFIGS[event_type].parent
    )
    config.setdefault("parallel", {}).update(execution_profile)
    config["sensitivity_experiment"] = {
        "experiment": "IDM_highD_subset_simulation_OAT_sensitivity_MC_reference",
        "event_type": event_type,
        "num_samples": MC_REFERENCE_SIZES[event_type],
        "seed": 42,
        "execution_profile": dict(execution_profile),
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
                "summary_path": Path(produced).name,
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


def _mark_skipped_after_quality_failures(
    spec: RunSpec,
    row: dict[str, Any],
) -> None:
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


def _run_selected_specs(
    *,
    selected: list[RunSpec],
    all_specs: list[RunSpec],
    rows: list[dict[str, Any]],
    configs: dict[str, dict[str, Any]],
    overwrite: bool,
    scheduler_workers: int,
    execution_profile: dict[str, int],
) -> None:
    """Run independent OAT settings concurrently without breaking seed policy.

    A setting has at most one active seed.  Therefore non-default settings can
    still stop after two consecutive execution/reliability failures, while
    unrelated settings occupy the remaining 2/4 scheduler slots.
    """
    index = {
        (row["event_type"], row["setting_id"], str(row["seed"])): row
        for row in rows
    }
    all_grouped: dict[tuple[str, str], list[RunSpec]] = {}
    for spec in all_specs:
        all_grouped.setdefault((spec.event_type, spec.setting_id), []).append(spec)
    for specs in all_grouped.values():
        specs.sort(key=lambda item: item.seed)
    grouped: dict[tuple[str, str], list[RunSpec]] = {}
    for spec in selected:
        grouped.setdefault((spec.event_type, spec.setting_id), []).append(spec)
    for specs in grouped.values():
        specs.sort(key=lambda item: item.seed)

    positions = {group: 0 for group in grouped}
    active_groups: set[tuple[str, str]] = set()
    futures: dict[Future[tuple[dict[str, Any], bool]], tuple[tuple[str, str], RunSpec]] = {}

    def prepare_next(group: tuple[str, str]) -> RunSpec | None:
        setting_specs = grouped[group]
        all_setting_specs = all_grouped[group]
        while positions[group] < len(setting_specs):
            spec = setting_specs[positions[group]]
            positions[group] += 1
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
                _mark_skipped_after_quality_failures(spec, row)
                continue
            if (
                not overwrite
                and row.get("execution_status") == "completed"
                and (spec.run_dir / "latent_subset_summary.json").is_file()
            ):
                continue
            row.update(
                {
                    "execution_status": "scheduled",
                    "quality_status": "pending",
                    "failure_reason": "",
                    "updated_utc": _utc_now(),
                }
            )
            return spec
        return None

    if scheduler_workers == 1:
        for group in grouped:
            while (spec := prepare_next(group)) is not None:
                row = index[(spec.event_type, spec.setting_id, str(spec.seed))]
                with _run_file_log(spec.run_dir / "run.log"):
                    updated, _quality_failed = _run_one(
                        spec,
                        configs[spec.event_type],
                        row,
                        overwrite=overwrite,
                        execution_profile=execution_profile,
                    )
                if updated is not row:
                    row.clear()
                    row.update(updated)
                _write_plan(rows)
        _write_plan(rows)
        return

    # Spawn avoids inheriting a CUDA context and works on both Windows and
    # Linux.  The parent is the sole writer of run_plan.csv.
    with ProcessPoolExecutor(
        max_workers=scheduler_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        while True:
            while len(futures) < scheduler_workers:
                next_group = next(
                    (group for group in grouped if group not in active_groups),
                    None,
                )
                if next_group is None:
                    break
                spec = prepare_next(next_group)
                if spec is None:
                    # This group has no runnable seed left; do not select it
                    # again in this scheduler pass.
                    active_groups.add(next_group)
                    continue
                row = index[(spec.event_type, spec.setting_id, str(spec.seed))]
                future = executor.submit(
                    _run_one_worker,
                    spec,
                    configs[spec.event_type],
                    dict(row),
                    overwrite,
                    execution_profile,
                )
                futures[future] = (next_group, spec)
                active_groups.add(next_group)
            _write_plan(rows)
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                group, spec = futures.pop(future)
                active_groups.discard(group)
                row = index[(spec.event_type, spec.setting_id, str(spec.seed))]
                try:
                    updated, _quality_failed = future.result()
                except Exception as exc:  # defensive: worker setup can fail first
                    reason = f"{type(exc).__name__}: {exc}"
                    row.update(
                        {
                            "execution_status": "failed",
                            "quality_status": "fail",
                            "failure_reason": reason,
                            "updated_utc": _utc_now(),
                        }
                    )
                    _json_write(
                        spec.run_dir / "run_status.json",
                        {
                            "execution_status": "failed",
                            "failure_reason": reason,
                            "traceback": traceback.format_exc(),
                            "updated_utc": row["updated_utc"],
                        },
                    )
                    LOGGER.exception(
                        "Scheduled sensitivity run failed before completion: %s/%s seed=%s",
                        spec.event_type,
                        spec.setting_id,
                        spec.seed,
                    )
                else:
                    row.clear()
                    row.update(updated)
            _write_plan(rows)


def _operational_repeat_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove run-location and provenance fields before repeat reuse checks."""
    normalized = copy.deepcopy(config)
    if isinstance(normalized.get("subset_simulation"), dict):
        normalized["subset_simulation"].pop("output_dir", None)
    # The repeat workflow only executes SS.  MC is paired separately through
    # the event-specific MC entrypoint, so its sample size/output settings do
    # not determine whether an SS seed result is reusable.
    normalized.pop("monte_carlo", None)
    for section in ("training", "context_sampling"):
        if isinstance(normalized.get(section), dict):
            normalized[section].pop("seed", None)
    for key in (
        "sensitivity_experiment",
        "cutin_default_calibration",
        "default_repeat_experiment",
    ):
        normalized.pop(key, None)
    return normalized


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _default_repeat_manifest(
    event_type: str,
    base_config: dict[str, Any],
    specs: list[RunSpec],
    execution_profile: dict[str, int],
) -> dict[str, Any]:
    repeat_root = DEFAULT_REPEAT_ROOTS[event_type]
    return {
        "workflow": "current_default_configuration_repeats",
        "event_type": event_type,
        "base_config": _repo_relative(BASE_CONFIGS[event_type]),
        "base_config_sha256": _sha256(BASE_CONFIGS[event_type]),
        "operational_config_sha256": _canonical_hash(
            _operational_repeat_config(base_config)
        ),
        "seeds": [int(spec.seed) for spec in specs],
        "execution_profile": dict(execution_profile),
        "output_root": _repo_relative(repeat_root),
        "paired_mc_reference_summary": _repo_relative(
            CURRENT_MC_REFERENCE_SUMMARIES[event_type]
        ),
    }


def _write_or_verify_default_repeat_manifest(
    event_type: str,
    base_config: dict[str, Any],
    specs: list[RunSpec],
    execution_profile: dict[str, int],
) -> None:
    repeat_root = DEFAULT_REPEAT_ROOTS[event_type]
    manifest_path = repeat_root / "default_repeat_manifest.json"
    current = _default_repeat_manifest(
        event_type, base_config, specs, execution_profile
    )
    if not manifest_path.exists():
        _json_write(manifest_path, current)
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    immutable_keys = (
        "workflow",
        "event_type",
        "operational_config_sha256",
        "seeds",
        "execution_profile",
        "output_root",
    )
    mismatches = [
        key for key in immutable_keys if existing.get(key) != current.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "Current default-repeat workflow does not match its existing manifest "
            f"for keys: {', '.join(mismatches)}. Do not mix incompatible "
            "configurations in the same repeat directory."
        )


def _validate_reusable_default_repeat(
    spec: RunSpec,
    base_config: dict[str, Any],
    execution_profile: dict[str, int],
) -> None:
    """Reject a stale seed result if its operational configuration differs."""
    config_path = spec.run_dir / "effective_config.json"
    if not config_path.is_file():
        return
    with config_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    expected = _effective_config(
        base_config,
        spec,
        execution_profile=execution_profile,
        default_repeat=True,
    )
    if _canonical_hash(_operational_repeat_config(existing)) != _canonical_hash(
        _operational_repeat_config(expected)
    ):
        raise RuntimeError(
            "Refusing to reuse an incompatible default-repeat result: "
            f"event={spec.event_type}, seed={spec.seed}. Use a new output "
            "directory or rerun the complete repeat set intentionally."
        )


def _repeat_record_from_existing(spec: RunSpec) -> dict[str, Any] | None:
    summary_path = spec.run_dir / "latent_subset_summary.json"
    if not summary_path.is_file():
        return None
    status_path = spec.run_dir / "run_status.json"
    status: dict[str, Any] = {}
    if status_path.is_file():
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    quality_status, failure_reason = _summary_quality(summary)
    return {
        "seed": int(spec.seed),
        "execution_status": "reused",
        "quality_status": quality_status,
        "failure_reason": failure_reason,
        "runtime_seconds": status.get("runtime_seconds", ""),
        "summary_path": summary_path.relative_to(spec.output_root).as_posix(),
        "summary": summary,
    }


def _default_repeat_pending_record(spec: RunSpec, status: str = "pending") -> dict[str, Any]:
    return {
        "seed": int(spec.seed),
        "execution_status": status,
        "quality_status": "pending" if status == "pending" else "fail",
        "failure_reason": "",
        "runtime_seconds": "",
        "summary_path": "",
        "summary": None,
    }


def _run_default_repeat_one(
    spec: RunSpec,
    base_config: dict[str, Any],
    *,
    execution_profile: dict[str, int],
    overwrite: bool,
) -> dict[str, Any]:
    row = {
        "event_type": spec.event_type,
        "setting_id": "default",
        "seed": int(spec.seed),
        "execution_status": "pending",
        "quality_status": "pending",
        "failure_reason": "",
        "runtime_seconds": "",
        "summary_path": "",
        "effective_config_sha256": "",
        "updated_utc": _utc_now(),
    }
    with _run_file_log(spec.run_dir / "run.log"):
        row, _ = _run_one(
            spec,
            base_config,
            row,
            overwrite=overwrite,
            execution_profile=execution_profile,
            result_root=spec.output_root,
            default_repeat=True,
        )
    summary_path = spec.run_dir / "latent_subset_summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    return {
        "seed": int(spec.seed),
        "execution_status": row["execution_status"],
        "quality_status": row["quality_status"],
        "failure_reason": row["failure_reason"],
        "runtime_seconds": row["runtime_seconds"],
        "summary_path": row["summary_path"],
        "summary": summary,
    }


def _write_default_repeat_summary(
    event_type: str,
    records: list[dict[str, Any]],
) -> None:
    """Persist a common five-seed audit table and uncertainty summary."""
    repeat_root = DEFAULT_REPEAT_ROOTS[event_type]
    rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: int(item["seed"])):
        summary = record.get("summary")
        counts = dict(summary.get("simulation_counts", {}) or {}) if summary else {}
        levels = list(summary.get("level_stats", []) or []) if summary else []
        rows.append(
            {
                "seed": record["seed"],
                "execution_status": record["execution_status"],
                "quality_status": record["quality_status"],
                "failure_reason": record["failure_reason"],
                "runtime_seconds": record["runtime_seconds"],
                "probability": "" if summary is None else summary.get("probability", ""),
                "num_levels": "" if summary is None else summary.get("num_levels", ""),
                "stop_reason": "" if summary is None else summary.get("stop_reason", ""),
                "closed_loop_evaluations": counts.get("closed_loop_evaluations", ""),
                "proposal_evaluations": counts.get("proposal_evaluations", ""),
                "mean_transition_acceptance_rate": (
                    levels[0].get("acceptance_rate", "") if levels else ""
                ),
            }
        )
        if summary is not None:
            completed.append(record)
    _csv_write(repeat_root / "seed_results.csv", rows)

    payload: dict[str, Any] = {
        "workflow": "current_default_configuration_repeats",
        "event_type": event_type,
        "num_requested_seed_runs": len(records),
        "num_completed_seed_runs": len(completed),
        "all_seed_runs_reliability_pass": len(completed) == len(records)
        and all(record["quality_status"] == "pass" for record in completed),
        "paired_mc_reference_summary": _repo_relative(
            CURRENT_MC_REFERENCE_SUMMARIES[event_type]
        ),
    }
    if completed:
        probabilities = [float(record["summary"]["probability"]) for record in completed]
        runtimes: list[float] = []
        for record in completed:
            try:
                runtime = float(record.get("runtime_seconds", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(runtime) and runtime >= 0.0:
                runtimes.append(runtime)
        payload["probability_mean"] = statistics.mean(probabilities)
        payload["num_seed_runs"] = len(completed)
        if len(probabilities) > 1:
            probability_std = statistics.stdev(probabilities)
            t_half_width = float(student_t.ppf(0.975, df=len(probabilities) - 1)) * (
                probability_std / math.sqrt(len(probabilities))
            )
            payload.update(
                {
                    "probability_std_across_seed": probability_std,
                    "probability_cv_across_seed": probability_std
                    / payload["probability_mean"],
                    "probability_ci95_across_seed": [
                        payload["probability_mean"] - t_half_width,
                        payload["probability_mean"] + t_half_width,
                    ],
                }
            )
        if runtimes:
            payload["runtime_seconds_mean"] = statistics.mean(runtimes)
            payload["runtime_seconds_max"] = max(runtimes)
        payload["mean_closed_loop_evaluations"] = statistics.mean(
            float(
                dict(record["summary"].get("simulation_counts", {}) or {}).get(
                    "closed_loop_evaluations", 0
                )
            )
            for record in completed
        )

    mc_path = CURRENT_MC_REFERENCE_SUMMARIES[event_type]
    if mc_path.is_file():
        with mc_path.open("r", encoding="utf-8") as handle:
            mc_summary = json.load(handle)
        mc_probability = float(mc_summary["probability"])
        payload["mc_reference"] = {
            "summary_path": _repo_relative(mc_path),
            "probability": mc_probability,
            "probability_ci95": [
                float(mc_summary["probability_ci95_lower"]),
                float(mc_summary["probability_ci95_upper"]),
            ],
            "num_samples": int(mc_summary["num_samples"]),
        }
        if "probability_mean" in payload:
            payload["relative_difference_vs_mc"] = (
                payload["probability_mean"] - mc_probability
            ) / mc_probability
        if "probability_ci95_across_seed" in payload:
            ss_lower, ss_upper = payload["probability_ci95_across_seed"]
            mc_lower, mc_upper = payload["mc_reference"]["probability_ci95"]
            payload["ss_mc_ci95_overlap"] = max(ss_lower, mc_lower) <= min(
                ss_upper, mc_upper
            )
    _json_write(repeat_root / "summary.json", payload)


def _run_current_default_repeats(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    execution_profile: dict[str, int],
) -> None:
    """Run or resume the current-default five-seed validation for one event."""
    event_type = args.event
    specs = build_default_repeat_specs(event_type)
    _write_or_verify_default_repeat_manifest(
        event_type, base_config, specs, execution_profile
    )
    selected_seeds = {
        spec.seed for spec in specs if args.seed is None or spec.seed == args.seed
    }
    if not selected_seeds:
        raise SystemExit(
            f"Seed {args.seed} is not in the predeclared default repeat set."
        )
    LOGGER.info(
        "Prepared %d current-default %s repeat(s) in %s",
        len(selected_seeds),
        event_type,
        DEFAULT_REPEAT_ROOTS[event_type],
    )
    LOGGER.info(
        "Default-repeat seeds run sequentially; --workers=%d applies only to "
        "the frozen OAT scheduler.",
        args.workers,
    )
    if args.dry_run:
        return

    records: list[dict[str, Any]] = []
    for spec in specs:
        existing = _repeat_record_from_existing(spec)
        if existing is not None and not args.overwrite:
            _validate_reusable_default_repeat(spec, base_config, execution_profile)
            records.append(existing)
            continue
        if spec.seed not in selected_seeds:
            records.append(_default_repeat_pending_record(spec))
            continue
        records.append(
            _run_default_repeat_one(
                spec,
                base_config,
                execution_profile=execution_profile,
                overwrite=bool(args.overwrite),
            )
        )
    _write_default_repeat_summary(event_type, records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        choices=("frozen-oat", "default-repeats"),
        default="frozen-oat",
        help=(
            "frozen-oat preserves the immutable sensitivity design; "
            "default-repeats validates the current default configuration for "
            "one explicitly selected event."
        ),
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2, 4),
        default=FORMAL_EXECUTION_DEFAULTS["scheduler_workers"],
        help=(
            "Concurrent independent SS settings to schedule (1, 2, or 4). "
            "Default 4 is validated with two CPU rollout workers per task."
        ),
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        choices=(0, 1, 2, 4),
        default=FORMAL_EXECUTION_DEFAULTS["rollout_workers"],
        help="CPU closed-loop rollout workers inside each SS task (0 disables pipeline).",
    )
    parser.add_argument(
        "--mcmc-batch-size",
        type=int,
        choices=(1, 64),
        default=FORMAL_EXECUTION_DEFAULTS["mcmc_batch_size"],
        help="Independent MH proposals decoded per GPU batch (validated values: 1 or 64).",
    )
    parser.add_argument(
        "--population-batch-size",
        type=int,
        default=FORMAL_EXECUTION_DEFAULTS["population_batch_size"],
        help="GPU decode batch size for initial populations, MCMC, and MC.",
    )
    parser.add_argument(
        "--rollout-prefetch-batches",
        type=int,
        default=FORMAL_EXECUTION_DEFAULTS["rollout_prefetch_batches"],
        help="Decoded batches retained while CPU rollout workers run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mcmc_batch_size <= 0:
        raise SystemExit("--mcmc-batch-size must be positive")
    if args.rollout_prefetch_batches <= 0:
        raise SystemExit("--rollout-prefetch-batches must be positive")
    if args.population_batch_size <= 0:
        raise SystemExit("--population-batch-size must be positive")
    setup_logging("INFO")
    execution_profile = {
        "mcmc_batch_size": int(args.mcmc_batch_size),
        "rollout_workers": int(args.rollout_workers),
        "rollout_prefetch_batches": int(args.rollout_prefetch_batches),
        "population_batch_size": int(args.population_batch_size),
    }
    if args.workflow == "default-repeats":
        if args.event == "all":
            raise SystemExit(
                "--workflow default-repeats requires --event following or --event cutin."
            )
        if args.setting is not None:
            raise SystemExit(
                "--setting is only available for --workflow frozen-oat; "
                "default-repeats always uses the current default setting."
            )
        if args.run_reference_mc:
            raise SystemExit(
                "Run the canonical event-specific MC wrapper separately. "
                "default-repeats only reads its paired MC reference."
            )
        config = load_yaml(BASE_CONFIGS[args.event])
        _run_current_default_repeats(args, config, execution_profile)
        return

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
            _run_reference_mc(
                event_type,
                configs[event_type],
                overwrite=args.overwrite,
                execution_profile=execution_profile,
            )

    _run_selected_specs(
        selected=selected,
        all_specs=all_specs,
        rows=rows,
        configs=configs,
        overwrite=args.overwrite,
        scheduler_workers=int(args.workers),
        execution_profile=execution_profile,
    )

    from IDM_subset.experiments.highd_ss_sensitivity.summarize_results import summarize

    summarize(results_root=RESULTS_ROOT)


if __name__ == "__main__":
    main()
