#!/usr/bin/env python3
"""Reproducible highD multi-policy subset-simulation experiment driver.

The driver deliberately reuses the policy-specific subset runners.  It only
freezes a common protocol, writes immutable effective configurations, and
builds audit/summary artifacts.  Existing single-run baseline results are not
modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
# Scripts under ``tools/`` are executed as files, for which Python only puts
# the script directory (not the repository root) on ``sys.path``.  The
# policy-specific runners are repository packages, so make the import contract
# explicit for both direct ``python`` and ``conda run`` invocation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "results" / "multi_policy_validation"
MANIFEST_PATH = RESULTS_ROOT / "experiment_manifest.json"
FROZEN_INPUTS_PATH = RESULTS_ROOT / "frozen_inputs.json"
RUN_PLAN_PATH = RESULTS_ROOT / "run_plan.csv"
REFERENCE_PLAN_PATH = RESULTS_ROOT / "reference_plan.csv"
SUMMARY_STATUS_PATH = RESULTS_ROOT / "summary_status.json"
LOGGER = logging.getLogger("multi_policy_validation")

EVENTS = ("following", "cutin")
SEEDS = (101, 202, 303, 404, 505)
MC_SAMPLES = {"following": 200_000, "cutin": 20_000}
SS_SETTINGS = {
    "following": {
        "num_samples": 3000,
        "p0": 0.20,
        "proposal_std": 0.12,
        "context_refresh_prob": 0.70,
        "mh_retries_per_sample": 6,
        "max_levels": 8,
        "adaptive_stop_enabled": False,
    },
    "cutin": {
        "num_samples": 1000,
        "p0": 0.10,
        "proposal_std": 0.10,
        "context_refresh_prob": 0.50,
        "mh_retries_per_sample": 4,
        "max_levels": 8,
        "adaptive_stop_enabled": True,
    },
}
EXECUTION_PROFILE = {
    "mcmc_batch_size": 64,
    "population_batch_size": 64,
    "rollout_workers": 2,
    "rollout_prefetch_batches": 2,
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    package: str
    policy_section: str | None

    def config_path(self, event: str) -> Path:
        return ROOT / self.package / "scripts" / "configs" / f"latent_subset_{event}.yaml"

    @property
    def policy_source(self) -> Path:
        names = {
            "IDM": ROOT / "tools" / "idm_ego.yaml",
            "A2C": ROOT / "A2C_subset" / "policies" / "a2c_policy.py",
            "PPO": ROOT / "PPO_subset" / "policies" / "ppo_policy.py",
            "SAIRL": ROOT / "SAIRL_subset" / "policies" / "sairl_policy.py",
        }
        return names[self.name]


POLICIES = {
    "IDM": PolicySpec("IDM", "IDM_subset", None),
    "A2C": PolicySpec("A2C", "A2C_subset", "a2c_policy"),
    "PPO": PolicySpec("PPO", "PPO_subset", "ppo_policy"),
    "SAIRL": PolicySpec("SAIRL", "SAIRL_subset", "sairl_policy"),
}

PLAN_FIELDS = (
    "experiment_group",
    "policy",
    "event_type",
    "seed",
    "execution_status",
    "quality_status",
    "failure_reason",
    "runtime_seconds",
    "summary_path",
    "effective_config_path",
    "effective_config_sha256",
    "updated_utc",
)
REFERENCE_FIELDS = (
    "experiment_group",
    "policy",
    "event_type",
    "seed",
    "num_samples",
    "execution_status",
    "failure_reason",
    "runtime_seconds",
    "summary_path",
    "effective_config_path",
    "effective_config_sha256",
    "updated_utc",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _relative_to_config(path: Path, config_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=config_dir.resolve())).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Frozen input does not exist: {resolved}")
    return {
        "path": _repo_relative(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "sha256": _sha256_tree(resolved) if resolved.is_dir() else _sha256_file(resolved),
        "bytes": None if resolved.is_dir() else int(resolved.stat().st_size),
    }


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise TypeError(f"Expected mapping YAML config: {path}")
    return content


def _resolve_config_path(value: str | Path, config_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_dir / candidate).resolve()


def _policy_checkpoint(spec: PolicySpec, config: dict[str, Any]) -> Path | None:
    if spec.name == "IDM":
        return None
    if not spec.policy_section:
        return None
    value = dict(config.get(spec.policy_section, {}) or {}).get("checkpoint_path")
    if not value:
        raise KeyError(f"{spec.name} config has no {spec.policy_section}.checkpoint_path")
    checkpoint = Path(str(value))
    return checkpoint.resolve() if checkpoint.is_absolute() else (ROOT / checkpoint).resolve()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _base_configs(policies: Iterable[str], events: Iterable[str]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(policy, event): _load_yaml(POLICIES[policy].config_path(event)) for policy in policies for event in events}


def _frozen_inputs(
    policies: Iterable[str],
    events: Iterable[str],
    configs: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    assets: dict[str, dict[str, Any]] = {}
    source_files = {
        "tools/risk.py": ROOT / "tools" / "risk.py",
        "tools/highd_cutin.py": ROOT / "tools" / "highd_cutin.py",
        "tools/idm_ego.yaml": ROOT / "tools" / "idm_ego.yaml",
        "driver": Path(__file__).resolve(),
    }
    for name, path in source_files.items():
        assets[name] = _fingerprint(path)
    for policy in policies:
        spec = POLICIES[policy]
        assets[f"policy_adapter:{policy}"] = _fingerprint(spec.policy_source)
        for event in events:
            config_path = spec.config_path(event)
            assets[f"base_config:{policy}:{event}"] = _fingerprint(config_path)
            config = configs[(policy, event)]
            config_dir = config_path.parent
            for key in (
                "natural_dataset_dir",
                "diffusion_checkpoint",
                "tail_context_path",
                "condition_distribution_path",
                "evt_model_path",
                "exposure_summary_path",
            ):
                value = dict(config.get("paths", {}) or {}).get(key)
                if value:
                    assets[f"{key}:{policy}:{event}"] = _fingerprint(
                        _resolve_config_path(str(value), config_dir)
                    )
            checkpoint = _policy_checkpoint(spec, config)
            if checkpoint is not None:
                assets[f"policy_checkpoint:{policy}"] = _fingerprint(checkpoint)
    return {
        "revision_id": "multi_policy_validation",
        "created_utc": _utc_now(),
        "git_revision": _git_revision(),
        "assets": assets,
    }


def _manifest_design(
    policies: Iterable[str],
    events: Iterable[str],
    frozen_inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "revision_id": "multi_policy_validation",
        "policies": list(policies),
        "events": list(events),
        "ss_seeds": list(SEEDS),
        "ss_settings": SS_SETTINGS,
        "mc_samples": MC_SAMPLES,
        "execution_profile": EXECUTION_PROFILE,
        "frozen_input_sha256": _canonical_hash(frozen_inputs["assets"]),
    }


def _environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
    }
    try:
        import torch

        payload.update(
            {
                "torch": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - only diagnostics
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import stable_baselines3

        payload["stable_baselines3"] = str(stable_baselines3.__version__)
    except Exception as exc:  # pragma: no cover - IDM does not need SB3
        payload["stable_baselines3_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _run_dir(policy: str, event: str, seed: int) -> Path:
    return RESULTS_ROOT / "ads" / policy.lower() / event / f"seed_{seed}"


def _reference_dir(policy: str, event: str) -> Path:
    return RESULTS_ROOT / "references" / policy.lower() / event


def _effective_config(
    spec: PolicySpec,
    event: str,
    seed: int,
    base_config: dict[str, Any],
    output_dir: Path,
    *,
    is_monte_carlo: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config_dir = spec.config_path(event).parent
    subset = config.setdefault("subset_simulation", {})
    subset.update(SS_SETTINGS[event])
    subset["output_dir"] = _relative_to_config(output_dir, config_dir)
    mc = config.setdefault("monte_carlo", {})
    mc["num_samples"] = int(MC_SAMPLES[event])
    mc["seed"] = int(seed)
    mc["output_dir"] = _relative_to_config(output_dir, config_dir)
    config.setdefault("training", {})["seed"] = int(seed)
    # All formal runs use the same frozen CUDA diffusion sampler.  The policy
    # adapters inherit this device unless they provide a policy-specific value.
    config["training"]["device"] = "cuda"
    config.setdefault("context_sampling", {})["seed"] = int(seed)
    config.setdefault("parallel", {}).update(EXECUTION_PROFILE)
    if spec.policy_section:
        policy_cfg = config.setdefault(spec.policy_section, {})
        policy_cfg["seed"] = int(seed)
        policy_cfg["deterministic"] = bool(policy_cfg.get("deterministic", True))
        if spec.name in {"A2C", "PPO"}:
            policy_cfg["device"] = "cuda"
    config["revision_experiment"] = {
        "revision_id": "multi_policy_validation",
        "experiment_group": "multi_ads_mc_reference" if is_monte_carlo else "multi_ads_ss",
        "policy": spec.name,
        "event_type": event,
        "seed": int(seed),
        "frozen_protocol": {
            "ss": SS_SETTINGS[event],
            "mc_samples": MC_SAMPLES[event],
            "execution_profile": EXECUTION_PROFILE,
        },
    }
    return config


def _empty_plan(policies: Iterable[str], events: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for policy in policies:
        for event in events:
            for seed in SEEDS:
                rows.append(
                    {
                        "experiment_group": "multi_ads_ss",
                        "policy": policy,
                        "event_type": event,
                        "seed": str(seed),
                        "execution_status": "planned",
                        "quality_status": "pending",
                        "failure_reason": "",
                        "runtime_seconds": "",
                        "summary_path": "",
                        "effective_config_path": "",
                        "effective_config_sha256": "",
                        "updated_utc": _utc_now(),
                    }
                )
    return rows


def _empty_reference_plan(policies: Iterable[str], events: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_group": "multi_ads_mc_reference",
            "policy": policy,
            "event_type": event,
            "seed": "42",
            "num_samples": str(MC_SAMPLES[event]),
            "execution_status": "planned",
            "failure_reason": "",
            "runtime_seconds": "",
            "summary_path": "",
            "effective_config_path": "",
            "effective_config_sha256": "",
            "updated_utc": _utc_now(),
        }
        for policy in policies
        for event in events
    ]


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(row["policy"], row["event_type"], int(row["seed"])): row for row in rows}


def _quality(summary: dict[str, Any]) -> tuple[str, str]:
    reliability = dict(summary.get("reliability", {}) or {})
    if reliability.get("status") == "pass":
        return "pass", ""
    reasons = list(reliability.get("failures", []) or []) + list(reliability.get("warnings", []) or [])
    if not reasons:
        reasons = [f"reliability_status={reliability.get('status', 'missing')}"]
    return "fail", "; ".join(str(item) for item in reasons)


def _expected_event(event: str) -> str:
    return "cut_in" if event == "cutin" else "following"


def _run_subset(
    policy: str,
    event: str,
    seed: int,
    base_config: dict[str, Any],
    row: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    spec = POLICIES[policy]
    run_dir = _run_dir(policy, event, seed)
    summary_path = run_dir / "latent_subset_summary.json"
    config_path = spec.config_path(event)
    if summary_path.is_file() and row.get("execution_status") == "completed" and not overwrite:
        return
    config = _effective_config(spec, event, seed, base_config, run_dir, is_monte_carlo=False)
    config_hash = _canonical_hash(config)
    effective_path = run_dir / "effective_config.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(effective_path, config)
    row.update(
        {
            "execution_status": "running",
            "quality_status": "pending",
            "failure_reason": "",
            "runtime_seconds": "",
            "summary_path": _repo_relative(summary_path),
            "effective_config_path": _repo_relative(effective_path),
            "effective_config_sha256": config_hash,
            "updated_utc": _utc_now(),
        }
    )
    _write_json(run_dir / "run_status.json", dict(row))
    start = time.monotonic()
    try:
        runner = importlib.import_module(f"{spec.package}.src.latent_subset_runner")
        produced = Path(
            runner.run_subset_from_config(
                config, config_path.parent, expected_event_type=_expected_event(event)
            )
        )
        summary = _read_json(produced)
        quality_status, reason = _quality(summary)
        elapsed = time.monotonic() - start
        row.update(
            {
                "execution_status": "completed",
                "quality_status": quality_status,
                "failure_reason": reason,
                "runtime_seconds": f"{elapsed:.6f}",
                "summary_path": _repo_relative(produced),
                "updated_utc": _utc_now(),
            }
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        row.update(
            {
                "execution_status": "failed",
                "quality_status": "fail",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": f"{elapsed:.6f}",
                "updated_utc": _utc_now(),
            }
        )
        row["traceback"] = traceback.format_exc()
        LOGGER.exception("SS run failed: %s/%s seed=%s", policy, event, seed)
    _write_json(run_dir / "run_status.json", row)


def _run_monte_carlo(
    policy: str,
    event: str,
    base_config: dict[str, Any],
    row: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    spec = POLICIES[policy]
    output_dir = _reference_dir(policy, event)
    summary_path = output_dir / "latent_monte_carlo_summary.json"
    config_path = spec.config_path(event)
    if summary_path.is_file() and row.get("execution_status") == "completed" and not overwrite:
        return
    config = _effective_config(spec, event, 42, base_config, output_dir, is_monte_carlo=True)
    config_hash = _canonical_hash(config)
    effective_path = output_dir / "effective_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(effective_path, config)
    row.update(
        {
            "execution_status": "running",
            "failure_reason": "",
            "runtime_seconds": "",
            "summary_path": _repo_relative(summary_path),
            "effective_config_path": _repo_relative(effective_path),
            "effective_config_sha256": config_hash,
            "updated_utc": _utc_now(),
        }
    )
    _write_json(output_dir / "run_status.json", dict(row))
    start = time.monotonic()
    try:
        runner = importlib.import_module(f"{spec.package}.src.latent_subset_runner")
        produced = Path(
            runner.run_monte_carlo_from_config(
                config, config_path.parent, expected_event_type=_expected_event(event)
            )
        )
        summary = _read_json(produced)
        actual = int(summary.get("num_samples", -1))
        if actual != MC_SAMPLES[event]:
            raise RuntimeError(f"MC wrote {actual} samples; expected {MC_SAMPLES[event]}")
        elapsed = time.monotonic() - start
        row.update(
            {
                "execution_status": "completed",
                "failure_reason": "",
                "runtime_seconds": f"{elapsed:.6f}",
                "summary_path": _repo_relative(produced),
                "updated_utc": _utc_now(),
            }
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        row.update(
            {
                "execution_status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": f"{elapsed:.6f}",
                "updated_utc": _utc_now(),
                "traceback": traceback.format_exc(),
            }
        )
        LOGGER.exception("MC run failed: %s/%s", policy, event)
    _write_json(output_dir / "run_status.json", row)


def _run_subset_worker(
    policy: str,
    event: str,
    seed: int,
    base_config: dict[str, Any],
    row: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Spawn-safe outer worker; the parent remains the sole plan writer."""
    # Per-batch progress messages are numerous for the formal MCMC budget and
    # can dominate Windows multi-process I/O.  Audit data live in the status,
    # effective-config and result artifacts, so retain warnings/errors here.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _run_subset(policy, event, seed, base_config, row, overwrite=overwrite)
    return row


def _run_monte_carlo_worker(
    policy: str,
    event: str,
    base_config: dict[str, Any],
    row: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Spawn-safe independent MC reference worker."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _run_monte_carlo(policy, event, base_config, row, overwrite=overwrite)
    return row


def _float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def _t_ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    margin = critical * _sample_std(values) / math.sqrt(len(values))
    return _mean(values) - margin, _mean(values) + margin


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + (z * z / n)
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _acceptance(summary: dict[str, Any]) -> float:
    values = [_float(item) for item in summary.get("acceptance_rates", [])]
    values = [item for item in values if math.isfinite(item)]
    return _mean(values)


def _core_config(config: dict[str, Any]) -> dict[str, Any]:
    subset = dict(config.get("subset_simulation", {}) or {})
    for key in ("output_dir",):
        subset.pop(key, None)
    mc = dict(config.get("monte_carlo", {}) or {})
    for key in ("output_dir", "seed"):
        mc.pop(key, None)
    return {
        "paths": {
            key: value
            for key, value in dict(config.get("paths", {}) or {}).items()
            if key != "idm_ego_config_path"
        },
        "event": config.get("event", {}),
        "env": config.get("env", {}),
        "dynamics": config.get("dynamics", {}),
        "closed_loop_risk": config.get("closed_loop_risk", {}),
        "closed_loop_risk_scoring": config.get("closed_loop_risk_scoring", {}),
        "evt": config.get("evt", {}),
        "physics": config.get("physics", {}),
        "sampling": config.get("sampling", {}),
        "subset_simulation": subset,
        "monte_carlo": mc,
        "mileage_return_period": config.get("mileage_return_period", {}),
        "parallel": config.get("parallel", {}),
    }


def _fairness_check(
    policies: Iterable[str], events: Iterable[str], plan_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    results: dict[str, Any] = {"status": "pass", "events": {}}
    for event in events:
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            baselines: dict[str, Any] = {}
            missing: list[str] = []
            for policy in policies:
                row = next(
                    (item for item in plan_rows if item["policy"] == policy and item["event_type"] == event and int(item["seed"]) == seed),
                    None,
                )
                if not row or not row.get("effective_config_path"):
                    missing.append(policy)
                    continue
                path = ROOT / row["effective_config_path"]
                if not path.is_file():
                    missing.append(policy)
                    continue
                baselines[policy] = _core_config(_read_json(path))
            hashes = {policy: _canonical_hash(value) for policy, value in baselines.items()}
            identical = len(set(hashes.values())) <= 1 and not missing
            per_seed[str(seed)] = {"status": "pass" if identical else "fail", "core_config_sha256": hashes, "missing": missing}
            if not identical:
                results["status"] = "fail"
        results["events"][event] = per_seed
    return results


def _seed_rows(plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in plan_rows:
        row = dict(plan)
        summary_path = ROOT / str(plan.get("summary_path", ""))
        if plan.get("execution_status") == "completed" and summary_path.is_file():
            summary = _read_json(summary_path)
            reliability = dict(summary.get("reliability", {}) or {})
            observed = dict(reliability.get("observed", {}) or {})
            counts = dict(summary.get("simulation_counts", {}) or {})
            mileage = dict(summary.get("mileage_return_period", {}) or {})
            row.update(
                {
                    "probability": _float(summary.get("probability")),
                    "analytic_se": _float(summary.get("probability_standard_error")),
                    "analytic_rse": _float(summary.get("probability_relative_standard_error")),
                    "num_levels": _int(summary.get("num_levels")),
                    "final_failure_fraction": _float(summary.get("final_failure_fraction")),
                    "mean_acceptance_rate": _acceptance(summary),
                    "closed_loop_evaluations": _int(counts.get("closed_loop_evaluations")),
                    "unique_contexts": _int(observed.get("unique_contexts")),
                    "unique_states": _int(observed.get("unique_states")),
                    "largest_context_share": _float(observed.get("largest_context_share")),
                    "largest_state_share": _float(observed.get("largest_state_share")),
                    "reliability_status_observed": str(reliability.get("status", "missing")),
                    "failure_threshold": _float(summary.get("failure_threshold")),
                    "evt_return_level_target": _float(summary.get("evt_return_level_target")),
                    "context_sampling_mode": str(summary.get("context_sampling_mode", "")),
                    "risk_intensity_per_mile": _float(mileage.get("risk_intensity_per_mile")),
                    "return_mileage": _float(mileage.get("return_mileage")),
                    "mileage_strict": str(mileage.get("strict_mileage_interpretation", "")),
                }
            )
        else:
            row.update(
                {
                    "probability": math.nan,
                    "analytic_se": math.nan,
                    "analytic_rse": math.nan,
                    "num_levels": 0,
                    "final_failure_fraction": math.nan,
                    "mean_acceptance_rate": math.nan,
                    "closed_loop_evaluations": 0,
                    "unique_contexts": 0,
                    "unique_states": 0,
                    "largest_context_share": math.nan,
                    "largest_state_share": math.nan,
                    "reliability_status_observed": "missing",
                    "failure_threshold": math.nan,
                    "evt_return_level_target": math.nan,
                    "context_sampling_mode": "",
                    "risk_intensity_per_mile": math.nan,
                    "return_mileage": math.nan,
                    "mileage_strict": "",
                }
            )
        rows.append(row)
    return rows


def _mc_summary(policy: str, event: str, reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = next((item for item in reference_rows if item["policy"] == policy and item["event_type"] == event), None)
    if not row or row.get("execution_status") != "completed":
        return {"status": "not_run"}
    path = ROOT / str(row.get("summary_path", ""))
    if not path.is_file():
        return {"status": "missing_summary"}
    summary = _read_json(path)
    n = _int(summary.get("num_samples"))
    probability = _float(summary.get("probability"))
    successes = int(round(probability * n)) if math.isfinite(probability) else 0
    low, high = _wilson_interval(successes, n)
    return {
        "status": "pass" if n == MC_SAMPLES[event] else "invalid_budget",
        "probability": probability,
        "num_samples": n,
        "successes": successes,
        "wilson_low": low,
        "wilson_high": high,
        "summary_path": row.get("summary_path", ""),
    }


def _summary_rows(
    policies: Iterable[str],
    events: Iterable[str],
    seed_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for event in events:
            selected = [item for item in seed_rows if item["policy"] == policy and item["event_type"] == event]
            completed = [item for item in selected if item.get("execution_status") == "completed"]
            valid = [item for item in completed if item.get("reliability_status_observed") == "pass"]
            estimates = [_float(item.get("probability")) for item in completed]
            estimates = [value for value in estimates if math.isfinite(value)]
            ci_low, ci_high = _t_ci95(estimates)
            mc = _mc_summary(policy, event, reference_rows)
            overlaps = bool(
                math.isfinite(ci_low)
                and math.isfinite(ci_high)
                and mc.get("status") == "pass"
                and ci_low <= float(mc["wilson_high"])
                and ci_high >= float(mc["wilson_low"])
            )
            all_pass = len(completed) == len(SEEDS) and len(valid) == len(SEEDS)
            robust = all_pass and overlaps
            row = {
                "policy": policy,
                "event_type": event,
                "planned_seeds": len(SEEDS),
                "completed_seeds": len(completed),
                "reliability_pass_seeds": len(valid),
                "mean_probability": _mean(estimates),
                "median_probability": median(estimates) if estimates else math.nan,
                "sample_std_probability": _sample_std(estimates),
                "t_ci95_low": ci_low,
                "t_ci95_high": ci_high,
                "total_closed_loop_evaluations": sum(_int(item.get("closed_loop_evaluations")) for item in completed),
                "mean_num_levels": _mean([_float(item.get("num_levels")) for item in completed]),
                "mean_final_failure_fraction": _mean([_float(item.get("final_failure_fraction")) for item in completed]),
                "mean_acceptance_rate": _mean([_float(item.get("mean_acceptance_rate")) for item in completed]),
                "mc_status": mc.get("status"),
                "mc_probability": mc.get("probability", math.nan),
                "mc_num_samples": mc.get("num_samples", 0),
                "mc_wilson_low": mc.get("wilson_low", math.nan),
                "mc_wilson_high": mc.get("wilson_high", math.nan),
                "ss_ci_overlaps_mc_wilson": overlaps,
                "robust_by_predeclared_rule": robust,
                "status": "pass" if robust else "fail",
            }
            rows.append(row)
    return rows


def _relative_risk_rows(
    policies: Iterable[str], events: Iterable[str], seed_rows: list[dict[str, Any]], *, samples: int = 10_000
) -> list[dict[str, Any]]:
    import numpy as np

    output: list[dict[str, Any]] = []
    for event in events:
        by_policy = {
            policy: {int(row["seed"]): _float(row.get("probability")) for row in seed_rows if row["policy"] == policy and row["event_type"] == event}
            for policy in policies
        }
        reference = by_policy.get("IDM", {})
        for policy in policies:
            if policy == "IDM":
                continue
            common = sorted(
                seed
                for seed in SEEDS
                if math.isfinite(reference.get(seed, math.nan)) and math.isfinite(by_policy[policy].get(seed, math.nan)) and reference[seed] > 0.0
            )
            ratios = [by_policy[policy][seed] / reference[seed] for seed in common]
            if ratios:
                rng = np.random.default_rng(20260804 + len(event) + len(policy))
                array = np.asarray(ratios, dtype=float)
                boot = np.mean(rng.choice(array, size=(samples, array.size), replace=True), axis=1)
                low, high = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
            else:
                low = high = math.nan
            output.append(
                {
                    "event_type": event,
                    "policy": policy,
                    "reference_policy": "IDM",
                    "paired_seeds": len(ratios),
                    "mean_relative_risk": _mean(ratios),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "interpretation": "no significant difference observed" if math.isfinite(low) and low <= 1.0 <= high else "interval does not cross 1" if math.isfinite(low) else "insufficient paired seeds",
                }
            )
    return output


def _figures(summary_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - rendering is a convenience artifact
        LOGGER.warning("Cannot generate figures: %s", exc)
        return
    figures = RESULTS_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric, output_name, ylabel in (
        ("mean_probability", "multi_ads_probability.png", "failure probability"),
        ("return_mileage", "multi_ads_return_mileage.png", "return mileage"),
    ):
        fig, axes = plt.subplots(1, len(EVENTS), figsize=(12, 4.5), squeeze=False)
        for axis, event in zip(axes[0], EVENTS):
            selected = [row for row in summary_rows if row["event_type"] == event]
            labels = [row["policy"] for row in selected]
            if metric == "mean_probability":
                values = [_float(row[metric], 0.0) for row in selected]
                errors = [
                    max(0.0, _float(row["t_ci95_high"], value) - value)
                    for row, value in zip(selected, values)
                ]
            else:
                values = []
                errors = []
                for row in selected:
                    policy_values = [
                        _float(seed["return_mileage"])
                        for seed in seed_rows
                        if seed["policy"] == row["policy"] and seed["event_type"] == event
                    ]
                    policy_values = [value for value in policy_values if math.isfinite(value)]
                    values.append(_mean(policy_values) if policy_values else 0.0)
                    errors.append(0.0)
            colors = ["#4472c4" if row.get("status") == "pass" else "#c0504d" for row in selected]
            axis.bar(labels, values, yerr=errors, capsize=4, color=colors)
            axis.set_title(event)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            if metric == "mean_probability":
                for idx, row in enumerate(selected):
                    if row.get("mc_status") == "pass":
                        axis.plot(idx, _float(row["mc_probability"]), "ko", label="MC" if idx == 0 else None)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=1)
        fig.tight_layout()
        fig.savefig(figures / output_name, dpi=180)
        plt.close(fig)


def _summarize(policies: Iterable[str], events: Iterable[str]) -> dict[str, Any]:
    plan_rows = _read_csv(RUN_PLAN_PATH)
    reference_rows = _read_csv(REFERENCE_PLAN_PATH)
    seed_rows = _seed_rows(plan_rows)
    summary_rows = _summary_rows(policies, events, seed_rows, reference_rows)
    fairness = _fairness_check(policies, events, plan_rows)
    ratio_rows = _relative_risk_rows(policies, events, seed_rows)
    tables = RESULTS_ROOT / "tables"
    seed_fields = tuple(dict.fromkeys([*PLAN_FIELDS, *[key for row in seed_rows for key in row.keys()]]))
    summary_fields = tuple(dict.fromkeys([key for row in summary_rows for key in row.keys()]))
    ratio_fields = tuple(dict.fromkeys([key for row in ratio_rows for key in row.keys()]))
    _write_csv(tables / "multi_ads_seed_level.csv", seed_fields, seed_rows)
    _write_csv(tables / "multi_ads_summary.csv", summary_fields, summary_rows)
    _write_csv(tables / "multi_ads_relative_risk.csv", ratio_fields, ratio_rows)
    _write_json(RESULTS_ROOT / "fairness_check.json", fairness)
    _figures(summary_rows, seed_rows)
    all_ss_completed = all(row.get("completed_seeds") == len(SEEDS) for row in summary_rows)
    all_robust = all(bool(row.get("robust_by_predeclared_rule")) for row in summary_rows)
    status = {
        "revision_id": "multi_policy_validation",
        "updated_utc": _utc_now(),
        "has_recorded_execution": any(row.get("execution_status") in {"completed", "failed"} for row in plan_rows),
        "all_ss_seed_runs_completed": all_ss_completed,
        "all_policy_event_results_robust": all_robust,
        "fairness_status": fairness["status"],
        "has_valid_multi_ads_result": bool(all_ss_completed and all_robust and fairness["status"] == "pass"),
        "required_ss_runs": len(list(policies)) * len(list(events)) * len(SEEDS),
        "completed_ss_runs": sum(row.get("execution_status") == "completed" for row in plan_rows),
        "failed_ss_runs": sum(row.get("execution_status") == "failed" for row in plan_rows),
        "completed_mc_references": sum(row.get("execution_status") == "completed" for row in reference_rows),
        "message": "A result is publication-ready only when all planned independent SS repeats pass reliability, the cross-seed SS interval overlaps the matching high-budget MC Wilson interval, and fairness passes.",
    }
    _write_json(SUMMARY_STATUS_PATH, status)
    return status


def _run_playbacks(policies: Iterable[str], events: Iterable[str], configs: dict[tuple[str, str], dict[str, Any]]) -> None:
    summary_rows = _read_csv(RESULTS_ROOT / "tables" / "multi_ads_summary.csv")
    seed_rows = _read_csv(RESULTS_ROOT / "tables" / "multi_ads_seed_level.csv")
    result_rows: list[dict[str, Any]] = []
    for event in events:
        candidates = [
            row
            for row in summary_rows
            if row["event_type"] == event and row.get("completed_seeds") == str(len(SEEDS))
        ]
        candidates.sort(key=lambda row: _float(row.get("mean_probability"), -math.inf), reverse=True)
        for candidate in candidates[:2]:
            policy = candidate["policy"]
            policy_seeds = [
                row
                for row in seed_rows
                if row["policy"] == policy and row["event_type"] == event and row.get("execution_status") == "completed"
            ]
            if not policy_seeds:
                continue
            selected = max(policy_seeds, key=lambda row: _float(row.get("probability"), -math.inf))
            seed = int(selected["seed"])
            spec = POLICIES[policy]
            config = _effective_config(spec, event, seed, configs[(policy, event)], _run_dir(policy, event, seed), is_monte_carlo=False)
            config["subset_simulation"]["output_dir"] = _relative_to_config(_run_dir(policy, event, seed), spec.config_path(event).parent)
            output_dir = RESULTS_ROOT / "playbacks" / event / policy.lower() / f"seed_{seed}"
            from tools import final_level_playback as module

            defaults = module.SCRIPT_DEFAULTS
            original = dict(defaults)
            try:
                defaults.update(
                    {
                        "samples_path": _relative_to_config(_run_dir(policy, event, seed) / "latent_subset_samples.npz", spec.config_path(event).parent),
                        "output_dir": _relative_to_config(output_dir, spec.config_path(event).parent),
                        "num_cases": 5,
                        "random_seed": 20260804,
                        "level": -1,
                        "render_gif": False,
                        "unique_test_scenarios": True,
                    }
                )
                manifest = module.replay_final_level(
                    config,
                    spec.config_path(event).parent,
                    subset_name=spec.package,
                    expected_event_type=_expected_event(event),
                )
                payload = _read_json(manifest)
                drift = [
                    abs(_float(case.get("stored_score")) - _float(case.get("replayed_score")))
                    for case in payload.get("cases", [])
                    if math.isfinite(_float(case.get("stored_score"))) and math.isfinite(_float(case.get("replayed_score")))
                ]
                result_rows.append(
                    {
                        "policy": policy,
                        "event_type": event,
                        "seed": seed,
                        "status": "pass",
                        "manifest_path": _repo_relative(manifest),
                        "num_cases": len(payload.get("cases", [])),
                        "max_score_drift": max(drift) if drift else math.nan,
                    }
                )
            except Exception as exc:
                LOGGER.exception("Playback failed: %s/%s seed=%s", policy, event, seed)
                result_rows.append(
                    {
                        "policy": policy,
                        "event_type": event,
                        "seed": seed,
                        "status": "fail",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                defaults.clear()
                defaults.update(original)
    fields = tuple(dict.fromkeys([key for row in result_rows for key in row.keys()]))
    _write_csv(RESULTS_ROOT / "tables" / "multi_ads_playback_validation.csv", fields, result_rows)


def _initialise(
    policies: list[str], events: list[str], configs: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frozen_inputs = _frozen_inputs(policies, events, configs)
    design = _manifest_design(policies, events, frozen_inputs)
    if MANIFEST_PATH.is_file():
        manifest = _read_json(MANIFEST_PATH)
        old_design = dict(manifest.get("frozen_design", {}) or {})
        if old_design != design:
            existing_rows = _read_csv(RUN_PLAN_PATH)
            if any(row.get("execution_status") == "completed" for row in existing_rows):
                raise RuntimeError("Existing revision result directory has a different frozen design or input hash; do not mix runs.")
            # A bootstrap/import failure before a single closed-loop evaluation
            # creates no scientific result.  Permit the corrected driver to
            # refresh its own fingerprint while preserving the failed rows and
            # their log files as an audit trail.
            history = list(manifest.get("initialisation_history", []) or [])
            history.append(
                {
                    "updated_utc": _utc_now(),
                    "reason": "No completed simulation existed; refreshed driver/input fingerprint after bootstrap repair.",
                    "previous_frozen_design_sha256": _canonical_hash(old_design),
                }
            )
            manifest.update(
                {
                    "frozen_design": design,
                    "environment": _environment(),
                    "initialisation_history": history,
                }
            )
            _write_json(FROZEN_INPUTS_PATH, frozen_inputs)
            _write_json(MANIFEST_PATH, manifest)
    else:
        _write_json(FROZEN_INPUTS_PATH, frozen_inputs)
        _write_json(
            MANIFEST_PATH,
            {
                "experiment": "highD_multi_ADS_subset_simulation",
                "created_utc": _utc_now(),
                "repository_root": ".",
                "frozen_design": design,
                "environment": _environment(),
                "runner_contract": "Each policy uses its existing run_subset_from_config/run_monte_carlo_from_config; this driver only freezes the common protocol and summaries.",
            },
        )
    plan_rows = _read_csv(RUN_PLAN_PATH) or _empty_plan(policies, events)
    reference_rows = _read_csv(REFERENCE_PLAN_PATH) or _empty_reference_plan(policies, events)
    _write_csv(RUN_PLAN_PATH, PLAN_FIELDS, plan_rows)
    _write_csv(REFERENCE_PLAN_PATH, REFERENCE_FIELDS, reference_rows)
    return plan_rows, reference_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(POLICIES) + ("all",), default="all")
    parser.add_argument("--event", choices=EVENTS + ("all",), default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Run only one SS seed.")
    parser.add_argument("--run-mc", action="store_true", help="Run high-budget independent MC references.")
    parser.add_argument("--run-playbacks", action="store_true", help="Replay two highest-risk policy cases per event after SS completion.")
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2, 4),
        default=4,
        help="Concurrent independent policy-event-seed jobs; 4 matches the validated IDM sensitivity scheduler.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun selected existing units.")
    parser.add_argument("--dry-run", action="store_true", help="Create/verify audit plan without simulation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    policies = list(POLICIES) if args.policy == "all" else [args.policy]
    events = list(EVENTS) if args.event == "all" else [args.event]
    configs = _base_configs(policies, events)
    plan_rows, reference_rows = _initialise(policies, events, configs)
    plan_index = _index_rows(plan_rows)
    reference_index = _index_rows(reference_rows)
    if not args.dry_run:
        selected_seeds = (args.seed,) if args.seed else SEEDS
        units = [(policy, event, seed) for policy in policies for event in events for seed in selected_seeds]
        if args.workers == 1:
            for policy, event, seed in units:
                row = plan_index[(policy, event, seed)]
                LOGGER.info("Run SS: policy=%s event=%s seed=%s", policy, event, seed)
                _run_subset(policy, event, seed, configs[(policy, event)], row, overwrite=args.overwrite)
                _write_csv(RUN_PLAN_PATH, PLAN_FIELDS, plan_rows)
        else:
            # Spawn avoids inheriting a live CUDA context.  Each worker writes
            # only its own run directory; this parent owns run_plan.csv.
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
                futures = {
                    executor.submit(
                        _run_subset_worker,
                        policy,
                        event,
                        seed,
                        configs[(policy, event)],
                        dict(plan_index[(policy, event, seed)]),
                        args.overwrite,
                    ): (policy, event, seed)
                    for policy, event, seed in units
                }
                for future in as_completed(futures):
                    policy, event, seed = futures[future]
                    row = plan_index[(policy, event, seed)]
                    try:
                        row.clear()
                        row.update(future.result())
                    except Exception as exc:  # defensive worker bootstrap audit
                        row.update(
                            {
                                "execution_status": "failed",
                                "quality_status": "fail",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "updated_utc": _utc_now(),
                            }
                        )
                        LOGGER.exception("Outer SS worker failed: %s/%s seed=%s", policy, event, seed)
                    _write_csv(RUN_PLAN_PATH, PLAN_FIELDS, plan_rows)
        if args.run_mc:
            mc_units = [(policy, event) for policy in policies for event in events]
            if args.workers == 1:
                for policy, event in mc_units:
                    row = reference_index[(policy, event, 42)]
                    LOGGER.info("Run MC: policy=%s event=%s N=%s", policy, event, MC_SAMPLES[event])
                    _run_monte_carlo(policy, event, configs[(policy, event)], row, overwrite=args.overwrite)
                    _write_csv(REFERENCE_PLAN_PATH, REFERENCE_FIELDS, reference_rows)
            else:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
                    futures = {
                        executor.submit(
                            _run_monte_carlo_worker,
                            policy,
                            event,
                            configs[(policy, event)],
                            dict(reference_index[(policy, event, 42)]),
                            args.overwrite,
                        ): (policy, event)
                        for policy, event in mc_units
                    }
                    for future in as_completed(futures):
                        policy, event = futures[future]
                        row = reference_index[(policy, event, 42)]
                        try:
                            row.clear()
                            row.update(future.result())
                        except Exception as exc:
                            row.update(
                                {
                                    "execution_status": "failed",
                                    "failure_reason": f"{type(exc).__name__}: {exc}",
                                    "updated_utc": _utc_now(),
                                }
                            )
                            LOGGER.exception("Outer MC worker failed: %s/%s", policy, event)
                        _write_csv(REFERENCE_PLAN_PATH, REFERENCE_FIELDS, reference_rows)
    status = _summarize(policies, events)
    if args.run_playbacks and not args.dry_run:
        _run_playbacks(policies, events, configs)
    LOGGER.info("Revision status: %s", json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
