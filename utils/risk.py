"""Unified safety/risk scoring used by highD, adversarial, and subset workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .evt import GPDTailModel, load_evt_model


DEFAULT_RISK_SCORING: dict[str, float] = {
    "ttc_weight": 2.0,
    "thw_weight": 1.0,
    "drac_weight": 2.0,
    "gap_weight": 1.0,
    "ttc_scale": 1.0,
    "thw_scale": 1.0,
    "drac_scale": 5.0,
    "gap_scale": 1.0,
    "ttc_eps": 0.2,
    "thw_eps": 0.2,
    "gap_eps": 0.5,
    "pool_beta": 8.0,
}


def resolve_risk_scoring(
    config: dict[str, Any],
    scoring_section: str = "risk_scoring",
) -> dict[str, float]:
    """Return the canonical risk-scoring config.

    `scoring_section` is the authoritative section.
    """
    cfg: dict[str, Any] = dict(DEFAULT_RISK_SCORING)
    cfg.update(dict(config.get(scoring_section, {})))
    return {key: float(value) for key, value in cfg.items()}


def softplus_np(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(value, 0.0)


def softmax_pool_np(value: np.ndarray, beta: float) -> float:
    if value.size == 0:
        return 0.0
    scaled = float(beta) * np.asarray(value, dtype=np.float64)
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(scaled)
    weights = weights / max(float(np.sum(weights)), 1.0e-12)
    return float(np.sum(weights * value))


def longitudinal_series_from_states(
    states: np.ndarray,
    ego_length: float,
    lead_length: float,
) -> dict[str, np.ndarray]:
    """Return per-frame longitudinal safety metrics for ego/lead states."""
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 2 or arr.shape[2] < 5:
        raise ValueError(
            "states must have shape (T, >=2, >=5) with ego at index 0 "
            "and lead at index 1"
        )
    ego = arr[:, 0]
    lead = arr[:, 1]
    return longitudinal_series_from_arrays(
        gap=lead[:, 0] - ego[:, 0] - 0.5 * (float(ego_length) + float(lead_length)),
        ego_speed=ego[:, 2],
        lead_speed=lead[:, 2],
        ego_accel=ego[:, 4],
    )


def longitudinal_series_from_arrays(
    *,
    gap: np.ndarray,
    ego_speed: np.ndarray,
    lead_speed: np.ndarray,
    ego_accel: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the shared longitudinal metric series from aligned arrays."""
    gap_arr = np.asarray(gap, dtype=np.float64)
    ego_v = np.asarray(ego_speed, dtype=np.float64)
    lead_v = np.asarray(lead_speed, dtype=np.float64)
    if gap_arr.shape != ego_v.shape or gap_arr.shape != lead_v.shape:
        raise ValueError("gap, ego_speed, and lead_speed must have matching shapes")
    closing = ego_v - lead_v
    positive_closing = closing > 1.0e-6
    valid_gap = gap_arr > 1.0e-6
    ttc = np.where(
        valid_gap & positive_closing,
        gap_arr / np.maximum(closing, 1.0e-6),
        1000.0,
    )
    thw = np.where(
        valid_gap & (ego_v > 1.0e-6),
        gap_arr / np.maximum(ego_v, 1.0e-6),
        1000.0,
    )
    drac = np.where(
        valid_gap & positive_closing,
        np.square(closing) / np.maximum(2.0 * gap_arr, 1.0e-6),
        0.0,
    )
    if ego_accel is None:
        accel = np.zeros_like(gap_arr)
    else:
        accel = np.asarray(ego_accel, dtype=np.float64)
        if accel.shape != gap_arr.shape:
            raise ValueError("ego_accel must match gap shape")
    return {
        "gap": gap_arr.astype(np.float32),
        "ego_speed": ego_v.astype(np.float32),
        "lead_speed": lead_v.astype(np.float32),
        "closing_speed": closing.astype(np.float32),
        "ttc": np.clip(ttc, 0.0, 1000.0).astype(np.float32),
        "thw": np.clip(thw, 0.0, 1000.0).astype(np.float32),
        "drac": np.clip(drac, 0.0, 1000.0).astype(np.float32),
        "ego_accel": accel.astype(np.float32),
    }


def longitudinal_proxy_from_series(
    series: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    scoring_section: str = "longitudinal_risk_scoring",
) -> dict[str, float]:
    """Score a longitudinal metric series without RSS terms."""
    gap = np.asarray(series.get("gap", []), dtype=np.float64)
    if gap.size == 0:
        zero_keys = (
            "ttc_objective",
            "thw_objective",
            "gap_objective",
            "drac_objective",
            "ttc_score",
            "thw_score",
            "gap_score",
            "drac_score",
            "proxy_risk_score",
        )
        return {key: 0.0 for key in zero_keys}

    scoring = resolve_risk_scoring(config, scoring_section)
    pool_beta = scoring["pool_beta"]
    ttc_eps = max(scoring["ttc_eps"], 1.0e-6)
    thw_eps = max(scoring["thw_eps"], 1.0e-6)
    gap_eps = max(scoring["gap_eps"], 1.0e-6)

    ttc = np.asarray(series["ttc"], dtype=np.float64)
    thw = np.asarray(series["thw"], dtype=np.float64)
    drac = np.asarray(series["drac"], dtype=np.float64)
    ttc_raw = softmax_pool_np(1.0 / np.maximum(ttc, ttc_eps), pool_beta)
    thw_raw = softmax_pool_np(1.0 / np.maximum(thw, thw_eps), pool_beta)
    gap_raw = softmax_pool_np(1.0 / np.maximum(gap, gap_eps), pool_beta)
    drac_raw = softmax_pool_np(np.maximum(drac, 0.0), pool_beta)

    ttc_objective = ttc_raw / max(scoring["ttc_scale"], 1.0e-6)
    thw_objective = thw_raw / max(scoring["thw_scale"], 1.0e-6)
    gap_objective = gap_raw / max(scoring["gap_scale"], 1.0e-6)
    drac_objective = drac_raw / max(scoring["drac_scale"], 1.0e-6)
    ttc_score = scoring["ttc_weight"] * ttc_objective
    thw_score = scoring["thw_weight"] * thw_objective
    gap_score = scoring["gap_weight"] * gap_objective
    drac_score = scoring["drac_weight"] * drac_objective
    proxy_score = ttc_score + thw_score + gap_score + drac_score
    return {
        "ttc_objective": float(ttc_objective),
        "thw_objective": float(thw_objective),
        "gap_objective": float(gap_objective),
        "drac_objective": float(drac_objective),
        "ttc_score": float(ttc_score),
        "thw_score": float(thw_score),
        "gap_score": float(gap_score),
        "drac_score": float(drac_score),
        "proxy_risk_score": float(proxy_score),
    }


def longitudinal_proxy_from_trace(
    trace: list[dict[str, float]],
    config: dict[str, Any],
    *,
    scoring_section: str = "longitudinal_risk_scoring",
) -> dict[str, float]:
    """Score a highway-env trace with the shared longitudinal formula."""
    if not trace:
        return longitudinal_proxy_from_series(
            {},
            config,
            scoring_section=scoring_section,
        )

    gap = np.asarray([row["gap"] for row in trace], dtype=np.float64)
    ego_speed = np.asarray([row["ego_speed"] for row in trace], dtype=np.float64)
    lead_speed = np.asarray(
        [row["lead_speed"] for row in trace],
        dtype=np.float64,
    )
    ego_accel = np.asarray(
        [row.get("ego_accel", row.get("ego_action_accel", 0.0)) for row in trace],
        dtype=np.float64,
    )
    series = longitudinal_series_from_arrays(
        gap=gap,
        ego_speed=ego_speed,
        lead_speed=lead_speed,
        ego_accel=ego_accel,
    )
    return longitudinal_proxy_from_series(
        series,
        config,
        scoring_section=scoring_section,
    )


def closed_loop_proxy_from_trace(
    trace: list[dict[str, float]],
    config: dict[str, Any],
    rss_cfg: Any | None = None,
    *,
    scoring_section: str = "longitudinal_risk_scoring",
) -> dict[str, float]:
    """Backward-compatible alias for the RSS-free longitudinal trace score."""
    return longitudinal_proxy_from_trace(
        trace,
        config,
        scoring_section=scoring_section,
    )


def _evt_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("evt", {}))


def _evt_model_path(config: dict[str, Any]) -> str | None:
    evt_cfg = _evt_config(config)
    if evt_cfg.get("model_path"):
        return str(evt_cfg["model_path"])
    paths = config.get("paths", {})
    if paths.get("evt_model_path"):
        return str(paths["evt_model_path"])
    return None


def _evt_score_space(config: dict[str, Any]) -> str:
    evt_cfg = _evt_config(config)
    if "score_space" in evt_cfg:
        return str(evt_cfg["score_space"]).lower()
    return str(config.get("closed_loop_risk", {}).get("score_space", "raw")).lower()


def evt_model_from_config(config: dict[str, Any]) -> tuple[GPDTailModel | None, str | None]:
    """Load and cache the configured EVT model when EVT scoring is requested."""
    score_space = _evt_score_space(config)
    if score_space not in {"evt", "s_evt"}:
        return None, None
    path_value = _evt_model_path(config)
    if not path_value:
        raise KeyError("EVT score_space requires evt.model_path or paths.evt_model_path")
    path = Path(path_value)
    if not path.exists() and not path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        candidates = (
            root / path,
            root / "subset" / "scripts" / "configs" / path,
            root / "adversaray" / "scripts" / "configs" / path,
        )
        path = next((candidate.resolve() for candidate in candidates if candidate.exists()), path)
    if not path.exists():
        raise FileNotFoundError(f"EVT model not found: {path}")
    cache = config.setdefault("_evt_model_cache", {})
    key = str(path.resolve())
    if key not in cache:
        cache[key] = load_evt_model(path)
    return cache[key], key


def apply_closed_loop_risk(
    metrics: dict[str, float],
    trace: list[dict[str, float]],
    config: dict[str, Any],
    rss_cfg: Any | None = None,
    *,
    scoring_section: str = "longitudinal_risk_scoring",
) -> float:
    """Update rollout metrics and return the canonical closed-loop risk."""
    cfg = config.get("closed_loop_risk", {})
    collision = float(metrics["collision"])
    hard_brake_threshold = float(cfg.get("hard_brake_threshold", -4.0))
    hard_brake = max(
        0.0,
        hard_brake_threshold - float(metrics["min_ego_accel"]),
    ) / max(abs(hard_brake_threshold), 1.0e-6)
    proxy = longitudinal_proxy_from_trace(
        trace,
        config,
        scoring_section=scoring_section,
    )
    collision_risk_score = float(cfg.get("collision_bonus", 20.0)) * collision
    near_collision_risk_score = float(cfg.get("near_collision_weight", 0.0)) * float(
        metrics.get("near_collision", 0.0)
    )
    hard_brake_risk_score = float(cfg.get("hard_brake_weight", 1.0)) * hard_brake
    y_long = float(
        collision_risk_score
        + near_collision_risk_score
        + proxy["proxy_risk_score"]
        + hard_brake_risk_score
    )
    evt_model, evt_path = evt_model_from_config(config)
    risk_score = y_long
    evt_tail_probability = float("nan")
    evt_return_level_target = float("nan")
    evt_failure_threshold = float("nan")
    if evt_model is not None:
        risk_score = float(evt_model.score(y_long))
        evt_tail_probability = float(evt_model.survival(y_long))
        return_period = int(_evt_config(config).get("return_period", 100))
        evt_return_level_target = float(evt_model.return_level(return_period))
        evt_failure_threshold = float(evt_model.score(evt_return_level_target))
    physics_penalty_score = -float(cfg.get("lead_physics_weight", 0.1)) * float(
        metrics["lead_physics_penalty"]
    )
    validity_penalized_score = float(risk_score + physics_penalty_score)
    metrics.update(
        {
            "risk_score": risk_score,
            "y_long": y_long,
            "proxy_risk_score": proxy["proxy_risk_score"],
            "collision_risk_score": float(collision_risk_score),
            "near_collision_risk_score": float(near_collision_risk_score),
            "ttc_objective": proxy["ttc_objective"],
            "thw_objective": proxy["thw_objective"],
            "drac_objective": proxy["drac_objective"],
            "gap_objective": proxy["gap_objective"],
            "ttc_risk_score": proxy["ttc_score"],
            "thw_risk_score": proxy["thw_score"],
            "drac_risk_score": proxy["drac_score"],
            "gap_risk_score": proxy["gap_score"],
            "hard_brake_severity": float(hard_brake),
            "hard_brake_risk_score": float(hard_brake_risk_score),
            "evt_tail_probability": evt_tail_probability,
            "evt_return_level_target": evt_return_level_target,
            "evt_failure_threshold": evt_failure_threshold,
            "evt_model_path": evt_path or "",
            "physics_penalty_score": float(physics_penalty_score),
            "validity_penalized_score": validity_penalized_score,
        }
    )
    return float(risk_score)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out.astype(np.float32)
    order = np.argsort(arr[finite], kind="mergesort")
    ranks = np.empty(int(finite.sum()), dtype=np.float64)
    ranks[order] = np.arange(1, int(finite.sum()) + 1, dtype=np.float64)
    out[finite] = ranks / max(int(finite.sum()), 1)
    return out.astype(np.float32)


def torch_plan_proxy_risk(
    kin: Any,
    config: dict[str, Any],
    rss_cfg: Any,
    *,
    reference_kin: Any | None = None,
    scoring_section: str = "risk_scoring",
) -> tuple[Any, dict[str, Any]]:
    """Torch version of the canonical differentiable proxy risk."""
    import torch
    import torch.nn.functional as F

    scoring = resolve_risk_scoring(config, scoring_section)
    pool_beta = float(scoring.get("pool_beta", float(rss_cfg.pool_beta)))
    temperature = max(float(rss_cfg.temperature), 1.0e-6)
    safe = _torch_rss_safe_distance(
        kin.ego_velocity,
        torch.clamp(kin.velocity, min=0.0),
        rss_cfg,
    )
    margin = kin.gap - safe

    closing_speed = kin.ego_velocity - kin.velocity
    ttc_eps = max(float(scoring.get("ttc_eps", 0.2)), 1.0e-6)
    gap_eps = max(float(scoring.get("gap_eps", 0.5)), 1.0e-6)
    positive_closing = closing_speed > 0.0
    ttc = torch.where(
        positive_closing,
        torch.clamp(kin.gap, min=gap_eps) / torch.clamp(closing_speed, min=ttc_eps),
        torch.full_like(kin.gap, 1000.0),
    )
    ttc_per_t = torch.where(
        positive_closing,
        1.0 / torch.clamp(ttc, min=ttc_eps),
        torch.zeros_like(ttc),
    )
    ttc_objective_raw = _torch_softmax_pool(ttc_per_t, beta=pool_beta, dim=1)

    drac = closing_speed.square() / torch.clamp(2.0 * kin.gap, min=gap_eps)
    drac = torch.where(positive_closing, drac, torch.zeros_like(drac))
    drac_objective_raw = _torch_softmax_pool(drac, beta=pool_beta, dim=1)

    gap_target = float(scoring.get("gap_target", 20.0))
    gap_objective_raw = _torch_softmax_pool(
        F.softplus(gap_target - kin.gap),
        beta=pool_beta,
        dim=1,
    )

    relative_safe_min = max(float(rss_cfg.relative_safe_min), 1.0e-6)
    relative_margin = margin / torch.clamp(safe, min=relative_safe_min)
    relative_violation = F.softplus(-relative_margin / temperature)
    relative_rss = _torch_softmax_pool(
        relative_violation,
        beta=pool_beta,
        dim=1,
    )

    if str(rss_cfg.mode).lower() == "delta":
        if reference_kin is None:
            raise ValueError("delta RSS mode requires reference_kin")
        ref_safe = _torch_rss_safe_distance(
            reference_kin.ego_velocity,
            torch.clamp(reference_kin.velocity, min=0.0),
            rss_cfg,
        )
        ref_margin = reference_kin.gap - ref_safe
        if margin.shape != ref_margin.shape:
            raise ValueError(
                "delta RSS requires matching adversary/reference horizons, "
                f"got {tuple(margin.shape)} and {tuple(ref_margin.shape)}"
            )
        delta_margin = ref_margin.detach() - margin
        delta_temperature = max(float(rss_cfg.delta_temperature), 1.0e-6)
        centered = F.softplus(delta_margin / delta_temperature)
        centered -= F.softplus(
            torch.zeros((), dtype=delta_margin.dtype, device=delta_margin.device)
        )
        delta_violation = torch.clamp(centered, min=0.0)
        delta_rss = _torch_softmax_pool(delta_violation, beta=pool_beta, dim=1)
    else:
        delta_margin = torch.zeros_like(margin)
        delta_violation = torch.zeros_like(margin)
        delta_rss = torch.zeros_like(torch.min(margin, dim=1).values)

    if kin.ego_action_accel is not None:
        ego_accel = kin.ego_action_accel
    else:
        ego_accel = kin.ego_acceleration
    insufficient_brake = torch.sigmoid(
        (ego_accel + float(rss_cfg.ego_min_brake))
        / max(float(rss_cfg.improper_temperature), 1.0e-6)
    )
    improper_soft = relative_violation * insufficient_brake
    improper_rss = _torch_softmax_pool(improper_soft, beta=pool_beta, dim=1)

    ttc_objective = ttc_objective_raw / max(
        float(scoring.get("ttc_scale", 1.0)), 1.0e-6
    )
    drac_objective = drac_objective_raw / max(
        float(scoring.get("drac_scale", 5.0)), 1.0e-6
    )
    gap_objective = gap_objective_raw / max(
        float(scoring.get("gap_scale", 20.0)), 1.0e-6
    )
    objective = (
        float(scoring.get("delta_rss_weight", 0.0)) * delta_rss
        + float(scoring.get("improper_rss_weight", 0.0)) * improper_rss
        + float(scoring.get("ttc_weight", 1.0)) * ttc_objective
        + float(scoring.get("drac_weight", 1.0)) * drac_objective
        + float(scoring.get("gap_weight", 0.5)) * gap_objective
    )
    return objective, {
        "rss_objective": delta_rss,
        "relative_rss_objective": relative_rss,
        "delta_rss_objective": delta_rss,
        "improper_rss_objective": improper_rss,
        "ttc_objective": ttc_objective,
        "drac_objective": drac_objective,
        "gap_objective": gap_objective,
        "risk_objective": objective,
        "rss_margin": margin,
        "rss_safe_distance": safe,
        "min_rss_margin": torch.min(margin, dim=1).values,
        "rss_violation_rate": (margin < 0.0).to(kin.gap.dtype).mean(dim=1),
        "relative_rss_margin": relative_margin,
        "relative_rss_violation_soft": relative_violation,
        "relative_rss_safe_distance": safe,
        "min_relative_rss_margin": torch.min(relative_margin, dim=1).values,
        "delta_rss_margin": delta_margin,
        "delta_rss_violation_soft": delta_violation,
        "max_delta_rss_margin": torch.max(delta_margin, dim=1).values,
        "rss_improper_soft": improper_soft,
        "rss_improper_fraction": (
            (relative_margin < 0.0) & (ego_accel > -float(rss_cfg.ego_min_brake))
        )
        .to(kin.gap.dtype)
        .mean(dim=1),
        "min_gap": torch.min(kin.gap, dim=1).values,
        "min_ttc": torch.min(torch.clamp(ttc, min=0.0, max=1000.0), dim=1).values,
    }


def _torch_rss_safe_distance(
    ego_velocity: Any,
    lead_velocity: Any,
    cfg: Any,
) -> Any:
    rho = float(cfg.response_time)
    ego_after_response = ego_velocity + rho * float(cfg.ego_max_accel)
    ego_distance = ego_velocity * rho + 0.5 * float(cfg.ego_max_accel) * rho * rho
    ego_brake_distance = ego_after_response.square() / max(
        2.0 * float(cfg.ego_min_brake),
        1.0e-6,
    )
    lead_brake_distance = lead_velocity.square() / max(
        2.0 * float(cfg.lead_max_brake),
        1.0e-6,
    )
    import torch

    return torch.clamp(
        ego_distance + ego_brake_distance - lead_brake_distance,
        min=0.0,
    )


def _torch_softmax_pool(x: Any, beta: float = 8.0, dim: int = 1) -> Any:
    import torch

    weights = torch.softmax(float(beta) * x, dim=dim)
    return torch.sum(weights * x, dim=dim)
