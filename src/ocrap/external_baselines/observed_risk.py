from __future__ import annotations

"""Deployable observation-only risk models shared by external baselines.

Teacher counterfactual tensors are deliberately excluded.  The module predicts a
small deterministic multi-modal actor set from visible history, caches that set
once per candidate group, and vectorizes candidate scoring.  It exposes both
scalar risk summaries and temporal/mode-resolved curves required by contingency
planning, predictive safety filters, calibration diagnostics, and videos.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ObservedRiskContext:
    hypothesis_names: tuple[str, ...]
    weights: np.ndarray                    # [H]
    times: np.ndarray                      # [T]
    actor_xy: np.ndarray                   # [H,A,T,2]
    actor_velocity: np.ndarray             # [H,A,T,2]
    actor_radius: np.ndarray               # [A]
    clearance_buffer_m: float
    # Earliest time at which two environment hypotheses are observably
    # distinguishable and remain distinguishable.  This is used by the robust
    # scenario-MPC adapter to enforce non-anticipativity until the obstacle mode
    # can be inferred, rather than (incorrectly) keying branching from ego-vs-
    # nominal trajectory divergence.
    mode_distinguish_step: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    mode_divergence_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=float))
    weight_source: str = "fixed_prior"


@dataclass(frozen=True)
class ObservedRiskProfile:
    losses: np.ndarray
    weights: np.ndarray
    margins: np.ndarray
    min_clearance: float
    min_ttc: float
    collision_probability: float
    expected_loss: float
    cvar_loss: float
    worst_loss: float
    backup_margin: float
    severity_proxy: float
    hypothesis_names: tuple[str, ...] = ()
    collision_probabilities: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    min_ttc_by_mode: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    closest_approach_time_by_mode: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    severity_by_mode: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    clearance_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    loss_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    backup_margin_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    mode_distinguish_step: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    mode_divergence_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=float))
    weight_source: str = "fixed_prior"


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    try:
        return float(pcfg.get(key, default))
    except Exception:
        return float(default)


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool = False) -> bool:
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    value = pcfg.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _weighted_upper_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    values, weights = values[valid], weights[valid]
    weights = weights / max(float(weights.sum()), 1e-9)
    order = np.argsort(values)[::-1]
    values, weights = values[order], weights[order]
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    total = acc = 0.0
    for value, weight in zip(values, weights):
        take = min(float(weight), alpha - total)
        if take <= 0:
            break
        acc += float(value) * take
        total += take
    return float(acc / max(total, 1e-9))


def _resample_xy(xy: np.ndarray, count: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return np.zeros((count, 2), dtype=float)
    if xy.shape[0] == count:
        return np.nan_to_num(xy[:, :2])
    src = np.linspace(0.0, 1.0, xy.shape[0])
    dst = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(dst, src, xy[:, 0]), np.interp(dst, src, xy[:, 1])], axis=-1)


def _ego_candidate(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 2:
        ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=float).reshape(-1)
        states = np.zeros((2, 9), dtype=float)
        states[:, : min(ego.size, 9)] = ego[:9]
    xy = states[:, :2]
    if states.shape[1] >= 7:
        speed = np.maximum(0.0, states[:, 6])
    elif states.shape[1] >= 4:
        speed = np.hypot(states[:, 2], states[:, 3])
    else:
        speed = np.zeros((states.shape[0],), dtype=float)
    length = float(np.nanmedian(states[:, 7])) if states.shape[1] >= 8 else 4.8
    width = float(np.nanmedian(states[:, 8])) if states.shape[1] >= 9 else 2.0
    return np.nan_to_num(xy), np.nan_to_num(speed), max(length, 1.0), max(width, 0.5)


def _last_observed_agents(d: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or valid.ndim != 2 or hist.shape[:2] != valid.shape:
        return []
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    for a in range(1, hist.shape[1]):  # index 0 is the SDC
        idx = np.where(valid[:, a])[0]
        if idx.size == 0:
            continue
        s = hist[int(idx[-1]), a]
        if s.size < 5:
            continue
        p = np.asarray(s[:2], dtype=float)
        v = np.asarray([s[3], s[4]], dtype=float)
        acc = np.asarray([s[5], s[6]], dtype=float) if s.size >= 7 else np.zeros(2, dtype=float)
        length = float(s[10]) if s.size >= 12 and np.isfinite(s[10]) and s[10] > 0 else 4.8
        width = float(s[11]) if s.size >= 12 and np.isfinite(s[11]) and s[11] > 0 else 2.0
        out.append((p, v, acc, length, width))
    return out


def _hypotheses() -> tuple[list[tuple[str, float, float, float]], np.ndarray]:
    specs = [
        ("constant_velocity", 1.00, 0.0, 0.0),
        ("yield", 0.65, -1.5, 0.0),
        ("accelerate", 1.20, 1.2, 0.0),
        ("hard_brake", 0.45, -3.5, 0.0),
        ("left_drift", 1.00, 0.0, 0.65),
        ("right_drift", 1.00, 0.0, -0.65),
        ("delay_noise", 1.08, 0.4, 0.25),
    ]
    weights = np.asarray([0.34, 0.14, 0.14, 0.10, 0.10, 0.10, 0.08], dtype=float)
    return specs, weights / weights.sum()


def _observation_conditioned_mode_weights(
    agents: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]],
    specs: list[tuple[str, float, float, float]],
    prior: np.ndarray,
    ego_xy: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Turn the shared mode bank into an observation-conditioned mixture.

    The external papers take multi-modal prediction probabilities (and RACP
    explicitly consumes beliefs) as inputs; OC-RAP does not expose their native
    prediction networks.  We therefore use one *shared*, label-free likelihood
    update for every external baseline.  It only uses current visible kinematics
    and is intentionally simple enough to be auditable.  This is an interface
    adapter, not a claim that the papers used this predictor.
    """
    prior = np.asarray(prior, dtype=float)
    if not agents or prior.size == 0:
        return prior / max(float(prior.sum()), 1e-9)

    p0 = np.stack([x[0] for x in agents], axis=0)
    v0 = np.stack([x[1] for x in agents], axis=0)
    a0 = np.stack([x[2] for x in agents], axis=0)
    speed = np.linalg.norm(v0, axis=-1)
    direction = np.divide(
        v0,
        speed[:, None],
        out=np.tile(np.asarray([[1.0, 0.0]]), (len(agents), 1)),
        where=speed[:, None] > 0.3,
    )
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)
    a_long = np.einsum("ad,ad->a", a0, direction)
    a_lat = np.einsum("ad,ad->a", a0, normal)

    # Nearby agents carry more evidence.  The floor prevents a distant but
    # visible actor from being completely discarded and keeps the posterior
    # stable in sparse scenes.
    ego0 = np.asarray(ego_xy[0] if ego_xy.ndim == 2 and ego_xy.shape[0] else np.zeros(2), dtype=float)
    dist = np.linalg.norm(p0 - ego0[None, :], axis=-1)
    rel = np.exp(-dist / max(_cfg_float(cfg, "risk_belief_distance_scale_m", 25.0), 1.0)) + 0.05
    rel /= max(float(rel.sum()), 1e-9)

    sigma_long = max(_cfg_float(cfg, "risk_belief_longitudinal_sigma_mps2", 2.0), 0.2)
    sigma_lat = max(_cfg_float(cfg, "risk_belief_lateral_sigma_mps2", 1.5), 0.2)
    transition_s = max(_cfg_float(cfg, "risk_belief_intent_transition_s", 1.5), 0.25)
    log_like = np.zeros(len(specs), dtype=float)
    for h, (_, speed_mult, accel_bias, lateral_drift) in enumerate(specs):
        target_long = accel_bias + (speed_mult - 1.0) * speed / transition_s
        target_lat = lateral_drift / transition_s
        err = 0.5 * ((a_long - target_long) / sigma_long) ** 2 + 0.5 * ((a_lat - target_lat) / sigma_lat) ** 2
        log_like[h] = -float(np.sum(rel * np.clip(err, 0.0, 25.0)))

    log_post = np.log(np.clip(prior, 1e-12, None)) + log_like
    log_post -= float(np.max(log_post))
    posterior = np.exp(log_post)
    posterior /= max(float(posterior.sum()), 1e-9)
    # A small prior blend prevents one noisy acceleration estimate from
    # spuriously collapsing the multi-modal set.
    prior_blend = float(np.clip(_cfg_float(cfg, "risk_belief_prior_blend", 0.30), 0.0, 1.0))
    posterior = prior_blend * prior + (1.0 - prior_blend) * posterior
    return posterior / max(float(posterior.sum()), 1e-9)


def _mode_distinguishability(
    actor_xy: np.ndarray,
    *,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise mode divergence curves and persistent distinction times.

    For point-valued interface predictions, two modes are considered
    distinguishable once their predicted environment states differ by the
    configured scene-level distance threshold and stay separated afterwards.
    This is the finite-prediction analogue of the set-disjointness condition in
    Batkovic et al.; it never reads future ground truth.
    """
    xy = np.asarray(actor_xy, dtype=float)
    if xy.ndim != 4:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0, 0), dtype=float)
    H, A, T, _ = xy.shape
    curves = np.zeros((H, H, T), dtype=float)
    steps = np.full((H, H), max(T - 1, 0), dtype=np.int32)
    if H == 0 or T == 0:
        return steps, curves
    for i in range(H):
        steps[i, i] = 0
        for j in range(i + 1, H):
            if A:
                d = np.linalg.norm(xy[i] - xy[j], axis=-1)  # [A,T]
                curve = np.max(d, axis=0)
            else:
                curve = np.zeros((T,), dtype=float)
            curves[i, j] = curves[j, i] = curve
            separated = curve >= float(threshold_m)
            # Earliest t such that all subsequent predictions stay separated.
            persistent = np.logical_and.accumulate(separated[::-1])[::-1]
            idx = np.where(persistent)[0]
            step = int(idx[0]) if idx.size else max(T - 1, 0)
            steps[i, j] = steps[j, i] = step
    return steps, curves


def build_observed_risk_context(d: dict[str, Any], cfg: dict[str, Any], *, horizon: int | None = None) -> ObservedRiskContext:
    """Predict visible actors once for all candidates in one scene-time group."""
    ego_xy, _, _, _ = _ego_candidate(d)
    T = max(int(horizon or ego_xy.shape[0]), 2)
    dt = _cfg_float(cfg, "risk_dt", _cfg_float(cfg, "contact_dt", 0.1))
    times = np.arange(T, dtype=float) * max(dt, 1e-3)
    agents = _last_observed_agents(d)
    specs, prior_weights = _hypotheses()
    # Preserve legacy behavior outside the audited Near-contact configuration.
    # Observation-conditioned probabilities are an explicit shared predictor
    # adapter for MARC/RACP/scenario-MPC in this reproduction, not a global
    # semantic change to unrelated Safe/Contact baselines.
    use_obs_belief = _cfg_bool(cfg, "risk_observation_conditioned_mode_weights", False)
    if use_obs_belief:
        weights = _observation_conditioned_mode_weights(agents, specs, prior_weights, ego_xy, cfg)
        weight_source = "observation_conditioned_kinematic_belief"
    else:
        weights = np.asarray(prior_weights, dtype=float)
        weights /= max(float(weights.sum()), 1e-9)
        weight_source = "fixed_shared_mode_prior"
    H, A = len(specs), len(agents)
    if A == 0:
        return ObservedRiskContext(tuple(x[0] for x in specs), weights, times,
                                   np.zeros((H, 0, T, 2), dtype=float),
                                   np.zeros((H, 0, T, 2), dtype=float),
                                   np.zeros((0,), dtype=float),
                                   _cfg_float(cfg, "risk_clearance_buffer_m", 0.75),
                                   np.full((H, H), max(T - 1, 0), dtype=np.int32),
                                   np.zeros((H, H, T), dtype=float),
                                   weight_source)

    p0 = np.stack([x[0] for x in agents], axis=0)                     # [A,2]
    v0 = np.stack([x[1] for x in agents], axis=0)
    a0 = np.stack([x[2] for x in agents], axis=0)
    dims = np.asarray([[x[3], x[4]] for x in agents], dtype=float)
    radii = 0.5 * np.hypot(np.maximum(dims[:, 0], 0.5), np.maximum(dims[:, 1], 0.3))
    speed = np.linalg.norm(v0, axis=-1)
    direction = np.divide(v0, speed[:, None], out=np.tile(np.asarray([[1.0, 0.0]]), (A, 1)), where=speed[:, None] > 0.3)
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)

    actor_xy = np.zeros((H, A, T, 2), dtype=float)
    actor_v = np.zeros_like(actor_xy)
    t = times[None, :, None]
    for h, (_, speed_mult, accel_bias, lateral_drift) in enumerate(specs):
        v = speed_mult * v0
        a = a0 + accel_bias * direction
        pred = p0[:, None, :] + t * v[:, None, :] + 0.5 * t**2 * a[:, None, :] + t * lateral_drift * normal[:, None, :]
        vel = v[:, None, :] + t * a[:, None, :] + lateral_drift * normal[:, None, :]
        if accel_bias < 0:
            along = np.einsum("atd,ad->at", pred - p0[:, None, :], direction)
            along = np.maximum(along, 0.0)
            pred = p0[:, None, :] + along[..., None] * direction[:, None, :] + t * lateral_drift * normal[:, None, :]
            stopped = along <= 1e-8
            vel = np.where(stopped[..., None], lateral_drift * normal[:, None, :], vel)
        actor_xy[h], actor_v[h] = pred, vel
    distinguish_step, divergence_curves = _mode_distinguishability(
        actor_xy,
        threshold_m=_cfg_float(cfg, "mode_distinguish_threshold_m", _cfg_float(cfg, "branch_divergence_threshold_m", 1.0)),
    )
    return ObservedRiskContext(
        tuple(x[0] for x in specs), weights, times, actor_xy, actor_v, radii,
        _cfg_float(cfg, "risk_clearance_buffer_m", 0.75), distinguish_step,
        divergence_curves, weight_source,
    )


def _ego_velocity(xy: np.ndarray, speed: np.ndarray, dt: float) -> np.ndarray:
    tangent = np.gradient(xy, max(float(dt), 1e-3), axis=0)
    norm = np.linalg.norm(tangent, axis=-1)
    direction = np.divide(tangent, norm[:, None], out=np.tile(np.asarray([[1.0, 0.0]]), (xy.shape[0], 1)), where=norm[:, None] > 1e-6)
    return direction * speed[:, None]


def _score_candidates_with_context(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any], context: ObservedRiskContext
) -> list[ObservedRiskProfile]:
    """Vectorized candidate scoring for one common horizon/context."""
    if not samples:
        return []
    T = int(context.times.size)
    H = len(context.hypothesis_names)
    ego_xy_list: list[np.ndarray] = []
    ego_speed_list: list[np.ndarray] = []
    ego_radius: list[float] = []
    for d in samples:
        xy, speed, length, width = _ego_candidate(d)
        ego_xy_list.append(_resample_xy(xy, T))
        src = np.linspace(0.0, 1.0, max(speed.size, 1))
        ego_speed_list.append(np.interp(np.linspace(0.0, 1.0, T), src, speed if speed.size else np.zeros(1)))
        ego_radius.append(0.5 * float(np.hypot(length, width)))
    ego_xy = np.stack(ego_xy_list, axis=0)                            # [N,T,2]
    ego_speed = np.stack(ego_speed_list, axis=0)                      # [N,T]
    ego_radius_arr = np.asarray(ego_radius, dtype=float)              # [N]
    N = int(ego_xy.shape[0])

    if context.actor_xy.shape[1] == 0:
        out: list[ObservedRiskProfile] = []
        for _ in range(N):
            losses = np.zeros((H,), dtype=float)
            margins = np.full((H,), 50.0, dtype=float)
            curves = np.full((H, T), 50.0, dtype=float)
            out.append(ObservedRiskProfile(
                losses=losses, weights=context.weights, margins=margins,
                min_clearance=50.0, min_ttc=float("inf"), collision_probability=0.0,
                expected_loss=0.0, cvar_loss=0.0, worst_loss=0.0, backup_margin=50.0,
                severity_proxy=0.0, hypothesis_names=context.hypothesis_names,
                collision_probabilities=np.zeros(H), min_ttc_by_mode=np.full(H, np.inf),
                closest_approach_time_by_mode=np.zeros(H), severity_by_mode=np.zeros(H),
                clearance_curves=curves, loss_curves=np.zeros((H, T)), backup_margin_curves=curves.copy(),
                mode_distinguish_step=context.mode_distinguish_step,
                mode_divergence_curves=context.mode_divergence_curves,
                weight_source=context.weight_source,
            ))
        return out

    # [N,H,A,T,2].  This replaces N repeated Python/NumPy candidate passes in
    # the closed-loop hot path while keeping the exact same geometry.
    delta = context.actor_xy[None, ...] - ego_xy[:, None, None, :, :]
    center = np.linalg.norm(delta, axis=-1)
    clearance = (
        center
        - ego_radius_arr[:, None, None, None]
        - context.actor_radius[None, None, :, None]
        - context.clearance_buffer_m
    )
    mode_clearance = np.min(clearance, axis=2)                         # [N,H,T]
    margins = np.min(mode_clearance, axis=2)                           # [N,H]
    closest_idx = np.argmin(mode_clearance, axis=2)                    # [N,H]
    closest_times = context.times[closest_idx]

    ttc_threshold = _cfg_float(cfg, "risk_ttc_clearance_threshold_m", 0.0)
    unsafe = mode_clearance <= ttc_threshold
    has_unsafe = np.any(unsafe, axis=2)
    first_idx = np.argmax(unsafe, axis=2)
    min_ttc_by_mode = np.where(has_unsafe, context.times[first_idx], np.inf)

    collision_temp = max(_cfg_float(cfg, "risk_collision_temperature_m", 0.8), 1e-3)
    loss_scale = max(_cfg_float(cfg, "risk_clearance_scale_m", 2.0), 1e-3)
    severity_speed = max(_cfg_float(cfg, "risk_severity_speed_mps", 12.0), 1e-3)
    collision_curve = 1.0 / (1.0 + np.exp(np.clip(mode_clearance / collision_temp, -40.0, 40.0)))
    proximity_curve = np.exp(-np.maximum(mode_clearance, 0.0) / loss_scale)
    penetration_curve = np.maximum(-mode_clearance, 0.0) / loss_scale

    nearest_actor = np.argmin(clearance, axis=2)                       # [N,H,T]
    dt = context.times[1] - context.times[0] if T > 1 else 0.1
    ego_v = np.stack([_ego_velocity(ego_xy[i], ego_speed[i], dt) for i in range(N)], axis=0)
    actor_v = np.broadcast_to(context.actor_velocity[None, ...], (N,) + context.actor_velocity.shape)
    gather_idx = nearest_actor[:, :, None, :, None]
    nearest_v = np.take_along_axis(actor_v, gather_idx, axis=2).squeeze(axis=2)
    severity_curve = np.clip(np.linalg.norm(ego_v[:, None, :, :] - nearest_v, axis=-1) / severity_speed, 0.0, 2.0)
    loss_curve = collision_curve + 0.35 * proximity_curve + 0.45 * collision_curve * severity_curve + 0.35 * penetration_curve
    loss_curve = np.clip(loss_curve, 0.0, 4.0)

    aggregation = str((((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {}).get("risk_temporal_aggregation", "max"))).lower()
    if aggregation == "mean":
        losses = np.mean(loss_curve, axis=2)
    elif aggregation == "discounted_mean":
        discount = np.exp(-_cfg_float(cfg, "risk_temporal_discount", 0.15) * context.times)
        losses = np.sum(loss_curve * discount[None, None, :], axis=2) / max(float(discount.sum()), 1e-9)
    else:
        losses = np.max(loss_curve, axis=2)

    decel = max(_cfg_float(cfg, "backup_deceleration_mps2", 5.0), 0.5)
    reaction = max(_cfg_float(cfg, "backup_reaction_time_s", 0.25), 0.0)
    stopping = ego_speed * reaction + ego_speed**2 / (2.0 * decel)
    backup_curves = mode_clearance - stopping[:, None, :]
    backup_margin = np.min(backup_curves, axis=(1, 2))
    weights = np.asarray(context.weights, dtype=float)
    expected = np.sum(losses * weights[None, :], axis=1)
    cvar = np.asarray([_weighted_upper_cvar(losses[i], weights, _cfg_float(cfg, "cvar_alpha", 0.2)) for i in range(N)])
    collision_prob_by_mode = np.max(collision_curve, axis=2)
    severity_by_mode = np.take_along_axis(severity_curve, closest_idx[..., None], axis=2).squeeze(axis=2)

    out: list[ObservedRiskProfile] = []
    for i in range(N):
        finite_ttc = min_ttc_by_mode[i][np.isfinite(min_ttc_by_mode[i])]
        out.append(ObservedRiskProfile(
            losses=np.asarray(losses[i], dtype=float), weights=weights, margins=np.asarray(margins[i], dtype=float),
            min_clearance=float(np.min(margins[i])), min_ttc=float(np.min(finite_ttc)) if finite_ttc.size else float("inf"),
            collision_probability=float(np.sum(weights * collision_prob_by_mode[i])), expected_loss=float(expected[i]),
            cvar_loss=float(cvar[i]), worst_loss=float(np.max(losses[i])), backup_margin=float(backup_margin[i]),
            severity_proxy=float(np.sum(weights * collision_prob_by_mode[i] * severity_by_mode[i])),
            hypothesis_names=context.hypothesis_names, collision_probabilities=np.asarray(collision_prob_by_mode[i], dtype=float),
            min_ttc_by_mode=np.asarray(min_ttc_by_mode[i], dtype=float), closest_approach_time_by_mode=np.asarray(closest_times[i], dtype=float),
            severity_by_mode=np.asarray(severity_by_mode[i], dtype=float), clearance_curves=np.asarray(mode_clearance[i], dtype=float),
            loss_curves=np.asarray(loss_curve[i], dtype=float), backup_margin_curves=np.asarray(backup_curves[i], dtype=float),
            mode_distinguish_step=context.mode_distinguish_step,
            mode_divergence_curves=context.mode_divergence_curves,
            weight_source=context.weight_source,
        ))
    return out


def score_candidate_with_context(d: dict[str, Any], cfg: dict[str, Any], context: ObservedRiskContext) -> ObservedRiskProfile:
    return _score_candidates_with_context([d], cfg, context)[0]


def observed_risk_profiles(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> list[ObservedRiskProfile]:
    """Score a candidate group while reusing actor forecasts for equal horizons."""
    if not samples:
        return []
    groups: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for i, d in enumerate(samples):
        xy, _, _, _ = _ego_candidate(d)
        T = max(int(xy.shape[0]), 2)
        groups.setdefault(T, []).append((i, d))
    out: list[ObservedRiskProfile | None] = [None] * len(samples)
    for T, items in groups.items():
        context = build_observed_risk_context(samples[0], cfg, horizon=T)
        batch = _score_candidates_with_context([d for _, d in items], cfg, context)
        for (i, _), profile in zip(items, batch):
            out[i] = profile
    return [p for p in out if p is not None]


def observed_risk_profile(d: dict[str, Any], cfg: dict[str, Any]) -> ObservedRiskProfile:
    context = build_observed_risk_context(d, cfg)
    return score_candidate_with_context(d, cfg, context)


def observed_risk_profiles_and_context(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[ObservedRiskProfile], ObservedRiskContext | None]:
    """Hot-path bundle: build the shared actor forecast once and return it.

    External paper-core ports such as DR-CVaR/CPSF need the actual predicted
    obstacle positions in addition to scalar risk profiles.  Returning the
    context avoids a second identical forecast construction per replan.
    """
    if not samples:
        return [], None
    horizons = []
    for d in samples:
        xy, _, _, _ = _ego_candidate(d)
        horizons.append(max(int(xy.shape[0]), 2))
    if len(set(horizons)) == 1:
        context = build_observed_risk_context(samples[0], cfg, horizon=horizons[0])
        return _score_candidates_with_context(samples, cfg, context), context
    # Preserve legacy mixed-horizon scoring semantics.  Ports can still use a
    # max-horizon context and resample it to their source horizon.
    profiles = observed_risk_profiles(samples, cfg)
    context = build_observed_risk_context(samples[0], cfg, horizon=max(horizons))
    return profiles, context
