from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observed_risk import (
    ObservedRiskContext, ObservedRiskProfile, build_observed_risk_context,
    observed_risk_profile, observed_risk_profiles, observed_risk_profiles_and_context,
)
from .paper_core_ports_v56 import (
    cpsf_constrained_projection_port, dr_cvar_safe_halfspace_port,
    integrated_postimpact_mpc_pso_port, postimpact_motion_tvlqr_port,
)
from .paper_core_ports_v57 import (
    compensatory_postimpact_mpc_port, post_collision_restoration_port,
    post_crash_braking_port, robust_postimpact_control_port,
)
from .paper_core_ports_v58 import severity_minimization_port


@dataclass
class ExternalSelection:
    selected_index: int
    reason: str
    admitted: np.ndarray
    score: np.ndarray
    selected_option: int | None = None


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return float(default)


def _betop_short_term_repulsive_cost(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> np.ndarray:
    """Finite-candidate analogue of BeTop Appendix-C short-term cost ``C_M``.

    BeTop planning inference combines trajectory confidence with a short-term
    repulsive-potential cost.  The released repository does not contain the
    nuPlan planning pipeline, and OC-RAP samples do not contain BeTop's learned
    multi-agent marginal futures.  We therefore preserve the published
    potential ``1 / (1 + d_min)`` and branching step ``t_b=3`` using only the
    current observed actors with constant-velocity extrapolation.  This helper
    intentionally uses no teacher/future labels.
    """
    n = len(samples)
    if n == 0:
        return np.zeros((0,), dtype=float)
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    pcfg = bcfg.get("policy", {}) if isinstance(bcfg.get("policy", {}), dict) else {}
    tb = max(1, int(pcfg.get("betop_branch_steps", 3)))
    dt = max(float(pcfg.get("betop_contingency_dt", 0.1)), 1.0e-3)

    d0 = samples[0]
    hist = np.asarray(d0.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(d0.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    actors: list[tuple[np.ndarray, np.ndarray]] = []
    if hist.ndim == 3 and hist.shape[1] > 1 and hist.shape[2] >= 2:
        H, A, _ = hist.shape
        if valid.ndim != 2 or valid.shape[:2] != (H, A):
            valid = np.isfinite(hist[..., :2]).all(axis=-1)
        for a in range(1, A):
            ids = np.flatnonzero(valid[:, a] & np.isfinite(hist[:, a, :2]).all(axis=-1))
            if ids.size == 0:
                continue
            st = hist[int(ids[-1]), a]
            p0 = np.asarray(st[:2], dtype=float)
            vel = np.asarray([st[3] if st.size > 3 else 0.0, st[4] if st.size > 4 else 0.0], dtype=float)
            if np.isfinite(p0).all() and np.isfinite(vel).all():
                actors.append((p0, vel))
    if not actors:
        return np.zeros((n,), dtype=float)

    ts = np.arange(tb, dtype=float)[:, None] * dt
    actor_xy = np.stack([p[None, :] + ts * v[None, :] for p, v in actors], axis=0)  # [A,tb,2]
    cost = np.zeros((n,), dtype=float)
    for i, d in enumerate(samples):
        st = np.asarray(d.get("prefix_states", np.zeros((0, 2))), dtype=float)
        if st.ndim != 2 or st.shape[0] == 0 or st.shape[1] < 2:
            cost[i] = 1.0
            continue
        xy = st[: min(tb, st.shape[0]), :2]
        if xy.shape[0] < tb:
            xy = np.concatenate([xy, np.repeat(xy[-1:], tb - xy.shape[0], axis=0)], axis=0)
        dist = np.linalg.norm(actor_xy - xy[None, :tb, :], axis=-1)
        dmin = float(np.nanmin(dist)) if dist.size else np.inf
        cost[i] = 0.0 if not np.isfinite(dmin) else 1.0 / (1.0 + max(dmin, 0.0))
    return cost


def _best(score: np.ndarray, mask: np.ndarray | None = None) -> int:
    score = np.asarray(score, dtype=float)
    if score.size == 0:
        return 0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.any():
            idxs = np.where(m)[0]
            return int(idxs[np.argmax(score[idxs])])
    return int(np.argmax(score))


def _admission_select(
    score: np.ndarray,
    admitted: np.ndarray,
    feasible: np.ndarray,
    *,
    fallback_score: np.ndarray,
    reason: str,
    prefer_nominal_if_admitted: bool = False,
) -> tuple[int, str]:
    """Select within the certified set, otherwise use an explicit safest fallback.

    Safety-filter and risk-constrained baselines must not silently turn into a
    utility maximizer when the admissible set is empty.  The original v50
    implementation did exactly that by evaluating ``score`` on all feasible
    candidates.  That makes a failed safety filter look successful while it can
    choose the highest-utility unsafe candidate.  We keep ``admitted`` false and
    choose a method-specific least-risk/maximum-backup candidate instead, with a
    distinct reason so closed-loop logs expose the infeasible-filter event.
    """
    admitted = np.asarray(admitted, dtype=bool)
    feasible = np.asarray(feasible, dtype=bool)
    if admitted.any():
        if prefer_nominal_if_admitted and admitted.size and bool(admitted[0]):
            return 0, reason
        return _best(score, admitted), reason
    idx = _best(np.asarray(fallback_score, dtype=float), feasible)
    return idx, f"{reason}_empty_admissible_set_safest_fallback"


def _valid_root_weights(d: dict[str, Any], K: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(d.get("root_probs", np.ones((K,), dtype=np.float32) / max(K, 1)), dtype=float).reshape(-1)[:K]
    if p.size < K:
        p = np.pad(p, (0, K - p.size))
    valid = np.asarray(d.get("root_valid", np.ones((K,), dtype=bool)), dtype=bool).reshape(-1)[:K]
    if valid.size < K:
        valid = np.pad(valid, (0, K - valid.size), constant_values=False)
    p = np.where(valid, np.clip(p, 0.0, None), 0.0)
    den = float(p.sum())
    return (p / den if den > 1e-8 else np.zeros(K, dtype=float)), valid


def _option_valid(d: dict[str, Any], L: int) -> np.ndarray:
    v = np.asarray(d.get("option_valid", np.ones((L,), dtype=bool)), dtype=bool).reshape(-1)[:L]
    if v.size < L:
        v = np.pad(v, (0, L - v.size), constant_values=False)
    return v


def _weighted_lower_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return 0.0
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    acc = 0.0
    total = 0.0
    for v, w in zip(values, weights):
        take = min(float(w), alpha - total)
        if take <= 0:
            break
        acc += float(v) * take
        total += take
    return float(acc / max(total, 1e-8))



def _weighted_upper_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Weighted CVaR of the upper tail of a nonnegative loss."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return 0.0
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)[::-1]
    values, weights = values[order], weights[order]
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    acc = 0.0
    total = 0.0
    for v, w in zip(values, weights):
        take = min(float(w), alpha - total)
        if take <= 0:
            break
        acc += float(v) * take
        total += take
    return float(acc / max(total, 1e-8))


def _effective_root_outcomes(d: dict[str, Any], alpha: float = 0.2, gamma: float = 0.0) -> dict[str, Any]:
    """Branch-wise existential margins and risk-loss samples.

    For a latent root z_k, branch-wise recovery is existential in the option
    dimension: the branch succeeds if any option g_l has margin m_{k,l} >= gamma.
    This is exactly the oracle order in the OC-RAP paper: max over options first,
    then aggregate over latent roots.
    """
    base = _branchwise_values(d, alpha=alpha)
    best = np.asarray(base.get("best_margins", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
    K = int(best.size)
    w, valid = _valid_root_weights(d, K)
    if K == 0:
        return {**base, "losses": np.zeros((0,), dtype=float), "risk_expected": 1.0, "risk_cvar": 1.0, "risk_worst": 1.0, "oracle_all_roots": False, "oracle_mass": 0.0}
    clipped = np.clip(best, -5.0, 5.0)
    losses = np.where(valid, np.maximum(0.0, float(gamma) - clipped), 5.0)
    risk_expected = float(np.sum(w * losses)) if w.size else float(np.mean(losses))
    risk_cvar = _weighted_upper_cvar(losses, w if w.size else np.ones_like(losses) / max(len(losses), 1), alpha=float(alpha))
    risk_worst = float(np.max(losses[valid])) if valid.any() else float(np.max(losses))
    oracle_ok = valid & (clipped >= float(gamma))
    all_roots = bool(valid.any() and np.all(oracle_ok[valid]))
    mass = float(np.sum(w * oracle_ok.astype(float))) if w.size else 0.0
    return {**base, "losses": losses, "risk_expected": risk_expected, "risk_cvar": risk_cvar, "risk_worst": risk_worst, "oracle_all_roots": all_roots, "oracle_mass": mass}


def _prefix_common_horizon(candidate: dict[str, Any], reference: dict[str, Any] | None, *, threshold: float = 1.0, max_fraction: float = 0.6) -> float:
    """Legacy ego-vs-reference prefix similarity helper.

    This is retained for compatibility with old diagnostic code.  It must not
    be used as the MARC scenario branch point: MARC branches from divergence
    among policy-conditioned *scenario futures*, not from candidate-vs-nominal
    ego deviation.  The main MARC adapter below uses `_marc_branch_step`.
    """
    if reference is None:
        return 0.0
    a = np.asarray(candidate.get("prefix_states", np.zeros((0, 0))), dtype=float)
    b = np.asarray(reference.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0 or a.shape[1] < 2 or b.shape[1] < 2:
        return 0.0
    T = min(a.shape[0], b.shape[0])
    if T <= 1:
        return 0.0
    dist = np.linalg.norm(a[:T, :2] - b[:T, :2], axis=-1)
    ok = np.where(dist <= float(threshold))[0]
    if ok.size == 0:
        return 0.0
    latest = int(ok[-1])
    cap = int(max(1, round(float(max_fraction) * (T - 1))))
    return float(min(latest, cap) / max(T - 1, 1))


def _candidate_xy(d: dict[str, Any], count: int | None = None) -> np.ndarray:
    s = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] < 2:
        xy = np.zeros((max(int(count or 2), 2), 2), dtype=float)
    else:
        xy = np.nan_to_num(s[:, :2])
    if count is None or xy.shape[0] == int(count):
        return xy
    src = np.linspace(0.0, 1.0, xy.shape[0])
    dst = np.linspace(0.0, 1.0, int(count))
    return np.stack([np.interp(dst, src, xy[:, 0]), np.interp(dst, src, xy[:, 1])], axis=-1)


def _candidate_controls(d: dict[str, Any], count: int | None = None) -> np.ndarray:
    u = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    if u.ndim != 2 or u.shape[0] == 0:
        return np.zeros((max(int(count or 2), 2), 2), dtype=float)
    if u.shape[1] < 2:
        u = np.pad(u, ((0, 0), (0, 2 - u.shape[1])))
    u = np.nan_to_num(u[:, :2])
    if count is None or u.shape[0] == int(count):
        return u
    src = np.linspace(0.0, 1.0, u.shape[0])
    dst = np.linspace(0.0, 1.0, int(count))
    return np.stack([np.interp(dst, src, u[:, 0]), np.interp(dst, src, u[:, 1])], axis=-1)


def _control_sequence_deviation(samples: list[dict[str, Any]], nominal_index: int, *, count: int | None = None) -> np.ndarray:
    """RMS input distance to the proposed/nominal controller command sequence."""
    if not samples:
        return np.zeros((0,), dtype=float)
    ref = _candidate_controls(samples[int(nominal_index)], count=count)
    out = np.zeros(len(samples), dtype=float)
    for i, d in enumerate(samples):
        u = _candidate_controls(d, count=ref.shape[0])
        # Normalize acceleration and steering to comparable units.  This is
        # only a metric for the safety-filter minimal-intervention objective.
        delta = (u - ref) / np.asarray([4.0, 0.6])[None, :]
        out[i] = float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))
    return out


def _prefix_compatibility_matrix(
    samples: list[dict[str, Any]],
    branch_step: int,
    *,
    state_threshold_m: float,
    accel_tolerance_mps2: float | None = None,
    steer_tolerance_rad: float | None = None,
) -> np.ndarray:
    """Finite-lattice non-anticipativity relation up to `branch_step`.

    State closeness is always required.  If control tolerances are supplied,
    inputs are also required to match within tolerance, reflecting the explicit
    input-equality constraint in robust scenario MPC.
    """
    n = len(samples)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)
    T = max(int(branch_step) + 1, 1)

    def _native_prefix_xy(d: dict[str, Any]) -> np.ndarray:
        xy_i = _candidate_xy(d)
        # Non-anticipativity is a *prefix* constraint.  Resampling an entire
        # trajectory to T points leaks post-branch tail differences into the
        # common prefix and can incorrectly reject valid contingent plans.
        return xy_i[:T] if xy_i.shape[0] >= T else _candidate_xy(d, count=T)

    def _native_prefix_controls(d: dict[str, Any]) -> np.ndarray:
        u_i = _candidate_controls(d)
        return u_i[:T] if u_i.shape[0] >= T else _candidate_controls(d, count=T)

    xy = np.stack([_native_prefix_xy(d) for d in samples], axis=0)
    dxy = np.linalg.norm(xy[:, None, :, :] - xy[None, :, :, :], axis=-1)
    compatible = np.max(dxy, axis=-1) <= float(state_threshold_m)
    if accel_tolerance_mps2 is not None or steer_tolerance_rad is not None:
        u = np.stack([_native_prefix_controls(d) for d in samples], axis=0)
        du = np.abs(u[:, None, :, :] - u[None, :, :, :])
        if accel_tolerance_mps2 is not None:
            compatible &= np.max(du[..., 0], axis=-1) <= float(accel_tolerance_mps2)
        if steer_tolerance_rad is not None:
            compatible &= np.max(du[..., 1], axis=-1) <= float(steer_tolerance_rad)
    np.fill_diagonal(compatible, True)
    return compatible


def _latest_divergence_branch_step(
    samples: list[dict[str, Any]],
    candidate_ids: list[int],
    *,
    horizon: int,
    threshold_m: float,
    max_fraction: float,
) -> int:
    """MARC scene-level branch time from scenario-conditioned ego futures."""
    if horizon <= 1 or len(candidate_ids) <= 1:
        return 0
    xy = np.stack([_candidate_xy(samples[i], count=horizon) for i in candidate_ids], axis=0)
    pair = np.linalg.norm(xy[:, None, :, :] - xy[None, :, :, :], axis=-1)
    divergence = np.max(pair, axis=(0, 1))
    cap = int(np.clip(round(float(max_fraction) * (horizon - 1)), 0, horizon - 1))
    ok = np.where(divergence[: cap + 1] < float(threshold_m))[0]
    return int(ok[-1]) if ok.size else 0


def _mode_tail_loss(profile: ObservedRiskProfile, mode: int, branch_step: int) -> float:
    curves = np.asarray(profile.loss_curves, dtype=float)
    if curves.ndim != 2 or mode >= curves.shape[0] or curves.shape[1] == 0:
        losses = np.asarray(profile.losses, dtype=float).reshape(-1)
        return float(losses[min(mode, max(losses.size - 1, 0))]) if losses.size else 0.0
    b = int(np.clip(branch_step, 0, curves.shape[1] - 1))
    return float(np.max(curves[mode, b:]))


def _mode_tail_collision(profile: ObservedRiskProfile, mode: int, branch_step: int, clearance_threshold_m: float) -> float:
    c = np.asarray(profile.clearance_curves, dtype=float)
    if c.ndim != 2 or mode >= c.shape[0] or c.shape[1] == 0:
        p = np.asarray(profile.collision_probabilities, dtype=float).reshape(-1)
        return float(p[min(mode, max(p.size - 1, 0))]) if p.size else float(profile.collision_probability)
    b = int(np.clip(branch_step, 0, c.shape[1] - 1))
    return float(np.any(c[mode, b:] <= float(clearance_threshold_m)))


def _robust_scenario_mpc_candidate_scores(
    samples: list[dict[str, Any]],
    profiles: list[ObservedRiskProfile],
    feasible: np.ndarray,
    utility: np.ndarray,
    smooth: np.ndarray,
    dev: np.ndarray,
    pcfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-lattice projection of Batkovic et al. problem (7).

    Each disturbance mode receives its own candidate continuation, while the
    input sequences of every mode pair are tied until that *pair* becomes
    distinguishable.  This mirrors Eq. (7f) more closely than collapsing all
    modes to one global branch time.  A bounded beam search solves the small
    multiple-choice candidate-tree problem without introducing a continuous
    optimizer into the common OC-RAP execution interface.
    """
    n = len(samples)
    score = np.full(n, -1.0e9, dtype=float)
    admitted = np.zeros(n, dtype=bool)
    branch_steps = np.zeros(n, dtype=int)
    if not profiles:
        return score, admitted, branch_steps
    c0 = np.asarray(profiles[0].clearance_curves, dtype=float)
    H, T = c0.shape if c0.ndim == 2 else (0, 0)
    if H == 0 or T == 0:
        return score, admitted, branch_steps

    weights = np.asarray(profiles[0].weights, dtype=float).reshape(-1)[:H]
    weights = weights / max(float(weights.sum()), 1e-9)
    distinguish = np.asarray(profiles[0].mode_distinguish_step, dtype=int)
    if distinguish.shape != (H, H):
        distinguish = np.zeros((H, H), dtype=int)
    cap_fraction = float(np.clip(pcfg.get("scenario_mpc_max_common_fraction", 1.0), 0.0, 1.0))
    cap_step = int(round(cap_fraction * max(T - 1, 0)))
    distinguish = np.minimum(np.maximum(distinguish, 0), cap_step)
    offdiag = distinguish[~np.eye(H, dtype=bool)] if H > 1 else np.zeros((0,), dtype=int)
    branch_steps[:] = int(np.max(offdiag)) if offdiag.size else 0

    state_tol = float(pcfg.get("scenario_mpc_state_tie_threshold_m", pcfg.get("branch_divergence_threshold_m", 1.0)))
    accel_tol = float(pcfg.get("scenario_mpc_control_tie_accel_tol_mps2", 0.75))
    steer_tol = float(pcfg.get("scenario_mpc_control_tie_steer_tol_rad", 0.15))

    # Eq. (7f) requires equal inputs for n < nbar_ij.  Cache a candidate
    # compatibility matrix for every distinct last-tied prefix index.  A value
    # of -1 means that the pair is already distinguishable and imposes no tie.
    tie_steps = sorted({int(distinguish[i, j]) - 1 for i in range(H) for j in range(i + 1, H) if int(distinguish[i, j]) > 0})
    compat_by_step: dict[int, np.ndarray] = {}
    for tie_step in tie_steps:
        compat_by_step[tie_step] = _prefix_compatibility_matrix(
            samples,
            tie_step,
            state_threshold_m=state_tol,
            accel_tolerance_mps2=accel_tol,
            steer_tolerance_rad=steer_tol,
        )

    clearance_gate = float(pcfg.get("scenario_mpc_min_clearance_gate_m", -0.50))
    # Batkovic et al. enforce physical/state/input constraints robustly across
    # modes while mode probabilities enter the expected objective.  Do not turn
    # our internal risk surrogate into an additional unpublished hard constraint.
    # A finite guard is still accepted when an old experiment config explicitly
    # requests it, but the paper-core default is unconstrained in this surrogate.
    max_mode_loss = float(
        pcfg.get(
            "scenario_mpc_max_mode_stage_loss_guard",
            pcfg.get("scenario_mpc_worst_risk_gate", float("inf")),
        )
    )
    uw = float(pcfg.get("scenario_mpc_utility_weight", 1.0))
    rw = float(pcfg.get("scenario_mpc_expected_weight", 1.5))
    ww = float(pcfg.get("scenario_mpc_worst_weight", 0.0))
    sw = float(pcfg.get("scenario_mpc_smoothness_weight", 0.12))
    dw = float(pcfg.get("scenario_mpc_deviation_weight", 0.05))
    beam_size = max(int(pcfg.get("scenario_mpc_pairwise_beam_size", 128)), 1)

    # Precompute mode-specific robust feasibility and costs for every executable
    # candidate.  Probabilities affect only the expected objective, not safety.
    mode_loss = np.full((n, H), np.inf, dtype=float)
    mode_value = np.full((n, H), -1.0e9, dtype=float)
    mode_safe = np.zeros((n, H), dtype=bool)
    for j in range(n):
        if not bool(feasible[j]):
            continue
        clr = np.asarray(profiles[j].clearance_curves, dtype=float)
        if clr.shape != (H, T):
            continue
        for h in range(H):
            loss_h = _mode_tail_loss(profiles[j], h, 0)
            safe_h = bool(np.all(clr[h] >= clearance_gate) and loss_h <= max_mode_loss)
            if not safe_h:
                continue
            mode_safe[j, h] = True
            mode_loss[j, h] = loss_h
            mode_value[j, h] = (
                uw * float(utility[j])
                - rw * loss_h
                - sw * float(smooth[j])
                - dw * float(dev[j])
            )

    options_by_mode = [np.where(mode_safe[:, h])[0].tolist() for h in range(H)]
    if any(len(x) == 0 for x in options_by_mode):
        return score, admitted, branch_steps

    def _pair_compatible(mode_a: int, cand_a: int, mode_b: int, cand_b: int) -> bool:
        nbar = int(distinguish[mode_a, mode_b])
        if nbar <= 0:
            return True
        tie_step = nbar - 1
        mat = compat_by_step.get(tie_step)
        return bool(mat[cand_a, cand_b]) if mat is not None else True

    # The returned OC-RAP candidate represents the single executable prefix
    # that all modes must share before *any* pair can be distinguished.  After
    # the first split, the beam tuple itself carries the mode/group-specific
    # continuations.  This avoids arbitrarily privileging one disturbance mode
    # as the representative full-horizon trajectory.
    positive_offdiag = offdiag[offdiag > 0] if offdiag.size else np.zeros((0,), dtype=int)
    common_tie_step = int(np.min(positive_offdiag) - 1) if positive_offdiag.size else -1
    root_compat = (
        _prefix_compatibility_matrix(
            samples, common_tie_step,
            state_threshold_m=state_tol,
            accel_tolerance_mps2=accel_tol,
            steer_tolerance_rad=steer_tol,
        )
        if common_tie_step >= 0 else np.ones((n, n), dtype=bool)
    )

    for i in range(n):
        if not bool(feasible[i]):
            continue
        root_clear = np.asarray(profiles[i].clearance_curves, dtype=float)
        if root_clear.shape != (H, T):
            continue
        if common_tie_step >= 0 and np.any(root_clear[:, : common_tie_step + 1] < clearance_gate):
            continue
        # beam item: (weighted objective so far, tuple(candidate id per mode),
        #             worst mode loss so far)
        beam: list[tuple[float, tuple[int, ...], float]] = [(0.0, tuple(), 0.0)]
        for h in range(H):
            expanded: list[tuple[float, tuple[int, ...], float]] = []
            for partial_score, assigned, worst_so_far in beam:
                for j in options_by_mode[h]:
                    if not bool(root_compat[i, j]):
                        continue
                    ok = True
                    for prev_h, prev_j in enumerate(assigned):
                        if not _pair_compatible(h, int(j), prev_h, int(prev_j)):
                            ok = False
                            break
                    if not ok:
                        continue
                    expanded.append((
                        partial_score + float(weights[h] * mode_value[j, h]),
                        assigned + (int(j),),
                        max(worst_so_far, float(mode_loss[j, h])),
                    ))
            if not expanded:
                beam = []
                break
            expanded.sort(key=lambda x: x[0] - ww * x[2], reverse=True)
            beam = expanded[:beam_size]
        if not beam:
            continue
        best = max(beam, key=lambda x: x[0] - ww * x[2])
        admitted[i] = True
        score[i] = float(best[0] - ww * best[2])
    return score, admitted, branch_steps

def _marc_candidate_scores(
    samples: list[dict[str, Any]],
    profiles: list[ObservedRiskProfile],
    feasible: np.ndarray,
    utility: np.ndarray,
    smooth: np.ndarray,
    dev: np.ndarray,
    macros: list[str],
    pcfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MARC policy-conditioned scenario tree + CVaR on a finite lattice."""
    n = len(samples)
    score = np.full(n, -1.0e9, dtype=float)
    admitted = np.zeros(n, dtype=bool)
    risks = np.full(n, np.inf, dtype=float)
    chances = np.ones(n, dtype=float)
    if not profiles:
        return score, admitted, risks, chances
    curves0 = np.asarray(profiles[0].loss_curves, dtype=float)
    H, T = curves0.shape if curves0.ndim == 2 else (0, 0)
    if H == 0 or T == 0:
        return score, admitted, risks, chances
    weights = np.asarray(profiles[0].weights, dtype=float).reshape(-1)[:H]
    weights = weights / max(float(weights.sum()), 1e-9)
    rho = float(np.clip(pcfg.get("marc_risk_tolerance", 0.35), 0.0, 0.999))
    # MARC's alpha is a confidence level: alpha->0 approaches expectation and
    # alpha->1 becomes more risk-averse.  Our CVaR helper accepts tail mass.
    tail_mass = max(1.0 - rho, 1e-3)
    threshold = float(pcfg.get("branch_divergence_threshold_m", 1.0))
    max_fraction = float(np.clip(pcfg.get("max_branch_fraction", 0.6), 0.0, 1.0))
    chance_clearance = float(pcfg.get("risk_ttc_clearance_threshold_m", 0.0))
    risk_threshold = float(pcfg.get("marc_risk_threshold", 1.0))
    chance_threshold = float(pcfg.get("marc_chance_threshold", 1.0))
    uw = float(pcfg.get("marc_utility_weight", 1.0))
    risk_w = float(pcfg.get("marc_safety_risk_weight", pcfg.get("marc_expected_risk_weight", 2.0)))
    chance_w = float(pcfg.get("marc_collision_probability_weight", 1.0))
    sw = float(pcfg.get("marc_smoothness_weight", 0.15))
    dw = float(pcfg.get("marc_deviation_weight", 0.10))

    loss_curves = np.stack([np.asarray(p.loss_curves, dtype=float) for p in profiles], axis=0)
    clearance_curves = np.stack([np.asarray(p.clearance_curves, dtype=float) for p in profiles], axis=0)
    if loss_curves.shape != (n, H, T) or clearance_curves.shape != (n, H, T):
        return score, admitted, risks, chances
    full_loss = np.max(loss_curves, axis=2)  # [candidate, mode]
    base_value = (
        uw * utility[:, None]
        - risk_w * full_loss
        - sw * smooth[:, None]
        - dw * dev[:, None]
    )

    macro_arr = np.asarray(macros, dtype=object)
    for macro in sorted(set(macros)):
        family_idx = np.where((macro_arr == macro) & feasible)[0]
        if family_idx.size == 0:
            continue
        # Policy-conditioned critical scenarios: best executable response in
        # this semantic family for each predicted mode.
        family_base = base_value[family_idx]  # [F,H]
        scenario_ids = [int(family_idx[int(np.argmax(family_base[:, h]))]) for h in range(H)]
        branch = _latest_divergence_branch_step(
            samples, scenario_ids, horizon=T, threshold_m=threshold, max_fraction=max_fraction
        )
        compat = _prefix_compatibility_matrix(samples, branch, state_threshold_m=threshold)
        F = int(family_idx.size)
        family_compat = compat[np.ix_(family_idx, family_idx)]  # [root,cont]

        tail_loss = np.max(loss_curves[family_idx, :, branch:], axis=2)  # [cont,H]
        tail_collision = np.any(clearance_curves[family_idx, :, branch:] <= chance_clearance, axis=2).astype(float)
        tail_value = (
            uw * utility[family_idx, None]
            - risk_w * tail_loss
            - sw * smooth[family_idx, None]
            - dw * dev[family_idx, None]
        )
        # For every possible shared root and mode, choose the best compatible
        # continuation in one broadcasted reduction.
        masked = np.where(family_compat[:, :, None], tail_value[None, :, :], -np.inf)
        best_local = np.argmax(masked, axis=1)  # [root,H], indices within family
        best_value = np.max(masked, axis=1)
        mode_idx = np.broadcast_to(np.arange(H, dtype=int)[None, :], best_local.shape)
        selected_tail_loss = tail_loss[best_local, mode_idx]
        selected_tail_collision = tail_collision[best_local, mode_idx]
        selected_global = family_idx[best_local]

        prefix_loss = np.max(loss_curves[family_idx, :, : branch + 1], axis=2)
        prefix_collision = np.any(clearance_curves[family_idx, :, : branch + 1] <= chance_clearance, axis=2).astype(float)
        mode_safety = np.maximum(prefix_loss, selected_tail_loss)
        mode_collision = np.maximum(prefix_collision, selected_tail_collision)
        mode_utility = utility[selected_global]
        mode_smooth = smooth[selected_global]
        mode_dev = dev[selected_global]

        cvar = np.asarray([_weighted_upper_cvar(mode_safety[r], weights, tail_mass) for r in range(F)], dtype=float)
        chance = np.sum(mode_collision * weights[None, :], axis=1)
        family_score = (
            uw * np.sum(mode_utility * weights[None, :], axis=1)
            - risk_w * cvar
            - chance_w * chance
            - sw * np.sum(mode_smooth * weights[None, :], axis=1)
            - dw * np.sum(mode_dev * weights[None, :], axis=1)
        )
        valid = np.all(np.isfinite(best_value), axis=1)
        gi = family_idx[valid]
        risks[gi] = cvar[valid]
        chances[gi] = chance[valid]
        score[gi] = family_score[valid]
        admitted[gi] = (cvar[valid] <= risk_threshold) & (chance[valid] <= chance_threshold)
    return score, admitted, risks, chances

def _racp_candidate_scores(
    samples: list[dict[str, Any]],
    profiles: list[ObservedRiskProfile],
    feasible: np.ndarray,
    utility: np.ndarray,
    smooth: np.ndarray,
    dev: np.ndarray,
    pcfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source-structured RACP shared-plan + belief-weighted contingent tails.

    The uploaded source computes a shared Frenet plan, then adds branch-weighted
    contingent-plan costs.  We preserve that cost topology while replacing the
    continuous Frenet/MPC solve with executable candidate enumeration.  The
    candidate recourse solve below is vectorized across roots and modes so the
    source-structured search does not dominate closed-loop runtime.
    """
    n = len(samples)
    score = np.full(n, -1.0e9, dtype=float)
    admitted = np.zeros(n, dtype=bool)
    risk_out = np.full(n, np.inf, dtype=float)
    chance_out = np.ones(n, dtype=float)
    if not profiles:
        return score, admitted, risk_out, chance_out
    curves0 = np.asarray(profiles[0].loss_curves, dtype=float)
    H, T = curves0.shape if curves0.ndim == 2 else (0, 0)
    if H == 0 or T == 0:
        return score, admitted, risk_out, chance_out
    weights = np.asarray(profiles[0].weights, dtype=float).reshape(-1)[:H]
    weights = weights / max(float(weights.sum()), 1e-9)
    branch_fraction = float(np.clip(pcfg.get("racp_branch_fraction", 0.40), 0.0, 1.0))
    branch = int(round(branch_fraction * max(T - 1, 0)))
    compat = _prefix_compatibility_matrix(
        samples, branch,
        state_threshold_m=float(pcfg.get("racp_nonanticipative_state_threshold_m", pcfg.get("branch_divergence_threshold_m", 1.0))),
    )
    chance_clearance = float(pcfg.get("risk_ttc_clearance_threshold_m", 0.0))
    risk_threshold = float(pcfg.get("racp_risk_threshold", pcfg.get("racp_delta", 0.75)))
    chance_threshold = float(pcfg.get("racp_chance_threshold", 1.0))
    uw = float(pcfg.get("racp_utility_weight", 1.0))
    rw = float(pcfg.get("racp_risk_weight", 2.5))
    cw = float(pcfg.get("racp_collision_probability_weight", 1.0))
    sw = float(pcfg.get("racp_smoothness_weight", 0.10))
    dw = float(pcfg.get("racp_deviation_weight", 0.05))
    backup_w = float(pcfg.get("racp_backup_margin_weight", 0.06))

    loss_curves = np.stack([np.asarray(p.loss_curves, dtype=float) for p in profiles], axis=0)
    clearance_curves = np.stack([np.asarray(p.clearance_curves, dtype=float) for p in profiles], axis=0)
    backup_curves = np.stack([np.asarray(p.backup_margin_curves, dtype=float) for p in profiles], axis=0)
    if loss_curves.shape != (n, H, T) or clearance_curves.shape != (n, H, T) or backup_curves.shape != (n, H, T):
        return score, admitted, risk_out, chance_out

    # [candidate, mode] precomputation for all contingent continuations.
    tail_loss = np.max(loss_curves[:, :, branch:], axis=2)
    tail_backup = np.min(backup_curves[:, :, branch:], axis=2)
    tail_chance = np.any(clearance_curves[:, :, branch:] <= chance_clearance, axis=2).astype(float)
    tail_value = (
        uw * utility[:, None]
        + backup_w * np.clip(tail_backup, -20.0, 20.0)
        - rw * tail_loss
        - sw * smooth[:, None]
        - dw * dev[:, None]
    )

    # Root i may choose contingent candidate j only if j shares the RACP common
    # prefix.  Broadcast the [root,candidate] relation over modes and solve all
    # mode-wise recourse argmax operations in one NumPy reduction.
    eligible = compat & feasible[None, :]
    any_eligible = np.any(eligible, axis=1)
    masked_value = np.where(eligible[:, :, None], tail_value[None, :, :], -np.inf)
    best_j = np.argmax(masked_value, axis=1)  # [root, mode]
    best_value = np.max(masked_value, axis=1)
    mode_idx = np.broadcast_to(np.arange(H, dtype=int)[None, :], best_j.shape)
    selected_tail_risk = tail_loss[best_j, mode_idx]
    selected_tail_chance = tail_chance[best_j, mode_idx]

    shared_mode_risk = np.max(loss_curves[:, :, : branch + 1], axis=2)
    shared_risk = np.sum(shared_mode_risk * weights[None, :], axis=1)
    contingent_risk = np.sum(np.maximum(shared_mode_risk, selected_tail_risk) * weights[None, :], axis=1)
    chance = np.sum(selected_tail_chance * weights[None, :], axis=1)
    shared_value = uw * utility - rw * shared_risk - sw * smooth - dw * dev
    total_score = shared_value + np.sum(best_value * weights[None, :], axis=1) - cw * chance

    valid = feasible & any_eligible & np.all(np.isfinite(best_value), axis=1)
    risk_out[valid] = contingent_risk[valid]
    chance_out[valid] = chance[valid]
    score[valid] = total_score[valid]
    admitted[valid] = (contingent_risk[valid] <= risk_threshold) & (chance[valid] <= chance_threshold)
    return score, admitted, risk_out, chance_out

def _control_smoothness_cost(d: dict[str, Any], dt: float = 0.2) -> float:
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    cost = 0.0
    if controls.ndim == 2 and controls.size:
        if controls.shape[1] >= 1:
            a = controls[:, 0]
            cost += float(np.nanmean(np.abs(a))) / 4.0
            if a.size > 1:
                cost += 0.25 * float(np.nanmax(np.abs(np.diff(a) / max(dt, 1e-3)))) / 8.0
        if controls.shape[1] >= 2:
            steer = controls[:, 1]
            cost += 0.5 * float(np.nanmean(np.abs(steer))) / 0.6
            if steer.size > 1:
                cost += 0.15 * float(np.nanmax(np.abs(np.diff(steer) / max(dt, 1e-3)))) / 1.0
    if states.ndim == 2 and states.shape[0] > 1 and states.shape[1] >= 5:
        # prefix_states schema: [x,y,vx,vy,heading,yaw_rate,speed,length,width].
        yaw = np.unwrap(states[:, 4])
        yr = np.diff(yaw) / max(dt, 1e-3)
        if yr.size:
            cost += 0.3 * float(np.nanmax(np.abs(yr))) / 1.0
    return float(np.nan_to_num(cost, nan=0.0, posinf=10.0, neginf=0.0))


def _nominal_deviation(samples: list[dict[str, Any]]) -> np.ndarray:
    if not samples:
        return np.zeros((0,), dtype=float)
    ref = np.asarray(samples[0].get("prefix_states", np.zeros((0, 0))), dtype=float)
    vals = []
    for d in samples:
        xy = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
        if ref.ndim != 2 or xy.ndim != 2 or ref.shape[0] == 0 or xy.shape[0] == 0 or ref.shape[1] < 2 or xy.shape[1] < 2:
            vals.append(0.0)
            continue
        T = min(ref.shape[0], xy.shape[0])
        vals.append(float(np.sqrt(np.mean(np.sum((xy[:T, :2] - ref[:T, :2]) ** 2, axis=-1))) / 5.0))
    return np.asarray(vals, dtype=float)


def _macro_names(samples: list[dict[str, Any]]) -> list[str]:
    out = []
    for d in samples:
        v = d.get("prefix_macro_name", d.get("macro_name", ""))
        try:
            v = np.asarray(v).item()
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
        except Exception:
            pass
        out.append(str(v))
    return out


def _posterior_root_values(d: dict[str, Any], alpha: float, temperature: float = 0.7) -> dict[str, Any]:
    eff = _effective_root_outcomes(d, alpha=alpha)
    margins = np.asarray(eff.get("best_margins", np.zeros((0,), dtype=float)), dtype=float)
    K = margins.size
    w, valid = _valid_root_weights(d, K)
    if K == 0 or not valid.any():
        return {**eff, "posterior_expected": 0.0, "entropy": 0.0, "posterior": w}
    logits = np.clip(margins / max(float(temperature), 1e-3), -20.0, 20.0)
    likelihood = np.exp(logits - np.nanmax(logits[valid]))
    post = np.where(valid, w * likelihood, 0.0)
    den = float(post.sum())
    post = post / den if den > 1e-8 else w
    entropy = float(-np.sum(post[post > 0] * np.log(post[post > 0])) / max(np.log(max(int(valid.sum()), 2)), 1e-8))
    return {**eff, "posterior_expected": float(np.sum(post * np.clip(margins, -5.0, 5.0))), "entropy": entropy, "posterior": post}

def _branchwise_values(d: dict[str, Any], alpha: float = 0.2) -> dict[str, Any]:
    M = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    if M.ndim != 2 or M.size == 0:
        return {"expected": 0.0, "cvar": 0.0, "worst": 0.0, "fail_prob": 1.0, "best_options": np.zeros((0,), dtype=int), "best_margins": np.zeros((0,), dtype=float)}
    K, L = M.shape
    opt_valid = _option_valid(d, L)
    masked = np.where(opt_valid[None, :], M, -1.0e9)
    best_options = np.argmax(masked, axis=1).astype(int)
    best_margins = masked[np.arange(K), best_options]
    w, valid = _valid_root_weights(d, K)
    best_margins = np.where(valid & np.isfinite(best_margins), best_margins, -1.0e9)
    expected = float(np.sum(w * np.clip(best_margins, -5.0, 5.0)))
    cvar = _weighted_lower_cvar(np.clip(best_margins, -5.0, 5.0), w, alpha=float(alpha))
    worst = float(np.min(np.clip(best_margins[valid], -5.0, 5.0))) if valid.any() else 0.0
    fail_prob = float(np.sum(w * (best_margins < 0.0))) if w.size else 1.0
    return {"expected": expected, "cvar": cvar, "worst": worst, "fail_prob": fail_prob, "best_options": best_options, "best_margins": best_margins}


def _shared_option_success_score(d: dict[str, Any], gamma: float = 0.0) -> tuple[int, float]:
    M = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    if M.ndim != 2 or M.size == 0:
        return 0, 0.0
    K, L = M.shape
    w, valid = _valid_root_weights(d, K)
    opt_valid = _option_valid(d, L)
    success = ((M >= float(gamma)) & valid[:, None] & opt_valid[None, :]).astype(float)
    mass = (success * w[:, None]).sum(axis=0)
    score = np.where(opt_valid, mass, -1.0e9)
    idx = int(np.argmax(score)) if score.size else 0
    return idx, float(max(score[idx], 0.0)) if score.size else 0.0


def _control_proxy(d: dict[str, Any]) -> tuple[float, float]:
    ctrl = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    if ctrl.ndim != 2 or ctrl.size == 0:
        return 0.0, 0.0
    accel = float(np.nanmax(np.abs(ctrl[:, 0]))) if ctrl.shape[1] >= 1 else 0.0
    steer = float(np.nanmax(np.abs(ctrl[:, 1]))) if ctrl.shape[1] >= 2 else 0.0
    return accel, steer


def _motion_stats(d: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float | np.ndarray]:
    """Kinematic/actuation statistics for finite-lattice contact baselines.

    OC-RAP prefix states follow F_EGO=[x,y,vx,vy,heading,yaw_rate,speed,length,width].
    Older adapters inferred yaw-rate from column 2, which is vx in this schema.
    This helper uses heading/yaw-rate columns when present and gracefully falls
    back to finite differences when samples are feature-only in closed loop.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    dt = float(pcfg.get("contact_dt", pcfg.get("postimpact_dt", 1.0 / float(cfg.get("sample_rate_hz", 10.0) or 10.0))))
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    out: dict[str, float | np.ndarray] = {
        "dt": dt,
        "yaw_rate": 0.0,
        "yaw_acc": 0.0,
        "terminal_speed": 0.0,
        "initial_speed": 0.0,
        "mean_speed": 0.0,
        "lateral_span": 0.0,
        "terminal_lateral_delta": 0.0,
        "heading_delta": 0.0,
        "accel_effort": 0.0,
        "brake_effort": 0.0,
        "steer_effort": 0.0,
        "jerk": 0.0,
        "steer_rate": 0.0,
        "adhesion_proxy": 0.0,
        "speed": np.zeros((0,), dtype=float),
        "yaw_rate_series": np.zeros((0,), dtype=float),
        "controls": controls,
        "states": states,
    }
    if states.ndim == 2 and states.shape[0] > 0:
        if states.shape[1] >= 7:
            speed = np.maximum(0.0, states[:, 6])
        elif states.shape[1] >= 4:
            speed = np.hypot(states[:, 2], states[:, 3])
        else:
            speed = np.zeros((states.shape[0],), dtype=float)
        out["speed"] = np.asarray(np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
        out["initial_speed"] = float(speed[0]) if speed.size else 0.0
        out["terminal_speed"] = float(speed[-1]) if speed.size else 0.0
        out["mean_speed"] = float(np.nanmean(speed)) if speed.size else 0.0
        if states.shape[1] >= 6:
            yr = np.asarray(states[:, 5], dtype=float)
            yr = np.nan_to_num(yr, nan=0.0, posinf=0.0, neginf=0.0)
        elif states.shape[1] >= 5 and states.shape[0] >= 2:
            heading = np.unwrap(states[:, 4])
            yr = np.gradient(heading, dt)
        else:
            yr = np.zeros((states.shape[0],), dtype=float)
        out["yaw_rate_series"] = yr
        out["yaw_rate"] = float(np.nanmax(np.abs(yr))) if yr.size else 0.0
        out["yaw_acc"] = float(np.nanmax(np.abs(np.diff(yr) / max(dt, 1e-3)))) if yr.size >= 2 else 0.0
        if states.shape[1] >= 2:
            y = np.asarray(states[:, 1], dtype=float)
            out["lateral_span"] = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
            out["terminal_lateral_delta"] = float(y[-1] - y[0]) if y.size else 0.0
        if states.shape[1] >= 5:
            heading = np.unwrap(np.asarray(states[:, 4], dtype=float))
            out["heading_delta"] = float(heading[-1] - heading[0]) if heading.size else 0.0
    if controls.ndim == 2 and controls.size:
        if controls.shape[1] >= 1:
            a = np.asarray(controls[:, 0], dtype=float)
            out["accel_effort"] = float(np.nanmean(np.abs(a))) if a.size else 0.0
            out["brake_effort"] = float(np.nanmean(np.maximum(0.0, -a))) if a.size else 0.0
            out["jerk"] = float(np.nanmax(np.abs(np.diff(a) / max(dt, 1e-3)))) if a.size >= 2 else 0.0
        if controls.shape[1] >= 2:
            steer = np.asarray(controls[:, 1], dtype=float)
            out["steer_effort"] = float(np.nanmean(np.abs(steer))) if steer.size else 0.0
            out["steer_rate"] = float(np.nanmax(np.abs(np.diff(steer) / max(dt, 1e-3)))) if steer.size >= 2 else 0.0
        # A compact friction/road-adhesion proxy: longitudinal acceleration plus
        # lateral acceleration implied by steering/yaw-rate should not exceed mu*g.
        mu = float(pcfg.get("postimpact_mu", pcfg.get("contact_mu", 0.75)))
        g = 9.81
        a_long = np.abs(controls[:, 0]) if controls.shape[1] >= 1 else np.zeros((controls.shape[0],), dtype=float)
        if states.ndim == 2 and states.shape[0] > 0:
            speed = np.asarray(out["speed"], dtype=float)
            yr = np.asarray(out["yaw_rate_series"], dtype=float)
            T = min(a_long.size, speed.size, yr.size)
            a_lat = np.abs(speed[:T] * yr[:T]) if T else np.zeros_like(a_long)
            a_long = a_long[:T] if T else a_long
        else:
            a_lat = np.zeros_like(a_long)
        usage = np.sqrt(a_long ** 2 + a_lat ** 2) / max(mu * g, 1e-3)
        out["adhesion_proxy"] = float(np.nanmax(usage)) if usage.size else 0.0
    return out


def _macro_is(d: dict[str, Any], names: set[str]) -> bool:
    v = d.get("prefix_macro_name", d.get("macro_name", ""))
    try:
        v = np.asarray(v).item()
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return str(v).strip().lower() in {str(x).lower() for x in names}


def _preferred_option_index(d: dict[str, Any], modes: list[str], gamma: float = 0.0) -> int:
    M = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    L = int(M.shape[1]) if M.ndim == 2 else 0
    if L <= 0:
        return 0
    modes_arr = np.asarray(d.get("recovery_modes", np.asarray([], dtype=object))).reshape(-1)
    preferred: list[int] = []
    wanted = {m.lower() for m in modes}
    for i, raw in enumerate(modes_arr.tolist()):
        val = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        if val.lower() in wanted and i < L:
            preferred.append(i)
    opt_valid = _option_valid(d, L)
    if preferred:
        w, valid = _valid_root_weights(d, int(M.shape[0]))
        scores = []
        for i in preferred:
            if not opt_valid[i]:
                scores.append(-1.0e9)
                continue
            col = M[:, i]
            succ = float(np.sum(w * (valid & np.isfinite(col) & (col >= float(gamma)))))
            val = float(np.sum(w * np.where(valid & np.isfinite(col), np.clip(col, -5.0, 5.0), 0.0)))
            scores.append(succ + 0.01 * val)
        if scores:
            return int(preferred[int(np.argmax(scores))])
    return int(_shared_option_success_score(d, gamma=gamma)[0])


def _front_obstacle_gap_and_speed(d: dict[str, Any]) -> tuple[float, float]:
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=float)
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=float).reshape(-1)
    if hist.ndim != 3 or hist.shape[0] == 0 or hist.shape[1] <= 1 or ego.size < 5:
        return float("inf"), 0.0
    last = hist[-1]
    vmask = valid[-1].astype(bool) if valid.ndim >= 2 and valid.shape[0] else np.ones((last.shape[0],), dtype=bool)
    if not bool(vmask[1:].any()):
        return float("inf"), 0.0
    ego_xy = ego[:2]
    heading = float(ego[4])
    forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
    rel = last[1:, :2] - ego_xy[None, :]
    lon = rel @ forward
    lat = np.abs(rel @ np.array([-forward[1], forward[0]], dtype=float))
    mask = vmask[1:] & (lon > 0.0) & (lat < 3.5)
    if not bool(mask.any()):
        return float("inf"), 0.0
    ids = np.where(mask)[0]
    j = int(ids[np.argmin(lon[mask])]) + 1
    gap = float(lon[j - 1])
    speed = float(np.hypot(last[j, 3], last[j, 4])) if last.shape[1] >= 5 else 0.0
    return gap, speed


def _safe_braking_distance_proxy(d: dict[str, Any], stats: dict[str, float | np.ndarray], cfg: dict[str, Any]) -> tuple[float, bool]:
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    m = float(pcfg.get("vehicle_mass", 1750.0))
    iz = float(pcfg.get("vehicle_iz", 2350.0))
    mu = float(pcfg.get("postimpact_mu", pcfg.get("contact_mu", 0.75)))
    gap, obstacle_v = _front_obstacle_gap_and_speed(d)
    v = float(stats.get("initial_speed", 0.0))
    yaw_rate = float(np.asarray(stats.get("yaw_rate_series", np.zeros((0,)))).reshape(-1)[0]) if np.asarray(stats.get("yaw_rate_series", np.zeros((0,)))).size else float(stats.get("yaw_rate", 0.0))
    exy = max(0.0, 0.5 * m * (v * v - obstacle_v * obstacle_v))
    ez = 0.5 * iz * yaw_rate * yaw_rate
    sbd = (exy + ez) / max(m * 9.81 * mu, 1e-3)
    feasible = bool(np.isfinite(gap) and (sbd + float(pcfg.get("postimpact_sbd_margin", 4.0)) <= gap))
    return float(sbd), feasible


def _postimpact_mpc_cost(d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None) -> tuple[float, dict[str, float]]:
    """Finite-lattice adapter of planning-integrated post-impact MPC.

    The score uses only quantities available online: candidate kinematics,
    tire-adhesion/stability proxies, safe-braking-distance mode selection, and
    an observation-conditioned multi-modal collision-risk forecast.  Teacher
    recoverability/harm tensors are intentionally excluded from action choice.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    utility = _scalar(d, "utility", 0.0)
    sbd, brake_feasible = _safe_braking_distance_proxy(d, stats, cfg)
    terminal_speed = float(stats["terminal_speed"])
    yaw_rate = float(stats["yaw_rate"])
    yaw_acc = float(stats["yaw_acc"])
    adhesion = float(stats["adhesion_proxy"])
    lateral_span = abs(float(stats["lateral_span"]))
    brake_macro = _macro_is(d, {"brake", "yield", "pull_over", "stabilize"})
    lane_macro = _macro_is(d, {"lane_shift", "merge", "pull_over"})
    if brake_feasible:
        decision_penalty = 0.0 if brake_macro else float(pcfg.get("postimpact_sbd_wrong_mode_penalty", 1.5))
        sbd_mode_cost = float(pcfg.get("postimpact_sbd_terminal_speed_weight", 0.35)) * terminal_speed
    else:
        decision_penalty = 0.0 if lane_macro else float(pcfg.get("postimpact_sbd_wrong_mode_penalty", 1.5))
        sbd_mode_cost = float(pcfg.get("postimpact_lane_change_lateral_weight", 0.08)) * max(0.0, 3.0 - lateral_span)
    stability_cost = (
        float(pcfg.get("postimpact_yaw_rate_weight", 1.4)) * yaw_rate
        + float(pcfg.get("postimpact_yaw_acc_weight", 0.15)) * yaw_acc
        + float(pcfg.get("postimpact_terminal_speed_weight", 0.25)) * terminal_speed
        + float(pcfg.get("postimpact_accel_weight", 0.08)) * float(stats["accel_effort"])
        + float(pcfg.get("postimpact_steer_weight", 0.08)) * float(stats["steer_effort"])
        + float(pcfg.get("postimpact_jerk_weight", 0.02)) * float(stats["jerk"])
        + float(pcfg.get("postimpact_adhesion_weight", 1.1)) * max(0.0, adhesion - 1.0)
    )
    obstacle_cost = (
        float(pcfg.get("postimpact_expected_risk_weight", 5.0)) * risk.expected_loss
        + float(pcfg.get("postimpact_cvar_risk_weight", 2.5)) * risk.cvar_loss
        + float(pcfg.get("postimpact_severity_weight", 1.5)) * risk.severity_proxy
    )
    rejoin_reward = float(pcfg.get("postimpact_rejoin_weight", 0.20)) * utility
    total = stability_cost + obstacle_cost + decision_penalty + sbd_mode_cost - rejoin_reward
    return float(total), {
        "yaw_rate": yaw_rate,
        "yaw_acc": yaw_acc,
        "terminal_speed": terminal_speed,
        "stable_stop_cost": float(stability_cost),
        "obstacle_cost": float(obstacle_cost),
        "sbd": float(sbd),
        "sbd_brake_feasible": float(brake_feasible),
        "adhesion_proxy": float(adhesion),
        "observed_expected_risk": float(risk.expected_loss),
        "observed_cvar_risk": float(risk.cvar_loss),
        "backup_margin": float(risk.backup_margin),
        "rejoin_reward": float(rejoin_reward),
    }


def _stable_stop_cost(d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None) -> tuple[float, dict[str, float]]:
    """Post-crash braking/stable-stop controller scored without oracle labels."""
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    utility = _scalar(d, "utility", 0.0)
    stop_macro = _macro_is(d, {"brake", "yield", "pull_over", "stabilize"})
    cost = (
        float(pcfg.get("stable_stop_terminal_speed_weight", 1.8)) * float(stats["terminal_speed"])
        + float(pcfg.get("stable_stop_yaw_rate_weight", 2.0)) * float(stats["yaw_rate"])
        + float(pcfg.get("stable_stop_yaw_acc_weight", 0.20)) * float(stats["yaw_acc"])
        + float(pcfg.get("stable_stop_expected_risk_weight", 6.0)) * risk.expected_loss
        + float(pcfg.get("stable_stop_cvar_risk_weight", 3.0)) * risk.cvar_loss
        + float(pcfg.get("stable_stop_steer_weight", 0.20)) * float(stats["steer_effort"])
        + float(pcfg.get("stable_stop_jerk_weight", 0.04)) * float(stats["jerk"])
        + (0.0 if stop_macro else float(pcfg.get("stable_stop_non_stop_macro_penalty", 2.0)))
        - float(pcfg.get("stable_stop_utility_tiebreak", 0.03)) * utility
    )
    return float(cost), {
        "terminal_speed": float(stats["terminal_speed"]),
        "yaw_rate": float(stats["yaw_rate"]),
        "stop_macro": float(stop_macro),
        "observed_expected_risk": float(risk.expected_loss),
        "backup_margin": float(risk.backup_margin),
    }


def _trajectory_restoration_cost(d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None) -> tuple[float, dict[str, float]]:
    """Steering/tractive-force post-collision restoration heuristic adapter."""
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    controls = np.asarray(stats["controls"], dtype=float)
    states = np.asarray(stats["states"], dtype=float)
    dt = float(stats["dt"])
    T = int(controls.shape[0]) if controls.ndim == 2 else 0
    t = np.arange(T, dtype=float) * dt
    tau0 = float(pcfg.get("restoration_tau0", 0.1))
    tau1 = float(pcfg.get("restoration_tau1", 0.45))
    tau2 = float(pcfg.get("restoration_tau2", 0.65))
    tau3 = float(pcfg.get("restoration_tau3", 0.95))
    tc1 = float(pcfg.get("restoration_tauc1", 0.35))
    tc2 = float(pcfg.get("restoration_tauc2", 0.85))
    A1 = float(pcfg.get("restoration_A1", 0.175))
    A2 = float(pcfg.get("restoration_A2", -0.10))
    Ac = float(pcfg.get("restoration_accel_pulse", 0.9))
    kdir = float(pcfg.get("restoration_kdir", 1.0))
    sign_src = 0.0
    if states.ndim == 2 and states.shape[0] > 0:
        sign_src += float(states[0, 1]) if states.shape[1] >= 2 else 0.0
        sign_src += float(states[0, 5]) if states.shape[1] >= 6 else 0.0
    direction = -np.sign(sign_src) if abs(sign_src) > 1e-6 else 1.0

    def window_sine(tt: np.ndarray, a: float, lo: float, hi: float) -> np.ndarray:
        if hi <= lo:
            return np.zeros_like(tt)
        mask = (tt >= lo) & (tt <= hi)
        out = np.zeros_like(tt)
        out[mask] = a * np.sin(np.pi * (tt[mask] - lo) / max(hi - lo, 1e-3))
        return out

    steer_ref = kdir * direction * (window_sine(t, A1, tau0, tau1) + window_sine(t, A2, tau2, tau3))
    accel_ref = window_sine(t, Ac, tc1, tc2)
    steer = controls[:, 1] if controls.ndim == 2 and controls.shape[1] >= 2 and T else np.zeros((T,), dtype=float)
    accel = controls[:, 0] if controls.ndim == 2 and controls.shape[1] >= 1 and T else np.zeros((T,), dtype=float)
    shape_cost = 0.0
    if T > 0:
        shape_cost = float(np.nanmean((steer - steer_ref) ** 2) / max(A1 * A1, 1e-4) + 0.25 * np.nanmean((accel - accel_ref) ** 2) / max(Ac * Ac, 1e-4))
    utility = _scalar(d, "utility", 0.0)
    terminal_y = abs(float(stats["terminal_lateral_delta"]))
    v0 = max(float(stats["initial_speed"]), 1e-3)
    speed_preservation_penalty = max(0.0, float(pcfg.get("restoration_min_speed_fraction", 0.45)) * v0 - float(stats["terminal_speed"])) / v0
    cost = (
        float(pcfg.get("restoration_shape_weight", 0.8)) * shape_cost
        + float(pcfg.get("restoration_yaw_rate_weight", 1.1)) * float(stats["yaw_rate"])
        + float(pcfg.get("restoration_lateral_weight", 0.25)) * terminal_y
        + float(pcfg.get("restoration_expected_risk_weight", 4.0)) * risk.expected_loss
        + float(pcfg.get("restoration_cvar_risk_weight", 2.0)) * risk.cvar_loss
        + float(pcfg.get("restoration_speed_preservation_weight", 2.0)) * speed_preservation_penalty
        + float(pcfg.get("restoration_adhesion_weight", 0.75)) * max(0.0, float(stats["adhesion_proxy"]) - 1.0)
        - float(pcfg.get("restoration_utility_weight", 0.25)) * utility
    )
    return float(cost), {
        "shape_cost": float(shape_cost),
        "terminal_speed": float(stats["terminal_speed"]),
        "yaw_rate": float(stats["yaw_rate"]),
        "observed_expected_risk": float(risk.expected_loss),
        "backup_margin": float(risk.backup_margin),
    }


def _severity_minimization_cost(d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None) -> tuple[float, dict[str, float]]:
    """Unavoidable-contact severity minimization using online observables only."""
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    utility = _scalar(d, "utility", 0.0)
    v0 = max(float(stats["initial_speed"]), 1e-3)
    dv_proxy = max(0.0, v0 - float(stats["terminal_speed"])) / v0
    residual_energy = (float(stats["terminal_speed"]) / v0) ** 2
    instability = float(stats["yaw_rate"]) + 0.15 * float(stats["yaw_acc"]) + max(0.0, float(stats["adhesion_proxy"]) - 1.0)
    contact_mode_bonus = float(_macro_is(d, {"brake", "yield", "lane_shift", "pull_over", "stabilize"}))
    cost = (
        float(pcfg.get("severity_collision_probability_weight", 8.0)) * risk.collision_probability
        + float(pcfg.get("severity_observed_risk_weight", 5.0)) * risk.expected_loss
        + float(pcfg.get("severity_tail_risk_weight", 2.5)) * risk.cvar_loss
        + float(pcfg.get("severity_relative_speed_weight", 3.0)) * risk.severity_proxy
        + float(pcfg.get("severity_delta_v_weight", 2.0)) * dv_proxy
        + float(pcfg.get("severity_residual_energy_weight", 0.8)) * residual_energy
        + float(pcfg.get("severity_instability_weight", 1.2)) * instability
        - float(pcfg.get("severity_backup_margin_weight", 0.08)) * np.clip(risk.backup_margin, -20.0, 20.0)
        - float(pcfg.get("severity_utility_tiebreak", 0.05)) * utility
        - float(pcfg.get("severity_contact_macro_bonus", 0.20)) * contact_mode_bonus
    )
    return float(cost), {
        "delta_v_proxy": float(dv_proxy),
        "instability": float(instability),
        "observed_expected_risk": float(risk.expected_loss),
        "observed_cvar_risk": float(risk.cvar_loss),
        "observed_severity": float(risk.severity_proxy),
        "backup_margin": float(risk.backup_margin),
    }



def _postimpact_motion_tvlqr_cost(
    d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None
) -> tuple[float, dict[str, float]]:
    """Finite-lattice adapter of Wang et al. (2022) post-impact planning/control.

    The paper combines polynomial/APF post-impact motion planning, TVLQR force
    tracking, and nonlinear control allocation. WOMD does not contain wheel-level
    force/torque states, so the common-lattice implementation retains the
    trajectory re-alignment, obstacle-potential, stability, and control-effort
    objectives while selecting among executable candidate prefixes.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    lateral = abs(float(stats["terminal_lateral_delta"]))
    heading = abs(float(stats["heading_delta"]))
    yaw_rate = float(stats["yaw_rate"])
    yaw_acc = float(stats["yaw_acc"])
    control = float(stats["accel_effort"]) + 0.8 * float(stats["steer_effort"])
    control_rate = float(stats["jerk"]) + 0.5 * float(stats["steer_rate"])
    # The APF obstacle term is represented by the observation-only risk field:
    # proximity drives expected loss/CVaR upward without accessing teacher labels.
    cost = (
        float(pcfg.get("tvlqr_lateral_weight", 1.2)) * lateral
        + float(pcfg.get("tvlqr_heading_weight", 1.0)) * heading
        + float(pcfg.get("tvlqr_yaw_rate_weight", 1.5)) * yaw_rate
        + float(pcfg.get("tvlqr_yaw_acc_weight", 0.15)) * yaw_acc
        + float(pcfg.get("tvlqr_apf_risk_weight", 4.5)) * risk.expected_loss
        + float(pcfg.get("tvlqr_cvar_weight", 2.0)) * risk.cvar_loss
        + float(pcfg.get("tvlqr_control_weight", 0.12)) * control
        + float(pcfg.get("tvlqr_control_rate_weight", 0.05)) * control_rate
        - float(pcfg.get("tvlqr_progress_weight", 0.10)) * _scalar(d, "utility", 0.0)
    )
    return float(cost), {
        "lateral_error": lateral,
        "heading_error": heading,
        "yaw_rate": yaw_rate,
        "observed_expected_risk": float(risk.expected_loss),
        "observed_cvar_risk": float(risk.cvar_loss),
    }


def _compensatory_postimpact_mpc_cost(
    d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None
) -> tuple[float, dict[str, float]]:
    """Finite-lattice FCC-MPC adapter for post-impact trajectory tracking.

    Cao et al. use active front steering plus differential torque vectoring to
    attenuate lateral/yaw deviations after impact.  OC-RAP cannot reproduce the
    wheel-level allocator from WOMD, so this adapter scores executable candidate
    prefixes by the same tracking/stability/control-effort objectives.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    lat_err = abs(float(stats["terminal_lateral_delta"]))
    heading_err = abs(float(stats["heading_delta"]))
    yaw_rate = float(stats["yaw_rate"])
    yaw_acc = float(stats["yaw_acc"])
    adhesion = max(0.0, float(stats["adhesion_proxy"]) - 1.0)
    control_effort = float(stats["steer_effort"]) + 0.35 * float(stats["accel_effort"])
    rate_effort = float(stats["steer_rate"]) + 0.20 * float(stats["jerk"])
    cost = (
        float(pcfg.get("comp_mpc_lateral_weight", 1.4)) * lat_err
        + float(pcfg.get("comp_mpc_heading_weight", 1.1)) * heading_err
        + float(pcfg.get("comp_mpc_yaw_rate_weight", 1.5)) * yaw_rate
        + float(pcfg.get("comp_mpc_yaw_acc_weight", 0.12)) * yaw_acc
        + float(pcfg.get("comp_mpc_control_weight", 0.16)) * control_effort
        + float(pcfg.get("comp_mpc_control_rate_weight", 0.05)) * rate_effort
        + float(pcfg.get("comp_mpc_adhesion_weight", 1.0)) * adhesion
        + float(pcfg.get("comp_mpc_expected_risk_weight", 4.0)) * risk.expected_loss
        + float(pcfg.get("comp_mpc_cvar_weight", 2.0)) * risk.cvar_loss
        - float(pcfg.get("comp_mpc_utility_weight", 0.10)) * _scalar(d, "utility", 0.0)
    )
    return float(cost), {
        "lateral_error": lat_err,
        "heading_error": heading_err,
        "yaw_rate": yaw_rate,
        "adhesion_proxy": float(stats["adhesion_proxy"]),
        "observed_cvar_risk": float(risk.cvar_loss),
    }


def _robust_postimpact_control_cost(
    d: dict[str, Any], cfg: dict[str, Any], risk: ObservedRiskProfile | None = None
) -> tuple[float, dict[str, float]]:
    """Sliding-surface/QP-allocation objective adapter for Ao et al. (2022).

    The original controller regulates course angle/lateral displacement with a
    sliding-mode upper layer and a fault-tolerant convex allocator.  WOMD has no
    post-impact wheel-torque state, therefore robust allocation is represented by
    candidate feasibility plus adhesion/control-rate penalties, not fabricated
    wheel-level dynamics.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    stats = _motion_stats(d, cfg)
    risk = risk or observed_risk_profile(d, cfg)
    lam_y = float(pcfg.get("robust_pic_lambda_y", 0.8))
    lam_psi = float(pcfg.get("robust_pic_lambda_heading", 0.7))
    e_y = float(stats["terminal_lateral_delta"])
    e_psi = float(stats["heading_delta"])
    yaw = float(stats["yaw_rate"])
    s_y = abs(e_y + lam_y * e_psi)
    s_psi = abs(e_psi + lam_psi * yaw)
    adhesion_excess = max(0.0, float(stats["adhesion_proxy"]) - float(pcfg.get("robust_pic_adhesion_target", 0.9)))
    cost = (
        float(pcfg.get("robust_pic_surface_y_weight", 1.4)) * s_y
        + float(pcfg.get("robust_pic_surface_heading_weight", 1.6)) * s_psi
        + float(pcfg.get("robust_pic_yaw_rate_weight", 1.2)) * yaw
        + float(pcfg.get("robust_pic_adhesion_weight", 1.2)) * adhesion_excess
        + float(pcfg.get("robust_pic_control_rate_weight", 0.08)) * (float(stats["steer_rate"]) + 0.25 * float(stats["jerk"]))
        + float(pcfg.get("robust_pic_expected_risk_weight", 4.5)) * risk.expected_loss
        + float(pcfg.get("robust_pic_cvar_weight", 2.5)) * risk.cvar_loss
    )
    return float(cost), {
        "sliding_surface_y": s_y,
        "sliding_surface_heading": s_psi,
        "yaw_rate": yaw,
        "adhesion_proxy": float(stats["adhesion_proxy"]),
        "observed_cvar_risk": float(risk.cvar_loss),
    }

def _temporal_contingency_risk(
    profile: ObservedRiskProfile,
    branch_fraction: float,
    *,
    alpha: float,
    tail_risk_weight: float,
    shared_prefix_weight: float,
    collision_threshold_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """Finite-lattice scenario-tree risk with a non-anticipative shared prefix.

    The candidate is shared until the branch point.  The prefix is therefore
    upper-tail aggregated across all modes; only the tail receives the
    belief-weighted expected/CVaR mixture.  This is a deployable approximation
    of MARC/RACP when their continuous optimizer is replaced by OC-RAP's
    executable candidate lattice.
    """
    curves = np.asarray(profile.loss_curves, dtype=float)
    clearance = np.asarray(profile.clearance_curves, dtype=float)
    weights = np.asarray(profile.weights, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0 or curves.shape[1] == 0:
        risk = (1.0 - tail_risk_weight) * float(profile.expected_loss) + tail_risk_weight * float(profile.cvar_loss)
        return risk, float(profile.expected_loss), float(profile.cvar_loss), float(profile.collision_probability)
    T = int(curves.shape[1])
    branch = int(np.clip(round(float(branch_fraction) * (T - 1)), 1, max(T - 1, 1))) if T > 1 else 0
    prefix_mode = np.max(curves[:, : branch + 1], axis=1)
    tail_mode = np.max(curves[:, branch:], axis=1)
    prefix_cvar = _weighted_upper_cvar(prefix_mode, weights, alpha=float(alpha))
    tail_expected = float(np.sum(weights * tail_mode))
    tail_cvar = _weighted_upper_cvar(tail_mode, weights, alpha=float(alpha))
    chance = float(np.sum(weights * np.any(clearance[:, branch:] <= float(collision_threshold_m), axis=1))) if clearance.shape == curves.shape else float(profile.collision_probability)
    tail = (1.0 - float(tail_risk_weight)) * tail_expected + float(tail_risk_weight) * tail_cvar
    total = float(shared_prefix_weight) * prefix_cvar + (1.0 - float(shared_prefix_weight)) * tail
    return total, tail_expected, tail_cvar, chance


def _pdm_source_projected_scores(
    samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    feasible: np.ndarray,
    profiles: list[ObservedRiskProfile],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Project tuPlan Garage's PDMScorer onto the common candidate lattice.

    The public scorer is ``prod(multiplicative_metrics) * weighted_average``
    with weights progress/TTC/comfort = 5/5/2.  nuPlan's polygon-based
    at-fault/drivable/direction metrics cannot be reproduced exactly from WOMD
    NPZs, so this bridge computes *binary observable proxies* from the executable
    candidate, its route, and predicted occupancy.  Importantly, no OC-RAP
    teacher utility/recovery label enters the score.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    n = len(samples)
    collision_threshold = float(pcfg.get("pdm_collision_probability_gate", 0.05))
    min_clearance_gate = float(pcfg.get("pdm_min_clearance_gate_m", 0.0))
    route_half_width = float(pcfg.get("pdm_route_half_width_m", 4.0))
    direction_backtrack = float(pcfg.get("pdm_direction_backtrack_threshold_m", pcfg.get("pdm_direction_backtrack_m", 2.0)))
    comfortable_jerk = float(pcfg.get("pdm_comfort_jerk_threshold_mps3", pcfg.get("pdm_comfort_jerk_mps3", 8.0)))
    comfortable_yaw_rate = float(pcfg.get("pdm_comfort_yaw_rate_threshold_rps", pcfg.get("pdm_comfort_yaw_rate_rps", 0.8)))
    comfortable_steer_rate = float(pcfg.get("pdm_comfort_steer_rate_threshold_rps", pcfg.get("pdm_comfort_steer_rate_rps", 1.2)))

    progress = np.zeros(n, dtype=float)
    ttc = np.ones(n, dtype=float)
    comfort = np.ones(n, dtype=float)
    no_collision = np.ones(n, dtype=float)
    direction = np.ones(n, dtype=float)
    drivable = np.ones(n, dtype=float)

    for i, (d, rp) in enumerate(zip(samples, profiles)):
        states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
        if states.ndim == 2 and states.shape[0] and states.shape[1] >= 2:
            xy = states[:, :2]
            # OC-RAP histories/routes are ego-centric.  Longitudinal route
            # progress is therefore the centerline-like x displacement when a
            # route projection is not available.
            route = np.asarray(d.get("route", np.zeros((0, 2))), dtype=float)
            if route.ndim == 2 and route.shape[0] >= 2 and route.shape[1] >= 2:
                rxy = route[:, :2]
                # Nearest route index is a lightweight approximation of PDMPath.project.
                d2 = ((xy[:, None, :] - rxy[None, :, :]) ** 2).sum(-1)
                nearest = d2.argmin(axis=1)
                seg = np.linalg.norm(np.diff(rxy, axis=0), axis=-1)
                arc = np.concatenate([[0.0], np.cumsum(seg)])
                progress[i] = max(0.0, float(arc[nearest[-1]] - arc[nearest[0]]))
                drivable[i] = float(np.sqrt(d2.min(axis=1)).max(initial=0.0) <= route_half_width)
                direction[i] = float(progress[i] >= 0.0 and float(xy[-1, 0] - xy[0, 0]) >= -direction_backtrack)
            else:
                progress[i] = max(0.0, float(xy[-1, 0] - xy[0, 0]))
                direction[i] = float(float(xy[-1, 0] - xy[0, 0]) >= -direction_backtrack)
        else:
            progress[i] = 0.0

        no_collision[i] = float(
            rp.collision_probability <= collision_threshold and rp.min_clearance >= min_clearance_gate
        )
        # The source TTC metric is binary.  Use the same observable collision
        # forecast as a projected binary TTC infraction rather than injecting a
        # continuous risk cost into the PDM weighted average.
        ttc[i] = no_collision[i]
        st = _motion_stats(d, cfg)
        comfort[i] = float(
            float(st["jerk"]) <= comfortable_jerk
            and float(st["yaw_rate"]) <= comfortable_yaw_rate
            and float(st["steer_rate"]) <= comfortable_steer_rate
        )

    multi = feasible.astype(float) * no_collision * direction * drivable
    gated_progress = progress * multi
    max_progress = float(np.max(gated_progress)) if gated_progress.size else 0.0
    if max_progress > float(pcfg.get("pdm_progress_distance_threshold_m", 0.1)):
        norm_progress = gated_progress / max_progress
    else:
        norm_progress = np.ones(n, dtype=float)
        norm_progress[multi <= 0.0] = 0.0

    wp = float(pcfg.get("pdm_progress_weight", 5.0))
    wt = float(pcfg.get("pdm_ttc_weight", 5.0))
    wc = float(pcfg.get("pdm_comfort_weight", 2.0))
    denom = max(wp + wt + wc, 1.0e-6)
    score = multi * (wp * norm_progress + wt * ttc + wc * comfort) / denom
    admitted = multi > 0.0
    parts = {
        "multiplicative": multi,
        "progress": norm_progress,
        "ttc": ttc,
        "comfort": comfort,
    }
    return score, admitted, parts


def select_external_policy(
    baseline: str,
    samples: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
    *,
    model_outputs: dict[str, np.ndarray] | None = None,
    precomputed_profiles: list[ObservedRiskProfile] | None = None,
    precomputed_context: ObservedRiskContext | None = None,
) -> ExternalSelection:
    """Select a candidate for an external baseline.

    Except for the explicitly named oracle upper bound, every selector uses only
    online-observable candidate/model quantities.  OC-RAP teacher labels remain
    available to the evaluator *after* selection, never as policy inputs.
    """
    cfg = cfg or {}
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    pcfg = bcfg.get("policy", {}) if isinstance(bcfg.get("policy", {}), dict) else {}
    baseline = str(baseline).lower()
    n = len(samples)
    if n == 0:
        return ExternalSelection(0, "empty_candidate_set", np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float))

    utility = np.asarray([_scalar(d, "utility", 0.0) for d in samples], dtype=float)
    feasible = np.asarray([_scalar(d, "feasible", 1.0) > 0.5 for d in samples], dtype=bool)

    # Predictor-free paper/source ports need neither nominal-deviation
    # bookkeeping nor OC-RAP's learned actor-risk predictor.  The v58 Parseh
    # severity planner is pre-impact Near-contact, while the four v57 recovery
    # controllers remain Contact; all are dispatched before generic risk work.
    predictor_free_paper_ports = {
        "post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop",
        "post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration",
        "compensatory_postimpact_mpc", "cao_postimpact_mpc",
        "robust_postimpact_control", "postimpact_sliding_mode", "ao_postimpact_control",
        "severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner",
    }
    if baseline in predictor_free_paper_ports:
        if baseline in {"post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop"}:
            port = post_crash_braking_port(samples, cfg)
            reason = "lu2017_postimpact_braking_abs_candidate_port"
        elif baseline in {"post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration"}:
            port = post_collision_restoration_port(samples, cfg)
            reason = "ghosh2026_open_loop_steering_tractive_force_candidate_port"
        elif baseline in {"compensatory_postimpact_mpc", "cao_postimpact_mpc"}:
            port = compensatory_postimpact_mpc_port(samples, cfg)
            reason = "cao2021_fcc_mpc_source_limited_structured_candidate_port"
        elif baseline in {"severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner"}:
            port = severity_minimization_port(samples, cfg)
            reason = "parseh2023_kudlich_slibar_postimpact_eq25_candidate_port"
        else:
            port = robust_postimpact_control_port(samples, cfg)
            reason = "ao2022_sliding_mode_fault_tolerant_exact_qp_candidate_port"
        idx, reason = _admission_select(
            port.score, port.admitted, feasible, fallback_score=port.fallback_score,
            reason=reason, prefer_nominal_if_admitted=False,
        )
        return ExternalSelection(idx, reason, port.admitted, port.score)

    dev = _nominal_deviation(samples)
    smooth = np.asarray([_control_smoothness_cost(d, dt=float(pcfg.get("dt", 0.2))) for d in samples], dtype=float)
    macros = _macro_names(samples)

    if baseline in {"nominal", "nominal_replay", "log_replay"}:
        admitted = np.zeros(n, dtype=bool)
        nominal = [i for i, d in enumerate(samples) if _scalar(d, "is_nominal", 0.0) > 0.5]
        idx = int(nominal[0] if nominal else 0)
        if not feasible[idx]:
            idx = _best(utility, feasible)
        admitted[idx] = True
        return ExternalSelection(idx, "logged_nominal_replay", admitted, utility.copy())

    if baseline in {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer"}:
        admitted = np.zeros(n, dtype=bool)
        if model_outputs is not None and "logits" in model_outputs:
            score = np.asarray(model_outputs["logits"], dtype=float).reshape(-1)[:n]
            reason = "wayformer_early_fusion_gmm_ego_bc_candidate_projection"
            idx = _best(score, feasible)
        else:
            score = -dev
            idx = 0 if feasible[0] else _best(score, feasible)
            reason = "wayformer_checkpoint_missing_nominal_fallback"
        admitted[idx] = True
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"gameformer", "gameformer_lite", "gameformer_levelk"}:
        admitted = np.zeros(n, dtype=bool)
        if model_outputs is not None and "logits" in model_outputs:
            score = np.asarray(model_outputs["logits"], dtype=float).reshape(-1)[:n]
            idx = _best(score, feasible)
            reason = "learned_gameformer_levelk_policy"
        else:
            score = -dev
            idx = 0 if feasible[0] else _best(score, feasible)
            reason = "gameformer_checkpoint_missing_nominal_fallback"
        admitted[idx] = True
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"betop", "betop_lite", "betopnet", "betopnet_lite"}:
        admitted = np.zeros(n, dtype=bool)
        if model_outputs is not None and "logits" in model_outputs:
            # Paper Appendix C.2: planning inference combines trajectory
            # confidence with a short-term repulsive-potential cost C_M, with
            # t_b=3 and lambda_m=0.5. The released repository does not include
            # the nuPlan planning pipeline, so at the common executable lattice
            # we use the published positive repulsive potential as a penalty
            # against normalized candidate confidence. Actor futures are
            # observation-only CV projections, never OC-RAP teacher futures.
            raw = np.asarray(model_outputs["logits"], dtype=float).reshape(-1)[:n]
            raw = raw - float(np.max(raw)) if raw.size else raw
            prob = np.exp(np.clip(raw, -60.0, 0.0))
            prob = prob / max(float(prob.sum()), 1.0e-12)
            cm = _betop_short_term_repulsive_cost(samples, cfg)
            lambda_m = float(pcfg.get("betop_short_term_cost_weight", 0.5))
            score = prob - lambda_m * cm
            idx = _best(score, feasible)
            reason = "betop_topology_guided_confidence_plus_short_term_contingency_cost_adapter"
        else:
            score = -dev
            idx = 0 if feasible[0] else _best(score, feasible)
            reason = "betop_checkpoint_missing_nominal_fallback"
        admitted[idx] = True
        return ExternalSelection(idx, reason, admitted, score)


    if baseline in {"plantf", "plan_tf", "plantf_adapter"}:
        admitted = np.zeros(n, dtype=bool)
        if model_outputs is not None and "logits" in model_outputs:
            score = np.asarray(model_outputs["logits"], dtype=float).reshape(-1)[:n]
            idx = _best(score, feasible)
            reason = "plantf_state_dropout_imitation_candidate_adapter"
        else:
            score = -dev
            idx = 0 if feasible[0] else _best(score, feasible)
            reason = "plantf_checkpoint_missing_nominal_fallback"
        admitted[idx] = True
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"pluto", "pluto_adapter"}:
        admitted = np.zeros(n, dtype=bool)
        if model_outputs is not None and "logits" in model_outputs:
            base_logits = np.asarray(model_outputs["logits"], dtype=float).reshape(-1)[:n]
            contrast = np.asarray(model_outputs.get("pluto_contrastive_logits", np.zeros_like(base_logits)), dtype=float).reshape(-1)[:n]
            score = base_logits + float(pcfg.get("pluto_contrastive_selection_weight", 0.15)) * contrast
            idx = _best(score, feasible)
            reason = "pluto_query_cil_candidate_adapter"
        else:
            score = -dev
            idx = 0 if feasible[0] else _best(score, feasible)
            reason = "pluto_checkpoint_missing_nominal_fallback"
        admitted[idx] = True
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"}:
        # Deliberate non-deployable upper bound.  This is the only selector that
        # may consume OC-RAP teacher tensors.
        alpha = float(pcfg.get("cvar_alpha", 0.2))
        gamma_o = float(pcfg.get("gamma_oracle_rec", pcfg.get("gamma_branch_rec", 0.0)))
        hard = np.asarray([_scalar(d, "hard_violation", 0.0) for d in samples], dtype=float)
        harm = np.asarray([_scalar(d, "harm_proxy", 0.0) for d in samples], dtype=float)
        branch_eff = [_effective_root_outcomes(d, alpha=alpha, gamma=gamma_o) for d in samples]
        oracle_all = np.asarray([bool(b["oracle_all_roots"]) for b in branch_eff], dtype=bool)
        branch_cvar = np.asarray([b["cvar"] for b in branch_eff], dtype=float)
        teacher_safe = feasible & (hard <= float(pcfg.get("gamma_H", 0.0))) & (harm <= float(pcfg.get("gamma_D", 5.0)))
        admitted = teacher_safe & oracle_all & (branch_cvar >= gamma_o)
        score = branch_cvar + float(pcfg.get("oracle_utility_tiebreak", 1.0e-3)) * utility
        idx = _best(score, admitted if admitted.any() else feasible)
        opts = np.asarray(branch_eff[idx].get("best_options", np.zeros((0,), dtype=int)))
        opt = int(opts[0]) if opts.size else None
        return ExternalSelection(idx, "teacher_only_branchwise_oracle_upper_bound", admitted, score, selected_option=opt)

    # v56 paper-core ports are dispatched before the legacy scalar candidate-
    # risk profiles.  The two Near-contact filters require the benchmark's
    # observation-only trajectory predictor.  The two Wang post-impact ports do
    # not: Wang 2023 uses constant-velocity obstacles (Eq. 15), while Wang 2022
    # uses fixed perceived obstacle coordinates in its APF (Eqs. 4-6).  Keeping
    # those Contact ports predictor-free is both source-faithful and faster.
    context_only_paper_ports = {
        "dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter",
        "conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf",
    }
    v56_predictor_free_paper_ports = {
        "postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc",
        "postimpact_motion_tvlqr", "postimpact_motion_planning", "wang2022_postimpact", "postimpact_tvlqr",
    }
    if baseline in context_only_paper_ports | v56_predictor_free_paper_ports:
        risk_context = None
        if baseline in context_only_paper_ports:
            risk_context = precomputed_context if precomputed_context is not None else build_observed_risk_context(samples[0], cfg)
        if baseline in {"dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter"}:
            assert risk_context is not None
            port = dr_cvar_safe_halfspace_port(samples, cfg, risk_context)
            reason = "safaoui_summers_drcvar_safe_halfspace_plus_mpc_candidate_port"
            prefer_nominal = False
        elif baseline in {"conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf"}:
            assert risk_context is not None
            port = cpsf_constrained_projection_port(samples, cfg, risk_context)
            reason = "strawn_ayanian_lindemann_cpsf_eq7_conformal_tube_candidate_port"
            prefer_nominal = True
        elif baseline in {"postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc"}:
            port = integrated_postimpact_mpc_pso_port(samples, cfg)
            reason = "wang2023_integrated_postimpact_mpc_sbd_constraints_pso_candidate_port"
            prefer_nominal = False
        else:
            port = postimpact_motion_tvlqr_port(samples, cfg)
            reason = "wang2022_quintic_apf_tvlqr_nonlinear_allocation_candidate_port"
            prefer_nominal = False
        idx, reason = _admission_select(
            port.score, port.admitted, feasible, fallback_score=port.fallback_score,
            reason=reason, prefer_nominal_if_admitted=prefer_nominal,
        )
        return ExternalSelection(idx, reason, port.admitted, port.score)

    # Deployable scenario-risk profiles shared by all remaining non-oracle planning/filter
    # baselines.  They are derived from candidate trajectories and observed agent
    # histories, not from m_star/r_orc/r_dep/harm labels.
    if precomputed_profiles is not None:
        profiles = precomputed_profiles
        risk_context = precomputed_context
    else:
        profiles, risk_context = observed_risk_profiles_and_context(samples, cfg)
    if len(profiles) != n:
        raise ValueError(f"precomputed_profiles length {len(profiles)} does not match candidate count {n}")
    exp_risk = np.asarray([p.expected_loss for p in profiles], dtype=float)
    cvar_risk = np.asarray([p.cvar_loss for p in profiles], dtype=float)
    worst_risk = np.asarray([p.worst_loss for p in profiles], dtype=float)
    collision_prob = np.asarray([p.collision_probability for p in profiles], dtype=float)
    backup_margin = np.asarray([p.backup_margin for p in profiles], dtype=float)
    min_clearance = np.asarray([p.min_clearance for p in profiles], dtype=float)
    severity = np.asarray([p.severity_proxy for p in profiles], dtype=float)


    if baseline in {"idm", "idm_planner"}:
        # Intelligent Driver Model longitudinal controller projected onto the
        # common executable candidate lattice. Candidate 0 is the route/nominal
        # reference; observed front-vehicle state determines the IDM acceleration.
        a_max = float(pcfg.get("idm_max_accel_mps2", 1.5))
        b = float(pcfg.get("idm_comfort_decel_mps2", 2.0))
        T_headway = float(pcfg.get("idm_time_headway_s", 1.5))
        s0 = float(pcfg.get("idm_min_gap_m", 2.0))
        v0 = max(float(pcfg.get("idm_target_speed_mps", 13.0)), 0.5)
        delta = float(pcfg.get("idm_accel_exponent", 4.0))
        candidate_accel = np.zeros(n, dtype=float)
        idm_target = np.zeros(n, dtype=float)
        for i, d in enumerate(samples):
            stats = _motion_stats(d, cfg)
            v = max(float(stats["initial_speed"]), 0.0)
            gap, lead_v = _front_obstacle_gap_and_speed(d)
            interaction = 0.0
            if np.isfinite(gap):
                dv = max(v - lead_v, 0.0)
                s_star = s0 + v * T_headway + v * dv / max(2.0 * np.sqrt(max(a_max * b, 1e-6)), 1e-3)
                interaction = (s_star / max(gap, 0.5)) ** 2
            idm_target[i] = a_max * (1.0 - (v / v0) ** delta - interaction)
            ctrl = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
            candidate_accel[i] = float(np.nanmean(ctrl[: min(5, len(ctrl)), 0])) if ctrl.ndim == 2 and ctrl.shape[0] and ctrl.shape[1] >= 1 else 0.0
        admitted = feasible & (collision_prob <= float(pcfg.get("idm_collision_probability_gate", 0.55)))
        score = (
            -float(pcfg.get("idm_accel_tracking_weight", 1.0)) * np.abs(candidate_accel - idm_target)
            -float(pcfg.get("idm_deviation_weight", 0.10)) * dev
            -float(pcfg.get("idm_risk_weight", 1.5)) * exp_risk
            +float(pcfg.get("idm_utility_weight", 0.15)) * utility
        )
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "idm_longitudinal_control_candidate_projection", admitted, score)

    if baseline in {"pdm_closed", "pdm_closed_adapter"}:
        score, admitted, _parts = _pdm_source_projected_scores(samples, cfg, feasible, profiles)
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "pdm_closed_source_scorer_projected_to_womd_candidate_lattice", admitted, score)

    if baseline in {"pdm_hybrid", "pdm_hybrid_adapter"}:
        # Source PDM-Hybrid executes PDM-Closed unchanged through the configured
        # correction horizon (2.0 s in the uploaded tuPlan Garage config) and
        # applies PDM-Offset only afterwards.  OC-RAP's standard closed-loop
        # protocol executes/replans a short prefix inside that horizon, so using
        # a learned logit to change the current candidate would be *less* faithful
        # than PDM-Closed.  Report the source-semantics projection explicitly.
        score, admitted, _parts = _pdm_source_projected_scores(samples, cfg, feasible, profiles)
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "pdm_hybrid_source_semantics_closed_prefix_before_2s_correction", admitted, score)

    if baseline in {"robust_scenario_mpc", "scenario_mpc", "batkovic_scenario_mpc"}:
        score, admitted, _branch = _robust_scenario_mpc_candidate_scores(
            samples, profiles, feasible, utility, smooth, dev, pcfg
        )
        idx, reason = _admission_select(
            score, admitted, feasible,
            fallback_score=-(worst_risk + 0.25 * cvar_risk + 0.10 * collision_prob),
            reason="robust_scenario_mpc_mode_distinction_nonanticipative_candidate_port",
        )
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"marc", "marc_lite", "marc_contingency"}:
        score, admitted, admission_risk, chance_risk = _marc_candidate_scores(
            samples, profiles, feasible, utility, smooth, dev, macros, pcfg
        )
        idx, reason = _admission_select(
            score, admitted, feasible,
            fallback_score=-(admission_risk + chance_risk),
            reason="marc_policy_conditioned_dynamic_scenario_tree_cvar_candidate_port",
        )
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"racp", "racp_lite", "risk_aware_contingency"}:
        score, admitted, contingent_risk, chance_risk = _racp_candidate_scores(
            samples, profiles, feasible, utility, smooth, dev, pcfg
        )
        idx, reason = _admission_select(
            score, admitted, feasible,
            fallback_score=-(contingent_risk + chance_risk),
            reason="racp_source_structured_shared_plus_belief_weighted_contingent_candidate_port",
        )
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"expected_risk", "expected_risk_filter", "expected_risk_planner"}:
        admitted = feasible & (exp_risk <= float(pcfg.get("expected_risk_threshold", 0.45)))
        score = utility - float(pcfg.get("expected_risk_weight", 3.0)) * exp_risk - float(pcfg.get("risk_deviation_weight", 0.05)) * dev
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "expected_observation_conditioned_collision_risk_filter", admitted, score)

    if baseline in {"cvar_risk", "cvar_risk_filter", "cvar_planner"}:
        admitted = feasible & (cvar_risk <= float(pcfg.get("cvar_risk_threshold", 0.55)))
        score = utility - float(pcfg.get("cvar_risk_weight", 3.0)) * cvar_risk - float(pcfg.get("risk_deviation_weight", 0.05)) * dev
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "cvar_observation_conditioned_tail_risk_filter", admitted, score)

    if baseline in {"dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter"}:
        # Legacy v48/v49 dispersion surrogate retained only for reproducibility;
        # it is deliberately not used by the v50 main near-contact table.
        ambiguity = float(pcfg.get("dro_ambiguity_radius", 0.10))
        dispersion = np.asarray([float(np.sqrt(np.sum(p.weights * (p.losses - p.expected_loss) ** 2))) for p in profiles], dtype=float)
        risk = cvar_risk + ambiguity * dispersion / max(float(pcfg.get("cvar_alpha", 0.2)), 1e-3)
        admitted = feasible & (risk <= float(pcfg.get("dro_cvar_threshold", 0.65)))
        score = utility - float(pcfg.get("dro_cvar_risk_weight", 3.5)) * risk - float(pcfg.get("risk_deviation_weight", 0.05)) * dev
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "legacy_wasserstein_inspired_dispersion_surrogate_not_main_table", admitted, score)

    if baseline in {"predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter"}:
        # Wabersich & Zeilinger: apply the proposed controller input unchanged
        # whenever its finite-horizon prediction satisfies state/input
        # constraints and reaches a terminal safe set. Otherwise solve a
        # minimally-invasive safety-filter problem.  Here that optimization is
        # projected onto the executable candidate lattice; no CBF constraint is
        # fabricated because it is not part of the cited PSF formulation.
        accel = np.zeros(n, dtype=float)
        steer = np.zeros(n, dtype=float)
        for i, d in enumerate(samples):
            accel[i], steer[i] = _control_proxy(d)
        ctrl_ok = (accel <= float(pcfg.get("psf_accel_gate", 6.0))) & (steer <= float(pcfg.get("psf_steer_gate", 0.75)))
        stage_margin = np.asarray([float(np.min(p.clearance_curves)) if np.asarray(p.clearance_curves).size else p.min_clearance for p in profiles], dtype=float)
        terminal_barrier = np.asarray([float(np.min(np.asarray(p.backup_margin_curves)[:, -1])) if np.asarray(p.backup_margin_curves).ndim == 2 and np.asarray(p.backup_margin_curves).shape[1] else p.backup_margin for p in profiles], dtype=float)
        stage_ok = stage_margin >= float(pcfg.get("psf_stage_clearance_margin_m", 0.0))
        terminal_ok = terminal_barrier >= float(pcfg.get("psf_terminal_backup_margin_m", pcfg.get("psf_backup_margin_m", 0.0)))
        admitted = feasible & ctrl_ok & stage_ok & terminal_ok
        nominal_ids = [i for i, d in enumerate(samples) if _scalar(d, "is_nominal", 0.0) > 0.5]
        nominal_idx = int(nominal_ids[0] if nominal_ids else 0)
        input_dev = _control_sequence_deviation(samples, nominal_idx)
        safety_tiebreak = float(pcfg.get("psf_safety_tiebreak_weight", 1.0e-3))
        score = -input_dev + safety_tiebreak * np.minimum(stage_margin, terminal_barrier)
        fallback_barrier = np.minimum(stage_margin, terminal_barrier)
        fallback_score = fallback_barrier - 1.0e-3 * input_dev
        idx, reason = _admission_select(
            score, admitted, feasible, fallback_score=fallback_score,
            reason="predictive_safety_filter_finite_horizon_terminal_safe_set_minimal_input_correction",
            prefer_nominal_if_admitted=False,
        )
        if admitted[nominal_idx]:
            idx = nominal_idx
        return ExternalSelection(idx, reason, admitted, score)

    # v57 legacy/dead compatibility branches below are retained only for patch/readability
    # comparison; registered aliases return through the predictor-free paper ports above.
    if baseline in {"post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop"}:
        details_list = []
        costs = []
        for d, p in zip(samples, profiles):
            c, details = _stable_stop_cost(d, cfg, p)
            costs.append(c); details_list.append(details)
        cost = np.asarray(costs, dtype=float)
        score = -cost
        stop_gate_speed = float(pcfg.get("stable_stop_terminal_speed_gate", 2.0))
        yaw_gate = float(pcfg.get("stable_stop_yaw_rate_gate", 1.4))
        stop_macro = np.asarray([_macro_is(d, {"brake", "yield", "pull_over", "stabilize"}) for d in samples], dtype=bool)
        stable_gate = np.asarray([x["terminal_speed"] <= stop_gate_speed and x["yaw_rate"] <= yaw_gate for x in details_list], dtype=bool)
        admitted = feasible & stop_macro & stable_gate & (exp_risk <= float(pcfg.get("stable_stop_risk_gate", 1.25)))
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "post_crash_braking_stable_stop_observed_risk", admitted, score)

    if baseline in {"post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration"}:
        details_list = []
        costs = []
        for d, p in zip(samples, profiles):
            c, details = _trajectory_restoration_cost(d, cfg, p)
            costs.append(c); details_list.append(details)
        cost = np.asarray(costs, dtype=float)
        score = -cost
        restoration_macro = np.asarray([_macro_is(d, {"stabilize", "lane_shift", "merge", "yield", "pull_over", "keep"}) for d in samples], dtype=bool)
        yaw_gate = float(pcfg.get("restoration_yaw_rate_gate", 2.2))
        speed_frac = float(pcfg.get("restoration_admit_min_speed_fraction", 0.30))
        speed_ok = []
        yaw_ok = []
        for d, det in zip(samples, details_list):
            st = _motion_stats(d, cfg)
            speed_ok.append(float(st["terminal_speed"]) >= speed_frac * max(float(st["initial_speed"]), 1e-3))
            yaw_ok.append(float(det["yaw_rate"]) <= yaw_gate)
        admitted = feasible & restoration_macro & np.asarray(speed_ok, dtype=bool) & np.asarray(yaw_ok, dtype=bool) & (cvar_risk <= float(pcfg.get("restoration_risk_gate", 1.5)))
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "post_collision_trajectory_restoration_observed_risk", admitted, score)


    if baseline in {"compensatory_postimpact_mpc", "cao_postimpact_mpc"}:
        details_list, costs = [], []
        for d, p in zip(samples, profiles):
            c, details = _compensatory_postimpact_mpc_cost(d, cfg, p)
            costs.append(c); details_list.append(details)
        cost = np.asarray(costs, dtype=float)
        score = -cost
        admitted = feasible & (np.asarray([x["yaw_rate"] for x in details_list]) <= float(pcfg.get("comp_mpc_yaw_rate_gate", 2.2))) & (cvar_risk <= float(pcfg.get("comp_mpc_risk_gate", 1.5)))
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "cao_compensatory_postimpact_fcc_mpc_candidate_adapter", admitted, score)

    if baseline in {"robust_postimpact_control", "postimpact_sliding_mode", "ao_postimpact_control"}:
        details_list, costs = [], []
        for d, p in zip(samples, profiles):
            c, details = _robust_postimpact_control_cost(d, cfg, p)
            costs.append(c); details_list.append(details)
        cost = np.asarray(costs, dtype=float)
        score = -cost
        adhesion = np.asarray([x["adhesion_proxy"] for x in details_list], dtype=float)
        admitted = feasible & (adhesion <= float(pcfg.get("robust_pic_adhesion_gate", 1.30))) & (cvar_risk <= float(pcfg.get("robust_pic_risk_gate", 1.5)))
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "ao_sliding_mode_fault_tolerant_postimpact_candidate_adapter", admitted, score)

    if baseline in {"severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner"}:
        details_list = []
        costs = []
        for d, p in zip(samples, profiles):
            c, details = _severity_minimization_cost(d, cfg, p)
            costs.append(c); details_list.append(details)
        cost = np.asarray(costs, dtype=float)
        score = -cost
        finite = np.isfinite(cost)
        threshold = float(pcfg.get("severity_admit_threshold", np.nanpercentile(cost[finite], 60.0) if finite.any() else 1.0))
        admitted = feasible & finite & (cost <= threshold)
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "unavoidable_collision_observed_severity_minimization", admitted, score)

    raise ValueError(f"Unknown external baseline {baseline!r}")

