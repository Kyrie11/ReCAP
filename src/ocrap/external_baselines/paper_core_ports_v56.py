from __future__ import annotations

"""Paper-core finite-candidate ports for audited external baselines (v56).

The functions in this module implement the mathematical mechanisms of the cited
methods and project their continuous optimizers onto OC-RAP's common executable
candidate lattice.  They intentionally do not consume OC-RAP teacher labels.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .observed_risk import ObservedRiskContext


_EPS = 1.0e-9


def _pcfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})


def _float(cfg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(_pcfg(cfg).get(key, default))
    except Exception:
        return float(default)


def _int(cfg: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(_pcfg(cfg).get(key, default))
    except Exception:
        return int(default)


def _arr(d: dict[str, Any], key: str, cols: int = 0) -> np.ndarray:
    x = np.asarray(d.get(key, np.zeros((0, cols))), dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(x: np.ndarray, T: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    T = max(int(T), 1)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == T:
        return x.copy()
    if x.shape[0] == 0:
        return np.zeros((T,) + x.shape[1:], dtype=float)
    if x.shape[0] == 1:
        return np.repeat(x, T, axis=0)
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, T)
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])], axis=-1)
    return out.reshape((T,) + x.shape[1:])


def _candidate_xy(d: dict[str, Any], T: int) -> np.ndarray:
    s = _arr(d, "prefix_states")
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] < 2:
        e = np.asarray(d.get("ego_state", np.zeros(9)), dtype=float).reshape(-1)
        xy = np.repeat(e[None, :2], max(T, 1), axis=0) if e.size >= 2 else np.zeros((T, 2))
        return xy
    return _resample(s[:, :2], T)


def _candidate_states(d: dict[str, Any], T: int) -> np.ndarray:
    s = _arr(d, "prefix_states")
    if s.ndim != 2 or s.shape[0] == 0:
        e = np.asarray(d.get("ego_state", np.zeros(9)), dtype=float).reshape(-1)
        s = np.repeat(e[None, :], max(T, 1), axis=0)
    return _resample(s, T)


def _candidate_controls(d: dict[str, Any], T: int) -> np.ndarray:
    u = _arr(d, "prefix_controls")
    if u.ndim != 2 or u.shape[0] == 0:
        return np.zeros((T, 2), dtype=float)
    if u.shape[1] < 2:
        u = np.pad(u, ((0, 0), (0, 2 - u.shape[1])))
    return _resample(u[:, :2], T)


def _candidate_radius(d: dict[str, Any]) -> float:
    s = _arr(d, "prefix_states")
    if s.ndim == 2 and s.shape[0] and s.shape[1] >= 9:
        length = max(float(np.nanmedian(s[:, 7])), 1.0)
        width = max(float(np.nanmedian(s[:, 8])), 0.5)
    else:
        length, width = 4.8, 2.0
    return 0.5 * float(np.hypot(length, width))


def _nominal_index(samples: Sequence[dict[str, Any]]) -> int:
    for i, d in enumerate(samples):
        try:
            if float(np.asarray(d.get("is_nominal", 0.0)).item()) > 0.5:
                return i
        except Exception:
            pass
    return 0


def _empirical_upper_cvar_equal(values: np.ndarray, alpha: float) -> float:
    """Empirical upper-tail CVaR for equal-mass samples, including fractional tail."""
    x = np.sort(np.asarray(values, dtype=float).reshape(-1))[::-1]
    if x.size == 0:
        return 0.0
    alpha = float(np.clip(alpha, 1.0 / max(x.size * 1000, 1000), 1.0))
    mass = 1.0 / x.size
    remain = alpha
    total = 0.0
    for v in x:
        take = min(mass, remain)
        if take <= 0:
            break
        total += float(v) * take
        remain -= take
    return total / max(alpha, _EPS)


def _systematic_mode_indices(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0.0, None)
    if w.size == 0:
        return np.zeros((max(n, 1),), dtype=int)
    w /= max(float(w.sum()), _EPS)
    cdf = np.cumsum(w)
    u = (np.arange(max(n, 1), dtype=float) + 0.5) / max(n, 1)
    return np.minimum(np.searchsorted(cdf, u, side="left"), w.size - 1).astype(int)


@dataclass
class PortResult:
    admitted: np.ndarray
    score: np.ndarray
    fallback_score: np.ndarray
    diagnostics: dict[str, Any]


def dr_cvar_safe_halfspace_port(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any], context: ObservedRiskContext
) -> PortResult:
    """Safaoui/Summers DR-CVaR halfspaces + finite-lattice MPC projection.

    This follows the released source's DRCVaRHalfspace and MPCFilter structure.
    For the source's unconstrained-support affine loss, the halfspace QP has the
    exact translation-invariant solution

      g* = CVaR_alpha(r - h^T xi) + eps/alpha - delta.

    That closed form is algebraically equivalent to the CVXPY program and avoids
    solving A*T tiny conic programs at every Waymax replanning step.
    """
    pc = _pcfg(cfg)
    n = len(samples)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return PortResult(z.astype(bool), z, z, {})
    nominal = _nominal_index(samples)
    source_horizon = max(1, _int(cfg, "dr_cvar_horizon_steps", 10))
    cand_T = max(_arr(samples[nominal], "prefix_states").shape[0], 2)
    T = min(source_horizon + 1, cand_T)
    T = max(T, 2)
    alpha = float(pc.get("dr_cvar_alpha", 0.20))
    eps = float(pc.get("dr_cvar_wasserstein_radius", 0.05))
    delta = float(pc.get("dr_cvar_loss_bound", 0.10))
    num_samples = max(1, int(pc.get("dr_cvar_num_samples", 20)))
    q = float(pc.get("dr_cvar_mpc_Q", 2.0))
    qt = float(pc.get("dr_cvar_mpc_QT", 5.0))
    r_input = float(pc.get("dr_cvar_mpc_R", 1.0))
    tol = float(pc.get("dr_cvar_halfspace_tolerance_m", 1.0e-6))

    ref_xy = _candidate_xy(samples[nominal], T)
    mode_ids = _systematic_mode_indices(context.weights, num_samples)
    actor = context.actor_xy
    A = int(actor.shape[1]) if actor.ndim == 4 else 0
    if A == 0:
        admitted = np.asarray([bool(float(np.asarray(d.get("feasible", 1.0)).item())) for d in samples], dtype=bool)
        score = np.zeros(n, dtype=float)
        return PortResult(admitted, score, score.copy(), {"halfspace_count": 0, "min_halfspace_margin_m": np.full(n, 50.0)})

    # Resample the predictor time axis once. [H,A,T,2]
    actor_T = np.empty((actor.shape[0], A, T, 2), dtype=float)
    for h in range(actor.shape[0]):
        for a in range(A):
            actor_T[h, a] = _resample(actor[h, a], T)
    xi = actor_T[mode_ids]  # [S,A,T,2]
    p_mean = np.mean(xi, axis=0)  # source builds normal from reference obstacle trajectory

    ego_radius = _candidate_radius(samples[nominal])
    h_all = np.zeros((A, T - 1, 2), dtype=float)
    b_all = np.zeros((A, T - 1), dtype=float)
    for a in range(A):
        padding = ego_radius + float(context.actor_radius[a])
        for k, t in enumerate(range(1, T)):
            vec = p_mean[a, t] - ref_xy[t]
            norm = float(np.linalg.norm(vec))
            if norm < 1.0e-7:
                vec = p_mean[a, t] - p_mean[a, max(t - 1, 0)]
                norm = float(np.linalg.norm(vec))
            h = vec / max(norm, 1.0e-7)
            loss_samples = padding - xi[:, a, t, :] @ h
            g = _empirical_upper_cvar_equal(loss_samples, alpha) + eps * float(np.linalg.norm(h)) / max(alpha, _EPS) - delta
            h_all[a, k] = h
            b_all[a, k] = -g

    xy = np.stack([_candidate_xy(d, T) for d in samples], axis=0)  # [N,T,2]
    # margin b-h*x, shape [N,A,T-1]
    lhs = np.einsum("nkd,akd->nak", xy[:, 1:, :], h_all)
    margins = b_all[None, :, :] - lhs
    min_margin = np.min(margins, axis=(1, 2))
    feasible = np.asarray([bool(float(np.asarray(d.get("feasible", 1.0)).item())) for d in samples], dtype=bool)
    admitted = feasible & (min_margin >= -tol)

    ref_states = _candidate_states(samples[nominal], T)
    ref_u = _candidate_controls(samples[nominal], T - 1)
    obj = np.zeros(n, dtype=float)
    for i, d in enumerate(samples):
        st = _candidate_states(d, T)
        u = _candidate_controls(d, T - 1)
        # Released MPCFilter: interior state tracking, terminal tracking, input effort.
        # Project only shared [x,y,vx,vy] states when present.
        D = min(st.shape[1], ref_states.shape[1], 4)
        interior = st[1:-2, :D] - ref_states[1:-2, :D] if T > 3 else np.zeros((0, D))
        terminal = st[-1, :D] - ref_states[-1, :D]
        # Acceleration/steering have different physical units; source controls are
        # 2-D accelerations. Scale steering to an equivalent lateral-control unit.
        us = u.copy()
        if us.shape[1] >= 2:
            us[:, 1] *= float(pc.get("dr_cvar_steering_input_scale", 6.0))
        obj[i] = q * q * float(np.sum(interior * interior)) + qt * qt * float(np.sum(terminal * terminal)) + r_input * r_input * float(np.sum(us * us))
    score = -obj
    fallback = min_margin - 1.0e-6 * obj
    return PortResult(admitted, score, fallback, {
        "halfspace_count": int(A * (T - 1)),
        "min_halfspace_margin_m": min_margin,
        "source_alpha": alpha,
        "source_epsilon": eps,
        "source_delta": delta,
        "source_num_samples": num_samples,
        "source_mpc_horizon_steps": source_horizon,
    })


def _parse_intervals(cfg: dict[str, Any], H: int) -> np.ndarray:
    raw = _pcfg(cfg).get("conformal_prediction_intervals_m", None)
    if raw is None:
        raise ValueError(
            "CPSF requires per-horizon conformal_prediction_intervals_m. "
            "Run tools/calibrate_external_baselines.py on calibration_near_contact with raw WOMD."
        )
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Invalid conformal_prediction_intervals_m calibration artifact")
    if arr.size < H:
        arr = np.pad(arr, (0, H - arr.size), constant_values=float(arr[-1]))
    return np.maximum(arr[:H], 0.0)


def cpsf_constrained_projection_port(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any], context: ObservedRiskContext
) -> PortResult:
    """Strawn/Ayanian/Lindemann CPSF Eq. (7) over the executable lattice.

    The paper learns a feed-forward network as an approximate solver for Eq. (7).
    OC-RAP instead enumerates the common executable candidate set, so this port
    solves that constrained objective exactly within the lattice: minimize the
    deviation from the nominal rollout subject to the conformal distance tubes.
    """
    n = len(samples)
    nominal = _nominal_index(samples)
    H = max(1, _int(cfg, "cpsf_prediction_horizon_steps", 7))
    cand_T = max(_arr(samples[nominal], "prefix_states").shape[0], 2)
    H = min(H, max(cand_T - 1, 1))
    C = _parse_intervals(cfg, H)
    epsilon = max(_float(cfg, "cpsf_collision_margin_m", 0.5), 0.0)
    T = H + 1
    actor = context.actor_xy
    A = int(actor.shape[1]) if actor.ndim == 4 else 0
    feasible = np.asarray([bool(float(np.asarray(d.get("feasible", 1.0)).item())) for d in samples], dtype=bool)
    nominal_xy = _candidate_xy(samples[nominal], T)
    xy = np.stack([_candidate_xy(d, T) for d in samples], axis=0)
    objective = np.sum((xy[:, 1:] - nominal_xy[None, 1:]) ** 2, axis=(1, 2))
    if A == 0:
        return PortResult(feasible, -objective, -objective, {"min_conformal_margin_m": np.full(n, 50.0), "intervals_m": C})

    # Paper uses one predicted trajectory per agent. With the common predictor
    # interface, use its probability-weighted point forecast; uncertainty around
    # that point is supplied by conformal calibration rather than by the mode bank.
    actor_T = np.empty((actor.shape[0], A, T, 2), dtype=float)
    for h in range(actor.shape[0]):
        for a in range(A):
            actor_T[h, a] = _resample(actor[h, a], T)
    point = np.tensordot(np.asarray(context.weights, dtype=float), actor_T, axes=(0, 0))  # [A,T,2]
    # Eq. (7) uses epsilon itself as the minimum collision-avoidance distance:
    # ||tau_bar^j - x_hat|| >= C_h + epsilon.  Do not add benchmark vehicle
    # bounding-circle radii here; that would change the cited CPSF constraint.
    required = C[None, None, :] + epsilon
    center = np.linalg.norm(xy[:, None, 1:, :] - point[None, :, 1:, :], axis=-1)  # [N,A,H]
    margins = center - required
    min_margin = np.min(margins, axis=(1, 2))
    admitted = feasible & (min_margin >= 0.0)
    score = -objective
    fallback = min_margin - 1.0e-6 * objective
    return PortResult(admitted, score, fallback, {
        "min_conformal_margin_m": min_margin,
        "intervals_m": C,
        "prediction_horizon_steps": H,
        "collision_margin_m": epsilon,
    })


def _kinematic_series(d: dict[str, Any], cfg: dict[str, Any], T: int | None = None) -> dict[str, np.ndarray]:
    pc = _pcfg(cfg)
    dt = float(pc.get("contact_dt", pc.get("postimpact_dt", 0.1)))
    s = _arr(d, "prefix_states")
    u = _arr(d, "prefix_controls")
    if s.ndim != 2 or s.shape[0] < 2:
        e = np.asarray(d.get("ego_state", np.zeros(9)), dtype=float).reshape(-1)
        s = np.repeat(e[None, :], 2, axis=0)
    if T is not None:
        s = _resample(s, T)
        u = _resample(u, max(T - 1, 1)) if u.ndim == 2 and u.shape[0] else np.zeros((max(T - 1, 1), 2))
    n = s.shape[0]
    x, y = s[:, 0], s[:, 1]
    if s.shape[1] >= 5:
        psi = np.unwrap(s[:, 4])
    else:
        dxy = np.gradient(s[:, :2], dt, axis=0)
        psi = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if s.shape[1] >= 4:
        vxg, vyg = s[:, 2], s[:, 3]
    else:
        vxg, vyg = np.gradient(x, dt), np.gradient(y, dt)
    speed = s[:, 6] if s.shape[1] >= 7 else np.hypot(vxg, vyg)
    r = s[:, 5] if s.shape[1] >= 6 else np.gradient(psi, dt)
    axg, ayg = np.gradient(vxg, dt), np.gradient(vyg, dt)
    rdot = np.gradient(r, dt)
    # Ground -> vehicle frame.
    c, sn = np.cos(psi), np.sin(psi)
    ux = c * vxg + sn * vyg
    uy = -sn * vxg + c * vyg
    beta = np.arctan2(uy, np.maximum(np.abs(ux), 1.0e-3))
    if u.ndim != 2 or u.shape[0] == 0:
        accel = np.zeros(max(n - 1, 1)); steer = np.zeros(max(n - 1, 1))
    else:
        accel = u[:, 0] if u.shape[1] else np.zeros(u.shape[0])
        steer = u[:, 1] if u.shape[1] >= 2 else np.zeros(u.shape[0])
    return {"x": x, "y": y, "psi": psi, "vxg": vxg, "vyg": vyg, "speed": speed, "r": r, "rdot": rdot,
            "axg": axg, "ayg": ayg, "ux": ux, "uy": uy, "beta": beta, "accel": accel, "steer": steer,
            "dt": np.asarray([dt])}


def _integrated_front_obstacle(d: dict[str, Any]) -> tuple[float, float]:
    hist = _arr(d, "agent_history")
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or hist.shape[0] == 0 or hist.shape[1] <= 1:
        return float("inf"), 0.0
    last = hist[-1]
    maskv = valid[-1] if valid.ndim == 2 and valid.shape == hist.shape[:2] else np.ones(last.shape[0], dtype=bool)
    ego = np.asarray(d.get("ego_state", np.zeros(9)), dtype=float).reshape(-1)
    heading = float(ego[4]) if ego.size >= 5 else 0.0
    f = np.array([np.cos(heading), np.sin(heading)])
    l = np.array([-f[1], f[0]])
    rel = last[1:, :2] - (ego[:2] if ego.size >= 2 else 0.0)
    lon, lat = rel @ f, np.abs(rel @ l)
    mask = maskv[1:] & (lon > 0.0) & (lat < 4.5)
    if not mask.any():
        return float("inf"), 0.0
    ids = np.where(mask)[0]
    j0 = int(ids[np.argmin(lon[mask])])
    st = last[j0 + 1]
    v = float(np.hypot(st[3], st[4])) if st.size >= 5 else 0.0
    return float(lon[j0]), v


def _paper_sbd(d: dict[str, Any], cfg: dict[str, Any], kin: dict[str, np.ndarray]) -> tuple[float, bool]:
    pc = _pcfg(cfg)
    m = float(pc.get("postimpact_vehicle_mass", pc.get("vehicle_mass", 1750.0)))
    iz = float(pc.get("postimpact_vehicle_iz", pc.get("vehicle_iz", 2350.0)))
    mu = max(float(pc.get("postimpact_mu", 0.8)), 0.05)
    gap, vob = _integrated_front_obstacle(d)
    vx = float(kin["vxg"][0]); vy = float(kin["vyg"][0]); yaw = float(kin["r"][0])
    Exy = 0.5 * m * max(vx * vx + vy * vy - vob * vob, 0.0)  # Eq. 27
    Ez = 0.5 * iz * yaw * yaw                              # Eq. 28
    D = (Exy + Ez) / max(m * 9.81 * mu, _EPS)              # Eq. 30
    safe = float(pc.get("postimpact_sbd_margin", 4.0))
    Sl = float(pc.get("postimpact_rhombus_longitudinal_m", 6.0))
    # Eq.31 includes the obstacle's own distance while decelerating to same speed.
    rhs_extra = vob * max(np.sqrt(max(vx * vx + vy * vy, 0.0)) - vob, 0.0) / max(9.81 * mu, _EPS)
    brake_feasible = bool(np.isfinite(gap) and D + safe + Sl <= gap + rhs_extra)
    return float(D), brake_feasible


def _octagon_axle_feasible(Fx: np.ndarray, Fy: np.ndarray, Mz: np.ndarray, cfg: dict[str, Any], *, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Wang 2023 Eqs. (10)-(14), transformed to front/rear axle forces."""
    pc = _pcfg(cfg)
    m = float(pc.get(f"{prefix}_vehicle_mass", pc.get("vehicle_mass", 1750.0)))
    mu = max(float(pc.get(f"{prefix}_mu", pc.get("postimpact_mu", 0.8))), 0.05)
    Lf = float(pc.get(f"{prefix}_Lf_m", 1.2)); Lr = float(pc.get(f"{prefix}_Lr_m", 1.6)); L = Lf + Lr
    Fyf = (Lr * Fy + Mz) / max(L, _EPS)
    Fyr = (Lf * Fy - Mz) / max(L, _EPS)
    Fxf = Lr / max(L, _EPS) * Fx
    Fxr = Lf / max(L, _EPS) * Fx
    Fzf = m * 9.81 * Lr / max(L, _EPS)
    Fzr = m * 9.81 * Lf / max(L, _EPS)
    # Eight line inequalities of the inscribed octagon. Normalize each side by its RHS.
    rt2 = np.sqrt(2.0)
    coeff = np.asarray([[rt2 - 1, 1], [rt2 + 1, 1], [1 - rt2, 1], [-rt2 - 1, 1],
                        [1 - rt2, -1], [-rt2 - 1, -1], [rt2 - 1, -1], [rt2 + 1, -1]], dtype=float)
    rhs_scale = np.asarray([1, rt2 + 1, 1, rt2 + 1, 1, rt2 + 1, 1, rt2 + 1], dtype=float)
    use = []
    for fxa, fya, fza in [(Fxf, Fyf, Fzf), (Fxr, Fyr, Fzr)]:
        lhs = coeff[:, 0, None] * fxa[None, :] + coeff[:, 1, None] * fya[None, :]
        rhs = rhs_scale[:, None] * fza * mu
        use.append(lhs / np.maximum(rhs, _EPS))
    usage = np.max(np.concatenate(use, axis=0), axis=0)
    return usage <= 1.0 + 1.0e-9, usage


def _constant_velocity_obstacles(d: dict[str, Any], T: int, dt: float) -> np.ndarray:
    """Wang 2023 Eq. (15): observed obstacles keep lane and constant velocity."""
    hist = _arr(d, "agent_history")
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or valid.ndim != 2 or hist.shape[:2] != valid.shape or hist.shape[1] <= 1:
        return np.zeros((0, T, 2), dtype=float)
    out = []
    tt = np.arange(T, dtype=float)[:, None] * float(dt)
    for a in range(1, hist.shape[1]):
        ids = np.where(valid[:, a])[0]
        if ids.size == 0:
            continue
        st = hist[int(ids[-1]), a]
        if st.size < 5 or not np.all(np.isfinite(st[[0, 1, 3, 4]])):
            continue
        p0 = np.asarray(st[:2], dtype=float)
        vel = np.asarray(st[3:5], dtype=float)
        out.append(p0[None, :] + tt * vel[None, :])
    return np.asarray(out, dtype=float) if out else np.zeros((0, T, 2), dtype=float)


def _rhombus_margin(samples: Sequence[dict[str, Any]], cfg: dict[str, Any], T: int) -> np.ndarray:
    """Dynamic rhombus obstacle envelope, Wang 2023 Eqs. (15)-(16).

    The cited paper does *not* use a learned/multimodal obstacle predictor here:
    each obstacle keeps its current lane and constant velocity over the short
    MPC horizon.  We preserve that model exactly from the observed OC-RAP
    history and evaluate the rhombus exclusion over the common candidate set.
    """
    n = len(samples)
    if n == 0:
        return np.zeros(0, dtype=float)
    Sl = max(_float(cfg, "postimpact_rhombus_longitudinal_m", 6.0), 0.1)
    Sw = max(_float(cfg, "postimpact_rhombus_lateral_m", 2.0), 0.1)
    dt = float(_pcfg(cfg).get("postimpact_dt", _pcfg(cfg).get("contact_dt", 0.1)))
    obs = _constant_velocity_obstacles(samples[0], T, dt)
    if obs.shape[0] == 0:
        return np.full(n, 50.0)
    xy = np.stack([_candidate_xy(d, T) for d in samples])
    rel = xy[:, None, :, :] - obs[None, :, :, :]
    # Interior of each paper rhombus: |dx|/Sl + |dy|/Sw < 1. Candidate-lattice
    # evaluation implements the same exclusion geometry without the native QP's
    # preselected linear safe-access side.
    rho = np.abs(rel[..., 0]) / Sl + np.abs(rel[..., 1]) / Sw
    return np.min(rho - 1.0, axis=(1, 2))


def _ltr_proxy(kin: dict[str, np.ndarray], cfg: dict[str, Any], *, prefix: str) -> np.ndarray:
    # Paper's full LTRsim uses suspension roll states. OC-RAP has no roll-state
    # channel, so use its quasi-static small-angle lateral-load-transfer term.
    pc = _pcfg(cfg)
    h = float(pc.get(f"{prefix}_cg_height_m", 0.55))
    track = float(pc.get(f"{prefix}_track_m", 1.60))
    ay = kin["ayg"]
    return 2.0 * h * ay / max(track * 9.81, _EPS)


def _vertical_loads(m: float, Lf: float, Lr: float, track: float, h: float, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    L = Lf + Lr
    # Wang 2022/2023 quasi-static Eq. (37)/(44). [T,4]
    f1 = m*9.81*Lr/(2*L) - m*h*ax/(2*L) - m*ay*h*Lr/(track*L)
    f2 = m*9.81*Lr/(2*L) - m*h*ax/(2*L) + m*ay*h*Lr/(track*L)
    f3 = m*9.81*Lf/(2*L) + m*h*ax/(2*L) - m*ay*h*Lf/(track*L)
    f4 = m*9.81*Lf/(2*L) + m*h*ax/(2*L) + m*ay*h*Lf/(track*L)
    return np.stack([f1, f2, f3, f4], axis=-1)


def _integrated_magic_lateral(steer: np.ndarray, ux: np.ndarray, uy: np.ndarray, yaw: np.ndarray, Fz: np.ndarray, Fxw: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Simplified lateral Magic Formula + friction similarity, Wang 2023 Eqs.35-40."""
    pc = _pcfg(cfg)
    Lf = float(pc.get("postimpact_Lf_m", 1.2)); Lr = float(pc.get("postimpact_Lr_m", 1.6)); track = float(pc.get("postimpact_track_m", 1.60))
    mu = max(float(pc.get("postimpact_mu", 0.8)), 0.05); mu0 = float(pc.get("postimpact_magic_mu0", 1.0)); xi = float(pc.get("postimpact_tire_xi", 0.95))
    # Wang 2023 Table I: lateral simplified Magic Formula fitted offline.
    dy1 = float(pc.get("postimpact_magic_dy1", -6.233)); dy2 = float(pc.get("postimpact_magic_dy2", 990.2))
    cy = float(pc.get("postimpact_magic_cy", 1.466)); by = float(pc.get("postimpact_magic_by", 0.1544))
    T = len(ux)
    out = np.zeros((T, 4), dtype=float)
    for i, (sx, sy) in enumerate([(1,1),(-1,1),(1,-1),(-1,-1)]):
        vx = ux + sx * track * yaw / 2.0
        vy = uy + sy * (Lf if i < 2 else Lr) * yaw
        alpha = (steer - np.arctan2(vy, np.maximum(np.abs(vx), 0.5))) if i < 2 else -np.arctan2(vy, np.maximum(np.abs(vx), 0.5))
        # Table-I Magic-Formula coefficients are fitted with vertical load in kN
        # and slip angle in degrees (the paper's force/slip curves use those units).
        # Eq. (39) applies friction similarity before evaluating the reference-
        # friction curve: Fy0(mu,alpha)=(mu/mu0)Fymf((mu0/mu)alpha,Fz).
        fz_n = np.maximum(Fz[:, i], 1.0)
        fz_kn = fz_n / 1000.0
        alpha_ref_deg = np.degrees((mu0 / max(mu, _EPS)) * alpha)
        base = (dy1 * fz_kn * fz_kn + dy2 * fz_kn) * np.sin(cy * np.arctan(by * alpha_ref_deg))
        fy0 = (mu / max(mu0, _EPS)) * base
        ratio = Fxw[:, i] / np.maximum(mu * xi * fz_n, 1.0)
        out[:, i] = fy0 * np.sqrt(np.maximum(1.0 - ratio * ratio, 0.0))
    return out


def _pso_allocation_residual(kin: dict[str, np.ndarray], cfg: dict[str, Any]) -> tuple[float, float]:
    """Vectorized Wang 2023 PSO allocator, Eqs. (32)-(45), for Nc control knots."""
    pc = _pcfg(cfg)
    m = float(pc.get("postimpact_vehicle_mass", pc.get("vehicle_mass", 1750.0)))
    iz = float(pc.get("postimpact_vehicle_iz", pc.get("vehicle_iz", 2350.0)))
    Lf = float(pc.get("postimpact_Lf_m", 1.2)); Lr = float(pc.get("postimpact_Lr_m", 1.6)); track = float(pc.get("postimpact_track_m", 1.60)); hcg=float(pc.get("postimpact_cg_height_m",0.55))
    mu = max(float(pc.get("postimpact_mu", 0.8)), 0.05); rw=float(pc.get("postimpact_wheel_radius_m",0.33)); xi=float(pc.get("postimpact_tire_xi",0.95))
    N = max(8, int(pc.get("postimpact_pso_particles", 500))); iters=max(1,int(pc.get("postimpact_pso_iterations",8)))
    c1=float(pc.get("postimpact_pso_c1",3.0)); c2=float(pc.get("postimpact_pso_c2",3.0)); wmax=float(pc.get("postimpact_pso_wmax",0.9)); wmin=float(pc.get("postimpact_pso_wmin",0.4))
    ex=float(pc.get("postimpact_pso_eps_x",9.0)); ey=float(pc.get("postimpact_pso_eps_y",1.0)); em=float(pc.get("postimpact_pso_eps_m",10.0)); es=float(pc.get("postimpact_pso_eps_steer",1.0))
    steer_max=float(pc.get("postimpact_steer_max_rad",0.55)); wheel_torque=float(pc.get("postimpact_wheel_torque_max_nm",2500.0))
    nc=min(max(1,int(pc.get("postimpact_control_horizon_steps",3))), max(len(kin["r"])-1,1))
    ids=np.linspace(0,max(len(kin["r"])-2,0),nc).round().astype(int)
    ux=kin["ux"][ids]; uy=kin["uy"][ids]; yaw=kin["r"][ids]; ax=kin["axg"][ids]; ay=kin["ayg"][ids]; rdot=kin["rdot"][ids]
    Fxo=m*(ax - yaw*uy); Fyo=m*(ay + yaw*ux); Mo=iz*rdot
    Fz=_vertical_loads(m,Lf,Lr,track,hcg,ax,ay)
    fx_lim=np.minimum(np.maximum(mu*xi*Fz,1.0), wheel_torque/max(rw,_EPS))
    rng=np.random.default_rng(int(pc.get("postimpact_pso_seed",2023)))
    # Solve each control knot independently but vectorize particles.
    best_vals=[]; best_use=[]
    for k in range(nc):
        lo=np.concatenate([[-steer_max],-fx_lim[k]]); hi=np.concatenate([[steer_max],fx_lim[k]])
        x=rng.uniform(lo,hi,size=(N,5)); v=rng.uniform(-0.1*(hi-lo),0.1*(hi-lo),size=(N,5))
        pbest=x.copy(); pval=np.full(N,np.inf); gbest=x[0].copy(); gval=np.inf
        def fitness(xx: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
            steer=xx[:,0]; fxw=xx[:,1:]
            # Tire lateral forces from the paper tire model.
            kk=np.full(N,k,dtype=int)
            # vectorize by treating particle index as time dimension
            fz=np.repeat(Fz[k:k+1],N,axis=0)
            fyw=_integrated_magic_lateral(steer,np.full(N,ux[k]),np.full(N,uy[k]),np.full(N,yaw[k]),fz,fxw,cfg)
            cs=np.cos(steer); sn=np.sin(steer)
            Fx=(fxw[:,0]+fxw[:,1])*cs-(fyw[:,0]+fyw[:,1])*sn+fxw[:,2]+fxw[:,3]
            Fy=(fxw[:,0]+fxw[:,1])*sn+(fyw[:,0]+fyw[:,1])*cs+fyw[:,2]+fyw[:,3]
            M=-(fxw[:,0]*cs-fyw[:,0]*sn)*track/2+(fxw[:,0]*sn+fyw[:,0]*cs)*Lf+(fxw[:,1]*cs-fyw[:,1]*sn)*track/2+(fxw[:,1]*sn+fyw[:,1]*cs)*Lf-fxw[:,2]*track/2-fyw[:,2]*Lr+fxw[:,3]*track/2-fyw[:,3]*Lr
            val=ex*(Fx-Fxo[k])**2+ey*(Fy-Fyo[k])**2+em*(M-Mo[k])**2+es*steer**2
            use=np.max(np.sqrt(fxw*fxw+fyw*fyw)/np.maximum(mu*fz,1.0),axis=1)
            val=val+1.0e9*np.maximum(use-1.0,0.0)**2
            return val,use
        for it in range(iters):
            val,use=fitness(x); improved=val<pval; pbest[improved]=x[improved]; pval[improved]=val[improved]
            j=int(np.argmin(val))
            if float(val[j])<gval: gval=float(val[j]); gbest=x[j].copy()
            w=wmax-(wmax-wmin)*(it/max(iters,1))
            v=w*v+c1*rng.random((N,5))*(pbest-x)+c2*rng.random((N,5))*(gbest[None,:]-x)
            vmax=0.1*(hi-lo); v=np.clip(v,-vmax,vmax); x=np.clip(x+v,lo,hi)
        vv,uu=fitness(gbest[None,:]); best_vals.append(float(vv[0])); best_use.append(float(uu[0]))
    # Normalize dimensional squared-error objective for cross-scene comparability.
    scale = ex*(m*9.81)**2 + ey*(m*9.81)**2 + em*(iz*2.0)**2 + es
    return float(np.mean(best_vals)/max(scale,_EPS)), float(np.max(best_use))


def _pso_allocation_residual_batch(
    kin_list: Sequence[dict[str, np.ndarray]], cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Batch the Wang-2023 PSO allocator across candidates/control knots.

    This is algebraically the same wheel-force/front-steer fitness and the same
    paper PSO hyperparameters as :func:`_pso_allocation_residual`; only the
    independent candidate/knot problems are carried as a leading NumPy batch
    dimension.  It removes the dominant Python loop in closed-loop evaluation
    without reducing the paper's 500-particle/8-iteration search budget.
    """
    n = len(kin_list)
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    pc = _pcfg(cfg)
    m = float(pc.get("postimpact_vehicle_mass", pc.get("vehicle_mass", 1750.0)))
    iz = float(pc.get("postimpact_vehicle_iz", pc.get("vehicle_iz", 2350.0)))
    Lf = float(pc.get("postimpact_Lf_m", 1.2)); Lr = float(pc.get("postimpact_Lr_m", 1.6))
    track = float(pc.get("postimpact_track_m", 1.60)); hcg = float(pc.get("postimpact_cg_height_m", 0.55))
    mu = max(float(pc.get("postimpact_mu", 0.8)), 0.05); rw = float(pc.get("postimpact_wheel_radius_m", 0.33))
    xi = float(pc.get("postimpact_tire_xi", 0.95))
    N = max(8, int(pc.get("postimpact_pso_particles", 500))); iters = max(1, int(pc.get("postimpact_pso_iterations", 8)))
    c1 = float(pc.get("postimpact_pso_c1", 3.0)); c2 = float(pc.get("postimpact_pso_c2", 3.0))
    wmax = float(pc.get("postimpact_pso_wmax", 0.9)); wmin = float(pc.get("postimpact_pso_wmin", 0.4))
    ex = float(pc.get("postimpact_pso_eps_x", 9.0)); ey = float(pc.get("postimpact_pso_eps_y", 1.0))
    em = float(pc.get("postimpact_pso_eps_m", 10.0)); es = float(pc.get("postimpact_pso_eps_steer", 1.0))
    steer_max = float(pc.get("postimpact_steer_max_rad", 0.55)); wheel_torque = float(pc.get("postimpact_wheel_torque_max_nm", 2500.0))
    horizon = min(len(k["r"]) for k in kin_list)
    nc = min(max(1, int(pc.get("postimpact_control_horizon_steps", 3))), max(horizon - 1, 1))
    ids = np.linspace(0, max(horizon - 2, 0), nc).round().astype(int)

    def gather(name: str) -> np.ndarray:
        return np.stack([np.asarray(k[name], dtype=float)[ids] for k in kin_list], axis=0).reshape(-1)

    ux, uy, yaw = gather("ux"), gather("uy"), gather("r")
    ax, ay, rdot = gather("axg"), gather("ayg"), gather("rdot")
    Fxo = m * (ax - yaw * uy); Fyo = m * (ay + yaw * ux); Mo = iz * rdot
    Fz = _vertical_loads(m, Lf, Lr, track, hcg, ax, ay)
    fx_lim = np.minimum(np.maximum(mu * xi * Fz, 1.0), wheel_torque / max(rw, _EPS))
    B = Fz.shape[0]
    lo = np.concatenate([np.full((B, 1), -steer_max), -fx_lim], axis=1)
    hi = np.concatenate([np.full((B, 1), steer_max), fx_lim], axis=1)
    rng = np.random.default_rng(int(pc.get("postimpact_pso_seed", 2023)))
    x = rng.uniform(lo[:, None, :], hi[:, None, :], size=(B, N, 5))
    v = rng.uniform(-0.1 * (hi - lo)[:, None, :], 0.1 * (hi - lo)[:, None, :], size=(B, N, 5))
    pbest = x.copy(); pval = np.full((B, N), np.inf)
    gbest = x[:, 0, :].copy(); gval = np.full(B, np.inf)

    def fitness(xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = xx.shape[1]
        steer = xx[..., 0]; fxw = xx[..., 1:]
        flat_s = steer.reshape(-1); flat_fx = fxw.reshape(-1, 4)
        fyw = _integrated_magic_lateral(
            flat_s,
            np.repeat(ux, P), np.repeat(uy, P), np.repeat(yaw, P),
            np.repeat(Fz, P, axis=0), flat_fx, cfg,
        ).reshape(B, P, 4)
        cs = np.cos(steer); sn = np.sin(steer)
        Fx = (fxw[..., 0] + fxw[..., 1]) * cs - (fyw[..., 0] + fyw[..., 1]) * sn + fxw[..., 2] + fxw[..., 3]
        Fy = (fxw[..., 0] + fxw[..., 1]) * sn + (fyw[..., 0] + fyw[..., 1]) * cs + fyw[..., 2] + fyw[..., 3]
        M = (-(fxw[..., 0] * cs - fyw[..., 0] * sn) * track / 2
             + (fxw[..., 0] * sn + fyw[..., 0] * cs) * Lf
             + (fxw[..., 1] * cs - fyw[..., 1] * sn) * track / 2
             + (fxw[..., 1] * sn + fyw[..., 1] * cs) * Lf
             - fxw[..., 2] * track / 2 - fyw[..., 2] * Lr
             + fxw[..., 3] * track / 2 - fyw[..., 3] * Lr)
        val = (ex * (Fx - Fxo[:, None])**2 + ey * (Fy - Fyo[:, None])**2
               + em * (M - Mo[:, None])**2 + es * steer**2)
        use = np.max(
            np.sqrt(fxw * fxw + fyw * fyw) / np.maximum(mu * Fz[:, None, :], 1.0),
            axis=2,
        )
        val = val + 1.0e9 * np.maximum(use - 1.0, 0.0)**2
        return val, use

    rows = np.arange(B)
    for it in range(iters):
        val, _ = fitness(x)
        improved = val < pval
        pbest[improved] = x[improved]; pval[improved] = val[improved]
        j = np.argmin(val, axis=1); cand = val[rows, j]
        better = cand < gval
        gbest[better] = x[rows[better], j[better]]; gval[better] = cand[better]
        w = wmax - (wmax - wmin) * (it / max(iters, 1))
        v = (w * v
             + c1 * rng.random((B, N, 5)) * (pbest - x)
             + c2 * rng.random((B, N, 5)) * (gbest[:, None, :] - x))
        vmax = 0.1 * (hi - lo)[:, None, :]
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lo[:, None, :], hi[:, None, :])

    val, use = fitness(gbest[:, None, :])
    scale = ex * (m * 9.81)**2 + ey * (m * 9.81)**2 + em * (iz * 2.0)**2 + es
    residual = (val[:, 0] / max(scale, _EPS)).reshape(n, nc).mean(axis=1)
    usage = use[:, 0].reshape(n, nc).max(axis=1)
    return residual, usage


def integrated_postimpact_mpc_pso_port(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any]
) -> PortResult:
    """Wang et al. T-IV 2023 paper-core MPC + SBD decision + PSO allocation port."""
    n=len(samples); pc=_pcfg(cfg)
    feasible=np.asarray([bool(float(np.asarray(d.get("feasible",1.0)).item())) for d in samples],dtype=bool)
    T=max(3,min(max(_arr(d,"prefix_states").shape[0] for d in samples),int(pc.get("postimpact_prediction_horizon_steps",10))+1))
    rhomb=_rhombus_margin(samples,cfg,T)
    kin_list=[_kinematic_series(d,cfg,T) for d in samples]
    pso_res,pso_use=_pso_allocation_residual_batch(kin_list,cfg)
    costs=np.zeros(n); hard=np.zeros(n,dtype=bool); fallback=np.zeros(n)
    oct_usage=np.zeros(n); ltr_max=np.zeros(n); sbd=np.zeros(n); brake=np.zeros(n,dtype=bool)
    for i,(d,kin) in enumerate(zip(samples,kin_list)):
        m=float(pc.get("postimpact_vehicle_mass",pc.get("vehicle_mass",1750.0))); iz=float(pc.get("postimpact_vehicle_iz",pc.get("vehicle_iz",2350.0)))
        Fx=m*kin["axg"]; Fy=m*kin["ayg"]; Mz=iz*kin["rdot"]
        ok_oct,usage=_octagon_axle_feasible(Fx,Fy,Mz,cfg,prefix="postimpact"); oct_usage[i]=float(np.max(usage))
        ltr=np.abs(_ltr_proxy(kin,cfg,prefix="postimpact")); ltr_max[i]=float(np.max(ltr))
        sbd[i],brake[i]=_paper_sbd(d,cfg,kin)
        # Paper decision references: brake => v_ref=0,Y current; lane-change => v_ob,Y +/- lane width.
        macro_raw=d.get("prefix_macro_name","")
        try: macro=np.asarray(macro_raw).item(); macro=macro.decode() if isinstance(macro,bytes) else str(macro)
        except Exception: macro=str(macro_raw)
        macro=macro.lower(); is_brake=any(x in macro for x in ("brake","yield","stabilize","pull_over")); is_lane=any(x in macro for x in ("lane_shift","merge","pull_over"))
        mode_pen=0.0 if ((brake[i] and is_brake) or ((not brake[i]) and is_lane)) else float(pc.get("postimpact_decision_mismatch_penalty",10.0))
        # Paper Qy Eq.49 projected onto available outputs: Xdot,Ydot,psidot,Y,psi.
        qdiag=np.asarray(pc.get("postimpact_Qy_diag",[20,400,1500,0,20000,40000,0,0]),dtype=float)
        vref=0.0 if brake[i] else _integrated_front_obstacle(d)[1]
        y0=float(kin["y"][0]); lane=float(pc.get("postimpact_lane_width_m",4.0)); yref=y0 if brake[i] else y0+(lane if kin["y"][-1]>=y0 else -lane)
        tracking=(qdiag[0]*np.mean((kin["vxg"]-vref)**2)+qdiag[1]*np.mean(kin["vyg"]**2)+qdiag[2]*np.mean(kin["r"]**2)+qdiag[4]*np.mean((kin["y"]-yref)**2)+qdiag[5]*np.mean(kin["psi"]**2))
        u=np.column_stack([Fx[:-1],Fy[:-1],Mz[:-1]]) if T>1 else np.zeros((1,3)); du=np.diff(u,axis=0) if len(u)>1 else u
        ru=float(pc.get("postimpact_Ru_scalar",5.0e-5)); mpc_cost=tracking/max(float(np.sum(np.maximum(qdiag,0)))+1.0,_EPS)+ru*float(np.mean(du*du))/(m*m+1.0)
        # Hard paper constraints: octagon adhesion, rhombus obstacle envelope, road/lattice feasibility, LTR <= .9, allocator tire saturation.
        hard[i]=bool(feasible[i] and np.all(ok_oct) and rhomb[i]>=0.0 and ltr_max[i]<=float(pc.get("postimpact_ltr_limit",0.9)) and pso_use[i]<=1.0+float(pc.get("postimpact_pso_usage_tolerance",0.02)))
        costs[i]=mpc_cost+mode_pen+float(pc.get("postimpact_pso_residual_weight",1.0))*pso_res[i]
        fallback[i]=min(rhomb[i],1.0-oct_usage[i],float(pc.get("postimpact_ltr_limit",0.9))-ltr_max[i],1.0-pso_use[i])-1.0e-6*costs[i]
    return PortResult(hard,-costs,fallback,{"rhombus_margin":rhomb,"octagon_usage":oct_usage,"ltr_abs_max":ltr_max,"pso_residual":pso_res,"pso_tire_usage":pso_use,"sbd_m":sbd,"brake_mode":brake})



def _fit_quintic_dimension(
    values: np.ndarray,
    times: np.ndarray,
    p0: float,
    v0: float,
    *,
    terminal_value: float | None = None,
    terminal_velocity: float | None = None,
) -> np.ndarray:
    """Least-squares quintic with Wang-2022 initial/terminal equalities.

    Eq. (2) fixes a0/a1 from the observed state. Eq. (9) fixes terminal
    Y/Ydot or psi/psidot when requested. The remaining a2..a5 are the paper's
    optimization variables; projecting one executable OC-RAP trajectory into
    that affine polynomial family is a tiny equality-constrained LS problem.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    t = np.asarray(times, dtype=float).reshape(-1)
    if y.size != t.size or y.size < 2:
        return np.asarray([p0, v0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    A = np.stack([t**2, t**3, t**4, t**5], axis=-1)
    b = y - (float(p0) + float(v0) * t)
    reg = 1.0e-9
    H = A.T @ A + reg * np.eye(4)
    rhs = A.T @ b
    C_rows = []
    d_rows = []
    tf = float(t[-1])
    if terminal_value is not None:
        C_rows.append([tf**2, tf**3, tf**4, tf**5])
        d_rows.append(float(terminal_value) - float(p0) - float(v0) * tf)
    if terminal_velocity is not None:
        C_rows.append([2.0 * tf, 3.0 * tf**2, 4.0 * tf**3, 5.0 * tf**4])
        d_rows.append(float(terminal_velocity) - float(v0))
    if C_rows:
        C = np.asarray(C_rows, dtype=float)
        dvec = np.asarray(d_rows, dtype=float)
        K = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
        rr = np.concatenate([rhs, dvec])
        try:
            sol = np.linalg.solve(K, rr)[:4]
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rr, rcond=None)[0][:4]
    else:
        sol = np.linalg.solve(H, rhs)
    return np.asarray([p0, v0, *sol], dtype=float)


def _poly_eval_all(coeff: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(coeff, dtype=float).reshape(6)
    t = np.asarray(times, dtype=float)
    p = sum(c[k] * t**k for k in range(6))
    v = sum(k * c[k] * t ** (k - 1) for k in range(1, 6))
    a = sum(k * (k - 1) * c[k] * t ** (k - 2) for k in range(2, 6))
    return np.asarray(p), np.asarray(v), np.asarray(a)


def _quintic_reference_kinematics(
    d: dict[str, Any], cfg: dict[str, Any], T: int
) -> tuple[dict[str, np.ndarray], float, dict[str, np.ndarray]]:
    """Project an executable candidate into Wang-2022 Eq. (1)/(2)/(9)."""
    raw = _kinematic_series(d, cfg, T)
    dt = float(raw["dt"][0])
    times = np.arange(T, dtype=float) * dt
    if T < 2 or times[-1] <= 0:
        return raw, 0.0, {"x": np.zeros(6), "y": np.zeros(6), "psi": np.zeros(6)}

    # Eq. (2): a0/b0/c0 and a1/b1/c1 come from the current post-impact state.
    x0, y0, p0 = float(raw["x"][0]), float(raw["y"][0]), float(raw["psi"][0])
    vx0, vy0, r0 = float(raw["vxg"][0]), float(raw["vyg"][0]), float(raw["r"][0])
    # Eq. (9): terminal lateral velocity and yaw rate approach zero. The lateral
    # displacement/yaw terminal targets are supplied by the candidate library,
    # which is the benchmark's common replacement for the native fmincon search.
    cx = _fit_quintic_dimension(raw["x"], times, x0, vx0)
    cy = _fit_quintic_dimension(raw["y"], times, y0, vy0, terminal_value=float(raw["y"][-1]), terminal_velocity=0.0)
    cp = _fit_quintic_dimension(np.unwrap(raw["psi"]), times, p0, r0, terminal_value=float(np.unwrap(raw["psi"])[-1]), terminal_velocity=0.0)
    x, xd, xdd = _poly_eval_all(cx, times)
    y, yd, ydd = _poly_eval_all(cy, times)
    psi, psid, psidd = _poly_eval_all(cp, times)
    c, sn = np.cos(psi), np.sin(psi)
    ux = c * xd + sn * yd
    uy = -sn * xd + c * yd
    speed = np.hypot(xd, yd)
    beta = np.arctan2(uy, np.maximum(np.abs(ux), 1.0e-3))
    u = _candidate_controls(d, max(T - 1, 1))
    kin = {
        "x": x, "y": y, "psi": psi, "vxg": xd, "vyg": yd, "speed": speed,
        "r": psid, "rdot": psidd, "axg": xdd, "ayg": ydd, "ux": ux, "uy": uy,
        "beta": beta, "accel": u[:, 0], "steer": u[:, 1], "dt": np.asarray([dt]),
    }
    pos_err = np.sqrt((x - raw["x"])**2 + (y - raw["y"])**2)
    yaw_err = np.abs(np.unwrap(psi) - np.unwrap(raw["psi"]))
    # Meters-equivalent RMS; yaw gets a 1 m/rad interface scale only for the
    # lattice projection check, not for the paper objective itself.
    fit_rms = float(np.sqrt(np.mean(pos_err**2 + yaw_err**2)))
    return kin, fit_rms, {"x": cx, "y": cy, "psi": cp}


def _static_obstacles(d: dict[str, Any]) -> np.ndarray:
    """Observed obstacle coordinates for Wang 2022 APF Eqs. (4)-(6).

    The cited planner writes each obstacle as a fixed perceived coordinate
    (X_b, Y_b) and the experiments use fixed traffic barrels.  Do not inject
    the benchmark's learned/multimodal predictor into this baseline: that would
    change the paper's APF objective and adds unnecessary closed-loop cost.
    """
    hist = _arr(d, "agent_history")
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or valid.ndim != 2 or hist.shape[:2] != valid.shape or hist.shape[1] <= 1:
        return np.zeros((0, 2), dtype=float)
    out: list[np.ndarray] = []
    for a in range(1, hist.shape[1]):
        ids = np.where(valid[:, a])[0]
        if ids.size == 0:
            continue
        p = np.asarray(hist[int(ids[-1]), a, :2], dtype=float)
        if p.size == 2 and np.all(np.isfinite(p)):
            out.append(p)
    return np.stack(out, axis=0) if out else np.zeros((0, 2), dtype=float)


def _apf_cost_from_kin(
    samples: Sequence[dict[str, Any]], kin_list: Sequence[dict[str, np.ndarray]], cfg: dict[str, Any], T: int,
) -> tuple[np.ndarray, np.ndarray]:
    pc = _pcfg(cfg); n = len(kin_list)
    k1=float(pc.get("tvlqr_k1",1.0)); k2=float(pc.get("tvlqr_k2",1.0))
    Dr=float(pc.get("tvlqr_obstacle_safety_radius_m",1.7)); Ds=float(pc.get("tvlqr_road_safety_distance_m",1.0))
    U=np.zeros(n); V=np.zeros(n); lane_half=float(pc.get("tvlqr_road_half_width_m",4.0))
    for i,(d,kin) in enumerate(zip(samples,kin_list)):
        xy=np.stack([kin["x"],kin["y"]],axis=-1)
        obs=_static_obstacles(d)
        if obs.shape[0]:
            dist=np.linalg.norm(xy[None,:,:]-obs[:,None,:],axis=-1)
            U1=np.exp(np.clip(-(dist-Dr),-40,40)); obs_term=np.max(U1,axis=0)
        else:
            obs_term=np.zeros(T)
        U2=np.exp(np.clip(-(np.abs(kin["y"]-lane_half)-Ds),-40,40))+np.exp(np.clip(-(np.abs(kin["y"]+lane_half)-Ds),-40,40))
        U[i]=float(np.max(k1*obs_term+k2*U2))
        V[i]=float(np.mean(np.abs(np.arctan2(kin["vyg"],np.maximum(np.abs(kin["vxg"]),1e-3))-kin["psi"])))
    return U,V

def _apf_cost(samples: Sequence[dict[str, Any]], cfg: dict[str, Any], context: ObservedRiskContext, T: int) -> tuple[np.ndarray,np.ndarray]:
    pc=_pcfg(cfg); n=len(samples); k1=float(pc.get("tvlqr_k1",1.0)); k2=float(pc.get("tvlqr_k2",1.0)); Dr=float(pc.get("tvlqr_obstacle_safety_radius_m",1.7)); Ds=float(pc.get("tvlqr_road_safety_distance_m",1.0))
    if context.actor_xy.ndim==4 and context.actor_xy.shape[1]:
        actor=np.empty((context.actor_xy.shape[0],context.actor_xy.shape[1],T,2))
        for h in range(context.actor_xy.shape[0]):
            for a in range(context.actor_xy.shape[1]): actor[h,a]=_resample(context.actor_xy[h,a],T)
        obs=np.tensordot(context.weights,actor,axes=(0,0))
    else: obs=np.zeros((0,T,2))
    U=np.zeros(n); V=np.zeros(n)
    lane_half=float(pc.get("tvlqr_road_half_width_m",4.0))
    for i,d in enumerate(samples):
        kin=_kinematic_series(d,cfg,T); xy=np.stack([kin["x"],kin["y"]],axis=-1)
        if obs.shape[0]:
            dist=np.linalg.norm(xy[None,:,:]-obs,axis=-1); U1=np.exp(np.clip(-(dist-Dr),-40,40)); obs_term=np.max(U1,axis=0)
        else: obs_term=np.zeros(T)
        U2=np.exp(np.clip(-(np.abs(kin["y"]-lane_half)-Ds),-40,40))+np.exp(np.clip(-(np.abs(kin["y"]+lane_half)-Ds),-40,40))
        U[i]=float(np.max(k1*obs_term+k2*U2))
        V[i]=float(np.mean(np.abs(np.arctan2(kin["vyg"],np.maximum(np.abs(kin["vxg"]),1e-3))-kin["psi"])))
    return U,V


def _tvlqr_tracking_cost(d: dict[str, Any], cfg: dict[str, Any], T: int, kin: dict[str, np.ndarray] | None = None) -> tuple[float,float,float]:
    """Wang 2022 Eqs.18-37: local-linear TVLQR with DARE iteration at each knot."""
    pc=_pcfg(cfg); kin=_kinematic_series(d,cfg,T) if kin is None else kin; dt=float(kin["dt"][0]); m=float(pc.get("tvlqr_vehicle_mass",1610.0)); iz=float(pc.get("tvlqr_vehicle_iz",2059.0))
    Q=np.diag(np.asarray(pc.get("tvlqr_Q_diag",[5,5,90,6e5,5e5,1e6]),dtype=float)); R=np.diag(np.asarray(pc.get("tvlqr_R_diag",[1e-4,1e-4,1e-4]),dtype=float))
    # Current observed state at first candidate knot versus candidate reference.
    ego=np.asarray(d.get("ego_state",np.zeros(9)),dtype=float).reshape(-1)
    psi=float(kin["psi"][0]); c=np.cos(psi); s=np.sin(psi)
    if ego.size>=4: ux0=c*ego[2]+s*ego[3]; uy0=-s*ego[2]+c*ego[3]
    else: ux0=float(kin["ux"][0]); uy0=float(kin["uy"][0])
    r0=float(ego[5]) if ego.size>=6 else float(kin["r"][0]); x0=float(ego[0]) if ego.size else float(kin["x"][0]); y0=float(ego[1]) if ego.size>1 else float(kin["y"][0]); psi0=float(ego[4]) if ego.size>4 else psi
    err=np.asarray([ux0-kin["ux"][0],uy0-kin["uy"][0],r0-kin["r"][0],x0-kin["x"][0],y0-kin["y"][0],psi0-kin["psi"][0]],dtype=float)
    total=0.0; max_u=0.0
    B=np.zeros((6,3)); B[0,0]=1/m; B[1,1]=1/m; B[2,2]=1/iz; Bd=B*dt
    I=np.eye(6)
    # DARE fixed-point iteration per paper (Matlab dare), warm-start P across time.
    P=Q.copy()
    for k in range(T-1):
        ux=float(kin["ux"][k]); uy=float(kin["uy"][k]); rr=float(kin["r"][k]); ph=float(kin["psi"][k])
        A=np.array([[0,rr,uy,0,0,0],[-rr,0,-ux,0,0,0],[0,0,0,0,0,0],[np.cos(ph),-np.sin(ph),0,0,0,-ux*np.sin(ph)-uy*np.cos(ph)],[np.sin(ph),np.cos(ph),0,0,0,ux*np.cos(ph)-uy*np.sin(ph)],[0,0,1,0,0,0]],dtype=float)
        Ad=I+A*dt
        # solve DARE by a handful of Riccati fixed-point iterations, tiny 6x6 matrices.
        for _ in range(int(pc.get("tvlqr_dare_iterations",20))):
            G=R+Bd.T@P@Bd
            K=np.linalg.solve(G+1e-12*np.eye(3),Bd.T@P@Ad)
            Pn=Q+Ad.T@P@Ad-Ad.T@P@Bd@K
            if np.max(np.abs(Pn-P))<1e-7: P=Pn; break
            P=Pn
        K=np.linalg.solve(R+Bd.T@P@Bd+1e-12*np.eye(3),Bd.T@P@Ad)
        du=-K@err; total+=float(err@Q@err+du@R@du); max_u=max(max_u,float(np.linalg.norm(du)))
        err=Ad@err+Bd@du
    return total/max(T-1,1),max_u,float(np.linalg.norm(err))


def _tvlqr_tracking_cost_batch(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any], T: int,
    kin_list: Sequence[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched Wang-2022 TVLQR Riccati recursion across candidate references.

    Every candidate retains its own time-varying A_k, Riccati matrix P_k and
    early-convergence mask.  The computation is mathematically identical to
    `_tvlqr_tracking_cost`; NumPy only batches the independent 6x6/3x3 solves.
    """
    n = len(samples)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return z, z.copy(), z.copy()
    pc = _pcfg(cfg)
    dt = float(kin_list[0]["dt"][0])
    m = float(pc.get("tvlqr_vehicle_mass", 1610.0)); iz = float(pc.get("tvlqr_vehicle_iz", 2059.0))
    Q = np.diag(np.asarray(pc.get("tvlqr_Q_diag", [5, 5, 90, 6e5, 5e5, 1e6]), dtype=float))
    R = np.diag(np.asarray(pc.get("tvlqr_R_diag", [1e-4, 1e-4, 1e-4]), dtype=float))
    err = np.zeros((n, 6), dtype=float)
    for i, (d, kin) in enumerate(zip(samples, kin_list)):
        ego = np.asarray(d.get("ego_state", np.zeros(9)), dtype=float).reshape(-1)
        psi = float(kin["psi"][0]); c = np.cos(psi); s = np.sin(psi)
        if ego.size >= 4:
            ux0 = c * ego[2] + s * ego[3]; uy0 = -s * ego[2] + c * ego[3]
        else:
            ux0 = float(kin["ux"][0]); uy0 = float(kin["uy"][0])
        r0 = float(ego[5]) if ego.size >= 6 else float(kin["r"][0])
        x0 = float(ego[0]) if ego.size else float(kin["x"][0])
        y0 = float(ego[1]) if ego.size > 1 else float(kin["y"][0])
        psi0 = float(ego[4]) if ego.size > 4 else psi
        err[i] = [ux0 - kin["ux"][0], uy0 - kin["uy"][0], r0 - kin["r"][0],
                  x0 - kin["x"][0], y0 - kin["y"][0], psi0 - kin["psi"][0]]

    B = np.zeros((6, 3), dtype=float); B[0, 0] = 1.0 / m; B[1, 1] = 1.0 / m; B[2, 2] = 1.0 / iz
    Bd = B * dt; I = np.eye(6)
    P = np.repeat(Q[None, :, :], n, axis=0)
    total = np.zeros(n, dtype=float); max_u = np.zeros(n, dtype=float)
    dare_iters = int(pc.get("tvlqr_dare_iterations", 20))
    eye3 = np.eye(3)[None, :, :]
    rows = np.arange(n)
    for k in range(T - 1):
        ux = np.asarray([kin["ux"][k] for kin in kin_list], dtype=float)
        uy = np.asarray([kin["uy"][k] for kin in kin_list], dtype=float)
        rr = np.asarray([kin["r"][k] for kin in kin_list], dtype=float)
        ph = np.asarray([kin["psi"][k] for kin in kin_list], dtype=float)
        A = np.zeros((n, 6, 6), dtype=float)
        A[:, 0, 1] = rr; A[:, 0, 2] = uy
        A[:, 1, 0] = -rr; A[:, 1, 2] = -ux
        A[:, 3, 0] = np.cos(ph); A[:, 3, 1] = -np.sin(ph)
        A[:, 3, 5] = -ux * np.sin(ph) - uy * np.cos(ph)
        A[:, 4, 0] = np.sin(ph); A[:, 4, 1] = np.cos(ph)
        A[:, 4, 5] = ux * np.cos(ph) - uy * np.sin(ph)
        A[:, 5, 2] = 1.0
        Ad = I[None, :, :] + A * dt
        active = np.ones(n, dtype=bool)
        for _ in range(dare_iters):
            PB = P @ Bd
            G = R[None, :, :] + np.matmul(Bd.T[None, :, :], PB)
            rhs = np.matmul(Bd.T[None, :, :], P @ Ad)
            K = np.linalg.solve(G + 1.0e-12 * eye3, rhs)
            AP = np.matmul(np.swapaxes(Ad, 1, 2), P)
            Pn = Q[None, :, :] + AP @ Ad - (AP @ Bd) @ K
            diff = np.max(np.abs(Pn - P), axis=(1, 2))
            P[active] = Pn[active]
            active &= diff >= 1.0e-7
            if not active.any():
                break
        G = R[None, :, :] + np.matmul(Bd.T[None, :, :], P @ Bd)
        rhs = np.matmul(Bd.T[None, :, :], P @ Ad)
        K = np.linalg.solve(G + 1.0e-12 * eye3, rhs)
        du = -np.einsum("nij,nj->ni", K, err)
        total += np.einsum("ni,ij,nj->n", err, Q, err) + np.einsum("ni,ij,nj->n", du, R, du)
        max_u = np.maximum(max_u, np.linalg.norm(du, axis=1))
        err = np.einsum("nij,nj->ni", Ad, err) + du @ Bd.T
    return total / max(T - 1, 1), max_u, np.linalg.norm(err, axis=1)


def _tvlqr_rear_axle_constraint(d: dict[str, Any], cfg: dict[str, Any], T: int, kin: dict[str, np.ndarray] | None = None) -> tuple[bool,float]:
    pc=_pcfg(cfg); kin=_kinematic_series(d,cfg,T) if kin is None else kin; m=float(pc.get("tvlqr_vehicle_mass",1610.0)); iz=float(pc.get("tvlqr_vehicle_iz",2059.0)); Lf=float(pc.get("tvlqr_Lf_m",1.05)); Lr=float(pc.get("tvlqr_Lr_m",1.61)); mu=float(pc.get("tvlqr_mu",0.9))
    Fy=m*(-kin["axg"]*np.sin(kin["psi"])+kin["ayg"]*np.cos(kin["psi"])); M=iz*kin["rdot"]; Fyr=(Lf*Fy-M)/max(Lf+Lr,_EPS); lim=m*9.81*Lf/max(Lf+Lr,_EPS)*mu
    usage=np.abs(Fyr)/max(lim,_EPS); accel=np.sqrt(kin["axg"]**2+kin["ayg"]**2)/(max(mu,0.05)*9.81)
    mx=max(float(np.max(usage)),float(np.max(accel))); return bool(mx<=1.0+1e-9),mx


def _tvlqr_magic_lateral_force(
    alpha_rad: np.ndarray, fz_n: float | np.ndarray, mu: float, cfg: dict[str, Any]
) -> np.ndarray:
    """Wang 2022 Eqs. (47)-(49): full Magic Formula + friction similarity.

    The paper's fitted coefficients use Fz in kN and alpha in degrees.  The
    returned force is in N; combined-slip ellipse coupling is applied by the
    caller because it also depends on longitudinal wheel force.
    """
    pc = _pcfg(cfg)
    Cy=float(pc.get("tvlqr_magic_Cy",1.141)); b1=float(pc.get("tvlqr_magic_b1",-5.98)); b2=float(pc.get("tvlqr_magic_b2",965.7)); b3=float(pc.get("tvlqr_magic_b3",2536.0)); b4=float(pc.get("tvlqr_magic_b4",2.071)); b5=float(pc.get("tvlqr_magic_b5",0.04436)); b6=float(pc.get("tvlqr_magic_b6",-0.04443)); b7=float(pc.get("tvlqr_magic_b7",0.5792)); b8=float(pc.get("tvlqr_magic_b8",-3.076))
    mu0=max(float(pc.get("tvlqr_magic_mu0",1.0)), _EPS)
    fz_kn=np.maximum(np.asarray(fz_n,dtype=float),1.0)/1000.0
    alpha_ref_deg=np.degrees((mu0/max(float(mu),_EPS))*np.asarray(alpha_rad,dtype=float))
    Dy=b1*fz_kn*fz_kn+b2*fz_kn
    By=b3*np.sin(b4*np.arctan(b5*fz_kn))/np.maximum(Cy*Dy,_EPS)
    Ey=b6*fz_kn*fz_kn+b7*fz_kn+b8
    z=By*alpha_ref_deg
    pure=Dy*np.sin(Cy*np.arctan(z-Ey*(z-np.arctan(z))))
    return (float(mu)/mu0)*pure


def _nonlinear_allocation_residual(d: dict[str, Any], cfg: dict[str, Any], T: int, kin: dict[str, np.ndarray] | None = None) -> tuple[float,float]:
    """Wang 2022 Eq.38-53 objective/combined-tire constraints via vectorized projected search.

    The paper uses fmincon/SQP with warm starting. For the finite candidate port,
    the desired resultant wrench comes from TVLQR/open-loop candidate kinematics;
    a deterministic projected coordinate refinement solves the same allocation
    objective and tire ellipse constraints without a SciPy/CVX dependency.
    """
    pc=_pcfg(cfg); kin=_kinematic_series(d,cfg,T) if kin is None else kin; m=float(pc.get("tvlqr_vehicle_mass",1610.0)); iz=float(pc.get("tvlqr_vehicle_iz",2059.0)); Lf=float(pc.get("tvlqr_Lf_m",1.05)); Lr=float(pc.get("tvlqr_Lr_m",1.61)); track=float(pc.get("tvlqr_track_m",1.565)); hcg=float(pc.get("tvlqr_cg_height_m",0.55)); mu=float(pc.get("tvlqr_mu",0.9)); xi=float(pc.get("tvlqr_tire_xi",0.95))
    e1,e2,e3=[float(x) for x in pc.get("tvlqr_allocation_eps",[9,1,10])]
    ids=np.linspace(0,max(T-2,0),min(3,max(T-1,1))).round().astype(int); vals=[]; uses=[]
    rw=float(pc.get("tvlqr_wheel_radius_m",0.347)); torque=float(pc.get("tvlqr_wheel_torque_max_nm",2500.0)); steer_lim=float(pc.get("tvlqr_steer_max_rad",0.55))
    rng=np.random.default_rng(int(pc.get("tvlqr_allocation_seed",2022))); N=max(64,int(pc.get("tvlqr_allocation_candidates",256)))
    for k in ids:
        ux=float(kin["ux"][k]); uy=float(kin["uy"][k]); r=float(kin["r"][k]); ax=float(kin["axg"][k]); ay=float(kin["ayg"][k]); rd=float(kin["rdot"][k])
        Fxo=m*(ax-r*uy); Fyo=m*(ay+r*ux); Mzo=iz*rd; Fz=_vertical_loads(m,Lf,Lr,track,hcg,np.asarray([ax]),np.asarray([ay]))[0]; fxlim=np.minimum(mu*xi*np.maximum(Fz,1),torque/max(rw,_EPS))
        steer=rng.uniform(-steer_lim,steer_lim,N); fx=rng.uniform(-fxlim,fxlim,size=(N,4));
        def evalv(st:np.ndarray,fxw:np.ndarray):
            fy=np.zeros_like(fxw)
            for wi,(sx,sy) in enumerate([(1,1),(-1,1),(1,-1),(-1,-1)]):
                vx=ux+sx*track*r/2; vy=uy+sy*(Lf if wi<2 else Lr)*r; al=(st-np.arctan2(vy,max(abs(vx),0.5))) if wi<2 else -np.arctan2(vy,max(abs(vx),0.5)); fz=max(Fz[wi],1.0)
                fy0=_tvlqr_magic_lateral_force(al,fz,mu,cfg)
                fy[:,wi]=fy0*np.sqrt(np.maximum(1-(fxw[:,wi]/max(mu*xi*fz,1.0))**2,0.0))
            cs=np.cos(st); sn=np.sin(st); Fx=(fxw[:,0]+fxw[:,1])*cs-(fy[:,0]+fy[:,1])*sn+fxw[:,2]+fxw[:,3]; Fy=(fxw[:,0]+fxw[:,1])*sn+(fy[:,0]+fy[:,1])*cs+fy[:,2]+fy[:,3]; M=-(fxw[:,0]*cs-fy[:,0]*sn)*track/2+(fxw[:,0]*sn+fy[:,0]*cs)*Lf+(fxw[:,1]*cs-fy[:,1]*sn)*track/2+(fxw[:,1]*sn+fy[:,1]*cs)*Lf-fxw[:,2]*track/2-fy[:,2]*Lr+fxw[:,3]*track/2-fy[:,3]*Lr; val=e1*(Fxo-Fx)**2+e2*(Fyo-Fy)**2+e3*(Mzo-M)**2; use=np.max(np.sqrt(fxw*fxw+fy*fy)/np.maximum(mu*Fz[None,:],1.0),axis=1); val+=1e9*np.maximum(use-1,0)**2; return val,use
        val,use=evalv(steer,fx); j=int(np.argmin(val)); best_s=float(steer[j]); best_fx=fx[j].copy(); best=float(val[j]); best_u=float(use[j])
        # Warm-style local projected refinement around current best.
        span_s=steer_lim; span_f=fxlim.copy()
        for _ in range(int(pc.get("tvlqr_allocation_refine_iterations",6))):
            span_s*=0.45; span_f*=0.45; st=np.clip(best_s+rng.normal(0,span_s,size=N),-steer_lim,steer_lim); f=np.clip(best_fx[None,:]+rng.normal(0,span_f,size=(N,4)),-fxlim,fxlim); v,u=evalv(st,f); jj=int(np.argmin(v));
            if float(v[jj])<best: best=float(v[jj]); best_s=float(st[jj]); best_fx=f[jj].copy(); best_u=float(u[jj])
        scale=e1*(m*9.81)**2+e2*(m*9.81)**2+e3*(iz*2)**2; vals.append(best/max(scale,_EPS)); uses.append(best_u)
    return float(np.mean(vals)),float(np.max(uses))


def _nonlinear_allocation_residual_batch(
    kin_list: Sequence[dict[str, np.ndarray]], cfg: dict[str, Any], T: int
) -> tuple[np.ndarray, np.ndarray]:
    """Batch Wang-2022 nonlinear tire allocation over candidates/control knots.

    The paper uses SQP/fmincon.  Our disclosed finite-lattice adapter keeps the
    same desired-wrench residual, full Magic Formula, friction-similarity law,
    and combined-tire ellipse used by the scalar projected-search port, while
    evaluating all independent candidate/knot searches in one vectorized batch.
    """
    n = len(kin_list)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return z, z.copy()
    pc = _pcfg(cfg)
    m=float(pc.get("tvlqr_vehicle_mass",1610.0)); iz=float(pc.get("tvlqr_vehicle_iz",2059.0))
    Lf=float(pc.get("tvlqr_Lf_m",1.05)); Lr=float(pc.get("tvlqr_Lr_m",1.61)); track=float(pc.get("tvlqr_track_m",1.565))
    hcg=float(pc.get("tvlqr_cg_height_m",0.55)); mu=float(pc.get("tvlqr_mu",0.9)); xi=float(pc.get("tvlqr_tire_xi",0.95))
    e1,e2,e3=[float(x) for x in pc.get("tvlqr_allocation_eps",[9,1,10])]
    ids=np.linspace(0,max(T-2,0),min(3,max(T-1,1))).round().astype(int); nc=len(ids)
    rw=float(pc.get("tvlqr_wheel_radius_m",0.347)); torque=float(pc.get("tvlqr_wheel_torque_max_nm",2500.0)); steer_lim=float(pc.get("tvlqr_steer_max_rad",0.55))
    N=max(64,int(pc.get("tvlqr_allocation_candidates",256))); refine=max(0,int(pc.get("tvlqr_allocation_refine_iterations",6)))

    def gather(name: str) -> np.ndarray:
        return np.stack([np.asarray(k[name],dtype=float)[ids] for k in kin_list],axis=0).reshape(-1)

    ux,uy,r=gather("ux"),gather("uy"),gather("r")
    ax,ay,rd=gather("axg"),gather("ayg"),gather("rdot")
    Fxo=m*(ax-r*uy); Fyo=m*(ay+r*ux); Mzo=iz*rd
    Fz=_vertical_loads(m,Lf,Lr,track,hcg,ax,ay)
    fxlim=np.minimum(mu*xi*np.maximum(Fz,1.0),torque/max(rw,_EPS))
    B=Fz.shape[0]
    rng=np.random.default_rng(int(pc.get("tvlqr_allocation_seed",2022)))
    steer=rng.uniform(-steer_lim,steer_lim,size=(B,N))
    fx=rng.uniform(-fxlim[:,None,:],fxlim[:,None,:],size=(B,N,4))

    def evalv(st: np.ndarray, fxw: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        fy=np.zeros_like(fxw)
        for wi,(sx,sy) in enumerate([(1,1),(-1,1),(1,-1),(-1,-1)]):
            vx=ux+sx*track*r/2.0
            vy=uy+sy*(Lf if wi<2 else Lr)*r
            slip=np.arctan2(vy,np.maximum(np.abs(vx),0.5))
            al=st-slip[:,None] if wi<2 else -slip[:,None]
            fz=np.maximum(Fz[:,wi],1.0)[:,None]
            fy0=_tvlqr_magic_lateral_force(al,fz,mu,cfg)
            fy[...,wi]=fy0*np.sqrt(np.maximum(1.0-(fxw[...,wi]/np.maximum(mu*xi*fz,1.0))**2,0.0))
        cs=np.cos(st); sn=np.sin(st)
        Fx=(fxw[...,0]+fxw[...,1])*cs-(fy[...,0]+fy[...,1])*sn+fxw[...,2]+fxw[...,3]
        Fy=(fxw[...,0]+fxw[...,1])*sn+(fy[...,0]+fy[...,1])*cs+fy[...,2]+fy[...,3]
        M=(-(fxw[...,0]*cs-fy[...,0]*sn)*track/2
           +(fxw[...,0]*sn+fy[...,0]*cs)*Lf
           +(fxw[...,1]*cs-fy[...,1]*sn)*track/2
           +(fxw[...,1]*sn+fy[...,1]*cs)*Lf
           -fxw[...,2]*track/2-fy[...,2]*Lr+fxw[...,3]*track/2-fy[...,3]*Lr)
        val=e1*(Fxo[:,None]-Fx)**2+e2*(Fyo[:,None]-Fy)**2+e3*(Mzo[:,None]-M)**2
        use=np.max(np.sqrt(fxw*fxw+fy*fy)/np.maximum(mu*Fz[:,None,:],1.0),axis=2)
        val+=1e9*np.maximum(use-1.0,0.0)**2
        return val,use

    rows=np.arange(B)
    val,use=evalv(steer,fx); j=np.argmin(val,axis=1)
    best_s=steer[rows,j].copy(); best_fx=fx[rows,j].copy(); best=val[rows,j].copy(); best_u=use[rows,j].copy()
    span_s=steer_lim; span_f=fxlim.copy()
    for _ in range(refine):
        span_s*=0.45; span_f*=0.45
        st=np.clip(best_s[:,None]+rng.normal(size=(B,N))*span_s,-steer_lim,steer_lim)
        f=np.clip(best_fx[:,None,:]+rng.normal(size=(B,N,4))*span_f[:,None,:],-fxlim[:,None,:],fxlim[:,None,:])
        v,u=evalv(st,f); jj=np.argmin(v,axis=1); cand=v[rows,jj]; better=cand<best
        best[better]=cand[better]; best_s[better]=st[rows[better],jj[better]]; best_fx[better]=f[rows[better],jj[better]]; best_u[better]=u[rows[better],jj[better]]
    scale=e1*(m*9.81)**2+e2*(m*9.81)**2+e3*(iz*2.0)**2
    return (best/max(scale,_EPS)).reshape(n,nc).mean(axis=1), best_u.reshape(n,nc).max(axis=1)


def postimpact_motion_tvlqr_port(
    samples: Sequence[dict[str, Any]], cfg: dict[str, Any]
) -> PortResult:
    """Wang et al. CJME 2022 quintic/APF planner + TVLQR + nonlinear allocation."""
    n=len(samples); pc=_pcfg(cfg)
    feasible=np.asarray([bool(float(np.asarray(d.get("feasible",1.0)).item())) for d in samples],dtype=bool)
    T=max(3,min(max(_arr(d,"prefix_states").shape[0] for d in samples),int(pc.get("tvlqr_planning_horizon_steps",37))))
    kin_list=[]; fit=np.zeros(n)
    for i,d in enumerate(samples):
        kin, fit_i, _ = _quintic_reference_kinematics(d,cfg,T)
        kin_list.append(kin)
        fit[i]=fit_i
    U,V=_apf_cost_from_kin(samples,kin_list,cfg,T)
    k3=float(pc.get("tvlqr_k3",1.0)); k4=float(pc.get("tvlqr_k4",0.9))
    costs=np.zeros(n); dyn_use=np.zeros(n); hard=np.zeros(n,dtype=bool); fallback=np.zeros(n)
    track,_,terminal_err=_tvlqr_tracking_cost_batch(samples,cfg,T,kin_list)
    alloc,alloc_use=_nonlinear_allocation_residual_batch(kin_list,cfg,T)
    fit_tol=float(pc.get("tvlqr_quintic_fit_tolerance_m",1.5)); fit_w=float(pc.get("tvlqr_quintic_fit_weight",1.0))
    for i,(d,kin) in enumerate(zip(samples,kin_list)):
        ok,dyn_use[i]=_tvlqr_rear_axle_constraint(d,cfg,T,kin=kin)
        terminal=float(pc.get("tvlqr_terminal_velocity_weight",1.0))*(kin["vyg"][-1]**2+kin["r"][-1]**2)
        costs[i]=k3*U[i]+k4*V[i]+fit_w*fit[i]+float(pc.get("tvlqr_tracking_cost_weight",1e-6))*track[i]+float(pc.get("tvlqr_allocation_residual_weight",1.0))*alloc[i]+terminal
        hard[i]=bool(feasible[i] and ok and fit[i]<=fit_tol and alloc_use[i]<=1.0+float(pc.get("tvlqr_allocation_usage_tolerance",0.02)))
        fallback[i]=min(1.0-dyn_use[i],1.0-alloc_use[i],fit_tol-fit[i])-1e-9*costs[i]
    return PortResult(hard,-costs,fallback,{
        "apf_U":U,"stability_V":V,"quintic_fit_rms":fit,"dynamics_usage":dyn_use,
        "tvlqr_cost":track,"allocation_residual":alloc,"allocation_tire_usage":alloc_use,
        "terminal_error":terminal_err,
    })
