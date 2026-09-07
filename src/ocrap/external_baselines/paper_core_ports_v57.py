from __future__ import annotations

"""Additional audited finite-candidate ports for post-impact baselines (v57).

These ports deliberately separate three things:
  1) mechanisms that are explicitly present in the cited source,
  2) the common OC-RAP executable-candidate projection, and
  3) quantities unavailable in WOMD/Waymax (driver pedal intent, wheel faults,
     impact-force history, wheel slip/torque states).

No function in this file consumes OC-RAP teacher labels or learned future-agent
predictions.  Contact metrics are computed by the common evaluator after the
selected executable candidate is rolled out.
"""

from functools import lru_cache
from itertools import product
from typing import Any, Sequence

import numpy as np

from .paper_core_ports_v56 import (
    PortResult,
    _candidate_controls,
    _candidate_states,
    _nominal_index,
    _pcfg,
)

_EPS = 1.0e-9
_G = 9.81


def _f(cfg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(_pcfg(cfg).get(key, default))
    except Exception:
        return float(default)


def _i(cfg: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(_pcfg(cfg).get(key, default))
    except Exception:
        return int(default)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _runtime_elapsed_s(sample: dict[str, Any]) -> float:
    for key in ("runtime_contact_elapsed_s", "contact_elapsed_s"):
        if key in sample:
            try:
                return max(0.0, float(np.asarray(sample[key]).item()))
            except Exception:
                pass
    return 0.0


def _feasible(samples: Sequence[dict[str, Any]]) -> np.ndarray:
    out = []
    for d in samples:
        try:
            out.append(float(np.asarray(d.get("feasible", 1.0)).item()) > 0.5)
        except Exception:
            out.append(True)
    return np.asarray(out, dtype=bool)


def _batch(samples: Sequence[dict[str, Any]], *, T_state: int | None = None, T_control: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.zeros((0, 1, 9), dtype=float), np.zeros((0, 1, 2), dtype=float)
    ni = _nominal_index(samples)
    s0 = np.asarray(samples[ni].get("prefix_states", np.zeros((0, 9))), dtype=float)
    u0 = np.asarray(samples[ni].get("prefix_controls", np.zeros((0, 2))), dtype=float)
    if T_state is None:
        T_state = max(int(s0.shape[0]) if s0.ndim == 2 else 0, 2)
    if T_control is None:
        T_control = max(int(u0.shape[0]) if u0.ndim == 2 else 0, max(int(T_state) - 1, 1))
    st = np.stack([_candidate_states(d, int(T_state)) for d in samples], axis=0)
    uu = np.stack([_candidate_controls(d, int(T_control)) for d in samples], axis=0)
    return np.nan_to_num(st), np.nan_to_num(uu)


def _observed_preimpact_reference(sample0: dict[str, Any], st: np.ndarray) -> tuple[float, float, float, str]:
    """Estimate the original straight driving path from *observed* ego history.

    OC-RAP histories are expressed in the current-ego frame.  For post-impact
    controllers this is useful: a pre-impact path appears as a line that need
    not pass through or align with the current post-impact pose.  We fit only
    the older observed portion of the SDC history, never future/teacher states.
    """
    ego = np.asarray(sample0.get("ego_state", np.zeros(9)), dtype=float).reshape(-1)
    x_cur = float(ego[0]) if ego.size >= 1 else float(st[0, 0, 0])
    y_cur = float(ego[1]) if ego.size >= 2 else float(st[0, 0, 1])
    psi_cur = float(ego[4]) if ego.size >= 5 else (float(st[0, 0, 4]) if st.shape[-1] > 4 else 0.0)

    hist = np.asarray(sample0.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(sample0.get("agent_valid", np.zeros((0, 0), dtype=bool)), dtype=bool)
    if hist.ndim != 3 or hist.shape[0] < 2 or hist.shape[1] < 1 or hist.shape[2] < 8:
        return x_cur, y_cur, psi_cur, "current_pose_fallback"
    if valid.shape[:2] != hist.shape[:2]:
        vmask = np.ones(hist.shape[0], dtype=bool)
    else:
        vmask = valid[:, 0].astype(bool)
    idx = np.where(vmask)[0]
    if idx.size < 2:
        return x_cur, y_cur, psi_cur, "current_pose_fallback"

    # Prefer the older 2/3 of the observed history so an impact near the current
    # decision does not rotate/translate the reference toward the disturbed pose.
    keep_n = max(2, int(np.ceil(2.0 * idx.size / 3.0)))
    idx = idx[:keep_n]
    pts = hist[idx, 0, :2]
    # Chronological least-squares velocity gives a signed reference direction.
    tt = np.arange(idx.size, dtype=float)
    tc = tt - np.mean(tt)
    pc = pts - np.mean(pts, axis=0, keepdims=True)
    denom = float(np.dot(tc, tc))
    direction = np.sum(pc * tc[:, None], axis=0) / max(denom, _EPS)
    if float(np.linalg.norm(direction)) < 0.05 and hist.shape[2] >= 5:
        direction = np.nanmedian(hist[idx, 0, 3:5], axis=0)
    if float(np.linalg.norm(direction)) < 0.05:
        headings = hist[idx, 0, 7]
        direction = np.array([np.mean(np.cos(headings)), np.mean(np.sin(headings))], dtype=float)
    if float(np.linalg.norm(direction)) < 1.0e-6:
        return x_cur, y_cur, psi_cur, "current_pose_fallback"
    psi_ref = float(np.arctan2(direction[1], direction[0]))

    # Fit a line with that direction through the older observed points.  The
    # reference origin need not equal the current position, thereby retaining
    # the post-impact lateral displacement Y used by Cao/Ao controllers.
    tangent = np.array([np.cos(psi_ref), np.sin(psi_ref)], dtype=float)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    mean_pt = np.nanmean(pts, axis=0)
    lateral_offset = float(np.nanmedian(pts @ normal))
    origin = mean_pt + (lateral_offset - float(mean_pt @ normal)) * normal
    return float(origin[0]), float(origin[1]), psi_ref, "observed_ego_history_line"


def _path_frame(st: np.ndarray, sample0: dict[str, Any]) -> dict[str, np.ndarray | float | str]:
    """Express candidates in the observed pre-impact straight-path frame."""
    x0, y0, psi0, ref_source = _observed_preimpact_reference(sample0, st)
    c, s = float(np.cos(psi0)), float(np.sin(psi0))
    dx = st[..., 0] - x0
    dy = st[..., 1] - y0
    x = c * dx + s * dy
    y = -s * dx + c * dy
    if st.shape[-1] >= 5:
        psi = _wrap(st[..., 4] - psi0)
    else:
        psi = np.zeros_like(x)
    if st.shape[-1] >= 6:
        r = st[..., 5]
    elif st.shape[1] >= 2:
        r = np.gradient(psi, axis=1)
    else:
        r = np.zeros_like(x)
    if st.shape[-1] >= 4:
        vxw, vyw = st[..., 2], st[..., 3]
        vx = c * vxw + s * vyw
        vy = -s * vxw + c * vyw
    else:
        vx = np.gradient(x, axis=1)
        vy = np.gradient(y, axis=1)
    if st.shape[-1] >= 7:
        speed = np.maximum(st[..., 6], 0.0)
    else:
        speed = np.hypot(vx, vy)
    return {
        "x": x, "y": y, "psi": psi, "r": r, "vx": vx, "vy": vy,
        "speed": speed, "reference_heading_rad": psi0,
        "reference_origin_x": x0, "reference_origin_y": y0,
        "reference_source": ref_source,
    }


def post_crash_braking_port(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> PortResult:
    """Lu et al. 2017 post-impact braking (PIB) branch on the candidate lattice.

    Source-faithful mechanism: after an impact, if there is no driver braking
    request, command autonomous braking up to ABS capability.  WOMD has no brake
    pedal/intention channel, therefore the benchmark explicitly evaluates the
    autonomous no-driver PIB branch, not PIBA.  Detailed stop/timeout values are
    adapter defaults motivated by related inventor patents and are recorded in
    diagnostics; they are not attributed to the SAE abstract.
    """
    n = len(samples)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return PortResult(z.astype(bool), z, z, {})
    st, u = _batch(samples)
    feas = _feasible(samples)
    dt = _f(cfg, "contact_dt", 0.1)
    mu = _f(cfg, "pib_abs_mu", _f(cfg, "contact_mu", 0.75))
    abs_scale = _f(cfg, "pib_abs_scale", 1.0)
    a_abs = max(0.1, min(_f(cfg, "pib_max_decel_mps2", 9.0), abs_scale * mu * _G))
    stop_speed = _f(cfg, "pib_motion_stop_threshold_mps", 8.0 / 3.6)
    max_duration = _f(cfg, "pib_max_active_duration_s", 2.5)
    min_brake_fraction = np.clip(_f(cfg, "pib_min_abs_fraction", 0.55), 0.0, 1.0)
    elapsed = _runtime_elapsed_s(samples[0])

    frame = _path_frame(st, samples[0])
    v0 = float(frame["speed"][0, 0])
    active = bool(v0 > stop_speed and elapsed < max_duration)
    T = u.shape[1]
    a = u[..., 0]
    steer = u[..., 1] if u.shape[-1] >= 2 else np.zeros_like(a)

    if active:
        # Avoid commanding negative speed: emulate ABS-limited deceleration until
        # the motion threshold, then release the brake.
        tref = np.arange(T, dtype=float) * dt
        vref = np.maximum(v0 - a_abs * tref, 0.0)
        a_ref = np.where(vref > stop_speed, -a_abs, 0.0)
    else:
        a_ref = np.zeros((T,), dtype=float)

    accel_fit = np.mean(((a - a_ref[None, :]) / max(a_abs, 1.0)) ** 2, axis=1)
    steer_cost = np.mean((steer / max(_f(cfg, "pib_steer_scale_rad", 0.25), 1.0e-3)) ** 2, axis=1)
    speed_target = max(stop_speed if active else v0, 0.0)
    terminal_speed = frame["speed"][:, -1]
    speed_cost = ((terminal_speed - speed_target) / max(v0, 2.0)) ** 2
    positive_drive = np.mean(np.maximum(a, 0.0) ** 2, axis=1) / max(a_abs * a_abs, 1.0)
    cost = accel_fit + _f(cfg, "pib_steer_tiebreak_weight", 0.02) * steer_cost + 0.15 * speed_cost + 2.0 * positive_drive

    if active:
        mean_decel = np.mean(np.maximum(-a, 0.0), axis=1)
        admitted = feas & (mean_decel >= min_brake_fraction * a_abs) & (np.max(a, axis=1) <= _f(cfg, "pib_positive_accel_tolerance_mps2", 0.25))
    else:
        admitted = feas & (np.mean(np.abs(a), axis=1) <= _f(cfg, "pib_release_accel_tolerance_mps2", 0.8))

    score = -cost
    fallback = -cost
    return PortResult(admitted, score, fallback, {
        "pib_branch": "autonomous_no_driver_brake",
        "pib_active": active,
        "elapsed_s": elapsed,
        "abs_deceleration_target_mps2": a_abs,
        "motion_stop_threshold_mps": stop_speed,
        "max_active_duration_s": max_duration,
        "terminal_speed_mps": terminal_speed,
    })


def _window_sine(t: np.ndarray, amp: float, lo: float, hi: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    if hi <= lo:
        return out
    mask = (t >= lo) & (t <= hi)
    out[mask] = float(amp) * np.sin(np.pi * (t[mask] - lo) / (hi - lo))
    return out


def post_collision_restoration_port(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> PortResult:
    """Ghosh et al. 2026 open-loop steering/tractive-force controller.

    Implements Eqs. (10), (13), (16), (18), (21) with paper-time (seconds), not
    a horizon-rescaled pulse.  The paper's Table 2 reports ``K1`` although the
    controller equations require ``A2`` and never define ``K1``; therefore v57
    does *not* silently reinterpret that table entry.  A2 defaults to zero unless
    the user supplies ``restoration_A2`` after source/author clarification.  The
    benchmark normally finishes before the second pulse starts at 10 s.
    """
    n = len(samples)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return PortResult(z.astype(bool), z, z, {})
    st, u = _batch(samples)
    feas = _feasible(samples)
    frame = _path_frame(st, samples[0])
    dt = _f(cfg, "contact_dt", 0.1)
    elapsed = _runtime_elapsed_s(samples[0])
    T = u.shape[1]
    t = elapsed + np.arange(T, dtype=float) * dt

    r0 = float(frame["r"][0, 0])
    vy0 = float(frame["vy"][0, 0])
    case2_threshold = _f(cfg, "restoration_case2_yaw_rate_threshold_radps", 0.10)
    case = 2 if abs(r0) >= case2_threshold else 1
    if case == 1:
        tau0, tau1 = 1.0, 3.0
        kdir_mag = 0.2
        table_k1 = -1.91
    else:
        tau0, tau1 = 1.0, 5.195
        kdir_mag = 0.5
        table_k1 = -1.4665
    tau0 = _f(cfg, f"restoration_case{case}_tau0_s", tau0)
    tau1 = _f(cfg, f"restoration_case{case}_tau1_s", tau1)
    tau2 = _f(cfg, "restoration_tau2_s", 10.0)
    tau3 = _f(cfg, "restoration_tau3_s", 11.0)
    tc1 = _f(cfg, "restoration_tauc1_s", 5.443)
    tc2 = _f(cfg, "restoration_tauc2_s", 10.0)
    A1 = _f(cfg, "restoration_A1", 0.175)
    # Paper Table 2 does not report A2.  Keep the omission explicit.
    A2 = _f(cfg, "restoration_A2", 0.0)
    Ac_N = _f(cfg, "restoration_Ac_N", 900.0)
    mass = _f(cfg, "restoration_vehicle_mass_kg", 1750.0)

    sign_src = vy0 if abs(vy0) > 1.0e-4 else r0
    mirror = 1.0 if sign_src >= 0.0 else -1.0
    kdir = -mirror * _f(cfg, f"restoration_case{case}_Kdir_abs", kdir_mag)
    steer_ref = kdir * (_window_sine(t, A1, tau0, tau1) + _window_sine(t, A2, tau2, tau3))
    fc = _window_sine(t, Ac_N, tc1, tc2)

    nominal = _nominal_index(samples)
    nominal_a = u[nominal, :, 0]
    # Eq. (18): F_xt = F_i + F_c.  WOMD has no wheel tractive-force state, so
    # the shared nominal longitudinal acceleration is the explicit Fi/m lattice
    # projection and only the paper's incremental Fc/m is imposed.
    accel_ref = nominal_a + fc / max(mass, 1.0)

    steer = u[..., 1]
    accel = u[..., 0]
    steer_scale = max(abs(A1 * kdir), _f(cfg, "restoration_steer_scale_floor_rad", 0.02))
    accel_scale = max(Ac_N / max(mass, 1.0), 0.25)
    steer_fit = np.mean(((steer - steer_ref[None, :]) / steer_scale) ** 2, axis=1)
    accel_fit = np.mean(((accel - accel_ref[None, :]) / accel_scale) ** 2, axis=1)
    cost = steer_fit + _f(cfg, "restoration_force_fit_weight", 0.25) * accel_fit

    # Paper explicitly says recovery is achieved without braking.  Small negative
    # candidate accelerations are tolerated because OC-RAP controls are vehicle
    # acceleration rather than wheel tractive force and include road/load effects.
    brake_tol = _f(cfg, "restoration_brake_tolerance_mps2", 0.75)
    admitted = feas & (np.min(accel, axis=1) >= -brake_tol)
    score = -cost
    return PortResult(admitted, score, score.copy(), {
        "source_case": case,
        "elapsed_s": elapsed,
        "source_open_loop": True,
        "source_A2_unreported": True,
        "source_table_unexplained_K1": table_k1,
        "steer_reference": steer_ref,
        "additional_tractive_force_N": fc,
        "lattice_accel_reference_mps2": accel_ref,
    })


def compensatory_postimpact_mpc_port(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> PortResult:
    """Source-limited structured port of Cao et al. 2021 FCC-MPC.

    Publicly accessible primary/transport-index metadata establishes FCC,
    reverse steering + differential torque vectoring, constraint transformation,
    and time-varying saturation on input/input-rate/slip-ratio.  The paywalled
    full equations/weights are not available from reliable online sources, so
    this function intentionally does *not* claim an equation-exact reproduction.
    It preserves those mechanisms as an explicit finite-lattice constraint and
    tracking adapter without adding OC-RAP predictor risk to the paper objective.
    """
    n = len(samples)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return PortResult(z.astype(bool), z, z, {})
    st, u = _batch(samples)
    feas = _feasible(samples)
    frame = _path_frame(st, samples[0])
    dt = _f(cfg, "contact_dt", 0.1)
    T = min(st.shape[1], u.shape[1])
    y = frame["y"][:, :T]
    psi = frame["psi"][:, :T]
    r = frame["r"][:, :T]
    vx = frame["vx"][:, :T]
    vy = frame["vy"][:, :T]
    speed = np.maximum(frame["speed"][:, :T], 0.5)
    beta = np.arctan2(vy, np.maximum(np.abs(vx), 0.5))
    a = u[:, :T, 0]
    steer = u[:, :T, 1]

    # FCC qualitative mechanism: reverse steering and differential yaw action
    # should oppose the collision-induced lateral/yaw state immediately.
    e0 = float(y[0, 0] + _f(cfg, "comp_mpc_fcc_yaw_mix_s", 0.8) * r[0, 0])
    if abs(e0) < 1.0e-5:
        e0 = float(vy[0, 0] + 0.5 * r[0, 0])
    desired_sign = -np.sign(e0) if abs(e0) > 1.0e-6 else 0.0
    first = max(1, min(T, _i(cfg, "comp_mpc_fcc_initial_steps", 5)))
    wrong_reverse = np.maximum(0.0, -desired_sign * steer[:, :first]) if desired_sign else np.zeros((n, first))
    rdot = np.gradient(r, dt, axis=1) if T >= 2 else np.zeros_like(r)
    wrong_yaw_comp = np.maximum(0.0, np.sign(r[:, :first]) * rdot[:, :first])

    # Constraint transformation: initial post-impact state may start outside the
    # nominal stability envelope.  Instead of softening the constraint, create a
    # contracting envelope that starts at the observed violation and converges to
    # the nominal envelope.  This is an explicit benchmark projection of the
    # mechanism, not a claim about Cao et al.'s unpublished transform equation.
    t = np.arange(T, dtype=float) * dt
    decay = np.exp(-_f(cfg, "comp_mpc_constraint_decay_rate", 1.25) * t)
    beta_base = _f(cfg, "comp_mpc_beta_limit_rad", 0.12)
    yaw_base = _f(cfg, "comp_mpc_yaw_rate_limit_radps", 0.80)
    beta_extra = max(0.0, abs(float(beta[0, 0])) - beta_base)
    yaw_extra = max(0.0, abs(float(r[0, 0])) - yaw_base)
    beta_bound = beta_base + beta_extra * decay
    yaw_bound = yaw_base + yaw_extra * decay

    steer_max = _f(cfg, "comp_mpc_steer_max_rad", 0.55)
    steer_rate_max = _f(cfg, "comp_mpc_steer_rate_max_radps", 1.5)
    slip_limit = _f(cfg, "comp_mpc_slip_ratio_limit", 0.20)
    mu = _f(cfg, "comp_mpc_mu", 0.85)
    # No wheel-speed/slip-ratio state in WOMD.  |a_x|/(mu g) is documented as
    # a conservative longitudinal utilisation proxy, not a fabricated slip state.
    slip_proxy = np.abs(a) / max(mu * _G, 1.0e-3)
    steer_rate = np.abs(np.diff(steer, axis=1, prepend=steer[:, :1])) / max(dt, 1.0e-3)
    constraint_excess = (
        np.maximum(np.abs(beta) - beta_bound[None, :], 0.0) ** 2
        + np.maximum(np.abs(r) - yaw_bound[None, :], 0.0) ** 2
        + np.maximum(np.abs(steer) - steer_max, 0.0) ** 2
        + np.maximum(steer_rate - steer_rate_max, 0.0) ** 2
        + np.maximum(slip_proxy - slip_limit, 0.0) ** 2
    )
    hard_ok = np.max(constraint_excess, axis=1) <= _f(cfg, "comp_mpc_constraint_tolerance", 1.0e-8)

    # Tracking part of MPC: fast attenuation of lateral and yaw deviations.
    wy = _f(cfg, "comp_mpc_tracking_y_weight", 1.0)
    wpsi = _f(cfg, "comp_mpc_tracking_heading_weight", 0.7)
    wr = _f(cfg, "comp_mpc_tracking_yaw_rate_weight", 1.0)
    wb = _f(cfg, "comp_mpc_tracking_sideslip_weight", 0.8)
    tracking = np.mean(wy * y * y + wpsi * psi * psi + wr * r * r + wb * beta * beta, axis=1)
    control = np.mean(0.04 * steer * steer + 0.002 * a * a, axis=1)
    fcc = np.mean(wrong_reverse * wrong_reverse + 0.25 * wrong_yaw_comp * wrong_yaw_comp, axis=1)
    cost = tracking + control + _f(cfg, "comp_mpc_fcc_weight", 1.0) * fcc + 1.0e3 * np.mean(constraint_excess, axis=1)
    admitted = feas & hard_ok
    return PortResult(admitted, -cost, -cost, {
        "fidelity": "source_limited_abstract_structured",
        "requires_full_paper_for_equation_exact_port": True,
        "desired_reverse_steering_sign": desired_sign,
        "beta_bound_rad": beta_bound,
        "yaw_rate_bound_radps": yaw_bound,
        "max_constraint_excess": np.max(constraint_excess, axis=1),
        "terminal_speed_mps": speed[:, -1],
        "reference_source": frame["reference_source"],
        "reference_heading_rad": frame["reference_heading_rad"],
    })


@lru_cache(maxsize=16)
def _active_statuses(dim: int = 4) -> tuple[tuple[int, ...], ...]:
    # -1 = lower bound, 0 = free, +1 = upper bound
    return tuple(product((-1, 0, 1), repeat=int(dim)))


@lru_cache(maxsize=32)
def _cached_box_qp_plan(
    h_key: tuple[float, ...], w_key: tuple[float, ...], xi: float, bound_key: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]]:
    """Pre-factorize all 3^4 active sets for the fixed four-wheel source QP."""
    H = np.asarray(h_key, dtype=float).reshape(2, 4)
    wdiag = np.asarray(w_key, dtype=float).reshape(4)
    bound = np.asarray(bound_key, dtype=float).reshape(4)
    Q = np.diag(wdiag) + float(xi) * (H.T @ H)
    plans = []
    for status in _active_statuses(4):
        status_a = np.asarray(status, dtype=int)
        fixed = np.where(status_a != 0)[0]
        free = np.where(status_a == 0)[0]
        fixed_values = status_a[fixed].astype(float) * bound[fixed] if fixed.size else np.zeros(0, dtype=float)
        if free.size:
            Qff = Q[np.ix_(free, free)]
            try:
                inv = np.linalg.inv(Qff)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(Qff)
            bias = fixed_values @ Q[np.ix_(fixed, free)] if fixed.size else np.zeros(free.size, dtype=float)
        else:
            inv = np.zeros((0, 0), dtype=float)
            bias = np.zeros(0, dtype=float)
        plans.append((fixed, free, fixed_values, inv, bias))
    return H, Q, tuple(plans)


def _solve_box_qp_batch(v: np.ndarray, H: np.ndarray, wdiag: np.ndarray, xi: float, bound: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact four-wheel box QP with cached active-set factorizations.

    min_u u^T W u + xi ||v-Hu||^2,  -bound <= u <= bound.
    The four actuator dimensions yield only 3^4=81 faces. Matrix inverses for
    those faces depend only on source/controller parameters, so they are cached
    across every candidate and every Waymax replan. No solver tolerance or
    objective is changed by this acceleration.
    """
    v = np.asarray(v, dtype=float).reshape(-1, 2)
    H0 = np.asarray(H, dtype=float).reshape(2, 4)
    w0 = np.maximum(np.asarray(wdiag, dtype=float).reshape(4), 1.0e-12)
    bnd = np.maximum(np.asarray(bound, dtype=float).reshape(4), 1.0e-9)
    # Rounded immutable keys prevent insignificant float-representation noise
    # from defeating cache reuse while remaining far below controller precision.
    h_key = tuple(np.round(H0.reshape(-1), 14).tolist())
    w_key = tuple(np.round(w0, 18).tolist())
    b_key = tuple(np.round(bnd, 12).tolist())
    Hc, _Q, plans = _cached_box_qp_plan(h_key, w_key, float(xi), b_key)
    rhs_all = float(xi) * (v @ Hc)  # [B,4] = xi H^T v per row
    best_u = np.zeros((v.shape[0], 4), dtype=float)
    best_obj = np.full(v.shape[0], np.inf, dtype=float)
    tol = 1.0e-8

    for fixed, free, fixed_values, inv, bias in plans:
        u = np.zeros((v.shape[0], 4), dtype=float)
        if fixed.size:
            u[:, fixed] = fixed_values[None, :]
        if free.size:
            sol = (rhs_all[:, free] - bias[None, :]) @ inv.T
            u[:, free] = sol
            valid = np.all(np.abs(sol) <= bnd[free][None, :] + tol, axis=1)
        else:
            valid = np.ones(v.shape[0], dtype=bool)
        if not valid.any():
            continue
        alloc = u @ Hc.T
        obj = np.sum((u * u) * w0[None, :], axis=1) + float(xi) * np.sum((v - alloc) ** 2, axis=1)
        improve = valid & (obj < best_obj)
        best_obj[improve] = obj[improve]
        best_u[improve] = u[improve]
    return best_u, best_obj


def robust_postimpact_control_port(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> PortResult:
    """Ao et al. 2022 sliding-mode + fault-tolerant QP candidate port.

    Implements the published sliding surface (17-19), reaching law (22), virtual
    yaw moment (23), motor fault model (26-27), utilisation objective (28-29),
    force/yaw mapping (30-31), torque bounds (32) and quadratic allocation
    (34).  The V2V impact impulse itself is not replayed because OC-RAP starts
    from observed post-impact states and WOMD does not expose impact forces/point.
    Once the impact interval has ended, those force terms are zero by design.
    """
    n = len(samples)
    if n == 0:
        z = np.zeros(0, dtype=float)
        return PortResult(z.astype(bool), z, z, {})
    st, u = _batch(samples)
    feas = _feasible(samples)
    frame = _path_frame(st, samples[0])
    dt = _f(cfg, "contact_dt", 0.1)
    T = min(st.shape[1], u.shape[1])
    y = frame["y"][:, :T]
    psi = frame["psi"][:, :T]
    r = frame["r"][:, :T]
    vx = frame["vx"][:, :T]
    vy = frame["vy"][:, :T]
    a_cmd = u[:, :T, 0]
    steer = u[:, :T, 1]

    # Source Table 1 / Table 2.
    m = _f(cfg, "robust_pic_vehicle_mass_kg", 1270.0)
    Iz = _f(cfg, "robust_pic_vehicle_iz_kgm2", 1536.7)
    af = _f(cfg, "robust_pic_a_m", 1.015)
    br = _f(cfg, "robust_pic_b_m", 1.895)
    wheel_r = _f(cfg, "robust_pic_wheel_radius_m", 0.325)
    track = _f(cfg, "robust_pic_track_m", 1.675)
    Kf = _f(cfg, "robust_pic_cornering_front_Nprad", 50000.0)
    Kr = _f(cfg, "robust_pic_cornering_rear_Nprad", 65000.0)
    mu = _f(cfg, "robust_pic_mu", 0.85)
    c1 = _f(cfg, "robust_pic_c1", 0.6)
    c2 = _f(cfg, "robust_pic_c2", 1.0)
    k1 = _f(cfg, "robust_pic_k1", 0.25)
    k2 = _f(cfg, "robust_pic_k2", 0.005)
    xi = _f(cfg, "robust_pic_qp_xi", 0.5)
    boundary = max(_f(cfg, "robust_pic_boundary_layer", 0.02), 1.0e-5)

    # Eq. (19), zero desired yaw rate and lateral deviation for the straight
    # post-impact reference path used in the paper.
    s = c1 * r + c2 * y

    # Approximate U(t) after the 0.2 s impact interval with the source's own
    # nominal cornering-stiffness bicycle terms. Impact-force terms are zero.
    vx_safe = np.where(np.abs(vx) >= 0.5, vx, np.sign(vx + _EPS) * 0.5)
    alpha_f = steer - (vy + af * r) / vx_safe
    alpha_r = -(vy - br * r) / vx_safe
    Fy_f = Kf * alpha_f
    Fy_r = Kr * alpha_r
    Uterm = c1 / max(Iz, 1.0) * (af * Fy_f - br * Fy_r)
    # Published Eq. (21) R(t)=c2(xdot*sin(phi)-ydot*cos(phi)).
    Rterm = c2 * (vx * np.sin(psi) - vy * np.cos(psi))
    sat_s = np.clip(s / boundary, -1.0, 1.0)
    delta_mz = Iz * (-k1 * s - k2 * sat_s - Uterm - Rterm)

    # Eq. (31) and (34).  Dataset has no diagnosed motor fault, so healthy
    # gains are the benchmark default. If an external fault vector is supplied,
    # the exact same QP uses it; we never invent a random failure.
    raw_fault = _pcfg(cfg).get("robust_pic_fault_factors", [0.0, 0.0, 0.0, 0.0])
    fault = np.clip(np.asarray(raw_fault, dtype=float).reshape(-1), 0.0, 1.0)
    if fault.size < 4:
        fault = np.pad(fault, (0, 4 - fault.size))
    fault = fault[:4]
    khat = 1.0 - fault
    signs = np.asarray([-1.0, 1.0, -1.0, 1.0])
    H = np.vstack((khat / wheel_r, signs * khat * track / (2.0 * wheel_r)))
    Fz = np.full(4, m * _G / 4.0, dtype=float)
    wdiag = (khat / np.maximum(mu * wheel_r * Fz, 1.0)) ** 2
    torque_max = _f(cfg, "robust_pic_wheel_torque_max_nm", 1600.0)
    bounds = np.full(4, torque_max, dtype=float)
    Fx = m * a_cmd
    vreq = np.stack((Fx, delta_mz), axis=-1).reshape(-1, 2)
    torque, qp_obj = _solve_box_qp_batch(vreq, H, wdiag, xi, bounds)
    achieved = torque @ H.T
    alloc_res = np.linalg.norm(vreq - achieved, axis=1).reshape(n, T)
    qp_obj = qp_obj.reshape(n, T)

    # Source verification focuses on lateral deviation, course angle, yaw rate,
    # sideslip, while longitudinal velocity is kept nearly unchanged. Use those
    # source quantities for candidate ranking; allocation residual is the hard
    # actuator-realizability term.
    beta = np.arctan2(vy, np.maximum(np.abs(vx), 0.5))
    tracking = np.sqrt(np.mean(y * y, axis=1)) + np.sqrt(np.mean(psi * psi, axis=1)) + np.sqrt(np.mean(r * r, axis=1)) + np.sqrt(np.mean(beta * beta, axis=1))
    v0 = np.maximum(frame["speed"][:, 0], 1.0)
    speed_loss = np.abs(frame["speed"][:, min(T - 1, frame["speed"].shape[1] - 1)] - v0) / v0
    res_scale = max(_f(cfg, "robust_pic_allocation_residual_scale", 300.0), 1.0)
    alloc_penalty = np.mean((alloc_res / res_scale) ** 2, axis=1)
    cost = tracking + _f(cfg, "robust_pic_speed_preservation_weight", 0.25) * speed_loss + _f(cfg, "robust_pic_allocation_weight", 1.0) * alloc_penalty + 1.0e-7 * np.mean(qp_obj, axis=1)

    residual_tol = _f(cfg, "robust_pic_allocation_residual_tolerance", 1200.0)
    admitted = feas & (np.max(alloc_res, axis=1) <= residual_tol)
    return PortResult(admitted, -cost, -cost, {
        "sliding_surface": s,
        "requested_delta_mz_Nm": delta_mz,
        "allocation_residual": alloc_res,
        "max_allocation_residual": np.max(alloc_res, axis=1),
        "fault_factors": fault,
        "qp_exact_active_set_enumeration": True,
        "impact_force_term": "zero_after_observed_impact_interval",
        "reference_source": frame["reference_source"],
        "reference_heading_rad": frame["reference_heading_rad"],
    })
