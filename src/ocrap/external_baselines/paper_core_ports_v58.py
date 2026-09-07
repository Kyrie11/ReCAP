from __future__ import annotations

"""Additional source-backed finite-candidate ports introduced in v58.

The primary implementation in this file is Parseh/Nybacka/Asplund's unavoidable-
collision severity planner.  The cited method chooses a *pre-impact* trajectory by
predicting collision impulse and post-impact target-vehicle motion, then minimizing
Eq. (25).  OC-RAP executes a common finite prefix lattice, so the native offline
optimal-control trajectory library is replaced only at that outer decision layer.

No OC-RAP teacher label, future root, or learned actor predictor is consumed here.
"""

from typing import Any, Sequence

import numpy as np

from .paper_core_ports_v56 import PortResult, _candidate_states, _nominal_index, _pcfg

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


def _wrap_pi_periodic(a: np.ndarray) -> np.ndarray:
    """Distance to the nearest multiple of pi, in [-pi/2, pi/2]."""
    x = np.asarray(a, dtype=float)
    return (x + 0.5 * np.pi) % np.pi - 0.5 * np.pi


def _observed_target(samples: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick one struck-target proxy from observation only.

    Parseh et al. assume the target trajectory is known.  The serialized OC-RAP
    samples intentionally do not store other-agent future trajectories.  We use
    the nearest currently observed non-SDC actor and a constant-velocity future;
    this limitation is surfaced in diagnostics/provenance.
    """
    if not samples:
        return None
    d = samples[0]
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or valid.ndim != 2 or hist.shape[:2] != valid.shape or hist.shape[1] <= 1:
        return None
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=float).reshape(-1)
    ego_xy = ego[:2] if ego.size >= 2 else np.zeros(2, dtype=float)
    rows: list[tuple[float, int, np.ndarray, float]] = []
    for a in range(1, hist.shape[1]):
        idx = np.where(valid[:, a])[0]
        if not idx.size:
            continue
        j = int(idx[-1])
        s = np.asarray(hist[j, a], dtype=float)
        if s.size < 5 or not np.isfinite(s[:5]).all():
            continue
        prev_heading = float(s[7]) if s.size > 7 else 0.0
        yaw_rate = 0.0
        if idx.size >= 2:
            j0 = int(idx[-2]); s0 = np.asarray(hist[j0, a], dtype=float)
            if s0.size > 7 and np.isfinite(s0[7]) and np.isfinite(prev_heading):
                dh = (prev_heading - float(s0[7]) + np.pi) % (2.0 * np.pi) - np.pi
                yaw_rate = dh / 0.1
        rows.append((float(np.linalg.norm(s[:2] - ego_xy)), a, s, yaw_rate))
    if not rows:
        return None
    rows.sort(key=lambda z: (z[0], z[1]))
    _dist, idx, s, yaw_rate = rows[0]
    return {
        "actor_index": int(idx),
        "position": np.asarray(s[:2], dtype=float),
        "velocity": np.asarray(s[3:5], dtype=float),
        "heading": float(s[7]) if s.size > 7 else float(np.arctan2(s[4], s[3] + _EPS)),
        "yaw_rate": float(yaw_rate),
        "length": float(s[10]) if s.size > 10 and np.isfinite(s[10]) and s[10] > 0 else 4.8,
        "width": float(s[11]) if s.size > 11 and np.isfinite(s[11]) and s[11] > 0 else 2.0,
    }


def _oriented_rect_corners(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    """Return a counter-clockwise rectangle polygon."""
    f = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
    l = np.asarray([-f[1], f[0]], dtype=float)
    hl, hw = 0.5 * float(length), 0.5 * float(width)
    return np.stack([
        center + hl * f + hw * l,
        center - hl * f + hw * l,
        center - hl * f - hw * l,
        center + hl * f - hw * l,
    ], axis=0)


def _convex_clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping for counter-clockwise convex polygons."""
    out = np.asarray(subject, dtype=float)
    clipper = np.asarray(clipper, dtype=float)
    if out.shape[0] == 0:
        return out
    for i in range(clipper.shape[0]):
        a = clipper[i]; b = clipper[(i + 1) % clipper.shape[0]]
        edge = b - a
        inp = out
        if inp.shape[0] == 0:
            break
        pieces: list[np.ndarray] = []
        def inside(p: np.ndarray) -> bool:
            q = p - a
            return float(edge[0] * q[1] - edge[1] * q[0]) >= -1.0e-10
        def intersect(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            d = q - p
            den = edge[0] * d[1] - edge[1] * d[0]
            if abs(float(den)) < 1.0e-12:
                return 0.5 * (p + q)
            ap = a - p
            t = (edge[0] * ap[1] - edge[1] * ap[0]) / den
            return p + float(t) * d
        prev = inp[-1]; prev_in = inside(prev)
        for cur in inp:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    pieces.append(intersect(prev, cur))
                pieces.append(cur)
            elif prev_in:
                pieces.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        out = np.asarray(pieces, dtype=float).reshape(-1, 2) if pieces else np.zeros((0, 2), dtype=float)
    return out


def _polygon_centroid(poly: np.ndarray) -> np.ndarray | None:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area2 = float(cross.sum())
    if abs(area2) < 1.0e-10:
        return poly.mean(axis=0)
    c = np.asarray([np.sum((x + xn) * cross), np.sum((y + yn) * cross)], dtype=float) / (3.0 * area2)
    return c if np.isfinite(c).all() else poly.mean(axis=0)


def _sat_collision_geometry(
    states: np.ndarray,
    target_xy: np.ndarray,
    target_heading: np.ndarray,
    target_length: float,
    target_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized SAT broad phase plus source Sutherland-Hodgman contact geometry.

    SAT cheaply finds each candidate's first rectangle overlap.  Only at that
    first impact do we execute the paper's Sutherland-Hodgman polygon clipping,
    use the overlap centroid as POI, and derive the contact-plane tangent from
    generated overlap-intersection vertices.  Degenerate corner/tangent contacts
    fall back to the minimum-penetration SAT axis.
    """
    N, T, _ = states.shape
    ego_xy = states[..., :2]
    yaw = states[..., 4] if states.shape[-1] > 4 else np.zeros((N, T), dtype=float)
    ego_len = np.maximum(states[..., 7] if states.shape[-1] > 7 else 4.8, 1.0)
    ego_wid = np.maximum(states[..., 8] if states.shape[-1] > 8 else 2.0, 0.5)
    ef = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    el = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
    th = np.broadcast_to(target_heading[None, :], (N, T))
    tf = np.stack([np.cos(th), np.sin(th)], axis=-1)
    tl = np.stack([-np.sin(th), np.cos(th)], axis=-1)
    axes = np.stack([ef, el, tf, tl], axis=2)  # [N,T,4,2]
    delta = target_xy[None, :, :] - ego_xy
    proj = np.abs(np.einsum("ntd,ntkd->ntk", delta, axes))
    er = 0.5 * ego_len[..., None] * np.abs(np.einsum("ntd,ntkd->ntk", ef, axes)) + 0.5 * ego_wid[..., None] * np.abs(np.einsum("ntd,ntkd->ntk", el, axes))
    tr = 0.5 * float(target_length) * np.abs(np.einsum("ntd,ntkd->ntk", tf, axes)) + 0.5 * float(target_width) * np.abs(np.einsum("ntd,ntkd->ntk", tl, axes))
    penetration = er + tr - proj
    collision = np.all(penetration >= 0.0, axis=-1)
    has = collision.any(axis=1)
    first = np.where(has, np.argmax(collision, axis=1), 0).astype(int)
    rows = np.arange(N)
    pen_first = penetration[rows, first]
    axis_idx = np.argmin(pen_first, axis=-1)
    normal = axes[rows, first, axis_idx]
    delta_first = delta[rows, first]
    sign = np.where(np.einsum("nd,nd->n", delta_first, normal) >= 0.0, 1.0, -1.0)
    normal = normal * sign[:, None]  # ego -> target
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=-1)

    ce = ego_xy[rows, first]
    ct = target_xy[first]
    efe = ef[rows, first]; ele = el[rows, first]
    tfe = tf[rows, first]; tle = tl[rows, first]
    le = ego_len[rows, first]; we = ego_wid[rows, first]
    re = 0.5 * le * np.abs(np.einsum("nd,nd->n", efe, normal)) + 0.5 * we * np.abs(np.einsum("nd,nd->n", ele, normal))
    rt = 0.5 * float(target_length) * np.abs(np.einsum("nd,nd->n", tfe, normal)) + 0.5 * float(target_width) * np.abs(np.einsum("nd,nd->n", tle, normal))
    support_e = ce + normal * re[:, None]
    support_t = ct - normal * rt[:, None]
    poi = 0.5 * (support_e + support_t)
    # Source paper: POI is the centroid of the Sutherland-Hodgman overlap
    # polygon.  SAT remains the vectorized broad phase and contact-normal
    # projection; exact clipping is paid only once per colliding candidate.
    for i in np.where(has)[0].tolist():
        ego_poly = _oriented_rect_corners(ce[i], float(yaw[i, first[i]]), float(le[i]), float(we[i]))
        tgt_poly = _oriented_rect_corners(ct[i], float(target_heading[first[i]]), float(target_length), float(target_width))
        overlap = _convex_clip_polygon(ego_poly, tgt_poly)
        centroid = _polygon_centroid(overlap)
        if centroid is not None:
            poi[i] = centroid
        # Parseh Sec. 3.2.4: identify overlap-polygon vertices that belong to
        # neither vehicle rectangle, fit the line through those intersection
        # vertices, then translate it through the POI.  That line is the CP and
        # is parallel to the tangential t-axis.  Use the farthest generated pair
        # when more than two vertices are available; degenerate corner contacts
        # fall back to the SAT minimum-penetration axis computed above.
        if overlap.shape[0] >= 2:
            d_ego = np.min(np.linalg.norm(overlap[:, None, :] - ego_poly[None, :, :], axis=-1), axis=1)
            d_tgt = np.min(np.linalg.norm(overlap[:, None, :] - tgt_poly[None, :, :], axis=-1), axis=1)
            generated = overlap[(d_ego > 1.0e-6) & (d_tgt > 1.0e-6)]
            if generated.shape[0] >= 2:
                dd = np.linalg.norm(generated[:, None, :] - generated[None, :, :], axis=-1)
                a, b = np.unravel_index(int(np.argmax(dd)), dd.shape)
                tv = generated[b] - generated[a]
                tn = float(np.linalg.norm(tv))
                if tn > 1.0e-9:
                    tv = tv / tn
                    nv = np.asarray([-tv[1], tv[0]], dtype=float)
                    if float(np.dot(ct[i] - ce[i], nv)) < 0.0:
                        nv = -nv
                    tangent[i] = tv
                    normal[i] = nv
    return has, first, normal, np.concatenate([tangent, poi], axis=-1)


def _collision_impulse(
    ego_state: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_heading: float,
    target_yaw_rate: float,
    normal: np.ndarray,
    tangent: np.ndarray,
    poi: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kudlich-Slibar-style full/sliding impulse in planar rigid-body form."""
    N = ego_state.shape[0]
    m1 = _f(cfg, "severity_target_mass_kg", 2150.0)
    m2 = _f(cfg, "severity_ego_mass_kg", 2150.0)
    I1 = _f(cfg, "severity_target_izz_kgm2", 3075.0)
    I2 = _f(cfg, "severity_ego_izz_kgm2", 3075.0)
    mu = _f(cfg, "severity_contact_mu", 0.4)
    v2 = ego_state[:, 2:4] if ego_state.shape[1] >= 4 else np.zeros((N, 2), dtype=float)
    w2 = ego_state[:, 5] if ego_state.shape[1] > 5 else np.zeros((N,), dtype=float)
    v1 = np.broadcast_to(np.asarray(target_vel, dtype=float), (N, 2)).copy()
    w1 = np.full((N,), float(target_yaw_rate), dtype=float)
    r1 = poi - target_pos
    r2 = poi - ego_state[:, :2]

    def perp(r: np.ndarray) -> np.ndarray:
        return np.stack([-r[:, 1], r[:, 0]], axis=-1)

    pr1 = perp(r1); pr2 = perp(r2)
    # Contact-point velocities v + omega x r = v + omega * perp(r)
    c1 = v1 + w1[:, None] * pr1
    c2 = v2 + w2[:, None] * pr2
    rel = c1 - c2
    B = np.stack([tangent, normal], axis=-1)  # [N,2,2], columns t,n
    rel_tn = np.einsum("nji,nj->ni", B, rel)

    # Generalized inverse-mass matrix at the POI.
    eye = np.eye(2, dtype=float)[None]
    Kg = (1.0 / m1 + 1.0 / m2) * eye + np.einsum("ni,nj->nij", pr1, pr1) / I1 + np.einsum("ni,nj->nij", pr2, pr2) / I2
    K = np.einsum("nji,njk,nkl->nil", B, Kg, B)
    # Compression impulse for sticking/full impact.
    det = K[:, 0, 0] * K[:, 1, 1] - K[:, 0, 1] * K[:, 1, 0]
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    invK = np.empty_like(K)
    invK[:, 0, 0] = K[:, 1, 1] / det
    invK[:, 1, 1] = K[:, 0, 0] / det
    invK[:, 0, 1] = -K[:, 0, 1] / det
    invK[:, 1, 0] = -K[:, 1, 0] / det
    jc = -np.einsum("nij,nj->ni", invK, rel_tn)
    Tc = jc[:, 0]; Nc = jc[:, 1]
    approaching = rel_tn[:, 1] < 0.0
    full = approaching & (Nc > 0.0) & (np.abs(Tc) <= mu * np.maximum(Nc, 0.0))

    # Sliding branch: T_c = -mu sign(v_rel,t) N_c and normal relative
    # velocity is zero at the end of compression, matching Eqs. (23-24).
    sg = np.where(rel_tn[:, 0] >= 0.0, 1.0, -1.0)
    den = K[:, 1, 1] - mu * sg * K[:, 1, 0]
    den = np.where(np.abs(den) < 1e-9, np.sign(den + 1e-12) * 1e-9, den)
    Ns = -rel_tn[:, 1] / den
    Ts = -mu * sg * Ns
    Tc_use = np.where(full, Tc, Ts)
    Nc_use = np.where(full, Nc, Ns)
    Nc_use = np.where(approaching, np.maximum(Nc_use, 0.0), 0.0)
    Tc_use = np.where(approaching, Tc_use, 0.0)

    # Eq. (24).  Its source speed convention is scenario-specific; use the
    # normal POI closing speed, the directly corresponding deployable quantity.
    closing = np.maximum(-rel_tn[:, 1], 1.0e-3)
    e = np.clip(2.5 / closing, 0.0, _f(cfg, "severity_restitution_max", 1.0))
    if "severity_restitution" in _pcfg(cfg):
        e[:] = np.clip(_f(cfg, "severity_restitution", 0.1), 0.0, 1.0)
    Jtn = np.stack([Tc_use * (1.0 + e), Nc_use * (1.0 + e)], axis=-1)
    J = np.einsum("nij,nj->ni", B, Jtn)
    v1_post = v1 + J / m1
    v2_post = v2 - J / m2
    cross1 = r1[:, 0] * J[:, 1] - r1[:, 1] * J[:, 0]
    cross2 = r2[:, 0] * J[:, 1] - r2[:, 1] * J[:, 0]
    w1_post = w1 + cross1 / I1
    w2_post = w2 - cross2 / I2
    return v1_post, w1_post, full.astype(bool), np.linalg.norm(J, axis=-1)


def _target_postimpact_rollout_cost(
    pos0: np.ndarray,
    vel0: np.ndarray,
    heading0: np.ndarray,
    yaw_rate0: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Source-parameter 3DOF post-impact rollout and Eq. (25) target cost."""
    N = pos0.shape[0]
    dt = _f(cfg, "severity_postimpact_dt", 0.02)
    steps = max(2, _i(cfg, "severity_postimpact_steps", int(round(2.0 / max(dt, 1e-3)))))
    m = _f(cfg, "severity_target_mass_kg", 2150.0)
    Izz = _f(cfg, "severity_target_izz_kgm2", 3075.0)
    lf = _f(cfg, "severity_lf_m", 1.4)
    lr = _f(cfg, "severity_lr_m", 1.45)
    Cf = _f(cfg, "severity_front_stiffness_factor", 22.0)
    Cr = _f(cfg, "severity_rear_stiffness_factor", 22.0)
    mu_s = _f(cfg, "severity_road_mu", 0.76)
    fr = _f(cfg, "severity_rolling_resistance", 0.013)

    h0 = np.asarray(heading0, dtype=float)
    f0 = np.stack([np.cos(h0), np.sin(h0)], axis=-1)
    l0 = np.stack([-np.sin(h0), np.cos(h0)], axis=-1)
    vx = np.einsum("nd,nd->n", vel0, f0)
    vy = np.einsum("nd,nd->n", vel0, l0)
    psi = h0.copy(); r = np.asarray(yaw_rate0, dtype=float).copy()
    X = pos0[:, 0].copy(); Y = pos0[:, 1].copy()
    ref_pos = pos0.copy(); ref_heading = h0.copy()

    Yc = np.zeros(N, dtype=float); psic = np.zeros(N, dtype=float)
    omegac = np.zeros(N, dtype=float); betac = np.zeros(N, dtype=float)
    Fzf_each = m * _G * lr / max(lf + lr, 1e-6) / 2.0
    Fzr_each = m * _G * lf / max(lf + lr, 1e-6) / 2.0
    for _ in range(steps):
        safe_vx = np.where(np.abs(vx) < 0.5, np.where(vx >= 0.0, 0.5, -0.5), vx)
        alpha_f = np.arctan2(vy + lf * r, safe_vx)
        alpha_r = np.arctan2(vy - lr * r, safe_vx)
        # Eq. (6), two tires per axle, zero commanded longitudinal tire force.
        Fyf = 2.0 * (-np.sin(np.arctan(Cf * alpha_f)) * (mu_s * Fzf_each))
        Fyr = 2.0 * (-np.sin(np.arctan(Cr * alpha_r)) * (mu_s * Fzr_each))
        Fx = -fr * m * _G * np.sign(vx)
        vxd = vy * r + Fx / m
        vyd = -vx * r + (Fyf + Fyr) / m
        rd = (lf * Fyf - lr * Fyr) / Izz
        Xd = vx * np.cos(psi) - vy * np.sin(psi)
        Yd = vx * np.sin(psi) + vy * np.cos(psi)
        vx += dt * vxd; vy += dt * vyd; r += dt * rd
        X += dt * Xd; Y += dt * Yd; psi += dt * r

        rel = np.stack([X - ref_pos[:, 0], Y - ref_pos[:, 1]], axis=-1)
        lat = np.einsum("nd,nd->n", rel, l0)
        psi_err = _wrap_pi_periodic(psi - ref_heading)
        beta = np.arctan2(vy, vx)
        beta_err = _wrap_pi_periodic(beta)
        Yc += lat * lat
        psic += psi_err * psi_err
        omegac += r * r
        betac += beta_err * beta_err

    w1 = _f(cfg, "severity_w1", 1.0); w2 = _f(cfg, "severity_w2", 1.0)
    w3 = _f(cfg, "severity_w3", 1.0); w4 = _f(cfg, "severity_w4", 1.0)
    J = np.sqrt(np.maximum(w1 * Yc + w2 * psic + w3 * omegac + w4 * betac, 0.0))
    return J, {"Yc": Yc, "psic": psic, "omegac": omegac, "betac": betac}


def severity_minimization_port(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> PortResult:
    """Parseh et al. paper-core severity-minimization candidate-lattice port."""
    n = len(samples)
    if n == 0:
        z = np.zeros((0,), dtype=float)
        return PortResult(z.astype(bool), z, z, {})
    nominal = _nominal_index(samples)
    feas = np.asarray([float(np.asarray(d.get("feasible", 1.0)).item()) > 0.5 for d in samples], dtype=bool)
    target = _observed_target(samples)
    if target is None:
        # No struck target can be reconstructed from deployment observations.
        score = np.full((n,), -1.0e6, dtype=float)
        score[nominal] = 0.0
        return PortResult(feas.copy(), score, score, {
            "fidelity": "paper_core_no_target_fallback",
            "target_future_source": "unavailable",
            "target_actor_index": -1,
        })

    raw_T = max(np.asarray(d.get("prefix_states", np.zeros((0, 9)))).shape[0] for d in samples)
    T = max(2, min(raw_T, _i(cfg, "severity_collision_horizon_steps", 20)))
    dt = _f(cfg, "severity_collision_dt", _f(cfg, "contact_dt", 0.1))
    states = np.stack([_candidate_states(d, T) for d in samples], axis=0)
    ts = np.arange(T, dtype=float) * dt
    target_xy = target["position"][None, :] + ts[:, None] * target["velocity"][None, :]
    target_heading = np.full((T,), float(target["heading"]), dtype=float)

    has, first, normal, tangent_poi = _sat_collision_geometry(
        states, target_xy, target_heading, float(target["length"]), float(target["width"])
    )
    tangent = tangent_poi[:, :2]; poi = tangent_poi[:, 2:]
    rows = np.arange(n)
    impact_state = states[rows, first]
    target_pos = target_xy[first]
    target_head = target_heading[first]
    v1_post, w1_post, full, impulse = _collision_impulse(
        impact_state, target_pos, np.asarray(target["velocity"], dtype=float), float(target["heading"]),
        float(target["yaw_rate"]), normal, tangent, poi, cfg,
    )
    J, parts = _target_postimpact_rollout_cost(target_pos, v1_post, target_head, w1_post, cfg)
    # No collision is strictly preferable when the benchmark candidate avoids an
    # impact; the paper's optimizer is only invoked once collision is deemed
    # unavoidable, so this extension is conservative rather than severity-seeking.
    J = np.where(has, J, 0.0)
    # The source optimizer is invoked only after collision is deemed unavoidable.
    # On OC-RAP's common lattice an avoiding candidate may nevertheless exist;
    # prefer it lexicographically without perturbing the source Eq. (25) ordering
    # among colliding candidates.  The tiny epsilon is an interface tie-break,
    # not a paper severity term.
    avoid_eps = max(_f(cfg, "severity_collision_avoidance_epsilon", 1.0e-6), 0.0)
    selection_cost = J + has.astype(float) * avoid_eps
    score = -selection_cost
    admitted = feas & np.isfinite(selection_cost)
    fallback = -selection_cost
    return PortResult(admitted, score, fallback, {
        "fidelity": "paper_core_kudlich_slibar_3dof_eq25_candidate_port",
        "source_temporal_semantics": "preimpact_unavoidable_collision_mitigation",
        "ocrap_evaluation_regime": "near_contact_legacy_control",
        "target_future_source": "observed_constant_velocity_projection",
        "contact_geometry": "vectorized_sat_broadphase_plus_source_sutherland_hodgman_overlap_centroid_poi",
        "contact_plane": "source_overlap_intersection_vertex_line_with_sat_degenerate_fallback",
        "full_impact_solver": "planar_rigid_body_sticking_projection_source_fullimpact_fsolve_not_reproduced",
        "target_actor_index": int(target["actor_index"]),
        "impact_predicted": has,
        "impact_step": first,
        "impact_type_full": full,
        "impact_impulse_ns": impulse,
        "collision_cost_J": J,
        "selection_cost_with_avoidance_tiebreak": selection_cost,
        "collision_avoidance_tiebreak": avoid_eps,
        "collision_cost_Yc": parts["Yc"],
        "collision_cost_psic": parts["psic"],
        "collision_cost_omegac": parts["omegac"],
        "collision_cost_betac": parts["betac"],
    })
