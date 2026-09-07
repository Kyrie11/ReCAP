from __future__ import annotations

"""V48.111 OC-CNGO: constraint-native candidate-agent geometry orientation audit.

V48.110 showed that candidate-dependent active-agent selection carries local
Support signal, but the high-dimensional coordinatewise token product is not a
population-invariant Support+Reserve source.  V48.111 keeps the V48.109/V48.110
strictly-convex closed-form ridge solver and the same observation-only fixed-CV
active-agent selector, while replacing raw agent-token coordinates by low-
dimensional *constraint-native candidate response coordinates*.

For every candidate and its scene-time nominal action, the same first eight
complete prefix states are evaluated against the current observed-agent set.
Candidate and nominal active agent/time pairs are selected by signed circle
clearance.  At those active pairs V48.111 measures candidate-minus-nominal
change in signed gap h and in its first-order normal flow h_dot.  These scalar
coordinates are invariant to global translation/rotation and vanish exactly for
the nominal action.

Registered nested families:
  base : historical candidate response u only, dimension 156;
  gap  : [u, delta-h at candidate-active1/2 and nominal-active1/2], dim 160;
  flow : gap plus delta-hdot at the same four active pairs, dim 164.

No teacher future, regime id, learned router, threshold/LR/capacity sweep,
source training, Stage-I/root training, or boundary transport is used.
"""

from dataclasses import dataclass
import numpy as np

from ocrap.v48_110_candidate_agent_topology_orientation import (
    RAW_CANDIDATE_DIM,
    AGENT_DIM,
    PREFIX_STATE_START,
    PREFIX_STATE_WIDTH,
    PREFIX_COMPLETE_STEPS,
)

ENGINEERING_VERSION = "v48.111.0-OC-CNGO"
ALGORITHM_NAME = "Observation-Consistent Constraint-Native Geometry Orientation Audit"
GAP_GEOMETRY_DIM = 4
FLOW_GEOMETRY_DIM = 4
GAP_FEATURE_DIM = RAW_CANDIDATE_DIM + GAP_GEOMETRY_DIM
FLOW_FEATURE_DIM = RAW_CANDIDATE_DIM + GAP_GEOMETRY_DIM + FLOW_GEOMETRY_DIM


def _decode_prefix_state(raw_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(raw_candidate, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("raw candidate shape mismatch")
    flat = r[:, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH]
    st = flat.reshape(len(r), PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH)
    ego_xy = r[:, :2]
    ego_len = np.maximum(np.abs(r[:, 7]), 1.0e-3)
    ego_wid = np.maximum(np.abs(r[:, 8]), 1.0e-3)
    ego_rad = 0.5 * np.hypot(ego_len, ego_wid)
    return st[:, :, :2], st[:, :, 2:4], ego_xy, ego_rad


def _agent_cv_geometry(
    raw_candidate: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed gap h[N,A,T], normal flow hdot[N,A,T], and valid mask.

    hdot > 0 means the candidate is opening the normal separation; hdot < 0
    means closing. Agent tokens contain current world-frame velocity, while
    prefix F_EGO states contain candidate world-frame ego velocity.
    """
    R = np.asarray(raw_candidate, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    M = np.asarray(M, dtype=bool)
    if A.ndim != 3 or A.shape[0] != len(R) or A.shape[2] != AGENT_DIM or M.shape != A.shape[:2]:
        raise ValueError("agent geometry shape mismatch")
    if not np.isfinite(sample_rate_hz) or float(sample_rate_hz) <= 0.0:
        raise ValueError("invalid sample rate")
    pxy, pvel, ego_xy, ego_rad = _decode_prefix_state(R)
    prefix_rel = pxy - ego_xy[:, None, :]
    rel0 = A[:, :, 0:2] * 80.0
    avel = A[:, :, 2:4] * 20.0
    alen = np.maximum(np.abs(A[:, :, 7] * 10.0), 1.0e-3)
    awid = np.maximum(np.abs(A[:, :, 8] * 5.0), 1.0e-3)
    arad = 0.5 * np.hypot(alen, awid)
    times = (np.arange(PREFIX_COMPLETE_STEPS, dtype=np.float64) + 1.0) / float(sample_rate_hz)
    agent_rel = rel0[:, :, None, :] + avel[:, :, None, :] * times[None, None, :, None]
    r = prefix_rel[:, None, :, :] - agent_rel
    dist = np.linalg.norm(r, axis=-1)
    h = dist - ego_rad[:, None, None] - arad[:, :, None]
    n = r / np.maximum(dist[..., None], 1.0e-9)
    vrel = pvel[:, None, :, :] - avel[:, :, None, :]
    hdot = np.sum(n * vrel, axis=-1)
    h = np.where(M[:, :, None], h, np.inf)
    hdot = np.where(M[:, :, None], hdot, 0.0)
    return h, hdot, M


def _active_pairs(h: np.ndarray, M: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return first/second active agent index and each agent's active time index.

    Exact clearance ties are broken by the observable agent-token content, not
    by tensor slot index, so input-agent permutation cannot change the selected
    physical content.  This only resolves a measure-zero tie ambiguity in the
    V48.110 minimum-clearance selector.
    """
    h = np.asarray(h, dtype=np.float64)
    M = np.asarray(M, dtype=bool)
    A = np.asarray(A, dtype=np.float64)
    if h.ndim != 3 or M.shape != h.shape[:2] or A.shape[:2] != M.shape:
        raise ValueError("active pair shape mismatch")
    n = h.shape[0]
    a1 = np.zeros(n, dtype=np.int64)
    a2 = np.zeros(n, dtype=np.int64)
    t1 = np.zeros(n, dtype=np.int64)
    t2 = np.zeros(n, dtype=np.int64)
    cmin = np.min(h, axis=2)
    tmin = np.argmin(h, axis=2)
    for k in range(n):
        ids = np.flatnonzero(M[k])
        if ids.size == 0:
            continue
        order = sorted((int(i) for i in ids), key=lambda i: (float(cmin[k, i]), *(float(x) for x in A[k, i].tolist())))
        a1[k] = int(order[0])
        a2[k] = int(order[1] if len(order) > 1 else order[0])
        t1[k] = int(tmin[k, a1[k]])
        t2[k] = int(tmin[k, a2[k]])
    return a1, a2, t1, t2


def _value_at_pairs(x: np.ndarray, a: np.ndarray, t: np.ndarray) -> np.ndarray:
    idx = np.arange(len(x), dtype=np.int64)
    return x[idx, np.asarray(a, dtype=np.int64), np.asarray(t, dtype=np.int64)]


def constraint_native_candidate_geometry(
    candidate_raw: np.ndarray,
    nominal_raw: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Candidate-relative signed-gap and normal-flow coordinates.

    The four registered active pairs are candidate active-1/2 and nominal
    active-1/2.  For each pair, delta-h and delta-hdot are candidate minus
    nominal at the *same agent and time*.  This makes the coordinates both
    action-relative and constraint-native while retaining active switching.
    """
    C = np.asarray(candidate_raw, dtype=np.float64)
    N0 = np.asarray(nominal_raw, dtype=np.float64)
    if N0.ndim == 1:
        N0 = np.repeat(N0[None, :], len(C), axis=0)
    if N0.shape != C.shape:
        raise ValueError("nominal/candidate raw shape mismatch")
    hc, vc, mask = _agent_cv_geometry(C, A, M, sample_rate_hz)
    hn, vn, _ = _agent_cv_geometry(N0, A, M, sample_rate_hz)
    ca1, ca2, ct1, ct2 = _active_pairs(hc, mask, A)
    na1, na2, nt1, nt2 = _active_pairs(hn, mask, A)
    pairs = ((ca1, ct1), (ca2, ct2), (na1, nt1), (na2, nt2))
    gap = np.stack([_value_at_pairs(hc, a, t) - _value_at_pairs(hn, a, t) for a, t in pairs], axis=1)
    flow = np.stack([_value_at_pairs(vc, a, t) - _value_at_pairs(vn, a, t) for a, t in pairs], axis=1)
    gap = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    diag = {
        "candidate_active1": ca1, "candidate_active2": ca2,
        "candidate_time1": ct1, "candidate_time2": ct2,
        "nominal_active1": na1, "nominal_active2": na2,
        "nominal_time1": nt1, "nominal_time2": nt2,
        "candidate_min_gap": np.min(hc, axis=(1, 2)),
        "nominal_min_gap": np.min(hn, axis=(1, 2)),
        "active1_switch": (ca1 != na1).astype(np.int64),
        "active2_switch": (ca2 != na2).astype(np.int64),
    }
    return gap, flow, diag


@dataclass
class ConstraintGeometryScaler:
    u_scale: np.ndarray
    gap_scale: np.ndarray
    flow_scale: np.ndarray


def _rms_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    return np.where(s > 1.0e-8, s, 1.0)


def fit_constraint_geometry_scaler(U: np.ndarray, gap: np.ndarray, flow: np.ndarray) -> ConstraintGeometryScaler:
    U = np.asarray(U, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    flow = np.asarray(flow, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("candidate response dimension mismatch")
    if gap.shape != (len(U), GAP_GEOMETRY_DIM) or flow.shape != (len(U), FLOW_GEOMETRY_DIM):
        raise ValueError("constraint geometry dimension mismatch")
    # Zero-centered RMS scaling preserves the physical zero boundary and the
    # exact nominal-zero intervention; no mean subtraction is allowed here.
    return ConstraintGeometryScaler(_rms_scale(U), _rms_scale(gap), _rms_scale(flow))


def base_features(U: np.ndarray, scaler: ConstraintGeometryScaler) -> np.ndarray:
    return np.asarray(U, dtype=np.float64) / scaler.u_scale


def gap_features(U: np.ndarray, gap: np.ndarray, scaler: ConstraintGeometryScaler) -> np.ndarray:
    us = base_features(U, scaler)
    gz = np.asarray(gap, dtype=np.float64) / scaler.gap_scale
    out = np.concatenate([us, gz], axis=1)
    if out.shape[1] != GAP_FEATURE_DIM:
        raise ValueError("gap feature dim mismatch")
    return out


def flow_features(U: np.ndarray, gap: np.ndarray, flow: np.ndarray, scaler: ConstraintGeometryScaler) -> np.ndarray:
    us = base_features(U, scaler)
    gz = np.asarray(gap, dtype=np.float64) / scaler.gap_scale
    fz = np.asarray(flow, dtype=np.float64) / scaler.flow_scale
    out = np.concatenate([us, gz, fz], axis=1)
    if out.shape[1] != FLOW_FEATURE_DIM:
        raise ValueError("flow feature dim mismatch")
    return out


def contract_checks() -> dict[str, bool]:
    rng = np.random.default_rng(48111)
    n, a = 6, 4
    C = np.zeros((n, RAW_CANDIDATE_DIM), dtype=np.float64)
    N = np.zeros((n, RAW_CANDIDATE_DIM), dtype=np.float64)
    C[:, 7] = N[:, 7] = 4.8
    C[:, 8] = N[:, 8] = 2.0
    for k in range(n):
        st0 = np.zeros((PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH), dtype=np.float64)
        st0[:, 0] = np.linspace(0.5, 4.0, PREFIX_COMPLETE_STEPS)
        st0[:, 2] = 5.0
        st0[:, 7] = 4.8; st0[:, 8] = 2.0
        stc = st0.copy(); stc[:, 1] = np.linspace(0.0, 0.8 * (k - 2), PREFIX_COMPLETE_STEPS)
        C[k, PREFIX_STATE_START:PREFIX_STATE_START + 72] = stc.reshape(-1)
        N[k, PREFIX_STATE_START:PREFIX_STATE_START + 72] = st0.reshape(-1)
    A = np.zeros((n, a, AGENT_DIM), dtype=np.float64); M = np.ones((n, a), dtype=bool)
    A[:, :, 7] = 0.48; A[:, :, 8] = 0.4
    A[:, 0, 0] = 0.05; A[:, 0, 2] = 0.10
    A[:, 1, 0] = 0.08; A[:, 1, 1] = 0.03
    A[:, 2, 0] = 0.08; A[:, 2, 1] = -0.03
    A[:, 3, 0] = -0.07
    gap, flow, d = constraint_native_candidate_geometry(C, N, A, M, 10.0)
    p = np.array([2, 0, 3, 1])
    g2, f2, d2 = constraint_native_candidate_geometry(C, N, A[:, p], M[:, p], 10.0)
    perm_ok = bool(np.allclose(gap, g2) and np.allclose(flow, f2) and np.array_equal(d["candidate_active1"], p[d2["candidate_active1"]]))
    z_gap, z_flow, _ = constraint_native_candidate_geometry(N, N, A, M, 10.0)
    U = rng.normal(size=(n, RAW_CANDIDATE_DIM))
    sc = fit_constraint_geometry_scaler(U, gap, flow)
    return {
        "raw_candidate_dim_156": RAW_CANDIDATE_DIM == 156,
        "gap_feature_dim_160": GAP_FEATURE_DIM == 160,
        "flow_feature_dim_164": FLOW_FEATURE_DIM == 164,
        "first_8_complete_prefix_states": PREFIX_COMPLETE_STEPS == 8,
        "constraint_native_permutation_invariant": perm_ok,
        "nominal_gap_delta_exact_zero": bool(np.count_nonzero(z_gap) == 0),
        "nominal_flow_delta_exact_zero": bool(np.count_nonzero(z_flow) == 0),
        "nominal_feature_exact_zero": bool(np.count_nonzero(flow_features(np.zeros((2, RAW_CANDIDATE_DIM)), z_gap[:2], z_flow[:2], sc)) == 0),
        "physical_zero_preserved_by_scaler": bool(np.all(np.isfinite(sc.gap_scale)) and np.all(np.isfinite(sc.flow_scale))),
    }
