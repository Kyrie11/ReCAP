from __future__ import annotations

"""V48.111 OC-CNRO: constraint-native candidate/agent recovery orientation audit.

V48.110 showed that raw active-agent topology is not a transferable Support/Reserve
source.  This audit keeps the first-eight-state observation-only CV geometry and the
same strictly-convex ridge owner, but replaces raw agent-token interactions with a
constraint-native *action-induced signed-clearance response*.

For candidate a and nominal a0, h_i^a(t) is the circle signed-clearance to observed
agent i under the same CV continuation.  The primitive is

    Delta h_i^a(t) = h_i^a(t) - h_i^a0(t).

It is nominal-zero by construction and is the finite-action analogue of the active
constraint normal response grad(h)^T Delta p.  A second channel multiplies Delta h
by nominal signed clearance, retaining boundary context without a learned threshold.

Two matched 188-D families are compared:
  nearest_pair = [u, geometry of the two currently nearest agents]
  active_pair  = [u, geometry of the two candidate-specific minimum-clearance agents]

The added geometry is exactly 32-D in both families (2 agents x 8 times x 2 channels),
so active-vs-nearest no longer has the 1716-vs-3588 capacity mismatch of V48.110.
Audit only; no model/planner/source parameters are trained.
"""

from dataclasses import dataclass
import numpy as np

from ocrap.v48_110_candidate_agent_topology_orientation import (
    RAW_CANDIDATE_DIM, AGENT_DIM, PREFIX_COMPLETE_STEPS,
    decode_prefix_xy_and_ego,
)

ENGINEERING_VERSION = "v48.111.0-OC-CNRO"
ALGORITHM_NAME = "Observation-Consistent Constraint-Native Recovery Orientation Audit"
PAIR_SIZE = 2
GEOMETRY_CHANNELS = 2
GEOMETRY_DIM = PAIR_SIZE * PREFIX_COMPLETE_STEPS * GEOMETRY_CHANNELS  # 32
MATCHED_DIM = RAW_CANDIDATE_DIM + GEOMETRY_DIM  # 188


def candidate_agent_signed_clearance_paths(
    raw_candidate: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    """Observation-only signed clearance [N,J,8] for candidate prefix vs CV agents."""
    R = np.asarray(raw_candidate, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    M = np.asarray(M, dtype=bool)
    if R.ndim != 2 or R.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("raw candidate shape mismatch")
    if A.ndim != 3 or A.shape[0] != len(R) or A.shape[2] != AGENT_DIM or M.shape != A.shape[:2]:
        raise ValueError("agent shape mismatch")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("invalid sample rate")

    pxy, ego_xy, ego_rad = decode_prefix_xy_and_ego(R)
    prefix_rel = pxy - ego_xy[:, None, :]
    rel0 = A[:, :, 0:2] * 80.0
    vel = A[:, :, 2:4] * 20.0
    alen = np.maximum(np.abs(A[:, :, 7] * 10.0), 1.0e-3)
    awid = np.maximum(np.abs(A[:, :, 8] * 5.0), 1.0e-3)
    arad = 0.5 * np.sqrt(alen * alen + awid * awid)
    times = (np.arange(PREFIX_COMPLETE_STEPS, dtype=np.float64) + 1.0) / float(sample_rate_hz)
    future = rel0[:, :, None, :] + vel[:, :, None, :] * times[None, None, :, None]
    delta = prefix_rel[:, None, :, :] - future
    clear = np.linalg.norm(delta, axis=-1) - ego_rad[:, None, None] - arad[:, :, None]
    return np.where(M[:, :, None], clear, np.inf)


def pair_indices_from_clearance(H: np.ndarray, A: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate-specific active pair and candidate-invariant nearest pair."""
    H = np.asarray(H, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    M = np.asarray(M, dtype=bool)
    if H.ndim != 3 or H.shape[:2] != M.shape or H.shape[2] != PREFIX_COMPLETE_STEPS:
        raise ValueError("clearance shape mismatch")
    rel0 = A[:, :, 0:2] * 80.0
    r2 = np.sum(rel0 * rel0, axis=-1)
    r2 = np.where(M, r2, np.inf)
    cmin = np.min(H, axis=2)
    n = len(H)
    active = np.zeros((n, PAIR_SIZE), dtype=np.int64)
    nearest = np.zeros((n, PAIR_SIZE), dtype=np.int64)
    for k in range(n):
        ids = np.flatnonzero(M[k])
        if ids.size == 0:
            continue
        oa = ids[np.argsort(cmin[k, ids], kind="stable")]
        on = ids[np.argsort(r2[k, ids], kind="stable")]
        active[k, 0] = int(oa[0]); active[k, 1] = int(oa[1] if len(oa) > 1 else oa[0])
        nearest[k, 0] = int(on[0]); nearest[k, 1] = int(on[1] if len(on) > 1 else on[0])
    return active, nearest


@dataclass
class ConstraintNativeScaler:
    u_scale: np.ndarray
    clearance_scale: float


def fit_constraint_native_scaler(U: np.ndarray, Hc: np.ndarray, H0: np.ndarray, M: np.ndarray) -> ConstraintNativeScaler:
    U = np.asarray(U, dtype=np.float64)
    Hc = np.asarray(Hc, dtype=np.float64)
    H0 = np.asarray(H0, dtype=np.float64)
    M = np.asarray(M, dtype=bool)
    if U.ndim != 2 or U.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("candidate response dimension mismatch")
    u_scale = np.sqrt(np.mean(U * U, axis=0, keepdims=True))
    u_scale = np.where(u_scale > 1.0e-8, u_scale, 1.0)
    finite = np.broadcast_to(M[:, :, None], Hc.shape) & np.isfinite(Hc) & np.isfinite(H0)
    vals = np.concatenate([Hc[finite], H0[finite]]) if np.any(finite) else np.array([1.0])
    hs = float(np.sqrt(np.mean(vals * vals)))
    if not np.isfinite(hs) or hs <= 1.0e-8:
        hs = 1.0
    return ConstraintNativeScaler(u_scale=u_scale, clearance_scale=hs)


def base_features(U: np.ndarray, scaler: ConstraintNativeScaler) -> np.ndarray:
    return np.asarray(U, dtype=np.float64) / scaler.u_scale


def _pair_geometry(Hc: np.ndarray, H0: np.ndarray, pair: np.ndarray, scaler: ConstraintNativeScaler) -> np.ndarray:
    n = len(Hc)
    ix = np.arange(n)[:, None]
    hc = Hc[ix, pair]
    h0 = H0[ix, pair]
    # No-agent rows arrive as inf; make them zero/noninformative rather than NaN.
    valid = np.isfinite(hc) & np.isfinite(h0)
    dh = np.where(valid, hc - h0, 0.0) / scaler.clearance_scale
    h0z = np.where(valid, h0, 0.0) / scaler.clearance_scale
    # Interleave two physically interpretable channels for every pair/time cell.
    g = np.stack([dh, dh * h0z], axis=-1).reshape(n, -1)
    if g.shape[1] != GEOMETRY_DIM:
        raise ValueError(f"geometry dim mismatch {g.shape[1]} != {GEOMETRY_DIM}")
    return g


def matched_features(U: np.ndarray, Hc: np.ndarray, H0: np.ndarray, pair: np.ndarray, scaler: ConstraintNativeScaler) -> np.ndarray:
    out = np.concatenate([base_features(U, scaler), _pair_geometry(Hc, H0, pair, scaler)], axis=1)
    if out.shape[1] != MATCHED_DIM:
        raise ValueError(f"matched dim mismatch {out.shape[1]} != {MATCHED_DIM}")
    return out


def contract_checks() -> dict[str, bool]:
    rng = np.random.default_rng(48111)
    n, j = 6, 5
    A = rng.normal(size=(n, j, AGENT_DIM)); M = np.ones((n, j), dtype=bool)
    A[:, :, 0:4] *= 0.05; A[:, :, 7] = 0.48; A[:, :, 8] = 0.4
    R0 = np.zeros((n, RAW_CANDIDATE_DIM)); R0[:, 7] = 4.8; R0[:, 8] = 2.0
    Rc = R0.copy()
    for k in range(n):
        st0 = np.zeros((PREFIX_COMPLETE_STEPS, 9)); stc = np.zeros_like(st0)
        st0[:, 0] = np.linspace(.2, 2.0, PREFIX_COMPLETE_STEPS)
        stc[:, 0] = st0[:, 0] + 0.1 * (k + 1)
        stc[:, 1] = 0.05 * ((-1) ** k) * np.arange(1, PREFIX_COMPLETE_STEPS + 1)
        st0[:, 7] = stc[:, 7] = 4.8; st0[:, 8] = stc[:, 8] = 2.0
        R0[k, 36:108] = st0.reshape(-1); Rc[k, 36:108] = stc.reshape(-1)
    Hc = candidate_agent_signed_clearance_paths(Rc, A, M, 10.0)
    H0 = candidate_agent_signed_clearance_paths(R0, A, M, 10.0)
    active, nearest = pair_indices_from_clearance(Hc, A, M)
    U = rng.normal(size=(n, RAW_CANDIDATE_DIM)); sc = fit_constraint_native_scaler(U, Hc, H0, M)
    fa = matched_features(U, Hc, H0, active, sc); fn = matched_features(U, Hc, H0, nearest, sc)
    z = matched_features(np.zeros_like(U), H0, H0, active, sc)
    p = np.array([2,4,1,0,3]); Hp = Hc[:, p]; Ap = A[:, p]; Mp = M[:, p]
    a2, n2 = pair_indices_from_clearance(Hp, Ap, Mp)
    # map permuted indices to original content ids
    perm_ok = bool(np.array_equal(active, p[a2]) and np.array_equal(nearest, p[n2]))
    return {
        "geometry_dim_32": GEOMETRY_DIM == 32,
        "matched_dim_188": MATCHED_DIM == 188,
        "matched_family_same_dim": fa.shape[1] == fn.shape[1] == MATCHED_DIM,
        "nominal_zero_exact": bool(np.count_nonzero(z) == 0),
        "candidate_agent_permutation_invariant": perm_ok,
        "clearance_scale_finite_positive": bool(np.isfinite(sc.clearance_scale) and sc.clearance_scale > 0),
    }
