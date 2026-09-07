from __future__ import annotations

"""V48.109 OC-RCSO: raw candidate-scene convex relational orientation audit.

V48.108.1 establishes two structural facts: the frozen raw->projected candidate
maps are injective, while the fixed whole raw candidate pathway still fails the
registered Support/Reserve transfer gates.  V48.109 therefore asks a narrower
question before any more encoder training: does a *single context-conditioned
signed orientation field* exist in a fixed raw candidate x scene feature space?

The score family is

    s(u, c) = <w, u> + <W, u c^T> = u^T (w + W c),

where u is the target-specific candidate-minus-nominal response coordinate and
c is a candidate-invariant raw scene summary.  This is bilinear in inputs but
linear in parameters (w, W).  With positive ridge regularization, fitting is a
strictly convex quadratic problem with a unique closed-form solution.  Thus a
STOP cannot be attributed to local minima, LR, epoch count, or Transformer
capacity.

Audit only.  No planner/Stage-I/root/source parameters are trained.
"""

from dataclasses import dataclass
import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.v48_108_raw_to_projected_action_pathway_audit import _layout_slices

ENGINEERING_VERSION = "v48.109.0-OC-RCSO"
ALGORITHM_NAME = "Observation-Consistent Raw Candidate-Scene Convex Relational Orientation Audit"
RAW_CANDIDATE_DIM = 156
RAW_SCENE_CONTEXT_DIM = 240
RELATIONAL_DIM = RAW_CANDIDATE_DIM + RAW_CANDIDATE_DIM * RAW_SCENE_CONTEXT_DIM


def raw_scene_context_summary(x: torch.Tensor, layout: FlatFeatureLayout) -> torch.Tensor:
    """Fixed candidate-invariant scene summary with permutation-invariant agents.

    Layout, chosen before observing V48.109 results:
      agent_summary[8]
      + agent-set mean/std/max/min over 10-D raw agent tokens [40]
      + BEV[14] + route[70] + map[70] + dynamics[38] = 240.

    No candidate-pathway quantity and no Safe/Near/Contact identifier is used.
    """
    if x.ndim != 2 or x.shape[1] != int(layout.total_dim):
        raise ValueError("raw_scene_context_summary expects [B,total_dim]")
    s = _layout_slices(layout)
    agent_summary = x[:, s["agent_summary"]].float()
    agents = x[:, s["agents"]].reshape(x.shape[0], layout.feature_max_agents, layout.agent_token_dim).float()
    mean = agents.mean(1)
    std = agents.std(1, unbiased=False)
    mx = agents.max(1).values
    mn = agents.min(1).values
    bev = x[:, s["bev"]].float()
    route = x[:, s["route"]].float()
    maps = x[:, s["map"]].float()
    dyn = x[:, s["dyn"]].float()
    out = torch.cat([agent_summary, mean, std, mx, mn, bev, route, maps, dyn], dim=-1)
    if out.shape[1] != RAW_SCENE_CONTEXT_DIM:
        raise ValueError(f"scene context dim mismatch {out.shape[1]} != {RAW_SCENE_CONTEXT_DIM}")
    return out


@dataclass
class ZeroAnchoredRelationalScaler:
    """Training-only conditioning that preserves u=0 exactly.

    Candidate response u is RMS-scaled without centering, hence nominal zero
    remains exact zero.  Scene context can be centered because every relational
    term is multiplied by u and therefore still vanishes for nominal response.
    """

    u_scale: np.ndarray
    c_mu: np.ndarray
    c_sd: np.ndarray


def fit_relational_scaler(U: np.ndarray, C: np.ndarray) -> ZeroAnchoredRelationalScaler:
    U = np.asarray(U, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("candidate response dimension mismatch")
    if C.ndim != 2 or C.shape[1] != RAW_SCENE_CONTEXT_DIM or len(C) != len(U):
        raise ValueError("scene context dimension mismatch")
    u_scale = np.sqrt(np.mean(U * U, axis=0, keepdims=True))
    u_scale = np.where(u_scale > 1.0e-8, u_scale, 1.0)
    c_mu = C.mean(axis=0, keepdims=True)
    c_sd = C.std(axis=0, keepdims=True)
    c_sd = np.where(c_sd > 1.0e-8, c_sd, 1.0)
    return ZeroAnchoredRelationalScaler(u_scale=u_scale, c_mu=c_mu, c_sd=c_sd)


def base_features(U: np.ndarray, scaler: ZeroAnchoredRelationalScaler) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    return U / scaler.u_scale


def relational_features(U: np.ndarray, C: np.ndarray, scaler: ZeroAnchoredRelationalScaler) -> np.ndarray:
    us = base_features(U, scaler)
    cz = (np.asarray(C, dtype=np.float64) - scaler.c_mu) / scaler.c_sd
    cross = np.einsum("ni,nj->nij", us, cz, optimize=True).reshape(len(us), -1)
    out = np.concatenate([us, cross], axis=1)
    if out.shape[1] != RELATIONAL_DIM:
        raise ValueError(f"relational feature dim mismatch {out.shape[1]} != {RELATIONAL_DIM}")
    return out


@dataclass
class ConvexRidgeOrientation:
    coef: np.ndarray
    ridge_lambda: float
    objective: float
    normal_equation_residual: float


def _balanced_weights(y01: np.ndarray) -> np.ndarray:
    y = np.asarray(y01, dtype=np.int64)
    n = len(y)
    pos = int(y.sum()); neg = int(n - pos)
    if n < 4 or pos == 0 or neg == 0:
        raise ValueError("convex orientation probe requires both classes")
    w = np.empty(n, dtype=np.float64)
    w[y == 1] = n / (2.0 * pos)
    w[y == 0] = n / (2.0 * neg)
    return w


def fit_closed_form_ridge(X: np.ndarray, y01: np.ndarray) -> ConvexRidgeOrientation:
    """Unique strongly-convex balanced ridge solution, solved in the sample dual.

    Objective:
      (1/sum a_i) sum_i a_i (x_i^T w - y_i)^2 + lambda ||w||^2,
      y_i in {-1,+1}, lambda = 1/N.

    The dual N x N solve avoids a 37k x 37k system for relational features.
    No iterative optimizer, LR, epoch, or regularization sweep is involved.
    """
    X = np.asarray(X, dtype=np.float64)
    y01 = np.asarray(y01, dtype=np.int64)
    if X.ndim != 2 or len(X) != len(y01):
        raise ValueError("ridge input shape mismatch")
    n = len(y01)
    a = _balanced_weights(y01)
    y = 2.0 * y01.astype(np.float64) - 1.0
    sw = np.sqrt(a / a.sum())
    A = X * sw[:, None]
    b = y * sw
    lam = 1.0 / float(n)
    K = A @ A.T
    K.flat[:: n + 1] += lam
    alpha = np.linalg.solve(K, b)
    coef = A.T @ alpha
    pred = X @ coef
    obj = float(np.sum(a * (pred - y) ** 2) / a.sum() + lam * np.dot(coef, coef))
    grad = 2.0 * (X.T @ (a * (pred - y)) / a.sum() + lam * coef)
    resid = float(np.linalg.norm(grad) / max(1.0, np.linalg.norm(coef)))
    return ConvexRidgeOrientation(coef=coef, ridge_lambda=lam, objective=obj, normal_equation_residual=resid)


def ridge_scores(model: ConvexRidgeOrientation, X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(model.coef, dtype=np.float64)


def dimension_checks() -> dict[str, bool]:
    L = FlatFeatureLayout()
    context_dim = int(L.agent_summary_dim + 4 * L.agent_token_dim + L.bev_dim + (L.route_stats_dim + L.route_flat_dim) + (L.map_stats_dim + L.map_flat_dim) + (L.dyn_stats_dim + L.dyn_flat_dim))
    return {
        "raw_candidate_dim_156": RAW_CANDIDATE_DIM == 156,
        "raw_scene_context_dim_240": context_dim == RAW_SCENE_CONTEXT_DIM == 240,
        "relational_dim_37596": RELATIONAL_DIM == 37596,
    }


def synthetic_agent_permutation_invariance() -> bool:
    L = FlatFeatureLayout(); torch.manual_seed(48109)
    x = torch.randn(3, L.total_dim)
    s = _layout_slices(L)
    agents = x[:, s["agents"]].reshape(3, L.feature_max_agents, L.agent_token_dim).clone()
    perm = torch.randperm(L.feature_max_agents)
    y = x.clone(); y[:, s["agents"]] = agents[:, perm].reshape(3, -1)
    return bool(torch.allclose(raw_scene_context_summary(x, L), raw_scene_context_summary(y, L), atol=1.0e-6, rtol=1.0e-6))


def synthetic_nominal_zero_feature() -> bool:
    U = np.zeros((4, RAW_CANDIDATE_DIM), dtype=np.float64)
    C = np.arange(4 * RAW_SCENE_CONTEXT_DIM, dtype=np.float64).reshape(4, RAW_SCENE_CONTEXT_DIM)
    sc = fit_relational_scaler(np.vstack([U, np.ones_like(U)]), np.vstack([C, C + 1.0]))
    z = relational_features(U, C, sc)
    return bool(np.count_nonzero(z) == 0)


def synthetic_relational_solution() -> bool:
    """XOR-like context dependence: base u cannot solve, u*c can."""
    U = np.zeros((8, RAW_CANDIDATE_DIM), dtype=np.float64)
    C = np.zeros((8, RAW_SCENE_CONTEXT_DIM), dtype=np.float64)
    us = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.float64)
    cs = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64)
    U[:, 0] = us; C[:, 0] = cs
    y = ((us * cs) > 0).astype(np.int64)
    sc = fit_relational_scaler(U, C)
    mb = fit_closed_form_ridge(base_features(U, sc), y)
    mr = fit_closed_form_ridge(relational_features(U, C, sc), y)
    sb = ridge_scores(mb, base_features(U, sc)); sr = ridge_scores(mr, relational_features(U, C, sc))
    base_acc = np.mean((sb > 0) == y); rel_acc = np.mean((sr > 0) == y)
    return bool(base_acc <= 0.75 and rel_acc == 1.0 and mr.normal_equation_residual < 1.0e-8)


def contract_checks() -> dict[str, bool]:
    return {
        **dimension_checks(),
        "agent_set_summary_permutation_invariant": synthetic_agent_permutation_invariance(),
        "nominal_zero_relational_feature": synthetic_nominal_zero_feature(),
        "closed_form_convex_relation_solves_context_dependence": synthetic_relational_solution(),
    }
