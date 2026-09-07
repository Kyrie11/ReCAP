from __future__ import annotations
import numpy as np

from ocrap.v48_111_constraint_native_geometry_orientation import (
    RAW_CANDIDATE_DIM, GAP_FEATURE_DIM, FLOW_FEATURE_DIM,
    PREFIX_STATE_START, PREFIX_STATE_WIDTH, PREFIX_COMPLETE_STEPS,
    constraint_native_candidate_geometry, fit_constraint_geometry_scaler,
    gap_features, flow_features, contract_checks,
)
from ocrap.v48_109_raw_candidate_scene_orientation import fit_closed_form_ridge


def _raw(n=3):
    R = np.zeros((n, RAW_CANDIDATE_DIM), dtype=np.float64); R[:, 7] = 4.8; R[:, 8] = 2.0
    for k in range(n):
        st = np.zeros((PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH), dtype=np.float64)
        st[:, 0] = np.linspace(0.5, 4.0, PREFIX_COMPLETE_STEPS)
        st[:, 2] = 5.0
        st[:, 7] = 4.8; st[:, 8] = 2.0
        R[k, PREFIX_STATE_START:PREFIX_STATE_START+72] = st.reshape(-1)
    return R


def _agents(n=3):
    A = np.zeros((n, 3, 10), dtype=np.float64); M = np.ones((n, 3), dtype=bool)
    A[:, :, 7] = .48; A[:, :, 8] = .4
    A[:, 0, 0] = .04; A[:, 0, 2] = .10
    A[:, 1, 0] = .07; A[:, 1, 1] = .03
    A[:, 2, 0] = .07; A[:, 2, 1] = -.04
    return A, M


def test_registered_dimensions_and_contract():
    assert GAP_FEATURE_DIM == 160 and FLOW_FEATURE_DIM == 164
    assert all(contract_checks().values())


def test_nominal_geometry_and_features_are_exact_zero():
    N = _raw(4); A, M = _agents(4)
    gap, flow, _ = constraint_native_candidate_geometry(N, N, A, M, 10.0)
    assert np.count_nonzero(gap) == 0 and np.count_nonzero(flow) == 0
    U = np.ones((4, RAW_CANDIDATE_DIM)); sc = fit_constraint_geometry_scaler(U, np.ones((4,4)), np.ones((4,4)))
    assert np.count_nonzero(gap_features(np.zeros_like(U), gap, sc)) == 0
    assert np.count_nonzero(flow_features(np.zeros_like(U), gap, flow, sc)) == 0


def test_agent_permutation_preserves_constraint_native_content():
    C = _raw(3); N = _raw(3); A, M = _agents(3)
    # Candidate 0 shifts left, candidate 1 right, candidate 2 stays nominal.
    for k, dy in enumerate((-1.0, 1.0, 0.0)):
        st = C[k, PREFIX_STATE_START:PREFIX_STATE_START+72].reshape(8,9).copy(); st[:,1] = np.linspace(0,dy,8)
        C[k, PREFIX_STATE_START:PREFIX_STATE_START+72] = st.reshape(-1)
    g, f, d = constraint_native_candidate_geometry(C, N, A, M, 10.0)
    p = np.array([2,0,1])
    gp, fp, dp = constraint_native_candidate_geometry(C, N, A[:,p], M[:,p], 10.0)
    assert np.allclose(g, gp) and np.allclose(f, fp)
    assert np.array_equal(d['candidate_active1'], p[dp['candidate_active1']])
    assert np.array_equal(d['nominal_active1'], p[dp['nominal_active1']])


def test_global_rotation_preserves_gap_and_normal_flow_delta():
    C = _raw(2); N = _raw(2); A, M = _agents(2)
    st = C[0, PREFIX_STATE_START:PREFIX_STATE_START+72].reshape(8,9).copy(); st[:,1] = np.linspace(0,1.0,8); st[:,3] = 1.0
    C[0, PREFIX_STATE_START:PREFIX_STATE_START+72] = st.reshape(-1)
    g, f, _ = constraint_native_candidate_geometry(C, N, A, M, 10.0)
    # Rotate all xy/vxy quantities 90 degrees. Ego current positions are zero.
    R = np.array([[0.,-1.],[1.,0.]])
    Cr, Nr, Ar = C.copy(), N.copy(), A.copy()
    for X in (Cr, Nr):
        states = X[:, PREFIX_STATE_START:PREFIX_STATE_START+72].reshape(len(X),8,9)
        states[:,:,:2] = states[:,:,:2] @ R.T
        states[:,:,2:4] = states[:,:,2:4] @ R.T
        X[:,0:2] = X[:,0:2] @ R.T
    Ar[:,:,:2] = Ar[:,:,:2] @ R.T
    Ar[:,:,2:4] = Ar[:,:,2:4] @ R.T
    gr, fr, _ = constraint_native_candidate_geometry(Cr, Nr, Ar, M, 10.0)
    assert np.allclose(g, gr, atol=1e-10) and np.allclose(f, fr, atol=1e-10)


def test_outward_candidate_has_positive_gap_delta_against_front_agent():
    C = _raw(1); N = _raw(1); A, M = _agents(1)
    # Move candidate laterally away from the front agent while nominal stays straight.
    st = C[0, PREFIX_STATE_START:PREFIX_STATE_START+72].reshape(8,9).copy(); st[:,1] = np.linspace(0,2.0,8); st[:,3] = 2.0
    C[0, PREFIX_STATE_START:PREFIX_STATE_START+72] = st.reshape(-1)
    gap, _, _ = constraint_native_candidate_geometry(C, N, A, M, 10.0)
    assert np.max(gap[0]) > 0.0


def test_low_dimensional_closed_form_solver_remains_unique():
    rng = np.random.default_rng(11); n = 40
    U = rng.normal(size=(n, RAW_CANDIDATE_DIM)); g = rng.normal(size=(n,4)); f = rng.normal(size=(n,4)); y = np.array([0,1]*(n//2))
    sc = fit_constraint_geometry_scaler(U,g,f)
    for X in (gap_features(U,g,sc), flow_features(U,g,f,sc)):
        m = fit_closed_form_ridge(X,y)
        assert m.ridge_lambda == 1/n and m.normal_equation_residual < 1e-8
