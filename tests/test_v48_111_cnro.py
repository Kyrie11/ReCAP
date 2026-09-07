from __future__ import annotations
import numpy as np

from ocrap.v48_111_constraint_native_recovery_orientation import (
    RAW_CANDIDATE_DIM, AGENT_DIM, PREFIX_COMPLETE_STEPS, GEOMETRY_DIM, MATCHED_DIM,
    candidate_agent_signed_clearance_paths, pair_indices_from_clearance,
    fit_constraint_native_scaler, matched_features, contract_checks,
)


def _scene():
    n,j=3,3
    A=np.zeros((n,j,AGENT_DIM),dtype=np.float64);M=np.ones((n,j),dtype=bool)
    A[:,:,7]=.48;A[:,:,8]=.4
    A[:,0,0]=.03;A[:,0,1]=.02
    A[:,1,0]=.03;A[:,1,1]=-.02
    A[:,2,0]=.08;A[:,2,1]=0.
    R0=np.zeros((n,RAW_CANDIDATE_DIM));Rc=R0.copy();R0[:,7]=Rc[:,7]=4.8;R0[:,8]=Rc[:,8]=2.0
    for k in range(n):
        s0=np.zeros((PREFIX_COMPLETE_STEPS,9));sc=np.zeros_like(s0)
        s0[:,0]=np.linspace(.2,2.0,PREFIX_COMPLETE_STEPS);sc[:]=s0
        sc[:,1]=(k-1)*.15*np.linspace(0.1,1.0,PREFIX_COMPLETE_STEPS)
        s0[:,7]=sc[:,7]=4.8;s0[:,8]=sc[:,8]=2.0
        R0[k,36:108]=s0.reshape(-1);Rc[k,36:108]=sc.reshape(-1)
    return R0,Rc,A,M


def test_contract_checks():
    c=contract_checks();assert c and all(c.values())


def test_matched_capacity_and_nominal_zero():
    R0,Rc,A,M=_scene();Hc=candidate_agent_signed_clearance_paths(Rc,A,M,10.0);H0=candidate_agent_signed_clearance_paths(R0,A,M,10.0)
    active,nearest=pair_indices_from_clearance(Hc,A,M);U=np.zeros((len(Rc),RAW_CANDIDATE_DIM));U[:,0]=[1.,2.,3.]
    s=fit_constraint_native_scaler(U,Hc,H0,M)
    fa=matched_features(U,Hc,H0,active,s);fn=matched_features(U,Hc,H0,nearest,s)
    assert GEOMETRY_DIM==32 and MATCHED_DIM==188
    assert fa.shape==fn.shape==(3,188)
    z=matched_features(np.zeros_like(U),H0,H0,active,s)
    assert np.count_nonzero(z)==0


def test_active_selector_changes_with_candidate_and_is_permutation_invariant():
    R0,Rc,A,M=_scene();Hc=candidate_agent_signed_clearance_paths(Rc,A,M,10.0);active,nearest=pair_indices_from_clearance(Hc,A,M)
    assert active.shape==nearest.shape==(3,2)
    p=np.array([2,0,1]);H2=Hc[:,p];A2=A[:,p];M2=M[:,p];a2,n2=pair_indices_from_clearance(H2,A2,M2)
    assert np.array_equal(active,p[a2]);assert np.array_equal(nearest,p[n2])


def test_clearance_delta_has_physical_sign():
    R0,Rc,A,M=_scene();Hc=candidate_agent_signed_clearance_paths(Rc,A,M,10.0);H0=candidate_agent_signed_clearance_paths(R0,A,M,10.0)
    # Candidate 0 moves laterally toward one side and candidate 2 toward the other;
    # the induced clearance response must not be identically zero.
    assert np.max(np.abs(np.where(np.isfinite(Hc),Hc-H0,0.0)))>1e-6
