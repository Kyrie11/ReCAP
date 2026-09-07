from __future__ import annotations
import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.v48_108_raw_to_projected_action_pathway_audit import _layout_slices
from ocrap.v48_109_raw_candidate_scene_orientation import fit_closed_form_ridge, ridge_scores
from ocrap.v48_110_candidate_agent_topology_orientation import (
    RAW_CANDIDATE_DIM,AGENT_DIM,NEAREST_DIM,TOPOLOGY_DIM,PREFIX_STATE_START,PREFIX_STATE_WIDTH,PREFIX_COMPLETE_STEPS,
    raw_agent_set,candidate_agent_clearance_topology,fit_active_topology_scaler,
    base_features,nearest_features,topology_features,contract_checks,
)


def _synthetic_raw(n=4):
    R=np.zeros((n,RAW_CANDIDATE_DIM),dtype=np.float64);R[:,7]=4.8;R[:,8]=2.0
    for i in range(n):
        st=np.zeros((PREFIX_COMPLETE_STEPS,PREFIX_STATE_WIDTH));st[:,0]=np.linspace(.5,4.0,PREFIX_COMPLETE_STEPS);st[:,1]=0.2*i;st[:,7]=4.8;st[:,8]=2.0
        R[i,PREFIX_STATE_START:PREFIX_STATE_START+PREFIX_COMPLETE_STEPS*PREFIX_STATE_WIDTH]=st.reshape(-1)
    return R


def test_registered_dimensions():
    assert RAW_CANDIDATE_DIM==156 and AGENT_DIM==10 and NEAREST_DIM==1716 and TOPOLOGY_DIM==3588


def test_raw_agent_set_shape_and_mask():
    L=FlatFeatureLayout();x=torch.zeros(3,L.total_dim);s=_layout_slices(L);ag=torch.zeros(3,L.feature_max_agents,L.agent_token_dim);ag[:,0,0]=1.;ag[:,1,1]=2.;x[:,s['agents']]=ag.reshape(3,-1);A,M=raw_agent_set(x,L)
    assert A.shape==(3,32,10) and M[:,:2].all() and not M[:,2:].any()


def test_active_selector_permutation_invariant_content():
    R=_synthetic_raw(3);A=np.zeros((3,4,10));M=np.ones((3,4),bool);A[:,:,7]=.48;A[:,:,8]=.4
    A[:,0,0]=.03;A[:,1,0]=.08;A[:,2,1]=.05;A[:,3,0]=-.07
    a1,a2,nr,t=candidate_agent_clearance_topology(R,A,M,10.)
    p=np.array([2,0,3,1]);b1,b2,bn,bt=candidate_agent_clearance_topology(R,A[:,p],M[:,p],10.)
    assert np.array_equal(a1,p[b1]) and np.array_equal(a2,p[b2]) and np.array_equal(nr,p[bn]) and np.allclose(t,bt)


def test_candidate_prefix_can_switch_active_agent():
    R=_synthetic_raw(2);A=np.zeros((2,2,10));M=np.ones((2,2),bool);A[:,:,7]=.48;A[:,:,8]=.4
    A[:,0,0]=.04;A[:,0,1]=.03;A[:,1,0]=.04;A[:,1,1]=-.03
    st0=np.zeros((PREFIX_COMPLETE_STEPS,PREFIX_STATE_WIDTH));st0[:,0]=np.linspace(.5,4,PREFIX_COMPLETE_STEPS);st0[:,1]=np.linspace(0,3,PREFIX_COMPLETE_STEPS);st0[:,7]=4.8;st0[:,8]=2
    st1=st0.copy();st1[:,1]*=-1
    R[0,PREFIX_STATE_START:PREFIX_STATE_START+72]=st0.reshape(-1);R[1,PREFIX_STATE_START:PREFIX_STATE_START+72]=st1.reshape(-1)
    a1,_,_,_=candidate_agent_clearance_topology(R,A,M,10.)
    assert a1[0]!=a1[1]


def test_nominal_zero_topology_feature():
    rng=np.random.default_rng(5);R=_synthetic_raw(6);A=rng.normal(size=(6,4,10));A[:,:,0:4]*=.1;A[:,:,7]=.48;A[:,:,8]=.4;M=np.ones((6,4),bool);a1,a2,nr,t=candidate_agent_clearance_topology(R,A,M,10.);U=rng.normal(size=(6,156));sc=fit_active_topology_scaler(U,A,M,t)
    z=topology_features(np.zeros((2,156)),A[:2],M[:2],a1[:2],a2[:2],t[:2],sc)
    assert np.count_nonzero(z)==0


def test_nearest_and_topology_shapes():
    rng=np.random.default_rng(6);R=_synthetic_raw(7);A=rng.normal(size=(7,5,10));A[:,:,0:4]*=.1;A[:,:,7]=.48;A[:,:,8]=.4;M=np.ones((7,5),bool);a1,a2,nr,t=candidate_agent_clearance_topology(R,A,M,10.);U=rng.normal(size=(7,156));sc=fit_active_topology_scaler(U,A,M,t)
    assert nearest_features(U,A,M,nr,sc).shape==(7,NEAREST_DIM)
    assert topology_features(U,A,M,a1,a2,t,sc).shape==(7,TOPOLOGY_DIM)


def test_same_closed_form_solver_unique():
    rng=np.random.default_rng(7);X=rng.normal(size=(20,30));y=np.array([0,1]*10);m=fit_closed_form_ridge(X,y)
    assert m.ridge_lambda==1/20 and m.normal_equation_residual<1e-9 and np.isfinite(ridge_scores(m,X)).all()


def test_runtime_contract_synthetics():
    assert all(contract_checks().values())
