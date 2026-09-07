from __future__ import annotations
import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.v48_108_raw_to_projected_action_pathway_audit import _layout_slices
from ocrap.v48_109_raw_candidate_scene_orientation import (
    RAW_CANDIDATE_DIM, RAW_SCENE_CONTEXT_DIM, RELATIONAL_DIM,
    raw_scene_context_summary, fit_relational_scaler, base_features, relational_features,
    fit_closed_form_ridge, ridge_scores, contract_checks,
)


def test_registered_dimensions():
    assert RAW_CANDIDATE_DIM == 156
    assert RAW_SCENE_CONTEXT_DIM == 240
    assert RELATIONAL_DIM == 37596


def test_scene_summary_agent_permutation_invariant():
    L=FlatFeatureLayout(); torch.manual_seed(1); x=torch.randn(4,L.total_dim); s=_layout_slices(L)
    ag=x[:,s["agents"]].reshape(4,L.feature_max_agents,L.agent_token_dim).clone()
    y=x.clone(); y[:,s["agents"]]=ag[:,torch.randperm(L.feature_max_agents)].reshape(4,-1)
    assert torch.allclose(raw_scene_context_summary(x,L),raw_scene_context_summary(y,L),atol=1e-6,rtol=1e-6)


def test_nominal_zero_is_structural_zero():
    rng=np.random.default_rng(2)
    U=rng.normal(size=(10,RAW_CANDIDATE_DIM)); C=rng.normal(size=(10,RAW_SCENE_CONTEXT_DIM))
    sc=fit_relational_scaler(U,C)
    z=relational_features(np.zeros((3,RAW_CANDIDATE_DIM)),C[:3],sc)
    assert np.count_nonzero(z)==0


def test_closed_form_ridge_unique_solution_residual():
    rng=np.random.default_rng(3); X=rng.normal(size=(18,25)); y=np.array([0,1]*9)
    m=fit_closed_form_ridge(X,y)
    assert m.ridge_lambda==1/18
    assert m.normal_equation_residual<1e-10
    assert np.isfinite(m.objective)


def test_relational_feature_solves_context_dependent_orientation():
    U=np.zeros((8,RAW_CANDIDATE_DIM));C=np.zeros((8,RAW_SCENE_CONTEXT_DIM))
    u=np.array([-1,1,-1,1,-1,1,-1,1.]);c=np.array([-1,-1,1,1,-1,-1,1,1.])
    U[:,0]=u;C[:,0]=c;y=((u*c)>0).astype(np.int64);sc=fit_relational_scaler(U,C)
    mb=fit_closed_form_ridge(base_features(U,sc),y);mr=fit_closed_form_ridge(relational_features(U,C,sc),y)
    ab=np.mean((ridge_scores(mb,base_features(U,sc))>0)==y)
    ar=np.mean((ridge_scores(mr,relational_features(U,C,sc))>0)==y)
    assert ab<=0.75 and ar==1.0


def test_context_summary_candidate_invariance_when_raw_scene_fixed():
    L=FlatFeatureLayout();torch.manual_seed(4);x=torch.randn(5,L.total_dim);s=_layout_slices(L)
    for g in ("agent_summary","agents","bev","route","map","dyn"):
        x[1:,s[g]]=x[0,s[g]]
    c=raw_scene_context_summary(x,L)
    assert torch.count_nonzero(c[1:]-c[0:1]).item()==0


def test_runtime_contract_synthetics():
    assert all(contract_checks().values())
