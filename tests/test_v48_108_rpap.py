from __future__ import annotations
import torch
from ocrap.models.encoders import FlatFeatureLayout,StructuredTokenEncoder
from ocrap.v48_108_raw_to_projected_action_pathway_audit import (
 action_features,candidate_pathway_dimension_check,contract_checks,projected_candidate_pathway,
 projection_full_column_rank,raw_candidate_pathway,raw_pathway_dim,reconstruct_raw_delta_from_projected,
 static_context_zero_delta_check,synthetic_projection_injectivity_check,
)

def _enc(d=32):
 return StructuredTokenEncoder(FlatFeatureLayout(),d_model=d,num_layers=2,num_heads=4).eval()

def test_dims_default_contract():
 assert raw_pathway_dim(FlatFeatureLayout())==156
 assert candidate_pathway_dimension_check(192)

def test_nominal_candidate_zero_delta_raw_and_projected():
 torch.manual_seed(1); enc=_enc(); x=torch.randn(4,enc.layout.total_dim);x[1]=x[0]
 for z in (raw_candidate_pathway(x,enc.layout),projected_candidate_pathway(enc,x)):
  _,d,c=action_features(z); assert torch.count_nonzero(d[0]).item()==0; assert torch.count_nonzero(c[0]).item()==0

def test_projection_is_full_column_rank_synthetic():
 assert projection_full_column_rank(_enc(192))
 assert synthetic_projection_injectivity_check(192)

def test_raw_delta_pseudoinverse_reconstruction():
 torch.manual_seed(2);enc=_enc(192);d=torch.randn(9,raw_pathway_dim(enc.layout));_,r=reconstruct_raw_delta_from_projected(enc,d)
 assert torch.allclose(d,r,atol=2e-5,rtol=1e-5)

def test_direct_projected_delta_matches_block_map():
 torch.manual_seed(3);enc=_enc(192);x=torch.randn(5,enc.layout.total_dim)
 rp=raw_candidate_pathway(x,enc.layout);pp=projected_candidate_pathway(enc,x)
 _,rd,_=action_features(rp);_,pd,_=action_features(pp);yd,_=reconstruct_raw_delta_from_projected(enc,rd)
 assert torch.allclose(pd,yd,atol=2e-5,rtol=1e-5)

def test_static_context_and_runtime_contract_checks():
 assert static_context_zero_delta_check()
 assert all(contract_checks(192).values())
