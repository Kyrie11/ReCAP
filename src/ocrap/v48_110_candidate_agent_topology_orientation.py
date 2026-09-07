from __future__ import annotations

"""V48.110 OC-CATO: candidate-to-agent active-constraint topology audit.

V48.109 ruled out a candidate-invariant global-summary bilinear field as a
sufficient explanation under a strictly-convex unique solver.  The next
preregistered question is whether the missing structure is *candidate-dependent
active-constraint switching*: different candidate prefixes can make different
observed agents become the limiting recovery constraint.

V48.110 keeps the exact V48.109 closed-form ridge solver.  It changes only the
fixed feature map.  For every candidate, an observation-only constant-velocity
(CV) clearance proxy is evaluated over the first eight complete historical
prefix states against every currently observed agent.  The smallest and
second-smallest clearance agents define a deterministic active set.  This
selector is candidate-dependent, content-based, and permutation invariant over
agent input order.  No teacher future, regime id, learned routing, threshold
sweep, or model training is used.

The registered families are:
  base:    candidate response u only (exact V48.109 convex base control),
  nearest: u plus interaction with the current nearest observed agent,
  topology:u plus interactions with the candidate-specific first/second active
           agents and two active-clearance scalars.

All fitted scores remain linear in parameters with positive ridge, so each fit
has a unique global optimum.  Nominal response u=0 maps to exact zero features.
"""

from dataclasses import dataclass
import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.v48_108_raw_to_projected_action_pathway_audit import _layout_slices

ENGINEERING_VERSION = "v48.110.0-OC-CATO"
ALGORITHM_NAME = "Observation-Consistent Candidate-to-Agent Active-Constraint Topology Orientation Audit"
RAW_CANDIDATE_DIM = 156
AGENT_DIM = 10
PREFIX_STATE_START = 36
PREFIX_STATE_WIDTH = 9
PREFIX_COMPLETE_STEPS = 8  # 80 flat values contain at least 8 complete 9-D states.
NEAREST_DIM = RAW_CANDIDATE_DIM + RAW_CANDIDATE_DIM * AGENT_DIM
TOPOLOGY_DIM = RAW_CANDIDATE_DIM + 2 * RAW_CANDIDATE_DIM * AGENT_DIM + 2 * RAW_CANDIDATE_DIM


def raw_agent_set(x: torch.Tensor, layout: FlatFeatureLayout) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.shape[1] != int(layout.total_dim):
        raise ValueError("raw_agent_set expects [B,total_dim]")
    s = _layout_slices(layout)
    a = x[:, s["agents"]].reshape(x.shape[0], layout.feature_max_agents, layout.agent_token_dim).float()
    mask = torch.linalg.vector_norm(a, dim=-1) > 1.0e-8
    return a, mask


def decode_prefix_xy_and_ego(raw_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the eight complete prefix xy states plus current ego size.

    Historical prefix states are 9-D F_EGO rows; the fixed 80-D flattened block
    therefore contains eight complete rows without using the truncated tail.
    """
    r = np.asarray(raw_candidate, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("raw candidate shape mismatch")
    flat = r[:, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH]
    st = flat.reshape(len(r), PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH)
    ego_xy = r[:, 0:2]
    ego_len = np.maximum(np.abs(r[:, 7]), 1.0e-3)
    ego_wid = np.maximum(np.abs(r[:, 8]), 1.0e-3)
    ego_rad = 0.5 * np.sqrt(ego_len * ego_len + ego_wid * ego_wid)
    return st[:, :, 0:2], ego_xy, ego_rad


def candidate_agent_clearance_topology(raw_candidate: np.ndarray, A: np.ndarray, M: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Return active1/active2/nearest indices and [c1, gap12] topology scalars.

    Raw agent token layout (historical feature builder):
      dx/80, dy/80, vx/20, vy/20, speed/20, sin, cos, length/10,
      width/5, object_type/10.
    Agent continuation is observation-only constant velocity.  Circle support is
    the same inexpensive geometry used by historical recovery-witness code.
    """
    R = np.asarray(raw_candidate,dtype=np.float64); A=np.asarray(A,dtype=np.float64); M=np.asarray(M,dtype=bool)
    if A.ndim != 3 or A.shape[0] != len(R) or A.shape[2] != AGENT_DIM or M.shape != A.shape[:2]:
        raise ValueError("agent topology shape mismatch")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("invalid sample rate")
    pxy, ego_xy, ego_rad = decode_prefix_xy_and_ego(R)
    prefix_rel = pxy - ego_xy[:,None,:]
    rel0 = A[:,:,0:2] * 80.0
    vel = A[:,:,2:4] * 20.0
    alen = np.maximum(np.abs(A[:,:,7] * 10.0),1.0e-3)
    awid = np.maximum(np.abs(A[:,:,8] * 5.0),1.0e-3)
    arad = 0.5 * np.sqrt(alen*alen + awid*awid)
    times = (np.arange(PREFIX_COMPLETE_STEPS,dtype=np.float64)+1.0) / float(sample_rate_hz)
    future = rel0[:,:,None,:] + vel[:,:,None,:] * times[None,None,:,None]
    delta = prefix_rel[:,None,:,:] - future
    clear = np.linalg.norm(delta,axis=-1) - ego_rad[:,None,None] - arad[:,:,None]
    clear = np.where(M[:,:,None],clear,np.inf)
    cmin = np.min(clear,axis=2)
    r2 = np.sum(rel0*rel0,axis=-1); r2=np.where(M,r2,np.inf)
    n=len(R); active1=np.zeros(n,dtype=np.int64);active2=np.zeros(n,dtype=np.int64);nearest=np.zeros(n,dtype=np.int64); topo=np.zeros((n,2),dtype=np.float64)
    for k in range(n):
        ids=np.flatnonzero(M[k])
        if ids.size==0:
            continue
        nearest[k]=int(ids[np.argmin(r2[k,ids])])
        order=ids[np.argsort(cmin[k,ids],kind='stable')]
        active1[k]=int(order[0]); active2[k]=int(order[1] if len(order)>1 else order[0])
        c1=float(cmin[k,active1[k]]); c2=float(cmin[k,active2[k]])
        topo[k,0]=c1; topo[k,1]=max(c2-c1,0.0)
    return active1,active2,nearest,topo


@dataclass
class ActiveTopologyScaler:
    u_scale: np.ndarray
    a_mu: np.ndarray
    a_sd: np.ndarray
    t_mu: np.ndarray
    t_sd: np.ndarray


def fit_active_topology_scaler(U: np.ndarray, A: np.ndarray, M: np.ndarray, topo: np.ndarray) -> ActiveTopologyScaler:
    U=np.asarray(U,dtype=np.float64);A=np.asarray(A,dtype=np.float64);M=np.asarray(M,dtype=bool);topo=np.asarray(topo,dtype=np.float64)
    u_scale=np.sqrt(np.mean(U*U,axis=0,keepdims=True));u_scale=np.where(u_scale>1e-8,u_scale,1.0)
    valid=A[M]
    if valid.size:
        a_mu=valid.mean(axis=0,keepdims=True);a_sd=valid.std(axis=0,keepdims=True);a_sd=np.where(a_sd>1e-8,a_sd,1.0)
    else:
        a_mu=np.zeros((1,AGENT_DIM));a_sd=np.ones((1,AGENT_DIM))
    t_mu=topo.mean(axis=0,keepdims=True);t_sd=topo.std(axis=0,keepdims=True);t_sd=np.where(t_sd>1e-8,t_sd,1.0)
    return ActiveTopologyScaler(u_scale,a_mu,a_sd,t_mu,t_sd)


def base_features(U: np.ndarray, scaler: ActiveTopologyScaler) -> np.ndarray:
    return np.asarray(U,dtype=np.float64)/scaler.u_scale


def _agent_z(A: np.ndarray, M: np.ndarray, idx: np.ndarray, scaler: ActiveTopologyScaler) -> np.ndarray:
    A=np.asarray(A,dtype=np.float64);M=np.asarray(M,dtype=bool);idx=np.asarray(idx,dtype=np.int64)
    z=(A[np.arange(len(A)),idx]-scaler.a_mu)/scaler.a_sd
    valid=M[np.arange(len(M)),idx]
    return np.where(valid[:,None],z,0.0)


def nearest_features(U: np.ndarray, A: np.ndarray, M: np.ndarray, nearest_idx: np.ndarray, scaler: ActiveTopologyScaler) -> np.ndarray:
    us=base_features(U,scaler);az=_agent_z(A,M,nearest_idx,scaler)
    cross=np.einsum('ni,nj->nij',us,az,optimize=True).reshape(len(us),-1)
    out=np.concatenate([us,cross],axis=1)
    if out.shape[1]!=NEAREST_DIM:raise ValueError("nearest dim mismatch")
    return out


def topology_features(U: np.ndarray, A: np.ndarray, M: np.ndarray, active1: np.ndarray, active2: np.ndarray, topo: np.ndarray, scaler: ActiveTopologyScaler) -> np.ndarray:
    us=base_features(U,scaler);a1=_agent_z(A,M,active1,scaler);a2=_agent_z(A,M,active2,scaler);tz=(np.asarray(topo,dtype=np.float64)-scaler.t_mu)/scaler.t_sd
    c1=np.einsum('ni,nj->nij',us,a1,optimize=True).reshape(len(us),-1)
    c2=np.einsum('ni,nj->nij',us,a2,optimize=True).reshape(len(us),-1)
    ts=np.einsum('ni,nj->nij',us,tz,optimize=True).reshape(len(us),-1)
    out=np.concatenate([us,c1,c2,ts],axis=1)
    if out.shape[1]!=TOPOLOGY_DIM:raise ValueError(f"topology dim mismatch {out.shape[1]} != {TOPOLOGY_DIM}")
    return out


def contract_checks() -> dict[str,bool]:
    dims={"raw_candidate_dim_156":RAW_CANDIDATE_DIM==156,"agent_dim_10":AGENT_DIM==10,"nearest_dim_1716":NEAREST_DIM==1716,"topology_dim_3588":TOPOLOGY_DIM==3588}
    rng=np.random.default_rng(48110);n=6;A=rng.normal(size=(n,5,AGENT_DIM));M=np.ones((n,5),bool)
    # Make plausible normalized geometry/size fields.
    A[:,:,0:2]*=.1;A[:,:,2:4]*=.1;A[:,:,7]=.48;A[:,:,8]=.4
    R=np.zeros((n,RAW_CANDIDATE_DIM));R[:,0:2]=0.;R[:,7]=4.8;R[:,8]=2.0
    # Eight complete prefix states; candidates move along x with distinct offsets.
    for i in range(n):
        st=np.zeros((PREFIX_COMPLETE_STEPS,PREFIX_STATE_WIDTH));st[:,0]=np.linspace(.5,4.0,PREFIX_COMPLETE_STEPS)+0.2*i;st[:,7]=4.8;st[:,8]=2.0
        R[i,PREFIX_STATE_START:PREFIX_STATE_START+72]=st.reshape(-1)
    a1,a2,nr,t=candidate_agent_clearance_topology(R,A,M,10.0)
    # Content-based selection must be permutation invariant after mapping indices back to content.
    p=np.array([2,4,1,0,3]);Ap=A[:,p];Mp=M[:,p];b1,b2,bn,bt=candidate_agent_clearance_topology(R,Ap,Mp,10.0)
    inv=np.argsort(p)
    perm_ok=bool(np.array_equal(a1,p[b1]) and np.array_equal(a2,p[b2]) and np.array_equal(nr,p[bn]) and np.allclose(t,bt))
    U=rng.normal(size=(n,RAW_CANDIDATE_DIM));sc=fit_active_topology_scaler(U,A,M,t);z=topology_features(np.zeros_like(U[:2]),A[:2],M[:2],a1[:2],a2[:2],t[:2],sc)
    # Candidate-dependent active switching synthetic.
    Rs=np.repeat(R[:1],2,axis=0);As=np.zeros((2,2,AGENT_DIM));Ms=np.ones((2,2),bool);As[:,:,7]=.48;As[:,:,8]=.4
    As[:,0,0]=.04;As[:,0,1]=.03;As[:,1,0]=.04;As[:,1,1]=-.03
    st0=np.zeros((PREFIX_COMPLETE_STEPS,PREFIX_STATE_WIDTH));st0[:,0]=np.linspace(.5,4,PREFIX_COMPLETE_STEPS);st0[:,1]=np.linspace(0,3,PREFIX_COMPLETE_STEPS);st0[:,7]=4.8;st0[:,8]=2
    st1=st0.copy();st1[:,1]*=-1
    Rs[0,PREFIX_STATE_START:PREFIX_STATE_START+72]=st0.reshape(-1);Rs[1,PREFIX_STATE_START:PREFIX_STATE_START+72]=st1.reshape(-1)
    s1,_,_,_=candidate_agent_clearance_topology(Rs,As,Ms,10.0)
    return {**dims,"active_selector_permutation_invariant":perm_ok,"nominal_zero_topology_feature":bool(np.count_nonzero(z)==0),"candidate_prefix_can_switch_active_agent":bool(s1[0]!=s1[1])}
