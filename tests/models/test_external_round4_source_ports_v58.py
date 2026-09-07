from __future__ import annotations

import numpy as np
import torch

import ocrap.external_baselines.policies as policies
from ocrap.external_baselines.data import _crossing_binary, _map_topology_arrays
from ocrap.external_baselines.models import TopoFuser, WayformerRouteBC
from ocrap.external_baselines.paper_core_ports_v57 import compensatory_postimpact_mpc_port
from ocrap.external_baselines.paper_core_ports_v58 import severity_minimization_port
from ocrap.external_baselines.policies import select_external_policy
from ocrap.external_baselines.train import _wayformer_native_loss
from ocrap.external_baselines.provenance import find_provenance


def test_betop_behavior_braid_uses_source_xt_intersection_not_distance_proxy() -> None:
    src = np.array([[0.0, -1.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    tgt = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    assert _crossing_binary(src, tgt)

    parallel_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    parallel_b = np.array([[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]], dtype=np.float32)
    assert not _crossing_binary(parallel_a, parallel_b)


def test_betop_map_braid_marks_only_source_nearest_polyline() -> None:
    T = 5
    st = np.zeros((T, 9), dtype=np.float32)
    st[:, 0] = np.linspace(0.0, 4.0, T)
    st[:, 7:9] = (4.8, 2.0)
    maps = np.zeros((2, 5, 10), dtype=np.float32)
    maps[0, :, 0] = np.linspace(0.0, 4.0, 5); maps[0, :, 1] = 0.5
    maps[1, :, 0] = np.linspace(0.0, 4.0, 5); maps[1, :, 1] = 1.5
    maps[:, :, 9] = 1.0
    d = {"prefix_states": st, "ego_state": st[0].copy(), "map_polylines": maps}
    cfg = {"external_baselines": {"model": {"future_len": T, "num_topology_map": 2}}}
    _feat, target, mask = _map_topology_arrays(d, cfg)
    assert mask.tolist() == [True, True]
    np.testing.assert_array_equal(target, np.array([1.0, 0.0], dtype=np.float32))


def test_betop_topofuser_uses_additive_previous_topology_feature() -> None:
    torch.manual_seed(0)
    fuser = TopoFuser(8, 0.0).eval()
    src = torch.randn(2, 3, 4, 8)
    tgt = torch.randn(2, 3, 4, 8)
    prev = torch.randn(2, 3, 4, 8)
    base = fuser(src, tgt, None)
    with_prev = fuser(src, tgt, prev)
    torch.testing.assert_close(with_prev - base, prev)


def _wayformer_inputs(scene_shift: float = 0.0):
    B, N, A, H, M, P, T = 2, 4, 3, 5, 4, 3, 6
    x = torch.zeros(B, N, 12)
    mask = torch.ones(B, N, dtype=torch.bool)
    ah = torch.zeros(B, A, H, 9)
    ah[..., 0] = torch.linspace(-2.0, 0.0, H)
    ah[:, 1, :, 1] = 2.0 + scene_shift
    av = torch.ones(B, A, H, dtype=torch.bool)
    mp = torch.zeros(B, M, P, 6)
    mp[..., 0] = torch.linspace(0.0, 2.0, P)
    mp[:, :, :, 1] = torch.arange(M, dtype=torch.float32)[None, :, None]
    mpv = torch.ones(B, M, P, dtype=torch.bool)
    meta = torch.zeros(B, M, 4)
    center = torch.zeros(B, M, 3)
    mv = torch.ones(B, M, dtype=torch.bool)
    prefix = torch.zeros(B, N, T, 2)
    prefix[..., 0] = torch.linspace(0.0, 5.0, T)
    for n in range(N):
        prefix[:, n, :, 1] = 0.2 * n
    pv = torch.ones(B, N, T, dtype=torch.bool)
    return dict(x=x, mask=mask, source_agent_history=ah, source_agent_valid=av,
                source_map_points=mp, source_map_point_valid=mpv,
                source_map_meta=meta, source_map_center=center,
                source_map_valid=mv, prefix_traj=prefix, prefix_valid=pv)


def test_wayformer_scene_early_fusion_changes_native_gmm_and_candidate_scores() -> None:
    torch.manual_seed(7)
    model = WayformerRouteBC(
        input_dim=12, max_candidates=4, d_model=32, num_heads=4, dropout=0.0,
        num_layers=1, num_encoder_layers=1, num_decoder_layers=1,
        num_latents=8, future_len=6, num_output_queries=4,
        num_mode_decoder_layers=1, max_history_steps=8, max_source_agents=4,
    ).eval()
    a = model(**_wayformer_inputs(scene_shift=0.0))
    b = model(**_wayformer_inputs(scene_shift=4.0))
    assert a["wayformer_mode_params"].shape == (2, 4, 6, 5)
    assert a["wayformer_mode_logits"].shape == (2, 4)
    assert torch.isfinite(a["logits"]).all()
    assert not torch.allclose(a["wayformer_mode_params"], b["wayformer_mode_params"])
    assert not torch.allclose(a["logits"], b["logits"])


def test_wayformer_native_bivariate_gmm_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(9)
    model = WayformerRouteBC(
        input_dim=12, max_candidates=4, d_model=32, num_heads=4, dropout=0.0,
        num_layers=1, num_encoder_layers=1, num_decoder_layers=1,
        num_latents=8, future_len=6, num_output_queries=4,
        num_mode_decoder_layers=1, max_history_steps=8, max_source_agents=4,
    )
    batch = _wayformer_inputs(scene_shift=0.0)
    batch["target_index"] = torch.tensor([0, 1], dtype=torch.long)
    out = model(**batch)
    loss = _wayformer_native_loss(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.mode_traj_head.weight.grad is not None
    assert torch.isfinite(model.mode_traj_head.weight.grad).all()


def _severity_sample(y_end: float, *, nominal: bool) -> dict:
    T, dt = 11, 0.1
    x = np.linspace(0.0, 12.0, T)
    y = np.linspace(0.0, y_end, T)
    st = np.zeros((T, 9), dtype=np.float32)
    st[:, 0] = x; st[:, 1] = y
    st[:, 2] = np.gradient(x, dt); st[:, 3] = np.gradient(y, dt)
    st[:, 4] = np.arctan2(st[:, 3], st[:, 2])
    st[:, 6] = np.hypot(st[:, 2], st[:, 3]); st[:, 7:9] = (4.8, 2.0)
    hist = np.zeros((3, 2, 16), dtype=np.float32)
    valid = np.ones((3, 2), dtype=bool)
    hist[:, 0, 0] = np.linspace(-2.0, 0.0, 3); hist[:, 0, 3] = 12.0; hist[:, 0, 10:12] = (4.8, 2.0)
    hist[:, 1, 0] = 10.0; hist[:, 1, 7] = 0.0; hist[:, 1, 10:12] = (4.8, 2.0)
    return {
        "prefix_states": st, "agent_history": hist, "agent_valid": valid,
        "ego_state": st[0].copy(), "feasible": np.int32(1),
        "is_nominal": np.int32(nominal), "utility": np.float32(0.0),
    }


def _severity_cfg() -> dict:
    return {"external_baselines": {"policy": {
        "severity_collision_horizon_steps": 11,
        "severity_collision_dt": 0.1,
        "severity_postimpact_steps": 20,
        "severity_postimpact_dt": 0.02,
        "severity_w1": 1.0, "severity_w2": 1.0, "severity_w3": 1.0, "severity_w4": 1.0,
    }}}


def test_parseh_severity_port_uses_collision_postimpact_cost_and_prefers_avoidance() -> None:
    samples = [_severity_sample(0.0, nominal=True), _severity_sample(5.0, nominal=False)]
    port = severity_minimization_port(samples, _severity_cfg())
    np.testing.assert_array_equal(port.diagnostics["impact_predicted"], np.array([True, False]))
    assert port.diagnostics["fidelity"] == "paper_core_kudlich_slibar_3dof_eq25_candidate_port"
    assert "fsolve_not_reproduced" in port.diagnostics["full_impact_solver"]
    assert port.diagnostics["contact_plane"] == "source_overlap_intersection_vertex_line_with_sat_degenerate_fallback"
    assert port.diagnostics["collision_avoidance_tiebreak"] > 0.0
    sel = select_external_policy("severity_minimization", samples, _severity_cfg())
    assert sel.selected_index == 1


def test_parseh_severity_port_is_predictor_free_and_teacher_invariant(monkeypatch) -> None:
    samples = [_severity_sample(0.0, nominal=True), _severity_sample(5.0, nominal=False)]
    def forbidden(*args, **kwargs):
        raise AssertionError("Parseh source port must not invoke the learned OC-RAP risk predictor")
    monkeypatch.setattr(policies, "observed_risk_profiles_and_context", forbidden)
    a = select_external_policy("severity_minimization", samples, _severity_cfg())
    for i, d in enumerate(samples):
        d["hard_violation"] = np.float32(1000 + i)
        d["harm_proxy"] = np.float32(-1000 - i)
        d["m_star"] = np.full((3, 2), 999.0 + i, dtype=np.float32)
    b = select_external_policy("severity_minimization", samples, _severity_cfg())
    assert a.selected_index == b.selected_index
    np.testing.assert_allclose(a.score, b.score)



def test_betop_inference_adds_source_short_term_repulsive_potential() -> None:
    T = 6
    def cand(y: float, nominal: bool) -> dict:
        st = np.zeros((T, 9), dtype=np.float32)
        st[:, 0] = np.linspace(0.0, 5.0, T)
        st[:, 1] = y
        hist = np.zeros((3, 2, 9), dtype=np.float32)
        valid = np.ones((3, 2), dtype=bool)
        hist[:, 0, 0] = np.linspace(-2.0, 0.0, 3)
        # actor sits on the nominal path near x=1 at the current time
        hist[:, 1, 0] = 1.0
        hist[:, 1, 1] = 0.0
        return {
            "prefix_states": st, "agent_history": hist, "agent_valid": valid,
            "feasible": np.int32(1), "is_nominal": np.int32(nominal),
            "utility": np.float32(0.0),
        }
    samples = [cand(0.0, True), cand(4.0, False)]
    cfg = {"external_baselines": {"policy": {
        "betop_branch_steps": 3, "betop_contingency_dt": 0.1,
        "betop_short_term_cost_weight": 0.5,
    }}}
    out = {"logits": np.array([0.0, 0.0], dtype=np.float32)}
    sel = select_external_policy("betopnet_lite", samples, cfg, model_outputs=out)
    assert sel.selected_index == 1
    assert "short_term_contingency_cost" in sel.reason

def test_cao2021_remains_source_limited_without_invented_equations() -> None:
    d = _severity_sample(0.0, nominal=True)
    p = compensatory_postimpact_mpc_port([d], {"external_baselines": {"policy": {}}})
    assert p.diagnostics["fidelity"] == "source_limited_abstract_structured"
    assert p.diagnostics["requires_full_paper_for_equation_exact_port"] is True
    prov = find_provenance("compensatory_postimpact_mpc")
    assert prov is not None
    assert "full-equation source unavailable" in prov.fidelity


def test_round4_reporting_keeps_wayformer_and_betop_as_safe_controls_and_moves_severity_to_near_control() -> None:
    w = find_provenance("wayformer_bc"); b = find_provenance("betopnet_lite"); s = find_provenance("severity_minimization")
    assert w is not None and w.regimes == ("safe",) and "adapter" in w.reporting_name.lower()
    assert b is not None and b.regimes == ("safe",) and "adapter" in b.reporting_name.lower()
    assert s is not None and s.regimes == ("near",) and "legacy" in s.reporting_name.lower()
