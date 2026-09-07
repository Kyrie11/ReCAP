from __future__ import annotations

import copy

import numpy as np

import ocrap.external_baselines.policies as policies
from ocrap.external_baselines.paper_core_ports_v57 import (
    _path_frame,
    _solve_box_qp_batch,
    post_collision_restoration_port,
    robust_postimpact_control_port,
)
from ocrap.external_baselines.policies import select_external_policy


def _sample(*, nominal: bool = False, accel: float = 0.0, steer: np.ndarray | float = 0.0,
            speed: float = 10.0, yaw_rate: float = 0.05, vy: float = 0.0,
            elapsed: float = 0.0, history_lane_y: float | None = None) -> dict:
    T = 11
    dt = 0.1
    st = np.zeros((T, 9), dtype=np.float32)
    st[:, 0] = np.arange(T, dtype=np.float32) * speed * dt
    st[:, 2] = speed
    st[:, 3] = vy
    st[:, 4] = 0.0
    st[:, 5] = yaw_rate
    st[:, 6] = np.hypot(speed, vy)
    st[:, 7:9] = (4.8, 2.0)
    u = np.zeros((T, 2), dtype=np.float32)
    u[:, 0] = accel
    if np.ndim(steer) == 0:
        u[:, 1] = float(steer)
    else:
        arr = np.asarray(steer, dtype=np.float32).reshape(-1)
        u[: min(T, arr.size), 1] = arr[:T]
        if arr.size and arr.size < T:
            u[arr.size :, 1] = arr[-1]

    H = 6
    hist = np.zeros((H, 1, 16), dtype=np.float32)
    valid = np.ones((H, 1), dtype=bool)
    hist[:, 0, 0] = np.linspace(-5.0, 0.0, H)
    hist[:, 0, 3] = speed
    hist[:, 0, 7] = 0.0
    hist[:, 0, 10:12] = (4.8, 2.0)
    if history_lane_y is not None:
        hist[:4, 0, 1] = history_lane_y
        hist[4:, 0, 1] = np.linspace(history_lane_y, 0.0, 2)

    return {
        "prefix_states": st,
        "prefix_controls": u,
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": st[0].copy(),
        "utility": np.float32(0.0),
        "feasible": np.int32(1),
        "is_nominal": np.int32(nominal),
        "prefix_macro_name": np.asarray("keep" if nominal else "stabilize"),
        "runtime_contact_elapsed_s": np.float32(elapsed),
    }


def _cfg() -> dict:
    return {"external_baselines": {"policy": {
        "contact_dt": 0.1,
        "contact_mu": 0.75,
        "pib_abs_mu": 0.75,
        "pib_max_decel_mps2": 9.0,
        "pib_motion_stop_threshold_mps": 8.0 / 3.6,
        "pib_max_active_duration_s": 2.5,
        "pib_min_abs_fraction": 0.55,
        "restoration_vehicle_mass_kg": 1750.0,
        "restoration_A1": 0.175,
        "restoration_A2": 0.0,
        "restoration_Ac_N": 900.0,
        "restoration_case1_Kdir_abs": 0.2,
        "restoration_case2_Kdir_abs": 0.5,
        "restoration_case1_tau0_s": 1.0,
        "restoration_case1_tau1_s": 3.0,
        "restoration_case2_tau0_s": 1.0,
        "restoration_case2_tau1_s": 5.195,
        "restoration_tau2_s": 10.0,
        "restoration_tau3_s": 11.0,
        "restoration_tauc1_s": 5.443,
        "restoration_tauc2_s": 10.0,
        "comp_mpc_beta_limit_rad": 0.12,
        "comp_mpc_yaw_rate_limit_radps": 0.8,
        "comp_mpc_steer_max_rad": 0.55,
        "comp_mpc_steer_rate_max_radps": 1.5,
        "comp_mpc_slip_ratio_limit": 0.2,
        "comp_mpc_mu": 0.85,
        "robust_pic_vehicle_mass_kg": 1270.0,
        "robust_pic_vehicle_iz_kgm2": 1536.7,
        "robust_pic_a_m": 1.015,
        "robust_pic_b_m": 1.895,
        "robust_pic_wheel_radius_m": 0.325,
        "robust_pic_track_m": 1.675,
        "robust_pic_cornering_front_Nprad": 50000.0,
        "robust_pic_cornering_rear_Nprad": 65000.0,
        "robust_pic_mu": 0.85,
        "robust_pic_c1": 0.6,
        "robust_pic_c2": 1.0,
        "robust_pic_k1": 0.25,
        "robust_pic_k2": 0.005,
        "robust_pic_qp_xi": 0.5,
        "robust_pic_fault_factors": [0.0, 0.0, 0.0, 0.0],
        "robust_pic_wheel_torque_max_nm": 1600.0,
        "robust_pic_allocation_residual_tolerance": 1.0e9,
    }}}


def test_lu2017_pib_selects_abs_like_braking_and_releases_after_timeout() -> None:
    a_abs = 0.75 * 9.81
    samples = [
        _sample(nominal=True, accel=0.0, speed=10.0, elapsed=0.0),
        _sample(accel=-a_abs, speed=10.0, elapsed=0.0),
    ]
    sel = select_external_policy("post_crash_braking", samples, _cfg())
    assert sel.reason == "lu2017_postimpact_braking_abs_candidate_port"
    assert sel.selected_index == 1
    assert bool(sel.admitted[1])

    timed_out = [copy.deepcopy(x) for x in samples]
    for d in timed_out:
        d["runtime_contact_elapsed_s"] = np.float32(2.6)
    sel2 = select_external_policy("post_crash_braking", timed_out, _cfg())
    assert sel2.selected_index == 0


def test_ghosh2026_restoration_uses_absolute_paper_time_and_exposes_A2_ambiguity() -> None:
    cfg = _cfg()
    elapsed = 1.5
    t = elapsed + np.arange(11) * 0.1
    target = -0.2 * 0.175 * np.where((t >= 1.0) & (t <= 3.0), np.sin(np.pi * (t - 1.0) / 2.0), 0.0)
    samples = [
        _sample(nominal=True, steer=0.0, speed=10.0, yaw_rate=0.05, vy=0.2, elapsed=elapsed),
        _sample(steer=target, speed=10.0, yaw_rate=0.05, vy=0.2, elapsed=elapsed),
    ]
    port = post_collision_restoration_port(samples, cfg)
    assert port.diagnostics["source_open_loop"] is True
    assert port.diagnostics["source_A2_unreported"] is True
    np.testing.assert_allclose(port.diagnostics["steer_reference"], target, atol=1e-7)
    sel = select_external_policy("post_collision_restoration", samples, cfg)
    assert sel.selected_index == 1
    for d in samples:
        d["runtime_contact_elapsed_s"] = np.float32(0.0)
    p0 = post_collision_restoration_port(samples, cfg)
    np.testing.assert_allclose(p0.diagnostics["steer_reference"][:10], 0.0, atol=1e-8)


def test_postimpact_reference_frame_preserves_observed_preimpact_lane_error() -> None:
    d = _sample(nominal=True, history_lane_y=1.0)
    st = np.asarray(d["prefix_states"])[None, ...]
    frame = _path_frame(st, d)
    assert frame["reference_source"] == "observed_ego_history_line"
    assert abs(float(np.asarray(frame["y"])[0, 0])) > 0.5


def test_ao2022_box_qp_matches_unconstrained_closed_form_and_respects_bounds() -> None:
    H = np.asarray([[1.0, 1.0, 1.0, 1.0], [-0.8, 0.8, -0.8, 0.8]])
    w = np.asarray([0.3, 0.4, 0.5, 0.6])
    xi = 0.5
    v = np.asarray([[1.0, 0.25], [-0.7, 0.8]])
    huge = np.full(4, 100.0)
    got, obj = _solve_box_qp_batch(v, H, w, xi, huge)
    Q = np.diag(w) + xi * H.T @ H
    expected = np.stack([np.linalg.solve(Q, xi * H.T @ vv) for vv in v])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
    assert np.isfinite(obj).all()

    bound = np.full(4, 0.08)
    clipped, obj2 = _solve_box_qp_batch(v * 100.0, H, w, xi, bound)
    assert np.all(np.abs(clipped) <= bound[None, :] + 1e-10)
    zero_obj = xi * np.sum((v * 100.0) ** 2, axis=1)
    assert np.all(obj2 <= zero_obj + 1e-8)


def test_ao2022_port_uses_published_sliding_surface_and_source_gains() -> None:
    cfg = _cfg()
    samples = [_sample(nominal=True, history_lane_y=1.0, yaw_rate=0.2)]
    port = robust_postimpact_control_port(samples, cfg)
    s = np.asarray(port.diagnostics["sliding_surface"])
    frame = _path_frame(np.asarray(samples[0]["prefix_states"])[None, ...], samples[0])
    expected = 0.6 * np.asarray(frame["r"]) + np.asarray(frame["y"])
    np.testing.assert_allclose(s, expected[:, : s.shape[1]], rtol=1e-8, atol=1e-8)
    assert port.diagnostics["qp_exact_active_set_enumeration"] is True


def test_four_v57_contact_ports_are_predictor_free_and_teacher_invariant(monkeypatch) -> None:
    samples = [_sample(nominal=True, accel=0.0), _sample(accel=-7.0, steer=-0.03)]

    def forbidden(*args, **kwargs):
        raise AssertionError("v57 source contact ports must not build learned risk predictions")

    monkeypatch.setattr(policies, "observed_risk_profiles_and_context", forbidden)
    methods = [
        "post_crash_braking",
        "post_collision_restoration",
        "compensatory_postimpact_mpc",
        "robust_postimpact_control",
    ]
    for method in methods:
        a = select_external_policy(method, samples, _cfg())
        mutated = copy.deepcopy(samples)
        for i, d in enumerate(mutated):
            d["hard_violation"] = np.float32(999.0 + i)
            d["harm_proxy"] = np.float32(-999.0 - i)
            d["m_star"] = np.full((3, 2), 1234.0 + i, dtype=np.float32)
        b = select_external_policy(method, mutated, _cfg())
        assert a.selected_index == b.selected_index, method
        np.testing.assert_allclose(a.score, b.score, err_msg=method)


def test_cao2021_port_reports_source_limited_fidelity_not_equation_exact() -> None:
    samples = [_sample(nominal=True), _sample(steer=-0.05)]
    from ocrap.external_baselines.paper_core_ports_v57 import compensatory_postimpact_mpc_port
    port = compensatory_postimpact_mpc_port(samples, _cfg())
    assert port.diagnostics["fidelity"] == "source_limited_abstract_structured"
    assert port.diagnostics["requires_full_paper_for_equation_exact_port"] is True
