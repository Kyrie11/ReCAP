from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ocrap.external_baselines.observed_risk import ObservedRiskContext, build_observed_risk_context
from ocrap.external_baselines.paper_core_ports_v56 import (
    _constant_velocity_obstacles,
    _integrated_magic_lateral,
    _kinematic_series,
    _nonlinear_allocation_residual_batch,
    _pso_allocation_residual_batch,
    _quintic_reference_kinematics,
    _static_obstacles,
    _tvlqr_tracking_cost,
    _tvlqr_tracking_cost_batch,
    _tvlqr_magic_lateral_force,
    cpsf_constrained_projection_port,
    dr_cvar_safe_halfspace_port,
)
from ocrap.external_baselines.policies import select_external_policy


def _sample(*, y_end: float = 0.0, nominal: bool = False, actor_x: float = 5.0, actor_vx: float = 0.0) -> dict:
    T = 11
    x = np.linspace(0.0, 10.0, T)
    y = np.linspace(0.0, y_end, T)
    st = np.zeros((T, 9), dtype=np.float32)
    st[:, 0], st[:, 1] = x, y
    st[:, 2], st[:, 3] = np.gradient(x, 0.1), np.gradient(y, 0.1)
    st[:, 4] = np.arctan2(st[:, 3], np.maximum(st[:, 2], 1e-3))
    st[:, 5] = np.gradient(st[:, 4], 0.1)
    st[:, 6] = np.hypot(st[:, 2], st[:, 3])
    st[:, 7], st[:, 8] = 4.8, 2.0
    hist = np.zeros((3, 2, 16), dtype=np.float32)
    valid = np.ones((3, 2), dtype=bool)
    hist[:, 0, 3] = 10.0
    hist[:, 0, 10:12] = (4.8, 2.0)
    hist[:, 1, 0] = actor_x
    hist[:, 1, 3] = actor_vx
    hist[:, 1, 10:12] = (4.8, 2.0)
    return {
        "prefix_states": st,
        "prefix_controls": np.zeros((T, 2), dtype=np.float32),
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": st[0],
        "utility": np.float32(0.0),
        "feasible": np.int32(1),
        "is_nominal": np.int32(nominal),
        "prefix_macro_name": np.asarray("keep" if nominal else "lane_shift"),
    }


def _static_context(x: float, horizon: int = 8) -> ObservedRiskContext:
    actor = np.zeros((1, 1, horizon, 2), dtype=float)
    actor[0, 0, :, 0] = x
    return ObservedRiskContext(
        hypothesis_names=("constant",), weights=np.asarray([1.0]),
        times=np.arange(horizon, dtype=float) * 0.1, actor_xy=actor,
        actor_velocity=np.zeros_like(actor), actor_radius=np.asarray([2.0]),
        clearance_buffer_m=0.0,
    )


def _near_cfg() -> dict:
    return {"external_baselines": {"policy": {
        "risk_dt": 0.1,
        "conformal_prediction_intervals_m": [0.0] * 7,
        "cpsf_prediction_horizon_steps": 7,
        "cpsf_collision_margin_m": 0.5,
        "dr_cvar_horizon_steps": 10,
        "dr_cvar_num_samples": 20,
        "dr_cvar_alpha": 0.2,
        "dr_cvar_wasserstein_radius": 0.05,
        "dr_cvar_loss_bound": 0.1,
        "dr_cvar_mpc_Q": 2.0,
        "dr_cvar_mpc_QT": 5.0,
        "dr_cvar_mpc_R": 1.0,
    }}}


def test_cpsf_algorithm1_exact_infinity_sentinel_and_quantile() -> None:
    path = Path(__file__).resolve().parents[2] / "tools" / "calibrate_external_baselines.py"
    spec = importlib.util.spec_from_file_location("calibrate_external_baselines_v56", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q, p = mod.cpsf_conformal_radius(np.asarray([1.0, 2.0, 3.0, 4.0]), 0.20)
    assert p == 4 and q == 4.0
    q, p = mod.cpsf_conformal_radius(np.asarray([1.0, 2.0, 3.0, 4.0]), 0.10)
    assert p == 5 and np.isinf(q)


def test_cpsf_calibration_restores_raw_track_order_when_sdc_is_not_zero() -> None:
    path = Path(__file__).resolve().parents[2] / "tools" / "calibrate_external_baselines.py"
    spec = importlib.util.spec_from_file_location("calibrate_external_baselines_track_order", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    states = np.zeros((2, 3, 16), dtype=np.float32)
    valid = np.ones((2, 3), dtype=bool)
    # Raw WOMD order is [track0, track1, SDC=track2].
    states[:, 2, 0] = 10.0
    states[0, 0, 0], states[1, 0, 0] = 12.0, 13.0
    states[0, 1, 0], states[1, 1, 0] = 15.0, 16.0
    raw = SimpleNamespace(agent_states=states, agent_valid=valid, sdc_track_index=2)

    hist = np.zeros((1, 3, 16), dtype=np.float32)
    # Serialized OC-RAP order is reconstructed as [SDC=2, raw0, raw1].
    hist[0, 0, 0] = 0.0; hist[0, 1, 0] = 2.0; hist[0, 2, 0] = 5.0
    sample = {"agent_history": hist, "agent_valid": np.ones((1, 3), dtype=bool)}
    point = np.zeros((2, 2, 2), dtype=float)
    point[0, 1, 0] = 2.5
    point[1, 1, 0] = 7.0
    mod._point_prediction = lambda sample, cfg, horizon: (point, [1, 2])
    score, counts, alignment = mod._group_nonconformity(raw, {"sample": sample, "time_index": 0}, {}, 1)
    np.testing.assert_allclose(score[0], np.sqrt(0.5**2 + 1.0**2), atol=1e-8)
    assert counts[0] == 2
    assert alignment < 1e-8


def test_cpsf_eq7_preserves_nominal_when_nominal_is_certified() -> None:
    samples = [_sample(nominal=True, actor_x=50.0), _sample(y_end=5.0, actor_x=50.0)]
    cfg = _near_cfg()
    ctx = _static_context(50.0, 8)
    port = cpsf_constrained_projection_port(samples, cfg, ctx)
    assert bool(port.admitted[0])
    sel = select_external_policy("conformal_predictive_safety_filter", samples, cfg, precomputed_context=ctx)
    assert sel.selected_index == 0


def test_cpsf_eq7_is_center_distance_C_plus_epsilon_without_extra_vehicle_radii() -> None:
    samples = [_sample(nominal=True, actor_x=5.0), _sample(y_end=7.0, actor_x=5.0)]
    cfg = _near_cfg()
    ctx = _static_context(30.0 / 7.0, 8)
    port = cpsf_constrained_projection_port(samples, cfg, ctx)
    assert not bool(port.admitted[0])
    assert bool(port.admitted[1])
    # The diagnostic is exactly distance - (C_h + epsilon), not distance minus
    # OC-RAP bounding circles, which are absent from the cited Eq. (7).
    assert port.diagnostics["collision_margin_m"] == 0.5


def test_dr_cvar_wasserstein_radius_cannot_relax_halfspaces() -> None:
    samples = [_sample(nominal=True), _sample(y_end=7.0)]
    cfg0 = _near_cfg()
    ctx = build_observed_risk_context(samples[0], cfg0, horizon=11)
    p0 = dr_cvar_safe_halfspace_port(samples, cfg0, ctx)
    cfg1 = _near_cfg()
    cfg1["external_baselines"]["policy"]["dr_cvar_wasserstein_radius"] = 0.5
    p1 = dr_cvar_safe_halfspace_port(samples, cfg1, ctx)
    np.testing.assert_array_less(
        np.asarray(p1.diagnostics["min_halfspace_margin_m"]),
        np.asarray(p0.diagnostics["min_halfspace_margin_m"]) + 1e-12,
    )


def test_wang2023_magic_formula_uses_kN_degrees_and_friction_similarity_units() -> None:
    cfg = {"external_baselines": {"policy": {
        "postimpact_mu": 1.0, "postimpact_magic_mu0": 1.0,
        "postimpact_magic_dy1": -6.233, "postimpact_magic_dy2": 990.2,
        "postimpact_magic_cy": 1.466, "postimpact_magic_by": 0.1544,
        "postimpact_Lf_m": 1.05, "postimpact_Lr_m": 1.61, "postimpact_track_m": 1.565,
    }}}
    alpha = np.deg2rad(5.0)
    out = _integrated_magic_lateral(
        np.asarray([alpha]), np.asarray([10.0]), np.asarray([0.0]), np.asarray([0.0]),
        np.asarray([[4000.0] * 4]), np.zeros((1, 4)), cfg,
    )
    dy1, dy2, cy, by = -6.233, 990.2, 1.466, 0.1544
    expected = (dy1 * 4.0**2 + dy2 * 4.0) * np.sin(cy * np.arctan(by * 5.0))
    np.testing.assert_allclose(out[0, 0], expected, rtol=1e-8, atol=1e-8)


def test_wang2022_full_magic_formula_uses_kN_degrees() -> None:
    cfg = {"external_baselines": {"policy": {
        "tvlqr_magic_mu0": 1.0,
        "tvlqr_magic_Cy": 1.141, "tvlqr_magic_b1": -5.98,
        "tvlqr_magic_b2": 965.7, "tvlqr_magic_b3": 2536.0,
        "tvlqr_magic_b4": 2.071, "tvlqr_magic_b5": 0.04436,
        "tvlqr_magic_b6": -0.04443, "tvlqr_magic_b7": 0.5792,
        "tvlqr_magic_b8": -3.076,
    }}}
    alpha = np.asarray([np.deg2rad(3.0), np.deg2rad(8.0)])
    got = _tvlqr_magic_lateral_force(alpha, 4000.0, 1.0, cfg)
    Cy, b1, b2, b3, b4, b5, b6, b7, b8 = 1.141, -5.98, 965.7, 2536.0, 2.071, 0.04436, -0.04443, 0.5792, -3.076
    Fz = 4.0
    Dy = b1 * Fz**2 + b2 * Fz
    By = b3 * np.sin(b4 * np.arctan(b5 * Fz)) / (Cy * Dy)
    Ey = b6 * Fz**2 + b7 * Fz + b8
    a = np.asarray([3.0, 8.0])
    z = By * a
    expected = Dy * np.sin(Cy * np.arctan(z - Ey * (z - np.arctan(z))))
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_wang2023_obstacle_prediction_is_constant_velocity_from_observation() -> None:
    d = _sample(nominal=True, actor_x=5.0, actor_vx=2.0)
    obs = _constant_velocity_obstacles(d, 4, 0.1)
    assert obs.shape == (1, 4, 2)
    np.testing.assert_allclose(obs[0, :, 0], [5.0, 5.2, 5.4, 5.6], atol=1e-7)
    np.testing.assert_allclose(obs[0, :, 1], 0.0, atol=1e-7)


def test_wang2022_quintic_projection_enforces_terminal_lateral_and_yaw_rates() -> None:
    d = _sample(y_end=3.0, nominal=True, actor_x=50.0)
    cfg = {"external_baselines": {"policy": {"contact_dt": 0.1}}}
    kin, fit, _ = _quintic_reference_kinematics(d, cfg, 11)
    assert np.isfinite(fit)
    assert abs(float(kin["vyg"][-1])) < 1e-7
    assert abs(float(kin["r"][-1])) < 1e-7


def test_wang2022_apf_uses_fixed_current_obstacle_coordinates() -> None:
    d = _sample(nominal=True, actor_x=5.0, actor_vx=20.0)
    obs = _static_obstacles(d)
    assert obs.shape == (1, 2)
    # Eq. (4) has one perceived (X_b,Y_b), not a learned future trajectory.
    np.testing.assert_allclose(obs[0], [5.0, 0.0], atol=1e-8)


def test_batched_tvlqr_riccati_matches_scalar_candidate_recursions() -> None:
    cfg = {"external_baselines": {"policy": {
        "contact_dt": 0.1,
        "tvlqr_vehicle_mass": 1610.0,
        "tvlqr_vehicle_iz": 2059.0,
        "tvlqr_Q_diag": [5, 5, 90, 6e5, 5e5, 1e6],
        "tvlqr_R_diag": [1e-4, 1e-4, 1e-4],
        "tvlqr_dare_iterations": 12,
    }}}
    samples = [_sample(y_end=1.5, nominal=True), _sample(y_end=-2.0)]
    for i, d in enumerate(samples):
        ego = np.asarray(d["ego_state"], dtype=float).copy()
        ego[1] += 0.25 * (i + 1); ego[4] += 0.03 * (i + 1); ego[5] += 0.05
        d["ego_state"] = ego
    kin = [_quintic_reference_kinematics(d, cfg, 11)[0] for d in samples]
    scalar = np.asarray([_tvlqr_tracking_cost(d, cfg, 11, kin=k) for d, k in zip(samples, kin)])
    batch = np.stack(_tvlqr_tracking_cost_batch(samples, cfg, 11, kin), axis=1)
    assert np.max(np.abs(scalar)) > 0.0
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-8)


def test_batched_contact_allocators_are_deterministic_and_finite() -> None:
    cfg = {"external_baselines": {"policy": {
        "contact_dt": 0.1,
        "postimpact_vehicle_mass": 1610.0, "postimpact_vehicle_iz": 2059.0,
        "postimpact_Lf_m": 1.05, "postimpact_Lr_m": 1.61, "postimpact_track_m": 1.565,
        "postimpact_cg_height_m": 0.55, "postimpact_mu": 0.8,
        "postimpact_pso_particles": 64, "postimpact_pso_iterations": 2,
        "tvlqr_vehicle_mass": 1610.0, "tvlqr_vehicle_iz": 2059.0,
        "tvlqr_Lf_m": 1.05, "tvlqr_Lr_m": 1.61, "tvlqr_track_m": 1.565,
        "tvlqr_cg_height_m": 0.55, "tvlqr_mu": 0.9,
        "tvlqr_allocation_candidates": 64, "tvlqr_allocation_refine_iterations": 2,
    }}}
    samples = [_sample(y_end=1.0, nominal=True), _sample(y_end=-1.0)]
    kin = [_kinematic_series(d, cfg, 11) for d in samples]
    a0 = _pso_allocation_residual_batch(kin, cfg); a1 = _pso_allocation_residual_batch(kin, cfg)
    for x, y in zip(a0, a1):
        assert np.all(np.isfinite(x)); np.testing.assert_allclose(x, y)
    qkin = [_quintic_reference_kinematics(d, cfg, 11)[0] for d in samples]
    b0 = _nonlinear_allocation_residual_batch(qkin, cfg, 11); b1 = _nonlinear_allocation_residual_batch(qkin, cfg, 11)
    for x, y in zip(b0, b1):
        assert np.all(np.isfinite(x)); np.testing.assert_allclose(x, y)
