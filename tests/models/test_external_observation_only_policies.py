from __future__ import annotations

import copy

import numpy as np

from ocrap.external_baselines.evaluate import _yaw_rate_violation_proxy
from ocrap.external_baselines.observed_risk import observed_risk_profile
from ocrap.external_baselines.policies import select_external_policy


def _sample(*, detour: bool, nominal: bool) -> dict:
    T = 11
    x = np.linspace(0.0, 10.0, T)
    y = np.linspace(0.0, 7.0 if detour else 0.0, T)
    states = np.zeros((T, 9), dtype=np.float32)
    states[:, 0] = x
    states[:, 1] = y
    states[:, 2] = np.gradient(x, 0.1)
    states[:, 3] = np.gradient(y, 0.1)
    states[:, 4] = np.arctan2(states[:, 3], np.maximum(states[:, 2], 1e-3))
    states[:, 5] = np.gradient(states[:, 4], 0.1)
    states[:, 6] = np.hypot(states[:, 2], states[:, 3])
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((T, 2), dtype=np.float32)

    hist = np.zeros((3, 2, 16), dtype=np.float32)
    valid = np.ones((3, 2), dtype=bool)
    hist[:, 0, 0] = 0.0
    hist[:, 0, 1] = 0.0
    hist[:, 0, 3] = 10.0
    hist[:, 0, 7] = 0.0
    hist[:, 0, 10] = 4.8
    hist[:, 0, 11] = 2.0
    hist[:, 1, 0] = 5.0
    hist[:, 1, 1] = 0.0
    hist[:, 1, 3] = 0.0
    hist[:, 1, 7] = 0.0
    hist[:, 1, 10] = 4.8
    hist[:, 1, 11] = 2.0

    return {
        "prefix_states": states,
        "prefix_controls": controls,
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": states[0],
        "utility": np.float32(0.0),
        "feasible": np.int32(1),
        "is_nominal": np.int32(nominal),
        "prefix_macro_name": np.asarray("keep" if nominal else "lane_shift"),
        "m_star": np.asarray([[2.0, -1.0], [2.0, -1.0]], dtype=np.float32),
        "root_probs": np.asarray([0.5, 0.5], dtype=np.float32),
        "root_valid": np.asarray([1, 1], dtype=bool),
        "option_valid": np.asarray([1, 1], dtype=bool),
        "r_orc_star": np.float32(2.0),
        "r_dep_star": np.float32(2.0),
        "hard_violation": np.float32(0.0),
        "harm_proxy": np.float32(0.0),
    }


def _cfg() -> dict:
    return {
        "external_baselines": {
            "policy": {
                "risk_dt": 0.1,
                "expected_risk_threshold": 2.0,
                "cvar_risk_threshold": 2.0,
                "dro_cvar_threshold": 2.0,
                "marc_risk_threshold": 2.0,
                "racp_risk_threshold": 2.0,
                "conformal_prediction_intervals_m": [0.0] * 7,
            }
        }
    }


def test_observed_risk_detects_conflicting_candidate() -> None:
    collision = observed_risk_profile(_sample(detour=False, nominal=True), _cfg())
    detour = observed_risk_profile(_sample(detour=True, nominal=False), _cfg())
    assert collision.expected_loss > detour.expected_loss
    assert collision.min_clearance < detour.min_clearance


def test_nonoracle_policies_are_invariant_to_teacher_label_mutation() -> None:
    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    mutated = copy.deepcopy(samples)
    for i, d in enumerate(mutated):
        d["m_star"] = -np.asarray(d["m_star"]) * (10.0 + i)
        d["r_orc_star"] = np.float32(-100.0 + i)
        d["r_dep_star"] = np.float32(100.0 - i)
        d["hard_violation"] = np.float32(50.0 * i)
        d["harm_proxy"] = np.float32(100.0 * (1 - i))
    methods = [
        "pdm_closed", "pdm_hybrid", "idm",
        "marc_lite", "racp_lite", "robust_scenario_mpc",
        "predictive_safety_filter", "dr_cvar_safety_filter",
        "conformal_predictive_safety_filter",
        "expected_risk_filter", "cvar_risk_filter", "dro_cvar_filter",
        "postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr",
        "post_collision_restoration", "compensatory_postimpact_mpc",
        "robust_postimpact_control", "severity_minimization",
    ]
    for method in methods:
        a = select_external_policy(method, samples, _cfg())
        b = select_external_policy(method, mutated, _cfg())
        assert a.selected_index == b.selected_index, method
        np.testing.assert_allclose(a.score, b.score, err_msg=method)


def test_learned_policy_selection_uses_logits_only() -> None:
    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    outputs = {
        "logits": np.asarray([0.0, 3.0]),
        "utility": np.asarray([1000.0, -1000.0]),
        "hard": np.asarray([0.0, 999.0]),
        "harm": np.asarray([0.0, 999.0]),
        "r_orc": np.asarray([999.0, -999.0]),
    }
    for method in ["gameformer_lite", "plantf", "pluto", "wayformer_bc", "betopnet_lite"]:
        sel = select_external_policy(method, samples, _cfg(), model_outputs=outputs)
        assert sel.selected_index == 1



def test_pdm_hybrid_matches_pdm_closed_before_correction_horizon() -> None:
    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    closed = select_external_policy("pdm_closed", samples, _cfg())
    hybrid = select_external_policy("pdm_hybrid", samples, _cfg())
    assert hybrid.selected_index == closed.selected_index
    np.testing.assert_allclose(hybrid.score, closed.score)
    assert "source_semantics_closed_prefix" in hybrid.reason

def test_yaw_rate_proxy_uses_schema_channel_five() -> None:
    d = _sample(detour=False, nominal=True)
    d["prefix_states"][:, 2] = 100.0  # vx must not be interpreted as heading
    d["prefix_states"][:, 5] = 0.2
    assert _yaw_rate_violation_proxy(d, yaw_rate_max=0.6) == 0.0
    d["prefix_states"][3, 5] = 0.8
    assert _yaw_rate_violation_proxy(d, yaw_rate_max=0.6) == 1.0


def test_control_smoothness_uses_heading_not_vx() -> None:
    from ocrap.external_baselines.policies import _control_smoothness_cost

    d = _sample(detour=False, nominal=True)
    states = d["prefix_states"].copy()
    states[:, 2] = np.linspace(-100.0, 100.0, states.shape[0])  # vx variation
    states[:, 4] = 0.0  # constant heading
    d["prefix_states"] = states
    assert _control_smoothness_cost(d, dt=0.1) < 1.0


def test_marc_never_selects_rejected_candidate_when_admitted_exists() -> None:
    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    samples[0]["utility"] = np.float32(1000.0)
    samples[1]["utility"] = np.float32(0.0)
    cfg = _cfg()
    p0 = observed_risk_profile(samples[0], cfg)
    p1 = observed_risk_profile(samples[1], cfg)
    tol = 0.35
    risk0 = (1.0 - tol) * p0.expected_loss + tol * p0.cvar_loss
    risk1 = (1.0 - tol) * p1.expected_loss + tol * p1.cvar_loss
    assert risk0 > risk1
    cfg["external_baselines"]["policy"].update({
        "marc_risk_threshold": 0.5 * (risk0 + risk1),
        "marc_utility_weight": 10.0,
    })
    sel = select_external_policy("marc_lite", samples, cfg)
    assert sel.admitted.any()
    assert sel.admitted[sel.selected_index]


def test_min_ttc_is_first_threshold_entry_not_closest_approach_time() -> None:
    d = _sample(detour=False, nominal=True)
    d["agent_history"][:, 1, 0] = 10.0
    profile = observed_risk_profile(d, _cfg())
    assert np.isclose(profile.min_ttc, 0.5)
    assert np.min(profile.closest_approach_time_by_mode) >= profile.min_ttc
    assert np.any(profile.closest_approach_time_by_mode > profile.min_ttc)


def test_group_risk_scoring_reuses_actor_forecast(monkeypatch) -> None:
    import ocrap.external_baselines.observed_risk as risk_module

    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    original = risk_module.build_observed_risk_context
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(risk_module, "build_observed_risk_context", counted)
    grouped = risk_module.observed_risk_profiles(samples, _cfg())
    assert calls == 1
    independent = [original(d, _cfg()) for d in samples]
    rescored = [risk_module.score_candidate_with_context(d, _cfg(), c) for d, c in zip(samples, independent)]
    for a, b in zip(grouped, rescored):
        np.testing.assert_allclose(a.losses, b.losses)
        assert np.isclose(a.expected_loss, b.expected_loss)
        assert np.isclose(a.cvar_loss, b.cvar_loss)


def test_imitation_target_never_falls_back_to_ocrap_feasibility() -> None:
    from ocrap.external_baselines.data import _target_index

    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    samples[0]["feasible"] = np.int32(0)
    samples[1]["feasible"] = np.int32(1)
    cfg = {"external_baselines": {"supervision_target": "logged_nominal", "allow_teacher_supervision": False}}
    assert _target_index(samples, "plantf", cfg) == 0


def test_main_near_filters_use_explicit_safest_fallback_when_admission_is_empty() -> None:
    unsafe = _sample(detour=False, nominal=True)
    safer = _sample(detour=True, nominal=False)
    # Make candidate 1 a controlled stop before the observed obstacle. Candidate
    # 0 gets huge utility so the old "score over all feasible" fallback would
    # incorrectly choose it whenever no safety constraint was satisfied.
    states = safer["prefix_states"].copy()
    states[:, 0] = 0.0
    states[:, 1] = 0.0
    states[:, 2:7] = 0.0
    safer["prefix_states"] = states
    safer["prefix_controls"][:] = 0.0
    unsafe["utility"] = np.float32(10000.0)
    safer["utility"] = np.float32(0.0)

    cfg = _cfg()
    cfg["external_baselines"]["policy"].update({
        "scenario_mpc_worst_risk_gate": -1.0,
        "scenario_mpc_min_clearance_gate_m": 100.0,
        "marc_risk_threshold": -1.0,
        "marc_chance_threshold": -1.0,
        "racp_risk_threshold": -1.0,
        "racp_chance_threshold": -1.0,
        "dr_cvar_threshold": -1.0,
        "dr_cvar_min_clearance_gate_m": 100.0,
        "conformal_prediction_intervals_m": [100.0] * 7,
        "psf_backup_margin_m": 100.0,
        "psf_terminal_backup_margin_m": 100.0,
    })
    methods = [
        "robust_scenario_mpc",
        "marc_lite",
        "racp_lite",
        "dr_cvar_safety_filter",
        "conformal_predictive_safety_filter",
        "predictive_safety_filter",
    ]
    for method in methods:
        sel = select_external_policy(method, [unsafe, safer], cfg)
        assert not sel.admitted.any(), method
        if method not in {"dr_cvar_safety_filter"}:
            assert sel.selected_index == 1, method
        assert "empty_admissible_set_safest_fallback" in sel.reason, method


def _fake_profile(*, clearance: np.ndarray, loss: np.ndarray, backup: np.ndarray, distinguish: np.ndarray | None = None) -> object:
    from ocrap.external_baselines.observed_risk import ObservedRiskProfile

    clearance = np.asarray(clearance, dtype=float)
    loss = np.asarray(loss, dtype=float)
    backup = np.asarray(backup, dtype=float)
    H, _ = clearance.shape
    weights = np.full(H, 1.0 / H, dtype=float)
    losses = np.max(loss, axis=1)
    return ObservedRiskProfile(
        losses=losses,
        weights=weights,
        margins=np.min(clearance, axis=1),
        min_clearance=float(np.min(clearance)),
        min_ttc=float("inf"),
        collision_probability=float(np.sum(weights * np.any(clearance <= 0.0, axis=1))),
        expected_loss=float(np.sum(weights * losses)),
        cvar_loss=float(np.max(losses)),
        worst_loss=float(np.max(losses)),
        backup_margin=float(np.min(backup)),
        severity_proxy=0.0,
        hypothesis_names=tuple(f"m{i}" for i in range(H)),
        collision_probabilities=np.any(clearance <= 0.0, axis=1).astype(float),
        min_ttc_by_mode=np.full(H, np.inf),
        closest_approach_time_by_mode=np.zeros(H),
        severity_by_mode=np.zeros(H),
        clearance_curves=clearance,
        loss_curves=loss,
        backup_margin_curves=backup,
        mode_distinguish_step=np.asarray(distinguish if distinguish is not None else np.zeros((H, H)), dtype=np.int32),
        weight_source="test",
    )


def test_predictive_safety_filter_preserves_safe_proposed_input_and_minimally_corrects_unsafe_one() -> None:
    nominal = _sample(detour=False, nominal=True)
    alternate = _sample(detour=True, nominal=False)
    alternate["utility"] = np.float32(10000.0)  # must not override minimal intervention
    safe = np.full((2, 11), 5.0)
    safe_backup = np.full((2, 11), 2.0)
    safe_loss = np.zeros((2, 11))
    profiles = [
        _fake_profile(clearance=safe, loss=safe_loss, backup=safe_backup),
        _fake_profile(clearance=safe, loss=safe_loss, backup=safe_backup),
    ]
    cfg = _cfg()
    sel = select_external_policy("predictive_safety_filter", [nominal, alternate], cfg, precomputed_profiles=profiles)
    assert sel.selected_index == 0
    assert sel.admitted[0]

    unsafe = safe.copy(); unsafe[:, 4:] = -1.0
    unsafe_backup = safe_backup.copy(); unsafe_backup[:, -1] = -1.0
    profiles[0] = _fake_profile(clearance=unsafe, loss=np.ones((2, 11)), backup=unsafe_backup)
    sel = select_external_policy("predictive_safety_filter", [nominal, alternate], cfg, precomputed_profiles=profiles)
    assert not sel.admitted[0]
    assert sel.admitted[1]
    assert sel.selected_index == 1


def test_robust_scenario_mpc_uses_mode_dependent_recourse_only_after_distinction() -> None:
    a = _sample(detour=False, nominal=True)
    b = _sample(detour=False, nominal=False)
    # Same non-anticipative prefix through step 2, different tails afterwards.
    b["prefix_states"] = a["prefix_states"].copy()
    b["prefix_controls"] = a["prefix_controls"].copy()
    b["prefix_states"][3:, 1] += 2.0
    b["prefix_controls"][3:, 1] += 0.2
    distinguish = np.asarray([[0, 2], [2, 0]], dtype=np.int32)
    c0 = np.full((2, 11), 2.0); c0[1, 3:] = -1.0
    c1 = np.full((2, 11), 2.0); c1[0, 3:] = -1.0
    p0 = _fake_profile(clearance=c0, loss=np.where(c0 < 0, 2.0, 0.1), backup=c0, distinguish=distinguish)
    p1 = _fake_profile(clearance=c1, loss=np.where(c1 < 0, 2.0, 0.1), backup=c1, distinguish=distinguish)
    cfg = _cfg()
    cfg["external_baselines"]["policy"].update({
        "scenario_mpc_min_clearance_gate_m": 0.0,
        "scenario_mpc_max_mode_stage_loss_guard": 4.0,
        "scenario_mpc_state_tie_threshold_m": 0.1,
        "scenario_mpc_control_tie_accel_tol_mps2": 0.1,
        "scenario_mpc_control_tie_steer_tol_rad": 0.1,
    })
    sel = select_external_policy("robust_scenario_mpc", [a, b], cfg, precomputed_profiles=[p0, p1])
    # Neither full trajectory is robust by itself; both roots become robust only
    # because the other candidate is a legal post-distinction recourse.
    assert sel.admitted.tolist() == [True, True]
    assert "mode_distinction_nonanticipative" in sel.reason



def test_robust_scenario_mpc_uses_pairwise_not_global_mode_distinction() -> None:
    # Mode 0 becomes distinguishable from {1,2} early, while modes 1 and 2
    # remain mutually indistinguishable.  Eq. (7f) permits an early 0-vs-{1,2}
    # split; a single global latest branch time would incorrectly forbid it.
    a = _sample(detour=False, nominal=True)
    b = _sample(detour=False, nominal=False)
    b["prefix_states"] = a["prefix_states"].copy()
    b["prefix_controls"] = a["prefix_controls"].copy()
    a["prefix_states"][2:, 1] += 2.0
    a["prefix_controls"][2:, 1] += 0.2
    distinguish = np.asarray([[0, 2, 2], [2, 0, 6], [2, 6, 0]], dtype=np.int32)
    ca = np.full((3, 11), 2.0); ca[1:, 2:] = -1.0
    cb = np.full((3, 11), 2.0); cb[0, 2:] = -1.0
    pa = _fake_profile(clearance=ca, loss=np.where(ca < 0, 2.0, 0.1), backup=ca, distinguish=distinguish)
    pb = _fake_profile(clearance=cb, loss=np.where(cb < 0, 2.0, 0.1), backup=cb, distinguish=distinguish)
    cfg = _cfg()
    cfg["external_baselines"]["policy"].update({
        "scenario_mpc_min_clearance_gate_m": 0.0,
        "scenario_mpc_max_mode_stage_loss_guard": 4.0,
        "scenario_mpc_state_tie_threshold_m": 0.1,
        "scenario_mpc_control_tie_accel_tol_mps2": 0.1,
        "scenario_mpc_control_tie_steer_tol_rad": 0.1,
        "scenario_mpc_pairwise_beam_size": 32,
    })
    sel = select_external_policy("robust_scenario_mpc", [a, b], cfg, precomputed_profiles=[pa, pb])
    assert sel.admitted.tolist() == [True, True]


def test_robust_scenario_mpc_does_not_make_internal_loss_an_unpublished_default_constraint() -> None:
    sample = _sample(detour=False, nominal=True)
    clearance = np.full((2, 11), 2.0)
    # Keep the physical clearance robustly safe but make the internal risk loss
    # much larger than the historical 4.0 guard.  The paper-core default should
    # still admit it; the loss belongs in the expected objective, not a hidden
    # robust constraint.
    profile = _fake_profile(
        clearance=clearance,
        loss=np.full((2, 11), 20.0),
        backup=clearance,
        distinguish=np.asarray([[0, 0], [0, 0]], dtype=np.int32),
    )
    cfg = _cfg()
    cfg["external_baselines"]["policy"].update({
        "scenario_mpc_min_clearance_gate_m": 0.0,
    })
    sel = select_external_policy(
        "robust_scenario_mpc", [sample], cfg, precomputed_profiles=[profile]
    )
    assert sel.admitted.tolist() == [True]

def test_marc_scene_branch_time_comes_from_scenario_conditioned_futures() -> None:
    from ocrap.external_baselines.policies import _latest_divergence_branch_step

    a = _sample(detour=False, nominal=True)
    b = _sample(detour=False, nominal=False)
    a["prefix_states"] = a["prefix_states"][:5].copy()
    b["prefix_states"] = a["prefix_states"].copy()
    b["prefix_states"][3:, 1] += 2.0
    branch = _latest_divergence_branch_step(
        [a, b], [0, 1], horizon=5, threshold_m=0.5, max_fraction=1.0
    )
    assert branch == 2


def test_shared_mode_probabilities_are_observation_conditioned_not_fixed_priors() -> None:
    from ocrap.external_baselines.observed_risk import build_observed_risk_context

    accelerating = _sample(detour=True, nominal=True)
    braking = copy.deepcopy(accelerating)
    accelerating["agent_history"][:, 1, 5] = 2.0
    braking["agent_history"][:, 1, 5] = -3.0
    cfg = _cfg()
    cfg["external_baselines"]["policy"]["risk_observation_conditioned_mode_weights"] = True
    ca = build_observed_risk_context(accelerating, cfg)
    cb = build_observed_risk_context(braking, cfg)
    assert ca.weight_source == "observation_conditioned_kinematic_belief"
    assert np.isclose(ca.weights.sum(), 1.0)
    assert np.isclose(cb.weights.sum(), 1.0)
    assert not np.allclose(ca.weights, cb.weights)
