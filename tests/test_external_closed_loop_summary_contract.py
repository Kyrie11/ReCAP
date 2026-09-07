from __future__ import annotations

import importlib.util
from pathlib import Path


def _summary_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "summarize_external_closed_loop.py"
    spec = importlib.util.spec_from_file_location("summarize_external_closed_loop", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def test_safe_publication_summary_keeps_nominal_safety_comfort_and_preservation_contract() -> None:
    m = _summary_module()
    expected = {
        "collision_scene_rate", "offroad_scene_rate", "minimum_clearance_m",
        "scene_min_clearance_m_median", "scene_min_clearance_m_p05",
        "minimum_ttc_s", "scene_ttc_s_median", "scene_ttc_s_p05",
        "acceleration_abs_p95_mps2", "jerk_p95", "yaw_rate_p95",
        "acceleration_max_mps2", "deceleration_max_mps2", "jerk_max_abs",
        "yaw_rate_max_abs", "route_progression_m", "closed_loop_bounded_NUP",
        "closed_loop_nominal_deviation", "intervention_rate", "intervention_scene_rate",
    }
    assert expected <= set(m.SAFE)

def test_near_publication_summary_keeps_full_low_headroom_recovery_contract() -> None:
    m = _summary_module()
    expected = {
        "collision_scene_rate",
        "minimum_clearance_m",
        "scene_min_clearance_m_p05",
        "minimum_ttc_s",
        "scene_ttc_s_p05",
        "near_contact_exposure_rate",
        "near_contact_exposure_duration_s",
        "near_contact_exposure_episode_count",
        "near_contact_longest_exposure_run_s",
        "critical_ttc_exposure_rate",
        "critical_ttc_exposure_duration_s",
        "critical_ttc_exposure_episode_count",
        "critical_ttc_longest_exposure_run_s",
        "near_zero_clearance_exposure_rate",
        "time_to_min_clearance_s",
        "terminal_clearance_m",
        "clearance_recovery_gain_m",
        "time_to_min_ttc_s",
        "terminal_ttc_s",
        "ttc_recovery_gain_s",
        "clearance_deficit_auc_m_s",
        "ttc_deficit_auc_s2",
        "closed_loop_FRA_exec",
        "closed_loop_FRA_cand",
        "closed_loop_DRS",
        "closed_loop_ODG",
        "closed_loop_bounded_NUP",
        "acceleration_abs_p95_mps2",
        "jerk_p95",
        "yaw_rate_p95",
        "intervention_rate",
        "intervention_scene_rate",
    }
    assert expected <= set(m.NEAR)


def test_contact_publication_summary_keeps_post_contact_recovery_contract() -> None:
    m = _summary_module()
    expected = {
        "overlap_episode_count",
        "overlap_duration_s",
        "longest_overlap_run_s",
        "post_contact_terminal_clearance_m",
        "post_contact_free_space_auc_m_s",
        "post_contact_free_space_auc_normalized_m",
        "post_contact_clearance_gain_m",
        "time_to_peak_post_contact_clearance_s",
        "post_contact_escape_scene_rate",
        "time_to_post_contact_escape_s",
        "recontact_scene_rate",
        "recontact_episode_count",
        "secondary_overlap_scene_rate",
        "new_stable_stop_scene_rate",
        "new_stable_stop_quality_scene_rate",
        "time_to_stable_stop_s",
        "time_to_stable_stop_quality_s",
        "post_contact_overlap_duration_s",
        "post_contact_overlap_rate",
        "post_contact_clearance_deficit_auc_m_s",
    }
    assert expected <= set(m.CONTACT)