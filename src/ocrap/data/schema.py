from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

AGENT_FEATURES = [
    "x", "y", "z", "vx", "vy", "ax", "ay", "heading", "sin_heading", "cos_heading",
    "length", "width", "height", "object_type", "valid", "observed_confidence",
]
F_AGENT = len(AGENT_FEATURES)
F_EGO = 9  # x,y,vx,vy,heading,yaw_rate,speed,length,width
F_CTRL = 4  # acceleration, steering, jerk, steering_rate


@dataclass
class RawScenario:
    scenario_id: str
    timestamps: np.ndarray
    sdc_track_index: int
    agent_states: np.ndarray  # [T,A,16]
    agent_valid: np.ndarray  # [T,A]
    map_polylines: np.ndarray  # [P,Q,F_map]
    map_valid: np.ndarray  # [P,Q]
    route: np.ndarray  # [R,F_route]
    dynamic_map: np.ndarray  # [T,B,F_signal]
    object_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneHistory:
    scene_id: str
    original_scenario_id: str
    time_index: int
    agent_history: np.ndarray
    agent_valid: np.ndarray
    map_polylines: np.ndarray
    map_valid: np.ndarray
    dynamic_map: np.ndarray
    route: np.ndarray
    occ_mask: np.ndarray
    ego_state: np.ndarray
    future_agent_states: np.ndarray
    future_agent_valid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePrefix:
    macro_id: int
    macro_name: str
    params: np.ndarray
    prefix_states: np.ndarray
    prefix_controls: np.ndarray
    utility: float
    feasible: bool
    hard_violation: float
    harm_proxy: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualFuture:
    future_id: int
    source: str
    prior: float
    agent_states: np.ndarray
    agent_valid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryOption:
    option_id: int
    mode: str
    params: np.ndarray
    valid: bool = True


@dataclass
class Observation:
    ego_state: np.ndarray
    boxes: np.ndarray  # [N,9]
    box_valid: np.ndarray
    occ_mask: np.ndarray
    contact_flag: bool
    stability_proxy: np.ndarray
    route_visible: np.ndarray | None = None


@dataclass
class RootClusteringResult:
    assignments: np.ndarray
    root_probs: np.ndarray
    root_signature: np.ndarray
    root_valid: np.ndarray
    representative_indices: np.ndarray
    future_to_root_weight: np.ndarray
    within_root_obs_dispersion: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSample:
    scene_id: str
    original_scenario_id: str
    time_index: int
    candidate_index: int
    split_id: str
    is_nominal: bool
    h_t: SceneHistory
    prefix: CandidatePrefix
    futures: list[CounterfactualFuture]
    future_probs: np.ndarray
    root_assignments: np.ndarray
    root_probs: np.ndarray
    root_signature: np.ndarray
    root_future_signature: np.ndarray
    root_valid: np.ndarray
    root_representative_future_id: np.ndarray
    future_to_root_weight: np.ndarray
    within_root_obs_dispersion: np.ndarray
    obs_distance: np.ndarray
    y_obs: np.ndarray
    c_star: np.ndarray
    recovery_options: list[RecoveryOption]
    m_star: np.ndarray
    option_valid: np.ndarray
    r_orc_star: float
    r_dep_star: float
    oracle_gap_star: float
    i_art_star: bool
    regime_label: dict[str, bool]
    valid_masks: dict[str, Any]
    teacher_diagnostics: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_npz_dict(self) -> dict[str, Any]:
        h = self.h_t
        p = self.prefix
        return {
            "scene_id": self.scene_id,
            "original_scenario_id": self.original_scenario_id,
            "official_scenario_id": str(h.metadata.get("official_scenario_id") or ""),
            "legacy_scenario_id": str(h.metadata.get("legacy_scenario_id") or ""),
            "source_scenario_index": np.int64(-1 if h.metadata.get("source_scenario_index", -1) is None else h.metadata.get("source_scenario_index", -1)),
            "scenario_id_source": str(h.metadata.get("scenario_id_source", "unknown")),
            "womd_source_role": str(h.metadata.get("womd_source_role", "unknown")),
            "womd_source_pattern": str(h.metadata.get("womd_source_pattern", "")),
            "waymax_max_num_objects": np.int64(h.metadata.get("waymax_max_num_objects", -1) or -1),
            "time_index": np.int64(self.time_index),
            "candidate_index": np.int64(self.candidate_index),
            "split_id": self.split_id,
            "is_nominal": np.int64(self.is_nominal),
            "agent_history": h.agent_history.astype(np.float32),
            "agent_valid": h.agent_valid.astype(np.float32),
            "map_polylines": h.map_polylines.astype(np.float32),
            "map_valid": h.map_valid.astype(np.float32),
            "dynamic_map": h.dynamic_map.astype(np.float32),
            "route": h.route.astype(np.float32),
            "bev_occ": h.occ_mask.astype(np.float32),
            "ego_state": h.ego_state.astype(np.float32),
            "prefix_states": p.prefix_states.astype(np.float32),
            "prefix_controls": p.prefix_controls.astype(np.float32),
            "prefix_macro_id": np.int64(p.macro_id),
            "prefix_macro_type_id": np.int64(p.diagnostics.get("macro_type_id", p.macro_id)),
            "prefix_macro_name": p.macro_name,
            "prefix_param": p.params.astype(np.float32),
            "utility": np.float32(p.utility),
            "hard_violation": np.float32(p.hard_violation),
            "harm_proxy": np.float32(p.harm_proxy),
            "feasible": np.int64(p.feasible),
            "prefix_diagnostics": json.dumps(p.diagnostics, sort_keys=True),
            "future_probs": self.future_probs.astype(np.float32),
            "future_sources": np.asarray([f.source for f in self.futures]),
            "future_metadata": json.dumps([f.metadata for f in self.futures], sort_keys=True),
            "future_valid": np.asarray([bool(f.agent_valid.any()) for f in self.futures], dtype=np.float32),
            "root_assignments": self.root_assignments.astype(np.int64),
            "root_probs": self.root_probs.astype(np.float32),
            "root_signature": self.root_signature.astype(np.float32),
            "root_future_signature": self.root_future_signature.astype(np.float32),
            "root_valid": self.root_valid.astype(np.float32),
            "root_representative_future_id": self.root_representative_future_id.astype(np.int64),
            "future_to_root_weight": self.future_to_root_weight.astype(np.float32),
            "within_root_obs_dispersion": self.within_root_obs_dispersion.astype(np.float32),
            "obs_distance": self.obs_distance.astype(np.float32),
            "y_obs": self.y_obs.astype(np.float32),
            "c_star": self.c_star.astype(np.float32),
            "m_star": self.m_star.astype(np.float32),
            "option_valid": self.option_valid.astype(np.float32),
            "recovery_modes": np.asarray([g.mode for g in self.recovery_options]),
            "recovery_params": pad_recovery_params(self.recovery_options).astype(np.float32),
            "r_orc_star": np.float32(self.r_orc_star),
            "r_dep_star": np.float32(self.r_dep_star),
            "oracle_gap_star": np.float32(self.oracle_gap_star),
            "i_art_star": np.int64(self.i_art_star),
            "regime_label": json.dumps(self.regime_label, sort_keys=True),
            "valid_masks": json.dumps(self.valid_masks, sort_keys=True),
            "teacher_diagnostics": json.dumps(self.teacher_diagnostics, sort_keys=True),
            "diagnostics": json.dumps(self.diagnostics, sort_keys=True),
        }


def pad_recovery_params(options: list[RecoveryOption], width: int = 3) -> np.ndarray:
    out = np.zeros((len(options), width), dtype=np.float32)
    for i, g in enumerate(options):
        p = np.asarray(g.params, dtype=np.float32).reshape(-1)
        out[i, : min(width, p.size)] = p[:width]
    return out


def dataclass_to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
