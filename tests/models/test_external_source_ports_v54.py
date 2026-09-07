from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.external_baselines.train import _forward_model, _loss_dict

ROOT = Path(__file__).resolve().parents[2]


def _cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/external_baselines/{name}.yaml").read_text())


def _batch(cfg: dict, *, batch: int = 2, candidates: int = 5) -> dict[str, torch.Tensor]:
    bcfg = cfg["external_baselines"]
    mcfg = bcfg["model"]
    future = int(mcfg.get("future_len", 20))
    history = int(mcfg.get("history_len", 11))
    agents = int(mcfg.get("source_max_agents", 9))
    maps = int(mcfg.get("source_map_polygons", 64))
    pts = int(mcfg.get("source_map_points", 20))
    feature_dim = 32
    prefix = torch.randn(batch, candidates, future, 2)
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    source_hist = torch.randn(batch, agents, history, 9)
    source_valid = torch.ones(batch, agents, history, dtype=torch.bool)
    source_current = source_hist[:, 0, -1, :6].clone()
    source_map = torch.randn(batch, maps, pts, 6)
    source_map_valid = torch.ones(batch, maps, pts, dtype=torch.bool)
    source_map_meta = torch.zeros(batch, maps, 4)
    source_map_center = source_map[:, :, 0, :3].clone()
    source_map_poly_valid = torch.ones(batch, maps, dtype=torch.bool)
    centerline = torch.randn(batch, 64, 3)
    target = torch.zeros(batch, dtype=torch.long)
    return {
        "x": torch.randn(batch, candidates, feature_dim),
        "mask": mask,
        "target_index": target,
        "utility": torch.zeros(batch, candidates),
        "hard": torch.zeros(batch, candidates),
        "harm": torch.zeros(batch, candidates),
        "r_orc": torch.zeros(batch, candidates),
        "r_dep": torch.zeros(batch, candidates),
        "prefix_traj": prefix,
        "prefix_valid": torch.ones(batch, candidates, future, dtype=torch.bool),
        "source_agent_history": source_hist,
        "source_agent_valid": source_valid,
        "source_current_state": source_current,
        "source_map_points": source_map,
        "source_map_point_valid": source_map_valid,
        "source_map_meta": source_map_meta,
        "source_map_center": source_map_center,
        "source_map_valid": source_map_poly_valid,
        "source_centerline": centerline,
    }


def test_source_ports_forward_native_loss_backward() -> None:
    for name in ("gameformer_lite", "plantf", "pluto"):
        cfg = _cfg(name)
        batch = _batch(cfg)
        model = build_model_from_cfg(batch["x"].shape[-1], cfg)
        out = _forward_model(model, batch, cfg)
        assert out["logits"].shape == batch["mask"].shape, name
        assert torch.isfinite(out["logits"]).all(), name
        losses = _loss_dict(out, batch, cfg)
        assert torch.isfinite(losses["loss"]), name
        losses["loss"].backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), name


def test_source_port_configs_are_observation_only_and_versioned() -> None:
    for name in ("gameformer_lite", "plantf", "pluto"):
        cfg = _cfg(name)["external_baselines"]
        assert cfg["model"]["implementation"] == "source_port_v54"
        assert cfg["model"].get("use_teacher_branch_context", False) is False
        assert cfg.get("allow_teacher_supervision", False) is False
        assert cfg.get("supervision_target") == "logged_nominal"


def test_source_scene_agent_cap_keeps_nearest_observed_neighbors() -> None:
    import numpy as np
    from ocrap.external_baselines.data import _source_scene_arrays

    cfg = {
        "external_baselines": {
            "model": {
                "history_len": 3,
                "source_max_agents": 3,
                "source_map_polygons": 2,
                "source_map_points": 2,
                "source_centerline_points": 4,
            }
        }
    }
    # Storage order deliberately puts the farthest neighbor first.  PlanTF's
    # source feature builder truncates after distance sorting, so the bridge
    # should retain agents at x=2 and x=5 rather than x=50.
    hist = np.zeros((3, 5, 16), dtype=np.float32)
    valid = np.ones((3, 5), dtype=bool)
    hist[:, 0, 0] = 0.0
    hist[:, 1, 0] = 50.0
    hist[:, 2, 0] = 5.0
    hist[:, 3, 0] = 2.0
    hist[:, 4, 0] = 20.0
    hist[:, :, 10] = 4.5
    hist[:, :, 11] = 1.9
    d = {
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": np.zeros((9,), dtype=np.float32),
        "map_polylines": np.zeros((0, 0, 0), dtype=np.float32),
        "route": np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
    }
    agent, agent_valid, *_ = _source_scene_arrays(d, cfg)
    assert agent_valid[:, -1].tolist() == [True, True, True]
    assert np.allclose(agent[:, -1, 0], np.asarray([0.0, 2.0, 5.0], dtype=np.float32))
