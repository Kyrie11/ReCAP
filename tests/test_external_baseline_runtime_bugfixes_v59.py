from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ocrap.external_baselines.source_ports import SourceAgentEncoder, SourceMapEncoder, SourcePointsEncoder

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_source_ports_bfloat16_autocast_scatter_is_dtype_safe() -> None:
    # CPU BF16 autocast reproduces the exact index_put dtype contract that used
    # to fail on CUDA: FP32 bridge tensors, BF16 Linear outputs.
    points = SourcePointsEncoder(6, 32).train()
    x = torch.randn(4, 5, 6, dtype=torch.float32)
    mask = torch.ones(4, 5, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = points(x, mask)
    assert y.shape == (4, 32)
    assert torch.isfinite(y.float()).all()

    agent = SourceAgentEncoder(32, state_channel=6, state_dropout=0.0, dropout=0.0).train()
    history = torch.randn(2, 3, 5, 9, dtype=torch.float32)
    valid = torch.ones(2, 3, 5, dtype=torch.bool)
    current = torch.randn(2, 9, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ay = agent(history, valid, current)
    assert ay.shape == (2, 3, 32)
    assert torch.isfinite(ay.float()).all()

    maps = SourceMapEncoder(32).train()
    mp = torch.randn(2, 4, 5, 6, dtype=torch.float32)
    mpv = torch.ones(2, 4, 5, dtype=torch.bool)
    meta = torch.zeros(2, 4, 4, dtype=torch.float32)
    meta[..., 3] = 10.0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        my = maps(mp, mpv, meta)
    assert my.shape == (2, 4, 32)
    assert torch.isfinite(my.float()).all()


def test_cpsf_calibration_uses_waymax_tfexample_selected_replay(monkeypatch, tmp_path: Path) -> None:
    mod = _load_tool("calibrate_external_baselines_runtime_fix", "tools/calibrate_external_baselines.py")
    sample = {
        "scene_id": np.asarray("official_scene__wx00000007"),
        "official_scenario_id": np.asarray("official_scene"),
        "source_scenario_index": np.asarray(7),
        "time_index": np.asarray(10),
    }
    row = {
        "group_index": 0,
        "scene_id": "official_scene__wx00000007",
        "scene_key": "official_scene",
        "match_ids": {"official_scene"},
        "source_scenario_index": 7,
        "time_index": 10,
        "sample": sample,
    }
    monkeypatch.setattr(mod, "load_config", lambda _: {"external_baselines": {"policy": {}}})
    monkeypatch.setattr(mod, "_target_groups", lambda *_: ([row], {"official_scene": [0]}))
    monkeypatch.setattr(
        mod,
        "resolve_womd_spec",
        lambda _: SimpleNamespace(valid=True, files=("fake_tfexample.tfrecord",), as_dict=lambda: {}),
    )
    calls = {"selected": 0, "legacy": 0}
    fake_raw = SimpleNamespace(
        scenario_id="official_scene__wx00000007",
        metadata={"_waymax_scenario_index": 7, "official_scenario_id": "official_scene"},
    )

    def selected(patterns, indices, parser_cfg=None):
        calls["selected"] += 1
        assert list(indices) == [7]
        yield fake_raw

    def legacy(*args, **kwargs):
        calls["legacy"] += 1
        raise AssertionError("Scenario/full-scan fallback must not be used when source indices exist")

    monkeypatch.setattr(mod, "iter_waymax_womd_scenarios_selected", selected)
    monkeypatch.setattr(mod, "iter_waymax_womd_scenarios", legacy)
    monkeypatch.setattr(
        mod,
        "_group_nonconformity",
        lambda raw, row, cfg, horizon: (np.full(horizon, 0.25), np.ones(horizon, dtype=int), 0.0),
    )
    out = tmp_path / "calib.json"
    argv = [
        "calibrate_external_baselines.py",
        "--config", "fake.yaml",
        "--dataset", str(tmp_path / "calibration_near_contact"),
        "--split", "calibration",
        "--womd-pattern", "fake@1",
        "--delta", "0.9",
        "--prediction-horizon", "2",
        "--mission-horizon", "1",
        "--allow-infinite",
        "--output", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    data = json.loads(out.read_text())
    assert calls == {"selected": 1, "legacy": 0}
    assert data["num_matched_groups"] == 1
    assert data["num_matched_scenes"] == 1


def test_cpsf_recovers_legacy_zero_source_index_from_scene_suffix() -> None:
    mod = _load_tool("calibrate_external_baselines_source_index", "tools/calibrate_external_baselines.py")
    assert mod._source_index_from_scene_id("abc__wx00000000") == 0
    assert mod._canonical_scene_id("abc__wx00000000") == "abc"


def test_jax_preflight_detects_plugin_jaxlib_version_mismatch() -> None:
    mod = _load_tool("check_jax_waymax_runtime", "tools/check_jax_waymax_runtime.py")
    mismatches = mod.obvious_stack_mismatches({
        "jaxlib": "0.4.33",
        "jax-cuda12-plugin": "0.4.35",
        "jax-cuda12-pjrt": "0.4.35",
    })
    assert any("jax-cuda12-plugin" in x for x in mismatches)
    assert any("jax-cuda12-pjrt" in x for x in mismatches)


def test_launchers_force_cpu_backend_for_cpu_only_waymax_tools() -> None:
    near = (ROOT / "scripts/run_near_contact_external_baselines_2gpu_optimized.sh").read_text()
    contact = (ROOT / "scripts/run_contact_external_baselines.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=\"\" JAX_PLATFORMS=cpu" in near
    assert "CUDA_VISIBLE_DEVICES=\"\" JAX_PLATFORMS=cpu" in contact
    for rel in (
        "scripts/run_safe_regime_external_baselines.sh",
        "scripts/run_near_contact_external_baselines_2gpu_optimized.sh",
        "scripts/run_contact_external_baselines.sh",
    ):
        text = (ROOT / rel).read_text()
        assert "tools/check_jax_waymax_runtime.py" in text
        assert "JAX_RUNTIME_PREFLIGHT" in text


def test_cpsf_source_index_accepts_legacy_alias_for_current_official_id(monkeypatch, tmp_path: Path) -> None:
    mod = _load_tool("calibrate_external_baselines_legacy_alias", "tools/calibrate_external_baselines.py")
    row = {
        "group_index": 0,
        "scene_id": "waymax_deadbeef__wx00000007",
        "scene_key": "waymax_deadbeef",
        "match_ids": {"waymax_deadbeef"},
        "source_scenario_index": 7,
        "time_index": 10,
        "sample": {"time_index": np.asarray(10)},
    }
    monkeypatch.setattr(mod, "load_config", lambda _: {"external_baselines": {"policy": {}}})
    monkeypatch.setattr(mod, "_target_groups", lambda *_: ([row], {"waymax_deadbeef": [0]}))
    monkeypatch.setattr(
        mod, "resolve_womd_spec",
        lambda _: SimpleNamespace(valid=True, files=("fake.tfrecord",), as_dict=lambda: {}),
    )
    raw = SimpleNamespace(
        scenario_id="0123456789abcdef__wx00000007",
        metadata={
            "_waymax_scenario_index": 7,
            "official_scenario_id": "0123456789abcdef",
            "original_scenario_id": "0123456789abcdef",
            "legacy_scenario_id": "waymax_deadbeef",
        },
    )
    monkeypatch.setattr(mod, "iter_waymax_womd_scenarios_selected", lambda *a, **k: iter([raw]))
    monkeypatch.setattr(mod, "iter_waymax_womd_scenarios", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback should not run")))
    monkeypatch.setattr(
        mod, "_group_nonconformity",
        lambda raw, row, cfg, horizon: (np.full(horizon, 0.25), np.ones(horizon, dtype=int), 0.0),
    )
    out = tmp_path / "calib.json"
    monkeypatch.setattr(sys, "argv", [
        "calibrate_external_baselines.py", "--config", "fake.yaml",
        "--dataset", str(tmp_path / "calibration_near_contact"), "--split", "calibration",
        "--womd-pattern", "fake@1", "--delta", "0.9",
        "--prediction-horizon", "2", "--mission-horizon", "1",
        "--allow-infinite", "--output", str(out),
    ])
    mod.main()
    data = json.loads(out.read_text())
    assert data["num_matched_groups"] == 1
    assert data["source_index_verified_groups"] == 1
    assert data["identity_fallback_groups"] == 0
    assert data["source_index_mismatch_count"] == 0


def test_cpsf_source_index_is_hint_and_falls_back_to_identity(monkeypatch, tmp_path: Path) -> None:
    mod = _load_tool("calibrate_external_baselines_identity_fallback", "tools/calibrate_external_baselines.py")
    row = {
        "group_index": 0,
        "scene_id": "waymax_target__wx00000007",
        "scene_key": "waymax_target",
        "match_ids": {"waymax_target"},
        "source_scenario_index": 7,
        "time_index": 10,
        "sample": {"time_index": np.asarray(10)},
    }
    monkeypatch.setattr(mod, "load_config", lambda _: {"external_baselines": {"policy": {}}})
    monkeypatch.setattr(mod, "_target_groups", lambda *_: ([row], {"waymax_target": [0]}))
    monkeypatch.setattr(
        mod, "resolve_womd_spec",
        lambda _: SimpleNamespace(valid=True, files=("fake.tfrecord",), as_dict=lambda: {}),
    )
    wrong = SimpleNamespace(
        scenario_id="official_wrong__wx00000007",
        metadata={"_waymax_scenario_index": 7, "official_scenario_id": "official_wrong", "legacy_scenario_id": "waymax_wrong"},
    )
    target = SimpleNamespace(
        scenario_id="official_target__wx00000011",
        metadata={"_waymax_scenario_index": 11, "official_scenario_id": "official_target", "legacy_scenario_id": "waymax_target"},
    )
    monkeypatch.setattr(mod, "iter_waymax_womd_scenarios_selected", lambda *a, **k: iter([wrong]))
    monkeypatch.setattr(mod, "iter_waymax_womd_scenarios", lambda *a, **k: iter([wrong, target]))
    monkeypatch.setattr(
        mod, "_group_nonconformity",
        lambda raw, row, cfg, horizon: (np.full(horizon, 0.5), np.ones(horizon, dtype=int), 0.0),
    )
    out = tmp_path / "calib.json"
    monkeypatch.setattr(sys, "argv", [
        "calibrate_external_baselines.py", "--config", "fake.yaml",
        "--dataset", str(tmp_path / "calibration_near_contact"), "--split", "calibration",
        "--womd-pattern", "fake@1", "--delta", "0.9",
        "--prediction-horizon", "2", "--mission-horizon", "1",
        "--allow-infinite", "--output", str(out),
    ])
    mod.main()
    data = json.loads(out.read_text())
    assert data["num_matched_groups"] == 1
    assert data["source_index_verified_groups"] == 0
    assert data["identity_fallback_groups"] == 1
    assert data["source_index_mismatch_count"] == 1


def test_near_launcher_defaults_closed_loop_to_standard_validation() -> None:
    text = (ROOT / "scripts/run_near_contact_external_baselines_2gpu_optimized.sh").read_text()
    assert ': "${CL_WOMD:=$WOMD_VAL}"' in text
