from __future__ import annotations

import functools
import hashlib
import os
import re
from typing import Any, Iterator
from types import SimpleNamespace

import numpy as np

from ocrap.data.schema import RawScenario


def _configure_tensorflow_for_waymax() -> None:
    """Keep TensorFlow on CPU without hiding the GPU from JAX/PyTorch.

    Waymax uses TensorFlow for TFRecord input preprocessing while its simulator is
    JAX-based.  In a shared JAX+Torch CUDA environment the plain ``tensorflow``
    wheel can discover the NVIDIA libraries installed by those frameworks.
    Setting ``CUDA_VISIBLE_DEVICES=""`` would also hide the GPU from JAX, so we
    instead use TensorFlow's own visibility API *before* Waymax imports its
    dataloader.

    Set OCRAP_TENSORFLOW_CPU_ONLY=0 only if GPU TensorFlow is intentionally
    required by a different workflow.
    """
    flag = os.environ.get("OCRAP_TENSORFLOW_CPU_ONLY", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        # TensorFlow visibility cannot be changed after TF has initialized its
        # runtime.  Fail closed rather than silently competing with JAX for GPU
        # memory or binding a possibly incompatible CUDA stack.
        visible = tf.config.get_visible_devices("GPU")
        if visible:
            raise RuntimeError(
                "TensorFlow initialized a GPU before Waymax setup. Import OC-RAP/Waymax "
                "before running TensorFlow GPU ops, or set OCRAP_TENSORFLOW_CPU_ONLY=0 "
                "only if GPU TensorFlow is intentional."
            ) from exc


def _require_waymax():
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        _configure_tensorflow_for_waymax()
        from waymax import config as wx_config  # type: ignore
        from waymax import dataloader as wx_dataloader  # type: ignore
        from waymax.dataloader import womd_factories  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "simulation_backend=waymax_closed_loop requires waymax, jax, jaxlib, "
            "tensorflow and WOMD TFExample access. Install the project with the "
            "waymax extra and verify that `python -c 'import waymax, jax'` works."
        ) from e
    return jax, jnp, wx_config, wx_dataloader, womd_factories


def _apply_jax_env(cfg: dict) -> None:
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    platforms = str(wx.get("jax_platforms", "cuda,cpu"))
    os.environ.setdefault("JAX_PLATFORMS", platforms)
    if not bool(wx.get("preallocate_gpu_memory", False)):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _as_np(x: Any) -> np.ndarray:
    try:
        import jax  # type: ignore

        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def _normalize_agent_time(x: Any, num_agents: int, num_steps: int | None = None, *, name: str = "field") -> np.ndarray:
    """Return an array in Waymax's agent-time layout ``(A, T)``.

    Waymax trajectory fields are mostly stored as ``(num_objects,
    num_timesteps)``, but some metadata-like fields can be ``(num_objects,)``
    or have a singleton batch axis depending on the dataloader/JAX path.  The
    OC-RAP raw schema expects time-major arrays later, so normalize once here
    instead of relying on ad-hoc broadcasting.
    """
    arr = np.asarray(x)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 0:
        if num_steps is None:
            raise ValueError(f"Cannot infer time dimension for scalar {name}")
        return np.full((num_agents, num_steps), arr, dtype=arr.dtype)

    if arr.ndim == 1:
        if arr.size == num_agents:
            if num_steps is None:
                raise ValueError(f"Cannot infer time dimension for per-agent {name} with shape {arr.shape}")
            return np.broadcast_to(arr[:, None], (num_agents, num_steps))
        if num_steps is not None and arr.size == num_steps:
            return np.broadcast_to(arr[None, :], (num_agents, num_steps))
        if arr.size == 1 and num_steps is not None:
            return np.full((num_agents, num_steps), arr.reshape(()), dtype=arr.dtype)
        raise ValueError(f"Cannot normalize {name} with shape {arr.shape}; expected agent dimension {num_agents}")

    if arr.ndim == 2:
        if arr.shape[0] == num_agents and (num_steps is None or arr.shape[1] == num_steps):
            return arr
        if arr.shape[1] == num_agents and (num_steps is None or arr.shape[0] == num_steps):
            return arr.T
        if num_steps is not None:
            if arr.shape == (num_agents, 1):
                return np.broadcast_to(arr, (num_agents, num_steps))
            if arr.shape == (1, num_agents):
                return np.broadcast_to(arr.reshape(num_agents, 1), (num_agents, num_steps))
            if arr.shape == (1, num_steps):
                return np.broadcast_to(arr, (num_agents, num_steps))
            if arr.shape == (num_steps, 1):
                return np.broadcast_to(arr.T, (num_agents, num_steps))
        raise ValueError(
            f"Cannot normalize {name} with shape {arr.shape}; expected "
            f"({num_agents}, T) or (T, {num_agents})"
        )

    squeezed = np.squeeze(arr)
    if squeezed.shape != arr.shape:
        return _normalize_agent_time(squeezed, num_agents, num_steps, name=name)
    if num_steps is not None and arr.shape[-2:] == (num_agents, num_steps):
        return arr.reshape(-1, num_agents, num_steps)[0]
    if num_steps is not None and arr.shape[-2:] == (num_steps, num_agents):
        return arr.reshape(-1, num_steps, num_agents)[0].T
    raise ValueError(f"Cannot normalize {name} with shape {arr.shape}")


def _agent_time_array(x: Any, num_agents: int, num_steps: int, name: str) -> np.ndarray:
    """Compatibility wrapper for Waymax per-agent/per-time fields.

    Older code paths used this helper when writing RawScenario feature
    channels.  Keep it as a thin alias so metadata-like fields such as
    length, width, height and object_type are consistently returned as
    agent-time arrays with shape (A, T).
    """
    return _normalize_agent_time(x, num_agents, num_steps, name=name)


def _decode_scenario_id(value: Any) -> str | None:
    """Decode WOMD ``scenario/id`` from scalar bytes or a uint8 byte vector."""
    try:
        arr = _as_np(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    try:
        if arr.dtype == np.uint8:
            text = bytes(arr.reshape(-1).tolist()).decode("utf-8", errors="strict")
        elif arr.size == 1:
            val = arr.reshape(()).item()
            if isinstance(val, bytes):
                text = val.decode("utf-8", errors="strict")
            else:
                text = str(val)
        else:
            return None
    except Exception:
        return None
    text = text.strip()
    return text if text and text not in {"None", "b''"} else None


def _legacy_scenario_id_from_state(state: Any) -> str:
    """Reproduce the pre-v48.28 fallback id for legacy target migration."""
    ids = _as_np(state.object_metadata.ids).reshape(-1)
    ts = _as_np(state.log_trajectory.timestamp_micros).reshape(-1)
    h = hashlib.sha1(ids.tobytes() + ts[: min(16, ts.size)].tobytes()).hexdigest()[:16]
    return f"waymax_{h}"


def _scenario_identity_from_payload(
    payload: dict[str, Any], idx: int, state: Any, cfg: dict | None = None
) -> tuple[str, str, str]:
    """Return saved id, official/legacy base id, and legacy compatibility id.

    Waymax's default dataloader discards the string ``scenario/id``.  v48.28
    preserves it explicitly.  The legacy state hash remains available only as a
    migration key for datasets built before the official id was retained.
    """
    cfg = cfg or {}
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    append_index = bool(wx.get("append_scenario_index_to_id", True))
    official = _decode_scenario_id(payload.get("scenario_id"))
    legacy = _legacy_scenario_id_from_state(state)
    base = official or legacy
    base = base.replace("/", "_").replace("\\", "_")
    saved = f"{base}__wx{idx:08d}" if append_index else base
    return saved, base, legacy


def _scenario_id_from_payload(payload: dict[str, Any], idx: int, state: Any, cfg: dict | None = None) -> str:
    """Backward-compatible wrapper returning the persisted scene id."""
    return _scenario_identity_from_payload(payload, idx, state, cfg)[0]


def _infer_womd_source_role(patterns: Any) -> str:
    text = _paths_to_waymax_path(patterns).lower()
    if "validation_interactive" in text:
        return "validation_interactive"
    if re.search(r"(^|[/_])validation([/_]|$)", text):
        return "validation"
    if "training" in text:
        return "training"
    if "testing" in text or "/test" in text:
        return "test"
    return "unknown"


def _preprocess_serialized_womd_with_id(serialized: Any, dataset_cfg: Any, wx_dataloader: Any):
    """Waymax preprocessor that retains the official WOMD ``scenario/id``.

    This follows the official Waymax custom-loader pattern.  It intentionally
    fails closed when the installed Waymax/TensorFlow API cannot expose the
    feature instead of silently reverting to a loader-order-dependent hash.
    """
    import tensorflow as tf  # type: ignore

    womd_utils = getattr(wx_dataloader, "womd_utils", None)
    if womd_utils is None:
        from waymax.dataloader import womd_utils  # type: ignore
    features = womd_utils.get_features_description(
        include_sdc_paths=bool(getattr(dataset_cfg, "include_sdc_paths", False)),
        max_num_rg_points=int(getattr(dataset_cfg, "max_num_rg_points", 30000)),
        num_paths=int(getattr(dataset_cfg, "num_paths", 45)),
        num_points_per_path=int(getattr(dataset_cfg, "num_points_per_path", 800)),
    )
    features["scenario/id"] = tf.io.FixedLenFeature([1], tf.string)
    parsed = tf.io.parse_example(serialized, features)
    scenario_id = parsed.pop("scenario/id")
    parsed["scenario/id"] = tf.io.decode_raw(scenario_id, tf.uint8)
    processed = wx_dataloader.preprocess_womd_example(
        parsed,
        aggregate_timesteps=bool(getattr(dataset_cfg, "aggregate_timesteps", True)),
        max_num_objects=int(getattr(dataset_cfg, "max_num_objects", 64)),
    )
    # preprocess_womd_example may preserve unknown keys in some versions and
    # discard them in others.  Re-attach the decoded bytes explicitly.
    processed["scenario/id"] = tf.io.decode_raw(scenario_id, tf.uint8)
    return processed


def _paths_to_waymax_path(patterns: Any) -> str:
    if isinstance(patterns, str):
        return patterns
    if isinstance(patterns, (list, tuple)) and len(patterns) == 1:
        return str(patterns[0])
    if isinstance(patterns, (list, tuple)):
        # Waymax DatasetConfig accepts a string path/pattern.  Keep the common
        # multi-shard glob form by joining only when the caller provided multiple
        # concrete values; TensorFlow's gfile glob can still handle brace/glob
        # syntax in each element on recent TF versions.
        return ",".join(str(x) for x in patterns)
    return str(patterns)


def _make_dataset_config(patterns: Any, cfg: dict):
    _, _, wx_config, _, _ = _require_waymax()
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    path = _paths_to_waymax_path(patterns)
    return wx_config.DatasetConfig(
        path=path,
        data_format=wx_config.DataFormat.TFRECORD,
        repeat=1,
        batch_dims=(),
        shuffle_seed=None,
        deterministic=True,
        include_sdc_paths=bool(wx.get("dataloader_include_sdc_paths", True)),
        aggregate_timesteps=True,
        max_num_rg_points=int(wx.get("max_num_rg_points", 30000)),
        max_num_objects=int(cfg.get("max_agents", 64)),
        num_paths=int(wx.get("num_paths", 45)),
        num_points_per_path=int(wx.get("num_points_per_path", 800)),
        drop_remainder=False,
        batch_by_scenario=True,
    )


def _route_from_sdc_paths(state: Any, max_points: int) -> np.ndarray:
    route = np.zeros((max_points, 6), dtype=np.float32)
    paths = getattr(state, "sdc_paths", None)
    if paths is not None:
        x = _as_np(paths.x)
        y = _as_np(paths.y)
        valid = _as_np(paths.valid).astype(bool)
        on_route = _as_np(paths.on_route).astype(bool)
        if x.ndim >= 2:
            candidates = np.where(on_route.reshape(-1))[0]
            if candidates.size == 0:
                candidates = np.arange(x.shape[-2])
            best = int(candidates[0])
            best_count = -1
            for c in candidates[: min(8, len(candidates))]:
                cnt = int(valid[c].sum())
                if cnt > best_count:
                    best = int(c)
                    best_count = cnt
            pts = np.stack([x[best], y[best]], axis=-1)[valid[best]]
            if len(pts) >= 2:
                idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
                pts = pts[idx]
                route[:, :2] = pts[:, :2]
                d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
                route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
                route[:, 3] = 13.4
                route[:, 5] = 1.0
                return route
    # Fallback to logged SDC path.  This is only a route proxy; diagnose will
    # still expose whether sdc_paths were available for true route metrics.
    meta = state.object_metadata
    sdc_idx = int(np.argmax(_as_np(meta.is_sdc).astype(bool)))
    tr = state.log_trajectory
    valid = _as_np(tr.valid)[sdc_idx].astype(bool)
    xy = np.stack([_as_np(tr.x)[sdc_idx], _as_np(tr.y)[sdc_idx]], axis=-1)[valid]
    if len(xy) < 2:
        xy = np.stack([np.arange(max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    idx = np.linspace(0, len(xy) - 1, max_points).round().astype(int)
    route[:, :2] = xy[idx, :2]
    d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
    route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    return route


def _map_from_waymax_roadgraph(state: Any, max_polylines: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    F = 10
    arr = np.zeros((max_polylines, max_points, F), dtype=np.float32)
    valid_out = np.zeros((max_polylines, max_points), dtype=bool)
    rg = getattr(state, "roadgraph_points", None)
    if rg is None:
        return arr, valid_out
    x = _as_np(rg.x).reshape(-1)
    y = _as_np(rg.y).reshape(-1)
    z = _as_np(rg.z).reshape(-1)
    dx = _as_np(rg.dir_x).reshape(-1)
    dy = _as_np(rg.dir_y).reshape(-1)
    typ = _as_np(rg.types).reshape(-1)
    val = _as_np(rg.valid).reshape(-1).astype(bool)
    ids = _as_np(rg.ids).reshape(-1)
    keep = np.where(val)[0]
    if keep.size == 0:
        return arr, valid_out
    # Preserve roadgraph feature identity where possible; fall back to chunks.
    groups: list[np.ndarray] = []
    for gid in np.unique(ids[keep])[:max_polylines]:
        idx = keep[ids[keep] == gid]
        if idx.size:
            groups.append(idx[:max_points])
        if len(groups) >= max_polylines:
            break
    if not groups:
        groups = [keep[i : i + max_points] for i in range(0, min(keep.size, max_polylines * max_points), max_points)]
    for p, idx in enumerate(groups[:max_polylines]):
        n = min(max_points, len(idx))
        ii = idx[:n]
        arr[p, :n, 0] = x[ii]
        arr[p, :n, 1] = y[ii]
        arr[p, :n, 2] = z[ii]
        arr[p, :n, 3] = dx[ii]
        arr[p, :n, 4] = dy[ii]
        arr[p, :n, 5] = typ[ii]
        arr[p, :n, 9] = 1.0
        valid_out[p, :n] = True
    return arr, valid_out


def _trajectory_for_raw_export(state: Any, mode: str = "log", splice_until: int | None = None) -> Any:
    """Choose/export trajectory used to construct RawScenario.

    `log` preserves dataset construction behavior.  `sim` exports the current
    simulated trajectory.  `closed_loop_splice` uses simulated history up to the
    current replanning timestep and log trajectory for the remaining future so
    candidate generation starts from the closed-loop state without losing future
    context.
    """
    if mode == "log" or not hasattr(state, "sim_trajectory"):
        return state.log_trajectory
    if mode == "sim":
        return state.sim_trajectory
    if mode != "closed_loop_splice":
        raise ValueError(f"Unknown trajectory export mode {mode!r}")
    log = state.log_trajectory
    sim = state.sim_trajectory
    fields = ["x", "y", "z", "vel_x", "vel_y", "yaw", "valid", "length", "width", "height", "timestamp_micros"]
    out = {}
    cut = int(splice_until if splice_until is not None else _as_np(getattr(state, "timestep", 0)).reshape(-1)[0])
    for name in fields:
        if not hasattr(log, name):
            continue
        lv = np.array(_as_np(getattr(log, name)))
        if not hasattr(sim, name):
            out[name] = lv
            continue
        sv = np.array(_as_np(getattr(sim, name)))
        if lv.shape != sv.shape or lv.ndim == 0:
            out[name] = lv
            continue
        arr = lv.copy()
        if arr.ndim >= 2:
            t_axis = -1 if arr.shape[-1] >= cut + 1 else 0
            sl = [slice(None)] * arr.ndim
            sl[t_axis] = slice(0, max(0, min(cut + 1, arr.shape[t_axis])))
            arr[tuple(sl)] = sv[tuple(sl)]
        else:
            arr[: max(0, min(cut + 1, arr.shape[0]))] = sv[: max(0, min(cut + 1, arr.shape[0]))]
        out[name] = arr
    return SimpleNamespace(**out)


def raw_scenario_from_waymax_state(state: Any, scenario_id: str, scenario_index: int, cfg: dict, trajectory_mode: str = "log", splice_until: int | None = None, static_template: RawScenario | None = None) -> RawScenario:
    tr = _trajectory_for_raw_export(state, trajectory_mode, splice_until)
    meta = state.object_metadata
    meta_ids = _as_np(meta.ids).reshape(-1)
    A = int(meta_ids.size) if meta_ids.size else int(getattr(state, "num_objects", 0))
    x = _normalize_agent_time(_as_np(tr.x), A, name="log_trajectory.x")
    T = int(x.shape[1])
    y = _normalize_agent_time(_as_np(tr.y), A, T, name="log_trajectory.y")
    z = _normalize_agent_time(_as_np(tr.z), A, T, name="log_trajectory.z")
    vx = _normalize_agent_time(_as_np(tr.vel_x), A, T, name="log_trajectory.vel_x")
    vy = _normalize_agent_time(_as_np(tr.vel_y), A, T, name="log_trajectory.vel_y")
    yaw = _normalize_agent_time(_as_np(tr.yaw), A, T, name="log_trajectory.yaw")
    valid = _normalize_agent_time(_as_np(tr.valid), A, T, name="log_trajectory.valid").astype(bool)
    length = _normalize_agent_time(_as_np(tr.length), A, T, name="log_trajectory.length")
    width = _normalize_agent_time(_as_np(tr.width), A, T, name="log_trajectory.width")
    height = _normalize_agent_time(_as_np(tr.height), A, T, name="log_trajectory.height")
    obj_type = _normalize_agent_time(_as_np(meta.object_types), A, T, name="object_metadata.object_types")
    states = np.zeros((T, A, 16), dtype=np.float32)
    states[..., 0] = x.T
    states[..., 1] = y.T
    states[..., 2] = z.T
    states[..., 3] = vx.T
    states[..., 4] = vy.T
    ax = np.gradient(vx, 0.1, axis=-1) if T > 1 else np.zeros_like(vx)
    ay = np.gradient(vy, 0.1, axis=-1) if T > 1 else np.zeros_like(vy)
    states[..., 5] = ax.T
    states[..., 6] = ay.T
    states[..., 7] = yaw.T
    states[..., 8] = np.sin(yaw).T
    states[..., 9] = np.cos(yaw).T
    states[..., 10] = _agent_time_array(length, A, T, "length").T
    states[..., 11] = _agent_time_array(width, A, T, "width").T
    states[..., 12] = _agent_time_array(height, A, T, "height").T
    states[..., 13] = _agent_time_array(obj_type, A, T, "object_type").T
    states[..., 14] = valid.T.astype(np.float32)
    states[..., 15] = valid.T.astype(np.float32)
    timestamps = _as_np(tr.timestamp_micros)
    if timestamps.ndim >= 2 and timestamps.shape[-1] == T:
        timestamps = timestamps.reshape(-1, T)[0]
    elif timestamps.ndim >= 2 and timestamps.shape[0] > 0:
        timestamps = timestamps[0]
    timestamps_s = timestamps.astype(np.float64) * 1e-6 if timestamps.size else np.arange(T, dtype=np.float32) * 0.1
    # Roadgraph, route, object ids and dynamic-map tensors are scenario-static.
    # Closed-loop replanning used to rebuild/copy them from JAX on every step,
    # which is especially expensive for ~30k roadgraph points.  Reuse the raw
    # scenario produced by the loader when supplied; only trajectories change.
    if static_template is not None:
        maps = static_template.map_polylines
        map_valid = static_template.map_valid
        route = static_template.route
        dyn = static_template.dynamic_map
        sdc_idx = int(static_template.sdc_track_index)
        object_ids = static_template.object_ids
    else:
        maps, map_valid = _map_from_waymax_roadgraph(state, int(cfg.get("max_map_polylines", 256)), int(cfg.get("max_polyline_points", 64)))
        route = _route_from_sdc_paths(state, int(cfg.get("route_points", 80)))
        dyn = np.zeros((T, int(cfg.get("max_dynamic_signals", 16)), 8), dtype=np.float32)
        sdc_idx = int(np.argmax(_as_np(meta.is_sdc).astype(bool)))
        object_ids = [str(int(v)) for v in meta_ids]
    return RawScenario(
        scenario_id=scenario_id,
        timestamps=timestamps_s[:T].astype(np.float32),
        sdc_track_index=sdc_idx,
        agent_states=states,
        agent_valid=valid.T,
        map_polylines=maps,
        map_valid=map_valid,
        route=route,
        dynamic_map=dyn,
        object_ids=object_ids,
        metadata={
            "source": "womd_waymax",
            "original_scenario_id": scenario_id,
            "_waymax_state": state,
            "_waymax_scenario_index": int(scenario_index),
            "waymax_sdc_paths_available": getattr(state, "sdc_paths", None) is not None,
            "_waymax_trajectory_mode": trajectory_mode,
            "_waymax_splice_until": -1 if splice_until is None else int(splice_until),
            "_waymax_branch_from_current": trajectory_mode in {"sim", "closed_loop_splice"},
        },
    )


def iter_waymax_womd_scenarios(patterns: Any, max_scenarios: int | None, parser_cfg: dict | None = None) -> Iterator[RawScenario]:
    cfg = parser_cfg or {}
    _apply_jax_env(cfg)
    _, _, _, wx_dataloader, womd_factories = _require_waymax()
    dataset_cfg = _make_dataset_config(patterns, cfg)

    def _postprocess(example):
        state = womd_factories.simulator_state_from_womd_dict(
            example,
            include_sdc_paths=bool((cfg.get("waymax", {}) or {}).get("dataloader_include_sdc_paths", True)),
        )
        return {"state": state, "scenario_id": example.get("scenario/id")}

    wx_cfg = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    retain_official_id = bool(wx_cfg.get("retain_official_scenario_id", True))
    if retain_official_id:
        parse = functools.partial(
            _preprocess_serialized_womd_with_id, dataset_cfg=dataset_cfg, wx_dataloader=wx_dataloader
        )
    else:
        parse = functools.partial(wx_dataloader.preprocess_serialized_womd_data, config=dataset_cfg)
    gen = wx_dataloader.get_data_generator(dataset_cfg, parse, _postprocess)
    start_index = max(0, int(cfg.get("scenario_start_index", 0)))
    stride = max(1, int(cfg.get("scenario_stride", 1)))
    worker_index = int(cfg.get("scenario_worker_index", 0)) % stride
    emitted = 0
    for i, payload in enumerate(gen):
        if i < start_index:
            continue
        if ((i - start_index) % stride) != worker_index:
            continue
        if max_scenarios is not None and emitted >= int(max_scenarios):
            break
        state = payload["state"] if isinstance(payload, dict) else payload
        saved_id, base_id, legacy_id = _scenario_identity_from_payload(
            payload if isinstance(payload, dict) else {}, i, state, cfg
        )
        raw = raw_scenario_from_waymax_state(state, saved_id, i, cfg)
        raw.metadata.update({
            "original_scenario_id": base_id,
            "official_scenario_id": base_id if not base_id.startswith("waymax_") else None,
            "legacy_scenario_id": legacy_id,
            "scenario_id_source": "official_womd" if not base_id.startswith("waymax_") else "legacy_state_hash",
            "womd_source_role": _infer_womd_source_role(patterns),
            "womd_source_pattern": _paths_to_waymax_path(patterns),
            "waymax_max_num_objects": int(getattr(dataset_cfg, "max_num_objects", cfg.get("max_agents", 64))),
        })
        yield raw
        emitted += 1


def iter_waymax_womd_scenarios_selected(
    patterns: Any,
    scenario_indices: Any,
    parser_cfg: dict | None = None,
) -> Iterator[RawScenario]:
    """Iterate only explicitly requested global Waymax scenario indices.

    This is an execution-equivalent targeted replay path for offline audits.
    The ordinary :func:`iter_waymax_womd_scenarios` materializes a full
    ``SimulatorState`` *and* converts it into ``RawScenario`` for every global
    index before callers can discard irrelevant rows.  V48.91 replays a sparse
    set of historical calibration indices, so doing that work for every scenario
    up to the maximum requested index is pure overhead.

    Here we keep the exact same TFExample parser and global enumeration order,
    but delay ``simulator_state_from_womd_dict`` and ``raw_scenario_from_waymax_state``
    until an index is actually requested.  Requested scenarios therefore use the
    same conversion path and metadata as the production iterator; unrequested
    scenarios are parsed only far enough to preserve deterministic enumeration.
    """
    targets = sorted({int(x) for x in scenario_indices if int(x) >= 0})
    if not targets:
        return
    target_set = set(targets)
    max_target = targets[-1]
    cfg = parser_cfg or {}
    _apply_jax_env(cfg)
    _, _, _, wx_dataloader, womd_factories = _require_waymax()
    dataset_cfg = _make_dataset_config(patterns, cfg)

    # Keep preprocessing identical to the production iterator, but postpone the
    # expensive WOMD-dict -> SimulatorState conversion until after index filtering.
    def _identity_postprocess(example):
        return example

    wx_cfg = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    retain_official_id = bool(wx_cfg.get("retain_official_scenario_id", True))
    if retain_official_id:
        parse = functools.partial(
            _preprocess_serialized_womd_with_id, dataset_cfg=dataset_cfg, wx_dataloader=wx_dataloader
        )
    else:
        parse = functools.partial(wx_dataloader.preprocess_serialized_womd_data, config=dataset_cfg)
    gen = wx_dataloader.get_data_generator(dataset_cfg, parse, _identity_postprocess)

    seen: set[int] = set()
    progress_every = max(0, int(cfg.get('_selected_replay_progress_every', 0)))
    for i, example in enumerate(gen):
        if i > max_target:
            break
        if progress_every and i > 0 and (i % progress_every) == 0:
            print(
                f"[ocrap-profile] sparse Waymax source scan index={i}/{max_target} "
                f"targets_materialized={len(seen)}/{len(target_set)}",
                flush=True,
            )
        if i not in target_set:
            continue
        state = womd_factories.simulator_state_from_womd_dict(
            example,
            include_sdc_paths=bool(wx_cfg.get("dataloader_include_sdc_paths", True)),
        )
        payload = {"state": state, "scenario_id": example.get("scenario/id")}
        saved_id, base_id, legacy_id = _scenario_identity_from_payload(payload, i, state, cfg)
        raw = raw_scenario_from_waymax_state(state, saved_id, i, cfg)
        raw.metadata.update({
            "original_scenario_id": base_id,
            "official_scenario_id": base_id if not base_id.startswith("waymax_") else None,
            "legacy_scenario_id": legacy_id,
            "scenario_id_source": "official_womd" if not base_id.startswith("waymax_") else "legacy_state_hash",
            "womd_source_role": _infer_womd_source_role(patterns),
            "womd_source_pattern": _paths_to_waymax_path(patterns),
            "waymax_max_num_objects": int(getattr(dataset_cfg, "max_num_objects", cfg.get("max_agents", 64))),
        })
        yield raw
        seen.add(i)
        if len(seen) == len(target_set):
            break
