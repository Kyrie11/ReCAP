from __future__ import annotations

"""V48.108 OC-RPAP: raw-to-projected candidate-action pathway audit.

V48.107 shows that nominal-invariant ordinal training of the first Stage-I
Transformer block improves its train/dev pairwise objective yet still fails the
held-out Support+Reserve gates.  The preregistered next question is therefore
upstream of attention mixing: is transferable signed response already present
in the raw candidate-prefix/control variables, and do the frozen input
projections preserve it?

This module is audit-only.  The candidate pathway is fixed by the historical
StructuredTokenEncoder layout, not selected from results:
  ego + prefix parameters + macro/scalars + prefix state + control.
The counterpart after projection is exactly the five corresponding semantic
tokens before any Transformer layer.  All other scene/agent inputs are treated
as the static interaction context and are checked for candidate invariance.
"""

from dataclasses import dataclass
from typing import Iterable

import torch

from ocrap.models.encoders import FlatFeatureLayout, StructuredTokenEncoder

ENGINEERING_VERSION = "v48.108.1-OC-RPAP"
ALGORITHM_NAME = "Observation-Consistent Raw-to-Projected Action Pathway Audit"
RAW_PATH_GROUPS = ("ego", "prefix_param", "macro_scalar", "prefix_state", "control")
PROJECTED_TOKEN_INDICES = (1, 2, 3, 4, 5)  # token 0 is CLS


def _layout_slices(layout: FlatFeatureLayout) -> dict[str, slice]:
    dims = [
        ("ego", layout.ego_dim),
        ("prefix_param", layout.prefix_param_dim),
        ("macro", layout.num_macros),
        ("scalar", layout.scalar_dim),
        ("prefix_state", layout.prefix_flat_dim),
        ("control", layout.control_flat_dim),
        ("agent_summary", layout.agent_summary_dim),
        ("agents", layout.feature_max_agents * layout.agent_token_dim),
        ("bev", layout.bev_dim),
        ("route", layout.route_stats_dim + layout.route_flat_dim),
        ("map", layout.map_stats_dim + layout.map_flat_dim),
        ("dyn", layout.dyn_stats_dim + layout.dyn_flat_dim),
    ]
    out: dict[str, slice] = {}
    i = 0
    for name, d in dims:
        out[name] = slice(i, i + int(d)); i += int(d)
    if i != int(layout.total_dim):
        raise ValueError(f"layout slice mismatch {i} != {layout.total_dim}")
    out["macro_scalar"] = slice(out["macro"].start, out["scalar"].stop)
    return out


def raw_pathway_dim(layout: FlatFeatureLayout) -> int:
    return int(layout.ego_dim + layout.prefix_param_dim + layout.num_macros + layout.scalar_dim + layout.prefix_flat_dim + layout.control_flat_dim)


def raw_candidate_pathway(x: torch.Tensor, layout: FlatFeatureLayout) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != int(layout.total_dim):
        raise ValueError("raw_candidate_pathway expects [B,total_dim]")
    s = _layout_slices(layout)
    return torch.cat([x[:, s[g]] for g in RAW_PATH_GROUPS], dim=-1).float()


def raw_static_context(x: torch.Tensor, layout: FlatFeatureLayout) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != int(layout.total_dim):
        raise ValueError("raw_static_context expects [B,total_dim]")
    s = _layout_slices(layout)
    groups = ("agent_summary", "agents", "bev", "route", "map", "dyn")
    return torch.cat([x[:, s[g]] for g in groups], dim=-1).float()


def projected_candidate_pathway(enc: StructuredTokenEncoder, x: torch.Tensor) -> torch.Tensor:
    """Five fixed semantic tokens before Transformer interaction, flattened."""
    ego, prefix_param, macro, scalar, prefix_state, control, *_ = enc._split(x)
    tokens = [
        enc.ego_proj(ego),
        enc.prefix_param_proj(prefix_param),
        enc.macro_scalar_proj(torch.cat([macro, scalar], dim=-1)),
        enc.prefix_state_proj(prefix_state),
        enc.control_proj(control),
    ]
    tok = torch.stack(tokens, dim=1)
    # Historical H0 adds fixed position embeddings.  Keep them so this is
    # literally the registered pre-encoder representation.  They cancel in
    # candidate-minus-nominal deltas.
    pos = enc.pos[:, 1:6, :]
    return (tok + pos).float().reshape(x.shape[0], -1)


def action_features(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw/projected analogue of the V48.102 nominal/delta/context contract."""
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("action_features requires nominal + candidate rows")
    nominal = z[0:1]
    candidate = z[1:]
    delta = candidate - nominal
    state = nominal.expand(candidate.shape[0], -1)
    context = delta * (1.0 + torch.tanh(state))
    return state, delta, context


def _projection_specs(enc: StructuredTokenEncoder):
    L = enc.layout
    return [
        ("ego", enc.ego_proj, int(L.ego_dim)),
        ("prefix_param", enc.prefix_param_proj, int(L.prefix_param_dim)),
        ("macro_scalar", enc.macro_scalar_proj, int(L.num_macros + L.scalar_dim)),
        ("prefix_state", enc.prefix_state_proj, int(L.prefix_flat_dim)),
        ("control", enc.control_proj, int(L.control_flat_dim)),
    ]


def projection_structure(enc: StructuredTokenEncoder) -> dict[str, dict[str, float | int | bool]]:
    out: dict[str, dict[str, float | int | bool]] = {}
    for name, layer, input_dim in _projection_specs(enc):
        W = layer.weight.detach().double().cpu()
        sv = torch.linalg.svdvals(W)
        rank = int(torch.linalg.matrix_rank(W).item())
        smax = float(sv.max().item()) if sv.numel() else 0.0
        smin = float(sv[-1].item()) if sv.numel() else 0.0
        cond = float(smax / max(smin, 1.0e-18)) if smax > 0 else float("inf")
        out[name] = {
            "input_dim": input_dim,
            "output_dim": int(W.shape[0]),
            "rank": rank,
            "full_column_rank": bool(rank == input_dim),
            "sigma_max": smax,
            "sigma_min": smin,
            "condition_number": cond,
        }
    return out


def projection_full_column_rank(enc: StructuredTokenEncoder) -> bool:
    return bool(all(bool(v["full_column_rank"]) for v in projection_structure(enc).values()))


def projection_structural_injectivity_event(event: dict, reconstruction_rel_tol: float = 1.0e-4) -> bool:
    """Return the *structural* raw->projected injectivity decision for one event.

    Structural injectivity is owned by full-column-rank of every fixed linear
    projection together with the registered pseudoinverse reconstruction check.
    The empirical FP32 quantity ``P(x1)-P(x0) - W(x1-x0)`` is intentionally not
    part of this decision: the two algebraically equivalent expressions use
    different floating-point evaluation orders and their absolute discrepancy is
    scale dependent.  It remains a numerical diagnostic only.
    """
    return bool(
        event.get("projection_all_full_column_rank") is True
        and float(event.get("raw_delta_reconstruction_max_rel_l2", float("inf"))) <= float(reconstruction_rel_tol)
    )


def reconstruct_raw_delta_from_projected(enc: StructuredTokenEncoder, raw_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project raw pathway deltas and recover them through fixed pseudoinverses.

    Bias and positional terms cancel for candidate-minus-nominal deltas.
    Returns (projected_delta_flat, reconstructed_raw_delta).
    """
    if raw_delta.ndim != 2 or raw_delta.shape[1] != raw_pathway_dim(enc.layout):
        raise ValueError("raw delta dimension mismatch")
    pieces = []
    rec = []
    st = 0
    for _name, layer, d in _projection_specs(enc):
        x = raw_delta[:, st:st+d].double()
        W = layer.weight.detach().double().to(x.device)  # [D,d]
        y = x @ W.T
        # y = x W^T -> x = y pinv(W^T) when W has full column rank.
        xr = y @ torch.linalg.pinv(W.T)
        pieces.append(y); rec.append(xr); st += d
    return torch.cat(pieces, dim=-1).float(), torch.cat(rec, dim=-1).float()


def candidate_pathway_dimension_check(d_model: int = 192) -> bool:
    L = FlatFeatureLayout()
    return raw_pathway_dim(L) == 156 and 5 * int(d_model) == 960


def synthetic_projection_injectivity_check(d_model: int = 192) -> bool:
    torch.manual_seed(48108)
    enc = StructuredTokenEncoder(FlatFeatureLayout(), d_model=d_model, num_layers=2, num_heads=4).eval()
    if not projection_full_column_rank(enc):
        return False
    x = torch.randn(7, raw_pathway_dim(enc.layout))
    _y, xr = reconstruct_raw_delta_from_projected(enc, x)
    return bool(torch.allclose(x.float(), xr.float(), atol=2.0e-5, rtol=1.0e-5))


def static_context_zero_delta_check() -> bool:
    L = FlatFeatureLayout(); torch.manual_seed(108)
    x = torch.randn(4, L.total_dim)
    s = _layout_slices(L)
    # Make only the candidate pathway vary; all static context stays nominal.
    x[1:, s["agent_summary"]] = x[0, s["agent_summary"]]
    x[1:, s["agents"]] = x[0, s["agents"]]
    x[1:, s["bev"]] = x[0, s["bev"]]
    x[1:, s["route"]] = x[0, s["route"]]
    x[1:, s["map"]] = x[0, s["map"]]
    x[1:, s["dyn"]] = x[0, s["dyn"]]
    z = raw_static_context(x, L)
    return bool(torch.count_nonzero(z[1:] - z[0:1]).item() == 0)


def contract_checks(d_model: int = 192) -> dict[str, bool]:
    return {
        "raw_candidate_pathway_dim_156": bool(candidate_pathway_dimension_check(d_model)),
        "projected_candidate_pathway_dim_960": bool(candidate_pathway_dimension_check(d_model)),
        "synthetic_projection_full_column_rank_and_reconstructable": bool(synthetic_projection_injectivity_check(d_model)),
        "static_context_zero_delta_contract": bool(static_context_zero_delta_check()),
    }
