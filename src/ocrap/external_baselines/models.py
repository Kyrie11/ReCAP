from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _make_transformer_encoder(layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
    """Build a pre-norm encoder without the inapplicable nested-tensor probe.

    PyTorch's nested-tensor fast path is disabled internally for ``norm_first``
    encoders, but the default constructor still emits a warning on every model
    creation.  Explicitly disabling it keeps logs clean without changing the
    executed attention path.  The fallback preserves compatibility with older
    supported PyTorch releases.
    """
    try:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
    except TypeError:  # PyTorch builds predating the keyword.
        return nn.TransformerEncoder(layer, num_layers=int(num_layers))
import torch.nn.functional as F

from ocrap.external_baselines.source_ports import (
    GameFormerSourcePort, PlanTFSourcePort, PLUTOSourcePort,
)


ACTOR_TOPO_FEATURE_DIM = 16
MAP_TOPO_FEATURE_DIM = 14
GAMEFORMER_STATE_DIM = 9


@dataclass
class ExternalBaselineBatch:
    x: torch.Tensor
    mask: torch.Tensor
    target_index: torch.Tensor
    utility: torch.Tensor
    hard: torch.Tensor
    harm: torch.Tensor
    r_orc: torch.Tensor
    r_dep: torch.Tensor
    feasible: torch.Tensor
    branch_margins: torch.Tensor | None = None
    root_features: torch.Tensor | None = None
    root_probs: torch.Tensor | None = None
    root_valid: torch.Tensor | None = None
    option_valid: torch.Tensor | None = None
    ego_history: torch.Tensor | None = None
    neighbor_history: torch.Tensor | None = None
    neighbor_valid: torch.Tensor | None = None
    prefix_traj: torch.Tensor | None = None
    prefix_valid: torch.Tensor | None = None
    actor_topology_features: torch.Tensor | None = None
    actor_topology_target: torch.Tensor | None = None
    actor_topology_mask: torch.Tensor | None = None
    map_topology_features: torch.Tensor | None = None
    map_topology_target: torch.Tensor | None = None
    map_topology_mask: torch.Tensor | None = None


class ResidualMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1, out_dim: int = 1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
            )
            for _ in range(int(num_layers))
        ])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(h)
        for block in self.layers:
            z = z + block(z)
        return self.out(F.gelu(z))


class ScalarHeads(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.utility = nn.Linear(d_model, 1)
        self.hard = nn.Linear(d_model, 1)
        self.harm = nn.Linear(d_model, 1)
        self.r_orc = nn.Linear(d_model, 1)
        self.r_dep = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "utility": self.utility(h).squeeze(-1),
            "hard": self.hard(h).squeeze(-1),
            "harm": self.harm(h).squeeze(-1),
            "r_orc": self.r_orc(h).squeeze(-1),
            "r_dep": self.r_dep(h).squeeze(-1),
        }


class WayformerRouteBC(nn.Module):
    """Wayformer architecture port for executable ego-candidate behavior cloning.

    Wayformer itself is a *motion-forecasting* model, not a planner.  The v57
    adapter applied latent attention only to the executable-candidate tokens and
    therefore did not preserve the paper's central early-fusion scene encoder.
    This v58 adapter keeps the source/paper mechanisms that are meaningful under
    the common OC-RAP action interface:

      * homogeneous early fusion of agent-history and vector-map tokens,
      * a Perceiver-style trainable latent-query bottleneck,
      * repeated latent decoder attention, and
      * multimodal/query decoding projected onto the finite executable lattice.

    The final query is one executable candidate rather than one future-trajectory
    mixture component.  This is therefore intentionally named an architecture-
    faithful ego-BC *adapter*, never an official Wayformer planner reproduction.
    All scene inputs are observation-only and are shared once per candidate set.
    """

    def __init__(
        self,
        input_dim: int,
        max_candidates: int = 32,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.15,
        mlp_hidden: int = 128,
        mlp_layers: int = 4,
        num_latents: int = 96,
        num_encoder_layers: int = 2,
        num_decoder_layers: int | None = None,
        source_agent_state_dim: int = 9,
        source_map_point_dim: int = 6,
        source_map_meta_dim: int = 4,
        source_map_center_dim: int = 3,
        max_history_steps: int = 32,
        max_source_agents: int = 32,
        future_len: int = 20,
        num_output_queries: int = 64,
        num_mode_decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.num_latents = max(1, int(num_latents))
        self.num_decoder_layers = int(num_decoder_layers if num_decoder_layers is not None else num_layers)
        self.future_len = max(2, int(future_len))
        self.num_output_queries = max(1, int(num_output_queries))

        # Candidate queries are the benchmark projection of Wayformer's output
        # queries. They never enter the scene encoder, so scene compute is shared
        # across all executable candidates.
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.candidate_pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        self.candidate_type = nn.Parameter(torch.zeros(1, 1, d_model))

        # Paper/source early-fusion inputs.  Map points are pooled *within each
        # polyline* before homogeneous fusion.  This is an explicit runtime
        # projection that avoids flattening thousands of padded map points while
        # retaining vector-map geometry and map semantics.
        self.agent_proj = nn.Sequential(nn.Linear(int(source_agent_state_dim), d_model), nn.GELU())
        self.map_point_proj = nn.Sequential(nn.Linear(int(source_map_point_dim), d_model), nn.GELU())
        self.map_meta_proj = nn.Sequential(
            nn.Linear(int(source_map_meta_dim) + int(source_map_center_dim), d_model), nn.GELU()
        )
        self.agent_type = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.map_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, 1, int(max_history_steps), d_model))
        self.agent_pos = nn.Parameter(torch.zeros(1, int(max_source_agents), 1, d_model))

        self.latents = nn.Parameter(torch.randn(1, self.num_latents, d_model) * 0.02)
        self.scene_to_latent = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.latent_in_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=int(num_heads), dim_feedforward=4 * d_model,
            dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.latent_encoder = _make_transformer_encoder(enc_layer, int(num_encoder_layers))

        # Wayformer's decoder produces a multi-modal trajectory distribution.
        # Keep that native forecasting objective and project its likelihood onto
        # executable ego candidates instead of pretending Wayformer itself is a
        # planning policy.
        self.output_queries = nn.Parameter(torch.randn(1, self.num_output_queries, d_model) * 0.02)
        # Source/Paper Perceiver-style decoder: learned output queries alternate
        # cross-attention to the scene latents with query self-attention/FFN.
        # The public benchmark configuration uses eight layers; v58 keeps a
        # smaller explicit runtime depth while preserving the decoder operator.
        n_mode_layers = max(1, int(num_mode_decoder_layers))
        self.mode_cross = nn.ModuleList([
            nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            for _ in range(n_mode_layers)
        ])
        self.mode_self = nn.ModuleList([
            nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            for _ in range(n_mode_layers)
        ])
        self.mode_norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mode_layers)])
        self.mode_norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mode_layers)])
        self.mode_norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_mode_layers)])
        self.mode_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
            for _ in range(n_mode_layers)
        ])
        # Per future step: mean x/y, log sigma x/y, correlation coefficient.
        self.mode_traj_head = nn.Linear(d_model, self.future_len * 5)
        self.mode_score_head = nn.Linear(d_model, 1)

        self.decoder_cross = nn.ModuleList([
            nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            for _ in range(self.num_decoder_layers)
        ])
        self.decoder_self = nn.ModuleList([
            nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            for _ in range(self.num_decoder_layers)
        ])
        self.decoder_norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_decoder_layers)])
        self.decoder_norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_decoder_layers)])
        self.decoder_norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_decoder_layers)])
        self.decoder_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
            for _ in range(self.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = ResidualMLP(
            d_model, hidden_dim=int(mlp_hidden), num_layers=int(mlp_layers), dropout=float(dropout), out_dim=1
        )
        self.scalar_heads = ScalarHeads(d_model)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)

    def _scene_tokens(
        self,
        B: int,
        device: torch.device,
        *,
        source_agent_history: torch.Tensor | None,
        source_agent_valid: torch.Tensor | None,
        source_map_points: torch.Tensor | None,
        source_map_point_valid: torch.Tensor | None,
        source_map_meta: torch.Tensor | None,
        source_map_center: torch.Tensor | None,
        source_map_valid: torch.Tensor | None,
        fallback: torch.Tensor,
        fallback_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_agent_history is None or source_map_points is None:
            valid = torch.ones(fallback.shape[:2], dtype=torch.bool, device=device) if fallback_mask is None else fallback_mask.bool()
            return fallback, valid

        ah = source_agent_history.to(device=device, dtype=torch.float32)
        av = source_agent_valid
        if av is None:
            av = ah.abs().sum(dim=-1) > 0
        else:
            av = av.to(device=device).bool()
        A, H = ah.shape[1], ah.shape[2]
        agent = self.agent_proj(ah)
        if H <= self.time_pos.shape[2]:
            tp = self.time_pos[:, :, :H]
        else:
            extra = self.time_pos[:, :, -1:].expand(1, 1, H - self.time_pos.shape[2], self.d_model)
            tp = torch.cat([self.time_pos, extra], dim=2)
        if A <= self.agent_pos.shape[1]:
            ap = self.agent_pos[:, :A]
        else:
            extra = self.agent_pos[:, -1:].expand(1, A - self.agent_pos.shape[1], 1, self.d_model)
            ap = torch.cat([self.agent_pos, extra], dim=1)
        agent = agent + tp + ap + self.agent_type
        agent = agent.reshape(B, A * H, self.d_model)
        av_flat = av.reshape(B, A * H)

        mp = source_map_points.to(device=device, dtype=torch.float32)
        mpv = source_map_point_valid
        if mpv is None:
            mpv = mp.abs().sum(dim=-1) > 0
        else:
            mpv = mpv.to(device=device).bool()
        point = self.map_point_proj(mp)
        map_tok = self._masked_mean(point, mpv, dim=2)
        mm = torch.zeros(B, mp.shape[1], 4, device=device, dtype=torch.float32) if source_map_meta is None else source_map_meta.to(device=device, dtype=torch.float32)
        mc = torch.zeros(B, mp.shape[1], 3, device=device, dtype=torch.float32) if source_map_center is None else source_map_center.to(device=device, dtype=torch.float32)
        map_tok = map_tok + self.map_meta_proj(torch.cat([mm, mc], dim=-1)) + self.map_type
        mv = mpv.any(dim=-1) if source_map_valid is None else source_map_valid.to(device=device).bool()

        scene = torch.cat([agent, map_tok], dim=1)
        valid = torch.cat([av_flat, mv], dim=1)
        # MultiheadAttention cannot consume a batch row whose entire memory is
        # masked.  Keep one neutral token in that pathological case.
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            scene = scene.clone(); valid = valid.clone()
            scene[empty, 0] = 0.0
            valid[empty, 0] = True
        return scene, valid

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        source_agent_history: torch.Tensor | None = None,
        source_agent_valid: torch.Tensor | None = None,
        source_map_points: torch.Tensor | None = None,
        source_map_point_valid: torch.Tensor | None = None,
        source_map_meta: torch.Tensor | None = None,
        source_map_center: torch.Tensor | None = None,
        source_map_valid: torch.Tensor | None = None,
        prefix_traj: torch.Tensor | None = None,
        prefix_valid: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        q = self.candidate_proj(x) + self._position(N) + self.candidate_type
        cand_valid = torch.ones((B, N), dtype=torch.bool, device=x.device) if mask is None else mask.bool()
        scene, scene_valid = self._scene_tokens(
            B, x.device,
            source_agent_history=source_agent_history,
            source_agent_valid=source_agent_valid,
            source_map_points=source_map_points,
            source_map_point_valid=source_map_point_valid,
            source_map_meta=source_map_meta,
            source_map_center=source_map_center,
            source_map_valid=source_map_valid,
            fallback=q,
            fallback_mask=cand_valid,
        )
        lat = self.latents.expand(B, -1, -1)
        z, _ = self.scene_to_latent(lat, scene, scene, key_padding_mask=~scene_valid, need_weights=False)
        lat = self.latent_encoder(self.latent_in_norm(lat + z))

        mode_h = self.output_queries.expand(B, -1, -1)
        for cross, self_attn, n1, n2, n3, ffn in zip(
            self.mode_cross, self.mode_self, self.mode_norm1, self.mode_norm2, self.mode_norm3, self.mode_ffn,
        ):
            z, _ = cross(n1(mode_h), lat, lat, need_weights=False)
            mode_h = mode_h + z
            qn = n2(mode_h)
            z, _ = self_attn(qn, qn, qn, need_weights=False)
            mode_h = mode_h + z
            mode_h = mode_h + ffn(n3(mode_h))
        mode_params = self.mode_traj_head(mode_h).view(B, self.num_output_queries, self.future_len, 5)
        mode_logits = self.mode_score_head(mode_h).squeeze(-1)

        key_padding_mask = ~cand_valid
        h = q
        for cross, self_attn, n1, n2, n3, ffn in zip(
            self.decoder_cross, self.decoder_self, self.decoder_norm1,
            self.decoder_norm2, self.decoder_norm3, self.decoder_ffn,
        ):
            z, _ = cross(n1(h), lat, lat, need_weights=False)
            h = h + z
            z, _ = self_attn(n2(h), n2(h), n2(h), key_padding_mask=key_padding_mask, need_weights=False)
            h = h + z
            h = h + ffn(n3(h))
        h = self.norm(h)
        fallback_logits = self.policy_head(h).squeeze(-1)
        logits = fallback_logits
        if prefix_traj is not None:
            cand = prefix_traj.to(device=x.device, dtype=torch.float32)
            if cand.shape[-2] != self.future_len:
                # Runtime contracts normally build exactly ``future_len`` points;
                # retain a deterministic interpolation fallback for old caches.
                flat = cand.reshape(B * cand.shape[1], cand.shape[2], 2).permute(0, 2, 1)
                cand = F.interpolate(flat, size=self.future_len, mode="linear", align_corners=True).permute(0, 2, 1).reshape(B, -1, self.future_len, 2)
            valid_t = torch.ones(cand.shape[:-1], dtype=torch.bool, device=x.device) if prefix_valid is None else prefix_valid.to(device=x.device).bool()
            if valid_t.shape[-1] != self.future_len:
                valid_t = F.interpolate(valid_t.float().reshape(B * valid_t.shape[1], 1, -1), size=self.future_len, mode="nearest").reshape(B, -1, self.future_len) > 0.5
            mu = mode_params[..., :2]
            log_s = mode_params[..., 2:4].clamp(-1.609, 5.0)
            rho = mode_params[..., 4].clamp(-0.5, 0.5)
            diff = cand[:, :, None, :, :] - mu[:, None, :, :, :]
            sx = torch.exp(log_s[..., 0])[:, None]
            sy = torch.exp(log_s[..., 1])[:, None]
            rr = rho[:, None]
            dx = diff[..., 0]; dy = diff[..., 1]
            one_m = (1.0 - rr.square()).clamp_min(1.0e-4)
            z2 = (dx / sx).square() + (dy / sy).square() - 2.0 * rr * dx * dy / (sx * sy)
            point_nll = log_s[..., 0][:, None] + log_s[..., 1][:, None] + 0.5 * torch.log(one_m) + 0.5 * z2 / one_m
            vm = valid_t[:, :, None, :].to(point_nll.dtype)
            mode_nll = (point_nll * vm).sum(-1) / vm.sum(-1).clamp_min(1.0)
            log_mix = torch.log_softmax(mode_logits.float(), dim=-1)[:, None, :]
            logits = torch.logsumexp(log_mix - mode_nll.float(), dim=-1)
        logits = logits.masked_fill(~cand_valid, -1.0e4)
        out = {
            "logits": logits,
            "wayformer_mode_params": mode_params,
            "wayformer_mode_logits": mode_logits,
            "wayformer_scene_latent_norm": lat.float().norm(dim=-1).mean(dim=-1),
        }
        out.update(self.scalar_heads(h))
        return out

    def _position(self, N: int) -> torch.Tensor:
        if N > self.candidate_pos.shape[1]:
            extra = self.candidate_pos[:, -1:, :].expand(1, N - self.candidate_pos.shape[1], -1)
            return torch.cat([self.candidate_pos, extra], dim=1)[:, :N]
        return self.candidate_pos[:, :N]


class GameFormerFutureEncoder(nn.Module):
    """FutureEncoder analogue from the uploaded GameFormer source."""

    def __init__(self, d_model: int, future_len: int, dropout: float) -> None:
        super().__init__()
        self.future_len = int(future_len)
        # Per-step state: x,y,heading,vx,vy,width,length,valid-like channel.
        self.step_mlp = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())

    @staticmethod
    def _stable_motion_state(traj_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return velocity and heading without an undefined atan2 backward.

        The decoder trajectory is a cumulative displacement from the current ego
        origin, so the first finite difference must use (0, 0) as its previous
        point.  The old code reused the first predicted point itself, artificially
        forcing the first velocity to zero and feeding ``atan2(0, 0)`` into the
        graph.  That angle has an undefined mathematical derivative and can yield
        non-finite CUDA/BF16 gradients even when every forward loss is finite.
        Genuine stationary steps still have no defined heading, so give them a
        neutral +x direction *before* atan2; the boolean branch deliberately
        carries zero heading gradient for those near-stationary steps.
        """
        # Do the finite-difference / angle math in FP32 even when the surrounding
        # GameFormer forward is under BF16 autocast.  This is tiny compared with
        # attention/LSTM compute and removes precision-sensitive angle gradients.
        traj32 = traj_xy.float()
        origin = torch.zeros_like(traj32[..., :1, :])
        prev = torch.cat([origin, traj32[..., :-1, :]], dim=-2)
        dxy = traj32 - prev
        motion_sq = dxy.square().sum(dim=-1)
        moving = motion_sq > 1.0e-8
        safe_dx = torch.where(moving, dxy[..., 0], torch.ones_like(dxy[..., 0]))
        safe_dy = torch.where(moving, dxy[..., 1], torch.zeros_like(dxy[..., 1]))
        heading = torch.atan2(safe_dy, safe_dx).unsqueeze(-1)
        vel = dxy / 0.1
        return vel, heading

    def forward(self, traj_xy: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # traj_xy: [B,N,M,T,2], scores: [B,N,M]
        B, N, M, T, _ = traj_xy.shape
        vel32, heading32 = self._stable_motion_state(traj_xy)
        # Keep the learned MLP in the surrounding autocast dtype for throughput;
        # only the numerically sensitive geometric transform above is FP32.
        state_dtype = traj_xy.dtype
        size = torch.ones(B, N, M, T, 2, device=traj_xy.device, dtype=state_dtype)
        valid = torch.ones(B, N, M, T, 1, device=traj_xy.device, dtype=state_dtype)
        state = torch.cat([traj_xy, heading32.to(state_dtype), vel32.to(state_dtype), size, valid], dim=-1)
        # This is the legacy/generic GameFormer adapter, not the source-faithful
        # GameFormerFutureEncoderSource used by the main-table source port.  Its
        # decoder trajectory is an internal learned prediction and must retain
        # gradient flow.  The source-faithful class keeps the paper/source
        # detach() semantics separately in source_ports.py.
        step = self.step_mlp(state)
        pooled = step.max(dim=-2).values
        # Softmax is cheap and is a second numerically sensitive reduction.
        # Evaluate it in FP32 and cast the probabilities back for the weighted sum.
        weights = torch.softmax(scores.float().clamp(-50.0, 50.0), dim=-1).to(pooled.dtype).unsqueeze(-1)
        return self.pool((pooled * weights).sum(dim=2))


class GameFormerLevelK(nn.Module):
    """GameFormer adapter that preserves encoder, multi-modal decoding and level-k reasoning.

    The uploaded GameFormer source uses AgentEncoder/LaneEncoder/CrosswalkEncoder,
    a Transformer fusion encoder, an InitialDecoder with learned modal/agent
    queries, then InteractionDecoder blocks whose FutureEncoder consumes the
    previous level's trajectories.  This adapter implements the same algorithmic
    pattern over OC-RAP grouped candidate-prefix tensors.
    """

    def __init__(self, input_dim: int, max_candidates: int = 32, d_model: int = 256, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.15, num_levels: int = 4, modalities: int = 6, future_len: int = 20, history_len: int = 11, neighbors_to_predict: int = 8, root_feature_dim: int = 18, num_roots: int = 10, num_options: int = 12, use_teacher_branch_context: bool = False, traj_step_scale: float = 1.5) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.num_levels = int(num_levels)
        self.modalities = int(modalities)
        self.future_len = int(future_len)
        self.history_len = int(history_len)
        self.neighbors_to_predict = int(neighbors_to_predict)
        self.num_roots = int(num_roots)
        self.num_options = int(num_options)
        self.root_feature_dim = int(root_feature_dim)
        self.use_teacher_branch_context = bool(use_teacher_branch_context)
        self.traj_step_scale = float(traj_step_scale)

        self.candidate_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        self.ego_encoder = nn.LSTM(GAMEFORMER_STATE_DIM, d_model // 2, num_layers=2, batch_first=True, dropout=dropout)
        self.agent_encoder = nn.LSTM(GAMEFORMER_STATE_DIM, d_model // 2, num_layers=2, batch_first=True, dropout=dropout)
        self.history_fuse = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4 * d_model, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True)
        self.fusion_encoder = _make_transformer_encoder(enc_layer, int(num_layers))

        branch_in = self.root_feature_dim + self.num_options + 2
        if self.use_teacher_branch_context:
            self.branch_point = nn.Sequential(nn.LayerNorm(branch_in), nn.Linear(branch_in, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
            self.branch_pool = nn.Linear(d_model, d_model)
        else:
            self.branch_point = None
            self.branch_pool = None
        self.neighbor_token_proj = nn.Sequential(nn.LayerNorm(d_model // 2), nn.Linear(d_model // 2, d_model), nn.GELU(), nn.Dropout(dropout))

        self.modal_query = nn.Embedding(self.modalities, d_model)
        self.level0_cross = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.level_cross = nn.ModuleList([nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True) for _ in range(max(self.num_levels, 1))])
        self.level_self = nn.ModuleList([
            _make_transformer_encoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4*d_model, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True), 1)
            for _ in range(max(self.num_levels, 1))
        ])
        self.future_encoder = GameFormerFutureEncoder(d_model, self.future_len, dropout)
        self.content_norm = nn.LayerNorm(d_model)
        self.traj_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout), nn.Linear(d_model, self.future_len * 4)) for _ in range(self.num_levels + 1)])
        self.score_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.ELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)) for _ in range(self.num_levels + 1)])
        self.policy_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(self.num_levels + 1)])
        self.scalar_heads = ScalarHeads(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, branch_margins: torch.Tensor | None = None, root_features: torch.Tensor | None = None, root_probs: torch.Tensor | None = None, root_valid: torch.Tensor | None = None, ego_history: torch.Tensor | None = None, neighbor_history: torch.Tensor | None = None, neighbor_valid: torch.Tensor | None = None, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        key_padding_mask = None if mask is None else ~mask.bool()
        scene = self.candidate_proj(x) + self._position(N)
        hist_ctx, neighbor_tokens, neighbor_mask = self._encode_history(B, N, x.device, ego_history, neighbor_history, neighbor_valid)
        branch_ctx = self._encode_branch(B, N, x.device, branch_margins, root_features, root_probs, root_valid)
        scene = scene + hist_ctx + branch_ctx
        scene = self.fusion_encoder(scene, src_key_padding_mask=key_padding_mask)

        content, traj, scores = self._initial_decode(scene, mask)
        level_logits: list[torch.Tensor] = []
        level_trajs: list[torch.Tensor] = []
        level_scores: list[torch.Tensor] = []
        logits0 = self._candidate_logits(content, scores, 0, mask)
        level_logits.append(logits0)
        level_trajs.append(traj)
        level_scores.append(scores)

        for k in range(1, self.num_levels + 1):
            # GameFormer level-k reasoning: the ego response at level k is
            # conditioned on the observed neighboring-agent tokens and on the
            # previous level's multimodal ego future.  Candidate alternatives are
            # not incorrectly treated as traffic agents.
            future_ctx = self.future_encoder(traj[..., :2], scores)
            interaction_tokens = torch.cat([future_ctx.unsqueeze(2), neighbor_tokens], dim=2)
            interaction_mask = torch.cat([
                torch.ones(B, N, 1, dtype=torch.bool, device=x.device), neighbor_mask
            ], dim=2)
            flat_tokens = interaction_tokens.reshape(B * N, interaction_tokens.shape[2], self.d_model)
            flat_mask = ~interaction_mask.reshape(B * N, interaction_mask.shape[2])
            inter = self.level_self[k - 1](flat_tokens, src_key_padding_mask=flat_mask)
            inter_ctx = inter[:, 0].reshape(B, N, self.d_model)
            q = (content + scene[:, :, None, :] + inter_ctx[:, :, None, :]).reshape(B * N, self.modalities, self.d_model)
            mem = neighbor_tokens.reshape(B * N, self.neighbors_to_predict, self.d_model)
            kmask = ~neighbor_mask.reshape(B * N, self.neighbors_to_predict)
            # MultiheadAttention cannot attend to an all-masked row. Keep one
            # neutral token active when no neighbor is observed.
            empty = kmask.all(dim=-1)
            if bool(empty.any()):
                kmask = kmask.clone()
                kmask[empty, 0] = False
                mem = mem.clone()
                mem[empty, 0] = 0.0
            q2, _ = self.level_cross[k - 1](q, mem, mem, key_padding_mask=kmask, need_weights=False)
            content = self.content_norm((q + q2).reshape(B, N, self.modalities, self.d_model))
            traj, scores = self._predict_traj(content, k)
            level_logits.append(self._candidate_logits(content, scores, k, mask))
            level_trajs.append(traj)
            level_scores.append(scores)

        h = self.final_norm(scene + content.mean(dim=2))
        out = {
            "logits": level_logits[-1],
            "level_logits": level_logits,
            "gameformer_level_trajs": level_trajs,
            "gameformer_level_scores": level_scores,
        }
        out.update(self.scalar_heads(h))
        return out

    def _position(self, N: int) -> torch.Tensor:
        if N > self.pos.shape[1]:
            extra = self.pos[:, -1:, :].expand(1, N - self.pos.shape[1], -1)
            return torch.cat([self.pos, extra], dim=1)[:, :N]
        return self.pos[:, :N]

    def _encode_history(self, B: int, N: int, device: torch.device, ego_history: torch.Tensor | None, neighbor_history: torch.Tensor | None, neighbor_valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ego_history is None:
            zeros = torch.zeros(B, N, self.d_model, device=device)
            return zeros, torch.zeros(B, N, self.neighbors_to_predict, self.d_model, device=device), torch.zeros(B, N, self.neighbors_to_predict, dtype=torch.bool, device=device)
        ego = self._pad_last2(ego_history.to(device=device, dtype=torch.float32), self.history_len, GAMEFORMER_STATE_DIM)
        ego_flat = ego.reshape(B * N, self.history_len, GAMEFORMER_STATE_DIM)
        _, (ego_h, _) = self.ego_encoder(ego_flat)
        ego_ctx = ego_h[-1].reshape(B, N, -1)
        if neighbor_history is None:
            nh = torch.zeros(B, N, self.neighbors_to_predict, self.d_model // 2, device=device)
            valid_actor = torch.zeros(B, N, self.neighbors_to_predict, dtype=torch.bool, device=device)
        else:
            neigh = self._pad_last3(neighbor_history.to(device=device, dtype=torch.float32), self.neighbors_to_predict, self.history_len, GAMEFORMER_STATE_DIM)
            neigh_flat = neigh.reshape(B * N * self.neighbors_to_predict, self.history_len, GAMEFORMER_STATE_DIM)
            _, (encoded, _) = self.agent_encoder(neigh_flat)
            nh = encoded[-1].reshape(B, N, self.neighbors_to_predict, -1)
            if neighbor_valid is not None:
                nv = self._pad_last2(neighbor_valid.to(device=device, dtype=torch.float32), self.neighbors_to_predict, self.history_len)
                valid_actor = nv.sum(dim=-1) > 0
            else:
                valid_actor = torch.ones(B, N, self.neighbors_to_predict, dtype=torch.bool, device=device)
        w = valid_actor.float().unsqueeze(-1)
        neigh_ctx = (nh * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        fused = self.history_fuse(torch.cat([ego_ctx, neigh_ctx], dim=-1))
        tokens = self.neighbor_token_proj(nh) * w
        return fused, tokens, valid_actor

    def _encode_branch(self, B: int, N: int, device: torch.device, branch_margins: torch.Tensor | None, root_features: torch.Tensor | None, root_probs: torch.Tensor | None, root_valid: torch.Tensor | None) -> torch.Tensor:
        K, L, Fdim = self.num_roots, self.num_options, self.root_feature_dim
        if not self.use_teacher_branch_context or self.branch_point is None or self.branch_pool is None:
            return torch.zeros(B, N, self.d_model, device=device)
        if root_features is None:
            root_features = torch.zeros(B, N, K, Fdim, device=device)
        else:
            root_features = self._pad_last2(root_features.to(device=device, dtype=torch.float32), K, Fdim)
        if branch_margins is None:
            branch_margins = torch.zeros(B, N, K, L, device=device)
        else:
            branch_margins = self._pad_last2(branch_margins.to(device=device, dtype=torch.float32), K, L)
            branch_margins = torch.nan_to_num(branch_margins, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        if root_probs is None:
            root_probs = torch.full((B, N, K), 1.0 / max(K, 1), device=device)
        else:
            root_probs = self._pad_1d(root_probs.to(device=device, dtype=torch.float32), K).clamp_min(0.0)
        if root_valid is None:
            root_valid = torch.ones(B, N, K, device=device, dtype=torch.float32)
        else:
            root_valid = self._pad_1d(root_valid.to(device=device, dtype=torch.float32), K)
        root_probs = root_probs * (root_valid > 0.5).float()
        root_probs = root_probs / root_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        point = torch.cat([root_features, branch_margins, root_probs.unsqueeze(-1), root_valid.unsqueeze(-1)], dim=-1)
        enc = self.branch_point(point)
        return self.branch_pool((enc * root_probs.unsqueeze(-1)).sum(dim=2))

    def _initial_decode(self, scene: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = scene.shape
        modal = self.modal_query.weight[None, None, :, :].expand(B, N, self.modalities, self.d_model)
        q = scene[:, :, None, :] + modal
        q_flat = q.reshape(B * N, self.modalities, self.d_model)
        mem = scene[:, None, :, :].expand(B, N, N, self.d_model).reshape(B * N, N, self.d_model)
        kmask = (~mask.bool())[:, None, :].expand(B, N, N).reshape(B * N, N) if mask is not None else None
        q2, _ = self.level0_cross(q_flat, mem, mem, key_padding_mask=kmask, need_weights=False)
        content = self.content_norm((q_flat + q2).reshape(B, N, self.modalities, self.d_model))
        traj, scores = self._predict_traj(content, 0)
        return content, traj, scores

    def _predict_traj(self, content: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, M, _ = content.shape
        raw = self.traj_heads[level](content).view(B, N, M, self.future_len, 4)
        # Cumulative displacement gives dynamically smoother trajectories than
        # unconstrained absolute points and mirrors GameFormer's trajectory head.
        xy = torch.cumsum(torch.tanh(raw[..., :2]) * self.traj_step_scale, dim=-2)
        logsig = raw[..., 2:4].clamp(-5.0, 5.0)
        traj = torch.cat([xy, logsig], dim=-1)
        scores = self.score_heads[level](content).squeeze(-1)
        return traj, scores

    def _candidate_logits(self, content: torch.Tensor, scores: torch.Tensor, level: int, mask: torch.Tensor | None) -> torch.Tensor:
        h = content.mean(dim=2)
        logits = self.policy_heads[level](h).squeeze(-1) + torch.logsumexp(scores, dim=-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        return logits

    @staticmethod
    def _pad_1d(x: torch.Tensor, n: int) -> torch.Tensor:
        if x.shape[-1] == n:
            return x
        out = x.new_zeros(*x.shape[:-1], n)
        m = min(n, x.shape[-1])
        if m > 0:
            out[..., :m] = x[..., :m]
        return out

    @staticmethod
    def _pad_last2(x: torch.Tensor, n0: int, n1: int) -> torch.Tensor:
        if x.shape[-2] == n0 and x.shape[-1] == n1:
            return x
        out = x.new_zeros(*x.shape[:-2], n0, n1)
        m0 = min(n0, x.shape[-2]); m1 = min(n1, x.shape[-1])
        if m0 > 0 and m1 > 0:
            out[..., :m0, :m1] = x[..., :m0, :m1]
        return out

    @staticmethod
    def _pad_last3(x: torch.Tensor, n0: int, n1: int, n2: int) -> torch.Tensor:
        if x.shape[-3:] == (n0, n1, n2):
            return x
        out = x.new_zeros(*x.shape[:-3], n0, n1, n2)
        m0 = min(n0, x.shape[-3]); m1 = min(n1, x.shape[-2]); m2 = min(n2, x.shape[-1])
        if m0 > 0 and m1 > 0 and m2 > 0:
            out[..., :m0, :m1, :m2] = x[..., :m0, :m1, :m2]
        return out


class TopoFuser(nn.Module):
    """Source-structured BeTop TopoFuser.

    The official implementation projects source and target features separately
    to ``d/2``, concatenates the pairwise embeddings, and *adds* the previous
    topology feature before the topology decoder.  v57 concatenated ``prev`` as
    a third raw input, which changes the iterative topology semantics.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        half = max(1, int(d_model) // 2)
        self.src_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_model, half))
        self.tgt_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_model, half))
        self.out_dim = 2 * half
        self.out_proj = nn.Identity() if self.out_dim == d_model else nn.Linear(self.out_dim, d_model)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
        feat = self.out_proj(torch.cat([self.src_mlp(src), self.tgt_mlp(tgt)], dim=-1))
        if prev is not None:
            feat = feat + prev
        return feat


class BeTopNetLite(nn.Module):
    """BeTopNet adapter preserving behavioral topology reasoning.

    The uploaded BeTopNet source builds an MTR-style encoder/decoder, predicts
    actor and map topology, selects topological neighbors, applies topology-aware
    attention, and uses focal/top-k topology losses.  This adapter keeps those
    core mechanisms while consuming OC-RAP grouped candidate tensors instead of
    BeTop's native WOMD cache.
    """

    def __init__(self, input_dim: int, max_candidates: int = 32, d_model: int = 256, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.15, actor_topology_feature_dim: int = ACTOR_TOPO_FEATURE_DIM, map_topology_feature_dim: int = MAP_TOPO_FEATURE_DIM, num_topology_agents: int = 16, num_topology_map: int = 64, num_topo: int = 16, mlp_hidden: int = 128, mlp_layers: int = 3) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.num_topology_agents = int(num_topology_agents)
        self.num_topology_map = int(num_topology_map)
        self.actor_topology_feature_dim = int(actor_topology_feature_dim)
        self.map_topology_feature_dim = int(map_topology_feature_dim)
        self.num_topo = int(num_topo)
        self.token_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4 * d_model, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True)
        self.scene_encoder = _make_transformer_encoder(enc_layer, int(num_layers))

        self.actor_proj = nn.Sequential(nn.LayerNorm(self.actor_topology_feature_dim), nn.Linear(self.actor_topology_feature_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.map_proj = nn.Sequential(nn.LayerNorm(self.map_topology_feature_dim), nn.Linear(self.map_topology_feature_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.actor_fusers = nn.ModuleList([TopoFuser(d_model, dropout) for _ in range(int(num_layers))])
        self.map_fusers = nn.ModuleList([TopoFuser(d_model, dropout) for _ in range(int(num_layers))])
        self.actor_topo_decoders = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)) for _ in range(int(num_layers))])
        self.map_topo_decoders = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)) for _ in range(int(num_layers))])
        self.actor_attn = nn.ModuleList([nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True) for _ in range(int(num_layers))])
        self.map_attn = nn.ModuleList([nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True) for _ in range(int(num_layers))])
        self.update_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(num_layers))])
        self.policy_head = ResidualMLP(d_model, hidden_dim=int(mlp_hidden), num_layers=int(mlp_layers), dropout=float(dropout), out_dim=1)
        self.scalar_heads = ScalarHeads(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, actor_topology_features: torch.Tensor | None = None, actor_topology_mask: torch.Tensor | None = None, map_topology_features: torch.Tensor | None = None, map_topology_mask: torch.Tensor | None = None, topology_features: torch.Tensor | None = None, topology_mask: torch.Tensor | None = None, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.token_proj(x) + self._position(N)
        h = self.scene_encoder(h, src_key_padding_mask=key_padding_mask)

        if actor_topology_features is None and topology_features is not None:
            actor_topology_features = topology_features
        if actor_topology_mask is None and topology_mask is not None:
            actor_topology_mask = topology_mask
        actor_mem, actor_mask = self._topology_memory(B, N, x.device, actor_topology_features, actor_topology_mask, self.num_topology_agents, self.actor_topology_feature_dim, self.actor_proj)
        map_mem, map_mask = self._topology_memory(B, N, x.device, map_topology_features, map_topology_mask, self.num_topology_map, self.map_topology_feature_dim, self.map_proj)

        actor_prev: torch.Tensor | None = None
        map_prev: torch.Tensor | None = None
        actor_logits_levels: list[torch.Tensor] = []
        map_logits_levels: list[torch.Tensor] = []
        for i in range(len(self.actor_fusers)):
            src_actor = h[:, :, None, :].expand(B, N, self.num_topology_agents, self.d_model)
            actor_feat = self.actor_fusers[i](src_actor, actor_mem, actor_prev)
            actor_logits = self.actor_topo_decoders[i](actor_feat).squeeze(-1)
            actor_logits = actor_logits.masked_fill(~actor_mask.bool(), -1.0e4)
            actor_prev = actor_feat
            h = self._apply_topo_attention(h, actor_mem, actor_logits, actor_mask, self.actor_attn[i], mask)

            src_map = h[:, :, None, :].expand(B, N, self.num_topology_map, self.d_model)
            map_feat = self.map_fusers[i](src_map, map_mem, map_prev)
            map_logits = self.map_topo_decoders[i](map_feat).squeeze(-1)
            map_logits = map_logits.masked_fill(~map_mask.bool(), -1.0e4)
            map_prev = map_feat
            h = self._apply_topo_attention(h, map_mem, map_logits, map_mask, self.map_attn[i], mask)
            h = self.update_norm[i](h)
            actor_logits_levels.append(actor_logits)
            map_logits_levels.append(map_logits)

        h = self.norm(h)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        out = {
            "logits": logits,
            "actor_topo_logits": actor_logits_levels[-1].unsqueeze(-1),
            "map_topo_logits": map_logits_levels[-1].unsqueeze(-1),
            "actor_topo_logits_levels": [z.unsqueeze(-1) for z in actor_logits_levels],
            "map_topo_logits_levels": [z.unsqueeze(-1) for z in map_logits_levels],
            # Legacy key for old policy code; shape [B,N,A,1].
            "topology_logits": actor_logits_levels[-1].unsqueeze(-1),
        }
        out.update(self.scalar_heads(h))
        return out

    def _position(self, N: int) -> torch.Tensor:
        if N > self.pos.shape[1]:
            extra = self.pos[:, -1:, :].expand(1, N - self.pos.shape[1], -1)
            return torch.cat([self.pos, extra], dim=1)[:, :N]
        return self.pos[:, :N]

    def _topology_memory(self, B: int, N: int, device: torch.device, feats: torch.Tensor | None, mask: torch.Tensor | None, count: int, fdim: int, proj: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        if feats is None:
            feats = torch.zeros(B, N, count, fdim, device=device)
        else:
            feats = self._pad_last2(feats.to(device=device, dtype=torch.float32), count, fdim)
        if mask is None:
            mask = feats.abs().sum(dim=-1) > 0
        else:
            mask = self._pad_1d(mask.to(device=device, dtype=torch.float32), count) > 0.5
        return proj(feats), mask.bool()

    def _apply_topo_attention(self, h: torch.Tensor, mem: torch.Tensor, logits: torch.Tensor, valid_mask: torch.Tensor, attn: nn.MultiheadAttention, cand_mask: torch.Tensor | None) -> torch.Tensor:
        B, N, Kall, D = mem.shape
        k = min(max(1, self.num_topo), Kall)
        score = torch.sigmoid(logits.detach()).masked_fill(~valid_mask.bool(), -1.0e4)
        idx = torch.topk(score, k=k, dim=-1).indices
        gather = idx.unsqueeze(-1).expand(B, N, k, D)
        selected = torch.gather(mem, dim=2, index=gather).reshape(B * N, k, D)
        selected_valid = torch.gather(valid_mask.bool(), dim=2, index=idx).reshape(B * N, k)
        empty = ~selected_valid.any(dim=-1)
        if bool(empty.any()):
            selected = selected.clone(); selected_valid = selected_valid.clone()
            selected[empty, 0] = 0.0
            selected_valid[empty, 0] = True
        q = h.reshape(B * N, 1, D)
        out, _ = attn(q, selected, selected, key_padding_mask=~selected_valid, need_weights=False)
        out = out.reshape(B, N, D)
        if cand_mask is not None:
            out = out * cand_mask.bool().unsqueeze(-1).float()
        return h + out

    @staticmethod
    def _pad_1d(x: torch.Tensor, n: int) -> torch.Tensor:
        if x.shape[-1] == n:
            return x
        out = x.new_zeros(*x.shape[:-1], n)
        m = min(n, x.shape[-1])
        if m > 0:
            out[..., :m] = x[..., :m]
        return out

    @staticmethod
    def _pad_last2(x: torch.Tensor, n0: int, n1: int) -> torch.Tensor:
        if x.shape[-2] == n0 and x.shape[-1] == n1:
            return x
        out = x.new_zeros(*x.shape[:-2], n0, n1)
        m0 = min(n0, x.shape[-2]); m1 = min(n1, x.shape[-1])
        if m0 > 0 and m1 > 0:
            out[..., :m0, :m1] = x[..., :m0, :m1]
        return out


class _SceneContextEncoder(nn.Module):
    """Observation-only scene encoder shared by direct-planning adapters.

    The external-baseline dataset exposes the same deployable history/map tensors
    to all adapters.  This encoder deliberately avoids OC-RAP teacher/root labels.
    """

    def __init__(self, d_model: int, dropout: float, actor_dim: int = ACTOR_TOPO_FEATURE_DIM, map_dim: int = MAP_TOPO_FEATURE_DIM) -> None:
        super().__init__()
        self.ego = nn.Sequential(nn.Linear(GAMEFORMER_STATE_DIM, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.neighbor = nn.Sequential(nn.Linear(GAMEFORMER_STATE_DIM, d_model), nn.GELU())
        self.actor = nn.Sequential(nn.Linear(actor_dim, d_model), nn.GELU())
        self.map = nn.Sequential(nn.Linear(map_dim, d_model), nn.GELU())
        self.fuse = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Dropout(dropout))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=dim)
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ego_history: torch.Tensor | None = None,
        neighbor_history: torch.Tensor | None = None,
        neighbor_valid: torch.Tensor | None = None,
        actor_topology_features: torch.Tensor | None = None,
        actor_topology_mask: torch.Tensor | None = None,
        map_topology_features: torch.Tensor | None = None,
        map_topology_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        z0 = x.new_zeros(B, N, self.ego[0].out_features)
        if ego_history is not None and ego_history.numel():
            z_ego = self.ego(ego_history[..., -1, :])
        else:
            z_ego = z0
        if neighbor_history is not None and neighbor_history.numel():
            nh = self.neighbor(neighbor_history)
            hist_mask = neighbor_valid.bool() if neighbor_valid is not None else None
            nh = self._masked_mean(nh, hist_mask, dim=-2)
            agent_mask = hist_mask.any(dim=-1) if hist_mask is not None else None
            z_nei = self._masked_mean(nh, agent_mask, dim=-2)
        else:
            z_nei = z0
        if actor_topology_features is not None and actor_topology_features.numel():
            af = self.actor(actor_topology_features)
            z_actor = self._masked_mean(af, actor_topology_mask.bool() if actor_topology_mask is not None else None, dim=-2)
        else:
            z_actor = z0
        if map_topology_features is not None and map_topology_features.numel():
            mf = self.map(map_topology_features)
            z_map = self._masked_mean(mf, map_topology_mask.bool() if map_topology_mask is not None else None, dim=-2)
        else:
            z_map = z0
        return self.fuse(torch.cat([z_ego, z_nei, z_actor, z_map], dim=-1))


class PlanTFAdapter(nn.Module):
    """PlanTF paper-core adapter over the common OC-RAP candidate lattice.

    It keeps the two design ideas that transfer cleanly across datasets: a compact
    vector/state representation fused by self-attention, and state-dropout during
    imitation training to reduce over-reliance on scene state.  The official
    nuPlan trajectory decoder is replaced by scoring executable candidate prefixes.
    """

    def __init__(self, input_dim: int, max_candidates: int = 32, d_model: int = 128, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.10, state_dropout: float = 0.25, actor_topology_feature_dim: int = ACTOR_TOPO_FEATURE_DIM, map_topology_feature_dim: int = MAP_TOPO_FEATURE_DIM) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.state_dropout = float(state_dropout)
        self.token = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
        self.context = _SceneContextEncoder(d_model, dropout, actor_topology_feature_dim, map_topology_feature_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4*d_model, dropout=float(dropout), activation='gelu', batch_first=True, norm_first=True)
        self.encoder = _make_transformer_encoder(layer, int(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.scalar_heads = ScalarHeads(d_model)

    def _position(self, n: int) -> torch.Tensor:
        if n <= self.pos.shape[1]:
            return self.pos[:, :n]
        return torch.cat([self.pos, self.pos[:, -1:].expand(1, n-self.pos.shape[1], -1)], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.token(x)
        ctx = self.context(x, **{k: kwargs.get(k) for k in ('ego_history','neighbor_history','neighbor_valid','actor_topology_features','actor_topology_mask','map_topology_features','map_topology_mask')})
        if self.training and self.state_dropout > 0:
            keep = (torch.rand(ctx.shape[:-1], device=ctx.device) >= self.state_dropout).to(ctx.dtype).unsqueeze(-1)
            ctx = ctx * keep
        h = h + ctx + self._position(h.shape[1])
        h = self.encoder(h, src_key_padding_mask=None if mask is None else ~mask.bool())
        h = self.norm(h)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        out = {'logits': logits}
        out.update(self.scalar_heads(h))
        return out


class PLUTOAdapter(nn.Module):
    """PLUTO query-style lateral/longitudinal imitation adapter.

    Maneuver queries model lateral and longitudinal intent jointly.  Candidate
    prefix displacement supplies the maneuver coordinate, while a scene-query
    contrastive head provides the CIL-style within-scene discrimination signal.
    """

    def __init__(self, input_dim: int, max_candidates: int = 32, d_model: int = 192, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.10, lateral_queries: int = 5, longitudinal_queries: int = 5, contrastive_temperature: float = 0.10, actor_topology_feature_dim: int = ACTOR_TOPO_FEATURE_DIM, map_topology_feature_dim: int = MAP_TOPO_FEATURE_DIM) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.temperature = max(float(contrastive_temperature), 1e-3)
        self.token = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
        self.context = _SceneContextEncoder(d_model, dropout, actor_topology_feature_dim, map_topology_feature_dim)
        self.maneuver = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.lat_q = nn.Parameter(torch.randn(1, int(lateral_queries), d_model) * 0.02)
        self.lon_q = nn.Parameter(torch.randn(1, int(longitudinal_queries), d_model) * 0.02)
        self.query_attn = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4*d_model, dropout=float(dropout), activation='gelu', batch_first=True, norm_first=True)
        self.encoder = _make_transformer_encoder(layer, int(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, 1)
        self.scalar_heads = ScalarHeads(d_model)
        self.scene_proj = nn.Linear(d_model, d_model)
        self.cand_proj = nn.Linear(d_model, d_model)

    def _position(self, n: int) -> torch.Tensor:
        if n <= self.pos.shape[1]: return self.pos[:, :n]
        return torch.cat([self.pos, self.pos[:, -1:].expand(1, n-self.pos.shape[1], -1)], dim=1)

    def _maneuver_features(self, x: torch.Tensor, prefix_traj: torch.Tensor | None, prefix_valid: torch.Tensor | None) -> torch.Tensor:
        B,N,_ = x.shape
        if prefix_traj is None or prefix_traj.numel() == 0:
            return x.new_zeros(B,N,4)
        pt = prefix_traj
        if prefix_valid is None:
            end = pt[..., -1, :]
        else:
            idx = prefix_valid.long().sum(dim=-1).clamp_min(1) - 1
            gather = idx[..., None, None].expand(B,N,1,2)
            end = torch.gather(pt, dim=-2, index=gather).squeeze(-2)
        start = pt[..., 0, :]
        d = end - start
        dist = torch.linalg.norm(d, dim=-1, keepdim=True)
        lat_ratio = d[..., 1:2] / dist.clamp_min(1e-3)
        return torch.cat([d, dist, lat_ratio], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, prefix_traj: torch.Tensor | None = None, prefix_valid: torch.Tensor | None = None, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        B,N,_ = x.shape
        ctx = self.context(x, **{k: kwargs.get(k) for k in ('ego_history','neighbor_history','neighbor_valid','actor_topology_features','actor_topology_mask','map_topology_features','map_topology_mask')})
        h = self.token(x) + ctx + self.maneuver(self._maneuver_features(x, prefix_traj, prefix_valid)) + self._position(N)
        queries = torch.cat([self.lat_q.expand(B,-1,-1), self.lon_q.expand(B,-1,-1)], dim=1)
        qctx,_ = self.query_attn(h, queries, queries, need_weights=False)
        h = self.encoder(h + qctx, src_key_padding_mask=None if mask is None else ~mask.bool())
        h = self.norm(h)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None: logits = logits.masked_fill(~mask.bool(), -1.0e4)
        valid = mask.to(h.dtype).unsqueeze(-1) if mask is not None else torch.ones_like(h[..., :1])
        scene = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        c = F.normalize(self.cand_proj(h), dim=-1)
        s = F.normalize(self.scene_proj(scene), dim=-1).unsqueeze(-1)
        contrastive_logits = torch.matmul(c, s).squeeze(-1) / self.temperature
        if mask is not None: contrastive_logits = contrastive_logits.masked_fill(~mask.bool(), -1.0e4)
        out={'logits': logits, 'pluto_contrastive_logits': contrastive_logits}
        out.update(self.scalar_heads(h))
        return out


class PDMHybridAdapter(nn.Module):
    """Legacy learned PDM-Hybrid adapter kept only for old checkpoint compatibility.

    v54 main-table PDM-Hybrid never constructs this module: source semantics keep
    the executed trajectory identical to PDM-Closed before the 2.0 s correction
    horizon.
    """

    def __init__(self, input_dim: int, max_candidates: int = 32, d_model: int = 128, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.max_candidates=int(max_candidates)
        self.token=nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim,d_model), nn.GELU())
        self.traj=nn.Sequential(nn.Linear(4,d_model), nn.GELU())
        self.pos=nn.Parameter(torch.zeros(1,self.max_candidates,d_model))
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=int(num_heads),dim_feedforward=4*d_model,dropout=float(dropout),activation='gelu',batch_first=True,norm_first=True)
        self.encoder=_make_transformer_encoder(layer,int(num_layers))
        self.norm=nn.LayerNorm(d_model)
        self.policy_head=nn.Linear(d_model,1)
        self.scalar_heads=ScalarHeads(d_model)

    def forward(self,x:torch.Tensor,mask:torch.Tensor|None=None,prefix_traj:torch.Tensor|None=None,prefix_valid:torch.Tensor|None=None,**_:torch.Tensor)->dict[str,torch.Tensor]:
        B,N,_=x.shape
        mf=x.new_zeros(B,N,4)
        if prefix_traj is not None and prefix_traj.numel():
            start=prefix_traj[...,0,:]
            if prefix_valid is None: end=prefix_traj[...,-1,:]
            else:
                idx=prefix_valid.long().sum(dim=-1).clamp_min(1)-1
                end=torch.gather(prefix_traj,-2,idx[...,None,None].expand(B,N,1,2)).squeeze(-2)
            d=end-start; mf=torch.cat([d, torch.linalg.norm(d,dim=-1,keepdim=True), d[...,1:2].abs()],dim=-1)
        pos=self.pos[:,:N] if N<=self.pos.shape[1] else torch.cat([self.pos,self.pos[:,-1:].expand(1,N-self.pos.shape[1],-1)],dim=1)
        h=self.encoder(self.token(x)+self.traj(mf)+pos,src_key_padding_mask=None if mask is None else ~mask.bool())
        h=self.norm(h); logits=self.policy_head(h).squeeze(-1)
        if mask is not None: logits=logits.masked_fill(~mask.bool(),-1e4)
        out={'logits':logits,'pdm_refinement_logits':logits}
        out.update(self.scalar_heads(h)); return out


CandidateSetTransformer = WayformerRouteBC


def build_model_from_cfg(input_dim: int, cfg: dict[str, Any]) -> nn.Module:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    mcfg = bcfg.get("model", {}) if isinstance(bcfg.get("model", {}), dict) else {}
    baseline = str(bcfg.get("baseline", "route_bc_lite")).lower()
    arch = str(mcfg.get("arch", "")).lower()
    max_candidates = int(mcfg.get("max_candidates", bcfg.get("max_candidates", 32)))
    common = dict(
        input_dim=int(input_dim),
        max_candidates=max_candidates,
        d_model=int(mcfg.get("d_model", 256 if ("gameformer" in baseline or "betop" in baseline) else 192)),
        num_layers=int(mcfg.get("num_layers", 3)),
        num_heads=int(mcfg.get("num_heads", 4)),
        dropout=float(mcfg.get("dropout", 0.15)),
    )
    implementation = str(mcfg.get("implementation", bcfg.get("implementation", ""))).lower()
    if implementation in {"source_port", "source_port_v54", "sourceported_v54"}:
        source_common = dict(
            d_model=int(mcfg.get("d_model", 256 if "gameformer" in baseline else 128)),
            num_layers=int(mcfg.get("num_layers", 6 if "gameformer" in baseline else 4)),
            num_heads=int(mcfg.get("num_heads", 8 if baseline != "pluto" else 4)),
            dropout=float(mcfg.get("dropout", 0.10)),
            future_len=int(mcfg.get("future_len", 20)),
        )
        if "gameformer" in baseline or arch in {"gameformer", "gameformer_lite", "gameformer_levelk"}:
            return GameFormerSourcePort(
                **source_common,
                num_levels=int(mcfg.get("num_levels", 4)),
                modalities=int(mcfg.get("modalities", 6)),
                source_max_agents=int(mcfg.get("source_max_agents", int(mcfg.get("neighbors_to_predict", 8)) + 1)),
                projection_beta=float(mcfg.get("projection_beta", 2.0)),
            )
        if baseline in {"plantf", "plan_tf", "plantf_adapter"} or arch in {"plantf", "plan_tf", "plantf_adapter"}:
            return PlanTFSourcePort(
                **source_common,
                state_dropout=float(mcfg.get("state_dropout", 0.75)),
                num_modes=int(mcfg.get("num_modes", 6)),
                projection_beta=float(mcfg.get("projection_beta", 2.0)),
            )
        if baseline in {"pluto", "pluto_adapter"} or arch in {"pluto", "pluto_adapter"}:
            return PLUTOSourcePort(
                **source_common,
                state_dropout=float(mcfg.get("state_dropout", 0.75)),
                decoder_depth=int(mcfg.get("decoder_depth", 4)),
                num_modes=int(mcfg.get("num_modes", 12)),
            )
    if arch in {"gameformer", "gameformer_levelk", "levelk"} or "gameformer" in baseline:
        return GameFormerLevelK(
            **common,
            num_levels=int(mcfg.get("num_levels", 4)),
            modalities=int(mcfg.get("modalities", 6)),
            future_len=int(mcfg.get("future_len", 20)),
            history_len=int(mcfg.get("history_len", 11)),
            neighbors_to_predict=int(mcfg.get("neighbors_to_predict", 8)),
            root_feature_dim=int(mcfg.get("root_feature_dim", 18)),
            num_roots=int(mcfg.get("num_roots", cfg.get("num_roots", 10))),
            num_options=int(mcfg.get("num_options", cfg.get("num_recovery_options", 12))),
            use_teacher_branch_context=bool(mcfg.get("use_teacher_branch_context", False)),
            traj_step_scale=float(mcfg.get("traj_step_scale", 1.5)),
        )
    if arch in {"plantf", "plan_tf", "plantf_adapter"} or baseline in {"plantf", "plan_tf", "plantf_adapter"}:
        return PlanTFAdapter(
            **common,
            state_dropout=float(mcfg.get("state_dropout", 0.25)),
            actor_topology_feature_dim=int(mcfg.get("actor_topology_feature_dim", ACTOR_TOPO_FEATURE_DIM)),
            map_topology_feature_dim=int(mcfg.get("map_topology_feature_dim", MAP_TOPO_FEATURE_DIM)),
        )
    if arch in {"pluto", "pluto_adapter"} or baseline in {"pluto", "pluto_adapter"}:
        return PLUTOAdapter(
            **common,
            lateral_queries=int(mcfg.get("lateral_queries", 5)),
            longitudinal_queries=int(mcfg.get("longitudinal_queries", 5)),
            contrastive_temperature=float(mcfg.get("contrastive_temperature", 0.10)),
            actor_topology_feature_dim=int(mcfg.get("actor_topology_feature_dim", ACTOR_TOPO_FEATURE_DIM)),
            map_topology_feature_dim=int(mcfg.get("map_topology_feature_dim", MAP_TOPO_FEATURE_DIM)),
        )
    if arch in {"pdm_hybrid", "pdm_hybrid_adapter"} or baseline in {"pdm_hybrid", "pdm_hybrid_adapter"}:
        return PDMHybridAdapter(**common)
    if arch in {"betop", "betop_lite", "betopnet", "betopnet_lite"} or "betop" in baseline:
        return BeTopNetLite(
            **common,
            actor_topology_feature_dim=int(mcfg.get("actor_topology_feature_dim", mcfg.get("topology_feature_dim", ACTOR_TOPO_FEATURE_DIM))),
            map_topology_feature_dim=int(mcfg.get("map_topology_feature_dim", MAP_TOPO_FEATURE_DIM)),
            num_topology_agents=int(mcfg.get("num_topology_agents", cfg.get("max_agents", 16))),
            num_topology_map=int(mcfg.get("num_topology_map", 64)),
            num_topo=int(mcfg.get("num_topo", 16)),
            mlp_hidden=int(mcfg.get("mlp_hidden", 128)),
            mlp_layers=int(mcfg.get("mlp_layers", 3)),
        )
    return WayformerRouteBC(
        **common,
        mlp_hidden=int(mcfg.get("mlp_hidden", 128)),
        mlp_layers=int(mcfg.get("mlp_layers", 4)),
        num_latents=int(mcfg.get("num_latents", 96)),
        num_encoder_layers=int(mcfg.get("num_encoder_layers", 2)),
        num_decoder_layers=int(mcfg.get("num_decoder_layers", mcfg.get("num_layers", 4))),
        max_history_steps=int(mcfg.get("max_history_steps", mcfg.get("history_len", 11))),
        max_source_agents=int(mcfg.get("source_max_agents", 32)),
        future_len=int(mcfg.get("future_len", 20)),
        num_output_queries=int(mcfg.get("num_output_queries", 64)),
        num_mode_decoder_layers=int(mcfg.get("num_mode_decoder_layers", 2)),
    )
