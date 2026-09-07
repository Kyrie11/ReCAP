"""Dependency-light source ports for the four uploaded Safe-regime baseline packages.

These modules intentionally sit between an *official implementation* and a generic
paper-inspired adapter.  Their block structure, inference decision, and training
losses follow the uploaded public repositories, while nuPlan/WOMD preprocessing is
replaced by :mod:`ocrap.external_baselines.data` and final trajectories are projected
onto OC-RAP's common executable candidate lattice.

The ports are therefore not checkpoint-compatible with the authors' repositories.
That distinction is recorded in ``docs/EXTERNAL_BASELINE_AUDIT_V54_ZH.md``.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _make_encoder(d_model: int, heads: int, depth: int, dropout: float, *, norm_first: bool = True) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    try:
        return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=depth)


def _last_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Gather the last valid temporal state from ``[B,A,H,D]``."""
    B, A, H, D = x.shape
    idx = valid.long().sum(dim=-1).clamp_min(1) - 1
    return torch.gather(x, 2, idx[..., None, None].expand(B, A, 1, D)).squeeze(2)


def _candidate_ade(
    pred_xy: torch.Tensor,
    candidate_xy: torch.Tensor,
    candidate_valid: torch.Tensor | None,
) -> torch.Tensor:
    """ADE between predicted modes and executable candidates.

    ``pred_xy``: [B,M,T,2], candidates: [B,N,T,2] -> [B,N,M].
    """
    T = min(pred_xy.shape[-2], candidate_xy.shape[-2])
    pred = pred_xy[..., :T, :].float()
    cand = candidate_xy[..., :T, :].float()
    dist = torch.linalg.norm(cand[:, :, None] - pred[:, None], dim=-1)
    if candidate_valid is None:
        return dist.mean(dim=-1)
    valid = candidate_valid[..., :T].bool()[:, :, None, :]
    dist = torch.where(valid, dist, torch.zeros_like(dist))
    return dist.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)


def _project_modes_to_candidates(
    pred_xy: torch.Tensor,
    mode_logits: torch.Tensor,
    candidate_xy: torch.Tensor,
    candidate_valid: torch.Tensor | None,
    candidate_mask: torch.Tensor | None,
    beta: float,
) -> torch.Tensor:
    ade = _candidate_ade(pred_xy, candidate_xy, candidate_valid)
    score = torch.logsumexp(mode_logits.float()[:, None, :] - float(beta) * ade, dim=-1)
    if candidate_mask is not None:
        score = score.masked_fill(~candidate_mask.bool(), -1.0e4)
    return score


class SourcePointsEncoder(nn.Module):
    """PlanTF/PLUTO PointNet encoder with a safe empty-mask path.

    The layer layout mirrors the public ``PointsEncoder``. BatchNorm is retained
    because it is part of the source implementation; the empty-mask path avoids
    invalid gathers for map-less synthetic/unit-test scenes.
    """

    def __init__(self, feat_channel: int, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bs, n, _ = x.shape
        if mask is None:
            mask = torch.ones(bs, n, dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        if not bool(mask.any()):
            return x.new_zeros(bs, self.encoder_channel)
        x_features = x.new_zeros(bs, n, 256)
        vals = x[mask]
        # BatchNorm1d cannot estimate a variance from one sample while training.
        # Duplicating that one feature is mathematically neutral after pooling and
        # keeps the source BN layout intact for tiny unit-test batches.
        if self.training and vals.shape[0] == 1:
            enc = self.first_mlp(torch.cat([vals, vals], dim=0))[:1]
        else:
            enc = self.first_mlp(vals)
        x_features[mask] = enc
        pooled = x_features.max(dim=1).values
        cat = torch.cat([x_features, pooled[:, None].expand(-1, n, -1)], dim=-1)
        res = x.new_zeros(bs, n, self.encoder_channel)
        vals2 = cat[mask]
        if self.training and vals2.shape[0] == 1:
            enc2 = self.second_mlp(torch.cat([vals2, vals2], dim=0))[:1]
        else:
            enc2 = self.second_mlp(vals2)
        res[mask] = enc2
        return res.max(dim=1).values


class LocalTemporalFPN(nn.Module):
    """Dependency-free substitute for PlanTF/PLUTO's NATTEN temporal encoder.

    The source repository uses ``natten.NeighborhoodAttention1D`` plus a 3-level
    FPN.  NATTEN is a compiled optional dependency and is not present in the
    OC-RAP environment.  This port keeps the tokenizer/downsample/FPN topology but
    replaces neighborhood attention with depthwise-local residual convolutions.
    """

    def __init__(self, in_chans: int = 9, dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        base = max(16, dim // 4)
        dims = [base, base * 2, dim]
        self.token = nn.Conv1d(in_chans, dims[0], 3, padding=1)
        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, d in enumerate(dims):
            blocks = []
            for _ in range(2):
                blocks.append(nn.Sequential(
                    nn.Conv1d(d, d, 3, padding=1, groups=d),
                    nn.GELU(),
                    nn.Conv1d(d, d, 1),
                    nn.Dropout(dropout),
                ))
            self.stages.append(nn.ModuleList(blocks))
            if i < len(dims) - 1:
                self.down.append(nn.Conv1d(d, dims[i + 1], 3, stride=2, padding=1))
        self.lateral = nn.ModuleList([nn.Conv1d(d, dim, 3, padding=1) for d in dims])
        self.fpn = nn.Conv1d(dim, dim, 3, padding=1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, d) for d in dims])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,C]
        z = self.token(x.transpose(1, 2))
        outs: list[torch.Tensor] = []
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                z = z + block(self.norms[i](z))
            outs.append(z)
            if i < len(self.down):
                z = self.down[i](z)
        laterals = [layer(o) for layer, o in zip(self.lateral, outs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-1], mode="linear", align_corners=False
            )
        return self.fpn(laterals[0])[..., -1]


class SourceStateAttentionEncoder(nn.Module):
    """PlanTF/PLUTO state-token encoder, including the source 3-token keep rule."""

    def __init__(self, state_channel: int, dim: int, state_dropout: float = 0.75) -> None:
        super().__init__()
        self.state_channel = int(state_channel)
        self.state_dropout = float(state_dropout)
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(self.state_channel)])
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.pos_embed = nn.Parameter(torch.empty(1, self.state_channel, dim))
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([linear(x[:, i, None]) for i, linear in enumerate(self.linears)], dim=1)
        tokens = tokens + self.pos_embed
        key_padding_mask = None
        if self.training and self.state_dropout > 0 and self.state_channel > 3:
            visible = torch.zeros(tokens.shape[0], 3, dtype=torch.bool, device=x.device)
            dropped = torch.rand(tokens.shape[0], self.state_channel - 3, device=x.device) < self.state_dropout
            key_padding_mask = torch.cat([visible, dropped], dim=1)
        query = self.query.expand(tokens.shape[0], -1, -1)
        return self.attn(query, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)[0][:, 0]


class SourceAgentEncoder(nn.Module):
    """PlanTF/PLUTO actor encoder over OC-RAP's observable actor history."""

    def __init__(self, dim: int, state_channel: int = 6, state_dropout: float = 0.75, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.history = LocalTemporalFPN(9, dim, dropout=dropout)
        self.ego_state = SourceStateAttentionEncoder(state_channel, dim, state_dropout)
        self.type_emb = nn.Embedding(4, dim)

    @staticmethod
    def _vector_feature(history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # history [B,A,H,9] = x,y,vx,vy,heading,length,width,type,valid
        pair = valid[..., 1:] & valid[..., :-1]
        pos = history[..., :2]
        vel = history[..., 2:4]
        heading = history[..., 4]
        dpos = pos[..., 1:, :] - pos[..., :-1, :]
        dvel = vel[..., 1:, :] - vel[..., :-1, :]
        dh = heading[..., 1:] - heading[..., :-1]
        shape = history[..., 1:, 5:7]
        feat = torch.cat([
            dpos,
            dvel,
            torch.stack([dh.cos(), dh.sin()], dim=-1),
            shape,
            pair.float().unsqueeze(-1),
        ], dim=-1)
        return torch.where(pair[..., None], feat, torch.zeros_like(feat))

    def forward(self, history: torch.Tensor, valid: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        B, A, _, _ = history.shape
        feat = self._vector_feature(history.float(), valid.bool())
        flat = feat.reshape(B * A, feat.shape[-2], feat.shape[-1])
        actor_valid = valid.any(dim=-1).reshape(-1)
        enc = history.new_zeros(B * A, self.dim)
        if bool(actor_valid.any()):
            enc[actor_valid] = self.history(flat[actor_valid])
        enc = enc.view(B, A, self.dim)
        # Source PlanTF/PLUTO deliberately replace ego history with current-state
        # attention when use_ego_history=False (the public config default).
        enc[:, 0] = self.ego_state(current_state[:, :6].float())
        last = _last_valid(history, valid.bool())
        typ = last[..., 7].long().clamp(0, 3)
        return enc + self.type_emb(typ)


class SourceMapEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.points = SourcePointsEncoder(6, dim)
        self.speed = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.type_emb = nn.Embedding(3, dim)
        self.route_emb = nn.Embedding(2, dim)
        self.tl_emb = nn.Embedding(4, dim)
        self.unknown_speed = nn.Embedding(1, dim)

    def forward(self, points: torch.Tensor, point_valid: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        B, M, P, C = points.shape
        x = self.points(points.reshape(B * M, P, C).float(), point_valid.reshape(B * M, P).bool()).view(B, M, -1)
        kind = meta[..., 0].long().clamp(0, 2)
        on_route = (meta[..., 1] > 0.5).long()
        tl = meta[..., 2].long().clamp(0, 3)
        speed = meta[..., 3].float()
        has_speed = torch.isfinite(speed) & (speed > 0)
        speed_emb = x.new_zeros(B, M, self.dim)
        if bool(has_speed.any()):
            speed_emb[has_speed] = self.speed(speed[has_speed].unsqueeze(-1))
        if bool((~has_speed).any()):
            speed_emb[~has_speed] = self.unknown_speed.weight[0]
        return x + self.type_emb(kind) + self.route_emb(on_route) + self.tl_emb(tl) + speed_emb


class SourceSceneEncoder(nn.Module):
    """Shared source-style actor/map scene encoder for PlanTF and PLUTO."""

    def __init__(self, dim: int, heads: int, depth: int, dropout: float, state_dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.agent = SourceAgentEncoder(dim, state_channel=6, state_dropout=state_dropout, dropout=dropout)
        self.map = SourceMapEncoder(dim)
        self.pos_emb = nn.Sequential(nn.Linear(4, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.encoder = _make_encoder(dim, heads, depth, dropout, norm_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        agent_history: torch.Tensor,
        agent_valid: torch.Tensor,
        current_state: torch.Tensor,
        map_points: torch.Tensor,
        map_point_valid: torch.Tensor,
        map_meta: torch.Tensor,
        map_center: torch.Tensor,
        map_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        agent_history = agent_history.float()
        agent_valid = agent_valid.bool()
        map_valid = map_valid.bool()
        x_agent = self.agent(agent_history, agent_valid, current_state)
        x_map = self.map(map_points, map_point_valid, map_meta)
        last = _last_valid(agent_history, agent_valid)
        apos = last[..., :2]
        ah = last[..., 4]
        mpos = map_center[..., :2].float()
        mh = map_center[..., 2].float()
        pos = torch.cat([
            torch.cat([apos, ah.cos()[..., None], ah.sin()[..., None]], dim=-1),
            torch.cat([mpos, mh.cos()[..., None], mh.sin()[..., None]], dim=-1),
        ], dim=1)
        x = torch.cat([x_agent, x_map], dim=1) + self.pos_emb(pos)
        key_padding = torch.cat([~agent_valid.any(dim=-1), ~map_valid], dim=1)
        # MultiheadAttention fails for an all-masked row. Ego is always a legal
        # neutral token even in synthetic tests with no scene history.
        empty = key_padding.all(dim=1)
        if bool(empty.any()):
            key_padding = key_padding.clone()
            key_padding[empty, 0] = False
        return self.norm(self.encoder(x, src_key_padding_mask=key_padding)), key_padding


class SourceTrajectoryDecoder(nn.Module):
    """PlanTF multimodal ego trajectory decoder."""

    def __init__(self, dim: int, num_modes: int, future_steps: int, out_channels: int = 4) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.future_steps = int(future_steps)
        self.out_channels = int(out_channels)
        self.multimodal_proj = nn.Linear(dim, self.num_modes * dim)
        hidden = 2 * dim
        self.loc = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, self.future_steps * self.out_channels))
        self.pi = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.multimodal_proj(x).view(-1, self.num_modes, x.shape[-1])
        loc = self.loc(z).view(-1, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(z).squeeze(-1)
        return loc, pi


class PlanTFSourcePort(nn.Module):
    """Source-derived PlanTF port with trajectory-to-candidate projection."""

    def __init__(self, d_model: int = 128, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, state_dropout: float = 0.75, num_modes: int = 6, future_len: int = 20, projection_beta: float = 2.0, **_: object) -> None:
        super().__init__()
        self.future_len = int(future_len)
        self.projection_beta = float(projection_beta)
        self.scene = SourceSceneEncoder(d_model, num_heads, num_layers, dropout, state_dropout)
        self.decoder = SourceTrajectoryDecoder(d_model, num_modes, self.future_len, out_channels=4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, prefix_traj: torch.Tensor | None = None, prefix_valid: torch.Tensor | None = None, **kw: torch.Tensor) -> dict[str, torch.Tensor]:
        required = ["source_agent_history", "source_agent_valid", "source_current_state", "source_map_points", "source_map_point_valid", "source_map_meta", "source_map_center", "source_map_valid"]
        if any(kw.get(k) is None for k in required):
            raise ValueError("PlanTF source port requires source_* scene tensors from ExternalGroupDataset")
        enc, _ = self.scene(*(kw[k] for k in required))
        traj, pi = self.decoder(enc[:, 0])
        if prefix_traj is None:
            raise ValueError("PlanTF source port requires executable prefix_traj for common-lattice projection")
        logits = _project_modes_to_candidates(traj[..., :2], pi, prefix_traj, prefix_valid, mask, self.projection_beta)
        return {"logits": logits, "plantf_trajectory": traj, "plantf_probability": pi}


class FourierEmbedding3(nn.Module):
    """Small Fourier embedding used for PLUTO reference-line poses."""

    def __init__(self, dim: int, bands: int = 16) -> None:
        super().__init__()
        self.bands = int(bands)
        self.proj = nn.Sequential(nn.Linear(3 * (2 * self.bands + 1), dim), nn.ReLU(), nn.Linear(dim, dim))
        freq = 2.0 ** torch.arange(self.bands, dtype=torch.float32)
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x[..., None] * self.freq
        feat = torch.cat([x, torch.sin(xx).flatten(-2), torch.cos(xx).flatten(-2)], dim=-1)
        return self.proj(feat)


class PLUTODecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.r2r = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.m2m = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim); self.norm3 = nn.LayerNorm(dim); self.norm4 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, ref_padding: torch.Tensor, mem_padding: torch.Tensor, m_pos: torch.Tensor) -> torch.Tensor:
        B, R, M, D = tgt.shape
        # reference-to-reference attention for each mode
        z = tgt.transpose(1, 2).reshape(B * M, R, D)
        z2 = self.norm1(z)
        z = z + self.drop(self.r2r(z2, z2, z2, key_padding_mask=ref_padding[:, None].expand(B, M, R).reshape(B * M, R), need_weights=False)[0])
        # mode-to-mode attention for each valid reference.  Do the full tensor
        # and neutralize padded references; this avoids Python/indexing overhead.
        z = z.reshape(B, M, R, D).transpose(1, 2).reshape(B * R, M, D)
        z2 = self.norm2(z)
        mp = m_pos.expand(B * R, -1, -1)
        z = z + self.drop(self.m2m(z2 + mp, z2 + mp, z2, need_weights=False)[0])
        valid_ref = (~ref_padding).reshape(B * R, 1, 1)
        z = torch.where(valid_ref, z, torch.zeros_like(z))
        z = z.reshape(B, R * M, D)
        z2 = self.norm3(z)
        z = z + self.drop(self.cross(z2, memory, memory, key_padding_mask=mem_padding, need_weights=False)[0])
        z = z + self.drop(self.ffn(self.norm4(z)))
        return z.reshape(B, R, M, D)


class PLUTOPlanningDecoder(nn.Module):
    """Source-shaped PLUTO reference-line x mode decoder."""

    def __init__(self, num_modes: int, depth: int, dim: int, heads: int, dropout: float, future_steps: int) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.future_steps = int(future_steps)
        self.blocks = nn.ModuleList([PLUTODecoderLayer(dim, heads, dropout) for _ in range(depth)])
        self.r_pos_emb = FourierEmbedding3(dim)
        self.r_encoder = SourcePointsEncoder(6, dim)
        self.q_proj = nn.Linear(2 * dim, dim)
        self.m_emb = nn.Parameter(torch.empty(1, 1, self.num_modes, dim))
        self.m_pos = nn.Parameter(torch.empty(1, self.num_modes, dim))
        self.loc_head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, self.future_steps * 2))
        self.yaw_head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, self.future_steps * 2))
        self.vel_head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, self.future_steps * 2))
        self.pi_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        nn.init.normal_(self.m_emb, std=0.01); nn.init.normal_(self.m_pos, std=0.01)

    @staticmethod
    def _references(prefix: torch.Tensor, valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # prefix [B,R,T,2] becomes PLUTO's reference-line features.
        B, R, T, _ = prefix.shape
        if valid is None:
            valid = torch.ones(B, R, T, dtype=torch.bool, device=prefix.device)
        else:
            valid = valid.bool()
        origin = prefix[..., :1, :]
        rel = prefix - origin
        prev = torch.cat([prefix[..., :1, :], prefix[..., :-1, :]], dim=-2)
        vec = prefix - prev
        theta = torch.atan2(vec[..., 1], vec[..., 0].clamp(min=1.0e-3))
        feat = torch.cat([rel, vec, theta.cos()[..., None], theta.sin()[..., None]], dim=-1)
        pose = torch.cat([prefix[..., 0, :], theta[..., 0, None]], dim=-1)
        return feat, valid, pose

    def forward(self, prefix: torch.Tensor, prefix_valid: torch.Tensor | None, memory: torch.Tensor, mem_padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat, valid, pose = self._references(prefix.float(), prefix_valid)
        B, R, T, C = feat.shape
        r_emb = self.r_encoder(feat.reshape(B * R, T, C), valid.reshape(B * R, T)).view(B, R, -1)
        r_emb = r_emb + self.r_pos_emb(pose)
        ref_padding = ~valid.any(dim=-1)
        q = self.q_proj(torch.cat([
            r_emb[:, :, None].expand(-1, -1, self.num_modes, -1),
            self.m_emb.expand(B, R, -1, -1),
        ], dim=-1))
        for blk in self.blocks:
            q = blk(q, memory, ref_padding, mem_padding, self.m_pos)
        loc = self.loc_head(q).view(B, R, self.num_modes, self.future_steps, 2)
        yaw = self.yaw_head(q).view(B, R, self.num_modes, self.future_steps, 2)
        vel = self.vel_head(q).view(B, R, self.num_modes, self.future_steps, 2)
        traj = torch.cat([loc, yaw, vel], dim=-1)
        pi = self.pi_head(q).squeeze(-1).masked_fill(ref_padding[..., None], -1.0e4)
        return traj, pi


class PLUTOSourcePort(nn.Module):
    """PLUTO source port: scene encoder + reference-line conditioned decoder."""

    def __init__(self, d_model: int = 128, num_layers: int = 4, num_heads: int = 4, dropout: float = 0.1, state_dropout: float = 0.75, decoder_depth: int = 4, num_modes: int = 12, future_len: int = 20, **_: object) -> None:
        super().__init__()
        self.scene = SourceSceneEncoder(d_model, num_heads, num_layers, dropout, state_dropout)
        self.decoder = PLUTOPlanningDecoder(num_modes, decoder_depth, d_model, num_heads, dropout, future_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, prefix_traj: torch.Tensor | None = None, prefix_valid: torch.Tensor | None = None, **kw: torch.Tensor) -> dict[str, torch.Tensor]:
        required = ["source_agent_history", "source_agent_valid", "source_current_state", "source_map_points", "source_map_point_valid", "source_map_meta", "source_map_center", "source_map_valid"]
        if any(kw.get(k) is None for k in required):
            raise ValueError("PLUTO source port requires source_* scene tensors from ExternalGroupDataset")
        if prefix_traj is None:
            raise ValueError("PLUTO source port requires executable prefix_traj as reference-line candidates")
        enc, key_padding = self.scene(*(kw[k] for k in required))
        traj, pi = self.decoder(prefix_traj, prefix_valid, enc, key_padding)
        logits = pi.max(dim=-1).values
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        return {"logits": logits, "pluto_trajectory": traj, "pluto_probability": pi}


class GameFormerAgentEncoder(nn.Module):
    """GameFormer AgentEncoder with the public 8-state + type structure."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.motion = nn.LSTM(8, dim, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, dim, padding_idx=0)

    @staticmethod
    def _state(history: torch.Tensor) -> torch.Tensor:
        # source order: x,y,heading,vx,vy,width,length,height ; type separate
        h = history.float()
        height = torch.full_like(h[..., :1], 1.5)
        return torch.cat([h[..., :2], h[..., 4:5], h[..., 2:4], h[..., 6:7], h[..., 5:6], height], dim=-1)

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        state = self._state(history)
        state = torch.where(valid[..., None].bool(), state, torch.zeros_like(state))
        # The public Encoder calls AgentEncoder independently for each actor.
        # Flatten B*A only around the LSTM to reproduce that computation without
        # a Python loop; nn.LSTM itself accepts [batch,time,channel], not 4-D.
        B, A, H, C = state.shape
        out, _ = self.motion(state.reshape(B * A, H, C))
        out = out.view(B, A, H, self.motion.hidden_size)
        last = _last_valid(history, valid.bool())
        typ = last[..., 7].long().clamp(0, 3)
        idx = valid.long().sum(dim=-1).clamp_min(1) - 1
        D = out.shape[-1]
        motion = torch.gather(out, 2, idx[..., None, None].expand(B, A, 1, D)).squeeze(2)
        return motion + self.type_emb(typ)


class GameFormerMapEncoder(nn.Module):
    """WOMD-vector-map bridge to GameFormer's 256-d scene-token interface."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.points = SourcePointsEncoder(6, dim)
        self.meta = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, points: torch.Tensor, valid: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        B, M, P, C = points.shape
        p = self.points(points.reshape(B * M, P, C).float(), valid.reshape(B * M, P).bool()).view(B, M, -1)
        # Scale speed before concatenating with categorical meta to avoid a single
        # high-magnitude channel dominating the source bridge.
        m = meta.float().clone()
        m[..., 3] = m[..., 3] / 20.0
        return p + self.meta(m)


class GameFormerFutureEncoderSource(nn.Module):
    """Public GameFormer FutureEncoder, including the crucial detach()."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, dim))
        self.type_emb = nn.Embedding(4, dim, padding_idx=0)

    def forward(self, traj_xy: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
        # traj [B,A,M,T,2], current [B,A,9] source GameFormer order.
        B, A, M, T, _ = traj_xy.shape
        cur = current_states[:, :, None].expand(-1, -1, M, -1)
        xy = torch.cat([cur[:, :, :, None, :2], traj_xy.float()], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1.0e-3))[..., None]
        size = cur[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        state = torch.cat([traj_xy.float(), theta, v, size], dim=-1)
        # This detach is present in the uploaded GameFormer source.  Omitting it
        # changes the level-k training graph and is therefore a fidelity bug.
        z = self.mlp(state.detach())
        typ = self.type_emb(cur[..., 8].long().clamp(0, 3))
        return z.max(dim=-2).values + typ


class GameFormerGMMPredictor(nn.Module):
    def __init__(self, dim: int, future_len: int) -> None:
        super().__init__()
        self.future_len = int(future_len)
        self.gaussian = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ELU(), nn.Dropout(0.1), nn.Linear(2 * dim, self.future_len * 4))
        self.score = nn.Sequential(nn.Linear(dim, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, D = q.shape
        return self.gaussian(q).view(B, M, self.future_len, 4), self.score(q).squeeze(-1)


class GameFormerSourcePort(nn.Module):
    """Source-derived GameFormer joint multi-agent level-k decoder.

    Source map tokens are shared across actors because OC-RAP stores a global
    ego-centric WOMD vector map rather than GameFormer's actor-indexed lane tensor.
    The decoder itself predicts *all observed actors* at every level and then
    projects only the ego modes to the common executable candidate lattice.
    """

    def __init__(self, d_model: int = 256, num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1, num_levels: int = 4, modalities: int = 6, future_len: int = 20, source_max_agents: int = 9, projection_beta: float = 2.0, **_: object) -> None:
        super().__init__()
        self.dim = int(d_model); self.levels = int(num_levels); self.modalities = int(modalities); self.future_len = int(future_len); self.max_agents = int(source_max_agents); self.projection_beta = float(projection_beta)
        self.agent_encoder = GameFormerAgentEncoder(self.dim)
        self.map_encoder = GameFormerMapEncoder(self.dim)
        # Uploaded GameFormer uses post-norm TransformerEncoderLayer.
        self.fusion = _make_encoder(self.dim, num_heads, num_layers, dropout, norm_first=False)
        self.modal_emb = nn.Embedding(self.modalities, self.dim)
        self.agent_emb = nn.Embedding(max(self.max_agents, 1), self.dim)
        self.init_cross = nn.MultiheadAttention(self.dim, num_heads, dropout=dropout, batch_first=True)
        self.init_norm1 = nn.LayerNorm(self.dim); self.init_norm2 = nn.LayerNorm(self.dim)
        self.init_ffn = nn.Sequential(nn.Linear(self.dim, 4 * self.dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * self.dim, self.dim))
        self.future_encoder = GameFormerFutureEncoderSource(self.dim)
        self.interaction = nn.ModuleList([_make_encoder(self.dim, num_heads, 1, dropout, norm_first=False) for _ in range(self.levels)])
        self.level_cross = nn.ModuleList([nn.MultiheadAttention(self.dim, num_heads, dropout=dropout, batch_first=True) for _ in range(self.levels)])
        self.level_norm1 = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(self.levels)])
        self.level_norm2 = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(self.levels)])
        self.level_ffn = nn.ModuleList([nn.Sequential(nn.Linear(self.dim, 4 * self.dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * self.dim, self.dim)) for _ in range(self.levels)])
        self.predictors = nn.ModuleList([GameFormerGMMPredictor(self.dim, self.future_len) for _ in range(self.levels + 1)])

    @staticmethod
    def _gf_current(history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        last = _last_valid(history.float(), valid.bool())
        height = torch.full_like(last[..., :1], 1.5)
        return torch.cat([last[..., :2], last[..., 4:5], last[..., 2:4], last[..., 6:7], last[..., 5:6], height, last[..., 7:8]], dim=-1)

    def _cross_block(self, q: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor, attn: nn.MultiheadAttention, n1: nn.LayerNorm, n2: nn.LayerNorm, ffn: nn.Module) -> torch.Tensor:
        z = attn(q, mem, mem, key_padding_mask=mem_mask, need_weights=False)[0]
        z = n1(z)
        return n2(z + ffn(z))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, prefix_traj: torch.Tensor | None = None, prefix_valid: torch.Tensor | None = None, **kw: torch.Tensor) -> dict[str, torch.Tensor]:
        hist = kw.get("source_agent_history"); valid = kw.get("source_agent_valid"); mp = kw.get("source_map_points"); mpv = kw.get("source_map_point_valid"); meta = kw.get("source_map_meta"); mv = kw.get("source_map_valid")
        if any(v is None for v in (hist, valid, mp, mpv, meta, mv)):
            raise ValueError("GameFormer source port requires source_* scene tensors")
        if prefix_traj is None:
            raise ValueError("GameFormer source port requires executable prefix_traj")
        hist = hist.float(); valid = valid.bool(); mv = mv.bool()
        B, A, _, _ = hist.shape
        if A > self.agent_emb.num_embeddings:
            raise ValueError(f"source_max_agents={self.agent_emb.num_embeddings} smaller than provided A={A}")
        actors = self.agent_encoder(hist, valid)
        maps = self.map_encoder(mp, mpv, meta)
        scene = torch.cat([actors, maps], dim=1)
        scene_mask = torch.cat([~valid.any(dim=-1), ~mv], dim=1)
        empty = scene_mask.all(dim=1)
        if bool(empty.any()):
            scene_mask = scene_mask.clone(); scene_mask[empty, 0] = False
        enc = self.fusion(scene, src_key_padding_mask=scene_mask)
        current = self._gf_current(hist, valid)
        modal = self.modal_emb(torch.arange(self.modalities, device=x.device))
        contents: list[torch.Tensor] = []
        trajs: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        # Source InitialDecoder is actor-indexed; loop count is small (<=9) and
        # avoids materializing B*A copies of the large scene memory.
        for a in range(A):
            q = enc[:, a:a+1] + modal[None] + self.agent_emb.weight[a][None, None]
            q = self._cross_block(q, enc, scene_mask, self.init_cross, self.init_norm1, self.init_norm2, self.init_ffn)
            tr, sc = self.predictors[0](q)
            tr = tr.clone(); tr[..., :2] = tr[..., :2] + current[:, a, None, None, :2]
            contents.append(q); trajs.append(tr); scores.append(sc)
        content = torch.stack(contents, dim=1)
        traj = torch.stack(trajs, dim=1)
        score = torch.stack(scores, dim=1)
        ego_trajs = [traj[:, 0]]; ego_scores = [score[:, 0]]
        level_logits = [_project_modes_to_candidates(traj[:, 0, ..., :2], score[:, 0], prefix_traj, prefix_valid, mask, self.projection_beta)]
        actor_padding = ~valid.any(dim=-1)
        for k in range(self.levels):
            multi = self.future_encoder(traj[..., :2], current)
            expected = (multi * score.float().softmax(dim=-1)[..., None]).mean(dim=2)
            interaction = self.interaction[k](expected, src_key_padding_mask=actor_padding)
            next_content=[]; next_traj=[]; next_score=[]
            mem_base = torch.cat([interaction, enc], dim=1)
            for a in range(A):
                mem_mask = torch.cat([actor_padding, scene_mask], dim=1).clone()
                mem_mask[:, a] = True  # source masks the current actor's own previous future
                q = content[:, a] + multi[:, a]
                q2 = self._cross_block(q, mem_base, mem_mask, self.level_cross[k], self.level_norm1[k], self.level_norm2[k], self.level_ffn[k])
                tr, sc = self.predictors[k + 1](q2)
                tr = tr.clone(); tr[..., :2] = tr[..., :2] + current[:, a, None, None, :2]
                next_content.append(q2); next_traj.append(tr); next_score.append(sc)
            content = torch.stack(next_content, dim=1); traj = torch.stack(next_traj, dim=1); score = torch.stack(next_score, dim=1)
            ego_trajs.append(traj[:, 0]); ego_scores.append(score[:, 0])
            level_logits.append(_project_modes_to_candidates(traj[:, 0, ..., :2], score[:, 0], prefix_traj, prefix_valid, mask, self.projection_beta))
        return {
            "logits": level_logits[-1],
            "level_logits": level_logits,
            "gameformer_ego_level_trajs": ego_trajs,
            "gameformer_ego_level_scores": ego_scores,
        }
