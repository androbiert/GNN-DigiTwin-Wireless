"""
student_film.py — WirelessNet-Fermi FiLM & Highway: Innovative Student GNN

A lightweight, physics-inspired student architecture for knowledge distillation
from WirelessNet-Fermi v3 (teacher). Instead of shrinking the teacher's MHA-heavy
design, this model re-engineers the information flow using three novel pillars:

  1. FiLM Modulation   — Link physical states (SINR, speed) generate per-element
                         scale (γ) and shift (β) coefficients that modulate Queue
                         representations.  Replaces LinkToQueueAttention.
                         Cost: O(D) vs O(H·Q·L·D).

  2. Lightweight Gated  — Flow→Queue aggregation uses scatter-mean + a learnable
     Aggregation          gate instead of full multi-head cross-attention.
                         Cost: O(F·D) vs O(F·H·D²).

  3. Dense Multi-Scale  — All K iteration states are retained and concatenated
     Skip-Highways        into a fused readout, so a shallow K=3 loop captures
                         both local and global graph structure.

Usage:
    from wireless_gnn.student_film import WirelessNetFermiStudent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from wireless_gnn.model import (
    FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM,
    scatter_sum, scatter_mean, grouped_softmax,
)


# --------------------------------------------------------------------------- #
# Building Block 1: FiLM Modulation
# --------------------------------------------------------------------------- #

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    A modulator node (e.g. Link) generates per-dimension scale (γ) and
    shift (β) parameters that are applied element-wise to a target node
    (e.g. Queue).  This mirrors the physics of wireless channels where
    SINR and distance *multiplicatively* attenuate capacity.

        FiLM(h_target | h_mod) = LayerNorm(γ ⊙ h_target + β)

    Replaces heavy multi-head cross-attention with O(D) element-wise ops.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        target_state:    torch.Tensor,   # [N_target, D]
        modulator_state: torch.Tensor,   # [N_mod, D]
        mapping:         torch.Tensor,   # [N_target] or [N_mod] → index
        mode:            str = "target_indexed",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mode : str
            "target_indexed" — mapping[i] gives the modulator index for
                               target node i.  len(mapping) == N_target.
            "modulator_indexed" — mapping[j] gives the target index for
                                  modulator j.  We scatter-mean the FiLM
                                  params to target nodes first.
        """
        if mode == "target_indexed":
            # Each target node has exactly one modulator (e.g. Queue→Link)
            mapped_mod = modulator_state[mapping]          # [N_target, D]
        else:
            # Multiple modulators map to one target (e.g. Link→Queue via l2q)
            # Scatter-mean the modulator states to target nodes
            mapped_mod = scatter_mean(modulator_state, mapping,
                                      target_state.size(0))  # [N_target, D]

        params = self.film_gen(mapped_mod)                 # [N_target, 2D]
        gamma, beta = torch.chunk(params, 2, dim=-1)       # [N_target, D] each

        # Affine modulation + residual + normalisation
        modulated = gamma * target_state + beta
        return self.norm(modulated + target_state)


# --------------------------------------------------------------------------- #
# Building Block 2: Lightweight Gated Aggregation (replaces MHA)
# --------------------------------------------------------------------------- #

class GatedScatterAggregation(nn.Module):
    """
    Lightweight alternative to FlowToQueueAttention.

    Instead of full multi-head attention (Q/K/V projections + grouped softmax),
    this module:
      1. Computes a per-flow relevance score via a small MLP.
      2. Uses grouped_softmax to normalise scores within each queue.
      3. Aggregates weighted flow values via scatter_sum.

    Parameters: 3·D² (vs 4·D² for MHA) — and no head splitting overhead.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        source_state: torch.Tensor,   # [N_src, D]  (e.g. flows)
        target_state: torch.Tensor,   # [N_tgt, D]  (e.g. queues)
        src_to_tgt:   torch.Tensor,   # [N_src] long — src i → tgt index
    ) -> torch.Tensor:                 # [N_tgt, D]
        n_src = source_state.size(0)
        n_tgt = target_state.size(0)

        # Pair each source with its target's state
        tgt_per_src = target_state[src_to_tgt]             # [N_src, D]
        pair = torch.cat([source_state, tgt_per_src], -1)  # [N_src, 2D]

        # Relevance scores → grouped softmax over each target group
        raw_scores = self.score_net(pair)                  # [N_src, 1]
        alpha = grouped_softmax(raw_scores, src_to_tgt, n_tgt)  # [N_src, 1]

        # Weighted values
        values = self.value_proj(source_state)             # [N_src, D]
        weighted = alpha * values                          # [N_src, D]
        return scatter_sum(weighted, src_to_tgt, n_tgt)    # [N_tgt, D]


# --------------------------------------------------------------------------- #
# Building Block 3: Compact FFN with Residual
# --------------------------------------------------------------------------- #

class CompactFFN(nn.Module):
    """
    Lightweight Feed-Forward block.  Expansion factor 1.5× (vs teacher's 2×)
    to save parameters while still providing non-linear per-node capacity.
    """

    def __init__(self, hidden_dim: int, expansion: float = 1.5,
                 dropout: float = 0.1):
        super().__init__()
        mid = int(hidden_dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# --------------------------------------------------------------------------- #
# Building Block 4: Sparse Flow Activity Gate
# --------------------------------------------------------------------------- #

class SparseActivityGate(nn.Module):
    """
    Learns a per-flow activity score from raw flow features and gates the
    hidden state.  Inactive flows (low offered load, zero delay) get their
    updates dampened, saving effective compute in downstream aggregation.

        gate_i = σ(MLP(raw_feat_i))
        h_i'   = gate_i ⊙ h_i
    """

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, raw_feat: torch.Tensor,
                hidden: torch.Tensor) -> torch.Tensor:
        g = self.gate(raw_feat)   # [F, D]
        return g * hidden         # [F, D]


# --------------------------------------------------------------------------- #
# Building Block 5: Lightweight Cross-Flow (within-queue) Mixing
# --------------------------------------------------------------------------- #

class CrossFlowMixing(nn.Module):
    """
    Lightweight alternative to full CrossFlowAttention.
    Uses scatter-mean within each queue group + a learnable gate,
    avoiding the O(F²) pairwise attention cost.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, flow_state: torch.Tensor,
                flow_to_queue: torch.Tensor) -> torch.Tensor:
        n_f = flow_state.size(0)
        n_q = int(flow_to_queue.max().item()) + 1 if n_f > 0 else 0

        # Queue-level mean of co-located flows
        q_mean = scatter_mean(flow_state, flow_to_queue, n_q)  # [Q, D]
        q_per_flow = q_mean[flow_to_queue]                     # [F, D]

        pair = torch.cat([flow_state, q_per_flow], dim=-1)     # [F, 2D]
        mixed = self.mix(pair)                                 # [F, D]
        g = torch.sigmoid(self.gate(pair))                     # [F, D]
        out = g * mixed + (1.0 - g) * flow_state
        return self.norm(out)


# --------------------------------------------------------------------------- #
# Main Student Model
# --------------------------------------------------------------------------- #

class WirelessNetFermiStudent(nn.Module):
    """
    WirelessNet-Fermi FiLM & Highway — Innovative Student GNN.

    A radically different architecture from the teacher that is specifically
    designed for efficient knowledge distillation and low-latency inference
    on wireless digital twin graphs.

    Architecture Summary
    --------------------
    ┌─────────────────────────────────────────────────────────┐
    │  Embedding:  2-layer MLPs (GELU) → D-dim               │
    │  Activity Gate:  raw flow features → sigmoid gate       │
    ├─────────────────────────────────────────────────────────┤
    │  Message Passing (×K, default 3):                       │
    │    ① F→Q  GatedScatterAggregation + GRU + LN            │
    │    ② L→L  Lightweight self-mixing (scatter + gate)      │
    │    ③ L→Q  FiLM Modulation + GRU + LN                    │
    │    ④ Q→L  FiLM Modulation (reverse direction)           │
    │    ⑤ Q→F  Gated message + GRU + LN + FFN                │
    │    → collect flow_state snapshot for highway             │
    ├─────────────────────────────────────────────────────────┤
    │  Multi-Scale Highway Fusion:                            │
    │    concat [h_F^(1) ‖ h_F^(2) ‖ … ‖ h_F^(K)] → Linear  │
    ├─────────────────────────────────────────────────────────┤
    │  Cross-Flow Mixing (within-queue scatter + gate)        │
    │  Readout MLP → scalar prediction                        │
    └─────────────────────────────────────────────────────────┘

    Parameters
    ----------
    hidden_dim   : int   embedding dimension D  (default 32)
    iterations   : int   message-passing rounds K  (default 3)
    dropout      : float dropout rate  (default 0.1)
    target       : str   'delay' or 'throughput'
    """

    def __init__(
        self,
        hidden_dim:   int   = 32,
        iterations:   int   = 3,
        dropout:      float = 0.1,
        target:       str   = 'throughput',
    ):
        assert target in ('delay', 'throughput'), \
            f"target must be 'delay' or 'throughput', got '{target}'"
        super().__init__()
        self.target     = target
        self.hidden_dim = hidden_dim
        self.iterations = iterations

        # ------------------------------------------------------------------ #
        # 1. Embeddings  (GELU, 2-layer)
        # ------------------------------------------------------------------ #
        self.flow_embedding = nn.Sequential(
            nn.Linear(FLOW_FEAT_DIM, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.GELU(),
        )
        self.queue_embedding = nn.Sequential(
            nn.Linear(QUEUE_FEAT_DIM, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),     nn.GELU(),
        )
        self.link_embedding = nn.Sequential(
            nn.Linear(LINK_FEAT_DIM, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.GELU(),
        )

        # ------------------------------------------------------------------ #
        # 2. Sparse Activity Gate  (prune inactive flows early)
        # ------------------------------------------------------------------ #
        self.activity_gate = SparseActivityGate(FLOW_FEAT_DIM, hidden_dim)

        # ------------------------------------------------------------------ #
        # 3. Message-Passing Modules
        # ------------------------------------------------------------------ #
        # Step ① — F → Q  (gated scatter aggregation)
        self.f2q_agg  = GatedScatterAggregation(hidden_dim, dropout)
        self.fq_gru   = nn.GRUCell(hidden_dim, hidden_dim)
        self.fq_norm  = nn.LayerNorm(hidden_dim)

        # Step ② — L → L  (lightweight self-mixing via scatter + gate)
        self.ll_mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ll_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ll_norm = nn.LayerNorm(hidden_dim)

        # Step ③ — L → Q  (FiLM modulation — the core innovation)
        self.lq_film = FiLMBlock(hidden_dim, dropout)
        self.lq_gru  = nn.GRUCell(hidden_dim, hidden_dim)
        self.lq_norm = nn.LayerNorm(hidden_dim)

        # Step ④ — Q → L  (FiLM modulation, reverse direction)
        self.ql_film = FiLMBlock(hidden_dim, dropout)
        self.ql_norm = nn.LayerNorm(hidden_dim)

        # Step ⑤ — Q → F  (gated message, same as teacher concept)
        self.qf_gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.qf_val_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.qf_gru       = nn.GRUCell(hidden_dim, hidden_dim)
        self.qf_norm      = nn.LayerNorm(hidden_dim)
        self.qf_ffn       = CompactFFN(hidden_dim, expansion=1.5, dropout=dropout)

        # ------------------------------------------------------------------ #
        # 4. Multi-Scale Dense Highway Fusion
        # ------------------------------------------------------------------ #
        self.highway_fuse = nn.Sequential(
            nn.Linear(hidden_dim * iterations, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.highway_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------ #
        # 5. Cross-Flow Mixing  (lightweight within-queue)
        # ------------------------------------------------------------------ #
        self.cross_flow = CrossFlowMixing(hidden_dim, dropout)

        # ------------------------------------------------------------------ #
        # 6. Readout MLP  (GELU activations)
        # ------------------------------------------------------------------ #
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ---------------------------------------------------------------------- #
    # Link Self-Mixing  (L → L)
    # ---------------------------------------------------------------------- #
    def _link_self_mix(self, link_state: torch.Tensor) -> torch.Tensor:
        """
        Lightweight alternative to full self-attention over links.
        Computes a global mean context and mixes it with each link via a gate.
        Cost: O(L·D) instead of O(L²·D).
        """
        n_l = link_state.size(0)
        if n_l <= 1:
            return link_state

        global_ctx = link_state.mean(dim=0, keepdim=True).expand(n_l, -1)  # [L, D]
        pair = torch.cat([link_state, global_ctx], dim=-1)                 # [L, 2D]
        mixed = self.ll_mix(pair)                                          # [L, D]
        gate = torch.sigmoid(self.ll_gate(pair))                           # [L, D]
        out = gate * mixed + (1.0 - gate) * link_state
        return self.ll_norm(out)

    # ---------------------------------------------------------------------- #
    # Q → F  Gated Message
    # ---------------------------------------------------------------------- #
    def _queue_to_flow_msg(
        self,
        flow_state:    torch.Tensor,   # [F, D]
        queue_state:   torch.Tensor,   # [Q, D]
        flow_to_queue: torch.Tensor,   # [F]
    ) -> torch.Tensor:
        q_per_flow = queue_state[flow_to_queue]            # [F, D]
        gate = torch.sigmoid(
            self.qf_gate_proj(
                torch.cat([flow_state, q_per_flow], dim=-1)
            )
        )                                                  # [F, D]
        return gate * self.qf_val_proj(q_per_flow)         # [F, D]

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        graph:   dict,
        history: Optional[list] = None,   # accepted for API compat, unused
    ) -> tuple:
        """
        Parameters
        ----------
        graph   : dict from graph_builder.build_graph()
        history : ignored (API compatibility with teacher)

        Returns
        -------
        pred       : [n_flows]   predicted delay (s) or throughput (bps)
        flow_state : [n_flows, D]
        """
        device = next(self.parameters()).device

        def _t(arr, dtype=torch.float32):
            return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

        # ------------------------------------------------------------------ #
        # Load tensors
        # ------------------------------------------------------------------ #
        flow_feat  = _t(graph["flow_feat"])            # [F, 7]
        queue_feat = _t(graph["queue_feat"])           # [Q, 5]
        link_feat  = _t(graph["link_feat"])            # [L, 6]

        f2q = _t(graph["flow_to_queue"], torch.long)   # [F]
        q2l = _t(graph["queue_to_link"], torch.long)   # [Q]
        l2q = _t(graph["link_to_queue"], torch.long)   # [L]

        n_f = flow_feat.size(0)
        n_q = queue_feat.size(0)

        if n_f == 0:
            pred = torch.zeros(0, device=device)
            flow_state = torch.zeros(0, self.hidden_dim, device=device)
            return pred, flow_state

        # Validate indices
        if f2q.numel() > 0:
            assert f2q.max() < n_q, \
                f"flow_to_queue index {f2q.max().item()} >= n_q={n_q}"
        if l2q.numel() > 0:
            assert l2q.max() < n_q, \
                f"link_to_queue index {l2q.max().item()} >= n_q={n_q}"

        # ------------------------------------------------------------------ #
        # Initial embeddings
        # ------------------------------------------------------------------ #
        flow_state  = self.flow_embedding(flow_feat)     # [F, D]
        queue_state = self.queue_embedding(queue_feat)   # [Q, D]
        link_state  = self.link_embedding(link_feat)     # [L, D]

        # ------------------------------------------------------------------ #
        # Sparse Activity Gate  (dampen inactive flows)
        # ------------------------------------------------------------------ #
        flow_state = self.activity_gate(flow_feat, flow_state)  # [F, D]

        # ------------------------------------------------------------------ #
        # Message Passing  (K iterations) + Highway Collection
        # ------------------------------------------------------------------ #
        highway_snapshots = []

        for _ in range(self.iterations):

            # ── Step ① F → Q  (gated scatter aggregation) ──────────────── #
            f2q_msg     = self.f2q_agg(flow_state, queue_state, f2q)
            queue_state = self.fq_norm(self.fq_gru(f2q_msg, queue_state))

            # ── Step ② L → L  (lightweight self-mixing) ────────────────── #
            link_state = self._link_self_mix(link_state)

            # ── Step ③ L → Q  (FiLM modulation) ────────────────────────── #
            q_modulated = self.lq_film(queue_state, link_state, l2q,
                                       mode="modulator_indexed")
            queue_state = self.lq_norm(self.lq_gru(q_modulated, queue_state))

            # ── Step ④ Q → L  (FiLM modulation, reverse) ──────────────── #
            link_state = self.ql_film(link_state, queue_state, l2q,
                                      mode="target_indexed")
            link_state = self.ql_norm(link_state)

            # ── Step ⑤ Q → F  (gated message + GRU) ───────────────────── #
            qf_msg     = self._queue_to_flow_msg(flow_state, queue_state, f2q)
            flow_state = self.qf_norm(self.qf_gru(qf_msg, flow_state))
            flow_state = self.qf_ffn(flow_state)

            # ── Collect highway snapshot ───────────────────────────────── #
            highway_snapshots.append(flow_state)

        # ------------------------------------------------------------------ #
        # Multi-Scale Dense Highway Fusion
        # ------------------------------------------------------------------ #
        highway_cat = torch.cat(highway_snapshots, dim=-1)  # [F, D*K]
        flow_state  = self.highway_norm(
            flow_state + self.highway_fuse(highway_cat)
        )                                                   # [F, D]

        # ------------------------------------------------------------------ #
        # Cross-Flow Mixing  (within-queue)
        # ------------------------------------------------------------------ #
        flow_state = self.cross_flow(flow_state, f2q)

        # ------------------------------------------------------------------ #
        # Readout
        # ------------------------------------------------------------------ #
        pred = self.readout(flow_state).squeeze(-1)  # [F]
        return pred, flow_state
