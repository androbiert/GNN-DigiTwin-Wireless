"""
model2.py — WirelessNet-Fermi v3: Enhanced Architecture

Same graph structure and attention mechanisms as v2, but with:
  1. LayerNorm after every GRU update  (stabilises deep message passing)
  2. FFN blocks with residual connections  (richer per-node transforms)
  3. GRU for Q → L  (replaces simple residual + tanh)
  4. GELU activations  (smoother gradients than ReLU)

Usage:
    from wireless_gnn.model2 import WirelessNetFermiV3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Re-use attention building blocks from model.py (no duplication)
from wireless_gnn.model import (
    FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM,
    scatter_sum, scatter_mean, grouped_softmax,
    FlowToQueueAttention, LinkSelfAttention, LinkToQueueAttention,
    QueueToFlowGating, TemporalAttention, CrossFlowAttention,
)


# --------------------------------------------------------------------------- #
# New building block: Feed-Forward Network with residual
# --------------------------------------------------------------------------- #

class FFNBlock(nn.Module):
    """
    Position-wise Feed-Forward Network (like in a Transformer block).
    
        FFN(x) = LayerNorm(x + Linear(GELU(Linear(x))))
    
    This gives each node the ability to perform a non-linear transform
    on its own state, independent of its neighbours.
    """
    def __init__(self, hidden_dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# --------------------------------------------------------------------------- #
# Main Model: WirelessNetFermi v3  (Enhanced)
# --------------------------------------------------------------------------- #

class WirelessNetFermiV3(nn.Module):
    """
    WirelessNet-Fermi v3 — Enhanced Architecture.

    Improvements over v2:
      - LayerNorm after every GRU update
      - FFN blocks (residual + GELU) after each message-passing step
      - GRU for Q → L (instead of simple tanh + residual)
      - GELU activations in embeddings and readout

    Parameters
    ----------
    hidden_dim   : int   embedding / hidden state size (default 64)
    num_heads    : int   attention heads (default 4)
    iterations   : int   message-passing rounds K (default 8)
    dropout      : float dropout (default 0.1)
    max_history  : int   temporal attention window (default 8 snapshots)
    use_temporal : bool  enable temporal attention (default True)
    target       : str   'delay' or 'throughput'
    """

    def __init__(
        self,
        hidden_dim:   int   = 64,
        num_heads:    int   = 4,
        iterations:   int   = 8,
        dropout:      float = 0.1,
        max_history:  int   = 8,
        use_temporal: bool  = True,
        target:       str   = 'delay',
    ):
        assert target in ('delay', 'throughput'), \
            f"target must be 'delay' or 'throughput', got '{target}'"
        self.target = target
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.iterations   = iterations
        self.use_temporal = use_temporal

        # ------------------------------------------------------------------ #
        # 1. Initial Embeddings  (GELU instead of ReLU)
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
        # 2. Attention modules  (same as v2)
        # ------------------------------------------------------------------ #
        # Step 1 — F → Q
        self.flow_to_queue_attn = FlowToQueueAttention(hidden_dim, num_heads)
        self.fq_gru             = nn.GRUCell(hidden_dim, hidden_dim)
        self.fq_norm            = nn.LayerNorm(hidden_dim)          # NEW
        self.fq_ffn             = FFNBlock(hidden_dim, dropout=dropout)  # NEW

        # Step 2a — L → L  (inter-cell interference)
        self.link_self_attn = LinkSelfAttention(hidden_dim, num_heads, dropout)

        # Step 2b — L → Q  (SINR-weighted GAT)
        self.link_to_queue_attn = LinkToQueueAttention(hidden_dim, num_heads)
        self.lq_gru             = nn.GRUCell(hidden_dim, hidden_dim)
        self.lq_norm            = nn.LayerNorm(hidden_dim)          # NEW
        self.lq_ffn             = FFNBlock(hidden_dim, dropout=dropout)  # NEW

        # Step 2c — Q → L  (GRU instead of simple residual + tanh)
        self.queue_to_link_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ql_gru             = nn.GRUCell(hidden_dim, hidden_dim)  # NEW
        self.ql_norm            = nn.LayerNorm(hidden_dim)            # NEW

        # Step 3 — Q → F  (soft gated)
        self.queue_to_flow_gate = QueueToFlowGating(hidden_dim)
        self.qf_gru             = nn.GRUCell(hidden_dim, hidden_dim)
        self.qf_norm            = nn.LayerNorm(hidden_dim)          # NEW
        self.qf_ffn             = FFNBlock(hidden_dim, dropout=dropout)  # NEW

        # ------------------------------------------------------------------ #
        # 3. Temporal Attention
        # ------------------------------------------------------------------ #
        if use_temporal:
            self.temporal_attn = TemporalAttention(hidden_dim, num_heads, max_history)

        # ------------------------------------------------------------------ #
        # 4. Cross-Flow Attention (before readout)
        # ------------------------------------------------------------------ #
        self.cross_flow_attn = CrossFlowAttention(hidden_dim, num_heads)

        # ------------------------------------------------------------------ #
        # 5. Readout MLP  (GELU instead of ReLU)
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
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        graph:   dict,
        history: Optional[list] = None,
    ) -> tuple:
        """
        Parameters
        ----------
        graph   : dict from graph_builder.build_graph()
        history : optional list of past [F_t, D] flow-state tensors

        Returns
        -------
        pred       : [n_flows]   predicted delay (s) OR throughput (bps)
        flow_state : [n_flows, D]
        """
        device = next(self.parameters()).device

        def _t(arr, dtype=torch.float32):
            return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

        # ------------------------------------------------------------------ #
        # Load tensors
        # ------------------------------------------------------------------ #
        flow_feat  = _t(graph["flow_feat"])           # [F, 8]
        queue_feat = _t(graph["queue_feat"])          # [Q, 2]
        link_feat  = _t(graph["link_feat"])           # [L, 4]

        f2q = _t(graph["flow_to_queue"], torch.long)  # [F]
        q2l = _t(graph["queue_to_link"], torch.long)  # [Q]
        l2q = _t(graph["link_to_queue"], torch.long)  # [L]

        n_f = flow_feat.size(0)
        n_q = queue_feat.size(0)

        if n_f == 0:
            pred = torch.zeros(0, device=device)
            flow_state = torch.zeros(0, self.hidden_dim, device=device)
            return pred, flow_state

        # Validate indices
        if f2q.numel() > 0:
            assert f2q.max() < n_q, \
                f"flow_to_queue has index {f2q.max().item()} but n_q={n_q}"
        if l2q.numel() > 0:
            assert l2q.max() < n_q, \
                f"link_to_queue has index {l2q.max().item()} but n_q={n_q}"

        # ------------------------------------------------------------------ #
        # Initial embeddings
        # ------------------------------------------------------------------ #
        flow_state  = self.flow_embedding(flow_feat)    # [F, D]
        queue_state = self.queue_embedding(queue_feat)  # [Q, D]
        link_state  = self.link_embedding(link_feat)    # [L, D]

        # ------------------------------------------------------------------ #
        # Message Passing  (K iterations)
        # ------------------------------------------------------------------ #
        for _ in range(self.iterations):

            # ── Step 1: F → Q ──────────────────────────────────────────────── #
            flow_agg    = self.flow_to_queue_attn(flow_state, queue_state, f2q)
            queue_state = self.fq_norm(self.fq_gru(flow_agg, queue_state))  # GRU + LN
            queue_state = self.fq_ffn(queue_state)                          # FFN

            # ── Step 2a: L → L ─────────────────────────────────────────────── #
            link_state = self.link_self_attn(link_state)

            # ── Step 2b: L → Q ─────────────────────────────────────────────── #
            link_msg    = self.link_to_queue_attn(queue_state, link_state, l2q)
            queue_state = self.lq_norm(self.lq_gru(link_msg, queue_state))  # GRU + LN
            queue_state = self.lq_ffn(queue_state)                          # FFN

            # ── Step 2c: Q → L  (GRU instead of simple residual) ───────────── #
            q_msg = self.queue_to_link_proj(queue_state[l2q])
            link_state = self.ql_norm(self.ql_gru(q_msg, link_state))       # GRU + LN

            # ── Step 3: Q → F ──────────────────────────────────────────────── #
            flow_msg   = self.queue_to_flow_gate(flow_state, queue_state, f2q)
            flow_state = self.qf_norm(self.qf_gru(flow_msg, flow_state))    # GRU + LN
            flow_state = self.qf_ffn(flow_state)                            # FFN

        # ------------------------------------------------------------------ #
        # Temporal Attention
        # ------------------------------------------------------------------ #
        if self.use_temporal and history:
            flow_state = self.temporal_attn(flow_state, history)

        # ------------------------------------------------------------------ #
        # Cross-Flow Attention
        # ------------------------------------------------------------------ #
        flow_state = self.cross_flow_attn(flow_state, f2q)

        # ------------------------------------------------------------------ #
        # Readout
        # ------------------------------------------------------------------ #
        pred = self.readout(flow_state).squeeze(-1)  # [F]
        return pred, flow_state
