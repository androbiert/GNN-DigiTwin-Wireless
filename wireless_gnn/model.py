"""
model.py — WirelessNet-Fermi v2: Full Attention Architecture

Two specialised instances are trained separately:
  - WirelessNetFermi(target='delay')      → predicts end-to-end delay
  - WirelessNetFermi(target='throughput') → predicts throughput

Architecture (5 attention mechanisms):
============================================================
  Nodes: Flow (F), Queue (Q), Link (L)

  Message Passing (K iterations):
    1. F → Q  : FlowToQueueAttention  — queues weight flows by criticality
    2. L → L  : LinkSelfAttention     — links model inter-cell interference
    3. L → Q  : LinkToQueueAttention  — SINR-weighted GAT (original)
    4. Q → L  : residual proj + gating
    5. Q → F  : QueueToFlowGating     — soft gate per flow (not uniform)

  Temporal:
    TemporalAttention — cross-snapshot memory (handover-aware)

  Readout:
    CrossFlowAttention → MLP → scalar pred (delay OR throughput)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# --------------------------------------------------------------------------- #
# Feature dimensions (must match graph_builder.py)
# --------------------------------------------------------------------------- #
FLOW_FEAT_DIM  = 8   # packet_size, interval, throughput, offered_load,
                      # packet_loss, harq_error_rate, harq_tx_attempts, delivery_ratio
QUEUE_FEAT_DIM = 2   # qsize_bytes, mac_buffer_overflow
LINK_FEAT_DIM  = 4   # sinr_dl, sinr_ul, distance, speed


# --------------------------------------------------------------------------- #
# Helper: scatter operations
# --------------------------------------------------------------------------- #

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out   = scatter_sum(src, index, dim_size)
    count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index.unsqueeze(1),
                       torch.ones(len(index), 1, dtype=src.dtype, device=src.device))
    return out / count.clamp(min=1.0)


def grouped_softmax(scores: torch.Tensor, index: torch.Tensor,
                    n_groups: int) -> torch.Tensor:
    """
    Numerically-stable softmax where each element i belongs to group index[i].
    scores : [N, H]   raw attention logits
    index  : [N]      group id for each row
    returns: [N, H]   attention weights, summing to 1 within each group
    """
    # Per-group max for stability
    max_per_group = torch.full((n_groups, scores.size(1)), float('-inf'),
                               dtype=scores.dtype, device=scores.device)
    max_per_group.scatter_reduce_(0,
                                  index.unsqueeze(1).expand_as(scores),
                                  scores, reduce='amax', include_self=True)
    scores_shifted = scores - max_per_group[index]          # [N, H]
    exp_s = torch.exp(scores_shifted)
    denom = torch.zeros(n_groups, scores.size(1),
                        dtype=scores.dtype, device=scores.device)
    denom.scatter_add_(0, index.unsqueeze(1).expand_as(exp_s), exp_s)
    return exp_s / (denom[index] + 1e-9)


# --------------------------------------------------------------------------- #
# 1. FlowToQueueAttention  (F → Q)
# --------------------------------------------------------------------------- #

class FlowToQueueAttention(nn.Module):
    """
    Each queue attends over its flows.
    Query = queue state,  Key/Value = flow states.
    Flows with high offered_load or packet_loss get higher attention.

    e_ij = (W_q·h_q_i)^T (W_k·h_f_j) / sqrt(d_h)
    α_ij = softmax over j (flows of queue i)
    h_q_i' = sum_j α_ij · W_v·h_f_j
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.H  = num_heads
        self.D  = hidden_dim
        self.dh = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_q  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.dh ** -0.5

    def forward(self,
                flow_state:    torch.Tensor,   # [F, D]
                queue_state:   torch.Tensor,   # [Q, D]
                flow_to_queue: torch.Tensor,   # [F]  long
                ) -> torch.Tensor:             # [Q, D]
        n_f = flow_state.size(0)
        n_q = queue_state.size(0)
        H, dh = self.H, self.dh

        Q = self.W_q(queue_state[flow_to_queue]).view(n_f, H, dh)  # [F, H, dh]
        K = self.W_k(flow_state).view(n_f, H, dh)                  # [F, H, dh]
        V = self.W_v(flow_state).view(n_f, H, dh)                  # [F, H, dh]

        scores = (Q * K).sum(-1) * self.scale                       # [F, H]
        alpha  = grouped_softmax(scores, flow_to_queue, n_q)        # [F, H]

        weighted = (alpha.unsqueeze(-1) * V).view(n_f, self.D)      # [F, D]
        agg = scatter_sum(weighted, flow_to_queue, n_q)             # [Q, D]
        return self.proj(agg)


# --------------------------------------------------------------------------- #
# 2. LinkSelfAttention  (L → L)
# --------------------------------------------------------------------------- #

class LinkSelfAttention(nn.Module):
    """
    Self-attention across ALL link nodes.
    Models inter-cell interference: a link with low SINR can attend to
    neighbouring high-power links that are causing the interference.

    Uses standard scaled dot-product attention (all-pairs, O(L²) — acceptable
    because L = number of UEs, typically < 30 in a scenario snapshot).
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, link_state: torch.Tensor) -> torch.Tensor:  # [L, D]
        x = link_state.unsqueeze(0)          # [1, L, D]  (batch=1)
        out, _ = self.attn(x, x, x)         # [1, L, D]
        return self.norm(link_state + out.squeeze(0))   # residual + LN


# --------------------------------------------------------------------------- #
# 3. LinkToQueueAttention  (L → Q)  — original GAT, kept & improved
# --------------------------------------------------------------------------- #

class LinkToQueueAttention(nn.Module):
    """
    Multi-head attention from Link nodes to Queue nodes.
    SINR-weighted: each queue aggregates link radio states.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.H  = num_heads
        self.D  = hidden_dim
        self.dh = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_q  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.dh ** -0.5

    def forward(self,
                queue_state:   torch.Tensor,   # [Q, D]
                link_state:    torch.Tensor,   # [L, D]
                link_to_queue: torch.Tensor,   # [L]  long
                ) -> torch.Tensor:             # [Q, D]
        n_l = link_state.size(0)
        n_q = queue_state.size(0)
        H, dh = self.H, self.dh

        Q = self.W_q(queue_state[link_to_queue]).view(n_l, H, dh)
        K = self.W_k(link_state).view(n_l, H, dh)
        V = self.W_v(link_state).view(n_l, H, dh)

        scores = (Q * K).sum(-1) * self.scale                        # [L, H]
        alpha  = grouped_softmax(scores, link_to_queue, n_q)         # [L, H]

        weighted = (alpha.unsqueeze(-1) * V).view(n_l, self.D)       # [L, D]
        agg = scatter_sum(weighted, link_to_queue, n_q)              # [Q, D]
        return self.proj(agg)


# --------------------------------------------------------------------------- #
# 4. QueueToFlowGating  (Q → F)
# --------------------------------------------------------------------------- #

class QueueToFlowGating(nn.Module):
    """
    Soft-gated message from queue to each flow.
    Instead of broadcasting the same queue state to all its flows,
    each flow computes a personalised gate:

        g_i = sigmoid(W_gate · [h_f_i || h_q_{f2q_i}])
        msg_i = g_i ⊙ W_v · h_q_{f2q_i}

    Flows with high packet_loss / low delivery_ratio open the gate wider,
    amplifying the queue signal for those flows that need it most.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_val  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                flow_state:    torch.Tensor,   # [F, D]
                queue_state:   torch.Tensor,   # [Q, D]
                flow_to_queue: torch.Tensor,   # [F]  long
                ) -> torch.Tensor:             # [F, D]
        q_per_flow = queue_state[flow_to_queue]             # [F, D]
        gate = torch.sigmoid(
            self.W_gate(torch.cat([flow_state, q_per_flow], dim=-1))
        )                                                   # [F, D]
        return gate * self.W_val(q_per_flow)               # [F, D]


# --------------------------------------------------------------------------- #
# 5. TemporalAttention  (cross-snapshot memory)
# --------------------------------------------------------------------------- #

class TemporalAttention(nn.Module):
    """
    Attends over the last T flow-state snapshots to build temporal context.
    Useful for tracking flows across handovers and mobility events.

    history : list of [F_t, D] tensors (variable F per snapshot — we use
              only the CURRENT flow dimension F_cur and align by position).
    Because flow count can change between snapshots, we pad/crop to n_cur.

    Architecture:
        Q = current flow state
        K = V = stacked history   [T, F_cur, D]
        output per flow = weighted average over T past states
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, max_history: int = 8):
        super().__init__()
        self.max_history = max_history
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_enc = nn.Embedding(max_history + 1, hidden_dim)  # positional bias

    def forward(self,
                flow_state: torch.Tensor,            # [F, D]  current
                history:    list,                    # list of past [F_t, D]
                ) -> torch.Tensor:                   # [F, D]
        if not history:
            return flow_state

        F, D = flow_state.shape
        device = flow_state.device

        # Align history snapshots to current F (pad/crop rows)
        aligned = []
        for t, h in enumerate(history[-self.max_history:]):
            h_t = h.detach()
            if h_t.size(0) < F:
                pad = torch.zeros(F - h_t.size(0), D, device=device, dtype=h_t.dtype)
                h_t = torch.cat([h_t, pad], dim=0)
            else:
                h_t = h_t[:F]
            pos = self.pos_enc(torch.tensor(t, device=device))  # [D]
            aligned.append(h_t + pos.unsqueeze(0))              # [F, D]

        # Stack: [T, F, D] → transpose to [F, T, D] for batch_first
        mem = torch.stack(aligned, dim=0).transpose(0, 1)       # [F, T, D]
        q   = flow_state.unsqueeze(1)                           # [F, 1, D]

        ctx, _ = self.attn(q, mem, mem)                         # [F, 1, D]
        ctx = ctx.squeeze(1)                                    # [F, D]
        return self.norm(flow_state + ctx)


# --------------------------------------------------------------------------- #
# 6. CrossFlowAttention  (readout refinement)
# --------------------------------------------------------------------------- #

class CrossFlowAttention(nn.Module):
    """
    Before the readout MLP, flows that share the same queue attend to each
    other's states. This captures correlations between co-located flows
    (e.g. two video streams from the same UE compete for the same buffer).

    Implementation: full self-attention within each queue group.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.H  = num_heads
        self.D  = hidden_dim
        self.dh = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_q  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = self.dh ** -0.5

    def forward(self,
                flow_state:    torch.Tensor,   # [F, D]
                flow_to_queue: torch.Tensor,   # [F]  long
                ) -> torch.Tensor:             # [F, D]
        n_f = flow_state.size(0)
        H, dh, D = self.H, self.dh, self.D

        Q = self.W_q(flow_state).view(n_f, H, dh)   # [F, H, dh]
        K = self.W_k(flow_state).view(n_f, H, dh)
        V = self.W_v(flow_state).view(n_f, H, dh)

        # [F_i, F_j, H] pairwise scores
        scores = torch.einsum('ihd,jhd->ijh', Q, K) * self.scale  # [F, F, H]

        # Mask: only attend within the same queue
        same_q = (flow_to_queue.unsqueeze(1) == flow_to_queue.unsqueeze(0))  # [F, F]
        mask = ~same_q                                               # True → ignore
        scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        alpha = torch.softmax(scores, dim=1)                        # [F, F, H]
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted aggregation: [F, F, H] x [F, H, dh] → [F, H, dh]
        agg = torch.einsum('ijh,jhd->ihd', alpha, V).reshape(n_f, D)  # [F, D]
        out = self.proj(agg)
        return self.norm(flow_state + out)   # residual + LayerNorm


# --------------------------------------------------------------------------- #
# Main Model: WirelessNetFermi v2
# --------------------------------------------------------------------------- #

class WirelessNetFermi(nn.Module):
    """
    WirelessNet-Fermi v2 — Full Attention Architecture.

    Parameters
    ----------
    hidden_dim   : int   embedding / hidden state size (default 64)
    num_heads    : int   attention heads (default 4)
    iterations   : int   message-passing rounds K (default 8)
    dropout      : float dropout on readout MLP (default 0.1)
    max_history  : int   temporal attention window (default 8 snapshots)
    use_temporal : bool  enable temporal attention (default True)
    """

    def __init__(
        self,
        hidden_dim:   int   = 64,
        num_heads:    int   = 4,
        iterations:   int   = 8,
        dropout:      float = 0.1,
        max_history:  int   = 8,
        use_temporal: bool  = True,
        target:       str   = 'delay',   # 'delay' or 'throughput'
    ):
        assert target in ('delay', 'throughput'), \
            f"target must be 'delay' or 'throughput', got '{target}'"
        self.target = target
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.iterations   = iterations
        self.use_temporal = use_temporal

        # ------------------------------------------------------------------ #
        # 1. Initial Embeddings
        # ------------------------------------------------------------------ #
        self.flow_embedding = nn.Sequential(
            nn.Linear(FLOW_FEAT_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ReLU(),
        )
        self.queue_embedding = nn.Sequential(
            nn.Linear(QUEUE_FEAT_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),     nn.ReLU(),
        )
        self.link_embedding = nn.Sequential(
            nn.Linear(LINK_FEAT_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        # 2. Attention modules
        # ------------------------------------------------------------------ #
        # Step 1 — F → Q
        self.flow_to_queue_attn = FlowToQueueAttention(hidden_dim, num_heads)
        self.fq_gru             = nn.GRUCell(hidden_dim, hidden_dim)

        # Step 2a — L → L  (inter-cell interference)
        self.link_self_attn = LinkSelfAttention(hidden_dim, num_heads, dropout)

        # Step 2b — L → Q  (SINR-weighted GAT)
        self.link_to_queue_attn = LinkToQueueAttention(hidden_dim, num_heads)
        self.lq_gru             = nn.GRUCell(hidden_dim, hidden_dim)

        # Step 2c — Q → L  (queue load back to link)
        self.queue_to_link_proj = nn.Linear(hidden_dim, hidden_dim)

        # Step 3 — Q → F  (soft gated)
        self.queue_to_flow_gate = QueueToFlowGating(hidden_dim)
        self.qf_gru             = nn.GRUCell(hidden_dim, hidden_dim)

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
        # 5. Readout MLP  (single output — specialised per target)
        # ------------------------------------------------------------------ #
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),   # scalar: delay OR throughput
        )

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        graph:   dict,
        history: Optional[list] = None,   # list of past flow_state tensors
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

            # ── Step 1: F → Q  (attention-weighted flow aggregation) ─────── #
            flow_agg    = self.flow_to_queue_attn(flow_state, queue_state, f2q)
            queue_state = self.fq_gru(flow_agg, queue_state)

            # ── Step 2a: L → L  (inter-link self-attention) ──────────────── #
            link_state = self.link_self_attn(link_state)

            # ── Step 2b: L → Q  (SINR-weighted GAT) ──────────────────────── #
            link_msg    = self.link_to_queue_attn(queue_state, link_state, l2q)
            queue_state = self.lq_gru(link_msg, queue_state)

            # ── Step 2c: Q → L  (buffer load back to link, residual) ─────── #
            q_msg_for_link = self.queue_to_link_proj(queue_state[l2q])
            link_state     = link_state + torch.tanh(q_msg_for_link)

            # ── Step 3: Q → F  (soft-gated queue → flow message) ─────────── #
            flow_msg   = self.queue_to_flow_gate(flow_state, queue_state, f2q)
            flow_state = self.qf_gru(flow_msg, flow_state)

        # ------------------------------------------------------------------ #
        # Temporal Attention  (cross-snapshot context)
        # ------------------------------------------------------------------ #
        if self.use_temporal and history:
            flow_state = self.temporal_attn(flow_state, history)

        # ------------------------------------------------------------------ #
        # Cross-Flow Attention  (co-located flow correlation)
        # ------------------------------------------------------------------ #
        flow_state = self.cross_flow_attn(flow_state, f2q)

        # ------------------------------------------------------------------ #
        # Readout
        # ------------------------------------------------------------------ #
        pred = self.readout(flow_state).squeeze(-1)  # [F]
        return pred, flow_state
