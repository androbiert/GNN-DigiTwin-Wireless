"""
model.py — WirelessNet-Fermi: Dynamic GNN for Wireless QoS Prediction

Architecture (mirrors RouteNet-Fermi's tripartite message passing):
============================================================
  Nodes: Flow (F), Queue (Q), Link (L)

  Message Passing (K iterations):
    1. F → Q  : flows push traffic demand into their UE queue
                (GRU update on queue state)
    2. L → Q  : link injects radio channel state into queue
                (GAT attention — SINR-weighted aggregation)
    3. Q → F  : queue feeds combined load+channel info back to flows
                (GRU update on flow state)

  Readout:
    MLP(flow_state) → [delay_pred, throughput_pred]

Key differences vs RouteNet-Fermi:
  - PyTorch instead of TensorFlow
  - GAT multi-head attention replaces plain GRU on link→queue step
  - Dual output: delay + throughput
  - Dynamic graph: topology rebuilt per timestamp (handover-aware)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
# Feature dimensions (must match graph_builder.py)
# --------------------------------------------------------------------------- #
FLOW_FEAT_DIM  = 8   # packet_size, interval, throughput, offered_load,
                      # packet_loss, harq_error_rate, harq_tx_attempts, delivery_ratio
QUEUE_FEAT_DIM = 3   # rlcDelay, qsize_bytes, mac_buffer_overflow
LINK_FEAT_DIM  = 4   # sinr_dl, sinr_ul, distance, speed


# --------------------------------------------------------------------------- #
# Helper: scatter operations (sum / mean over variable-size neighbour sets)
# --------------------------------------------------------------------------- #

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Aggregate src[i] into output[index[i]] by summation.
    src   : [N, D]
    index : [N]   (long)
    out   : [dim_size, D]
    """
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out   = scatter_sum(src, index, dim_size)
    count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones(len(index), 1, dtype=src.dtype, device=src.device))
    count = count.clamp(min=1.0)
    return out / count


# --------------------------------------------------------------------------- #
# GAT Attention: Link → Queue
# --------------------------------------------------------------------------- #

class LinkToQueueAttention(nn.Module):
    """
    Multi-head attention from Link nodes to Queue nodes.

    Each queue attends over the links that serve it (usually just one,
    but during handover there could briefly be two competing gNBs).

    We use a lightweight additive-attention (Bahdanau-style):
        e_ij = v^T · tanh(W_q · h_q_i  +  W_l · h_l_j)
        α_ij = softmax over j for each i
        h_q_i' = sum_j α_ij · h_l_j

    Since each queue usually has exactly one link, the attention also
    functions as a gating mechanism: if SINR is low the gate suppresses
    the link contribution.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads  = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_val   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale    = (hidden_dim // num_heads) ** -0.5

    def forward(
        self,
        queue_state: torch.Tensor,  # [n_queues, D]
        link_state:  torch.Tensor,  # [n_links,  D]
        link_to_queue: torch.Tensor,  # [n_links]  long — which queue each link belongs to
    ) -> torch.Tensor:
        """Returns updated queue state aggregated from links. [n_queues, D]"""
        n_q = queue_state.size(0)
        n_l = link_state.size(0)
        D   = self.hidden_dim
        H   = self.num_heads
        d_h = D // H

        # Project
        Q = self.W_query(queue_state[link_to_queue])  # [n_links, D]
        K = self.W_key(link_state)                    # [n_links, D]
        V = self.W_val(link_state)                    # [n_links, D]

        # Reshape for multi-head
        Q = Q.view(n_l, H, d_h)   # [n_links, H, d_h]
        K = K.view(n_l, H, d_h)
        V = V.view(n_l, H, d_h)

        # Scaled dot-product score per (link, head)
        scores = (Q * K).sum(dim=-1) * self.scale  # [n_links, H]
        # We need per-queue softmax: group links by their queue
        # Build attention weights using exp → scatter_sum → normalise
        exp_scores = torch.exp(scores - scores.max(dim=0, keepdim=True).values)  # [n_links, H]

        # Sum of exp scores per queue (for softmax denominator)
        idx_exp = link_to_queue.unsqueeze(1).expand_as(exp_scores)   # [n_links, H]
        denom   = torch.zeros(n_q, H, dtype=exp_scores.dtype, device=exp_scores.device)
        denom.scatter_add_(0, idx_exp, exp_scores)                   # [n_queues, H]
        denom   = denom[link_to_queue] + 1e-9                         # [n_links, H]

        alpha = exp_scores / denom  # [n_links, H]  (soft attention weights)

        # Weighted sum of values
        weighted_V = alpha.unsqueeze(-1) * V  # [n_links, H, d_h]
        weighted_V = weighted_V.view(n_l, D)  # [n_links, D]

        # Scatter to queue nodes
        agg = scatter_sum(weighted_V, link_to_queue, n_q)  # [n_queues, D]
        return self.out_proj(agg)


# --------------------------------------------------------------------------- #
# Main Model: WirelessNetFermi
# --------------------------------------------------------------------------- #

class WirelessNetFermi(nn.Module):
    """
    WirelessNet-Fermi: RouteNet-Fermi adapted for 5G wireless networks.

    Parameters
    ----------
    hidden_dim  : int   embedding / hidden state size (default 64)
    num_heads   : int   attention heads for Link→Queue GAT (default 4)
    iterations  : int   message-passing rounds K (default 8)
    dropout     : float dropout on readout MLP (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads:  int = 4,
        iterations: int = 8,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.iterations = iterations

        # ------------------------------------------------------------------ #
        # 1. Initial Embeddings (raw features → hidden_dim)
        # ------------------------------------------------------------------ #
        self.flow_embedding = nn.Sequential(
            nn.Linear(FLOW_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.queue_embedding = nn.Sequential(
            nn.Linear(QUEUE_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.link_embedding = nn.Sequential(
            nn.Linear(LINK_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        # 2. Message Passing Cells
        # ------------------------------------------------------------------ #
        # Step 1: F → Q  (flows aggregate into queue)
        self.flow_to_queue_gru  = nn.GRUCell(hidden_dim, hidden_dim)
        # Step 2: L → Q  (links attend into queue)  [GAT]
        self.link_to_queue_attn = LinkToQueueAttention(hidden_dim, num_heads)
        self.queue_update_gru   = nn.GRUCell(hidden_dim, hidden_dim)
        # Step 3: Q → F  (queue informs flow)
        self.queue_to_flow_gru  = nn.GRUCell(hidden_dim, hidden_dim)

        # Projection for link update from queue (queue→link direction)
        self.queue_to_link_proj = nn.Linear(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # 3. Readout MLP  (flow_state → [delay, throughput])
        # ------------------------------------------------------------------ #
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),   # [delay, throughput]
        )

    # ---------------------------------------------------------------------- #
    # Forward (single graph snapshot)
    # ---------------------------------------------------------------------- #
    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        graph : dict returned by graph_builder.build_graph()
            Keys needed: flow_feat, queue_feat, link_feat,
                         flow_to_queue, queue_to_link, link_to_queue

        Returns
        -------
        delay_pred      : [n_flows]   predicted end-to-end delay (s)
        throughput_pred : [n_flows]   predicted throughput (bps)
        """
        device = next(self.parameters()).device

        def _t(arr, dtype=torch.float32):
            return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

        # ------------------------------------------------------------------ #
        # Load tensors
        # ------------------------------------------------------------------ #
        flow_feat    = _t(graph["flow_feat"])          # [F, 8]
        queue_feat   = _t(graph["queue_feat"])         # [Q, 3]
        link_feat    = _t(graph["link_feat"])          # [L, 4]

        f2q = _t(graph["flow_to_queue"],  torch.long)  # [F]
        q2l = _t(graph["queue_to_link"],  torch.long)  # [Q]
        l2q = _t(graph["link_to_queue"],  torch.long)  # [L]

        n_f = flow_feat.size(0)
        n_q = queue_feat.size(0)
        n_l = link_feat.size(0)

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

            # ── Step 1: F → Q ─────────────────────────────────────────── #
            # Each queue aggregates the mean of its flows' states
            flow_agg = scatter_mean(flow_state, f2q, n_q)   # [Q, D]
            queue_state = self.flow_to_queue_gru(flow_agg, queue_state)

            # ── Step 2: L → Q  (GAT attention — channel quality aware) ── #
            link_msg    = self.link_to_queue_attn(queue_state, link_state, l2q)  # [Q, D]
            queue_state = self.queue_update_gru(link_msg, queue_state)

            # ── Step 2b: Q → L  (queue state propagates back to link) ─── #
            # Allows link to "know" about the buffer load of its UE
            queue_msg_for_link = self.queue_to_link_proj(queue_state[l2q])  # [L, D]
            link_state = link_state + torch.tanh(queue_msg_for_link)         # residual

            # ── Step 3: Q → F ─────────────────────────────────────────── #
            queue_msg = queue_state[f2q]                    # [F, D]
            flow_state = self.queue_to_flow_gru(queue_msg, flow_state)

        # ------------------------------------------------------------------ #
        # Readout
        # ------------------------------------------------------------------ #
        out = self.readout(flow_state)          # [F, 2]
        delay_pred      = out[:, 0]             # [F]
        throughput_pred = out[:, 1]             # [F]

        return delay_pred, throughput_pred
