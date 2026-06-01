"""
baseline_gnn.py — Simple Homogeneous GNN Baseline

A baseline model that removes graph heterogeneity and attention mechanisms.
It pads all node types (Flow, Queue, Link) to the same feature dimension,
constructs a single homogeneous graph, and uses simple mean-pooling
for message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from wireless_gnn.model import FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM

# We pad all node features to the max dimension (which is Flow: 8)
HOMOGENEOUS_FEAT_DIM = max(FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM)

class SimpleGNNLayer(nn.Module):
    """
    A very simple message passing layer.
    h_i^{l+1} = ReLU( Linear( h_i^l + mean_{j in N(i)} h_j^l ) )
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Aggregate messages from neighbors
        msg = x[src]
        
        out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(count)[src])
        
        # Mean pooling
        agg = out / count.clamp(min=1.0)
        
        # Add self-connection (x) and apply linear transformation
        h_new = self.linear(x + agg)
        return F.relu(h_new)

class BaselineGNN(nn.Module):
    """
    Homogeneous GNN Baseline without Attention.
    """
    def __init__(
        self,
        hidden_dim:  int   = 64,
        iterations:  int   = 3,
        dropout:     float = 0.1,
        target:      str   = 'delay',
        # Unused kwargs for compatibility
        num_heads:   int   = 4,
        max_history: int   = 8,
        use_temporal: bool = False
    ):
        super().__init__()
        assert target in ('delay', 'throughput'), f"Invalid target: {target}"
        self.target = target
        self.hidden_dim = hidden_dim
        self.iterations = iterations

        # Single shared embedding for the padded homogeneous features
        self.node_emb = nn.Sequential(
            nn.Linear(HOMOGENEOUS_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Simple GNN layers
        self.layers = nn.ModuleList([
            SimpleGNNLayer(hidden_dim) for _ in range(iterations)
        ])

        # Readout MLP (predicts target for flow nodes only)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim , 1)
        )

    def forward(
        self,
        graph:   dict,
        history: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        
        flow_feat  = torch.tensor(np.asarray(graph["flow_feat"]), dtype=torch.float32, device=device)
        queue_feat = torch.tensor(np.asarray(graph["queue_feat"]), dtype=torch.float32, device=device)
        link_feat  = torch.tensor(np.asarray(graph["link_feat"]), dtype=torch.float32, device=device)

        n_f = flow_feat.size(0)
        n_q = queue_feat.size(0)
        n_l = link_feat.size(0)
        n_total = n_f + n_q + n_l

        if n_f == 0:
            return torch.zeros(0, device=device), torch.zeros(0, self.hidden_dim, device=device)

        # 1. Pad features to HOMOGENEOUS_FEAT_DIM
        flow_pad  = F.pad(flow_feat,  (0, HOMOGENEOUS_FEAT_DIM - FLOW_FEAT_DIM))
        queue_pad = F.pad(queue_feat, (0, HOMOGENEOUS_FEAT_DIM - QUEUE_FEAT_DIM))
        link_pad  = F.pad(link_feat,  (0, HOMOGENEOUS_FEAT_DIM - LINK_FEAT_DIM))

        # 2. Concatenate into a single node matrix
        X = torch.cat([flow_pad, queue_pad, link_pad], dim=0) # [n_total, 8]
        X = self.node_emb(X) # [n_total, hidden_dim]

        # 3. Build edge list
        f2q = torch.tensor(np.asarray(graph["flow_to_queue"]), dtype=torch.long, device=device)
        q2l = torch.tensor(np.asarray(graph["queue_to_link"]), dtype=torch.long, device=device)
        l2q = torch.tensor(np.asarray(graph["link_to_queue"]), dtype=torch.long, device=device)

        src_list = []
        dst_list = []

        # Flow <-> Queue edges
        if len(f2q) > 0:
            f_idx = torch.arange(n_f, device=device)
            q_idx = f2q + n_f
            src_list.extend([f_idx, q_idx])
            dst_list.extend([q_idx, f_idx])

        # Queue <-> Link edges
        if len(q2l) > 0:
            q_idx2 = torch.arange(n_q, device=device) + n_f
            l_idx = q2l + n_f + n_q
            src_list.extend([q_idx2, l_idx])
            dst_list.extend([l_idx, q_idx2])
            
        # Link <-> Queue edges
        if len(l2q) > 0:
            l_idx2 = torch.arange(n_l, device=device) + n_f + n_q
            q_idx3 = l2q + n_f
            src_list.extend([l_idx2, q_idx3])
            dst_list.extend([q_idx3, l_idx2])

        if len(src_list) > 0:
            src = torch.cat(src_list)
            dst = torch.cat(dst_list)
            
            # 4. Message Passing
            for layer in self.layers:
                X = layer(X, src, dst, n_total)

        # 5. Readout (only extract Flow nodes)
        flow_states = X[:n_f]
        pred = self.readout(flow_states).squeeze(-1) # [n_f]

        return pred, flow_states
