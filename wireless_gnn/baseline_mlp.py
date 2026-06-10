"""
baseline_mlp.py — MLP Baseline for Throughput Prediction

A simple feed-forward MLP that predicts per-flow throughput
WITHOUT using any graph structure or message passing.

For each flow, the input is the concatenation of:
  - Flow features      (7 dims)
  - Queue features      (5 dims)
  - Link features       (6 dims)

Total input: 18 dims -> MLP -> 1 scalar

This serves as a baseline to measure the added value of the GNN architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from wireless_gnn.model import FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM


# Total input dimension per flow
BASELINE_INPUT_DIM = FLOW_FEAT_DIM + QUEUE_FEAT_DIM + LINK_FEAT_DIM  # 7 + 5 + 6 = 18


class BaselineMLP(nn.Module):
    """
    MLP Baseline — no graph structure, no message passing.

    For each flow, concatenates [flow_feat || queue_feat || link_feat]
    and predicts throughput (or delay) with a standard feed-forward network.

    Parameters
    ----------
    hidden_dim : int    hidden layer size (default 128)
    num_layers : int    number of hidden layers (default 3)
    dropout    : float  dropout rate (default 0.1)
    target     : str    'delay' or 'throughput'
    """

    def __init__(
        self,
        hidden_dim:  int   = 128,
        num_layers:  int   = 3,
        dropout:     float = 0.1,
        target:      str   = 'throughput',
        # Unused kwargs for compatibility with train_scenarios.py interface
        num_heads:   int   = 4,
        iterations:  int   = 8,
    ):
        super().__init__()
        assert target in ('delay', 'throughput'), \
            f"target must be 'delay' or 'throughput', got '{target}'"
        self.target     = target
        self.hidden_dim = hidden_dim

        # Build MLP layers
        layers = []
        in_dim = BASELINE_INPUT_DIM

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def _build_flat_input(self, graph: dict, device: torch.device) -> torch.Tensor:
        """
        For each flow, concatenate its own features with its queue's
        and link's features to create a flat input vector.

        Returns: [n_flows, 17]
        """
        flow_feat  = torch.tensor(np.asarray(graph["flow_feat"]),
                                  dtype=torch.float32, device=device)    # [F, 7]
        queue_feat = torch.tensor(np.asarray(graph["queue_feat"]),
                                  dtype=torch.float32, device=device)    # [Q, 4]
        link_feat  = torch.tensor(np.asarray(graph["link_feat"]),
                                  dtype=torch.float32, device=device)    # [L, 6]

        f2q = torch.tensor(np.asarray(graph["flow_to_queue"]),
                           dtype=torch.long, device=device)              # [F]
        q2l = torch.tensor(np.asarray(graph["queue_to_link"]),
                           dtype=torch.long, device=device)              # [Q]

        # Gather queue and link features for each flow
        queue_per_flow = queue_feat[f2q]                     # [F, 2]
        link_per_flow  = link_feat[q2l[f2q]]                 # [F, 4]

        # Concatenate: [flow || queue || link]
        flat = torch.cat([flow_feat, queue_per_flow, link_per_flow], dim=-1)  # [F, 18]
        return flat

    def forward(
        self,
        graph:   dict,
        history: Optional[list] = None,   # ignored — kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        graph   : dict from graph_builder.build_graph()
        history : ignored (kept for interface compatibility)

        Returns
        -------
        pred       : [n_flows]   predicted throughput (or delay)
        flow_state : [n_flows, hidden_dim]  (dummy, for compat)
        """
        device = next(self.parameters()).device
        n_f = len(graph["flow_feat"])

        if n_f == 0:
            pred = torch.zeros(0, device=device)
            dummy = torch.zeros(0, self.hidden_dim, device=device)
            return pred, dummy

        flat_input = self._build_flat_input(graph, device)   # [F, 17]

        pred = self.mlp(flat_input).squeeze(-1)              # [F]

        # Return dummy flow_state for compatibility
        dummy_state = torch.zeros(n_f, self.hidden_dim, device=device)
        return pred, dummy_state
