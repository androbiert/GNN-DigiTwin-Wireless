"""
baseline_lstm.py — LSTM Baseline for Throughput/Delay Prediction

A bidirectional LSTM that predicts per-flow throughput (or delay)
WITHOUT using any graph structure or message passing.

For each flow, the input is the concatenation of:
  - Flow features      (8 dims)
  - Queue features      (2 dims)  ← the queue this flow belongs to
  - Link features       (4 dims)  ← the link associated with the flow's queue

Total input: 14 dims → Linear projection → BiLSTM → MLP → 1 scalar

Flows are ordered by queue assignment to give the LSTM some structural
information, but this is far weaker than the GNN's explicit graph topology
with attention-weighted message passing.

This serves as a baseline to measure the added value of the GNN architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from wireless_gnn.model import FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM


# Total input dimension per flow
BASELINE_INPUT_DIM = FLOW_FEAT_DIM + QUEUE_FEAT_DIM + LINK_FEAT_DIM  # 8 + 2 + 4 = 14


class BaselineLSTM(nn.Module):
    """
    Bidirectional LSTM Baseline — no graph structure, no message passing.

    For each flow, concatenates [flow_feat || queue_feat || link_feat],
    orders flows by queue index, runs through a BiLSTM, and predicts
    throughput (or delay) with a readout MLP.

    Parameters
    ----------
    hidden_dim : int    hidden layer size (default 128)
    num_layers : int    number of LSTM layers (default 2)
    dropout    : float  dropout rate (default 0.1)
    target     : str    'delay' or 'throughput'
    """

    def __init__(
        self,
        hidden_dim:  int   = 128,
        num_layers:  int   = 2,
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

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(BASELINE_INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project bidirectional output back to hidden_dim
        self.bidir_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_flat_input(self, graph: dict, device: torch.device) -> torch.Tensor:
        """
        For each flow, concatenate its own features with its queue's
        and link's features to create a flat input vector.

        Returns: [n_flows, 14]
        """
        flow_feat  = torch.tensor(np.asarray(graph["flow_feat"]),
                                  dtype=torch.float32, device=device)    # [F, 8]
        queue_feat = torch.tensor(np.asarray(graph["queue_feat"]),
                                  dtype=torch.float32, device=device)    # [Q, 2]
        link_feat  = torch.tensor(np.asarray(graph["link_feat"]),
                                  dtype=torch.float32, device=device)    # [L, 4]

        f2q = torch.tensor(np.asarray(graph["flow_to_queue"]),
                           dtype=torch.long, device=device)              # [F]
        q2l = torch.tensor(np.asarray(graph["queue_to_link"]),
                           dtype=torch.long, device=device)              # [Q]

        # Gather queue and link features for each flow
        queue_per_flow = queue_feat[f2q]                     # [F, 2]
        link_per_flow  = link_feat[q2l[f2q]]                 # [F, 4]

        # Concatenate: [flow || queue || link]
        flat = torch.cat([flow_feat, queue_per_flow, link_per_flow], dim=-1)  # [F, 14]
        return flat, f2q

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

        flat_input, f2q = self._build_flat_input(graph, device)   # [F, 14], [F]

        # Sort flows by queue index to give LSTM sequential structure
        sort_idx = torch.argsort(f2q)
        sorted_input = flat_input[sort_idx]                       # [F, 14]

        # Project input
        x = self.input_proj(sorted_input)                         # [F, hidden_dim]

        # Add batch dimension for LSTM: [1, F, hidden_dim]
        x = x.unsqueeze(0)

        # BiLSTM forward
        lstm_out, _ = self.lstm(x)                                # [1, F, 2*hidden_dim]
        lstm_out = lstm_out.squeeze(0)                            # [F, 2*hidden_dim]

        # Project back to hidden_dim
        h = self.bidir_proj(lstm_out)                             # [F, hidden_dim]

        # Readout
        pred_sorted = self.readout(h).squeeze(-1)                 # [F]

        # Unsort to original flow order
        unsort_idx = torch.argsort(sort_idx)
        pred = pred_sorted[unsort_idx]                            # [F]

        # Return dummy flow_state for compatibility
        dummy_state = torch.zeros(n_f, self.hidden_dim, device=device)
        return pred, dummy_state
