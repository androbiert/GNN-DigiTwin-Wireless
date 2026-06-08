"""
baseline_lstm.py — Simple LSTM Baseline for Throughput/Delay Prediction

A minimal single-layer unidirectional LSTM that predicts per-flow throughput
WITHOUT using any graph structure or message passing.

For each flow, the input is the concatenation of:
  - Flow features      (7 dims)
  - Queue features      (5 dims)
  - Link features       (6 dims)

Total input: 18 dims -> LSTM -> Linear -> 1 scalar

This serves as a simple baseline to demonstrate the superiority
of the GNN attention-based architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from wireless_gnn.model import FLOW_FEAT_DIM, QUEUE_FEAT_DIM, LINK_FEAT_DIM

BASELINE_INPUT_DIM = FLOW_FEAT_DIM + QUEUE_FEAT_DIM + LINK_FEAT_DIM  # 18


class BaselineLSTM(nn.Module):
    """
    Simple LSTM Baseline — no graph structure, no message passing, no attention.

    Parameters
    ----------
    hidden_dim : int    LSTM hidden size (default 64)
    target     : str    'delay' or 'throughput'
    """

    def __init__(
        self,
        hidden_dim:  int   = 64,
        target:      str   = 'throughput',
        # Unused kwargs for compatibility with train_scenarios.py interface
        num_heads:   int   = 4,
        iterations:  int   = 8,
        dropout:     float = 0.1,
        num_layers:  int   = 1,
    ):
        super().__init__()
        assert target in ('delay', 'throughput')
        self.target     = target
        self.hidden_dim = hidden_dim

        # Single-layer unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=BASELINE_INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Simple linear readout
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        graph:   dict,
        history: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        n_f = len(graph["flow_feat"])

        if n_f == 0:
            return torch.zeros(0, device=device), torch.zeros(0, self.hidden_dim, device=device)

        # Build flat input per flow: [flow || queue || link]
        flow_feat  = torch.tensor(np.asarray(graph["flow_feat"]),  dtype=torch.float32, device=device)
        queue_feat = torch.tensor(np.asarray(graph["queue_feat"]), dtype=torch.float32, device=device)
        link_feat  = torch.tensor(np.asarray(graph["link_feat"]),  dtype=torch.float32, device=device)
        f2q = torch.tensor(np.asarray(graph["flow_to_queue"]), dtype=torch.long, device=device)
        q2l = torch.tensor(np.asarray(graph["queue_to_link"]), dtype=torch.long, device=device)

        flat = torch.cat([flow_feat, queue_feat[f2q], link_feat[q2l[f2q]]], dim=-1)  # [F, 17]

        # LSTM: treat flows as a sequence [1, F, 14]
        lstm_out, _ = self.lstm(flat.unsqueeze(0))  # [1, F, hidden_dim]
        lstm_out = lstm_out.squeeze(0)               # [F, hidden_dim]

        pred = self.readout(lstm_out).squeeze(-1)    # [F]

        dummy_state = torch.zeros(n_f, self.hidden_dim, device=device)
        return pred, dummy_state
