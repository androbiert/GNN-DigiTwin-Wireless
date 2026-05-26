import torch
import torch.nn as nn

class ModelV(nn.Module):
    """
    ModelV: Pure Temporal Baseline (Non-GNN, Non-Attention)
    Uses an MLP per timestep to extract features from global snapshot statistics,
    followed by a GRU to aggregate temporal history, and an MLP readout.
    """
    def __init__(self, input_dim=31, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Step 1: Encode each timestep
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Step 2: Temporal aggregation (GRU)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Step 3: Readout from final hidden state
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch_size, seq_len, input_dim]
            
        Returns
        -------
        pred : torch.Tensor
            Shape: [batch_size]
        """
        B, S, D = x.size()
        
        # encode each timestep independently
        x_flat = x.view(B * S, D)
        encoded_flat = self.encoder(x_flat)
        encoded = encoded_flat.view(B, S, -1)  # [B, S, hidden_dim]
        
        # temporal rnn
        out, h_n = self.rnn(encoded)  # out: [B, S, hidden_dim]
        
        # take the last timestep's output to predict the target for the current window
        last_out = out[:, -1, :]  # [B, hidden_dim]
        
        pred = self.readout(last_out).squeeze(-1)  # [B]
        return pred
