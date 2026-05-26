import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from wireless_gnn.graph_builder import build_graph

class GlobalFeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        # x is [N, seq_len, dim]
        self.mean = np.mean(x, axis=(0, 1), keepdims=True)
        self.std = np.std(x, axis=(0, 1), keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.tolist() if self.mean is not None else None,
                "std": self.std.tolist() if self.std is not None else None}

    def load_state_dict(self, state):
        if state["mean"] is not None:
            self.mean = np.array(state["mean"])
            self.std = np.array(state["std"])

def extract_flow_features(g: dict, i: int) -> np.ndarray:
    """
    Extract concatenated flow-specific features: flow, queue, and link.
    Returns np.ndarray of shape [14] (8 flow + 2 queue + 4 link)
    """
    f_feat = g["flow_feat"][i]
    qi = g["flow_to_queue"][i]
    q_feat = g["queue_feat"][qi]
    li = g["queue_to_link"][qi]
    l_feat = g["link_feat"][li]
    return np.concatenate([f_feat, q_feat, l_feat])

def load_temporal_scenarios(data_paths, seq_len=8):
    all_sequences_x = []
    all_sequences_y_d = []
    all_sequences_y_t = []
    
    for path in data_paths:
        with open(path, "r") as f:
            data = json.load(f)
            
        flow_history = {}
        
        for t, snap in enumerate(data):
            g = build_graph(snap)
            if g is None:
                continue
                
            flows = snap.get("flows", [])
            active_flows = [
                f for f in flows
                if (f.get("delay", 0) > 0 or f.get("throughput", 0) > 0)
                and f.get("dst", "").startswith("ue")
            ]
            
            n_flows = g.get("n_flows", 0)
            if n_flows == 0 or len(active_flows) != n_flows:
                continue
                
            for i, f in enumerate(active_flows):
                dst = f["dst"]
                feat = extract_flow_features(g, i)
                td = g["target_delay"][i]
                tt = g["target_throughput"][i]
                
                if dst not in flow_history:
                    flow_history[dst] = []
                flow_history[dst].append((t, feat, td, tt))
                
        # Build sliding windows per individual flow (must be contiguous in time)
        for dst, history in flow_history.items():
            if len(history) < seq_len:
                continue
                
            for i in range(len(history) - seq_len + 1):
                window = history[i:i+seq_len]
                t_start = window[0][0]
                is_contiguous = True
                for idx, (t_val, _, _, _) in enumerate(window):
                    if t_val != t_start + idx:
                        is_contiguous = False
                        break
                        
                if is_contiguous:
                    x_win = np.stack([item[1] for item in window])  # [seq_len, 14]
                    y_d_val = window[-1][2]
                    y_t_val = window[-1][3]
                    
                    all_sequences_x.append(x_win)
                    all_sequences_y_d.append(y_d_val)
                    all_sequences_y_t.append(y_t_val)
                    
    if not all_sequences_x:
        return None, None, None
        
    X = np.stack(all_sequences_x)      # [N, seq_len, 14]
    Y_d = np.array(all_sequences_y_d)   # [N]
    Y_t = np.array(all_sequences_y_t)   # [N]
    
    return X, Y_d, Y_t

class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def build_temporal_datasets(data_paths, target='delay', seq_len=8, split_dir=None):
    print(f"[dataset_v] Loading temporal windows (seq_len={seq_len}) from {len(data_paths)} files...")
    X, Y_d, Y_t = load_temporal_scenarios(data_paths, seq_len)
    
    if X is None:
        raise ValueError("No valid sequences found in data_paths.")
        
    Y = Y_d if target == 'delay' else Y_t
    N = len(X)
    print(f"[dataset_v] Extracted {N} windows.")
    
    split_file = os.path.join(split_dir, "split.json") if split_dir else None
    
    if split_file and os.path.isfile(split_file):
        print(f"[dataset_v] Loading existing chronological split from {split_file}")
        with open(split_file, "r") as f:
            meta = json.load(f)
        train_idx = meta["train_idx"]
        val_idx = meta["val_idx"]
        test_idx = meta["test_idx"]
    else:
        print(f"[dataset_v] Creating NEW chronological split (70/15/15)")
        n_train = int(0.70 * N)
        n_val = int(0.15 * N)
        
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, N))
        
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
            with open(split_file, "w") as f:
                json.dump({
                    "n_total": N,
                    "seq_len": seq_len,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx
                }, f)
                
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    
    normalizer = GlobalFeatureNormalizer()
    normalizer.fit(X_train)
    
    X_train_norm = normalizer.transform(X_train)
    X_val_norm = normalizer.transform(X[val_idx])
    X_test_norm = normalizer.transform(X[test_idx])
    
    train_ds = SequenceDataset(X_train_norm, Y_train)
    val_ds = SequenceDataset(X_val_norm, Y[val_idx])
    test_ds = SequenceDataset(X_test_norm, Y[test_idx])
    
    return train_ds, val_ds, test_ds, normalizer
