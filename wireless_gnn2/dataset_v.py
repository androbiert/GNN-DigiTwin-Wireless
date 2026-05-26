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

def extract_global_features(graph: dict) -> tuple:
    """
    Extract global average and max features for flow, queue, and link attributes.
    Returns (feat_array, target_delay, target_throughput)
    Input dim = 16 (flow) + 4 (queue) + 8 (link) + 3 (counts) = 31
    """
    def agg(arr):
        if len(arr) == 0:
            return np.zeros(arr.shape[1] * 2, dtype=np.float32)
        return np.concatenate([np.mean(arr, axis=0), np.max(arr, axis=0)])

    f_agg = agg(graph["flow_feat"])
    q_agg = agg(graph["queue_feat"])
    l_agg = agg(graph["link_feat"])
    counts = np.array([graph["n_flows"], graph["n_queues"], graph["n_links"]], dtype=np.float32)
    
    feat = np.concatenate([f_agg, q_agg, l_agg, counts])
    
    td = np.mean(graph["target_delay"]) if graph["n_flows"] > 0 else 0.0
    tt = np.mean(graph["target_throughput"]) if graph["n_flows"] > 0 else 0.0
    
    return feat, td, tt

def load_temporal_scenarios(data_paths, seq_len=8):
    all_sequences_x = []
    all_sequences_y_d = []
    all_sequences_y_t = []
    
    for path in data_paths:
        with open(path, "r") as f:
            data = json.load(f)
            
        # extract sequentially
        seq_x, seq_d, seq_t = [], [], []
        for snap in data:
            g = build_graph(snap)
            if g is not None:
                feat, td, tt = extract_global_features(g)
                seq_x.append(feat)
                seq_d.append(td)
                seq_t.append(tt)
                
        if len(seq_x) < seq_len:
            continue
            
        x = np.stack(seq_x)
        d = np.array(seq_d)
        t = np.array(seq_t)
        
        # Build windows (sliding window)
        windows_x, windows_d, windows_t = [], [], []
        for i in range(len(x) - seq_len + 1):
            windows_x.append(x[i:i+seq_len])
            # Target is the value at the LAST step of the window
            windows_d.append(d[i+seq_len-1])
            windows_t.append(t[i+seq_len-1])
            
        all_sequences_x.extend(windows_x)
        all_sequences_y_d.extend(windows_d)
        all_sequences_y_t.extend(windows_t)
        
    if not all_sequences_x:
        return None, None, None
        
    X = np.stack(all_sequences_x)  # [N, seq_len, dim]
    Y_d = np.array(all_sequences_y_d) # [N]
    Y_t = np.array(all_sequences_y_t) # [N]
    
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
