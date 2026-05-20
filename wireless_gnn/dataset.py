"""
dataset.py — WirelessNet-Fermi Dataset (Scenario-Aware)

Supports two modes:
  1. Legacy: loads from hardcoded SCENARIO_DIRS (backward compatible)
  2. Scenario-aware: loads from a list of SimConfig objects from scenario_registry

Folder names (relative to project root):
  Data/SC01/simulations/01)SC01-P=0.01W-S=PF-Q=50KiB/data.json
  Data/SC02/simulations/...
"""

import json
import os
import numpy as np
from typing import Optional, List, Tuple
from torch.utils.data import Dataset

from wireless_gnn.graph_builder import build_graph


# --------------------------------------------------------------------------- #
# Statistics (z-score normalisation, computed on the fly from training set)
# --------------------------------------------------------------------------- #

class FeatureNormalizer:
    """Collects mean/std across all training samples, then normalises."""

    def __init__(self):
        self._flow_vals   = []
        self._queue_vals  = []
        self._link_vals   = []
        self._delay_vals  = []
        self._tput_vals   = []
        self.fitted = False
        self.log_delay = True   # delay targets are log1p-transformed

        # Will be set after fit()
        self.flow_mean  = self.flow_std  = None
        self.queue_mean = self.queue_std = None
        self.link_mean  = self.link_std  = None
        self.delay_mean = self.delay_std = None  # mean/std of log1p(delay)
        self.tput_mean  = self.tput_std  = None

    def accumulate(self, graph: dict):
        self._flow_vals.append(graph["flow_feat"])
        self._queue_vals.append(graph["queue_feat"])
        self._link_vals.append(graph["link_feat"])
        # Store log1p(delay) for normalization — compresses multi-order-of-magnitude range
        self._delay_vals.append(np.log1p(graph["target_delay"].astype(np.float64)).astype(np.float32))
        self._tput_vals.append(graph["target_throughput"])

    def fit(self, eps: float = 1e-8):
        def _ms(lst):
            arr = np.concatenate(lst, axis=0)
            m   = arr.mean(axis=0)
            s   = arr.std(axis=0)
            s   = np.where(s < eps, 1.0, s)   # avoid division by zero
            return m.astype(np.float32), s.astype(np.float32)

        self.flow_mean,  self.flow_std  = _ms(self._flow_vals)
        self.queue_mean, self.queue_std = _ms(self._queue_vals)
        self.link_mean,  self.link_std  = _ms(self._link_vals)
        self.delay_mean, self.delay_std = _ms(self._delay_vals)
        self.tput_mean,  self.tput_std  = _ms(self._tput_vals)
        self.fitted = True

    def normalize(self, graph: dict) -> dict:
        assert self.fitted, "Call fit() first."
        g = dict(graph)
        g["flow_feat"]         = (graph["flow_feat"]   - self.flow_mean)  / self.flow_std
        g["queue_feat"]        = (graph["queue_feat"]  - self.queue_mean) / self.queue_std
        g["link_feat"]         = (graph["link_feat"]   - self.link_mean)  / self.link_std
        # Targets: normalise for loss, but keep originals for metric reporting
        log_delay = np.log1p(graph["target_delay"].astype(np.float64)).astype(np.float32)
        g["target_delay_norm"]     = (log_delay - self.delay_mean) / self.delay_std
        g["target_throughput_norm"]= (graph["target_throughput"] - self.tput_mean)  / self.tput_std
        return g


# --------------------------------------------------------------------------- #
# Legacy file loading (backward compatible)
# --------------------------------------------------------------------------- #

# Folder names that contain a data.json (legacy — kept for backward compat)
SCENARIO_DIRS = [
    "01)SC-01-P=0.01,Sch=PF,Qs=100KiB",
    "02)SC01-P=0.01,S=PF,Q=2MiB",
    "03)SC01-P=0.01,S=PF,Q=10MiB",
]


def _find_data_files(project_root: str) -> List[str]:
    """Return absolute paths to all data.json files that exist (legacy mode)."""
    found = []
    for sdir in SCENARIO_DIRS:
        path = os.path.join(project_root, sdir, "data.json")
        if os.path.isfile(path):
            found.append(path)
        else:
            print(f"[dataset] WARNING: not found -> {path}")
    return found


def load_all_snapshots(project_root: str) -> List[dict]:
    """
    Load every valid graph snapshot from all three scenario data.json files.
    Returns a list of raw graph dicts (not yet normalised).
    
    LEGACY MODE — for backward compatibility with train.py.
    """
    files = _find_data_files(project_root)
    if not files:
        raise FileNotFoundError(
            f"No data.json found under {project_root}. "
            f"Expected subdirs: {SCENARIO_DIRS}"
        )

    all_graphs = []
    for fpath in files:
        scenario_name = os.path.basename(os.path.dirname(fpath))
        print(f"[dataset] Loading {scenario_name} ...")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        n_valid = 0
        for snapshot in data:
            g = build_graph(snapshot)
            if g is not None:
                g["scenario"] = scenario_name
                all_graphs.append(g)
                n_valid += 1

        print(f"[dataset]   -> {n_valid}/{len(data)} valid snapshots")

    print(f"[dataset] Total samples before filtering: {len(all_graphs)}")
    
    # --- Delete outlier snapshots (max delay > 95th percentile) ---
    all_delays = []
    for g in all_graphs:
        all_delays.extend(g["target_delay"])
    
    if all_delays:
        threshold_95 = float(np.percentile(all_delays, 95))
        print(f"[dataset] 95th percentile delay threshold: {threshold_95:.5f}s")
        
        filtered_graphs = []
        for g in all_graphs:
            if max(g["target_delay"]) <= threshold_95:
                filtered_graphs.append(g)
        
        print(f"[dataset] Filtered out {len(all_graphs) - len(filtered_graphs)} outlier snapshots.")
        all_graphs = filtered_graphs
    
    print(f"[dataset] Total samples after filtering: {len(all_graphs)}")
    return all_graphs


# --------------------------------------------------------------------------- #
# Scenario-aware file loading  (NEW)
# --------------------------------------------------------------------------- #

def load_scenario_snapshots(
    data_paths: List[str],
    scenario_id: str = "",
    target: str = "delay",
    filter_outliers: bool = True,
    outlier_percentile: float = 95.0,
    verbose: bool = True,
) -> List[dict]:
    """
    Load all valid graph snapshots from a list of data.json paths.

    Parameters
    ----------
    data_paths : List[str]
        Absolute paths to data.json files.
    scenario_id : str
        Scenario identifier (e.g. "SC01") — stored as metadata on each graph.
    target : str
        "delay" or "throughput" — used for outlier filtering.
    filter_outliers : bool
        If True, remove snapshots with extreme target values.
    outlier_percentile : float
        Percentile threshold for outlier filtering.
    verbose : bool
        Print loading progress.

    Returns
    -------
    List[dict]
        List of raw graph dicts (not yet normalised).
    """
    all_graphs = []

    for fpath in data_paths:
        folder_name = os.path.basename(os.path.dirname(fpath))
        if verbose:
            print(f"[dataset] Loading {folder_name} ...")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[dataset] WARNING: cannot read {fpath}: {e}")
            continue

        n_valid = 0
        for snapshot in data:
            g = build_graph(snapshot)
            if g is not None:
                g["scenario"] = scenario_id
                g["config_folder"] = folder_name
                all_graphs.append(g)
                n_valid += 1

        if verbose:
            print(f"[dataset]   -> {n_valid}/{len(data)} valid snapshots")

    if verbose:
        print(f"[dataset] Total samples before filtering: {len(all_graphs)}")

    # --- Outlier filtering ---
    if filter_outliers and all_graphs:
        if target == "delay":
            all_targets = []
            for g in all_graphs:
                all_targets.extend(g["target_delay"])
        else:
            all_targets = []
            for g in all_graphs:
                all_targets.extend(g["target_throughput"])

        if all_targets:
            threshold = float(np.percentile(all_targets, outlier_percentile))
            if verbose:
                print(f"[dataset] {outlier_percentile}th percentile threshold: {threshold:.5f}")

            before = len(all_graphs)
            if target == "delay":
                all_graphs = [g for g in all_graphs if max(g["target_delay"]) <= threshold]
            else:
                # For throughput, remove extreme high values
                all_graphs = [g for g in all_graphs if max(g["target_throughput"]) <= threshold]

            if verbose:
                print(f"[dataset] Filtered out {before - len(all_graphs)} outlier snapshots.")

    if verbose:
        print(f"[dataset] Total samples after filtering: {len(all_graphs)}")

    return all_graphs


# --------------------------------------------------------------------------- #
# PyTorch Dataset
# --------------------------------------------------------------------------- #

class WirelessDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-built (and optionally normalised) graph dicts.
    Each item is a dict ready for WirelessNetFermi.forward().
    """

    def __init__(self, graphs: List[dict], normalizer: Optional[FeatureNormalizer] = None):
        self.graphs     = graphs
        self.normalizer = normalizer

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        g = self.graphs[idx]
        if self.normalizer is not None:
            g = self.normalizer.normalize(g)
        return g


# --------------------------------------------------------------------------- #
# Build datasets — Legacy (backward compatible)
# --------------------------------------------------------------------------- #

def build_datasets(
    project_root: str,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple["WirelessDataset", "WirelessDataset", "WirelessDataset", FeatureNormalizer]:
    """
    Load all snapshots, split into train/val/test, fit normaliser on training set.
    LEGACY MODE — backward compatible with original train.py.

    Returns (train_dataset, val_dataset, test_dataset, normalizer).
    """
    all_graphs = load_all_snapshots(project_root)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_graphs))

    n_train = int(len(all_graphs) * train_ratio)
    n_val   = int(len(all_graphs) * val_ratio)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs   = [all_graphs[i] for i in val_idx]
    test_graphs  = [all_graphs[i] for i in test_idx]

    # Fit normaliser ONLY on training data
    norm = FeatureNormalizer()
    for g in train_graphs:
        norm.accumulate(g)
    norm.fit()

    print(f"[dataset] Split -> train={len(train_graphs)}, "
          f"val={len(val_graphs)}, test={len(test_graphs)}")

    return (
        WirelessDataset(train_graphs, norm),
        WirelessDataset(val_graphs,   norm),
        WirelessDataset(test_graphs,  norm),
        norm,
    )


# --------------------------------------------------------------------------- #
# Build datasets — Scenario-aware  (NEW)
# --------------------------------------------------------------------------- #

def build_scenario_datasets(
    data_paths:   List[str],
    scenario_id:  str   = "",
    target:       str   = "delay",
    train_ratio:  float = 0.7,
    val_ratio:    float = 0.15,
    seed:         int   = 42,
    filter_outliers:    bool  = True,
    outlier_percentile: float = 95.0,
) -> Tuple["WirelessDataset", "WirelessDataset", "WirelessDataset", FeatureNormalizer]:
    """
    Load snapshots from given data_paths, split, normalise.

    Parameters
    ----------
    data_paths : List[str]
        Absolute paths to data.json files for this scenario.
    scenario_id : str
        Scenario name (e.g. "SC01").
    target : str
        "delay" or "throughput" — affects outlier filtering.
    train_ratio, val_ratio : float
        Split ratios.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset, normalizer)
    """
    all_graphs = load_scenario_snapshots(
        data_paths=data_paths,
        scenario_id=scenario_id,
        target=target,
        filter_outliers=filter_outliers,
        outlier_percentile=outlier_percentile,
    )

    if not all_graphs:
        raise ValueError(
            f"No valid graphs loaded for {scenario_id} (target={target}). "
            f"Checked {len(data_paths)} data.json files."
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_graphs))

    n_train = int(len(all_graphs) * train_ratio)
    n_val   = int(len(all_graphs) * val_ratio)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs   = [all_graphs[i] for i in val_idx]
    test_graphs  = [all_graphs[i] for i in test_idx]

    # Fit normaliser ONLY on training data
    norm = FeatureNormalizer()
    for g in train_graphs:
        norm.accumulate(g)
    norm.fit()

    print(f"[dataset] {scenario_id} split -> train={len(train_graphs)}, "
          f"val={len(val_graphs)}, test={len(test_graphs)}")

    return (
        WirelessDataset(train_graphs, norm),
        WirelessDataset(val_graphs,   norm),
        WirelessDataset(test_graphs,  norm),
        norm,
    )


# --------------------------------------------------------------------------- #
# Collate function (variable-size graphs -> list batching)
# --------------------------------------------------------------------------- #

def collate_fn(batch: List[dict]) -> List[dict]:
    """
    Since each graph has a different number of nodes and edges,
    we keep the batch as a Python list (no padding needed).
    WirelessNetFermi processes each graph independently and stacks the losses.
    """
    return batch
