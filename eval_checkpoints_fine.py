import sys
import os
import argparse
import glob
import json
import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from wireless_gnn.model import WirelessNetFermi
from wireless_gnn.dataset import build_scenario_datasets, collate_fn, WirelessDataset
from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario, filter_for_target
from evaluate_models import compute_metrics

def load_old_checkpoint(ckpt_path, device):
    """Load a checkpoint that might have different feature dimensions."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    hidden_dim = ckpt.get("hidden_dim", 64)
    num_heads  = ckpt.get("num_heads", 4)
    iterations = ckpt.get("iterations", 8)
    target     = ckpt.get("target", "throughput")
    state_dict = ckpt.get("model", ckpt)
    
    model = WirelessNetFermi(
        hidden_dim=hidden_dim, 
        num_heads=num_heads,
        iterations=iterations, 
        target=target
    )
    
    # Detect old feature dimensions from state_dict
    flow_dim  = state_dict["flow_embedding.0.weight"].shape[1]
    queue_dim = state_dict["queue_embedding.0.weight"].shape[1]
    link_dim  = state_dict["link_embedding.0.weight"].shape[1]
    
    print(f"  [Checkpoint] Detected old feature dims: flow={flow_dim}, queue={queue_dim}, link={link_dim}")
    
    # Patch the model's embeddings to match the checkpoint
    model.flow_embedding[0] = nn.Linear(flow_dim, hidden_dim)
    model.queue_embedding[0] = nn.Linear(queue_dim, hidden_dim)
    model.link_embedding[0] = nn.Linear(link_dim, hidden_dim)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, ckpt, (flow_dim, queue_dim, link_dim)

def adapt_graph(graph, old_dims):
    """Adapt graph features to match old dimensions expected by the checkpoint."""
    f_dim, q_dim, l_dim = old_dims
    new_graph = dict(graph)
    
    # Flow
    f_feat = np.array(new_graph["flow_feat"])
    if f_feat.shape[1] < f_dim:
        pad = np.zeros((f_feat.shape[0], f_dim - f_feat.shape[1]), dtype=f_feat.dtype)
        f_feat = np.concatenate([f_feat, pad], axis=1)
    elif f_feat.shape[1] > f_dim:
        f_feat = f_feat[:, :f_dim]
    new_graph["flow_feat"] = f_feat
    
    # Queue
    q_feat = np.array(new_graph["queue_feat"])
    if q_feat.shape[1] > q_dim:
        new_graph["queue_feat"] = q_feat[:, :q_dim]
    elif q_feat.shape[1] < q_dim:
        pad = np.zeros((q_feat.shape[0], q_dim - q_feat.shape[1]), dtype=q_feat.dtype)
        new_graph["queue_feat"] = np.concatenate([q_feat, pad], axis=1)

    # Link
    l_feat = np.array(new_graph["link_feat"])
    if l_feat.shape[1] > l_dim:
        new_graph["link_feat"] = l_feat[:, :l_dim]
    elif l_feat.shape[1] < l_dim:
        pad = np.zeros((l_feat.shape[0], l_dim - l_feat.shape[1]), dtype=l_feat.dtype)
        new_graph["link_feat"] = np.concatenate([l_feat, pad], axis=1)
        
    return new_graph

@torch.no_grad()
def predict_adapted(model, graph, normalizer, device, old_dims):
    model.eval()
    adapted_graph = adapt_graph(graph, old_dims)
    
    # Normalize after adapting dimensions
    if normalizer is not None:
        adapted_graph = normalizer.normalize(adapted_graph)
        
    pred, _ = model(adapted_graph)
    
    # Convert to physical values
    mean = torch.tensor(normalizer.tput_mean, device=device)
    std  = torch.tensor(normalizer.tput_std,  device=device)
    true = np.asarray(graph["target_throughput"])
    pred_phys = torch.clamp(pred * std + mean, min=0.0).cpu().numpy()
    
    return pred_phys, true

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine checkpoints with automatic feature dimension patching.")
    parser.add_argument("--data-dir", default="data_cleaned", help="Data directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints_fine", help="Checkpoints directory")
    parser.add_argument("--checkpoint-name", default="best.pt", help="Specific checkpoint file to load (e.g. epoch_008_mape_0.05.pt)")
    parser.add_argument("--scenario", default="SC03", help="Scenario to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Discovering scenarios...")
    all_configs = discover_scenarios(_project_root, data_dir=args.data_dir, validate=True, verbose=False)
    groups = group_by_scenario(all_configs)

    sc = args.scenario.upper()
    if sc not in groups:
        print(f"ERROR: Scenario '{sc}' not found.")
        sys.exit(1)
        
    cfgs = filter_for_target(groups[sc], "throughput")
    if not cfgs:
        print("No throughput configs found.")
        sys.exit(1)

    ckpt_path = os.path.join(args.checkpoint_dir, sc, "throughput", args.checkpoint_name)
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Evaluating General Model for {sc} (Throughput) with Adaptive Loading")
    print(f"Checkpoint File: {args.checkpoint_name}")
    print(f"{'='*70}")

    model, ckpt, old_dims = load_old_checkpoint(ckpt_path, device)

    data_paths = [c.data_path for c in cfgs]
    ckpt_dir = os.path.join(args.checkpoint_dir, sc, "throughput")
    print(f"[{sc}] Building full dataset...")
    
    _, _, full_test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id=sc,
        target="throughput",
        seed=42,
        split_dir=ckpt_dir,
    )

    if "normalizer" in ckpt:
        print(f"[{sc}] Loading normalizer from checkpoint.")
        normalizer.load_state(ckpt["normalizer"])
    else:
        print(f"[{sc}] WARNING: No normalizer in checkpoint, using recomputed one.")

    policy_folders = defaultdict(set)
    for c in cfgs:
        policy_folders[c.scheduler].add(c.folder_name)

    results = []
    
    for policy, folders in policy_folders.items():
        policy_graphs = [g for g in full_test_ds.graphs if g["config_folder"] in folders]
        if not policy_graphs:
            continue

        print(f"\n  [{policy}] Test graphs: {len(policy_graphs)}")
        
        # Disable normalizer here to avoid shape crashes; we normalize inside predict_adapted
        pol_test_ds = WirelessDataset(policy_graphs, normalizer=None)
        loader = DataLoader(pol_test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        all_pred, all_true = [], []
        
        for batch in loader:
            for graph in batch:
                pred_phys, true_phys = predict_adapted(model, graph, normalizer, device, old_dims)
                all_pred.append(pred_phys)
                all_true.append(true_phys)
        
        if all_pred:
            pred = np.concatenate(all_pred)
            true = np.concatenate(all_true)
            metrics = compute_metrics(pred, true)
            
            res_entry = {
                "Scenario": sc,
                "Policy": policy,
                "MAE": float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "MAPE": float(metrics["MAPE (%)"]),
                "R2": float(metrics["R²"])
            }
            results.append(res_entry)
            
            scale = 1e-3
            print(f"    -> MAE: {metrics['MAE'] * scale:.2f} kbps | RMSE: {metrics['RMSE'] * scale:.2f} kbps | MAPE: {metrics['MAPE (%)']:.2f}% | R²: {metrics['R²']:.4f}")

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - Throughput Evaluation Across Policies")
    print(f"{'='*70}")
    print(f"{'Scenario':<10} {'Policy':<12} {'MAE (kbps)':>12} {'RMSE (kbps)':>12} {'MAPE (%)':>10} {'R²':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['Scenario']:<10} {r['Policy']:<12} {r['MAE']*1e-3:>12.2f} {r['RMSE']*1e-3:>12.2f} {r['MAPE']:>10.2f} {r['R2']:>10.4f}")
    
if __name__ == "__main__":
    main()
