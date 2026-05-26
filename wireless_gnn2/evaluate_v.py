import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario
from wireless_gnn2.dataset_v import build_temporal_datasets, GlobalFeatureNormalizer
from wireless_gnn2.model_v import ModelV

def compute_metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> dict:
    errors = pred - true
    abs_errors = np.abs(errors)
    
    mae  = np.mean(abs_errors)
    mse  = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    denom = np.abs(true) + eps
    mape  = np.mean(abs_errors / denom) * 100
    
    smape_denom = (np.abs(true) + np.abs(pred)) / 2.0 + eps
    smape = np.mean(abs_errors / smape_denom) * 100
    
    median_ae = np.median(abs_errors)
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + eps))
    
    buckets = {
        "0-100ms": (0.0, 0.1),
        "100-500ms": (0.1, 0.5),
        "500-2000ms": (0.5, 2.0),
        ">2000ms": (2.0, float("inf"))
    }
    mae_buckets = {}
    for name, (low, high) in buckets.items():
        mask = (true >= low) & (true < high)
        if np.any(mask):
            mae_buckets[name] = np.mean(abs_errors[mask])
        else:
            mae_buckets[name] = float("nan")
            
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape,
        "MedianAE": median_ae,
        "R2": r2,
        "Bucket_MAE": mae_buckets
    }

def evaluate_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    configs = discover_scenarios(".", data_dir=args.data_dir)
    groups = group_by_scenario(configs)
    
    if args.scenario not in groups:
        raise ValueError(f"Scenario {args.scenario} not found.")
        
    cfgs = groups[args.scenario]
    if args.queue_size:
        cfgs = [c for c in cfgs if c.queue_size == args.queue_size]
        
    data_paths = [c.data_path for c in cfgs]
    
    ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.scenario}_{args.queue_size or 'all'}_seq{args.seq_len}", args.target)
    
    _, _, test_ds, normalizer = build_temporal_datasets(
        data_paths=data_paths,
        target=args.target,
        seq_len=args.seq_len,
        split_dir=ckpt_dir
    )
    
    # Reload normalizer from disk if exists
    norm_path = os.path.join(ckpt_dir, "normalizer.json")
    if os.path.isfile(norm_path):
        with open(norm_path, "r") as f:
            normalizer.load_state_dict(json.load(f))
            
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    input_dim = test_ds.X.shape[2]
    print(f"Dynamically determined input dimension: {input_dim}")
    model = ModelV(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.0)
    
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y_batch.numpy())
            
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    
    if args.target == 'delay':
        all_pred = np.expm1(all_pred)
        all_pred = np.clip(all_pred, a_min=0.0, a_max=None)
        
    metrics = compute_metrics(all_pred, all_true)
    
    scale = 1000.0 if args.target == 'delay' else 1.0
    unit = "ms" if args.target == 'delay' else "bps"
    
    print(f"\n========================================")
    print(f"EVALUATION: {args.scenario} {args.queue_size}")
    print(f"========================================")
    print(f"MAE:       {metrics['MAE'] * scale:.2f} {unit}")
    print(f"MedAE:     {metrics['MedianAE'] * scale:.2f} {unit}")
    print(f"RMSE:      {metrics['RMSE'] * scale:.2f} {unit}")
    print(f"SMAPE:     {metrics['SMAPE']:.2f}%")
    print(f"R²:        {metrics['R2']:.4f}")
    
    if args.target == 'delay':
        print(f"\n[Bucket MAE]")
        for b_name, b_val in metrics["Bucket_MAE"].items():
            v = f"{b_val * scale:.2f} ms" if not np.isnan(b_val) else "N/A"
            print(f"  {b_name:<10}: {v}")
            
    print(f"\n[Sample Comparison]")
    print(f"{'Index':<8} {'GT':>12} {'Pred':>12} {'Abs Err':>14}")
    print("-" * 50)
    idxs = np.random.choice(len(all_true), min(20, len(all_true)), replace=False)
    for idx in idxs:
        gt = all_true[idx] * scale
        pr = all_pred[idx] * scale
        err = abs(gt - pr)
        print(f"{idx:<8} {gt:>12.3f} {pr:>12.3f} {err:>14.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_cleaned")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_model_v")
    parser.add_argument("--scenario", type=str, default="SC01")
    parser.add_argument("--queue-size", type=str, default=None)
    parser.add_argument("--target", type=str, default="delay")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    evaluate_model(args)
