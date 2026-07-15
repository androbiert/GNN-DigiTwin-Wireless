"""
eval_student.py — Standalone Evaluation of Distilled WirelessNet-Fermi GNN Student Model

Loads a trained student model checkpoint, filters the data configuration to the test split
defined in split.json, runs inference, and computes performance and latency metrics.
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluate_models import (
    compute_metrics,
    predict_with_timing,
    plot_scatter_eval,
    plot_error_distribution,
    plot_residuals
)
from wireless_gnn.student_film import WirelessNetFermiStudent
from wireless_gnn.dataset import build_scenario_datasets, collate_fn
from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario, filter_for_target
torch.set_num_threads(4)
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    else:
        return obj

def load_student_model(ckpt_path: str, device: torch.device):
    """Loads the distilled student GNN model from checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
    print(f"Loading student checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Verify architecture
    arch = ckpt.get("architecture")
    if arch != "FiLM_Highway" and "student" not in ckpt:
        print(f"WARNING: Checkpoint architecture is '{arch}', which might not be a Student GNN.")
        
    config = ckpt.get("config", {})
    hidden_dim = config.get("student_dim", 32)
    iterations = config.get("student_iters", 3)
    target = config.get("target", "throughput")
    dropout = config.get("dropout", 0.1)
    state_dict = ckpt.get("student", ckpt)
    
    print(f"Instantiating WirelessNetFermiStudent(hidden_dim={hidden_dim}, iterations={iterations}, target='{target}')")
    model = WirelessNetFermiStudent(
        hidden_dim=hidden_dim,
        iterations=iterations,
        dropout=dropout,
        target=target
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, target, ckpt

def main():
    parser = argparse.ArgumentParser(description="Evaluate a distilled GNN student model.")
    parser.add_argument("--checkpoint", default="checkpoints_distilled_film_throughput/SC01/throughput/best.pt",
                        help="Path to the student model checkpoint best.pt")
    parser.add_argument("--data-dir", default="data_cleaned",
                        help="Data directory containing scenarios")
    parser.add_argument("--scenario", default="SC01",
                        help="Scenario ID to evaluate (e.g. SC01)")
    parser.add_argument("--output-dir", default="evaluation_results_student",
                        help="Directory to save plots and JSON summary")
    parser.add_argument("--device", default="auto",
                        help="Device to run evaluation: 'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Evaluation Device: {device}")
    
    # 1. Load model
    try:
        model, target, ckpt = load_student_model(args.checkpoint, device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
        
    # 2. Discover datasets
    print(f"Discovering scenario configs for {args.scenario} ({target})...")
    all_configs = discover_scenarios(_project_root, data_dir=args.data_dir, validate=True, verbose=False, use_cache=True, scenario_filter=args.scenario)
    groups = group_by_scenario(all_configs)
    
    if args.scenario not in groups:
        print(f"ERROR: Scenario '{args.scenario}' not found in {args.data_dir}")
        sys.exit(1)
        
    cfgs = filter_for_target(groups[args.scenario], target)
    if not cfgs:
        print(f"ERROR: No configuration files found for scenario {args.scenario} and target {target}")
        sys.exit(1)
        
    data_paths = [c.data_path for c in cfgs]
    print(f"Found {len(data_paths)} data files for {args.scenario} / {target}")
    
    # 3. Load dataset splits using split.json
    split_dir = os.path.dirname(args.checkpoint)
    print(f"Loading dataset splits using split.json under {split_dir}")
    _, _, test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id=args.scenario,
        target=target,
        seed=ckpt.get("config", {}).get("seed", 42),
        split_dir=split_dir
    )
    
    # Load normalizer if present in checkpoint
    if "normalizer" in ckpt:
        print("Loading normalizer parameters from checkpoint.")
        normalizer.load_state(ckpt["normalizer"])
    else:
        print("WARNING: Normalizer not found in checkpoint. Relying on dynamically computed normalizer.")
        
    print(f"Test split size: {len(test_ds)} snapshots")
    if len(test_ds) == 0:
        print("ERROR: Test split is empty.")
        sys.exit(1)
        
    # 4. Predict
    all_pred = []
    all_true = []
    all_times = []
    all_flows = []
    
    print("Running inference...")
    for i, graph in enumerate(test_ds):
        pred_phys, true_phys, elapsed, n_flows = predict_with_timing(
            model, graph, normalizer, device
        )
        all_pred.append(pred_phys)
        all_true.append(true_phys)
        all_times.append(elapsed)
        all_flows.append(n_flows)
        
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    
    # Compute metrics
    metrics = compute_metrics(pred, true)
    
    # Timing calculations
    times = np.array(all_times)
    flows = np.array(all_flows)
    metrics["Avg Inference (ms)"]  = np.mean(times) * 1000
    metrics["P95 Inference (ms)"]  = np.percentile(times, 95) * 1000
    metrics["Total Inference (s)"] = np.sum(times)
    metrics["Avg Flows/Graph"]     = np.mean(flows)
    
    # Print results
    scale = 1000.0 if target == "delay" else 1e-3
    unit  = "ms" if target == "delay" else "kbps"
    
    print(f"\n{'='*60}")
    print(f"  STUDENT EVALUATION SUMMARY — {args.scenario} / {target.upper()}")
    print(f"{'='*60}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Student Size   : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Test Flows     : {metrics['n_samples']:,}")
    print(f"  MAE            : {metrics['MAE'] * scale:.4f} {unit}")
    print(f"  RMSE           : {metrics['RMSE'] * scale:.4f} {unit}")
    print(f"  MAPE           : {metrics['MAPE (%)']:.2f}%")
    print(f"  Median AE      : {metrics['Median AE'] * scale:.4f} {unit}")
    print(f"  R² Score       : {metrics['R²']:.6f}")
    print(f"  Acc@10%        : {metrics['Acc@10%']:.2f}%")
    print(f"  Acc@20%        : {metrics['Acc@20%']:.2f}%")
    print(f"  Acc@50%        : {metrics['Acc@50%']:.2f}%")
    print(f"  Avg Inference  : {metrics['Avg Inference (ms)']:.2f} ms/graph")
    print(f"  P95 Inference  : {metrics['P95 Inference (ms)']:.2f} ms/graph")
    print(f"  Total Inference: {metrics['Total Inference (s)']:.2f} s")
    print(f"{'='*60}\n")
    
    # Print sample predictions
    print("  [Sample Predictions Table]")
    num_samples = min(15, len(true))
    sample_indices = np.random.choice(len(true), num_samples, replace=False)
    print(f"  {'Index':<8} {f'GT ({unit})':>12} {f'Pred ({unit})':>12} {f'Abs Err ({unit})':>14} {'Rel Err (%)':>12}")
    print("  " + "-"*62)
    for idx in sample_indices:
        gt_val = true[idx] * scale
        pred_val = pred[idx] * scale
        abs_err = abs(gt_val - pred_val)
        rel_err = (abs_err / (abs(gt_val) + 1e-6)) * 100
        print(f"  {idx:<8} {gt_val:>12.3f} {pred_val:>12.3f} {abs_err:>14.3f} {rel_err:>12.2f}%")
    print()
    
    # Save plots
    plot_dir = os.path.join(args.output_dir, f"{args.scenario}_{target}")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Generating evaluation plots in {plot_dir}...")
    plot_scatter_eval(pred, true, "Student", target, plot_dir)
    plot_error_distribution(pred, true, "Student", target, plot_dir)
    plot_residuals(pred, true, "Student", target, plot_dir)
    
    # Save JSON summary
    summary_path = os.path.join(plot_dir, "summary.json")
    with open(summary_path, "w") as f:
        summary_data = {
            "checkpoint": args.checkpoint,
            "scenario": args.scenario,
            "target": target,
            "parameters": sum(p.numel() for p in model.parameters()),
            "metrics": make_json_serializable(metrics)
        }
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON saved to {summary_path}")
    print(" Done!")

if __name__ == "__main__":
    main()
