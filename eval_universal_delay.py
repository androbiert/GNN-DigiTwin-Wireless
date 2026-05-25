import sys
import os
import argparse
import glob
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluate_models import (
    load_model_from_checkpoint, 
    compute_metrics, 
    predict_with_timing
)
from wireless_gnn.dataset import build_scenario_datasets, collate_fn, WirelessDataset
from wireless_gnn.scenario_registry import discover_scenarios, filter_for_target

def main():
    parser = argparse.ArgumentParser(description="Evaluate Universal Delay model across all scenarios and policies.")
    parser.add_argument("--data-dir", default="data", help="Data directory (e.g. Data)")
    parser.add_argument("--checkpoint-path", default=os.path.join("wireless_gnn", "checkpoints", "delay", "best.pt"), help="Path to universal delay checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover scenarios and configs
    print("Discovering scenarios...")
    all_configs = discover_scenarios(_project_root, data_dir=args.data_dir, validate=True, verbose=False)
    
    # Filter for delay target
    delay_cfgs = filter_for_target(all_configs, "delay")
    if not delay_cfgs:
        print("No configs found for target 'delay'.")
        return

    print(f"Found {len(delay_cfgs)} delay configs across {len(set(c.scenario_id for c in delay_cfgs))} scenarios.")

    if not os.path.exists(args.checkpoint_path):
        print(f"No checkpoint found at {args.checkpoint_path}")
        return

    print(f"\n{'='*70}")
    print(f"Evaluating Universal Model: {args.checkpoint_path}")
    print(f"{'='*70}")

    # Load the universal model
    model, arch_name, ckpt = load_model_from_checkpoint(args.checkpoint_path, device)
    model.eval()

    # Build the full universal dataset to get the exact normalizer and test split as training
    data_paths = [c.data_path for c in delay_cfgs]
    print(f"Building full dataset from {len(data_paths)} configs to get general normalizer & test split...")
    
    # We pass scenario_id="ALL" so it knows it's a combined dataset
    _, _, full_test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id="ALL",
        target="delay",
        seed=42
    )

    # Check if normalizer is in the checkpoint
    if "normalizer" in ckpt:
        print(f"Loading normalizer stats from checkpoint.")
        normalizer.load_state(ckpt["normalizer"])
    else:
        print(f"Normalizer not in checkpoint, relying on dynamically built normalizer (WARNING: fragile!)")

    # Group test graphs by Scenario and Policy
    folder_to_policy = {c.folder_name: c.scheduler for c in delay_cfgs}
    
    grouped_graphs = defaultdict(list)
    for g in full_test_ds.graphs:
        sc = g["scenario"]
        folder = g["config_folder"]
        policy = folder_to_policy.get(folder, "UNKNOWN")
        grouped_graphs[(sc, policy)].append(g)

    results = []

    # Sort keys for consistent display
    sorted_keys = sorted(grouped_graphs.keys(), key=lambda x: (x[0], x[1]))

    # Evaluate on each (Scenario, Policy) combination
    for sc, policy in sorted_keys:
        graphs = grouped_graphs[(sc, policy)]
        print(f"\n  [{sc} | {policy}] Test graphs: {len(graphs)}")
        
        if len(graphs) == 0:
            continue
            
        pol_test_ds = WirelessDataset(graphs, normalizer=normalizer)
        loader = DataLoader(pol_test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        all_pred = []
        all_true = []

        with torch.no_grad():
            for batch in loader:
                for graph in batch:
                    pred_phys, true_phys, _, _ = predict_with_timing(model, graph, normalizer, device)
                    all_pred.append(pred_phys)
                    all_true.append(true_phys)
        
        if all_pred:
            pred = np.concatenate(all_pred)
            true = np.concatenate(all_true)
            
            # Predict returns in physical values. Compute metrics based on these.
            metrics = compute_metrics(pred, true)
            
            # For delay, we usually display in ms, so multiply by 1000
            scale = 1000.0
            
            res_entry = {
                "Scenario": sc,
                "Policy": policy,
                "MAE": float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "MAPE": float(metrics["MAPE (%)"]),
                "R2": float(metrics["R²"])
            }
            results.append(res_entry)
            
            print(f"    -> MAE: {metrics['MAE'] * scale:.2f} ms | RMSE: {metrics['RMSE'] * scale:.2f} ms | MAPE: {metrics['MAPE (%)']:.2f}% | R²: {metrics['R²']:.4f}")
            
            print(f"    [Sample Comparison for {sc} - {policy}]")
            num_samples_to_print = min(30, len(true))
            sample_indices = np.random.choice(len(true), num_samples_to_print, replace=False)
            print(f"    {'Index':<8} {'GT (ms)':>12} {'Pred (ms)':>12} {'Abs Err (ms)':>14} {'Rel Err (%)':>12}")
            print("    " + "-"*62)
            for idx in sample_indices:
                gt_val = true[idx] * scale
                pred_val = pred[idx] * scale
                abs_err = abs(gt_val - pred_val)
                rel_err = (abs_err / (abs(gt_val) + 1e-6)) * 100
                print(f"    {idx:<8} {gt_val:>12.3f} {pred_val:>12.3f} {abs_err:>14.3f} {rel_err:>12.2f}%")

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - Universal Delay Model Evaluation Across All Combinations")
    print(f"{'='*70}")
    print(f"{'Scenario':<10} {'Policy':<12} {'MAE (ms)':>12} {'RMSE (ms)':>12} {'MAPE (%)':>10} {'R²':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['Scenario']:<10} {r['Policy']:<12} {r['MAE']*1000.0:>12.2f} {r['RMSE']*1000.0:>12.2f} {r['MAPE']:>10.2f} {r['R2']:>10.4f}")
    
    # Save results to JSON
    out_file = "evaluation_delay_universal.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
