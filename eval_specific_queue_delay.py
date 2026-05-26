import sys
import os
import argparse
import glob
import json
import torch
import numpy as np
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
from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario, filter_for_target

def main():
    parser = argparse.ArgumentParser(description="Evaluate specific Delay models for each scenario and queue_size combination.")
    parser.add_argument("--data-dir", default="data_cleaned", help="Data directory (e.g. Data_cleaned or Data)")
    parser.add_argument("--checkpoint-dir", default="checkpoints_queue", help="Checkpoints directory containing SC01_50KiB, etc.")
    parser.add_argument("--scenario", default=None, help="Scenario to evaluate (e.g., SC01 or SC03). Default: evaluate all available.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Discovering scenarios...")
    all_configs = discover_scenarios(_project_root, data_dir=args.data_dir, validate=True, verbose=False)
    groups = group_by_scenario(all_configs)

    results = []

    # Iterate through all scenario groups (SC01, SC02, etc.)
    for sc_id, cfgs in groups.items():
        if args.scenario and sc_id.upper() != args.scenario.upper():
            continue
        delay_cfgs = filter_for_target(cfgs, "delay")
        if not delay_cfgs:
            continue

        # Group configs by queue size
        queue_groups = {}
        for c in delay_cfgs:
            queue_groups.setdefault(c.queue_size, []).append(c)

        # Sort queue groups for clean output
        def parse_qsize(q):
            try:
                val = float(''.join(c for c in q if c.isdigit() or c == '.'))
                if 'MiB' in q: val *= 1024
                return val
            except: return 0.0

        sorted_qsizes = sorted(queue_groups.keys(), key=parse_qsize)

        for qsize in sorted_qsizes:
            q_cfgs = queue_groups[qsize]
            model_name = f"{sc_id}_{qsize}"
            ckpt_path = os.path.join(args.checkpoint_dir, model_name, "delay", "best.pt")
            
            if not os.path.exists(ckpt_path):
                continue

            print(f"\n{'='*70}")
            print(f"Evaluating Specific Model: {model_name} (Delay)")
            print(f"{'='*70}")

            model, arch_name, ckpt = load_model_from_checkpoint(ckpt_path, device)
            model.eval()

            # Build the specific dataset for this combination to get the test split
            data_paths = [c.data_path for c in q_cfgs]
            print(f"[{model_name}] Building dataset from {len(data_paths)} configs to get test split...")
            _, _, test_ds, normalizer = build_scenario_datasets(
                data_paths=data_paths,
                scenario_id=model_name,
                target="delay",
                seed=42,
                subsample_ratio=0.2,
                split_dir=os.path.dirname(ckpt_path)
            )

            # Check if normalizer is in the checkpoint
            if "normalizer" in ckpt:
                print(f"  [{model_name}] Loading normalizer stats from checkpoint.")
                normalizer.load_state(ckpt["normalizer"])
            else:
                print(f"  [{model_name}] Normalizer not in checkpoint, relying on dynamically built normalizer (WARNING: fragile!)")

            if len(test_ds) == 0:
                print(f"  [{model_name}] No test graphs found.")
                continue

            print(f"  [{model_name}] Test graphs: {len(test_ds)}")
            
            loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

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
                
                metrics = compute_metrics(pred, true)
                scale = 1000.0  # seconds to ms
                
                res_entry = {
                    "Scenario": sc_id,
                    "Queue Size": qsize,
                    "MAE": float(metrics["MAE"]),
                    "MedAE": float(metrics["Median AE"]),
                    "RMSE": float(metrics["RMSE"]),
                    "MAPE": float(metrics["MAPE (%)"]),
                    "SMAPE": float(metrics["SMAPE (%)"]),
                    "R2": float(metrics["R²"]),
                    "Bucket_MAE": metrics["Bucket_MAE"]
                }
                results.append(res_entry)
                
                print(f"    -> MAE: {metrics['MAE'] * scale:.2f} ms | MedAE: {metrics['Median AE'] * scale:.2f} ms | RMSE: {metrics['RMSE'] * scale:.2f} ms | SMAPE: {metrics['SMAPE (%)']:.2f}% | R²: {metrics['R²']:.4f}")
                
                print(f"    [Bucket MAE]")
                for b_name, b_val in metrics["Bucket_MAE"].items():
                    val_str = f"{b_val * scale:.2f} ms" if not np.isnan(b_val) else "N/A"
                    print(f"      {b_name:<10}: {val_str}")
                
                print(f"\n    [Sample Comparison for {model_name}]")
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
                print("\n")

    if results:
        print(f"\n{'='*115}")
        print(f"FINAL SUMMARY - Specific Queue-Size Delay Models Evaluation")
        print(f"{'='*115}")
        print(f"{'Scenario':<10} {'Queue Size':<12} {'MAE':>8} {'MedAE':>8} {'RMSE':>8} {'SMAPE':>8} {'R²':>8} | {'0-100ms':>10} {'100-500ms':>10} {'500-2k ms':>10} {'>2000ms':>10}")
        print("-" * 115)
        for r in results:
            b = r["Bucket_MAE"]
            b1 = f"{b['0-100ms']*1000:.1f}" if not np.isnan(b['0-100ms']) else "N/A"
            b2 = f"{b['100-500ms']*1000:.1f}" if not np.isnan(b['100-500ms']) else "N/A"
            b3 = f"{b['500-2000ms']*1000:.1f}" if not np.isnan(b['500-2000ms']) else "N/A"
            b4 = f"{b['>2000ms']*1000:.1f}" if not np.isnan(b['>2000ms']) else "N/A"
            
            print(f"{r['Scenario']:<10} {r['Queue Size']:<12} {r['MAE']*1000.0:>8.1f} {r['MedAE']*1000.0:>8.1f} {r['RMSE']*1000.0:>8.1f} {r['SMAPE']:>8.1f}% {r['R2']:>8.4f} | {b1:>10} {b2:>10} {b3:>10} {b4:>10}")
        
        # Save results to JSON
        out_file = "evaluation_delay_specific_queue.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {out_file}")
    else:
        print("\nNo models evaluated. Check your data-dir and checkpoint-dir arguments.")

if __name__ == "__main__":
    main()
