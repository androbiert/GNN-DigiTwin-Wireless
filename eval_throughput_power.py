"""
eval_throughput_power.py — Evaluate best model per scenario on each emission power level for throughput.

Groups test data by tx_power (e.g. 0.01W, 0.1W, 0.5W, 2W) and evaluates
the general per-scenario model on each power group separately.

Usage:
  python eval_throughput_power.py
  python eval_throughput_power.py --scenario SC01
  python eval_throughput_power.py --checkpoint-dir checkpoints_v3
"""

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
from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario, filter_for_target

def main():
    parser = argparse.ArgumentParser(description="Evaluate best model per scenario on each emission power level for throughput.")
    parser.add_argument("--data-dir", default="data_cleaned", help="Data directory (e.g. Data_cleaned)")
    parser.add_argument("--checkpoint-dir", default="checkpoints_v3", help="Checkpoints directory")
    parser.add_argument("--recache", action="store_true", help="Force re-scanning of scenarios (ignore cache)")
    parser.add_argument("--scenario", default=None, help="Evaluate only this scenario (e.g. SC03). Default: all.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover scenarios and configs
    print("Discovering scenarios...")
    all_configs = discover_scenarios(_project_root, data_dir=args.data_dir, validate=True, verbose=False, use_cache=not args.recache)
    groups = group_by_scenario(all_configs)

    # Filter to requested scenario if specified
    if args.scenario:
        sc = args.scenario.upper()
        if sc not in groups:
            print(f"ERROR: Scenario '{sc}' not found. Available: {list(groups.keys())}")
            sys.exit(1)
        groups = {sc: groups[sc]}

    results = []

    # Iterate through all scenario groups (SC01, SC02, etc.)
    for sc_id, cfgs in groups.items():
        # Find throughput configs
        tgt_cfgs = filter_for_target(cfgs, "throughput")
        if not tgt_cfgs:
            continue

        # Check if we have a general model checkpoint for this scenario
        ckpt_path = os.path.join(args.checkpoint_dir, sc_id, "throughput", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint found for {sc_id}/throughput at {ckpt_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating General Model for {sc_id} (Throughput by Emission Power)")
        print(f"{'='*70}")

        # Load the general model
        model, arch_name, ckpt = load_model_from_checkpoint(ckpt_path, device)
        model.eval()

        # Build the full dataset to get the test split
        data_paths = [c.data_path for c in tgt_cfgs]
        ckpt_dir = os.path.join(args.checkpoint_dir, sc_id, "throughput")
        print(f"[{sc_id}] Building full dataset from {len(data_paths)} configs to get test split...")
        _, _, full_test_ds, normalizer = build_scenario_datasets(
            data_paths=data_paths,
            scenario_id=sc_id,
            target="throughput",
            seed=42,
            split_dir=ckpt_dir,
        )

        # Override normalizer with the one from the checkpoint (critical!)
        if "normalizer" in ckpt:
            print(f"[{sc_id}] Loading normalizer from checkpoint (matches training).")
            normalizer.load_state(ckpt["normalizer"])
        else:
            print(f"[{sc_id}] WARNING: No normalizer in checkpoint, using recomputed one (may cause errors).")

        # Now group the configs by emission power (tx_power)
        power_folders = defaultdict(set)
        for c in tgt_cfgs:
            power_folders[c.tx_power].add(c.folder_name)

        # Evaluate on each power level
        for power, folders in sorted(power_folders.items(), key=lambda x: float(x[0].replace("W", ""))):
            # Filter the test dataset for this power level
            power_graphs = [g for g in full_test_ds.graphs if g["config_folder"] in folders]

            if not power_graphs:
                print(f"  [P={power}] No test graphs found.")
                continue

            print(f"  [P={power}] Test graphs: {len(power_graphs)}")

            # Predict
            all_pred = []
            all_true = []

            pol_test_ds = WirelessDataset(power_graphs, normalizer=normalizer)
            loader = DataLoader(pol_test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

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

                res_entry = {
                    "Scenario": sc_id,
                    "Power": power,
                    "MAE": float(metrics["MAE"]),
                    "RMSE": float(metrics["RMSE"]),
                    "MAPE": float(metrics["MAPE (%)"]),
                    "R2": float(metrics["R²"]),
                }
                results.append(res_entry)

                scale = 1e-3
                print(f"    -> MAE: {metrics['MAE'] * scale:.2f} kbps | RMSE: {metrics['RMSE'] * scale:.2f} kbps | MAPE: {metrics['MAPE (%)']:.2f}% | R²: {metrics['R²']:.4f}")

                print(f"    [Sample Comparison for P={power}]")
                num_samples_to_print = min(30, len(true))
                sample_indices = np.random.choice(len(true), num_samples_to_print, replace=False)
                print(f"    {'Index':<8} {'GT (kbps)':>12} {'Pred (kbps)':>12} {'Abs Err (kbps)':>14} {'Rel Err (%)':>12}")
                print("    " + "-"*62)
                for idx in sample_indices:
                    gt_val = true[idx] * scale
                    pred_val = pred[idx] * scale
                    abs_err = abs(gt_val - pred_val)
                    rel_err = (abs_err / (abs(gt_val) + 1e-6)) * 100
                    print(f"    {idx:<8} {gt_val:>12.3f} {pred_val:>12.3f} {abs_err:>14.3f} {rel_err:>12.2f}%")
                print("\n")

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - Throughput Evaluation Across Emission Power Levels")
    print(f"{'='*70}")
    print(f"{'Scenario':<10} {'Power':<12} {'MAE (kbps)':>12} {'RMSE (kbps)':>12} {'MAPE (%)':>10} {'R²':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['Scenario']:<10} {r['Power']:<12} {r['MAE']*1e-3:>12.2f} {r['RMSE']*1e-3:>12.2f} {r['MAPE']:>10.2f} {r['R2']:>10.4f}")

    # Save results to JSON
    out_file = "evaluation_throughput_power.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
