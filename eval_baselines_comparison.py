"""
eval_baselines_comparison.py — Compare per-scenario GNN vs single MLP/LSTM baselines

The GNN has one model per scenario (checkpoints/SC01/throughput/best.pt).
The baselines have ONE model trained on all scenarios (checkpoints_baseline/ALL/throughput/best.pt).

This script evaluates all three on the SAME test set per scenario, prints
a side-by-side comparison table, and shows 10 sample predictions per scenario.

Usage:
  python eval_baselines_comparison.py
  python eval_baselines_comparison.py --scenario SC01
  python eval_baselines_comparison.py --target delay
"""

import sys
import os
import argparse
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
    predict_with_timing,
)
from wireless_gnn.dataset import build_scenario_datasets, collate_fn, WirelessDataset, FeatureNormalizer
from wireless_gnn.scenario_registry import (
    discover_scenarios, group_by_scenario, filter_for_target,
)
from wireless_gnn.baseline_mlp import BaselineMLP
from wireless_gnn.baseline_lstm import BaselineLSTM


# --------------------------------------------------------------------------- #
# Model loading for baselines
# --------------------------------------------------------------------------- #

def load_baseline_model(ckpt_path: str, model_class, device: torch.device):
    """Load a baseline model (MLP or LSTM) from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hidden_dim = ckpt.get("hidden_dim", 128)
    target     = ckpt.get("target", "throughput")

    state_dict = ckpt.get("model", ckpt)

    model = model_class(
        hidden_dim=hidden_dim,
        target=target,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, model_class.__name__, ckpt


def collect_predictions(model, test_ds, normalizer, device):
    """Run model on test set and return (all_pred, all_true) as flat arrays."""
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            for graph in batch:
                try:
                    pred_phys, true_phys, _, _ = predict_with_timing(
                        model, graph, normalizer, device
                    )
                    all_pred.append(pred_phys)
                    all_true.append(true_phys)
                except Exception:
                    continue

    if not all_pred:
        return None, None

    return np.concatenate(all_pred), np.concatenate(all_true)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Compare per-scenario GNN vs single MLP/LSTM baselines."
    )
    parser.add_argument("--data-dir", default="data_cleaned")
    parser.add_argument("--gnn-checkpoint-dir", default="checkpoints",
                        help="GNN checkpoint directory (per-scenario models)")
    parser.add_argument("--mlp-checkpoint-dir", default="checkpoints_baseline",
                        help="MLP baseline checkpoint directory (single 'ALL' model)")
    parser.add_argument("--lstm-checkpoint-dir", default="checkpoints_baseline_lstm",
                        help="LSTM baseline checkpoint directory (single 'ALL' model)")
    parser.add_argument("--target", default="throughput", choices=["delay", "throughput"])
    parser.add_argument("--scenario", default=None,
                        help="Evaluate only this scenario (e.g., SC01)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of sample predictions to show per scenario")
    parser.add_argument("--recache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {args.target}")

    if args.target == "throughput":
        scale, unit = 1e-3, "kbps"
    else:
        scale, unit = 1000.0, "ms"

    # ── Discover scenarios ────────────────────────────────────────────────── #
    print("Discovering scenarios...")
    all_configs = discover_scenarios(
        _project_root, data_dir=args.data_dir, validate=True,
        verbose=False, use_cache=not args.recache
    )
    groups = group_by_scenario(all_configs)

    # We will load baseline models dynamically per-scenario to allow fair comparison
    # (checking for SC-specific checkpoints first, then falling back to ALL)

    # ── Evaluate per scenario ─────────────────────────────────────────────── #
    all_results = []

    for sc_id, cfgs in groups.items():
        if args.scenario and sc_id.upper() != args.scenario.upper():
            continue

        tgt_cfgs = filter_for_target(cfgs, args.target)
        if not tgt_cfgs:
            continue

        # Check GNN checkpoint
        gnn_ckpt_path = os.path.join(args.gnn_checkpoint_dir, sc_id, args.target, "best.pt")
        if not os.path.exists(gnn_ckpt_path):
            print(f"\n[{sc_id}] No GNN checkpoint at {gnn_ckpt_path}, skipping.")
            continue

        print(f"\n{'='*90}")
        print(f"  SCENARIO: {sc_id} — {args.target.upper()} COMPARISON")
        print(f"{'='*90}")

        # Build test dataset using the GNN's split
        data_paths = [c.data_path for c in tgt_cfgs]
        gnn_split_dir = os.path.join(args.gnn_checkpoint_dir, sc_id, args.target)

        print(f"  Building dataset ({len(data_paths)} configs, GNN split)...")
        _, _, test_ds, normalizer = build_scenario_datasets(
            data_paths=data_paths,
            scenario_id=sc_id,
            target=args.target,
            seed=42,
            split_dir=gnn_split_dir,
        )

        print(f"  Test set: {len(test_ds)} graphs")
        if len(test_ds) == 0:
            continue

        scenario_results = []
        # Store predictions for sample comparison: { model_name: (pred, true) }
        model_predictions = {}

        # ── 1. Evaluate GNN (per-scenario model) ─────────────────────────── #
        print(f"\n  [GNN (Ours)] Loading per-scenario model...")
        try:
            gnn_model, gnn_arch, gnn_ckpt_data = load_model_from_checkpoint(gnn_ckpt_path, device)
            gnn_model.eval()
            gnn_params = sum(p.numel() for p in gnn_model.parameters())

            # Use GNN's normalizer
            gnn_normalizer = FeatureNormalizer()
            gnn_normalizer.load_state(normalizer.get_state())  # copy base
            if "normalizer" in gnn_ckpt_data:
                gnn_normalizer.load_state(gnn_ckpt_data["normalizer"])

            test_ds.normalizer = gnn_normalizer
            gnn_pred, gnn_true = collect_predictions(gnn_model, test_ds, gnn_normalizer, device)
            if gnn_pred is not None:
                gnn_metrics = compute_metrics(gnn_pred, gnn_true)
                model_predictions["GNN (Ours)"] = (gnn_pred, gnn_true)
                print(f"  [GNN (Ours)] MAE: {gnn_metrics['MAE']*scale:.2f} {unit} | "
                      f"MAPE: {gnn_metrics['MAPE (%)']:.2f}% | R²: {gnn_metrics['R²']:.4f}")
                scenario_results.append({
                    "Scenario": sc_id, "Model": "GNN (Ours)",
                    "Architecture": gnn_arch, "Parameters": gnn_params,
                    "MAE": float(gnn_metrics["MAE"]),
                    "RMSE": float(gnn_metrics["RMSE"]),
                    "MAPE": float(gnn_metrics["MAPE (%)"]),
                    "R2": float(gnn_metrics["R²"]),
                    "MedAE": float(gnn_metrics["Median AE"]),
                    "Acc@10%": float(gnn_metrics["Acc@10%"]),
                    "Acc@20%": float(gnn_metrics["Acc@20%"]),
                    "n_samples": int(gnn_metrics["n_samples"]),
                })
        except Exception as e:
            print(f"  [GNN] ERROR: {e}")

        # ── 2. Evaluate baselines ─────────────────────────────────────────── #
        baseline_configs = [
            ("MLP Baseline", args.mlp_checkpoint_dir, BaselineMLP),
            ("LSTM Baseline", args.lstm_checkpoint_dir, BaselineLSTM)
        ]

        for bl_name, bl_dir, bl_class in baseline_configs:
            # Check for per-scenario checkpoint first (fair comparison)
            bl_ckpt = os.path.join(bl_dir, sc_id, args.target, "best.pt")
            is_all_model = False
            
            # Fallback to ALL-scenarios checkpoint
            if not os.path.exists(bl_ckpt):
                bl_ckpt = os.path.join(bl_dir, "ALL", args.target, "best.pt")
                is_all_model = True

            if not os.path.exists(bl_ckpt):
                print(f"  [{bl_name}] No checkpoint found (checked {sc_id} and ALL). Skipping.")
                continue

            print(f"  [{bl_name}] Loading model ({'ALL scenarios' if is_all_model else 'per-scenario'})...")
            bl_model, bl_arch, bl_ckpt_data = load_baseline_model(bl_ckpt, bl_class, device)
            bl_params = sum(p.numel() for p in bl_model.parameters())

            # Use the BASELINE's normalizer (trained on all scenarios)
            bl_normalizer = FeatureNormalizer()
            bl_normalizer.load_state(normalizer.get_state())  # copy base
            if "normalizer" in bl_ckpt_data:
                bl_normalizer.load_state(bl_ckpt_data["normalizer"])

            test_ds.normalizer = bl_normalizer
            bl_pred, bl_true = collect_predictions(bl_model, test_ds, bl_normalizer, device)
            if bl_pred is not None:
                bl_metrics = compute_metrics(bl_pred, bl_true)
                model_predictions[bl_name] = (bl_pred, bl_true)
                print(f"  [{bl_name}] MAE: {bl_metrics['MAE']*scale:.2f} {unit} | "
                      f"MAPE: {bl_metrics['MAPE (%)']:.2f}% | R²: {bl_metrics['R²']:.4f}")
                scenario_results.append({
                    "Scenario": sc_id, "Model": bl_name,
                    "Architecture": bl_arch, "Parameters": bl_params,
                    "MAE": float(bl_metrics["MAE"]),
                    "RMSE": float(bl_metrics["RMSE"]),
                    "MAPE": float(bl_metrics["MAPE (%)"]),
                    "R2": float(bl_metrics["R²"]),
                    "MedAE": float(bl_metrics["Median AE"]),
                    "Acc@10%": float(bl_metrics["Acc@10%"]),
                    "Acc@20%": float(bl_metrics["Acc@20%"]),
                    "n_samples": int(bl_metrics["n_samples"]),
                })

        # ── Scenario comparison table ─────────────────────────────────────── #
        if len(scenario_results) > 1:
            print(f"\n  {'─'*90}")
            print(f"  COMPARISON — {sc_id} ({args.target.upper()})")
            print(f"  {'─'*90}")
            print(f"  {'Model':<18} {'Params':>10} {'MAE ('+unit+')':>12} {'RMSE ('+unit+')':>13} {'MAPE (%)':>10} {'R²':>10} {'Acc@10%':>10}")
            print(f"  {'─'*90}")
            for r in scenario_results:
                print(
                    f"  {r['Model']:<18} {r['Parameters']:>10,} "
                    f"{r['MAE']*scale:>12.2f} {r['RMSE']*scale:>13.2f} "
                    f"{r['MAPE']:>10.2f} {r['R2']:>10.4f} {r['Acc@10%']:>9.1f}%"
                )
            print(f"  {'─'*90}")

            # GNN improvement
            gnn_r = next((r for r in scenario_results if "GNN" in r["Model"]), None)
            if gnn_r:
                for r in scenario_results:
                    if "GNN" not in r["Model"] and r["MAPE"] > 0:
                        mape_improv = ((r["MAPE"] - gnn_r["MAPE"]) / r["MAPE"]) * 100
                        mae_improv = ((r["MAE"] - gnn_r["MAE"]) / r["MAE"]) * 100
                        print(f"  → GNN improves over {r['Model']}: "
                              f"MAPE -{mape_improv:.1f}% | MAE -{mae_improv:.1f}%")

        # ── Sample-by-sample comparison ───────────────────────────────────── #
        if len(model_predictions) > 1:
            # Get the ground truth from the GNN predictions (all should share same GT)
            ref_name = list(model_predictions.keys())[0]
            ref_true = model_predictions[ref_name][1]

            # Use the minimum length across all models
            min_len = min(len(v[0]) for v in model_predictions.values())
            n_show = min(args.n_samples, min_len)

            if n_show > 0:
                # Pick random sample indices (fixed seed for reproducibility)
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(min_len, size=n_show, replace=False)
                sample_idx = np.sort(sample_idx)

                model_names = list(model_predictions.keys())

                print(f"\n  {'─'*120}")
                print(f"  SAMPLE PREDICTIONS — {sc_id} ({n_show} samples, {args.target.upper()})")
                print(f"  {'─'*120}")

                # Header
                header = f"  {'#':<4} {'Ground Truth':>14}"
                for mn in model_names:
                    short = mn.replace(" Baseline", "").replace(" (Ours)", "")
                    header += f" {'|':>3} {short+' Pred':>14} {short+' Err':>12}"
                print(header)
                print(f"  {'─'*120}")

                for rank, idx in enumerate(sample_idx):
                    gt_val = ref_true[idx] * scale
                    row = f"  {rank+1:<4} {gt_val:>12.2f} {unit}"

                    for mn in model_names:
                        pred_arr = model_predictions[mn][0]
                        pred_val = pred_arr[idx] * scale
                        err_val = pred_val - gt_val
                        row += f"   | {pred_val:>12.2f} {err_val:>+12.2f}"

                    print(row)

                print(f"  {'─'*120}")

        all_results.extend(scenario_results)

    # ── Final cross-scenario summary ──────────────────────────────────────── #
    if all_results:
        print(f"\n\n{'='*100}")
        print(f"  FINAL SUMMARY — ALL SCENARIOS ({args.target.upper()})")
        print(f"  GNN = per-scenario model | Baselines = single model trained on ALL scenarios")
        print(f"{'='*100}")
        print(f"  {'Scenario':<10} {'Model':<18} {'Params':>10} {'MAE ('+unit+')':>12} {'RMSE ('+unit+')':>13} {'MAPE (%)':>10} {'R²':>10}")
        print(f"  {'─'*95}")

        prev_sc = None
        for r in sorted(all_results, key=lambda x: (x["Scenario"], 0 if "GNN" in x["Model"] else 1)):
            if prev_sc and prev_sc != r["Scenario"]:
                print(f"  {'─'*95}")
            prev_sc = r["Scenario"]
            print(
                f"  {r['Scenario']:<10} {r['Model']:<18} {r['Parameters']:>10,} "
                f"{r['MAE']*scale:>12.2f} {r['RMSE']*scale:>13.2f} "
                f"{r['MAPE']:>10.2f} {r['R2']:>10.4f}"
            )
        print(f"{'='*100}")

        # Overall averages per model
        print(f"\n  AVERAGE ACROSS ALL SCENARIOS:")
        print(f"  {'Model':<18} {'Avg MAE ('+unit+')':>14} {'Avg MAPE (%)':>14} {'Avg R²':>10}")
        print(f"  {'─'*60}")
        model_names = sorted(set(r["Model"] for r in all_results),
                             key=lambda x: 0 if "GNN" in x else 1)
        for m in model_names:
            m_results = [r for r in all_results if r["Model"] == m]
            avg_mae = np.mean([r["MAE"] for r in m_results])
            avg_mape = np.mean([r["MAPE"] for r in m_results])
            avg_r2 = np.mean([r["R2"] for r in m_results])
            print(f"  {m:<18} {avg_mae*scale:>14.2f} {avg_mape:>14.2f} {avg_r2:>10.4f}")
        print()

        # Save
        out_file = f"evaluation_baselines_{args.target}.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {out_file}")
    else:
        print("\nNo models evaluated. Check checkpoint directories.")


if __name__ == "__main__":
    main()
