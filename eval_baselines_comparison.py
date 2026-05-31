"""
eval_baselines_comparison.py — Compare per-scenario GNN vs single MLP/LSTM baselines

The GNN has one model per scenario (checkpoints/SC01/throughput/best.pt).
The baselines have ONE model trained on all scenarios (checkpoints_baseline/ALL/throughput/best.pt).

This script evaluates all three on the SAME test set per scenario and prints
a side-by-side comparison showing GNN superiority.

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
from wireless_gnn.dataset import build_scenario_datasets, collate_fn, WirelessDataset
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


def evaluate_model_on_test(model, test_ds, normalizer, device, target):
    """Evaluate a model on a test dataset, returning metrics."""
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
        return None

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    return compute_metrics(pred, true)


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

    # ── Pre-load baseline models (single model for all scenarios) ─────────── #
    baseline_models = {}

    # MLP
    mlp_ckpt = os.path.join(args.mlp_checkpoint_dir, "ALL", args.target, "best.pt")
    if os.path.exists(mlp_ckpt):
        print(f"Loading MLP baseline from {mlp_ckpt}")
        mlp_model, mlp_arch, mlp_ckpt_data = load_baseline_model(mlp_ckpt, BaselineMLP, device)
        mlp_params = sum(p.numel() for p in mlp_model.parameters())
        baseline_models["MLP Baseline"] = {
            "model": mlp_model, "ckpt": mlp_ckpt_data, "params": mlp_params, "arch": mlp_arch
        }
        print(f"  MLP: {mlp_arch} | {mlp_params:,} params")
    else:
        print(f"MLP checkpoint not found at {mlp_ckpt}")

    # LSTM
    lstm_ckpt = os.path.join(args.lstm_checkpoint_dir, "ALL", args.target, "best.pt")
    if os.path.exists(lstm_ckpt):
        print(f"Loading LSTM baseline from {lstm_ckpt}")
        lstm_model, lstm_arch, lstm_ckpt_data = load_baseline_model(lstm_ckpt, BaselineLSTM, device)
        lstm_params = sum(p.numel() for p in lstm_model.parameters())
        baseline_models["LSTM Baseline"] = {
            "model": lstm_model, "ckpt": lstm_ckpt_data, "params": lstm_params, "arch": lstm_arch
        }
        print(f"  LSTM: {lstm_arch} | {lstm_params:,} params")
    else:
        print(f"LSTM checkpoint not found at {lstm_ckpt}")

    if not baseline_models:
        print("\nERROR: No baseline models found. Train them first:")
        print("  python train_baselines.py --model all --target throughput --epochs 50")
        sys.exit(1)

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

        # ── 1. Evaluate GNN (per-scenario model) ─────────────────────────── #
        print(f"\n  [GNN (Ours)] Loading per-scenario model...")
        try:
            gnn_model, gnn_arch, gnn_ckpt_data = load_model_from_checkpoint(gnn_ckpt_path, device)
            gnn_model.eval()
            gnn_params = sum(p.numel() for p in gnn_model.parameters())

            # Use GNN's normalizer
            gnn_normalizer = normalizer
            if "normalizer" in gnn_ckpt_data:
                gnn_normalizer.load_state(gnn_ckpt_data["normalizer"])

            gnn_metrics = evaluate_model_on_test(gnn_model, test_ds, gnn_normalizer, device, args.target)
            if gnn_metrics:
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

        # ── 2. Evaluate baselines (single model for all scenarios) ────────── #
        for bl_name, bl_info in baseline_models.items():
            print(f"  [{bl_name}] Evaluating single-model baseline...")

            bl_model = bl_info["model"]
            bl_ckpt = bl_info["ckpt"]

            # Use the BASELINE's normalizer (trained on all scenarios)
            bl_normalizer = normalizer  # re-create from scratch
            if "normalizer" in bl_ckpt:
                bl_normalizer.load_state(bl_ckpt["normalizer"])

            bl_metrics = evaluate_model_on_test(bl_model, test_ds, bl_normalizer, device, args.target)
            if bl_metrics:
                print(f"  [{bl_name}] MAE: {bl_metrics['MAE']*scale:.2f} {unit} | "
                      f"MAPE: {bl_metrics['MAPE (%)']:.2f}% | R²: {bl_metrics['R²']:.4f}")
                scenario_results.append({
                    "Scenario": sc_id, "Model": bl_name,
                    "Architecture": bl_info["arch"], "Parameters": bl_info["params"],
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
