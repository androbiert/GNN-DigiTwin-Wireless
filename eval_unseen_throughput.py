"""
eval_unseen_throughput.py — Evaluate SC01 Throughput Model on UNSEEN Data

Loads the SC01 throughput checkpoint and evaluates it against the cleaned
unseen test data in GNN_UNSEEN_cleaned/.

The unseen files encode different network configurations in their filename:
  UNSEEN_DATA-U={num_ue}-P={power}-S={scheduler}-Q={queue_size}.json

Usage:
  python eval_unseen_throughput.py
  python eval_unseen_throughput.py --checkpoint checkpoints_fine/SC01/throughput/best.pt
  python eval_unseen_throughput.py --unseen-dir GNN_UNSEEN_cleaned --checkpoint checkpoints/SC01/throughput/best.pt
"""

import sys
import os
import re
import argparse
import glob
import json
import time
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
    predict_with_timing,
)
from wireless_gnn.dataset import (
    WirelessDataset,
    FeatureNormalizer,
    collate_fn,
    load_scenario_snapshots,
)
from wireless_gnn.graph_builder import build_graph


# --------------------------------------------------------------------------- #
# Parse unseen filename to extract configuration parameters
# --------------------------------------------------------------------------- #

def parse_unseen_filename(filename: str) -> dict:
    """
    Parse UNSEEN_DATA-U=20-P=0.05W-S=PF-Q=500KiB.json into a config dict.

    Returns dict with keys: num_ue, power, scheduler, queue_size, or empty
    dict if parsing fails.
    """
    # Remove leading number prefix like "01)"
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"^\d+\)", "", base)

    config = {}

    m_ue = re.search(r"U=(\d+)", base)
    if m_ue:
        config["num_ue"] = int(m_ue.group(1))

    m_power = re.search(r"P=([\d.]+)W", base)
    if m_power:
        config["power"] = f"{m_power.group(1)}W"

    m_sched = re.search(r"S=(\w+?)(?=-|$)", base)
    if m_sched:
        config["scheduler"] = m_sched.group(1)

    m_queue = re.search(r"Q=([\d.]+(?:KiB|MiB|GiB))", base, re.IGNORECASE)
    if m_queue:
        config["queue_size"] = m_queue.group(1)

    return config


def _qsize_to_bytes(qsize_str: str) -> float:
    """Convert queue size string to bytes for sorting."""
    m = re.match(r"([\d.]+)\s*(MiB|KiB|GiB)", qsize_str, re.IGNORECASE)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2).upper()
    multipliers = {"KIB": 1024, "MIB": 1024**2, "GIB": 1024**3}
    return val * multipliers.get(unit, 1.0)


def config_label(cfg: dict) -> str:
    """Create a human-readable label from a config dict."""
    parts = []
    if "num_ue" in cfg:
        parts.append(f"U={cfg['num_ue']}")
    if "power" in cfg:
        parts.append(f"P={cfg['power']}")
    if "scheduler" in cfg:
        parts.append(f"S={cfg['scheduler']}")
    if "queue_size" in cfg:
        parts.append(f"Q={cfg['queue_size']}")
    return " | ".join(parts) if parts else "unknown"


# --------------------------------------------------------------------------- #
# Load unseen data from a single JSON file
# --------------------------------------------------------------------------- #

def load_unseen_graphs(json_path: str, target: str = "throughput") -> list:
    """
    Load all valid graph snapshots from a single unseen JSON file.

    Parameters
    ----------
    json_path : str
        Path to the unseen JSON file.
    target : str
        "throughput" or "delay" — controls feature engineering.

    Returns
    -------
    list of dict
        List of raw graph dicts (not yet normalised).
    """
    print(f"  Loading {os.path.basename(json_path)} ...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graphs = []
    for snapshot in data:
        g = build_graph(snapshot)
        if g is not None:
            g["scenario"] = "UNSEEN"
            g["config_folder"] = os.path.basename(json_path)

            if target == "throughput":
                g["flow_feat"] = g["flow_feat"].copy()
                g["flow_feat"][:, 2] = g["target_delay"]

            graphs.append(g)

    print(f"    -> {len(graphs)}/{len(data)} valid snapshots")
    return graphs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SC01 Throughput Model on UNSEEN cleaned data."
    )
    parser.add_argument(
        "--unseen-dir",
        default="GNN_UNSEEN_cleaned",
        help="Directory containing unseen JSON files (default: GNN_UNSEEN_cleaned)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/SC01/throughput/best.pt",
        help="Path to SC01 throughput checkpoint (default: checkpoints/SC01/throughput/best.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for DataLoader (default: 64)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_unseen_throughput.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load SC01 throughput model ────────────────────────────────────── #
    ckpt_path = os.path.join(_project_root, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Loading SC01 Throughput Model")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    model, arch_name, ckpt = load_model_from_checkpoint(ckpt_path, device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture:  {arch_name}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  Target:        {ckpt.get('target', 'throughput')}")
    print(f"  Best epoch:    {ckpt.get('epoch', '?')}")
    print(f"  Best val loss: {ckpt.get('val_mape', '?')}")

    # ── Load normalizer from checkpoint ───────────────────────────────── #
    normalizer = FeatureNormalizer()
    if "normalizer" in ckpt:
        print(f"  Normalizer:    loaded from checkpoint ✓")
        normalizer.load_state(ckpt["normalizer"])
    else:
        print(f"  WARNING: No normalizer in checkpoint! Results may be unreliable.")
        sys.exit(1)

    # ── Find unseen data files ────────────────────────────────────────── #
    unseen_dir = os.path.join(_project_root, args.unseen_dir)
    if not os.path.isdir(unseen_dir):
        print(f"ERROR: Unseen directory not found: {unseen_dir}")
        sys.exit(1)

    unseen_files = sorted(glob.glob(os.path.join(unseen_dir, "*.json")))
    if not unseen_files:
        print(f"ERROR: No JSON files found in {unseen_dir}")
        sys.exit(1)

    print(f"\nFound {len(unseen_files)} unseen data files:")
    for f in unseen_files:
        cfg = parse_unseen_filename(f)
        print(f"  • {os.path.basename(f)}  →  {config_label(cfg)}")

    # ── Evaluate each unseen file ─────────────────────────────────────── #
    results = []
    all_pred_global = []
    all_true_global = []

    for fpath in unseen_files:
        fname = os.path.basename(fpath)
        cfg = parse_unseen_filename(fname)

        print(f"\n{'─'*70}")
        print(f"  Evaluating: {fname}")
        print(f"  Config:     {config_label(cfg)}")
        print(f"{'─'*70}")

        # Load graphs
        graphs = load_unseen_graphs(fpath, target="throughput")
        if not graphs:
            print(f"  ⚠ No valid graphs, skipping.")
            continue

        # Create dataset with the SC01 normalizer
        unseen_ds = WirelessDataset(graphs, normalizer=normalizer)
        loader = DataLoader(
            unseen_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        # Run predictions
        all_pred = []
        all_true = []
        all_times = []

        with torch.no_grad():
            for batch in loader:
                for graph in batch:
                    pred_phys, true_phys, elapsed, n_flows = predict_with_timing(
                        model, graph, normalizer, device
                    )
                    all_pred.append(pred_phys)
                    all_true.append(true_phys)
                    all_times.append(elapsed)

        if not all_pred:
            print(f"  ⚠ No predictions generated, skipping.")
            continue

        pred = np.concatenate(all_pred)
        true = np.concatenate(all_true)
        metrics = compute_metrics(pred, true)

        all_pred_global.append(pred)
        all_true_global.append(true)

        avg_infer_ms = (sum(all_times) / len(all_times)) * 1000.0

        scale = 1e-3  # bps → kbps

        # Print results
        print(f"\n  {'─'*50}")
        print(f"  RESULTS: {config_label(cfg)}")
        print(f"  {'─'*50}")
        print(f"  {'Graphs:':<22} {len(graphs)}")
        print(f"  {'Total flows:':<22} {len(pred):,}")
        print(f"  {'MAE:':<22} {metrics['MAE'] * scale:.2f} kbps")
        print(f"  {'RMSE:':<22} {metrics['RMSE'] * scale:.2f} kbps")
        print(f"  {'MAPE:':<22} {metrics['MAPE (%)']:.2f}%")
        print(f"  {'SMAPE:':<22} {metrics['SMAPE (%)']:.2f}%")
        print(f"  {'R²:':<22} {metrics['R²']:.6f}")
        print(f"  {'Acc@10%:':<22} {metrics['Acc@10%']:.1f}%")
        print(f"  {'Acc@20%:':<22} {metrics['Acc@20%']:.1f}%")
        print(f"  {'Acc@50%:':<22} {metrics['Acc@50%']:.1f}%")
        print(f"  {'Avg Inference:':<22} {avg_infer_ms:.2f} ms/graph")
        print(f"  {'─'*50}")

        # Print sample comparison
        print(f"\n  [Sample Comparison]")
        num_samples = min(20, len(true))
        sample_indices = np.random.choice(len(true), num_samples, replace=False)
        print(f"  {'Index':<8} {'GT (kbps)':>12} {'Pred (kbps)':>12} {'Abs Err (kbps)':>14} {'Rel Err (%)':>12}")
        print("  " + "-" * 62)
        for idx in sample_indices:
            gt_val = true[idx] * scale
            pred_val = pred[idx] * scale
            abs_err = abs(gt_val - pred_val)
            rel_err = (abs_err / (abs(gt_val) + 1e-6)) * 100
            print(f"  {idx:<8} {gt_val:>12.3f} {pred_val:>12.3f} {abs_err:>14.3f} {rel_err:>12.2f}%")

        # Store result
        res = {
            "File": fname,
            "NumUE": cfg.get("num_ue", "?"),
            "Power": cfg.get("power", "?"),
            "Scheduler": cfg.get("scheduler", "?"),
            "QueueSize": cfg.get("queue_size", "?"),
            "NumGraphs": len(graphs),
            "NumFlows": int(len(pred)),
            "MAE": float(metrics["MAE"]),
            "RMSE": float(metrics["RMSE"]),
            "MAPE": float(metrics["MAPE (%)"]),
            "SMAPE": float(metrics["SMAPE (%)"]),
            "R2": float(metrics["R²"]),
            "Acc10": float(metrics["Acc@10%"]),
            "Acc20": float(metrics["Acc@20%"]),
            "Acc50": float(metrics["Acc@50%"]),
            "InferTime_ms": float(avg_infer_ms),
        }
        results.append(res)

    # ── Global aggregate metrics ──────────────────────────────────────── #
    if all_pred_global:
        global_pred = np.concatenate(all_pred_global)
        global_true = np.concatenate(all_true_global)
        global_metrics = compute_metrics(global_pred, global_true)

        scale = 1e-3

        print(f"\n\n{'='*70}")
        print(f"  GLOBAL METRICS — SC01 Model on ALL Unseen Data")
        print(f"{'='*70}")
        print(f"  {'Total flows:':<22} {len(global_pred):,}")
        print(f"  {'MAE:':<22} {global_metrics['MAE'] * scale:.2f} kbps")
        print(f"  {'RMSE:':<22} {global_metrics['RMSE'] * scale:.2f} kbps")
        print(f"  {'MAPE:':<22} {global_metrics['MAPE (%)']:.2f}%")
        print(f"  {'SMAPE:':<22} {global_metrics['SMAPE (%)']:.2f}%")
        print(f"  {'R²:':<22} {global_metrics['R²']:.6f}")
        print(f"  {'Acc@10%:':<22} {global_metrics['Acc@10%']:.1f}%")
        print(f"  {'Acc@20%:':<22} {global_metrics['Acc@20%']:.1f}%")
        print(f"  {'Acc@50%:':<22} {global_metrics['Acc@50%']:.1f}%")

    # ── Summary table ─────────────────────────────────────────────────── #
    print(f"\n\n{'='*130}")
    print(f"  SUMMARY TABLE — SC01 Throughput Model on Unseen Configurations")
    print(f"{'='*130}")
    header = (
        f"  {'UEs':<5} {'Power':<8} {'Sched':<8} {'QSize':<10} "
        f"{'Graphs':>7} {'Flows':>8} "
        f"{'MAE(kbps)':>10} {'RMSE(kbps)':>11} {'MAPE%':>8} {'SMAPE%':>8} "
        f"{'R²':>8} {'Acc@10':>8} {'Acc@20':>8} {'Acc@50':>8}"
    )
    print(header)
    print("  " + "─" * 126)

    for r in results:
        print(
            f"  {str(r['NumUE']):<5} {str(r['Power']):<8} {str(r['Scheduler']):<8} {str(r['QueueSize']):<10} "
            f"{r['NumGraphs']:>7} {r['NumFlows']:>8} "
            f"{r['MAE']*1e-3:>10.2f} {r['RMSE']*1e-3:>11.2f} {r['MAPE']:>8.2f} {r['SMAPE']:>8.2f} "
            f"{r['R2']:>8.4f} {r['Acc10']:>7.1f}% {r['Acc20']:>7.1f}% {r['Acc50']:>7.1f}%"
        )

    # ── Breakdown by dimension ────────────────────────────────────────── #
    # Group by number of UEs
    ue_groups = defaultdict(list)
    for r in results:
        ue_groups[r["NumUE"]].append(r)

    if len(ue_groups) > 1:
        print(f"\n  ── Breakdown by Number of UEs ──")
        for ue, group in sorted(ue_groups.items()):
            avg_mape = np.mean([r["MAPE"] for r in group])
            avg_r2 = np.mean([r["R2"] for r in group])
            avg_acc20 = np.mean([r["Acc20"] for r in group])
            total_flows = sum(r["NumFlows"] for r in group)
            print(
                f"    U={ue:>2}  |  Avg MAPE: {avg_mape:>7.2f}%  |  Avg R²: {avg_r2:>7.4f}  "
                f"|  Avg Acc@20: {avg_acc20:>6.1f}%  |  Flows: {total_flows:>8,}"
            )

    # Group by power
    power_groups = defaultdict(list)
    for r in results:
        power_groups[r["Power"]].append(r)

    if len(power_groups) > 1:
        print(f"\n  ── Breakdown by TX Power ──")
        for pwr, group in sorted(power_groups.items()):
            avg_mape = np.mean([r["MAPE"] for r in group])
            avg_r2 = np.mean([r["R2"] for r in group])
            avg_acc20 = np.mean([r["Acc20"] for r in group])
            total_flows = sum(r["NumFlows"] for r in group)
            print(
                f"    P={pwr:<6}  |  Avg MAPE: {avg_mape:>7.2f}%  |  Avg R²: {avg_r2:>7.4f}  "
                f"|  Avg Acc@20: {avg_acc20:>6.1f}%  |  Flows: {total_flows:>8,}"
            )

    # Group by queue size
    qsize_groups = defaultdict(list)
    for r in results:
        qsize_groups[r["QueueSize"]].append(r)

    if len(qsize_groups) > 1:
        print(f"\n  ── Breakdown by Queue Size ──")
        for qs, group in sorted(qsize_groups.items(), key=lambda x: _qsize_to_bytes(str(x[0]))):
            avg_mape = np.mean([r["MAPE"] for r in group])
            avg_r2 = np.mean([r["R2"] for r in group])
            avg_acc20 = np.mean([r["Acc20"] for r in group])
            total_flows = sum(r["NumFlows"] for r in group)
            print(
                f"    Q={qs:<8}  |  Avg MAPE: {avg_mape:>7.2f}%  |  Avg R²: {avg_r2:>7.4f}  "
                f"|  Avg Acc@20: {avg_acc20:>6.1f}%  |  Flows: {total_flows:>8,}"
            )

    print(f"\n{'='*130}")

    # ── Save results to JSON ──────────────────────────────────────────── #
    output = {
        "checkpoint": args.checkpoint,
        "architecture": arch_name,
        "n_params": n_params,
        "per_file_results": results,
    }

    if all_pred_global:
        output["global_metrics"] = {
            "total_flows": int(len(global_pred)),
            "MAE": float(global_metrics["MAE"]),
            "RMSE": float(global_metrics["RMSE"]),
            "MAPE": float(global_metrics["MAPE (%)"]),
            "SMAPE": float(global_metrics["SMAPE (%)"]),
            "R2": float(global_metrics["R²"]),
            "Acc10": float(global_metrics["Acc@10%"]),
            "Acc20": float(global_metrics["Acc@20%"]),
            "Acc50": float(global_metrics["Acc@50%"]),
        }

    out_path = os.path.join(_project_root, args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n Results saved to {out_path}")


if __name__ == "__main__":
    main()
