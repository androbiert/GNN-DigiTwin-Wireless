"""
wireless_gnn_main.py — WirelessNet-Fermi Entry Point

Usage examples:
  # Train with defaults (all 3 scenarios, 50 epochs)
  python wireless_gnn_main.py

  # Custom hyperparameters
  python wireless_gnn_main.py --epochs 100 --hidden_dim 128 --heads 8 --lr 5e-4

  # Evaluate a saved checkpoint on a specific graph
  python wireless_gnn_main.py --mode eval --checkpoint checkpoints/best_model.pt

  # Quick smoke test (5 epochs)
  python wireless_gnn_main.py --epochs 5
"""

import os
import sys
import argparse
import json
from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves PNG files)

# ---- ensure project root is on sys.path ----------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wireless_gnn.train   import train, predict
from wireless_gnn.dataset import build_datasets, collate_fn
from wireless_gnn.model   import WirelessNetFermi
from wireless_gnn.graph_builder import build_graph


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="WirelessNet-Fermi: Dynamic GNN for 5G NR QoS Prediction"
    )
    p.add_argument("--mode",          type=str,   default="train",
                   choices=["train", "eval"],
                   help="train: full training pipeline | eval: load checkpoint and evaluate")
    p.add_argument("--project_root",  type=str,   default=PROJECT_ROOT,
                   help="Path to GNN-DigiTwin-Wireless project root")
    p.add_argument("--checkpoint_dir",type=str,   default=os.path.join(PROJECT_ROOT, "checkpoints"),
                   help="Directory to save/load model checkpoints")
    # Model hyperparameters
    p.add_argument("--hidden_dim",    type=int,   default=64,
                   help="Hidden embedding dimension (default 64)")
    p.add_argument("--heads",         type=int,   default=4,
                   help="GAT attention heads (default 4, must divide hidden_dim)")
    p.add_argument("--iterations",    type=int,   default=8,
                   help="Message-passing rounds K (default 8)")
    p.add_argument("--dropout",       type=float, default=0.1,
                   help="Dropout rate in readout MLP (default 0.1)")
    # Training hyperparameters
    p.add_argument("--epochs",        type=int,   default=50,
                   help="Training epochs (default 50)")
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="Adam learning rate (default 1e-3)")
    p.add_argument("--weight_decay",  type=float, default=1e-4,
                   help="Adam weight decay (default 1e-4)")
    p.add_argument("--patience",      type=int,   default=10,
                   help="Early stopping patience (default 10)")
    p.add_argument("--device",        type=str,   default="auto",
                   help="Device: auto | cpu | cuda | cuda:0")
    # Eval only
    p.add_argument("--checkpoint",    type=str,   default=None,
                   help="Path to a .pt checkpoint file (eval mode only)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_loss_curves(history: dict, save_dir: str):
    """Save training/validation loss curve as PNG."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], label="Train MAPE", linewidth=2, color="#4C72B0")
    ax.plot(history["val"],   label="Val MAPE",   linewidth=2, color="#DD8452")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Combined MAPE Loss", fontsize=13)
    ax.set_title("WirelessNet-Fermi — Training Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Loss curve saved -> {path}")


def plot_predictions(results_list: List[dict], save_dir: str, n_samples: int = 200):
    """
    Scatter plot: predicted vs actual delay and throughput.
    results_list: list of dicts from predict()
    """
    os.makedirs(save_dir, exist_ok=True)

    all_delay_true = np.concatenate([r["delay_true"]      for r in results_list])
    all_delay_pred = np.concatenate([r["delay_pred"]      for r in results_list])
    all_tput_true  = np.concatenate([r["throughput_true"] for r in results_list])
    all_tput_pred  = np.concatenate([r["throughput_pred"] for r in results_list])

    # Sub-sample for readability
    idx = np.random.choice(len(all_delay_true), min(n_samples, len(all_delay_true)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Delay
    ax = axes[0]
    ax.scatter(all_delay_true[idx] * 1000,   # convert s -> ms
               all_delay_pred[idx] * 1000,
               alpha=0.6, s=25, color="#4C72B0", edgecolors="none")
    lim = max(all_delay_true.max(), all_delay_pred.max()) * 1000 * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("True Delay (ms)", fontsize=12)
    ax.set_ylabel("Predicted Delay (ms)", fontsize=12)
    ax.set_title("Delay Prediction", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1]
    ax.scatter(all_tput_true[idx] / 1000,   # bps -> kbps
               all_tput_pred[idx] / 1000,
               alpha=0.6, s=25, color="#DD8452", edgecolors="none")
    lim = max(all_tput_true.max(), all_tput_pred.max()) / 1000 * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("True Throughput (kbps)", fontsize=12)
    ax.set_ylabel("Predicted Throughput (kbps)", fontsize=12)
    ax.set_title("Throughput Prediction", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("WirelessNet-Fermi — Predicted vs Actual QoS",
                 fontsize=14, fontweight="bold", y=1.01)
    path = os.path.join(save_dir, "predictions_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Scatter plot saved -> {path}")


# --------------------------------------------------------------------------- #
# Eval mode: load checkpoint and run on test set
# --------------------------------------------------------------------------- #

def run_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    _, _, test_ds, normalizer = build_datasets(args.project_root)

    model = WirelessNetFermi(
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        iterations=args.iterations,
        dropout=args.dropout,
    ).to(device)

    ckpt = args.checkpoint or os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.isfile(ckpt):
        print(f"[eval] ERROR: checkpoint not found at {ckpt}")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[eval] Loaded checkpoint: {ckpt}")

    results = []
    for graph in test_ds:
        r = predict(model, graph, normalizer, device)
        results.append(r)

    all_dt = np.concatenate([r["delay_true"]  for r in results])
    all_dp = np.concatenate([r["delay_pred"]  for r in results])
    all_tt = np.concatenate([r["throughput_true"] for r in results])
    all_tp = np.concatenate([r["throughput_pred"] for r in results])

    mape_d = float(np.mean(np.abs((all_dp - all_dt) / (np.abs(all_dt) + 1e-6))))
    mape_t = float(np.mean(np.abs((all_tp - all_tt) / (np.abs(all_tt) + 1e-6))))

    print(f"\n[eval] Test MAPE Delay      : {mape_d:.4%}")
    print(f"[eval] Test MAPE Throughput : {mape_t:.4%}")

    plot_dir = os.path.join(args.checkpoint_dir, "plots")
    plot_predictions(results, plot_dir)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    print("=" * 60)
    print("  WirelessNet-Fermi — Dynamic GNN for 5G NR")
    print("  Inspired by RouteNet-Fermi (UPC)")
    print("  Innovation: Dynamic Links + GAT Attention + Dual Output")
    print("=" * 60)
    print(f"\n  Mode       : {args.mode}")
    print(f"  Project    : {args.project_root}")
    print(f"  hidden_dim : {args.hidden_dim}")
    print(f"  GAT heads  : {args.heads}")
    print(f"  Iterations : {args.iterations}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Device     : {args.device}")
    print()

    if args.mode == "eval":
        run_eval(args)
        return

    # ── Train Mode ──────────────────────────────────────────────────── #
    results = train(
        project_root  = args.project_root,
        hidden_dim    = args.hidden_dim,
        num_heads     = args.heads,
        iterations    = args.iterations,
        epochs        = args.epochs,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        patience      = args.patience,
        device_str    = args.device,
        checkpoint_dir= args.checkpoint_dir,
    )

    # ── Plots ────────────────────────────────────────────────────────── #
    plot_dir = os.path.join(args.checkpoint_dir, "plots")
    plot_loss_curves(results["history"], plot_dir)

    # Generate prediction scatter on test set
    device = next(results["model"].parameters()).device
    test_results = []
    _, _, test_ds, norm = build_datasets(args.project_root)
    for graph in test_ds:
        r = predict(results["model"], graph, norm, device)
        test_results.append(r)
    plot_predictions(test_results, plot_dir)

    # ── Save final summary ───────────────────────────────────────────── #
    summary = {
        "test_mape_loss":       results["test_loss"],
        "test_mape_delay":      results["test_mape_delay"],
        "test_mape_throughput": results["test_mape_tput"],
        "best_val_loss":        results["best_val_loss"],
        "config": {
            "hidden_dim": args.hidden_dim,
            "heads":      args.heads,
            "iterations": args.iterations,
            "epochs":     args.epochs,
            "lr":         args.lr,
        }
    }
    summary_path = os.path.join(args.checkpoint_dir, "summary.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[main] Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
