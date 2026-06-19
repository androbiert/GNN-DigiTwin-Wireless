"""
finetune_unseen.py — Fine-tune a GNN Throughput Model on Unseen Configurations

Loads a pre-trained throughput model (e.g. SC01), freezes its normalizer,
splits the unseen cleaned data into a fine-tuning set (80%) and a test set (20%),
trains the model weights for a few epochs at a low learning rate,
and evaluates performance before vs after adaptation.

Usage:
  python finetune_unseen.py --checkpoint checkpoints_v3/SC01/throughput/best.pt --epochs 15
"""

import sys
import os
import argparse
import glob
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
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
)
from wireless_gnn.graph_builder import build_graph
from eval_unseen_throughput import parse_unseen_filename, config_label


# Loss functions matching train_scenarios.py
def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps)))

def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def load_unseen_graphs(json_path: str) -> list:
    """Load all valid graph snapshots and apply throughput feature replacement."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graphs = []
    for snapshot in data:
        g = build_graph(snapshot)
        if g is not None:
            g["scenario"] = "UNSEEN"
            g["config_folder"] = os.path.basename(json_path)
            # Replace offered load at index 2 with target delay for throughput prediction
            g["flow_feat"] = g["flow_feat"].copy()
            g["flow_feat"][:, 2] = g["target_delay"]
            graphs.append(g)
    return graphs


def run_eval(model, loader, normalizer, device):
    """Run model evaluations and return predictions, ground truths, and metrics."""
    all_pred = []
    all_true = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for graph in batch:
                pred_phys, true_phys, _, _ = predict_with_timing(
                    model, graph, normalizer, device
                )
                all_pred.append(pred_phys)
                all_true.append(true_phys)
    
    if not all_pred:
        return None, None, None
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    return pred, true, compute_metrics(pred, true)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GNN Throughput model on unseen configurations."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_v3/SC01/throughput/best.pt",
        help="Path to pre-trained throughput model (default: checkpoints_v3/SC01/throughput/best.pt)",
    )
    parser.add_argument(
        "--unseen-dir",
        default="GNN_UNSEEN_cleaned",
        help="Directory containing cleaned unseen JSON files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of fine-tuning epochs (default: 15)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader (default: 32)",
    )
    parser.add_argument(
        "--loss-type",
        default="mape",
        choices=["mape", "mae"],
        help="Loss function to optimize (default: mape)",
    )
    parser.add_argument(
        "--output-suffix",
        default="finetuned",
        help="Suffix for saved fine-tuned checkpoint (e.g. best_finetuned.pt)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load Pre-trained GNN Model ─────────────────────────────────────── #
    ckpt_path = os.path.join(_project_root, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"\nLoading Pre-trained GNN checkpoint: {args.checkpoint}")
    model, arch_name, ckpt = load_model_from_checkpoint(ckpt_path, device)
    
    # Extract normalizer from checkpoint
    normalizer = FeatureNormalizer()
    if "normalizer" in ckpt:
        normalizer.load_state(ckpt["normalizer"])
        print("  Normalizer loaded successfully.")
    else:
        print("ERROR: Checkpoint has no normalizer. Fine-tuning cannot proceed safely.")
        sys.exit(1)

    # ── Load and Split Unseen Data ──────────────────────────────────────── #
    unseen_dir = os.path.join(_project_root, args.unseen_dir)
    unseen_files = sorted(glob.glob(os.path.join(unseen_dir, "*.json")))
    if not unseen_files:
        print(f"ERROR: No JSON files found in {unseen_dir}")
        sys.exit(1)

    print(f"\nLoading unseen files and splitting 80% train / 20% test...")
    train_graphs = []
    test_graphs = []

    # Fix seed for reproducible splitting
    random.seed(42)
    
    for f in unseen_files:
        cfg = parse_unseen_filename(f)
        cfg_str = config_label(cfg)
        graphs = load_unseen_graphs(f)
        if not graphs:
            continue
        
        # Shuffle snapshots of this specific configuration
        random.shuffle(graphs)
        n_train = int(len(graphs) * 0.8)
        
        train_graphs.extend(graphs[:n_train])
        test_graphs.extend(graphs[n_train:])
        print(f"  • {os.path.basename(f)} ({cfg_str}) -> {n_train} train graphs, {len(graphs)-n_train} test graphs")

    print(f"\nTotal Train Snapshots: {len(train_graphs)}")
    print(f"Total Test Snapshots:  {len(test_graphs)}")

    # Create Datasets using the original GNN normalizer (normalizer stays frozen)
    train_ds = WirelessDataset(train_graphs, normalizer=normalizer)
    test_ds  = WirelessDataset(test_graphs,  normalizer=normalizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ── Evaluate BEFORE Fine-tuning ────────────────────────────────────── #
    print(f"\n======================================================================")
    print(f"  EVALUATING MODEL PERFORMANCE BEFORE FINE-TUNING (20% TEST SPLIT)")
    print(f"======================================================================")
    pred_before, true_before, metrics_before = run_eval(model, test_loader, normalizer, device)
    
    scale = 1e-3  # bps -> kbps
    print(f"  MAE:      {metrics_before['MAE']*scale:.2f} kbps")
    print(f"  RMSE:     {metrics_before['RMSE']*scale:.2f} kbps")
    print(f"  MAPE:     {metrics_before['MAPE (%)']:.2f}%")
    print(f"  R²:       {metrics_before['R²']:.6f}")
    print(f"  Acc@20%:  {metrics_before['Acc@20%']:.1f}%")

    # ── Fine-tuning Loop ────────────────────────────────────────────────── #
    print(f"\n======================================================================")
    print(f"  STARTING FINE-TUNING FOR {args.epochs} EPOCHS")
    print(f"  Loss function: {args.loss_type.upper()} | Learning Rate: {args.lr}")
    print(f"======================================================================")

    # Setup optimizer (only fine-tune model parameters, normalizer is constant)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    mean = torch.tensor(normalizer.tput_mean, device=device, dtype=torch.float32)
    std  = torch.tensor(normalizer.tput_std,  device=device, dtype=torch.float32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            n_flows = 0
            
            for graph in batch:
                pred, _ = model(graph)
                pred = pred.float()
                
                # Denormalize GNN predictions to physical throughput
                pred_phys = pred * std + mean
                true_raw = torch.tensor(
                    np.asarray(graph["target_throughput"]),
                    dtype=torch.float32, device=device
                )
                
                if args.loss_type == "mae":
                    loss = mae_loss(pred_phys, true_raw)
                else:
                    loss = mape_loss(pred_phys, true_raw)
                
                batch_loss += loss
                n_flows += len(true_raw)
            
            if batch_loss > 0:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1
                
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # ── Evaluate AFTER Fine-tuning ─────────────────────────────────────── #
    print(f"\n======================================================================")
    print(f"  EVALUATING MODEL PERFORMANCE AFTER FINE-TUNING (20% TEST SPLIT)")
    print(f"======================================================================")
    pred_after, true_after, metrics_after = run_eval(model, test_loader, normalizer, device)
    
    print(f"  MAE:      {metrics_after['MAE']*scale:.2f} kbps (Before: {metrics_before['MAE']*scale:.2f} kbps)")
    print(f"  RMSE:     {metrics_after['RMSE']*scale:.2f} kbps (Before: {metrics_before['RMSE']*scale:.2f} kbps)")
    print(f"  MAPE:     {metrics_after['MAPE (%)']:.2f}% (Before: {metrics_before['MAPE (%)']:.2f}%)")
    print(f"  R²:       {metrics_after['R²']:.6f} (Before: {metrics_before['R²']:.6f})")
    print(f"  Acc@20%:  {metrics_after['Acc@20%']:.1f}% (Before: {metrics_before['Acc@20%']:.1f}%)")

    # ── Save Checkpoint ────────────────────────────────────────────────── #
    dir_name = os.path.dirname(ckpt_path)
    base_name = os.path.basename(ckpt_path)
    new_name = base_name.replace(".pt", f"_{args.output_suffix}.pt")
    save_path = os.path.join(dir_name, new_name)

    print(f"\nSaving fine-tuned checkpoint to: {save_path}")
    torch.save({
        "scenario": ckpt.get("scenario", "UNSEEN"),
        "target": ckpt.get("target", "throughput"),
        "model": model.state_dict(),
        "normalizer": normalizer.get_state(),
        "hidden_dim": ckpt.get("hidden_dim", 64),
        "num_heads": ckpt.get("num_heads", 4),
        "iterations": ckpt.get("iterations", 8),
        "epoch": args.epochs,
    }, save_path)
    print("Fine-tuning completed successfully! ✓")


if __name__ == "__main__":
    main()
