import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario
from wireless_gnn2.dataset_v import build_temporal_datasets
from wireless_gnn2.model_v import ModelV

def train_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Discover data
    configs = discover_scenarios(args.data_dir)
    groups = group_by_scenario(configs)
    
    if args.scenario not in groups:
        raise ValueError(f"Scenario {args.scenario} not found. Available: {list(groups.keys())}")
        
    cfgs = groups[args.scenario]
    if args.queue_size:
        cfgs = [c for c in cfgs if c.queue_size == args.queue_size]
        
    data_paths = [c.data_path for c in cfgs]
    if not data_paths:
        raise ValueError("No data paths found matching the criteria.")
        
    print(f"Found {len(data_paths)} data files.")
    
    # 2. Build datasets
    ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.scenario}_{args.queue_size or 'all'}_seq{args.seq_len}", args.target)
    
    train_ds, val_ds, test_ds, normalizer = build_temporal_datasets(
        data_paths=data_paths,
        target=args.target,
        seq_len=args.seq_len,
        split_dir=ckpt_dir
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model
    model = ModelV(input_dim=31, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.target == 'delay':
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.SmoothL1Loss()
        
    # 4. Train Loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save normalizer
    normalizer_path = os.path.join(ckpt_dir, "normalizer.json")
    with open(normalizer_path, "w") as f:
        json.dump(normalizer.state_dict(), f)
        
    losses = {"train": [], "val": []}
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            if args.target == 'delay':
                y_target = torch.log1p(y_batch)
            else:
                y_target = y_batch
                
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                if args.target == 'delay':
                    y_target = torch.log1p(y_batch)
                else:
                    y_target = y_batch
                    
                pred = model(X_batch)
                loss = criterion(pred, y_target)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_ds)
        
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "normalizer": normalizer.state_dict()
            }, ckpt_path)
            star = "★"
        else:
            star = ""
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} {star}")
            
    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"Checkpoint saved to {ckpt_dir}/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_cleaned")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_model_v")
    parser.add_argument("--scenario", type=str, default="SC01")
    parser.add_argument("--queue-size", type=str, default=None)
    parser.add_argument("--target", type=str, default="delay")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
        
    train_model(args)
