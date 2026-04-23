"""
train.py — Training & Evaluation Loop for WirelessNet-Fermi

Loss: MAPE (Mean Absolute Percentage Error) on both delay and throughput.
      MAPE is preferred over MSE here because:
        - delay varies over several orders of magnitude (0.004 s -> 2+ s)
        - throughput varies widely (631 bps -> 125 kbps)
      MAPE treats all samples equally regardless of magnitude.
"""

import os
import time
import copy
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wireless_gnn.model   import WirelessNetFermi
from wireless_gnn.dataset import WirelessDataset, FeatureNormalizer, collate_fn


# --------------------------------------------------------------------------- #
# Loss Functions
# --------------------------------------------------------------------------- #

def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean Absolute Percentage Error — safe against zero targets."""
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps)))


def combined_loss(
    delay_pred:      torch.Tensor,
    throughput_pred: torch.Tensor,
    delay_true:      torch.Tensor,
    throughput_true: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combined MAPE loss.
    alpha: weight for delay loss (1-alpha for throughput).
    """
    loss_d = mape_loss(delay_pred,      delay_true)
    loss_t = mape_loss(throughput_pred, throughput_true)
    return alpha * loss_d + (1.0 - alpha) * loss_t, loss_d.item(), loss_t.item()


# --------------------------------------------------------------------------- #
# Single-graph forward pass with loss
# --------------------------------------------------------------------------- #

def process_graph(
    model: WirelessNetFermi,
    graph: dict,
    device: torch.device,
    normalizer: FeatureNormalizer,
) -> tuple[torch.Tensor, float, float]:
    """
    Run model on one graph snapshot.
    Returns (loss, mape_delay, mape_throughput).
    Predictions are in normalised space; targets are un-normalised for metrics.
    """
    delay_pred, tput_pred = model(graph)

    # Denormalise predictions back to physical units for computing MAPE
    delay_mean = torch.tensor(normalizer.delay_mean, device=device)
    delay_std  = torch.tensor(normalizer.delay_std,  device=device)
    tput_mean  = torch.tensor(normalizer.tput_mean,  device=device)
    tput_std   = torch.tensor(normalizer.tput_std,   device=device)

    delay_pred_phys = delay_pred * delay_std + delay_mean
    tput_pred_phys  = tput_pred  * tput_std  + tput_mean

    # Ground-truth targets in physical units
    delay_true = torch.tensor(
        np.asarray(graph["target_delay"]), dtype=torch.float32, device=device
    )
    tput_true = torch.tensor(
        np.asarray(graph["target_throughput"]), dtype=torch.float32, device=device
    )

    loss, ld, lt = combined_loss(delay_pred_phys, tput_pred_phys, delay_true, tput_true)
    return loss, ld, lt


# --------------------------------------------------------------------------- #
# Epoch helpers
# --------------------------------------------------------------------------- #

def run_epoch(
    model:      WirelessNetFermi,
    loader:     DataLoader,
    device:     torch.device,
    normalizer: FeatureNormalizer,
    optimizer:  Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float]:
    """
    One training or evaluation epoch.
    If optimizer is None -> eval mode.
    Returns (avg_loss, avg_mape_delay, avg_mape_throughput).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = total_ld = total_lt = 0.0
    n_samples = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            # batch is a list of graph dicts (see collate_fn)
            for graph in batch:
                loss, ld, lt = process_graph(model, graph, device, normalizer)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                total_loss += loss.item()
                total_ld   += ld
                total_lt   += lt
                n_samples  += 1

    n = max(n_samples, 1)
    return total_loss / n, total_ld / n, total_lt / n


# --------------------------------------------------------------------------- #
# Main train function
# --------------------------------------------------------------------------- #

def train(
    project_root: str,
    hidden_dim:   int   = 64,
    num_heads:    int   = 4,
    iterations:   int   = 8,
    epochs:       int   = 50,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 10,
    device_str:   str   = "auto",
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """
    Full training pipeline.

    Returns a results dict with train/val/test metrics and the trained model.
    """
    from wireless_gnn.dataset import build_datasets

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_ds, val_ds, test_ds, normalizer = build_datasets(project_root)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = WirelessNetFermi(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        iterations=iterations,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    history        = {"train": [], "val": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  "
          f"{'MAPE Delay':>12}  {'MAPE Tput':>12}  {'Time(s)':>8}")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, device, normalizer, optimizer)
        val_loss, val_ld, val_lt = run_epoch(model, val_loader, device, normalizer)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{val_ld:>12.4%}  {val_lt:>12.4%}  {elapsed:>8.1f}")

        # Early stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(best_state, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[train] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # ------------------------------------------------------------------ #
    # Final evaluation on test set
    # ------------------------------------------------------------------ #
    model.load_state_dict(best_state)
    test_loss, test_ld, test_lt = run_epoch(model, test_loader, device, normalizer)

    print(f"\n{'='*72}")
    print(f"[train] TEST RESULTS:")
    print(f"  Total MAPE Loss  : {test_loss:.6f}")
    print(f"  MAPE Delay       : {test_ld:.4%}")
    print(f"  MAPE Throughput  : {test_lt:.4%}")
    print(f"{'='*72}\n")

    return {
        "model":        model,
        "normalizer":   normalizer,
        "history":      history,
        "test_loss":    test_loss,
        "test_mape_delay": test_ld,
        "test_mape_tput":  test_lt,
        "best_val_loss":   best_val_loss,
    }


# --------------------------------------------------------------------------- #
# Inference helper
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(
    model:      WirelessNetFermi,
    graph:      dict,
    normalizer: FeatureNormalizer,
    device:     torch.device,
) -> dict:
    """
    Run inference on a single graph snapshot.
    Returns a dict with 'delay' and 'throughput' predictions (physical units).
    """
    model.eval()
    delay_pred, tput_pred = model(graph)

    delay_mean = torch.tensor(normalizer.delay_mean, device=device)
    delay_std  = torch.tensor(normalizer.delay_std,  device=device)
    tput_mean  = torch.tensor(normalizer.tput_mean,  device=device)
    tput_std   = torch.tensor(normalizer.tput_std,   device=device)

    delay_phys = (delay_pred * delay_std + delay_mean).cpu().numpy()
    tput_phys  = (tput_pred  * tput_std  + tput_mean ).cpu().numpy()

    return {
        "delay_pred":      delay_phys,
        "throughput_pred": tput_phys,
        "delay_true":      np.asarray(graph["target_delay"]),
        "throughput_true": np.asarray(graph["target_throughput"]),
    }
