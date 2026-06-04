"""
train.py — WirelessNet-Fermi : training d'un seul modèle (Delay OU Throughput)

Usage:
python wireless_gnn/train.py --target delay --epochs 100 --resume
python wireless_gnn/train.py --target delay --epochs 100 --resume auto

Checkpoints:
  checkpoints/<target>/epoch_001_mape_0.1234.pt  <- traces de chaque époque
  checkpoints/<target>/best.pt                   <- meilleur val-loss
"""

import sys, os as _os
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import time
import copy
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from wireless_gnn.model   import WirelessNetFermi
from wireless_gnn.dataset import WirelessDataset, FeatureNormalizer, collate_fn


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps)))


def log_huber_loss(pred_log: torch.Tensor, true_log: torch.Tensor,
                   delta: float = 1.0) -> torch.Tensor:
    """Huber loss in log-space — robust to scale differences."""
    return nn.functional.huber_loss(pred_log, true_log, delta=delta)


# --------------------------------------------------------------------------- #
# Single-graph forward + loss
# --------------------------------------------------------------------------- #

def process_graph(
    model:      WirelessNetFermi,
    graph:      dict,
    device:     torch.device,
    normalizer: FeatureNormalizer,
) -> Tuple[torch.Tensor, float]:
    pred, _ = model(graph)

    if model.target == 'delay':
        mean = torch.tensor(normalizer.delay_mean, device=device)
        std  = torch.tensor(normalizer.delay_std,  device=device)
        true_raw = torch.tensor(np.asarray(graph["target_delay"]),
                                dtype=torch.float32, device=device)

        # Model predicts in log-space (normalizer stores log1p stats)
        pred_log = pred * std + mean           # log1p(delay) scale
        true_log = torch.log1p(true_raw)       # log1p(raw delay)

        # Training loss: Huber in log-space (scale-invariant)
        loss = log_huber_loss(pred_log, true_log)

        # Track log-space loss as metric (MAPE in physical space is misleading)
        metric = loss.item()

    else:
        mean = torch.tensor(normalizer.tput_mean, device=device)
        std  = torch.tensor(normalizer.tput_std,  device=device)
        true_raw = torch.tensor(np.asarray(graph["target_throughput"]),
                                dtype=torch.float32, device=device)
        pred_phys = pred * std + mean
        loss = mape_loss(pred_phys, true_raw)
        metric = loss.item()

    return loss, metric


# --------------------------------------------------------------------------- #
# Epoch runner
# --------------------------------------------------------------------------- #

def run_epoch(
    model:      WirelessNetFermi,
    loader:     DataLoader,
    device:     torch.device,
    normalizer: FeatureNormalizer,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    desc:       str = "",
) -> float:
    training = optimizer is not None
    model.train(training)

    total = 0.0
    n     = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            for graph in batch:
                loss, mape = process_graph(model, graph, device, normalizer)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                total += mape
                n     += 1
            pbar.set_postfix(mape=f"{total/max(n,1):.4f}")

    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Train one model (Delay OU Throughput)
# --------------------------------------------------------------------------- #

def train(
    target:         str,               # 'delay' or 'throughput'
    project_root:   str,
    hidden_dim:     int   = 64,
    num_heads:      int   = 4,
    iterations:     int   = 8,
    dropout:        float = 0.1,
    epochs:         int   = 50,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    patience:       int   = 10,
    device_str:     str   = "auto",
    checkpoint_dir: str   = "checkpoints",
    resume:         Optional[str] = None,
) -> dict:
    """
    Entraîne un seul modèle spécialisé (delay OU throughput).

    Checkpoints sauvegardés :
      <checkpoint_dir>/<target>/epoch_NNN.pt   — toutes les époques (traces)
      <checkpoint_dir>/<target>/best.pt        — meilleur val-loss
    """
    from wireless_gnn.dataset import build_datasets

    assert target in ('delay', 'throughput'), \
        f"--target must be 'delay' or 'throughput', got '{target}'"

    label = "Delay" if target == "delay" else "Throughput"

    # ── Device ────────────────────────────────────────────────────────────── #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n[{label}] Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────── #
    train_ds, val_ds, test_ds, normalizer = build_datasets(project_root, target=target)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────── #
    model = WirelessNetFermi(
        hidden_dim = hidden_dim,
        num_heads  = num_heads,
        iterations = iterations,
        dropout    = dropout,
        target     = target,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{label}] Paramètres : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Dossier checkpoints ───────────────────────────────────────────────── #
    ckpt_dir = os.path.join(checkpoint_dir, target)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")

    # ── Training loop ─────────────────────────────────────────────────────── #
    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train": [], "val": []}
    start_epoch = 1

    if str(resume).lower() in ("auto", "latest", "true"):
        import glob
        pattern = os.path.join(ckpt_dir, "epoch_*_mape_*.pt")
        ckpts = glob.glob(pattern)
        if ckpts:
            ckpts.sort()
            resume = ckpts[-1]
            print(f"[{label}] Auto-detected latest checkpoint: {resume}")
        else:
            print(f"[{label}] No checkpoints found in {ckpt_dir} to resume from.")
            resume = None

    if resume and os.path.isfile(resume):
        print(f"[{label}] Reprise depuis le checkpoint : {resume}")
        checkpoint = torch.load(resume, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "val_mape" in checkpoint:
                best_val = checkpoint["val_mape"]
        else:
            model.load_state_dict(checkpoint)
        best_state = copy.deepcopy(model.state_dict())
    elif resume:
        print(f"[{label}] Checkpoint introuvable : {resume}")

    print(f"\n[{label}] {'Epoch':>6}  {'Train MAPE':>12}  {'Val MAPE':>12}  {'Best?':>6}  {'LR':>10}  {'Time':>7}")
    print(f"[{label}] " + "-" * 65)

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc=f"[{label}]", unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        t0 = time.time()

        train_mape = run_epoch(model, train_loader, device, normalizer, optimizer,
                               desc=f"  train ep{epoch:03d}")
        val_mape   = run_epoch(model, val_loader,   device, normalizer,
                               desc=f"  val   ep{epoch:03d}")

        scheduler.step(val_mape)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        history["train"].append(train_mape)
        history["val"].append(val_mape)

        # ── Sauvegarde trace de chaque époque ──────────────────────────────── #
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_mape_{val_mape:.4f}.pt")
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "train_mape": train_mape,
            "val_mape":   val_mape,
        }, epoch_ckpt)

        # ── Meilleur modèle ───────────────────────────────────────────────── #
        is_best = val_mape < best_val
        if is_best:
            best_val   = val_mape
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, best_ckpt)

        else:
            no_improve += 1

        flag = "★" if is_best else ""
        tqdm.write(
            f"[{label}] {epoch:>6}  {train_mape:>12.4%}  {val_mape:>12.4%}  "
            f"{flag:>6}  {cur_lr:>10.2e}  {elapsed:>6.1f}s"
        )

        epoch_bar.set_postfix(
            train=f"{train_mape:.4%}", val=f"{val_mape:.4%}",
            best=f"{best_val:.4%}", s=f"{elapsed:.1f}s"
        )

        if no_improve >= patience:
            tqdm.write(f"\n[{label}] Early stop à l'époque {epoch} (patience={patience})")
            break

    # ── Test final ────────────────────────────────────────────────────────── #
    model.load_state_dict(best_state)
    test_mape = run_epoch(model, test_loader, device, normalizer, desc="  test")

    print(f"\n{'='*60}")
    print(f"[{label}] RÉSULTATS TEST")
    print(f"  MAPE {label:>12} : {test_mape:.4%}")
    print(f"  Meilleur val-MAPE  : {best_val:.4%}")
    print(f"  Checkpoints traces : {ckpt_dir}/epoch_NNN_mape_XXXX.pt")
    print(f"  Meilleur checkpoint: {best_ckpt}")
    print(f"{'='*60}\n")

    return {
        "model":      model,
        "normalizer": normalizer,
        "history":    history,
        "test_mape":  test_mape,
        "best_val":   best_val,
        "ckpt_dir":   ckpt_dir,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(
    model:      WirelessNetFermi,
    graph:      dict,
    normalizer: FeatureNormalizer,
    device:     torch.device,
) -> dict:
    model.eval()
    pred, _ = model(graph)

    if model.target == 'delay':
        mean = torch.tensor(normalizer.delay_mean, device=device)
        std  = torch.tensor(normalizer.delay_std,  device=device)
        key_pred, key_true = "delay_pred", "delay_true"
        true = np.asarray(graph["target_delay"])

        # Denormalize from log-space: z → log1p(delay) → delay
        pred_log  = pred * std + mean
        pred_phys = torch.expm1(pred_log).cpu().numpy()

        return {key_pred: pred_phys, key_true: true}

    else:
        mean = torch.tensor(normalizer.tput_mean, device=device)
        std  = torch.tensor(normalizer.tput_std,  device=device)
        key_pred, key_true = "throughput_pred", "throughput_true"
        true = np.asarray(graph["target_throughput"])

        return {
            key_pred: (pred * std + mean).cpu().numpy(),
            key_true: true,
        }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_loss_curve(history: dict, target: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], label="Train MAPE", linewidth=2, color="#4C72B0")
    ax.plot(history["val"],   label="Val MAPE",   linewidth=2, color="#DD8452")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(f"MAPE Loss ({target.capitalize()})", fontsize=13)
    ax.set_title(f"Training Curve - {target.capitalize()}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    path = os.path.join(save_dir, f"loss_curve_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{target.capitalize()}] Loss curve saved -> {path}")

def plot_scatter(results_list: list, target: str, save_dir: str, n_samples: int = 200):
    os.makedirs(save_dir, exist_ok=True)
    
    key_true = f"{target}_true"
    key_pred = f"{target}_pred"
    
    all_true = np.concatenate([r[key_true] for r in results_list])
    all_pred = np.concatenate([r[key_pred] for r in results_list])

    idx = np.random.choice(len(all_true), min(n_samples, len(all_true)), replace=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    if target == "delay":
        x_true = all_true[idx] * 1000
        x_pred = all_pred[idx] * 1000
        color = "#4C72B0"
        xlabel = "True Delay (ms)"
        ylabel = "Predicted Delay (ms)"
    else:
        x_true = all_true[idx] / 1000
        x_pred = all_pred[idx] / 1000
        color = "#DD8452"
        xlabel = "True Throughput (kbps)"
        ylabel = "Predicted Throughput (kbps)"

    ax.scatter(x_true, x_pred, alpha=0.6, s=25, color=color, edgecolors="none")
    lim = max(x_true.max(), x_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Prediction Scatter - {target.capitalize()}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"scatter_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{target.capitalize()}] Scatter plot saved -> {path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entraîner WirelessNet-Fermi sur Delay OU Throughput"
    )
    parser.add_argument(
        "--target", required=True, choices=["delay", "throughput"],
        help="Choisir le modèle à entraîner : 'delay' ou 'throughput'"
    )
    parser.add_argument("--root",           default=".",  help="Racine du projet")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--hidden-dim",     type=int,   default=64)
    parser.add_argument("--num-heads",      type=int,   default=4)
    parser.add_argument("--iterations",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--resume",         type=str, nargs='?', const='auto', default=None, help="Chemin du checkpoint, ou 'auto' pour le dernier")
    args = parser.parse_args()

    root = args.root if args.root != "." else _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__))
    )

    results = train(
        target         = args.target,
        project_root   = root,
        hidden_dim     = args.hidden_dim,
        num_heads      = args.num_heads,
        iterations     = args.iterations,
        dropout        = args.dropout,
        epochs         = args.epochs,
        lr             = args.lr,
        patience       = args.patience,
        device_str     = args.device,
        checkpoint_dir = args.checkpoint_dir,
        resume         = args.resume,
    )

    # ── Plots ────────────────────────────────────────────────────────── #
    plot_dir = os.path.join(results["ckpt_dir"], "plots")
    plot_loss_curve(results["history"], args.target, plot_dir)

    from wireless_gnn.dataset import build_datasets
    _, _, test_ds, norm = build_datasets(root, target=args.target)
    
    device = next(results["model"].parameters()).device
    test_results = []
    for graph in test_ds:
        r = predict(results["model"], graph, norm, device)
        test_results.append(r)
        
    plot_scatter(test_results, args.target, plot_dir)
