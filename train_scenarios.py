"""
train_scenarios.py — Per-Scenario Training for WirelessNet-Fermi

Trains one GNN model per (Scenario × Target) combination.
All configurations (power, scheduler, queue size) within a scenario
are pooled together into one dataset.

Usage:
  # Discover all scenarios (dry run)
  python train_scenarios.py --dry-run

  # Train ALL scenarios for delay
  python train_scenarios.py --target delay --epochs 50

  # Train ALL scenarios for throughput
  python train_scenarios.py --target throughput --epochs 50

  # Train both targets for all scenarios
  python train_scenarios.py --target all --epochs 50

  # Train a single scenario
  python train_scenarios.py --scenario SC01 --target delay --epochs 50

Checkpoints:
  checkpoints/<scenario>/<target>/best.pt
  checkpoints/<scenario>/<target>/epoch_NNN_mape_XXXX.pt
"""

import sys
import os
import time
import copy
import json
from typing import Optional, Tuple, List, Dict

# Ensure project root is on path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn as nn
# Try to import from modern torch.amp to avoid deprecation warnings in PyTorch 2.1+
try:
    import torch.amp
    _use_modern_amp = True
    GradScaler = torch.amp.GradScaler
except (ImportError, AttributeError):
    _use_modern_amp = False
    from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wireless_gnn.model import WirelessNetFermi
from wireless_gnn.dataset import (
    WirelessDataset,
    FeatureNormalizer,
    collate_fn,
    build_scenario_datasets,
)
from wireless_gnn.scenario_registry import (
    discover_scenarios,
    group_by_scenario,
    filter_for_target,
    print_summary,
    SimConfig,
)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps)))


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
        true = torch.tensor(np.asarray(graph["target_delay"]),
                            dtype=torch.float32, device=device)
    else:
        mean = torch.tensor(normalizer.tput_mean, device=device)
        std  = torch.tensor(normalizer.tput_std,  device=device)
        true = torch.tensor(np.asarray(graph["target_throughput"]),
                            dtype=torch.float32, device=device)

    pred_phys = pred * std + mean
    loss = mape_loss(pred_phys, true)
    return loss, loss.item()


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
    scaler:     Optional[GradScaler] = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total = 0.0
    n     = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    use_cuda = (device.type == "cuda")
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            for graph in batch:
                amp_ctx = torch.amp.autocast(device.type, enabled=use_cuda) if _use_modern_amp else autocast(enabled=use_cuda)
                with amp_ctx:
                    loss, mape = process_graph(model, graph, device, normalizer)
                
                if training:
                    optimizer.zero_grad()
                    if scaler is not None and use_cuda:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                total += mape
                n     += 1
            pbar.set_postfix(mape=f"{total/max(n,1):.4f}")

    return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Train one scenario × target
# --------------------------------------------------------------------------- #

def train_scenario(
    scenario_id:    str,
    target:         str,
    data_paths:     List[str],
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
    seed:           int   = 42,
    subsample_ratio: float = 1.0,
) -> dict:
    """
    Train one model for a specific scenario × target.

    Returns dict with model, normalizer, history, test_mape, etc.
    """
    label = f"{scenario_id}/{target.capitalize()}"

    # ── Device ────────────────────────────────────────────────────────────── #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n[{label}] Device : {device}")
    print(f"[{label}] Data files: {len(data_paths)}")

    # ── Data ──────────────────────────────────────────────────────────────── #
    train_ds, val_ds, test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id=scenario_id,
        target=target,
        seed=seed,
        subsample_ratio=subsample_ratio,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn , pin_memory  = True )
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    print(f"[{label}] Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────── #
    model = WirelessNetFermi(
        hidden_dim = hidden_dim,
        num_heads  = num_heads,
        iterations = iterations,
        dropout    = dropout,
        target     = target,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{label}] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    if device.type == "cuda":
        scaler = GradScaler("cuda") if _use_modern_amp else GradScaler()
    else:
        scaler = None

    # ── Checkpoint directory ──────────────────────────────────────────────── #
    ckpt_dir = os.path.join(checkpoint_dir, scenario_id, target)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")

    # ── Training loop ─────────────────────────────────────────────────────── #
    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train": [], "val": []}

    print(f"\n[{label}] {'Epoch':>6}  {'Train MAPE':>12}  {'Val MAPE':>12}  {'Best?':>6}  {'LR':>10}  {'Time':>7}")
    print(f"[{label}] " + "-" * 65)

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{label}]", unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        t0 = time.time()

        train_mape = run_epoch(
            model, train_loader, device, normalizer, optimizer,
            desc=f"  train ep{epoch:03d}", scaler=scaler
        )
        val_mape   = run_epoch(
            model, val_loader,   device, normalizer,
            desc=f"  val   ep{epoch:03d}", scaler=scaler
        )

        scheduler.step(val_mape)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        history["train"].append(train_mape)
        history["val"].append(val_mape)

        # ── Save epoch checkpoint ──────────────────────────────────────────── #
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_mape_{val_mape:.4f}.pt")
        torch.save({
            "epoch":       epoch,
            "scenario":    scenario_id,
            "target":      target,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "train_mape":  train_mape,
            "val_mape":    val_mape,
        }, epoch_ckpt)

        # ── Best model ────────────────────────────────────────────────────── #
        is_best = val_mape < best_val
        if is_best:
            best_val   = val_mape
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "scenario":   scenario_id,
                "target":     target,
                "model":      best_state,
                "val_mape":   best_val,
                "epoch":      epoch,
                "hidden_dim": hidden_dim,
                "num_heads":  num_heads,
                "iterations": iterations,
            }, best_ckpt)
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
            tqdm.write(f"\n[{label}] Early stop at epoch {epoch} (patience={patience})")
            break

    # ── Test final ────────────────────────────────────────────────────────── #
    model.load_state_dict(best_state)
    test_mape = run_epoch(
        model, test_loader, device, normalizer,
        desc="  test", scaler=scaler
    )

    print(f"\n{'='*60}")
    print(f"[{label}] TEST RESULTS")
    print(f"  MAPE:              {test_mape:.4%}")
    print(f"  Best val MAPE:     {best_val:.4%}")
    print(f"  Checkpoint:        {best_ckpt}")
    print(f"{'='*60}\n")

    return {
        "scenario":   scenario_id,
        "target":     target,
        "model":      model,
        "normalizer": normalizer,
        "history":    history,
        "test_mape":  test_mape,
        "best_val":   best_val,
        "ckpt_dir":   ckpt_dir,
        "train_ds":   train_ds,
        "val_ds":     val_ds,
        "test_ds":    test_ds,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_loss_curve(history: dict, scenario_id: str, target: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], label="Train MAPE", linewidth=2, color="#4C72B0")
    ax.plot(history["val"],   label="Val MAPE",   linewidth=2, color="#DD8452")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(f"MAPE Loss ({target.capitalize()})", fontsize=13)
    ax.set_title(f"{scenario_id} — Training Curve ({target.capitalize()})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    path = os.path.join(save_dir, f"loss_curve_{scenario_id}_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{scenario_id}] Loss curve saved -> {path}")


@torch.no_grad()
def predict(model, graph, normalizer, device):
    model.eval()
    pred, _ = model(graph)

    if model.target == 'delay':
        mean = torch.tensor(normalizer.delay_mean, device=device)
        std  = torch.tensor(normalizer.delay_std,  device=device)
        key_pred, key_true = "delay_pred", "delay_true"
        true = np.asarray(graph["target_delay"])
    else:
        mean = torch.tensor(normalizer.tput_mean, device=device)
        std  = torch.tensor(normalizer.tput_std,  device=device)
        key_pred, key_true = "throughput_pred", "throughput_true"
        true = np.asarray(graph["target_throughput"])

    return {
        key_pred: (pred * std + mean).cpu().numpy(),
        key_true: true,
    }


def plot_scatter(results_list: list, scenario_id: str, target: str,
                 save_dir: str, n_samples: int = 200):
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
    ax.set_title(f"{scenario_id} — Prediction Scatter ({target.capitalize()})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"scatter_{scenario_id}_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{scenario_id}] Scatter plot saved -> {path}")


# --------------------------------------------------------------------------- #
# Summary comparison table
# --------------------------------------------------------------------------- #

def print_results_table(all_results: List[dict]):
    """Print a summary table comparing all trained models."""
    print(f"\n{'='*70}")
    print(f"  TRAINING RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<10} {'Target':<12} {'Train MAPE':>12} {'Val MAPE':>12} {'Test MAPE':>12}")
    print(f"  {'-'*58}")

    for r in sorted(all_results, key=lambda x: (x["scenario"], x["target"])):
        train_final = r["history"]["train"][-1] if r["history"]["train"] else 0
        print(
            f"  {r['scenario']:<10} {r['target']:<12} "
            f"{train_final:>12.4%} {r['best_val']:>12.4%} {r['test_mape']:>12.4%}"
        )

    print(f"{'='*70}\n")


def save_results_json(all_results: List[dict], save_path: str):
    """Save results summary to a JSON file."""
    summary = []
    for r in all_results:
        summary.append({
            "scenario":   r["scenario"],
            "target":     r["target"],
            "test_mape":  r["test_mape"],
            "best_val":   r["best_val"],
            "train_final": r["history"]["train"][-1] if r["history"]["train"] else 0,
            "epochs_run":  len(r["history"]["train"]),
            "ckpt_dir":   r["ckpt_dir"],
        })

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved -> {save_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train WirelessNet-Fermi per scenario × target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_scenarios.py --dry-run
  python train_scenarios.py --target delay --epochs 50
  python train_scenarios.py --target all --epochs 50
  python train_scenarios.py --scenario SC01 --target delay --epochs 30
        """,
    )
    parser.add_argument("--target", default="all", choices=["delay", "throughput", "all"],
                        help="Target to train: 'delay', 'throughput', or 'all'")
    parser.add_argument("--scenario", default=None,
                        help="Train only this scenario (e.g. SC01). Default: all.")
    parser.add_argument("--root", default=".",
                        help="Project root directory")
    parser.add_argument("--data-dir", default="Data",
                        help="Data subdirectory name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample", type=float, default=0.2,
                        help="Subsample ratio of snapshots (e.g. 0.2 for 20% to speed up training)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only discover scenarios; don't train")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip data validation during discovery (faster)")

    args = parser.parse_args()

    # Resolve project root
    root = args.root
    if root == ".":
        root = os.path.dirname(os.path.abspath(__file__))

    # ── Step 1: Discover ──────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("  PHASE 1: SCENARIO DISCOVERY")
    print("=" * 70)

    all_configs = discover_scenarios(
        root,
        data_dir=args.data_dir,
        validate=not args.no_validate,
        verbose=True,
    )
    print_summary(all_configs)

    if args.dry_run:
        print("[dry-run] Exiting without training.\n")
        sys.exit(0)

    # ── Step 2: Determine what to train ────────────────────────────────────── #
    targets = ["delay", "throughput"] if args.target == "all" else [args.target]
    groups  = group_by_scenario(all_configs)

    # Filter to requested scenario if specified
    if args.scenario:
        sc = args.scenario.upper()
        if sc not in groups:
            print(f"\nERROR: Scenario '{sc}' not found. Available: {list(groups.keys())}")
            sys.exit(1)
        groups = {sc: groups[sc]}

    # Build training plan
    training_plan = []  # list of (scenario_id, target, data_paths)
    for sc_id, cfgs in groups.items():
        for tgt in targets:
            valid_cfgs = filter_for_target(cfgs, tgt)
            if not valid_cfgs:
                print(f"[SKIP] {sc_id}/{tgt}: no valid data")
                continue
            paths = [c.data_path for c in valid_cfgs]
            training_plan.append((sc_id, tgt, paths))

    print(f"\n{'='*70}")
    print(f"  PHASE 2: TRAINING PLAN")
    print(f"{'='*70}")
    for sc_id, tgt, paths in training_plan:
        print(f"  {sc_id} × {tgt:<12} → {len(paths)} data files")
    print(f"\n  Total models to train: {len(training_plan)}")
    print(f"{'='*70}\n")

    if not training_plan:
        print("Nothing to train. Check your --scenario / --target filters.")
        sys.exit(1)

    # ── Step 3: Train each ────────────────────────────────────────────────── #
    all_results = []

    for i, (sc_id, tgt, paths) in enumerate(training_plan):
        print(f"\n{'#'*70}")
        print(f"  MODEL {i+1}/{len(training_plan)}: {sc_id} × {tgt}")
        print(f"{'#'*70}")

        try:
            result = train_scenario(
                scenario_id   = sc_id,
                target        = tgt,
                data_paths    = paths,
                project_root  = root,
                hidden_dim    = args.hidden_dim,
                num_heads     = args.num_heads,
                iterations    = args.iterations,
                dropout       = args.dropout,
                epochs        = args.epochs,
                lr            = args.lr,
                patience      = args.patience,
                device_str    = args.device,
                checkpoint_dir= args.checkpoint_dir,
                seed          = args.seed,
                subsample_ratio=args.subsample,
            )

            # ── Plots ────────────────────────────────────────────────────── #
            plot_dir = os.path.join(result["ckpt_dir"], "plots")
            plot_loss_curve(result["history"], sc_id, tgt, plot_dir)

            device = next(result["model"].parameters()).device
            test_results = []
            for graph in result["test_ds"]:
                r = predict(result["model"], graph, result["normalizer"], device)
                test_results.append(r)
            if test_results:
                plot_scatter(test_results, sc_id, tgt, plot_dir)

            all_results.append(result)

        except Exception as e:
            print(f"\n[ERROR] {sc_id}/{tgt} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Step 4: Summary ────────────────────────────────────────────────────── #
    if all_results:
        print_results_table(all_results)
        summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
        save_results_json(all_results, summary_path)

    print(f"\n✅ Done! Trained {len(all_results)}/{len(training_plan)} models.\n")
