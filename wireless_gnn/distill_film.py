"""
distill_film.py — Knowledge Distillation with FiLM & Highway Student

Trains the innovative WirelessNet-Fermi FiLM & Highway student model under
guidance from a pretrained WirelessNet-Fermi v3 teacher.

Supports three distillation losses:
  α · L_hard  (MAPE vs ground truth)
  β · L_soft  (MSE vs teacher predictions)
  γ · L_feat  (MSE of projected flow states vs teacher flow states)

Usage:
  python wireless_gnn/distill_film.py \
      --teacher_ckpt checkpoints/delay/best.pt \
      --target delay \
      --epochs 80
"""

import sys
import os as _os
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import time
import copy
import argparse
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from wireless_gnn.model2 import WirelessNetFermiV3
from wireless_gnn.student_film import WirelessNetFermiStudent
from wireless_gnn.dataset import FeatureNormalizer, build_datasets, build_scenario_datasets, collate_fn
from wireless_gnn.scenario_registry import discover_scenarios, filter_for_target



# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def mape_loss(pred: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    return torch.mean(torch.abs((pred - target) / (target.abs() + eps)))


class FeatureProjection(nn.Module):
    """Learns a linear projection from student dim to teacher dim.
    Discarded at inference time (zero overhead)."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# --------------------------------------------------------------------------- #
# Epoch runner
# --------------------------------------------------------------------------- #

def run_distill_epoch(
    teacher:    WirelessNetFermiV3,
    student:    WirelessNetFermiStudent,
    proj:       FeatureProjection,
    loader:     DataLoader,
    device:     torch.device,
    normalizer: FeatureNormalizer,
    optimizer:  torch.optim.Optimizer = None,
    alpha: float = 0.3,
    beta:  float = 0.5,
    gamma: float = 0.2,
    desc:  str   = "",
) -> Tuple[float, float]:
    """Single epoch of distillation training or validation."""
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    proj.train(training)

    total_loss = 0.0
    total_mape = 0.0
    n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch",
                    dynamic_ncols=True)
        for batch in pbar:
            for graph in batch:
                # Select normalisation constants and ground-truth targets
                if student.target == 'delay':
                    mean = torch.tensor(normalizer.delay_mean, device=device)
                    std  = torch.tensor(normalizer.delay_std,  device=device)
                    true_phys = torch.tensor(
                        np.asarray(graph["target_delay"]),
                        dtype=torch.float32, device=device
                    )
                else:
                    mean = torch.tensor(normalizer.tput_mean, device=device)
                    std  = torch.tensor(normalizer.tput_std,  device=device)
                    true_phys = torch.tensor(
                        np.asarray(graph["target_throughput"]),
                        dtype=torch.float32, device=device
                    )

                # Teacher forward (no grads)
                with torch.no_grad():
                    teacher_pred, teacher_flow_state = teacher(graph)

                # Student forward
                student_pred, student_flow_state = student(graph)

                # ── Loss 1: Hard target (ground truth MAPE) ─────────────── #
                student_pred_phys = student_pred * std + mean
                loss_hard = mape_loss(student_pred_phys, true_phys)

                # ── Loss 2: Soft target (match teacher output, MSE) ─────── #
                loss_soft = F.mse_loss(student_pred, teacher_pred.detach())

                # ── Loss 3: Feature hint (projected flow states) ────────── #
                if student_flow_state.size(0) > 0:
                    proj_student = proj(student_flow_state)
                    loss_feat = F.mse_loss(
                        proj_student, teacher_flow_state.detach()
                    )
                else:
                    loss_feat = torch.tensor(0.0, device=device)

                # Combined loss
                loss = alpha * loss_hard + beta * loss_soft + gamma * loss_feat

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        student.parameters(), max_norm=5.0
                    )
                    if (student_flow_state.size(0) > 0
                            and hasattr(proj.proj, 'weight')):
                        nn.utils.clip_grad_norm_(
                            proj.parameters(), max_norm=5.0
                        )
                    optimizer.step()

                total_loss += loss.item()
                total_mape += loss_hard.item()
                n += 1

            pbar.set_postfix(
                mape=f"{total_mape/max(n,1):.4f}",
                loss=f"{total_loss/max(n,1):.4f}",
            )

    return total_loss / max(n, 1), total_mape / max(n, 1)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train_distilled(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve teacher checkpoint path ───────────────────────────────────── #
    if args.teacher_ckpt is None:
        if args.scenario:
            possible_dirs = [
                os.path.join("checkpoints"),
                os.path.join("checkpoints_v3"),
                os.path.join("..", "checkpoints"),
                os.path.join("..", "checkpoints_v3"),
            ]
            found = False
            for base in possible_dirs:
                candidate = os.path.join(base, args.scenario, args.target, "best.pt")
                if os.path.isfile(candidate):
                    args.teacher_ckpt = candidate
                    found = True
                    break
            if not found:
                # Default to parent directory checkpoints path for error reporting
                args.teacher_ckpt = os.path.join("..", "checkpoints_v3", args.scenario, args.target, "best.pt")
            print(f"Auto-resolved teacher checkpoint to: {args.teacher_ckpt}")
        else:
            raise ValueError("Please specify --teacher_ckpt or --scenario to locate the teacher model.")
    else:
        # Check if the user specified path directly exists relative to current dir or parent dir
        if not os.path.exists(args.teacher_ckpt):
            parent_candidate = os.path.join("..", args.teacher_ckpt)
            if os.path.exists(parent_candidate):
                args.teacher_ckpt = parent_candidate

        # Now handle directory vs file checks
        if os.path.isdir(args.teacher_ckpt):
            if args.scenario:
                args.teacher_ckpt = os.path.join(args.teacher_ckpt, args.scenario, args.target, "best.pt")
                print(f"Auto-resolved teacher checkpoint directory to file: {args.teacher_ckpt}")
            else:
                direct_file = os.path.join(args.teacher_ckpt, "best.pt")
                if os.path.isfile(direct_file):
                    args.teacher_ckpt = direct_file
                    print(f"Auto-resolved teacher checkpoint to: {args.teacher_ckpt}")

    # ── Checkpoint directory ──────────────────────────────────────────────── #
    if args.scenario:
        ckpt_dir = os.path.join(args.checkpoint_dir, args.scenario, args.target)
    else:
        ckpt_dir = os.path.join(args.checkpoint_dir, f"distilled_film_{args.target}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    print(f"Checkpoint directory resolved to: {ckpt_dir}")

    # ── Datasets ──────────────────────────────────────────────────────────── #
    if args.scenario:
        all_configs = discover_scenarios(
            args.root,
            data_dir=args.data_dir,
            validate=True,
            verbose=True,
            use_cache=True,
        )
        sc = args.scenario.upper()
        scenario_configs = [c for c in all_configs if c.scenario_id.upper() == sc]
        if not scenario_configs:
            raise ValueError(f"Scenario '{sc}' not found. Available: {sorted(list(set(c.scenario_id for c in all_configs)))}")

        valid_configs = filter_for_target(scenario_configs, args.target)
        if not valid_configs:
            raise ValueError(f"No valid configs for scenario {sc} and target {args.target}")

        data_paths = [c.data_path for c in valid_configs]
        print(f"[{sc}/{args.target}] Training on {len(data_paths)} scenario data files.")

        train_ds, val_ds, test_ds, normalizer = build_scenario_datasets(
            data_paths=data_paths,
            scenario_id=sc,
            target=args.target,
            seed=args.seed,
            subsample_ratio=args.subsample,
            split_dir=ckpt_dir,
        )
    else:
        train_ds, val_ds, test_ds, normalizer = build_datasets(
            args.root, target=args.target
        )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # ── Teacher ───────────────────────────────────────────────────────────── #
    teacher = WirelessNetFermiV3(
        hidden_dim=args.teacher_dim,
        num_heads=args.teacher_heads,
        iterations=args.teacher_iters,
        target=args.target,
    ).to(device)

    print(f"Loading teacher checkpoint: {args.teacher_ckpt}")
    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    teacher.load_state_dict(ckpt)
    teacher.eval()

    # ── Student (FiLM & Highway) ──────────────────────────────────────────── #
    student = WirelessNetFermiStudent(
        hidden_dim=args.student_dim,
        iterations=args.student_iters,
        dropout=args.dropout,
        target=args.target,
    ).to(device)

    proj = FeatureProjection(args.student_dim, args.teacher_dim).to(device)

    # ── Parameter counts ──────────────────────────────────────────────────── #
    t_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    s_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    p_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"  ARCHITECTURE COMPARISON")
    print(f"  Teacher (v3):  {t_params:>10,} params  "
          f"(D={args.teacher_dim}, H={args.teacher_heads}, K={args.teacher_iters})")
    print(f"  Student (FiLM): {s_params:>9,} params  "
          f"(D={args.student_dim}, K={args.student_iters})")
    print(f"  Projection:    {p_params:>10,} params  (discarded at inference)")
    print(f"  Compression:   {s_params/t_params:.1%} of teacher size")
    print(f"{'='*60}\n")

    # ── Optimiser ─────────────────────────────────────────────────────────── #
    params_to_opt = list(student.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(
        params_to_opt, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    start_epoch = 1
    best_val_mape = float("inf")
    best_state = copy.deepcopy(student.state_dict())
    no_improve = 0

    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if args.resume:
        if os.path.isfile(latest_ckpt):
            print(f"★ Resuming from latest checkpoint: {latest_ckpt}")
            ckpt_data = torch.load(latest_ckpt, map_location=device, weights_only=False)
            student.load_state_dict(ckpt_data["student"])
            proj.load_state_dict(ckpt_data["proj"])
            optimizer.load_state_dict(ckpt_data["optimizer"])
            if "scheduler" in ckpt_data and scheduler is not None:
                scheduler.load_state_dict(ckpt_data["scheduler"])
            start_epoch = ckpt_data.get("epoch", 0) + 1
            best_val_mape = ckpt_data.get("best_val_mape", float("inf"))
            no_improve = ckpt_data.get("no_improve", 0)
            if no_improve >= args.patience:
                print(f"★ Resetting loaded patience counter 'no_improve' to 0 (previously {no_improve}) to prevent immediate early stopping upon resumption.")
                no_improve = 0
            print(f"★ Successfully resumed from epoch {start_epoch - 1}. Training will continue from epoch {start_epoch}.")
        else:
            print(f"⚠️ Warning: --resume specified, but no latest checkpoint file found at: {latest_ckpt}")
            print(f"  Starting training from scratch (epoch 1).")

        if os.path.isfile(best_ckpt):
            best_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            if "student" in best_data:
                best_state = best_data["student"]
            else:
                best_state = copy.deepcopy(student.state_dict())

    print(f"Distillation training loop starting...")
    print(f"Loss weights: α(hard)={args.alpha}, β(soft)={args.beta}, "
          f"γ(feat)={args.gamma}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train MAPE':>12}  "
          f"{'Val MAPE':>12}  {'Best?':>6}  {'LR':>10}")
    print("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_mape = run_distill_epoch(
            teacher, student, proj, train_loader, device, normalizer,
            optimizer,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            desc=f"  train ep{epoch:03d}",
        )

        val_loss, val_mape = run_distill_epoch(
            teacher, student, proj, val_loader, device, normalizer,
            optimizer=None,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            desc=f"  val ep{epoch:03d}",
        )

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        is_best = val_mape < best_val_mape
        if is_best:
            best_val_mape = val_mape
            best_state = copy.deepcopy(student.state_dict())
            no_improve = 0
            torch.save({
                "student": best_state,
                "proj": proj.state_dict(),
                "val_mape": val_mape,
                "epoch": epoch,
                "config": vars(args),
                "architecture": "FiLM_Highway",
                "normalizer": normalizer.get_state(),
            }, best_ckpt)
        else:
            no_improve += 1

        # Save latest checkpoint for resuming
        latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
        torch.save({
            "epoch": epoch,
            "student": student.state_dict(),
            "proj": proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val_mape": best_val_mape,
            "no_improve": no_improve,
            "config": vars(args),
            "normalizer": normalizer.get_state(),
        }, latest_ckpt)

        flag = "★" if is_best else ""
        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_mape:>12.4%}  "
              f"{val_mape:>12.4%}  {flag:>6}  {cur_lr:>10.2e}")

        if no_improve >= args.patience:
            print(f"\nEarly stop at epoch {epoch} "
                  f"(patience={args.patience})")
            break

    # ── Final evaluation on test set ──────────────────────────────────────── #
    student.load_state_dict(best_state)
    _, test_mape = run_distill_epoch(
        teacher, student, proj, test_loader, device, normalizer,
        optimizer=None,
        alpha=1.0, beta=0.0, gamma=0.0,
        desc="  test evaluation",
    )

    print(f"\n{'='*60}")
    print(f"  DISTILLATION RESULTS — FiLM & Highway Student")
    print(f"  Target         : {args.target.upper()}")
    print(f"  Final Test MAPE: {test_mape:.4%}")
    print(f"  Best Val MAPE  : {best_val_mape:.4%}")
    print(f"  Saved to       : {best_ckpt}")
    print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation — FiLM & Highway Student"
    )
    parser.add_argument("--teacher_ckpt", default=None,
                        help="Path to pretrained teacher best.pt. Defaults to checkpoints_v3/{scenario}/{target}/best.pt if --scenario is specified.")
    parser.add_argument("--scenario", default=None,
                        help="Scenario identifier to train on (e.g. SC01, SC02). Default: None (legacy all).")
    parser.add_argument("--data_dir", default="data_cleaned",
                        help="Data directory containing scenario folders (default: data_cleaned)")
    parser.add_argument("--target", default="throughput",
                        choices=["delay", "throughput"])
    parser.add_argument("--root", default=".",
                        help="Project root path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--checkpoint_dir", default="checkpoints_distilled_film_throughput")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting scenario dataset")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Subsample ratio of snapshots (e.g., 0.2 to use 20%% of data)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint if one exists")

    # Loss weights
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Hard target loss weight")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Soft target loss weight")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Feature representation loss weight")

    # Teacher config
    parser.add_argument("--teacher_dim", type=int, default=64)
    parser.add_argument("--teacher_heads", type=int, default=4)
    parser.add_argument("--teacher_iters", type=int, default=8)

    # Student config
    parser.add_argument("--student_dim", type=int, default=32,
                        help="Student hidden dimension D")
    parser.add_argument("--student_iters", type=int, default=3,
                        help="Student GNN iterations K")
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    if args.root == ".":
        args.root = _os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))
        )

    train_distilled(args)
