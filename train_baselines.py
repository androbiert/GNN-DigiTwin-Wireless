"""
train_baselines.py — Train MLP/LSTM/GNN Baselines

Trains baseline models on data pooled from ALL scenarios (default),
or on a SINGLE scenario if --scenario is specified.

Usage:
  # Train MLP baseline on all scenarios
  python train_baselines.py --model baseline --target throughput --epochs 50

  # Train LSTM baseline on a specific scenario
  python train_baselines.py --model lstm --scenario SC01 --target throughput --epochs 50

  # Train all baselines on all scenarios
  python train_baselines.py --model all --target throughput --epochs 50
"""

import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from wireless_gnn.baseline_mlp import BaselineMLP
from wireless_gnn.baseline_lstm import BaselineLSTM
from wireless_gnn.baseline_gnn import BaselineGNN
from wireless_gnn.scenario_registry import (
    discover_scenarios, group_by_scenario, filter_for_target, print_summary,
)
from train_scenarios import train_scenario, plot_loss_curve, predict, plot_scatter


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MLP/LSTM baselines on ALL scenarios or a specific one",
    )
    parser.add_argument("--scenario", default=None,
                        help="Train on a specific scenario (e.g. SC01). Default: ALL scenarios combined.")
    parser.add_argument("--model", default="all", choices=["baseline", "lstm", "baseline_gnn", "all"],
                        help="'baseline' (MLP), 'lstm', 'baseline_gnn', or 'all' (all of them)")
    parser.add_argument("--target", default="throughput", choices=["delay", "throughput"],
                        help="Target to train")
    parser.add_argument("--loss", default="mape", choices=["mape", "mae"],
                        help="Loss function to use for throughput (default: mape)")
    parser.add_argument("--data-dir", default="data_cleaned")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dim for baselines (default 128 to give baselines fair capacity)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample", type=float, default=0.2,
                        help="Subsample ratio of snapshots")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--recache", action="store_true")

    args = parser.parse_args()

    # ── Discover scenarios ─────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("  DISCOVERING SCENARIOS")
    print("=" * 70)

    all_configs = discover_scenarios(
        _project_root, data_dir=args.data_dir,
        validate=True, verbose=True, use_cache=not args.recache,
    )
    print_summary(all_configs)

    groups = group_by_scenario(all_configs)

    # Filter to requested scenario if specified
    if args.scenario:
        sc = args.scenario.upper()
        if sc not in groups:
            print(f"ERROR: Scenario '{sc}' not found. Available: {list(groups.keys())}")
            sys.exit(1)
        groups = {sc: groups[sc]}
        scenario_label = sc
        print(f"\n  Training on scenario: {sc}")
    else:
        scenario_label = "ALL"
        print(f"\n  Training on ALL scenarios combined")

    # Pool data paths
    all_data_paths = []
    scenario_count = 0
    for sc_id, cfgs in groups.items():
        valid_cfgs = filter_for_target(cfgs, args.target)
        if valid_cfgs:
            paths = [c.data_path for c in valid_cfgs]
            all_data_paths.extend(paths)
            scenario_count += 1
            print(f"  [{sc_id}] {len(paths)} data files for {args.target}")

    if not all_data_paths:
        print(f"ERROR: No valid data found for target '{args.target}'")
        sys.exit(1)

    print(f"\n  Total: {len(all_data_paths)} data files from {scenario_count} scenarios")

    # ── Determine which models to train ───────────────────────────────────── #
    models_to_train = []
    if args.model in ("baseline", "all"):
        models_to_train.append(("baseline", BaselineMLP, "checkpoints_baseline"))
    if args.model in ("lstm", "all"):
        models_to_train.append(("lstm", BaselineLSTM, "checkpoints_baseline_lstm"))
    if args.model in ("baseline_gnn", "all"):
        models_to_train.append(("baseline_gnn", BaselineGNN, "checkpoints_baseline_gnn"))

    # ── Train each baseline ───────────────────────────────────────────────── #
    for model_name, model_cls, ckpt_dir in models_to_train:
        print(f"\n{'#'*70}")
        print(f"  TRAINING: {model_cls.__name__} on {scenario_label} ({args.target})")
        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"{'#'*70}")

        try:
            result = train_scenario(
                scenario_id     = scenario_label,
                target         = args.target,
                data_paths     = all_data_paths,
                project_root   = _project_root,
                hidden_dim     = args.hidden_dim,
                num_heads      = 4,
                iterations     = 8,
                dropout        = args.dropout,
                epochs         = args.epochs,
                lr             = args.lr,
                patience       = args.patience,
                device_str     = args.device,
                checkpoint_dir = ckpt_dir,
                seed           = args.seed,
                subsample_ratio= args.subsample,
                model_class    = model_cls,
                resume         = args.resume,
                loss_type      = args.loss,
            )

            # Plots
            plot_dir = os.path.join(result["ckpt_dir"], "plots")
            plot_loss_curve(result["history"], f"{scenario_label}_{model_name}", args.target, plot_dir)

            device = next(result["model"].parameters()).device
            test_results = []
            for graph in result["test_ds"]:
                r = predict(result["model"], graph, result["normalizer"], device)
                test_results.append(r)
            if test_results:
                plot_scatter(test_results, f"{scenario_label}_{model_name}", args.target, plot_dir)

            print(f"\n✅ {model_cls.__name__} training complete!")
            print(f"   Checkpoint: {result['ckpt_dir']}/best.pt")

        except Exception as e:
            print(f"\n❌ ERROR training {model_cls.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  ALL BASELINES TRAINED")
    print("=" * 70)
    print("\nNext step: Run comparison with:")
    print("  python eval_baselines_comparison.py")


if __name__ == "__main__":
    main()
