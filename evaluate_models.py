"""
evaluate_models.py — Comprehensive Evaluation of Trained Policy Models

Evaluates each best.pt checkpoint found under a given checkpoint directory.
Produces per-model metrics and comparison plots.

Usage:
  # Evaluate all models in a checkpoint directory
  python evaluate_models.py --checkpoint-dir checkpoints --data-dir Data_cleaned

  # Evaluate a specific policy model
  python evaluate_models.py --checkpoint-dir checkpoints --model-name SC01_PF

  # Compare v2 vs v3
  python evaluate_models.py --checkpoint-dir checkpoints_v2 checkpoints_v3 --data-dir Data_cleaned
"""

import sys, os
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import glob
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from wireless_gnn.model   import WirelessNetFermi
from wireless_gnn.model2  import WirelessNetFermiV3
from wireless_gnn.dataset import (
    WirelessDataset, FeatureNormalizer, collate_fn,
    build_scenario_datasets,
)
from wireless_gnn.scenario_registry import (
    discover_scenarios, group_by_scenario, filter_for_target,
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> dict:
    """Compute a full suite of regression metrics."""
    errors = pred - true
    abs_errors = np.abs(errors)
    
    # Basic metrics
    mae  = np.mean(abs_errors)
    mse  = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    # Percentage metrics
    denom = np.abs(true) + eps
    mape  = np.mean(abs_errors / denom) * 100
    
    # Median absolute error (robust to outliers)
    median_ae = np.median(abs_errors)
    
    # R² score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + eps))
    
    # Percentile errors
    p50 = np.percentile(abs_errors, 50)
    p90 = np.percentile(abs_errors, 90)
    p95 = np.percentile(abs_errors, 95)
    p99 = np.percentile(abs_errors, 99)
    
    # Accuracy within thresholds (% of predictions within X% of true value)
    rel_errors = abs_errors / denom
    acc_10  = np.mean(rel_errors < 0.10) * 100   # within 10%
    acc_20  = np.mean(rel_errors < 0.20) * 100   # within 20%
    acc_50  = np.mean(rel_errors < 0.50) * 100   # within 50%
    
    # Max error
    max_err = np.max(abs_errors)
    
    return {
        "MAE":            mae,
        "MSE":            mse,
        "RMSE":           rmse,
        "MAPE (%)":       mape,
        "Median AE":      median_ae,
        "R²":             r2,
        "P50 Error":      p50,
        "P90 Error":      p90,
        "P95 Error":      p95,
        "P99 Error":      p99,
        "Max Error":      max_err,
        "Acc@10%":        acc_10,
        "Acc@20%":        acc_20,
        "Acc@50%":        acc_50,
        "n_samples":      len(pred),
    }


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load a model from a best.pt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle older checkpoints that are just a raw state_dict
    if "model" not in ckpt and not any(k in ckpt for k in ["hidden_dim", "num_heads", "iterations"]):
        state_dict = ckpt
        hidden_dim = 64
        num_heads  = 4
        iterations = 8
        target     = "delay"
        model_type = None
    else:
        hidden_dim = ckpt.get("hidden_dim", 64)
        num_heads  = ckpt.get("num_heads", 4)
        iterations = ckpt.get("iterations", 8)
        target     = ckpt.get("target", "delay")
        model_type = ckpt.get("model_type", None)
        state_dict = ckpt.get("model", ckpt)
    
    # Detect model type from state_dict keys
    has_ffn = any("ffn" in k for k in state_dict.keys())
    
    if has_ffn or model_type == "v3":
        model = WirelessNetFermiV3(
            hidden_dim=hidden_dim, num_heads=num_heads,
            iterations=iterations, target=target,
        )
        arch_name = "WirelessNetFermiV3"
    else:
        model = WirelessNetFermi(
            hidden_dim=hidden_dim, num_heads=num_heads,
            iterations=iterations, target=target,
        )
        arch_name = "WirelessNetFermi"
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, arch_name, ckpt


# --------------------------------------------------------------------------- #
# Prediction with timing
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_with_timing(model, graph, normalizer, device):
    """Run prediction and measure inference time."""
    model.eval()
    
    # Warm-up (first run is always slower)
    _ = model(graph)
    
    # Timed run
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    pred, _ = model(graph)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    # Convert to physical values
    if model.target == 'delay':
        mean = torch.tensor(normalizer.delay_mean, device=device)
        std  = torch.tensor(normalizer.delay_std,  device=device)
        true = np.asarray(graph["target_delay"])
        log_pred = pred * std + mean
        pred_phys = torch.clamp(torch.expm1(log_pred), min=0.0).cpu().numpy()
    else:
        mean = torch.tensor(normalizer.tput_mean, device=device)
        std  = torch.tensor(normalizer.tput_std,  device=device)
        true = np.asarray(graph["target_throughput"])
        pred_phys = torch.clamp(pred * std + mean, min=0.0).cpu().numpy()
    
    n_flows = len(pred_phys)
    return pred_phys, true, elapsed, n_flows


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_scatter_eval(pred, true, model_name, target, save_dir):
    """Scatter plot: predicted vs true."""
    os.makedirs(save_dir, exist_ok=True)
    scale = 1000.0 if target == "delay" else 1e-3
    unit  = "ms" if target == "delay" else "kbps"
    
    fig, ax = plt.subplots(figsize=(8, 7))
    x_true = true * scale
    x_pred = pred * scale
    
    ax.scatter(x_true, x_pred, alpha=0.3, s=10, color="#4C72B0", edgecolors="none")
    lim = max(x_true.max(), x_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=2, label="y = x (perfect)")
    ax.set_xlabel(f"True {target.capitalize()} ({unit})", fontsize=13)
    ax.set_ylabel(f"Predicted {target.capitalize()} ({unit})", fontsize=13)
    ax.set_title(f"{model_name} — Scatter ({target.capitalize()})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(save_dir, f"scatter_{model_name}_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_distribution(pred, true, model_name, target, save_dir):
    """Histogram of absolute errors."""
    os.makedirs(save_dir, exist_ok=True)
    scale = 1000.0 if target == "delay" else 1e-3
    unit  = "ms" if target == "delay" else "kbps"
    
    abs_errors = np.abs(pred - true) * scale
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of absolute errors
    axes[0].hist(abs_errors, bins=80, color="#4C72B0", alpha=0.7, edgecolor="white")
    axes[0].axvline(np.median(abs_errors), color="red", linestyle="--", label=f"Median: {np.median(abs_errors):.1f} {unit}")
    axes[0].axvline(np.mean(abs_errors), color="orange", linestyle="--", label=f"Mean: {np.mean(abs_errors):.1f} {unit}")
    axes[0].set_xlabel(f"Absolute Error ({unit})", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title(f"{model_name} — Error Distribution", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Relative error histogram
    rel_errors = np.abs(pred - true) / (np.abs(true) + 1e-6) * 100
    rel_errors_clipped = np.clip(rel_errors, 0, 200)
    axes[1].hist(rel_errors_clipped, bins=80, color="#DD8452", alpha=0.7, edgecolor="white")
    axes[1].axvline(np.median(rel_errors), color="red", linestyle="--", label=f"Median: {np.median(rel_errors):.1f}%")
    axes[1].set_xlabel("Relative Error (%)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title(f"{model_name} — Relative Error", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, f"errors_{model_name}_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residuals(pred, true, model_name, target, save_dir):
    """Residual plot: error vs true value."""
    os.makedirs(save_dir, exist_ok=True)
    scale = 1000.0 if target == "delay" else 1e-3
    unit  = "ms" if target == "delay" else "kbps"
    
    residuals = (pred - true) * scale
    x_true = true * scale
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x_true, residuals, alpha=0.3, s=10, color="#55A868", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"True {target.capitalize()} ({unit})", fontsize=12)
    ax.set_ylabel(f"Residual ({unit})", fontsize=12)
    ax.set_title(f"{model_name} — Residuals vs True Value", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(save_dir, f"residuals_{model_name}_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison_bar(all_metrics: dict, metric_name: str, save_dir: str, target: str):
    """Bar chart comparing a single metric across all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(all_metrics.keys())
    values = [all_metrics[m][metric_name] for m in models]
    
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    bars = ax.bar(models, values, color=sns.color_palette("Set2", len(models)), edgecolor="white", linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Model Comparison — {metric_name} ({target.capitalize()})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    
    safe_name = metric_name.replace(" ", "_").replace("%", "pct").replace("²", "2").replace("@", "at")
    path = os.path.join(save_dir, f"compare_{safe_name}_{target}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #

def find_best_checkpoints(checkpoint_dirs):
    """Find all best.pt files under given checkpoint directories."""
    found = {}
    for ckpt_dir in checkpoint_dirs:
        for best_pt in glob.glob(os.path.join(ckpt_dir, "**", "best.pt"), recursive=True):
            # Extract model name from path: checkpoints/SC01_PF/delay/best.pt → SC01_PF
            parts = os.path.normpath(best_pt).split(os.sep)
            # Find the checkpoint dir index, model_name is next, then target
            try:
                base_idx = next(i for i, p in enumerate(parts) if p == os.path.basename(ckpt_dir))
                model_name = parts[base_idx + 1]
                target     = parts[base_idx + 2]
            except (StopIteration, IndexError):
                model_name = os.path.basename(os.path.dirname(os.path.dirname(best_pt)))
                target     = os.path.basename(os.path.dirname(best_pt))
            
            # Add checkpoint dir name as prefix if multiple dirs
            if len(checkpoint_dirs) > 1:
                dir_label = os.path.basename(ckpt_dir)
                key = f"{dir_label}/{model_name}"
            else:
                key = model_name
            
            found[key] = {
                "path": best_pt,
                "model_name": model_name,
                "target": target,
                "ckpt_dir": ckpt_dir,
            }
    return found


def evaluate_model(model_key, info, data_dir, device, output_dir):
    """Evaluate a single model on its test set."""
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {model_key} ({info['target']})")
    print(f"{'='*70}")
    
    # Load model
    model, arch_name, ckpt = load_model_from_checkpoint(info["path"], device)
    target = info["target"]
    model_name = info["model_name"]
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture:  {arch_name}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  Target:        {target}")
    print(f"  Best epoch:    {ckpt.get('epoch', '?')}")
    print(f"  Best val loss: {ckpt.get('val_mape', '?')}")
    
    # Discover data for this model
    root = os.path.dirname(os.path.abspath(__file__))
    all_configs = discover_scenarios(root, data_dir=data_dir, validate=True, verbose=False)
    groups = group_by_scenario(all_configs)
    
    # Figure out which configs belong to this model
    # Model name could be SC01, SC01_PF, SC01_DRR, etc.
    parts = model_name.split("_")
    sc_id = parts[0]  # SC01
    policy = "_".join(parts[1:]) if len(parts) > 1 else None
    
    if sc_id not in groups:
        print(f"  ERROR: Scenario {sc_id} not found in {data_dir}")
        return None
    
    cfgs = filter_for_target(groups[sc_id], target)
    if policy:
        cfgs = [c for c in cfgs if c.scheduler == policy]
    
    if not cfgs:
        print(f"  ERROR: No valid configs found for {model_name}/{target}")
        return None
    
    data_paths = [c.data_path for c in cfgs]
    print(f"  Data files:    {len(data_paths)}")
    
    # Build dataset
    _, _, test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id=model_name,
        target=target,
        seed=42,
        split_dir=os.path.dirname(ckpt_path)
    )

    if "normalizer" in ckpt_data:
        print(f"  Loading normalizer stats from checkpoint.")
        normalizer.load_state(ckpt_data["normalizer"])
    else:
        print(f"  Normalizer not in checkpoint, relying on dynamically built normalizer (WARNING: fragile!)")

    print(f"  Test samples:  {len(test_ds)}")
    
    # Run predictions
    all_pred = []
    all_true = []
    all_times = []
    all_flows = []
    
    print(f"  Running inference on {len(test_ds)} graphs...")
    for i, graph in enumerate(test_ds):
        pred_phys, true_phys, elapsed, n_flows = predict_with_timing(
            model, graph, normalizer, device
        )
        all_pred.append(pred_phys)
        all_true.append(true_phys)
        all_times.append(elapsed)
        all_flows.append(n_flows)
    
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    
    # Compute metrics
    metrics = compute_metrics(pred, true)
    
    # Inference timing stats
    times = np.array(all_times)
    flows = np.array(all_flows)
    metrics["Avg Inference (ms)"]  = np.mean(times) * 1000
    metrics["P95 Inference (ms)"]  = np.percentile(times, 95) * 1000
    metrics["Total Inference (s)"] = np.sum(times)
    metrics["Avg Flows/Graph"]     = np.mean(flows)
    
    # Scale for display
    scale = 1000.0 if target == "delay" else 1e-3
    unit  = "ms" if target == "delay" else "kbps"
    
    # Print results
    print(f"\n  {'─'*50}")
    print(f"  RESULTS: {model_key}")
    print(f"  {'─'*50}")
    print(f"  {'MAE:':<22} {metrics['MAE'] * scale:.4f} {unit}")
    print(f"  {'RMSE:':<22} {metrics['RMSE'] * scale:.4f} {unit}")
    print(f"  {'MAPE:':<22} {metrics['MAPE (%)']:.2f}%")
    print(f"  {'Median AE:':<22} {metrics['Median AE'] * scale:.4f} {unit}")
    print(f"  {'R²:':<22} {metrics['R²']:.6f}")
    print(f"  {'─'*50}")
    print(f"  {'P50 Error:':<22} {metrics['P50 Error'] * scale:.4f} {unit}")
    print(f"  {'P90 Error:':<22} {metrics['P90 Error'] * scale:.4f} {unit}")
    print(f"  {'P95 Error:':<22} {metrics['P95 Error'] * scale:.4f} {unit}")
    print(f"  {'P99 Error:':<22} {metrics['P99 Error'] * scale:.4f} {unit}")
    print(f"  {'Max Error:':<22} {metrics['Max Error'] * scale:.4f} {unit}")
    print(f"  {'─'*50}")
    print(f"  {'Acc@10%:':<22} {metrics['Acc@10%']:.2f}%")
    print(f"  {'Acc@20%:':<22} {metrics['Acc@20%']:.2f}%")
    print(f"  {'Acc@50%:':<22} {metrics['Acc@50%']:.2f}%")
    print(f"  {'─'*50}")
    print(f"  {'Avg Inference:':<22} {metrics['Avg Inference (ms)']:.2f} ms/graph")
    print(f"  {'P95 Inference:':<22} {metrics['P95 Inference (ms)']:.2f} ms/graph")
    print(f"  {'Total Inference:':<22} {metrics['Total Inference (s)']:.2f} s ({len(test_ds)} graphs)")
    print(f"  {'Avg Flows/Graph:':<22} {metrics['Avg Flows/Graph']:.1f}")
    print(f"  {'Total Test Flows:':<22} {metrics['n_samples']:,}")
    
    # Generate plots
    model_plot_dir = os.path.join(output_dir, model_key.replace("/", "_"))
    print(f"\n  Generating plots...")
    plot_scatter_eval(pred, true, model_key.replace("/", "_"), target, model_plot_dir)
    plot_error_distribution(pred, true, model_key.replace("/", "_"), target, model_plot_dir)
    plot_residuals(pred, true, model_key.replace("/", "_"), target, model_plot_dir)
    print(f"  Plots saved to: {model_plot_dir}/")
    
    return {
        "model_key":  model_key,
        "arch":       arch_name,
        "target":     target,
        "n_params":   n_params,
        "metrics":    metrics,
        "pred":       pred,
        "true":       true,
    }


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def print_comparison_table(results: list, target: str):
    """Print a formatted comparison table of all models."""
    scale = 1000.0 if target == "delay" else 1e-3
    unit  = "ms" if target == "delay" else "kbps"
    
    print(f"\n{'='*120}")
    print(f"  COMPARISON TABLE — {target.upper()}")
    print(f"{'='*120}")
    
    header = (
        f"  {'Model':<25} {'Arch':<22} {'Params':>8}  "
        f"{'MAE':>10}  {'RMSE':>10}  {'MAPE%':>8}  "
        f"{'R²':>8}  {'Acc@10%':>8}  {'Acc@20%':>8}  {'Acc@50%':>8}  "
        f"{'Infer(ms)':>10}"
    )
    print(header)
    print(f"  {'─'*118}")
    
    for r in sorted(results, key=lambda x: x["metrics"]["MAE"]):
        m = r["metrics"]
        print(
            f"  {r['model_key']:<25} {r['arch']:<22} {r['n_params']:>8,}  "
            f"{m['MAE']*scale:>10.2f}  {m['RMSE']*scale:>10.2f}  {m['MAPE (%)']:>7.2f}%  "
            f"{m['R²']:>8.4f}  {m['Acc@10%']:>7.2f}%  {m['Acc@20%']:>7.2f}%  {m['Acc@50%']:>7.2f}%  "
            f"{m['Avg Inference (ms)']:>10.2f}"
        )
    
    print(f"{'='*120}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained WirelessNet-Fermi models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", nargs="+", default=["checkpoints"],
                        help="One or more checkpoint directories to scan for best.pt files")
    parser.add_argument("--data-dir", default="Data_cleaned",
                        help="Data directory name")
    parser.add_argument("--model-name", default=None,
                        help="Evaluate only this model (e.g., SC01_PF)")
    parser.add_argument("--output-dir", default="evaluation_results",
                        help="Directory to save evaluation results and plots")
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cpu', or 'cuda'")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Find checkpoints
    checkpoints = find_best_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print(f"ERROR: No best.pt files found in {args.checkpoint_dir}")
        sys.exit(1)
    
    # Filter by model name if specified
    if args.model_name:
        checkpoints = {k: v for k, v in checkpoints.items() if args.model_name in k}
        if not checkpoints:
            print(f"ERROR: No model matching '{args.model_name}' found.")
            sys.exit(1)
    
    print(f"\nFound {len(checkpoints)} model(s) to evaluate:")
    for k, v in checkpoints.items():
        print(f"  {k:<30} -> {v['path']}")
    
    # Evaluate each model
    all_results = []
    for model_key, info in checkpoints.items():
        try:
            result = evaluate_model(
                model_key, info, args.data_dir, device, args.output_dir
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR evaluating {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo models were successfully evaluated.")
        sys.exit(1)
    
    # Group results by target for comparison
    targets_seen = set(r["target"] for r in all_results)
    for tgt in targets_seen:
        tgt_results = [r for r in all_results if r["target"] == tgt]
        if len(tgt_results) > 1:
            print_comparison_table(tgt_results, tgt)
            
            # Generate comparison bar charts
            compare_dir = os.path.join(args.output_dir, "comparison")
            all_metrics_dict = {r["model_key"]: r["metrics"] for r in tgt_results}
            
            for metric in ["MAE", "RMSE", "MAPE (%)", "R²", "Acc@10%", "Acc@20%", "Acc@50%", "Avg Inference (ms)"]:
                plot_comparison_bar(all_metrics_dict, metric, compare_dir, tgt)
            print(f"  Comparison plots saved to: {compare_dir}/")
        elif len(tgt_results) == 1:
            print_comparison_table(tgt_results, tgt)
    
    # Save JSON summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = []
    for r in all_results:
        entry = {
            "model_key": r["model_key"],
            "arch":      r["arch"],
            "target":    r["target"],
            "n_params":  r["n_params"],
            "metrics":   {k: float(v) for k, v in r["metrics"].items()},
        }
        summary.append(entry)
    
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved to: {summary_path}")
    print(f"✅ Done! Evaluated {len(all_results)} model(s).\n")
