import sys
import os
import argparse
import glob
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluate_models import load_model_from_checkpoint, predict_with_timing
from wireless_gnn.dataset import build_scenario_datasets, collate_fn

def main():
    parser = argparse.ArgumentParser(description="Evaluate Inference Time Scaling of the Model.")
    parser.add_argument("--data-dir", default="data_cleaned", help="Data directory (e.g. data_cleaned)")
    parser.add_argument("--checkpoint-dir", default="checkpoints_v3", help="Checkpoints directory")
    parser.add_argument("--scenario", default="SC03", help="Scenario to use for fetching varying sized graphs")
    parser.add_argument("--target", default="throughput", help="Target (delay or throughput)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover and build the dataset
    sc = args.scenario.upper()
    ckpt_dir = os.path.join(args.checkpoint_dir, sc, args.target)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    model, arch_name, ckpt = load_model_from_checkpoint(ckpt_path, device)
    model.eval()

    # Load dataset to get graphs of varying sizes
    print("Loading dataset to extract graphs...")
    # Find all data.json for this scenario
    data_paths = glob.glob(os.path.join(args.data_dir, sc, "simulations", "*", "data.json"))
    if not data_paths:
        print(f"ERROR: No data files found for {sc} in {args.data_dir}")
        sys.exit(1)
        
    _, _, test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id=sc,
        target=args.target,
        seed=42,
        split_dir=ckpt_dir,
    )
    
    if "normalizer" in ckpt:
        normalizer.load_state(ckpt["normalizer"])

    print(f"Loaded {len(test_ds)} test graphs. Benchmarking...")

    times = []
    n_flows_list = []
    
    # We sample a subset to keep benchmarking fast (e.g., 2000 random graphs)
    sample_indices = np.random.choice(len(test_ds), min(2000, len(test_ds)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        graph = test_ds[idx]
        
        # Ensure it's evaluated properly (single batch)
        with torch.no_grad():
            _, _, elapsed, n_flows = predict_with_timing(model, graph, normalizer, device)
            
        # Skip the first few to avoid CUDA warmup spikes
        if i > 10:
            times.append(elapsed * 1000.0) # convert to ms
            n_flows_list.append(n_flows)
            
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1} / {len(sample_indices)} graphs...")

    # Aggregate by number of flows
    avg_times = {}
    for f, t in zip(n_flows_list, times):
        if f not in avg_times:
            avg_times[f] = []
        avg_times[f].append(t)
        
    x = sorted(list(avg_times.keys()))
    y_mean = [np.mean(avg_times[f]) for f in x]
    y_std  = [np.std(avg_times[f]) for f in x]

    # Overall average
    print("\n" + "="*50)
    print("INFERENCE TIME EVALUATION")
    print("="*50)
    print(f"Hardware:     {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Architecture: {arch_name}")
    print(f"Total Graphs: {len(times)}")
    print(f"Average Time: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    print(f"Min Time:     {np.min(times):.2f} ms")
    print(f"Max Time:     {np.max(times):.2f} ms")
    print("="*50)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    plt.plot(x, y_mean, marker='o', linewidth=2, color='#2ecc71', label=f"{arch_name} ({device.type.upper()})")
    plt.fill_between(x, np.array(y_mean) - np.array(y_std), np.array(y_mean) + np.array(y_std), alpha=0.2, color='#2ecc71')
    
    plt.xlabel("Graph Size (Number of Flows)", fontsize=12)
    plt.ylabel("Inference Time (ms)", fontsize=12)
    plt.title(f"Scalability: Inference Time vs Network Size\n{sc} - {args.target.capitalize()}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    out_file = f"plot_inference_time_{sc}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Scalability plot saved to {out_file}")

if __name__ == "__main__":
    main()
