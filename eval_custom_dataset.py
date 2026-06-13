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

from evaluate_models import load_model_from_checkpoint, predict_with_timing, compute_metrics
from wireless_gnn.dataset import WirelessDataset, collate_fn, FeatureNormalizer
from wireless_gnn.graph_builder import build_graph

def main():
    parser = argparse.ArgumentParser(description="Evaluate a custom folder of JSON data.")
    parser.add_argument("--test-dir", default="GNN_test", help="Directory containing custom JSON files")
    parser.add_argument("--checkpoint-path", default=None, required=True, help="Path to the trained model (e.g. checkpoints_v3/SC01/delay/best.pt)")
    parser.add_argument("--target", default="delay", help="Target variable (delay or throughput)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)

    # Load model and normalizer
    print(f"Loading model from {args.checkpoint_path}...")
    model, arch_name, ckpt = load_model_from_checkpoint(args.checkpoint_path, device)
    model.eval()

    normalizer = FeatureNormalizer()
    if "normalizer" in ckpt:
        normalizer.load_state(ckpt["normalizer"])
        print("Loaded normalizer from checkpoint.")
    else:
        print("WARNING: No normalizer found in checkpoint! Evaluation will be unnormalized.")

    # Find JSON files
    json_files = glob.glob(os.path.join(args.test_dir, "*.json"))
    if not json_files:
        print(f"ERROR: No JSON files found in {args.test_dir}")
        sys.exit(1)

    all_results = []
    
    # Process each file independently
    for fpath in json_files:
        fname = os.path.basename(fpath)
        print(f"\nProcessing {fname}...")
        
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        graphs = []
        for snapshot in data:
            g = build_graph(snapshot)
            if g is not None:
                # If target is throughput, swap it out for delay in input feature
                if args.target == "throughput" and "target_delay" in g:
                    g["flow_feat"] = g["flow_feat"].copy()
                    g["flow_feat"][:, 2] = g["target_delay"]
                graphs.append(g)
                
        print(f"  -> Built {len(graphs)} valid graphs.")
        if len(graphs) == 0:
            continue
            
        # Create dataset and loader
        test_ds = WirelessDataset(graphs, normalizer=normalizer)
        loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        all_pred = []
        all_true = []
        all_times = []
        
        with torch.no_grad():
            for batch in loader:
                for graph in batch:
                    pred_phys, true_phys, elapsed, _ = predict_with_timing(model, graph, normalizer, device)
                    all_pred.append(pred_phys)
                    all_true.append(true_phys)
                    all_times.append(elapsed * 1000.0) # ms
                    
        pred = np.concatenate(all_pred)
        true = np.concatenate(all_true)
        metrics = compute_metrics(pred, true)
        
        avg_infer_ms = float(np.mean(all_times))
        
        res = {
            "File": fname,
            "Graphs": len(graphs),
            "MAE": float(metrics["MAE"]),
            "RMSE": float(metrics["RMSE"]),
            "MAPE": float(metrics["MAPE (%)"]),
            "SMAPE": float(metrics["SMAPE (%)"]),
            "R2": float(metrics["R²"]),
            "Acc10": float(metrics["Acc@10%"]),
            "Acc20": float(metrics["Acc@20%"]),
            "Time_ms": avg_infer_ms
        }
        all_results.append(res)
        
        scale = 1000.0 if args.target == "delay" else 1e-3
        unit = "ms" if args.target == "delay" else "kbps"
        
        print(f"  -> MAE: {metrics['MAE'] * scale:.2f} {unit} | MAPE: {metrics['MAPE (%)']:.2f}% | R²: {metrics['R²']:.4f}")
        print(f"  -> Acc@10: {metrics['Acc@10%']:.1f}% | Acc@20: {metrics['Acc@20%']:.1f}%")
        print(f"  -> Infer Time: {avg_infer_ms:.2f} ms/graph")

    # Final Summary Table
    print(f"\n{'='*105}")
    print(f"FINAL SUMMARY - CUSTOM DATASET EVALUATION ({args.target.upper()})")
    print(f"{'='*105}")
    unit = "ms" if args.target == "delay" else "kbps"
    print(f"{'Filename':<20} {'Graphs':<8} {f'MAE ({unit})':>12} {'RMSE':>12} {'MAPE (%)':>10} {'SMAPE (%)':>10} {'R²':>8} {'Acc@10':>8} {'Acc@20':>8} {'Time(ms)':>8}")
    print("-" * 105)
    
    scale = 1000.0 if args.target == "delay" else 1e-3
    for r in sorted(all_results, key=lambda x: x["File"]):
        print(f"{r['File']:<20} {r['Graphs']:<8} {r['MAE']*scale:>12.2f} {r['RMSE']*scale:>12.2f} {r['MAPE']:>10.2f} {r['SMAPE']:>10.2f} {r['R2']:>8.4f} {r['Acc10']:>7.1f}% {r['Acc20']:>7.1f}% {r['Time_ms']:>8.2f}")
        
    out_file = f"evaluation_custom_dataset_{args.target}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
