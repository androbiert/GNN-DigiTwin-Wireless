import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_clipped_delays(root_dir):
    data_dir = os.path.join(root_dir, "Data")
    scenarios = [d for d in os.listdir(data_dir) if d.startswith("SC")]
    scenarios.sort()

    out_dir = os.path.join(root_dir, "eda_results", "delay_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    scenario_delays = {}
    all_delays_global = []
    
    for sc in scenarios:
        sc_dir = os.path.join(data_dir, sc, "simulations")
        if not os.path.isdir(sc_dir):
            continue
            
        folders = sorted(glob.glob(os.path.join(sc_dir, "*")))
        all_delays = []
        for folder in folders:
            fpath = os.path.join(folder, "data.json")
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for snap in data:
                        for flow in snap.get("flows", []):
                            d = float(flow.get("delay", 0))
                            rlc = float(flow.get("rlcDelay", 0))
                            total_delay = d + rlc
                            if total_delay > 0:
                                all_delays.append(total_delay)
                except Exception:
                    pass
        
        delays = np.array(all_delays) * 1000  # convert to ms
        if len(delays) > 0:
            scenario_delays[sc] = delays
            all_delays_global.extend(delays)

    all_delays_global = np.array(all_delays_global)
    if len(all_delays_global) == 0:
        return
        
    p95 = np.percentile(all_delays_global, 95)
    p99 = np.percentile(all_delays_global, 99)
    print(f"Global 95th percentile: {p95:.2f} ms")
    print(f"Global 99th percentile: {p99:.2f} ms")

    # Let's clip at P99 (e.g. remove the most extreme 1%)
    # Wait, the user said "Clipping (on remove the outliers)". 
    # We will literally filter them out for visualization, or use np.clip. Let's do both or just clip.
    clip_threshold = p99
    
    scenario_clipped = {}
    for sc, arr in scenario_delays.items():
        # Clip the data
        clipped = np.clip(arr, a_min=0, a_max=clip_threshold)
        scenario_clipped[sc] = clipped
    
    # Plotting the clearly visible distribution after clipping
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. KDE Plot showing density up to P99
    for sc, arr in scenario_clipped.items():
        sns.kdeplot(arr, ax=axes[0], label=sc, fill=True, alpha=0.3)
    
    axes[0].set_title(f"Density of Delay (Clipped at P99: {clip_threshold:.1f} ms)")
    axes[0].set_xlabel("Delay (ms)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    # 2. Boxplot (which now shouldn't have infinite tails)
    data_list = [scenario_clipped[sc] for sc in scenarios if sc in scenario_clipped]
    labels = [sc for sc in scenarios if sc in scenario_clipped]
    
    sns.boxplot(data=data_list, ax=axes[1], palette="Set2")
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Delay (ms)")
    axes[1].set_title(f"Boxplot of Delay (Clipped at P99: {clip_threshold:.1f} ms)")
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "delay_clipped_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Saved clipped distribution to {out_path}")

if __name__ == "__main__":
    root = r"c:\Users\DELL\Desktop\GNN-DigiTwin-Wireless"
    analyze_clipped_delays(root)
