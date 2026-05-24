import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_single_file(fpath, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from {fpath}...")
    all_delays = []
    
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for snap in data:
            for flow in snap.get("flows", []):
                d = float(flow.get("delay", 0))
                rlc = float(flow.get("rlcDelay", 0))
                total_delay = d + rlc
                if total_delay > 0:
                    all_delays.append(total_delay)
        
    delays = np.array(all_delays) * 1000  # convert to ms
    if len(delays) == 0:
        print("No delays found.")
        return
        
    p95 = np.percentile(delays, 95)
    p99 = np.percentile(delays, 99)
    print(f"Total samples: {len(delays):,}")
    print(f"95th percentile: {p95:.2f} ms")
    print(f"99th percentile: {p99:.2f} ms")
    print(f"Max: {delays.max():.2f} ms")

    clip_threshold = p99
    
    clipped = np.clip(delays, a_min=0, a_max=clip_threshold)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KDE
    sns.kdeplot(clipped, ax=axes[0], fill=True, alpha=0.5, color="coral")
    axes[0].set_title(f"Density of Delay (Clipped at P99: {clip_threshold:.1f} ms)")
    axes[0].set_xlabel("Delay (ms)")
    axes[0].set_ylabel("Density")
    
    # Histogram
    sns.histplot(clipped, bins=50, ax=axes[1], color="teal", kde=False)
    axes[1].set_title(f"Histogram of Delay (Clipped at P99: {clip_threshold:.1f} ms)")
    axes[1].set_xlabel("Delay (ms)")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "single_file_clipped_delay.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Saved clipped distribution to {out_path}")

if __name__ == "__main__":
    file_path = r"c:\Users\DELL\Desktop\GNN-DigiTwin-Wireless\Data\SC01\simulations\sim_config_001\data.json"
    out = r"c:\Users\DELL\Desktop\GNN-DigiTwin-Wireless\eda_results\delay_diagnostics"
    analyze_single_file(file_path, out)
