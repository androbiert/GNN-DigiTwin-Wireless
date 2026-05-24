import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_delays(root_dir):
    data_dir = os.path.join(root_dir, "Data")
    scenarios = [d for d in os.listdir(data_dir) if d.startswith("SC")]
    scenarios.sort()

    out_dir = os.path.join(root_dir, "eda_results", "delay_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("DELAY EDA PER SCENARIO")
    print("=" * 80)

    scenario_delays = {}
    
    for sc in scenarios:
        sc_dir = os.path.join(data_dir, sc, "simulations")
        if not os.path.isdir(sc_dir):
            continue
            
        folders = sorted(glob.glob(os.path.join(sc_dir, "*")))
        
        all_delays = []
        for folder in folders:  # read all configs for better stats
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
                except Exception as e:
                    pass
        
        delays = np.array(all_delays)
        if len(delays) == 0:
            continue
            
        scenario_delays[sc] = delays
        
        print(f"\n[{sc}] DELAY DISTRIBUTION (seconds)")
        print("-" * 50)
        print(f"  Count:    {len(delays):,}")
        print(f"  Min:      {delays.min():.10f}")
        print(f"  Max:      {delays.max():.6f}")
        print(f"  Mean:     {delays.mean():.6f}")
        print(f"  Median:   {np.median(delays):.6f}")
        print(f"  Std:      {delays.std():.6f}")
        print(f"  CoV:      {delays.std()/delays.mean():.3f} (Coefficient of Variation)")
        
        # Percentiles
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  P{p:02d}:      {np.percentile(delays, p)*1000:10.4f} ms")
            
        print(f"  < 5 ms:   {(delays < 0.005).sum() / len(delays) * 100:.2f}%")
        print(f"  > 100 ms: {(delays > 0.1).sum() / len(delays) * 100:.2f}%")
        print(f"  > 1 s:    {(delays > 1.0).sum() / len(delays) * 100:.2f}%")
        
        dyn_range = delays.max() / delays.min()
        print(f"  Dynamic Range: {dyn_range:.1f}x ({np.log10(dyn_range):.2f} orders of magnitude)")
        print()

    # Visualizations
    if not scenario_delays:
        print("No delay data found.")
        return

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_list = [scenario_delays[sc] * 1000 for sc in scenarios if sc in scenario_delays] # in ms
    labels = [sc for sc in scenarios if sc in scenario_delays]
    
    sns.boxplot(data=data_list, ax=ax, showfliers=False, palette="Set2")
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Delay Distribution per Scenario (Without Outliers)")
    plt.savefig(os.path.join(out_dir, "delay_boxplot_no_outliers.png"), dpi=150)
    plt.close()

    # Violinplot (Log Scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=data_list, ax=ax, palette="Set2", log_scale=True)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delay (ms) - Log Scale")
    ax.set_yscale('log')
    ax.set_title("Delay Distribution per Scenario (Log Scale)")
    plt.savefig(os.path.join(out_dir, "delay_violin_log_scale.png"), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {out_dir}")

if __name__ == "__main__":
    root = r"c:\Users\DELL\Desktop\GNN-DigiTwin-Wireless"
    analyze_delays(root)
