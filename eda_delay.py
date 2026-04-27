import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wireless_gnn.dataset import load_all_snapshots

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading datasets...")
    # Load all graph snapshots
    all_graphs = load_all_snapshots(root_dir)
    
    # Extract delay data and scenario information
    delays = []
    scenarios = []
    
    for g in all_graphs:
        # target_delay is a list/array of delays for flows in this snapshot
        d = g["target_delay"]
        delays.extend(d)
        scenarios.extend([g["scenario"]] * len(d))
        
    delays = np.array(delays)
    
    print("\n" + "="*40)
    print("--- Delay Data Statistics ---")
    print("="*40)
    print(f"Total delay samples : {len(delays):,}")
    print(f"Mean Delay          : {np.mean(delays)*1000:.3f} ms")
    print(f"Median Delay        : {np.median(delays)*1000:.3f} ms")
    print(f"Std Dev Delay       : {np.std(delays)*1000:.3f} ms")
    print(f"Min Delay           : {np.min(delays)*1000:.3f} ms")
    print(f"Max Delay           : {np.max(delays)*1000:.3f} ms")
    print(f"90th Percentile     : {np.percentile(delays, 90)*1000:.3f} ms")
    print(f"95th Percentile     : {np.percentile(delays, 95)*1000:.3f} ms")
    print(f"99th Percentile     : {np.percentile(delays, 99)*1000:.3f} ms")
    
    # Create EDA directory
    eda_dir = os.path.join(root_dir, "eda_results")
    os.makedirs(eda_dir, exist_ok=True)
    
    # 1. Histogram of all delays
    plt.figure(figsize=(10, 6))
    sns.histplot(delays * 1000, bins=50, kde=True, color='blue')
    plt.title("Distribution of Delays (Overall)")
    plt.xlabel("Delay (ms)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(eda_dir, "delay_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Boxplot by Scenario
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=scenarios, y=delays * 1000, hue=scenarios, palette="Set2", legend=False)
    plt.title("Delay Distribution per Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Delay (ms)")
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "delay_by_scenario_box.png"), dpi=150)
    plt.close()

    # 3. Violin plot by Scenario
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=scenarios, y=delays * 1000, hue=scenarios, palette="Set3", legend=False)
    plt.title("Delay Violin Plot per Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Delay (ms)")
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "delay_by_scenario_violin.png"), dpi=150)
    plt.close()

    print(f"\nEDA plots have been saved to the folder: {eda_dir}")

if __name__ == '__main__':
    main()
