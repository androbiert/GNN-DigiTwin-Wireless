import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from wireless_gnn.scenario_registry import discover_scenarios, filter_for_target
from wireless_gnn.dataset import load_scenario_snapshots

def main():
    print("=" * 70)
    print("  EDA: Delay vs Power (tx_power) & Scheduler")
    print("=" * 70)

    data_dir = "Data_cleaned"
    print(f"Discovering scenarios in {data_dir}...")
    all_configs = discover_scenarios(_project_root, data_dir=data_dir, validate=True, verbose=False)
    
    # We only care about configs that have delay
    delay_configs = filter_for_target(all_configs, "delay")
    
    if not delay_configs:
        print("No delay configurations found!")
        return
        
    records = []
    
    print(f"Loading data for {len(delay_configs)} configurations to extract delay and power...")
    
    for cfg in delay_configs:
        # Load the raw snapshots without outlier filtering to see the true distribution
        graphs = load_scenario_snapshots(
            data_paths=[cfg.data_path],
            scenario_id=cfg.scenario_id,
            target="delay",
            filter_outliers=False,
            verbose=False
        )
        
        # Get raw power value (e.g., "0.01W" -> 0.01)
        try:
            power_val = float(cfg.tx_power.replace('W', ''))
        except ValueError:
            power_val = cfg.tx_power
            
        for g in graphs:
            delays = g["target_delay"] * 1000.0  # Convert to ms
            for d in delays:
                records.append({
                    "Scenario": cfg.scenario_id,
                    "Scheduler": cfg.scheduler,
                    "Power (W)": power_val,
                    "Power Label": cfg.tx_power,
                    "Delay (ms)": d
                })
                
    if not records:
        print("No delay data extracted.")
        return
        
    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} total delay samples.")
    
    out_dir = os.path.join(_project_root, "eda_results", "deep")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Boxplot: Delay vs Power (Overall)
    print("Plotting Delay vs Power (Overall Boxplot)...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Power Label", y="Delay (ms)", showfliers=False, order=sorted(df['Power Label'].unique()))
    plt.title("Distribution of Delay vs. Transmission Power (Outliers Hidden)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out1 = os.path.join(out_dir, "delay_vs_power_boxplot.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    
    # 2. Violinplot: Delay vs Power grouped by Scheduler
    print("Plotting Delay vs Power grouped by Scheduler...")
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=df, x="Power Label", y="Delay (ms)", hue="Scheduler", cut=0, scale="width", order=sorted(df['Power Label'].unique()))
    plt.title("Distribution of Delay vs. Transmission Power (Grouped by Scheduler)")
    plt.legend(title="Scheduler", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    out2 = os.path.join(out_dir, "delay_vs_power_scheduler_violin.png")
    plt.savefig(out2, dpi=150)
    plt.close()

    # 3. Summary Statistics Table
    print("\nSummary Statistics: Mean Delay (ms) grouped by Power and Scheduler")
    summary = df.groupby(["Power Label", "Scheduler"])["Delay (ms)"].agg(['count', 'mean', 'std', 'median']).reset_index()
    print("-" * 75)
    print(f"{'Power':<10} {'Scheduler':<15} {'Samples':<10} {'Mean (ms)':<12} {'Std (ms)':<12} {'Median (ms)':<12}")
    print("-" * 75)
    for _, row in summary.iterrows():
        print(f"{row['Power Label']:<10} {row['Scheduler']:<15} {int(row['count']):<10} {row['mean']:<12.3f} {row['std']:<12.3f} {row['median']:<12.3f}")
    
    # Save summary to CSV
    csv_out = os.path.join(out_dir, "delay_power_summary.csv")
    summary.to_csv(csv_out, index=False)
    
    print(f"\nAll plots and summaries saved to {out_dir}")
    
if __name__ == "__main__":
    main()
