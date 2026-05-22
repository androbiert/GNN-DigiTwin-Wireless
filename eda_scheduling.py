import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_scheduling_policies():
    base_dir = "Data_cleaned/SC01/simulations"
    out_dir = "Schu_policy_models/EDA"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Finding data files...")
    files = glob.glob(os.path.join(base_dir, "*", "data.json"))
    
    data_records = []
    
    print("Parsing data and extracting policies...")
    for f in tqdm(files):
        # folder name example: 01)SC01-P=0.01W-S=PF-Q=50KiB
        folder_name = os.path.basename(os.path.dirname(f))
        
        # Extract scheduling policy (S=...)
        policy = "Unknown"
        parts = folder_name.split("-")
        for p in parts:
            if p.startswith("S="):
                policy = p.split("=")[1]
                break
                
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                
            for snap in data:
                for flow in snap.get("flows", []):
                    delay = flow.get("delay", 0.0)
                    rlc = flow.get("rlcDelay", 0.0)
                    tot_delay = (delay + rlc) * 1000 # to ms
                    data_records.append({
                        "Policy": policy,
                        "Delay (ms)": tot_delay
                    })
        except Exception as e:
            continue
            
    if not data_records:
        print("No data found!")
        return
        
    df = pd.DataFrame(data_records)
    print(f"Loaded {len(df)} flow records.")
    
    # Generate statistics
    stats = df.groupby("Policy")["Delay (ms)"].describe(percentiles=[.5, .75, .90, .95, .99])
    print("\nDelay Statistics by Scheduling Policy:")
    print(stats)
    
    stats_file = os.path.join(out_dir, "policy_delay_stats.txt")
    with open(stats_file, "w") as f:
        f.write("Delay Statistics by Scheduling Policy:\n")
        f.write(stats.to_string())
        
    print("Generating Boxplot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Policy", y="Delay (ms)", ax=ax, palette="Set2")
    ax.set_title("Delay Distribution by Scheduling Policy (SC01 Cleaned)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "delay_boxplot_by_policy.png"), dpi=150)
    plt.close()
    
    print("Generating KDE Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in df["Policy"].unique():
        sns.kdeplot(data=df[df["Policy"] == policy], x="Delay (ms)", label=policy, fill=True, alpha=0.3, ax=ax)
    ax.set_title("Density of Delay by Scheduling Policy", fontsize=14)
    ax.set_xlim(0, df["Delay (ms)"].quantile(0.95)) # limit to 95th percentile for better view
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "delay_kde_by_policy.png"), dpi=150)
    plt.close()
    
    print(f"EDA successfully saved in {out_dir}")

if __name__ == "__main__":
    analyze_scheduling_policies()
