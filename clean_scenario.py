import os
import sys
import json
import glob
import numpy as np
from tqdm import tqdm
import argparse

def clean_scenario(scenario: str):
    src_dir = f"Data/{scenario}/simulations"
    dst_dir = f"Data_cleaned/{scenario}/simulations"
    
    if not os.path.exists(src_dir):
        print(f"Source {src_dir} not found.")
        return
        
    print(f"[{scenario}] Finding files in {src_dir}...")
    files = glob.glob(os.path.join(src_dir, "*", "data.json"))
    
    all_delays = []
    all_tputs = []
    
    print(f"[{scenario}] Computing percentiles across {len(files)} files...")
    for f in tqdm(files, desc="Scanning"):
        with open(f, 'r') as fp:
            try:
                data = json.load(fp)
                for snap in data:
                    for flow in snap.get("flows", []):
                        all_delays.append(flow.get("delay", 0) + flow.get("rlcDelay", 0))
                        all_tputs.append(flow.get("throughput", 0))
            except Exception as e:
                pass
                
    if not all_delays:
        print(f"[{scenario}] No valid flows found.")
        return
        
    p90_delay = np.percentile(all_delays, 90)
    p90_tput = np.percentile(all_tputs, 90)
    
    print(f"[{scenario}] 90th Percentile Delay: {p90_delay*1000:.2f} ms ({p90_delay} s)")
    print(f"[{scenario}] 90th Percentile Throughput: {p90_tput/1000:.2f} kbps ({p90_tput} bps)")
    
    print(f"[{scenario}] Clipping and saving to {dst_dir}...")
    for f in tqdm(files, desc="Cleaning"):
        rel_path = os.path.relpath(f, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        with open(f, 'r') as fp:
            try:
                data = json.load(fp)
            except:
                continue
                
        for snap in data:
            for flow in snap.get("flows", []):
                # Clip delay
                orig_d = flow.get("delay", 0)
                orig_rlc = flow.get("rlcDelay", 0)
                tot = orig_d + orig_rlc
                if tot > p90_delay:
                    scale = p90_delay / tot if tot > 0 else 0
                    flow["delay"] = orig_d * scale
                    flow["rlcDelay"] = orig_rlc * scale
                    
                # Clip throughput
                orig_tput = flow.get("throughput", 0)
                if orig_tput > p90_tput:
                    flow["throughput"] = p90_tput
                    
        with open(dst_path, 'w') as fp:
            json.dump(data, fp)

    print(f"[{scenario}] Done! Cleaned dataset is in Data_cleaned/{scenario}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and clip simulation data for a scenario")
    parser.add_argument("scenario", type=str, help="Scenario name (e.g. SC03)")
    args = parser.parse_args()
    clean_scenario(args.scenario)
