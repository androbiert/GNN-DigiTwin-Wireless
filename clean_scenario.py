import os
import sys
import json
import glob
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def scan_single_file(f):
    delays = []
    tputs = []
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
            for snap in data:
                for flow in snap.get("flows", []):
                    delays.append(flow.get("delay", 0) + flow.get("rlcDelay", 0))
                    tputs.append(flow.get("throughput", 0))
    except Exception as e:
        pass
    return delays, tputs

def clean_single_file(args):
    f, src_dir, dst_dir, p90_delay, p90_tput = args
    try:
        rel_path = os.path.relpath(f, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        with open(f, 'r') as fp:
            data = json.load(fp)
            
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
        return True
    except Exception as e:
        return False

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
    
    print(f"[{scenario}] Computing percentiles across {len(files)} files in parallel...")
    # Use ProcessPoolExecutor for parallel file scanning
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(scan_single_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Scanning"):
            delays, tputs = future.result()
            all_delays.extend(delays)
            all_tputs.extend(tputs)
                
    if not all_delays:
        print(f"[{scenario}] No valid flows found.")
        return
        
    p90_delay = np.percentile(all_delays, 90)
    p90_tput = np.percentile(all_tputs, 90)
    
    print(f"[{scenario}] 90th Percentile Delay: {p90_delay*1000:.2f} ms ({p90_delay} s)")
    print(f"[{scenario}] 90th Percentile Throughput: {p90_tput/1000:.2f} kbps ({p90_tput} bps)")
    
    print(f"[{scenario}] Clipping and saving to {dst_dir} in parallel...")
    clean_args = [(f, src_dir, dst_dir, p90_delay, p90_tput) for f in files]
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(clean_single_file, arg): arg[0] for arg in clean_args}
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(files), desc="Cleaning"):
            if future.result():
                success_count += 1
                
    print(f"[{scenario}] Done! Cleaned {success_count}/{len(files)} files in Data_cleaned/{scenario}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and clip simulation data for a scenario in parallel")
    parser.add_argument("scenario", type=str, help="Scenario name (e.g. SC03)")
    args = parser.parse_args()
    clean_scenario(args.scenario)
