import json
import glob
import numpy as np

files = glob.glob("data/SC01/*/data.json")
if not files:
    print("No files found!")
else:
    print(f"Found {len(files)} files.")
    delays = []
    tputs = []
    
    for f in files[:5]: # just check first 5
        with open(f) as fp:
            data = json.load(fp)
            for flow in data.get("flows", []):
                delays.append(flow.get("delay", 0))
                tputs.append(flow.get("throughput", 0))
                
    if delays:
        delays = np.array(delays)
        tputs = np.array(tputs)
        print(f"Delay: min={delays.min():.6f}, max={delays.max():.6f}, mean={delays.mean():.6f}")
        print(f"Tput: min={tputs.min():.6f}, max={tputs.max():.6f}, mean={tputs.mean():.6f}")
