"""Quick diagnostic: Why is SC01 Delay MAPE so high?"""
import json, os, glob, numpy as np

root = r"c:\Users\DELL\Desktop\GNN-DigiTwin-Wireless\Data\SC01\simulations"
folders = sorted(glob.glob(os.path.join(root, "*")))

all_delays = []
all_tputs = []

for folder in folders[:5]:  # sample 5 configs
    fpath = os.path.join(folder, "data.json")
    if not os.path.isfile(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for snap in data[:50]:  # sample 50 snapshots per config
        for flow in snap.get("flows", []):
            d = float(flow.get("delay", 0))
            t = float(flow.get("throughput", 0))
            rlc = float(flow.get("rlcDelay", 0))
            total_delay = d + rlc
            if total_delay > 0 or t > 0:
                all_delays.append(total_delay)
                all_tputs.append(t)

delays = np.array(all_delays)
tputs = np.array(all_tputs)

print("=" * 60)
print("DELAY DISTRIBUTION (seconds)")
print("=" * 60)
print(f"  Count:    {len(delays)}")
print(f"  Min:      {delays.min():.10f}")
print(f"  Max:      {delays.max():.6f}")
print(f"  Mean:     {delays.mean():.8f}")
print(f"  Median:   {np.median(delays):.8f}")
print(f"  Std:      {delays.std():.8f}")
print()

# Percentile breakdown
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:02d}:      {np.percentile(delays, p):.10f} s  ({np.percentile(delays, p)*1000:.6f} ms)")

print()
print(f"  Zeros:    {(delays == 0).sum()} ({(delays == 0).mean()*100:.1f}%)")
print(f"  < 1e-6:   {(delays < 1e-6).sum()} ({(delays < 1e-6).mean()*100:.1f}%)")
print(f"  < 1e-4:   {(delays < 1e-4).sum()} ({(delays < 1e-4).mean()*100:.1f}%)")
print(f"  < 1e-3:   {(delays < 1e-3).sum()} ({(delays < 1e-3).mean()*100:.1f}%)")
print(f"  < 1e-2:   {(delays < 1e-2).sum()} ({(delays < 1e-2).mean()*100:.1f}%)")
print(f"  > 0.1:    {(delays > 0.1).sum()} ({(delays > 0.1).mean()*100:.1f}%)")
print(f"  > 1.0:    {(delays > 1.0).sum()} ({(delays > 1.0).mean()*100:.1f}%)")

# Dynamic range
nonzero = delays[delays > 0]
if len(nonzero) > 0:
    print(f"\n  Dynamic range (max/min of nonzero): {nonzero.max()/nonzero.min():.0f}x")
    print(f"  Orders of magnitude: {np.log10(nonzero.max()/nonzero.min()):.1f}")

print()
print("=" * 60)
print("THROUGHPUT DISTRIBUTION (bps)")
print("=" * 60)
print(f"  Count:    {len(tputs)}")
print(f"  Min:      {tputs.min():.2f}")
print(f"  Max:      {tputs.max():.2f}")
print(f"  Mean:     {tputs.mean():.2f}")
print(f"  Median:   {np.median(tputs):.2f}")
print(f"  Std:      {tputs.std():.2f}")
print(f"  Zeros:    {(tputs == 0).sum()} ({(tputs == 0).mean()*100:.1f}%)")
print(f"  CoV:      {tputs.std()/tputs.mean():.3f}")

print()
print("=" * 60)
print("MAPE SENSITIVITY ANALYSIS")
print("=" * 60)
# Simulate: if model predicts mean delay for all, what MAPE?
mean_delay = delays.mean()
eps = 1e-6
mask = delays > eps
simulated_mape = np.mean(np.abs(mean_delay - delays[mask]) / delays[mask])
print(f"  If model predicts MEAN ({mean_delay*1000:.4f} ms) for all:")
print(f"    MAPE = {simulated_mape*100:.1f}%")

median_delay = np.median(delays)
simulated_mape2 = np.mean(np.abs(median_delay - delays[mask]) / delays[mask])
print(f"  If model predicts MEDIAN ({median_delay*1000:.4f} ms) for all:")
print(f"    MAPE = {simulated_mape2*100:.1f}%")

# Perfect predictor with 5ms noise
noise = np.random.normal(0, 0.005, size=mask.sum())
simulated_mape3 = np.mean(np.abs(noise) / delays[mask])
print(f"  If PERFECT predictor + 5ms gaussian noise:")
print(f"    MAPE = {simulated_mape3*100:.1f}%")
