"""
gpu_inference_profiler.py — GPU Resource Profiler for WirelessNet-Fermi Inference

Monitors and reports detailed GPU metrics during inference:
  - VRAM allocation (current, peak, reserved)
  - GPU utilisation % and SM clock speed
  - Temperature and power draw
  - Per-graph inference latency (with CUDA events for accurate timing)
  - Per-layer breakdown (optional, via PyTorch profiler)
  - Model parameter memory footprint
  - Throughput in graphs/sec and flows/sec

Usage:
  # Basic profiling with a checkpoint
  python gpu_inference_profiler.py --checkpoint checkpoints_v3/SC03/throughput/best.pt \
        --data-dir data_cleaned --scenario SC03 --target throughput

  # With detailed per-layer profiling
  python gpu_inference_profiler.py --checkpoint checkpoints_v3/SC03/throughput/best.pt \
        --data-dir data_cleaned --scenario SC03 --target throughput --profile-layers

  # Save report as JSON
  python gpu_inference_profiler.py --checkpoint checkpoints_v3/SC03/throughput/best.pt \
        --data-dir data_cleaned --scenario SC03 --target throughput --save-json gpu_report.json

  # Compare teacher vs distilled student
  python gpu_inference_profiler.py --checkpoint checkpoints/distilled_delay/best.pt \
        --data-dir data_cleaned --scenario SC03 --target delay --is-student
"""

import sys
import os
import argparse
import time
import json
import glob
import threading
import numpy as np
import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluate_models import load_model_from_checkpoint
from wireless_gnn.dataset import build_scenario_datasets, collate_fn


# --------------------------------------------------------------------------- #
# GPU Info Utilities
# --------------------------------------------------------------------------- #

def get_gpu_properties(device_idx: int = 0) -> dict:
    """Return static GPU properties (name, memory, compute capability)."""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(device_idx)
    return {
        "available": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_GB": round(props.total_mem / (1024 ** 3), 2),
        "sm_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "pytorch_version": torch.__version__,
    }


def get_gpu_dynamic_stats(device_idx: int = 0) -> dict:
    """Return live GPU stats (utilisation, temperature, power) via nvidia-smi."""
    stats = {}
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_idx}",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,"
                "power.draw,power.limit,clocks.sm,clocks.mem,fan.speed",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 8:
                stats = {
                    "gpu_util_pct": float(parts[0]) if parts[0] != "[N/A]" else None,
                    "mem_util_pct": float(parts[1]) if parts[1] != "[N/A]" else None,
                    "temperature_C": float(parts[2]) if parts[2] != "[N/A]" else None,
                    "power_draw_W": float(parts[3]) if parts[3] != "[N/A]" else None,
                    "power_limit_W": float(parts[4]) if parts[4] != "[N/A]" else None,
                    "sm_clock_MHz": float(parts[5]) if parts[5] != "[N/A]" else None,
                    "mem_clock_MHz": float(parts[6]) if parts[6] != "[N/A]" else None,
                    "fan_speed_pct": float(parts[7]) if parts[7] != "[N/A]" else None,
                }
    except Exception:
        pass
    return stats


def get_vram_stats(device_idx: int = 0) -> dict:
    """Return VRAM allocation stats from PyTorch."""
    return {
        "allocated_MB": round(torch.cuda.memory_allocated(device_idx) / (1024 ** 2), 2),
        "reserved_MB": round(torch.cuda.memory_reserved(device_idx) / (1024 ** 2), 2),
        "max_allocated_MB": round(torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2), 2),
        "max_reserved_MB": round(torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2), 2),
    }


def get_model_memory_footprint(model: torch.nn.Module) -> dict:
    """Estimate the model parameter + buffer memory footprint."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable,
        "param_memory_MB": round(param_bytes / (1024 ** 2), 4),
        "buffer_memory_MB": round(buffer_bytes / (1024 ** 2), 4),
        "total_footprint_MB": round((param_bytes + buffer_bytes) / (1024 ** 2), 4),
    }


# --------------------------------------------------------------------------- #
# Background GPU Sampler  (polls nvidia-smi during inference)
# --------------------------------------------------------------------------- #

class GPUSampler:
    """Lightweight background thread that samples GPU stats at a given interval."""

    def __init__(self, device_idx: int = 0, interval_sec: float = 0.25):
        self.device_idx = device_idx
        self.interval = interval_sec
        self.samples = []
        self._stop = threading.Event()

    def _poll(self):
        while not self._stop.is_set():
            sample = get_gpu_dynamic_stats(self.device_idx)
            sample["timestamp"] = time.perf_counter()
            # Also sample PyTorch VRAM
            sample["vram_allocated_MB"] = round(
                torch.cuda.memory_allocated(self.device_idx) / (1024 ** 2), 2
            )
            self.samples.append(sample)
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        def _safe_stat(key):
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            if not vals:
                return None
            return {"min": min(vals), "max": max(vals), "mean": round(sum(vals) / len(vals), 2)}

        return {
            "n_samples": len(self.samples),
            "gpu_util_pct": _safe_stat("gpu_util_pct"),
            "mem_util_pct": _safe_stat("mem_util_pct"),
            "temperature_C": _safe_stat("temperature_C"),
            "power_draw_W": _safe_stat("power_draw_W"),
            "sm_clock_MHz": _safe_stat("sm_clock_MHz"),
            "vram_allocated_MB": _safe_stat("vram_allocated_MB"),
        }


# --------------------------------------------------------------------------- #
# Inference Benchmark
# --------------------------------------------------------------------------- #

@torch.no_grad()
def benchmark_inference(model, graphs, device, n_warmup=10):
    """
    Run inference on a list of graphs and measure per-graph latency
    using CUDA events for accurate GPU timing.

    Returns:
        latencies_ms: list of per-graph latencies in milliseconds
        total_flows: total flows processed
    """
    model.eval()

    # ── Warmup ──
    for i in range(min(n_warmup, len(graphs))):
        _ = model(graphs[i])
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    total_flows = 0

    for graph in graphs:
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        pred, _ = model(graph)

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            # CPU fallback (less precise)
            elapsed_ms = 0.0  # handled separately for CPU

        latencies_ms.append(elapsed_ms)
        total_flows += pred.shape[0]

    return latencies_ms, total_flows


@torch.no_grad()
def benchmark_inference_cpu_fallback(model, graphs, n_warmup=5):
    """CPU timing fallback using time.perf_counter."""
    model.eval()

    for i in range(min(n_warmup, len(graphs))):
        _ = model(graphs[i])

    latencies_ms = []
    total_flows = 0

    for graph in graphs:
        t0 = time.perf_counter()
        pred, _ = model(graph)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        total_flows += pred.shape[0]

    return latencies_ms, total_flows


# --------------------------------------------------------------------------- #
# Per-Layer Profiling  (optional --profile-layers)
# --------------------------------------------------------------------------- #

def profile_layers(model, sample_graphs, device, n_graphs=5):
    """
    Use torch.profiler to get a per-layer breakdown.
    Returns a sorted list of (layer_name, self_cuda_time_ms) tuples.
    """
    from torch.profiler import profile as torch_profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    graphs_to_profile = sample_graphs[:n_graphs]

    with torch_profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for g in graphs_to_profile:
                _ = model(g)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Extract key averages
    table = prof.key_averages()
    results = []
    for evt in table:
        name = evt.key
        if device.type == "cuda":
            t_ms = evt.self_cuda_time_total / 1000.0  # μs → ms
        else:
            t_ms = evt.self_cpu_time_total / 1000.0
        mem_mb = (evt.self_cpu_memory_usage or 0) / (1024 ** 2)
        results.append({
            "name": name,
            "calls": evt.count,
            "self_time_ms": round(t_ms, 4),
            "cpu_time_ms": round(evt.self_cpu_time_total / 1000.0, 4),
            "memory_MB": round(mem_mb, 4),
        })

    # Sort by self_time_ms descending
    results.sort(key=lambda x: x["self_time_ms"], reverse=True)
    return results


# --------------------------------------------------------------------------- #
# Pretty Printing
# --------------------------------------------------------------------------- #

def _hr(char="─", width=72):
    return char * width

def _section(title, char="═"):
    pad = (70 - len(title)) // 2
    return f"\n{char * pad} {title} {char * pad}"


def print_report(report: dict):
    """Print a rich, formatted GPU inference report to the console."""

    print(_section("GPU INFERENCE PROFILER REPORT"))

    # ── GPU Hardware ──
    hw = report.get("gpu_hardware", {})
    if hw.get("available"):
        print(f"\n  GPU Device       : {hw['name']}")
        print(f"  Compute Cap.     : {hw['compute_capability']}")
        print(f"  Total VRAM       : {hw['total_memory_GB']} GB")
        print(f"  SM Count         : {hw['sm_count']}")
        print(f"  CUDA / cuDNN     : {hw['cuda_version']} / {hw.get('cudnn_version', 'N/A')}")
        print(f"  PyTorch          : {hw['pytorch_version']}")
    else:
        print("\n  ⚠  No CUDA GPU detected — running on CPU")

    # ── Model Footprint ──
    fp = report.get("model_footprint", {})
    print(f"\n{_hr()}")
    print(f"  Model Architecture : {report.get('architecture', '?')}")
    print(f"  Total Parameters   : {fp.get('total_params', 0):,}")
    print(f"  Trainable Params   : {fp.get('trainable_params', 0):,}")
    print(f"  Param Memory       : {fp.get('param_memory_MB', 0):.2f} MB")
    print(f"  Buffer Memory      : {fp.get('buffer_memory_MB', 0):.2f} MB")
    print(f"  Total Footprint    : {fp.get('total_footprint_MB', 0):.2f} MB")

    # ── VRAM During Inference ──
    vram = report.get("vram_during_inference", {})
    if vram:
        print(f"\n{_hr()}")
        print("  VRAM During Inference:")
        print(f"    After model load : {vram.get('after_model_load_MB', '?')} MB")
        print(f"    Peak allocated   : {vram.get('peak_allocated_MB', '?')} MB")
        print(f"    Peak reserved    : {vram.get('peak_reserved_MB', '?')} MB")
        inference_delta = vram.get("inference_delta_MB")
        if inference_delta is not None:
            print(f"    Inference Δ      : +{inference_delta} MB  (activation tensors)")

    # ── GPU Utilisation (sampled) ──
    sampled = report.get("gpu_sampled_stats", {})
    if sampled:
        print(f"\n{_hr()}")
        print("  GPU Stats (sampled during inference):")
        for key, label in [
            ("gpu_util_pct", "GPU Utilisation"),
            ("mem_util_pct", "Memory Util."),
            ("temperature_C", "Temperature"),
            ("power_draw_W", "Power Draw"),
            ("sm_clock_MHz", "SM Clock"),
            ("vram_allocated_MB", "VRAM Allocated"),
        ]:
            stat = sampled.get(key)
            if stat:
                unit = {"gpu_util_pct": "%", "mem_util_pct": "%", "temperature_C": "°C",
                        "power_draw_W": "W", "sm_clock_MHz": "MHz", "vram_allocated_MB": "MB"}.get(key, "")
                print(f"    {label:<20}: min={stat['min']}{unit}  avg={stat['mean']}{unit}  max={stat['max']}{unit}")

    # ── Latency Stats ──
    lat = report.get("latency", {})
    if lat:
        print(f"\n{_hr()}")
        print("  Inference Latency:")
        print(f"    Graphs profiled  : {lat.get('n_graphs', '?')}")
        print(f"    Total flows      : {lat.get('total_flows', '?'):,}")
        print(f"    Mean latency     : {lat.get('mean_ms', 0):.3f} ms/graph")
        print(f"    Median latency   : {lat.get('median_ms', 0):.3f} ms/graph")
        print(f"    Std deviation    : {lat.get('std_ms', 0):.3f} ms")
        print(f"    P90 latency      : {lat.get('p90_ms', 0):.3f} ms")
        print(f"    P95 latency      : {lat.get('p95_ms', 0):.3f} ms")
        print(f"    P99 latency      : {lat.get('p99_ms', 0):.3f} ms")
        print(f"    Min / Max        : {lat.get('min_ms', 0):.3f} / {lat.get('max_ms', 0):.3f} ms")
        thr = lat.get("throughput", {})
        if thr:
            print(f"    Throughput       : {thr.get('graphs_per_sec', 0):.1f} graphs/sec"
                  f"  |  {thr.get('flows_per_sec', 0):.0f} flows/sec")

    # ── Per-Layer Breakdown ──
    layers = report.get("layer_profile", [])
    if layers:
        print(f"\n{_hr()}")
        print("  Per-Layer Time Breakdown (top 20):")
        print(f"    {'Layer':<55} {'Calls':>6} {'Self Time (ms)':>14}")
        print(f"    {'─' * 55} {'─' * 6} {'─' * 14}")
        for entry in layers[:20]:
            name = entry["name"][:55]
            print(f"    {name:<55} {entry['calls']:>6} {entry['self_time_ms']:>14.4f}")

    print(f"\n{'═' * 72}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="GPU Inference Profiler for WirelessNet-Fermi models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data-dir", default="data_cleaned", help="Data directory")
    parser.add_argument("--scenario", default="SC03", help="Scenario ID")
    parser.add_argument("--target", default="throughput", choices=["delay", "throughput"])
    parser.add_argument("--n-graphs", type=int, default=500,
                        help="Number of graphs to benchmark (default 500)")
    parser.add_argument("--profile-layers", action="store_true",
                        help="Enable per-layer profiling via torch.profiler")
    parser.add_argument("--save-json", default=None,
                        help="Save the full report as a JSON file")
    parser.add_argument("--is-student", action="store_true",
                        help="Checkpoint is a distilled student model")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu', 'cuda'")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    device_idx = device.index or 0 if device.type == "cuda" else 0

    print(f"\n{'═' * 72}")
    print(f"  GPU INFERENCE PROFILER  —  WirelessNet-Fermi")
    print(f"{'═' * 72}")
    print(f"  Device         : {device}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Scenario       : {args.scenario}")
    print(f"  Target         : {args.target}")
    print(f"  Graphs to test : {args.n_graphs}")

    report = {}

    # ── 1. GPU Hardware Info ──
    report["gpu_hardware"] = get_gpu_properties(device_idx) if device.type == "cuda" else {"available": False}

    # ── 2. Load Model ──
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_idx)

    if args.is_student:
        # Student checkpoint has different structure
        from wireless_gnn.model2 import WirelessNetFermiV3
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = WirelessNetFermiV3(
            hidden_dim=cfg.get("student_dim", 32),
            num_heads=cfg.get("student_heads", 2),
            iterations=cfg.get("student_iters", 3),
            target=args.target,
        ).to(device)
        model.load_state_dict(ckpt["student"])
        arch_name = f"WirelessNetFermiV3-Student (d={cfg.get('student_dim', 32)}, " \
                    f"h={cfg.get('student_heads', 2)}, K={cfg.get('student_iters', 3)})"
    else:
        model, arch_name, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    model.eval()
    report["architecture"] = arch_name
    report["model_footprint"] = get_model_memory_footprint(model)

    vram_report = {}
    if device.type == "cuda":
        vram_report["after_model_load_MB"] = round(
            torch.cuda.memory_allocated(device_idx) / (1024 ** 2), 2
        )

    # ── 3. Load Dataset ──
    sc = args.scenario.upper()
    ckpt_dir = os.path.dirname(args.checkpoint)

    data_paths = sorted(glob.glob(os.path.join(args.data_dir, sc, "simulations", "*", "data.json")))
    if not data_paths:
        print(f"  ERROR: No data files found for {sc} in {args.data_dir}")
        sys.exit(1)

    # Use first config only
    data_paths = [data_paths[0]]
    print(f"  Data file      : {data_paths[0]}")

    _, _, test_ds, normalizer = build_scenario_datasets(
        data_paths=data_paths,
        scenario_id=sc,
        target=args.target,
        seed=42,
        split_dir=ckpt_dir,
    )

    if "normalizer" in ckpt:
        normalizer.load_state(ckpt["normalizer"])

    n_available = len(test_ds)
    n_use = min(args.n_graphs, n_available)
    indices = np.random.RandomState(42).choice(n_available, n_use, replace=False)
    graphs = [test_ds[i] for i in indices]
    print(f"  Test graphs    : {n_use} / {n_available} available")

    # ── 4. Reset memory stats & start GPU sampler ──
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_idx)

    sampler = None
    if device.type == "cuda":
        sampler = GPUSampler(device_idx=device_idx, interval_sec=0.2)
        sampler.start()

    # ── 5. Benchmark Inference ──
    print(f"\n  Running inference benchmark ({n_use} graphs)...")

    if device.type == "cuda":
        latencies_ms, total_flows = benchmark_inference(model, graphs, device, n_warmup=10)
    else:
        latencies_ms, total_flows = benchmark_inference_cpu_fallback(model, graphs, n_warmup=5)

    # Stop sampler
    if sampler:
        sampler.stop()
        report["gpu_sampled_stats"] = sampler.summary()

    # ── 6. VRAM after inference ──
    if device.type == "cuda":
        vram_report["peak_allocated_MB"] = round(
            torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2), 2
        )
        vram_report["peak_reserved_MB"] = round(
            torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2), 2
        )
        vram_report["inference_delta_MB"] = round(
            vram_report["peak_allocated_MB"] - vram_report["after_model_load_MB"], 2
        )
        report["vram_during_inference"] = vram_report

    # ── 7. Latency Statistics ──
    lats = np.array(latencies_ms)
    total_time_sec = lats.sum() / 1000.0
    report["latency"] = {
        "n_graphs": n_use,
        "total_flows": int(total_flows),
        "mean_ms": round(float(lats.mean()), 4),
        "median_ms": round(float(np.median(lats)), 4),
        "std_ms": round(float(lats.std()), 4),
        "min_ms": round(float(lats.min()), 4),
        "max_ms": round(float(lats.max()), 4),
        "p90_ms": round(float(np.percentile(lats, 90)), 4),
        "p95_ms": round(float(np.percentile(lats, 95)), 4),
        "p99_ms": round(float(np.percentile(lats, 99)), 4),
        "throughput": {
            "graphs_per_sec": round(n_use / total_time_sec, 2) if total_time_sec > 0 else 0,
            "flows_per_sec": round(total_flows / total_time_sec, 2) if total_time_sec > 0 else 0,
        },
    }

    # ── 8. Per-Layer Profiling (optional) ──
    if args.profile_layers:
        print("  Running per-layer profiling (torch.profiler)...")
        report["layer_profile"] = profile_layers(model, graphs, device, n_graphs=5)
    else:
        report["layer_profile"] = []

    # ── 9. Print Report ──
    print_report(report)

    # ── 10. Save JSON ──
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  ✅ JSON report saved to: {args.save_json}\n")


if __name__ == "__main__":
    main()
