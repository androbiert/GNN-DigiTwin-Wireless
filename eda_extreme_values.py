import json
import math
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wireless_gnn.graph_builder import build_graph
from wireless_gnn.scenario_registry import discover_scenarios, filter_for_target


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, "eda_results", "extreme_values")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_qsize_bytes(qsize: str) -> float:
    qsize = str(qsize).strip().upper()
    if qsize.endswith("KIB"):
        return float(qsize[:-3]) * 1024
    if qsize.endswith("MIB"):
        return float(qsize[:-3]) * 1024 * 1024
    if qsize.endswith("GIB"):
        return float(qsize[:-3]) * 1024 * 1024 * 1024
    return float(qsize[:-1]) if qsize.endswith("B") else float(qsize)


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f}"


def savefig(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.relpath(path, ROOT_DIR)}")


def load_delay_rows(configs):
    rows = []
    graph_counter = 0

    for idx, cfg in enumerate(configs, start=1):
        print(f"[load] {idx}/{len(configs)} {cfg.folder_name}")
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            snapshots = json.load(f)

        for snap in snapshots:
            graph = build_graph(snap)
            if graph is None:
                continue

            graph_counter += 1
            queue_feat = graph["queue_feat"]
            link_feat = graph["link_feat"]

            for flow_idx, flow_feat in enumerate(graph["flow_feat"]):
                queue_idx = int(graph["flow_to_queue"][flow_idx])
                link_idx = int(graph["queue_to_link"][queue_idx])

                rows.append({
                    "scenario": cfg.scenario_id,
                    "scheduler": cfg.scheduler,
                    "tx_power": cfg.tx_power,
                    "queue_size": cfg.queue_size,
                    "queue_size_bytes": parse_qsize_bytes(cfg.queue_size),
                    "config_folder": cfg.folder_name,
                    "timestamp": float(graph.get("timestamp", 0.0)),
                    "delay_s": float(graph["target_delay"][flow_idx]),
                    "throughput_bps": float(graph["target_throughput"][flow_idx]),
                    "packet_size": float(flow_feat[0]),
                    "interval": float(flow_feat[1]),
                    "flow_throughput_bps": float(flow_feat[2]),
                    "offered_load_bps": float(flow_feat[3]),
                    "packet_loss": float(flow_feat[4]),
                    "harq_error_rate": float(flow_feat[5]),
                    "harq_tx_attempts": float(flow_feat[6]),
                    "delivery_ratio": float(flow_feat[7]),
                    "queue_bytes": float(queue_feat[queue_idx][0]),
                    "mac_buffer_overflow": float(queue_feat[queue_idx][1]),
                    "sinr_dl": float(link_feat[link_idx][0]),
                    "sinr_ul": float(link_feat[link_idx][1]),
                    "distance": float(link_feat[link_idx][2]),
                    "speed": float(link_feat[link_idx][3]),
                })

    print(f"[load] collected {len(rows):,} flow rows from {graph_counter:,} valid graphs")
    return rows


def percentile_summary(values):
    ps = [50, 75, 90, 95, 99, 99.5, 99.9]
    return {p: float(np.percentile(values, p)) for p in ps}


def iqr_bounds(values):
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    return q1, q3, iqr, q3 + 1.5 * iqr, q3 + 3.0 * iqr


def robust_outlier_rate(values):
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad == 0.0:
        return 0.0, med, mad
    robust_z = 0.6745 * (values - med) / mad
    return float(np.mean(np.abs(robust_z) > 3.5)), med, mad


def summarize_group(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["delay_s"])

    summary = []
    for name, vals in grouped.items():
        arr = np.array(vals, dtype=np.float64)
        summary.append({
            key: name,
            "count": int(arr.size),
            "mean_ms": float(arr.mean() * 1000),
            "p95_ms": float(np.percentile(arr, 95) * 1000),
            "p99_ms": float(np.percentile(arr, 99) * 1000),
            "max_ms": float(arr.max() * 1000),
        })

    summary.sort(key=lambda x: (x["p99_ms"], x["max_ms"]), reverse=True)
    return summary


def top_extreme_configs(rows, threshold):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config_folder"]].append(row)

    summary = []
    for folder, items in grouped.items():
        delays = np.array([x["delay_s"] for x in items], dtype=np.float64)
        extreme_share = float(np.mean(delays >= threshold))
        summary.append({
            "config_folder": folder,
            "scenario": items[0]["scenario"],
            "scheduler": items[0]["scheduler"],
            "tx_power": items[0]["tx_power"],
            "queue_size": items[0]["queue_size"],
            "count": int(len(items)),
            "extreme_share_pct": extreme_share * 100,
            "p99_ms": float(np.percentile(delays, 99) * 1000),
            "max_ms": float(delays.max() * 1000),
        })

    summary.sort(key=lambda x: (x["extreme_share_pct"], x["p99_ms"], x["max_ms"]), reverse=True)
    return summary


def plot_delay_distributions(delays_s):
    delays_ms = delays_s * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(delays_ms, bins=120, color="#1f77b4", alpha=0.85)
    axes[0].set_title("Delay Distribution (Raw)")
    axes[0].set_xlabel("Delay (ms)")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(delays_ms), bins=120, color="#d62728", alpha=0.85)
    axes[1].set_title("Delay Distribution (log1p ms)")
    axes[1].set_xlabel("log1p(delay_ms)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Delay Tail Shape", fontweight="bold")
    savefig(fig, "delay_raw_vs_log.png")

    clip_p99 = float(np.percentile(delays_ms, 99))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delays_ms[delays_ms <= clip_p99], bins=100, color="#2ca02c", alpha=0.85)
    ax.set_title(f"Delay Distribution Clipped at p99 ({clip_p99:.2f} ms)")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Count")
    savefig(fig, "delay_clipped_p99.png")


def plot_top_groups(group_summary, group_key, filename, top_n=12):
    top = group_summary[:top_n]
    labels = [x[group_key] for x in top][::-1]
    p99_vals = [x["p99_ms"] for x in top][::-1]
    max_vals = [x["max_ms"] for x in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    ax.barh(y, p99_vals, color="#ff7f0e", alpha=0.85, label="p99")
    ax.barh(y, max_vals, color="#9467bd", alpha=0.35, label="max")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Delay (ms)")
    ax.set_title(f"Groups with Heaviest Delay Tails by {group_key}")
    ax.legend()
    savefig(fig, filename)


def plot_extreme_feature_boxplots(rows, extreme_threshold):
    delays = np.array([r["delay_s"] for r in rows], dtype=np.float64)
    mask = delays >= extreme_threshold
    normal = [rows[i] for i in range(len(rows)) if not mask[i]]
    extreme = [rows[i] for i in range(len(rows)) if mask[i]]

    features = [
        ("packet_loss", "Packet loss"),
        ("harq_error_rate", "HARQ error rate"),
        ("harq_tx_attempts", "HARQ tx attempts"),
        ("delivery_ratio", "Delivery ratio"),
        ("sinr_dl", "SINR DL"),
        ("queue_bytes", "Queue bytes"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, features):
        normal_vals = np.array([x[key] for x in normal], dtype=np.float64)
        extreme_vals = np.array([x[key] for x in extreme], dtype=np.float64)
        ax.boxplot([normal_vals, extreme_vals], tick_labels=["Non-extreme", "Extreme"], showfliers=False)
        ax.set_title(title)
        if key == "queue_bytes":
            ax.set_yscale("log")

    fig.suptitle("Feature Shift for Extreme Delay Samples", fontweight="bold")
    savefig(fig, "extreme_feature_boxplots.png")


def write_report(rows, summary):
    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Extreme Value EDA\n\n")
        f.write(f"- Flow rows analyzed: {summary['n_rows']:,}\n")
        f.write(f"- Delay configs analyzed: {summary['n_configs']}\n")
        f.write(f"- Delay scenarios discovered: {', '.join(summary['scenarios'])}\n")
        f.write(f"- Median delay: {summary['pcts'][50] * 1000:.3f} ms\n")
        f.write(f"- p95 delay: {summary['pcts'][95] * 1000:.3f} ms\n")
        f.write(f"- p99 delay: {summary['pcts'][99] * 1000:.3f} ms\n")
        f.write(f"- Max delay: {summary['delay_max_ms']:.3f} ms\n")
        f.write(f"- IQR upper fence: {summary['iqr_hi_ms']:.3f} ms\n")
        f.write(f"- Extreme threshold used: p99 = {summary['extreme_threshold_ms']:.3f} ms\n")
        f.write(f"- Share above p99 threshold: {summary['extreme_share_pct']:.3f}%\n")
        f.write(f"- Delay skewness proxy (mean/median): {summary['mean_over_median']:.2f}x\n")
        f.write("\n## Interpretation\n\n")
        f.write(summary["interpretation"] + "\n\n")

        f.write("## Heaviest Tail by Scenario\n\n")
        for row in summary["by_scenario"][:10]:
            f.write(
                f"- {row['scenario']}: count={row['count']:,}, "
                f"mean={row['mean_ms']:.3f} ms, p95={row['p95_ms']:.3f} ms, "
                f"p99={row['p99_ms']:.3f} ms, max={row['max_ms']:.3f} ms\n"
            )

        f.write("\n## Heaviest Tail by Scheduler\n\n")
        for row in summary["by_scheduler"][:10]:
            f.write(
                f"- {row['scheduler']}: count={row['count']:,}, "
                f"mean={row['mean_ms']:.3f} ms, p95={row['p95_ms']:.3f} ms, "
                f"p99={row['p99_ms']:.3f} ms, max={row['max_ms']:.3f} ms\n"
            )

        f.write("\n## Heaviest Tail by Queue Size\n\n")
        for row in summary["by_queue_size"][:10]:
            f.write(
                f"- {row['queue_size']}: count={row['count']:,}, "
                f"mean={row['mean_ms']:.3f} ms, p95={row['p95_ms']:.3f} ms, "
                f"p99={row['p99_ms']:.3f} ms, max={row['max_ms']:.3f} ms\n"
            )

        f.write("\n## Configurations Contributing Most Extreme Delays\n\n")
        for row in summary["top_configs"][:15]:
            f.write(
                f"- {row['config_folder']} ({row['scenario']}, {row['scheduler']}, "
                f"{row['tx_power']}, {row['queue_size']}): "
                f"extreme_share={row['extreme_share_pct']:.2f}%, "
                f"p99={row['p99_ms']:.3f} ms, max={row['max_ms']:.3f} ms\n"
            )

        f.write("\n## Robust Feature Outlier Rates\n\n")
        for feature, rate in summary["feature_outlier_rates"]:
            f.write(f"- {feature}: {rate * 100:.3f}%\n")

    print(f"[saved] {os.path.relpath(report_path, ROOT_DIR)}")


def main():
    ensure_dir(OUT_DIR)

    configs = discover_scenarios(ROOT_DIR, validate=True, verbose=True, use_cache=True)
    delay_configs = filter_for_target(configs, "delay")
    if not delay_configs:
        raise RuntimeError("No delay-enabled configs found.")

    rows = load_delay_rows(delay_configs)
    if not rows:
        raise RuntimeError("No flow rows were extracted from delay configs.")

    delays = np.array([r["delay_s"] for r in rows], dtype=np.float64)
    pcts = percentile_summary(delays)
    q1, q3, iqr, iqr_hi, extreme_iqr_hi = iqr_bounds(delays)
    extreme_threshold = pcts[99]
    extreme_mask = delays >= extreme_threshold

    by_scenario = summarize_group(rows, "scenario")
    by_scheduler = summarize_group(rows, "scheduler")
    by_queue_size = summarize_group(rows, "queue_size")
    top_configs = top_extreme_configs(rows, extreme_threshold)

    feature_keys = [
        "packet_loss",
        "harq_error_rate",
        "harq_tx_attempts",
        "delivery_ratio",
        "sinr_dl",
        "sinr_ul",
        "distance",
        "speed",
        "queue_bytes",
        "offered_load_bps",
    ]
    feature_outlier_rates = []
    for key in feature_keys:
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        rate, _, _ = robust_outlier_rate(arr)
        feature_outlier_rates.append((key, rate))
    feature_outlier_rates.sort(key=lambda x: x[1], reverse=True)

    overflow_rate_extreme = float(np.mean([r["mac_buffer_overflow"] for r in rows if r["delay_s"] >= extreme_threshold]))
    overflow_rate_all = float(np.mean([r["mac_buffer_overflow"] for r in rows]))

    interpretation_parts = []
    if pcts[99] > 2.0 * pcts[95]:
        interpretation_parts.append("The delay distribution is strongly right-skewed, with a very heavy upper tail.")
    else:
        interpretation_parts.append("The delay distribution has some tail risk, but the upper tail is not dramatically separated from the bulk.")
    if overflow_rate_extreme > overflow_rate_all * 1.5:
        interpretation_parts.append("Extreme delays align with MAC buffer overflow much more often than average, which suggests congestion rather than random noise.")
    if by_queue_size and by_queue_size[0]["queue_size"] in {"50KiB", "100KiB"}:
        interpretation_parts.append("Smaller queue sizes appear near the top of the tail summary, so some extremes may be structurally induced by tight buffers.")
    if not interpretation_parts:
        interpretation_parts.append("The largest delays do not appear to come from a single obvious failure mode, so they should be examined per configuration.")

    plot_delay_distributions(delays)
    plot_top_groups(by_scenario, "scenario", "tail_by_scenario.png", top_n=min(10, len(by_scenario)))
    plot_top_groups(by_scheduler, "scheduler", "tail_by_scheduler.png", top_n=min(10, len(by_scheduler)))
    plot_top_groups(by_queue_size, "queue_size", "tail_by_queue_size.png", top_n=min(10, len(by_queue_size)))
    plot_extreme_feature_boxplots(rows, extreme_threshold)

    summary = {
        "n_rows": len(rows),
        "n_configs": len(delay_configs),
        "scenarios": sorted(set(r["scenario"] for r in rows)),
        "pcts": pcts,
        "delay_max_ms": float(delays.max() * 1000),
        "iqr_hi_ms": float(iqr_hi * 1000),
        "extreme_threshold_ms": float(extreme_threshold * 1000),
        "extreme_share_pct": float(np.mean(extreme_mask) * 100),
        "mean_over_median": float(delays.mean() / max(pcts[50], 1e-12)),
        "interpretation": " ".join(interpretation_parts),
        "by_scenario": by_scenario,
        "by_scheduler": by_scheduler,
        "by_queue_size": by_queue_size,
        "top_configs": top_configs,
        "feature_outlier_rates": feature_outlier_rates,
    }
    write_report(rows, summary)

    print("\n[done] Extreme value EDA complete.")
    print(f"[done] Results saved to {os.path.relpath(OUT_DIR, ROOT_DIR)}")


if __name__ == "__main__":
    main()
