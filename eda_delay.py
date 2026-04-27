"""
eda_deep.py — Deep EDA for WirelessNet-Fermi Dataset
======================================================
Runs after the existing delay_distribution / boxplot / violin plots.
Covers:
  1. Graph topology stats          (node/edge count distributions)
  2. Feature dimension inventory   (shape of flow/queue/link feat per graph)
  3. Per-feature distributions     (flow_feat, queue_feat, link_feat, per dim)
  4. Feature–target correlations   (Pearson + Spearman heatmaps)
  5. Cross-scenario distribution shift (per-feature KDE overlays)
  6. Target joint distribution     (delay vs throughput scatter + hexbin)
  7. Outlier profile               (Z-score flags per feature dim)
  8. Pairwise target scatter by scenario

All plots are saved to  <project_root>/eda_results/deep/
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
from scipy import stats

from wireless_gnn.dataset import load_all_snapshots

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {os.path.relpath(path)}")


def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


def to2d(arr):
    """Ensure arr is 2-D (n_samples, n_dims). Handles scalars and 1-D arrays."""
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def robust_concat(list_of_arrays):
    """Stack a list of arrays that may be 1-D or 2-D into (N, D)."""
    return np.vstack([to2d(a) for a in list_of_arrays])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect raw arrays
# ─────────────────────────────────────────────────────────────────────────────

def collect(all_graphs):
    """Return per-scenario and global numpy arrays for all feature types."""
    by_sc = defaultdict(lambda: defaultdict(list))
    glob  = defaultdict(list)

    for g in all_graphs:
        sc = g["scenario"]
        for key in ("flow_feat", "queue_feat", "link_feat"):
            arr = to2d(g[key])
            by_sc[sc][key].append(arr)
            glob[key].append(arr)
        for key in ("target_delay", "target_throughput"):
            arr = np.atleast_1d(np.array(g[key], dtype=np.float32))
            by_sc[sc][key].append(arr)
            glob[key].append(arr)
        # topology
        by_sc[sc]["n_nodes"].append(g.get("num_nodes", g.get("n_nodes", np.nan)))
        by_sc[sc]["n_edges"].append(g.get("num_edges", g.get("n_edges", np.nan)))
        glob["n_nodes"].append(by_sc[sc]["n_nodes"][-1])
        glob["n_edges"].append(by_sc[sc]["n_edges"][-1])

    # Stack
    for sc in by_sc:
        for key in ("flow_feat", "queue_feat", "link_feat"):
            by_sc[sc][key] = robust_concat(by_sc[sc][key])
        for key in ("target_delay", "target_throughput"):
            by_sc[sc][key] = np.concatenate(by_sc[sc][key])
        for key in ("n_nodes", "n_edges"):
            by_sc[sc][key] = np.array([x for x in by_sc[sc][key]
                                        if not (isinstance(x, float) and np.isnan(x))],
                                       dtype=np.float32)

    for key in ("flow_feat", "queue_feat", "link_feat"):
        glob[key] = robust_concat(glob[key])
    for key in ("target_delay", "target_throughput"):
        glob[key] = np.concatenate(glob[key])
    for key in ("n_nodes", "n_edges"):
        glob[key] = np.array([x for x in glob[key]
                               if not (isinstance(x, float) and np.isnan(x))],
                              dtype=np.float32)

    return by_sc, glob


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph Topology
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology(by_sc, glob, out):
    print("\n[1] Graph topology stats ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, label in zip(axes,
                               ("n_nodes", "n_edges"),
                               ("Node count per graph", "Edge count per graph")):
        for sc, d in by_sc.items():
            if len(d[key]):
                ax.hist(d[key], bins=30, alpha=0.55, label=sc[-12:], density=True)
        ax.set_title(label)
        ax.set_xlabel(label.split()[0])
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Graph Topology Distributions by Scenario", fontweight="bold")
    save(fig, os.path.join(out, "topology_distributions.png"))

    # Print summary
    for key, label in (("n_nodes", "Nodes"), ("n_edges", "Edges")):
        arr = glob[key]
        if len(arr):
            print(f"  {label}: mean={arr.mean():.1f}  std={arr.std():.1f}  "
                  f"min={arr.min():.0f}  max={arr.max():.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-feature dimension distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_dims(glob, out):
    print("\n[2] Per-feature dimension distributions ...")
    for feat_key, label in (("flow_feat",  "Flow Features"),
                             ("queue_feat", "Queue Features"),
                             ("link_feat",  "Link Features")):
        arr = glob[feat_key]          # shape (N, D)
        D   = arr.shape[1]
        ncols = min(D, 6)
        nrows = int(np.ceil(D / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3 * ncols, 2.5 * nrows),
                                 squeeze=False)
        for d in range(D):
            r, c = divmod(d, ncols)
            ax = axes[r][c]
            col = arr[:, d]
            ax.hist(col, bins=40, color="steelblue", alpha=0.75)
            ax.set_title(f"dim {d}", fontsize=9)
            ax.set_xlabel("")
            # Flag if near-constant
            if col.std() < 1e-6:
                ax.set_facecolor("#fff0f0")
                ax.set_title(f"dim {d} ⚠ const", fontsize=9, color="red")
        # Hide unused axes
        for d in range(D, nrows * ncols):
            r, c = divmod(d, ncols)
            axes[r][c].set_visible(False)
        fig.suptitle(f"{label}  ({D} dims, {arr.shape[0]:,} samples)",
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        fname = feat_key + "_dim_distributions.png"
        save(fig, os.path.join(out, fname))
        print(f"  {label}: shape {arr.shape} | "
              f"near-zero-std dims: "
              f"{[d for d in range(D) if arr[:,d].std() < 1e-6]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature–target correlation heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_target_correlations(glob, out):
    print("\n[3] Feature–target correlation heatmaps ...")
    delay     = glob["target_delay"]
    tput      = glob["target_throughput"]

    # We align per-flow targets with flow features (same length)
    for feat_key, label in (("flow_feat",  "Flow"),
                             ("queue_feat", "Queue"),
                             ("link_feat",  "Link")):
        arr = glob[feat_key]   # (N_feat, D)
        D   = arr.shape[1]

        # Use per-flow delay; if sizes don't match, skip gracefully
        n  = min(len(arr), len(delay))
        X  = arr[:n]
        yd = delay[:n]
        yt = tput[:n]

        pearson_d  = np.array([stats.pearsonr(X[:, d], yd)[0]  for d in range(D)])
        spearman_d = np.array([stats.spearmanr(X[:, d], yd)[0] for d in range(D)])
        pearson_t  = np.array([stats.pearsonr(X[:, d], yt)[0]  for d in range(D)])
        spearman_t = np.array([stats.spearmanr(X[:, d], yt)[0] for d in range(D)])

        mat = np.vstack([pearson_d, spearman_d, pearson_t, spearman_t])

        fig, ax = plt.subplots(figsize=(max(8, D * 0.6 + 2), 4))
        sns.heatmap(mat,
                    xticklabels=[f"d{i}" for i in range(D)],
                    yticklabels=["Pearson/delay", "Spearman/delay",
                                 "Pearson/tput",  "Spearman/tput"],
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=(D <= 20), fmt=".2f", linewidths=0.3, ax=ax)
        ax.set_title(f"{label} Feature × Target Correlations", fontweight="bold")
        plt.tight_layout()
        save(fig, os.path.join(out, f"{feat_key}_target_corr.png"))

        # Warn about high correlations (potential leakage) or zero correlations
        top = np.argsort(np.abs(pearson_d))[::-1][:3]
        print(f"  {label} – top-3 |Pearson/delay| dims: "
              f"{[(d, round(pearson_d[d],3)) for d in top]}")
        zero = [d for d in range(D) if abs(pearson_d[d]) < 0.01 and
                abs(pearson_t[d]) < 0.01]
        if zero:
            print(f"  {label} – dims with ~0 corr to both targets: {zero} (check these!)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-scenario distribution shift (KDE overlays)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_shift(by_sc, out):
    print("\n[4] Cross-scenario distribution shift ...")
    scenarios = list(by_sc.keys())
    colors    = sns.color_palette("Set1", len(scenarios))

    for feat_key, label in (("flow_feat",  "Flow"),
                             ("queue_feat", "Queue"),
                             ("link_feat",  "Link")):
        arrs = {sc: by_sc[sc][feat_key] for sc in scenarios}
        D    = arrs[scenarios[0]].shape[1]
        ncols = min(D, 6)
        nrows = int(np.ceil(D / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3 * ncols, 2.5 * nrows),
                                 squeeze=False)
        for d in range(D):
            r, c = divmod(d, ncols)
            ax   = axes[r][c]
            for sc, col_arr, col in zip(scenarios, arrs.values(), colors):
                col_data = col_arr[:, d]
                if col_data.std() > 1e-9:
                    sns.kdeplot(col_data, ax=ax, label=sc[-8:],
                                color=col, fill=True, alpha=0.25, linewidth=1.2)
            ax.set_title(f"dim {d}", fontsize=8)
            ax.set_yticks([])

        # Legend on first axis
        axes[0][0].legend(fontsize=6)
        # Hide unused
        for d in range(D, nrows * ncols):
            r, c = divmod(d, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle(f"{label} Feature – KDE by Scenario  ({D} dims)",
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        save(fig, os.path.join(out, f"{feat_key}_scenario_kde.png"))

        # KS test: does the distribution differ significantly?
        if len(scenarios) >= 2:
            sc0, sc1 = scenarios[0], scenarios[1]
            diffs = []
            for d in range(D):
                ks = stats.ks_2samp(arrs[sc0][:, d], arrs[sc1][:, d])
                if ks.pvalue < 0.01:
                    diffs.append(d)
            print(f"  {label}: dims with significant shift ({sc0[-8:]} vs {sc1[-8:]}, "
                  f"KS p<0.01): {diffs}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Joint target distribution (delay vs throughput)
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_joint(by_sc, glob, out):
    print("\n[5] Target joint distribution ...")
    delay = glob["target_delay"] * 1000       # → ms
    tput  = glob["target_throughput"] / 1e6   # → Mbps (adjust if already Mbps)

    # Overall hexbin
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(delay, tput, gridsize=50, cmap="YlOrRd", mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Joint Distribution: Delay × Throughput", fontweight="bold")
    save(fig, os.path.join(out, "target_joint_hexbin.png"))

    # Per-scenario scatter
    scenarios = list(by_sc.keys())
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, sc in enumerate(scenarios):
        ax  = axes[0][i]
        d_  = by_sc[sc]["target_delay"] * 1000
        t_  = by_sc[sc]["target_throughput"] / 1e6
        ax.scatter(d_, t_, s=4, alpha=0.3, color=sns.color_palette("Set1", n)[i])
        r, p = stats.pearsonr(d_[:5000], t_[:5000])   # subsample for speed
        ax.set_title(f"{sc[-14:]}\nr={r:.3f}", fontsize=9)
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Throughput (Mbps)")
    fig.suptitle("Delay vs Throughput per Scenario", fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(out, "target_joint_by_scenario.png"))

    # Global Pearson
    r, p = stats.pearsonr(delay, tput)
    print(f"  Global Pearson(delay, tput) = {r:.4f}  (p={p:.2e})")
    if abs(r) > 0.8:
        print("  ⚠  Very high delay–throughput correlation – "
              "check if targets are co-derived (possible data leakage).")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Outlier detection (Z-score)
# ─────────────────────────────────────────────────────────────────────────────

def plot_outliers(glob, out):
    print("\n[6] Outlier detection (|Z| > 4) ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    summary   = {}

    for ax, feat_key, label in zip(axes,
                                   ("flow_feat",  "queue_feat",  "link_feat"),
                                   ("Flow",        "Queue",        "Link")):
        arr    = glob[feat_key]
        z      = np.abs(stats.zscore(arr, axis=0))
        pct    = (z > 4).mean(axis=0) * 100   # % outlier per dim
        summary[label] = pct
        ax.bar(range(len(pct)), pct, color="tomato", alpha=0.8)
        ax.set_title(f"{label} – % samples with |Z| > 4 per dim")
        ax.set_xlabel("Feature dim")
        ax.set_ylabel("% outliers")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    save(fig, os.path.join(out, "outlier_zscore.png"))

    for label, pct in summary.items():
        bad = [(d, round(p, 2)) for d, p in enumerate(pct) if p > 1.0]
        if bad:
            print(f"  {label}: dims with >1 % outliers: {bad}")
        else:
            print(f"  {label}: no dims with >1 % outliers ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Target distribution by scenario – violin + stats table
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_by_scenario(by_sc, out):
    print("\n[7] Target distributions per scenario ...")
    scenarios = list(by_sc.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, unit, scale in zip(axes,
                                    ("target_delay",      "target_throughput"),
                                    ("ms",                "Mbps"),
                                    (1000,                1e-6)):
        data   = [by_sc[sc][key] * scale for sc in scenarios]
        labels = [sc[-14:] for sc in scenarios]
        parts  = ax.violinplot(data, showmedians=True, showextrema=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel(f"{key.replace('target_','').title()} ({unit})")
        ax.set_title(f"{key.replace('target_','').title()} by Scenario")

    fig.suptitle("Target Value Distributions per Scenario", fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(out, "target_by_scenario_violin.png"))

    # Print per-scenario stats
    print(f"\n  {'Scenario':<35} {'Delay mean(ms)':>15} {'Delay std':>10} "
          f"{'Tput mean(Mbps)':>16} {'Tput std':>10}")
    for sc in scenarios:
        d  = by_sc[sc]["target_delay"]     * 1000
        t  = by_sc[sc]["target_throughput"] / 1e6
        print(f"  {sc[-35:]:<35} {d.mean():>15.3f} {d.std():>10.3f} "
              f"{t.mean():>16.3f} {t.std():>10.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Feature variance ratio across scenarios  (ANOVA F-stat per dim)
# ─────────────────────────────────────────────────────────────────────────────

def plot_anova(by_sc, out):
    print("\n[8] Feature ANOVA across scenarios ...")
    scenarios = list(by_sc.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, feat_key, label in zip(axes,
                                   ("flow_feat", "queue_feat", "link_feat"),
                                   ("Flow",       "Queue",      "Link")):
        D      = by_sc[scenarios[0]][feat_key].shape[1]
        f_vals = []
        for d in range(D):
            groups = [by_sc[sc][feat_key][:, d] for sc in scenarios]
            try:
                f, _ = stats.f_oneway(*groups)
            except Exception:
                f = 0.0
            f_vals.append(f if np.isfinite(f) else 0.0)
        f_arr = np.array(f_vals)
        ax.bar(range(D), np.log1p(f_arr), color="mediumorchid", alpha=0.8)
        ax.set_title(f"{label} – log(1+F) ANOVA across scenarios")
        ax.set_xlabel("Feature dim")
        ax.set_ylabel("log(1 + F-statistic)")
        # Annotate top-3
        top3 = np.argsort(f_arr)[::-1][:3]
        for d in top3:
            ax.annotate(f"d{d}", (d, np.log1p(f_arr[d])),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color="darkred")

        print(f"  {label}: top-3 scenario-discriminative dims "
              f"{[(d, round(float(f_arr[d]),1)) for d in top3]}")

    plt.tight_layout()
    save(fig, os.path.join(out, "feature_anova.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    out      = ensure(os.path.join(root_dir, "eda_results", "deep"))

    print("=" * 55)
    print("  WirelessNet-Fermi  —  Deep EDA")
    print("=" * 55)
    print(f"Output dir: {out}\n")

    print("Loading all snapshots ...")
    all_graphs = load_all_snapshots(root_dir)

    print("\nCollecting arrays by scenario ...")
    by_sc, glob = collect(all_graphs)

    # Print quick inventory
    print("\n--- Feature inventory ---")
    for k in ("flow_feat", "queue_feat", "link_feat"):
        print(f"  {k}: shape {glob[k].shape}")
    print(f"  target_delay:       {glob['target_delay'].shape}")
    print(f"  target_throughput:  {glob['target_throughput'].shape}")

    plot_topology(by_sc, glob, out)
    plot_feature_dims(glob, out)
    plot_feature_target_correlations(glob, out)
    plot_distribution_shift(by_sc, out)
    plot_target_joint(by_sc, glob, out)
    plot_outliers(glob, out)
    plot_target_by_scenario(by_sc, out)
    plot_anova(by_sc, out)

    print(f"\n{'='*55}")
    print(f"  All EDA plots saved to:  eda_results/deep/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()