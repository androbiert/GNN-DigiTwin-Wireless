"""
scenario_registry.py — Auto-discovery & grouping of simulation scenarios

Scans Data/SC*/simulations/*/data.json, parses configuration from folder names,
validates which scenarios have usable data, and groups them for training.

Folder name pattern:
    NN)SCxx-P=<power>W-S=<scheduler>-Q=<queue_size>
    e.g. 01)SC01-P=0.01W-S=PF-Q=50KiB

Usage:
    from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario
    configs = discover_scenarios("path/to/project")
    groups  = group_by_scenario(configs)   # { "SC01": [...], "SC02": [...], ... }
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# --------------------------------------------------------------------------- #
# Data class for a single simulation configuration
# --------------------------------------------------------------------------- #

@dataclass
class SimConfig:
    """One simulation run = one data.json file with its parsed metadata."""
    scenario_id:    str                # SC01, SC02, ...
    tx_power:       str                # "0.01W", "0.1W", "0.5W", "2W"
    scheduler:      str                # PF, MAXCI, DRR, MAXCI_MB, ALLOCATOR_BESTFIT
    queue_size:     str                # "50KiB", "100KiB", "2MiB", "10MiB"
    data_path:      str                # absolute path to data.json
    folder_name:    str                # original folder name
    n_snapshots:    int   = 0          # total snapshots in data.json
    n_with_flows:   int   = 0          # snapshots that have active flows
    has_delay:      bool  = False      # flows contain "delay" field
    has_throughput: bool  = False       # flows contain "throughput" field
    traffic_types:  list  = field(default_factory=list)  # unique traffic types found


# --------------------------------------------------------------------------- #
# Folder name parser
# --------------------------------------------------------------------------- #

_FOLDER_PATTERN = re.compile(
    r"^\d+\)"                          # "01)"
    r"(?P<sc>SC\d+)"                   # "SC01"
    r"-P=(?P<power>[\d.]+)W"           # "-P=0.01W"
    r"-S=(?P<sched>[A-Za-z_]+)"        # "-S=PF"
    r"-Q=(?P<qsize>\w+)$"             # "-Q=50KiB"
)


def parse_folder_name(name: str) -> Optional[dict]:
    """Parse a simulation folder name into config components.

    Returns dict with keys: sc, power, sched, qsize  — or None if no match.
    """
    m = _FOLDER_PATTERN.match(name)
    if not m:
        return None
    return {
        "sc":    m.group("sc").upper(),
        "power": m.group("power") + "W",
        "sched": m.group("sched"),
        "qsize": m.group("qsize"),
    }


# --------------------------------------------------------------------------- #
# Scenario validation (quick scan of data.json)
# --------------------------------------------------------------------------- #

def _validate_data(data_path: str, quick_limit: int = 10) -> dict:
    """Quick-scan a data.json to determine what it contains.

    Reads the first `quick_limit` snapshots with flows to check:
      - number of total snapshots
      - number with active flows
      - whether flows have delay / throughput fields
      - traffic types present in node data

    Returns dict with validation info.
    """
    info = {
        "n_snapshots": 0,
        "n_with_flows": 0,
        "has_delay": False,
        "has_throughput": False,
        "traffic_types": set(),
    }

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [registry] WARNING: cannot read {data_path}: {e}")
        return info

    info["n_snapshots"] = len(data)
    checked = 0

    for snap in data:
        flows = snap.get("flows", [])
        if flows:
            info["n_with_flows"] += 1

            if checked < quick_limit:
                for fl in flows:
                    if "delay" in fl:
                        info["has_delay"] = True
                    if "throughput" in fl:
                        info["has_throughput"] = True

                for node in snap.get("nodes", []):
                    tt = node.get("traffic_type")
                    if tt:
                        info["traffic_types"].add(tt)

                checked += 1

    info["traffic_types"] = sorted(info["traffic_types"])
    return info


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_scenarios(
    project_root: str,
    data_dir:     str = "Data",
    validate:     bool = True,
    verbose:      bool = True,
) -> List[SimConfig]:
    """Discover all simulation configs under <project_root>/<data_dir>/SC*/simulations/.

    Parameters
    ----------
    project_root : str
        Path to the project root.
    data_dir : str
        Subdirectory under project_root containing SC* folders (default "Data").
    validate : bool
        If True, quick-scans each data.json to check for flows/delay/throughput.
    verbose : bool
        Print discovery progress.

    Returns
    -------
    List[SimConfig]
        All valid simulation configurations found.
    """
    base = os.path.join(project_root, data_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Data directory not found: {base}")

    configs = []
    skipped_no_parse = 0
    skipped_no_data  = 0

    # Iterate SC* folders
    sc_dirs = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.upper().startswith("SC")
    ])

    if verbose:
        print(f"\n[registry] Scanning {base}")
        print(f"[registry] Found {len(sc_dirs)} scenario folders: {sc_dirs}")

    for sc_dir in sc_dirs:
        sim_base = os.path.join(base, sc_dir, "simulations")
        if not os.path.isdir(sim_base):
            if verbose:
                print(f"  [{sc_dir}] No 'simulations' subdirectory — skipping")
            continue

        sim_folders = sorted([
            d for d in os.listdir(sim_base)
            if os.path.isdir(os.path.join(sim_base, d))
        ])

        sc_count = 0
        for folder in sim_folders:
            parsed = parse_folder_name(folder)
            if parsed is None:
                skipped_no_parse += 1
                continue

            data_path = os.path.join(sim_base, folder, "data.json")
            if not os.path.isfile(data_path):
                skipped_no_data += 1
                continue

            cfg = SimConfig(
                scenario_id = parsed["sc"],
                tx_power    = parsed["power"],
                scheduler   = parsed["sched"],
                queue_size  = parsed["qsize"],
                data_path   = data_path,
                folder_name = folder,
            )

            if validate:
                info = _validate_data(data_path)
                cfg.n_snapshots    = info["n_snapshots"]
                cfg.n_with_flows   = info["n_with_flows"]
                cfg.has_delay      = info["has_delay"]
                cfg.has_throughput  = info["has_throughput"]
                cfg.traffic_types  = info["traffic_types"]

            configs.append(cfg)
            sc_count += 1

        if verbose:
            print(f"  [{sc_dir}] {sc_count} valid simulation configs found")

    if verbose:
        print(f"\n[registry] Total: {len(configs)} configs")
        if skipped_no_parse:
            print(f"[registry] Skipped {skipped_no_parse} folders (name parse failed)")
        if skipped_no_data:
            print(f"[registry] Skipped {skipped_no_data} folders (no data.json)")

    return configs


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #

def group_by_scenario(configs: List[SimConfig]) -> Dict[str, List[SimConfig]]:
    """Group configs by scenario_id. Returns {"SC01": [...], "SC02": [...], ...}."""
    groups: Dict[str, List[SimConfig]] = {}
    for cfg in configs:
        groups.setdefault(cfg.scenario_id, []).append(cfg)
    return dict(sorted(groups.items()))


def filter_for_target(
    configs: List[SimConfig],
    target:  str,
) -> List[SimConfig]:
    """Filter configs that support the given target ('delay' or 'throughput').

    - For delay:  must have has_delay=True AND n_with_flows > 0
    - For throughput: must have has_throughput=True AND n_with_flows > 0
    """
    filtered = []
    for cfg in configs:
        if cfg.n_with_flows == 0:
            continue
        if target == "delay" and not cfg.has_delay:
            continue
        if target == "throughput" and not cfg.has_throughput:
            continue
        filtered.append(cfg)
    return filtered


# --------------------------------------------------------------------------- #
# Summary printer
# --------------------------------------------------------------------------- #

def print_summary(configs: List[SimConfig]):
    """Print a human-readable summary table of discovered scenarios."""
    groups = group_by_scenario(configs)

    print(f"\n{'='*80}")
    print(f"  SCENARIO REGISTRY SUMMARY")
    print(f"{'='*80}")

    for sc_id, cfgs in groups.items():
        n_with_flows = sum(1 for c in cfgs if c.n_with_flows > 0)
        n_delay      = sum(1 for c in cfgs if c.has_delay)
        n_tput       = sum(1 for c in cfgs if c.has_throughput)
        schedulers   = sorted(set(c.scheduler for c in cfgs))
        powers       = sorted(set(c.tx_power for c in cfgs))
        q_sizes      = sorted(set(c.queue_size for c in cfgs))
        ttypes       = sorted(set(t for c in cfgs for t in c.traffic_types))
        total_snaps  = sum(c.n_with_flows for c in cfgs)

        print(f"\n  {sc_id}:")
        print(f"    Configs:      {len(cfgs)}")
        print(f"    With flows:   {n_with_flows}/{len(cfgs)}")
        print(f"    Has delay:    {n_delay}/{len(cfgs)}")
        print(f"    Has tput:     {n_tput}/{len(cfgs)}")
        print(f"    Total snaps:  {total_snaps}")
        print(f"    Schedulers:   {schedulers}")
        print(f"    Powers:       {powers}")
        print(f"    Queue sizes:  {q_sizes}")
        if ttypes:
            print(f"    Traffic:      {ttypes}")

    # Summary of what can be trained
    print(f"\n{'-'*80}")
    print(f"  TRAINABLE COMBINATIONS:")
    for sc_id, cfgs in groups.items():
        delay_ok = any(c.has_delay and c.n_with_flows > 0 for c in cfgs)
        tput_ok  = any(c.has_throughput and c.n_with_flows > 0 for c in cfgs)
        targets  = []
        if delay_ok:  targets.append("delay")
        if tput_ok:   targets.append("throughput")
        if targets:
            print(f"    {sc_id} -> {', '.join(targets)}")
        else:
            print(f"    {sc_id} -> [no trainable data]")
    print(f"{'='*80}\n")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover and summarize simulation scenarios")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--data-dir", default="Data", help="Data subdirectory name")
    args = parser.parse_args()

    configs = discover_scenarios(args.root, args.data_dir, validate=True, verbose=True)
    print_summary(configs)
