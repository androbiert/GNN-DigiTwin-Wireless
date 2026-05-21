"""
graph_builder.py — Dynamic Tripartite Graph Builder for WirelessNet-Fermi

For each JSON timestamp snapshot, constructs the tripartite graph:
  - Flow nodes (F): one per active DL flow
  - Queue nodes (Q): one per UE with an active flow
  - Link nodes (L): one per (UE, serving_gNB) connection

Edges:
  F → Q : flow pushes demand into its UE's queue
  Q → L : queue connects to its serving gNB link
  L → Q : link injects radio state back into queue  (reverse for attention)
  Q → F : queue feeds back to flow for QoS readout
"""

import math
import re
from typing import Optional, Dict, Any
import numpy as np


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def _parse_qsize_bytes(qsize_str: str) -> float:
    """Convert queue-size strings like '10MiB', '100KiB', '2MiB' to bytes."""
    qsize_str = str(qsize_str).strip()
    m = re.match(r"([\d.]+)\s*(MiB|KiB|GiB|MB|KB|GB)?", qsize_str, re.IGNORECASE)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    multipliers = {"KIB": 1024, "MIB": 1024**2, "GIB": 1024**3,
                   "KB": 1000, "MB": 1000**2, "GB": 1000**3}
    return val * multipliers.get(unit, 1.0)


def _euclidean_distance(pos1: dict, pos2: dict) -> float:
    """3-D Euclidean distance between two node position dicts."""
    dx = pos1.get("x", 0.0) - pos2.get("x", 0.0)
    dy = pos1.get("y", 0.0) - pos2.get("y", 0.0)
    dz = pos1.get("z", 0.0) - pos2.get("z", 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _ue_index(ue_id: str) -> int:
    """'ue3' → 3, 'ue14' → 14"""
    return int(re.search(r"\d+", ue_id).group())


def _flow_ue_index(dst_str: str) -> int:
    """'ue[3]' → 3"""
    return int(re.search(r"\d+", dst_str).group())


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #

def build_graph(snapshot: dict) -> Optional[dict]:
    """
    Convert one JSON timestamp entry into a structured graph dict.

    Returns None if the snapshot has no active flows (delay == 0 for all flows).

    Returned dict keys
    ------------------
    flow_feat   : np.ndarray  [n_flows, 8]
                    0  packet_size          (bytes)
                    1  interval             (seconds between packets)
                    2  throughput           (bps, observed)
                    3  offered_load         (bps, = packet_size / interval — offered traffic)
                    4  packet_loss          (ratio [0,1])
                    5  harq_error_rate      (ratio [0,1])
                    6  harq_tx_attempts     (average HARQ rounds per TB)
                    7  delivery_ratio       (sentPacketToUpperLayer / receivedPacketFromLowerLayer)
    queue_feat  : np.ndarray  [n_queues, 2]
                    0  qsize_bytes          (bytes)
                    1  mac_buffer_overflow  (bool 0/1 — buffer saturation flag)
    link_feat   : np.ndarray  [n_links, 4]   (sinr_dl, sinr_ul, distance, speed)

    flow_to_queue  : np.ndarray [n_flows]         flow_i → queue index
    queue_to_link  : np.ndarray [n_queues]         queue_i → link index
    link_to_queue  : np.ndarray [n_links]          link_i → queue index  (same as above, reversed)

    target_delay       : np.ndarray [n_flows]
    target_throughput  : np.ndarray [n_flows]
    """
    nodes = snapshot.get("nodes", [])
    flows = snapshot.get("flows", [])

    if not flows:
        return None

    # ------------------------------------------------------------------ #
    # 1. Index UE and gNB nodes
    # ------------------------------------------------------------------ #
    ue_info  = {}   # ue_id  → node dict
    gnb_info = {}   # gnb_id → node dict

    for node in nodes:
        nid = node["id"]
        if nid.startswith("ue"):
            ue_info[nid] = node
        elif nid.startswith("gnb"):
            gnb_info[nid] = node

    # ------------------------------------------------------------------ #
    # 2. Identify active flows  (delay > 0 OR throughput > 0)
    # ------------------------------------------------------------------ #
    active_flows = [
        f for f in flows
        if (f.get("delay", 0) > 0 or f.get("throughput", 0) > 0)
        and f.get("dst", "").startswith("ue")
    ]

    if not active_flows:
        return None

    # ------------------------------------------------------------------ #
    # 3. Build QUEUE index  (one per UE that has at least one active flow)
    # ------------------------------------------------------------------ #
    active_ue_ids = sorted(
        set(_flow_ue_index(f["dst"]) for f in active_flows)
    )  # sorted list of UE integers
    queue_idx_map = {ue_int: i for i, ue_int in enumerate(active_ue_ids)}
    # ue_id_from_int: e.g. 3 → 'ue3'
    ue_id_for_int = {i: f"ue{i}" for i in active_ue_ids}

    # ------------------------------------------------------------------ #
    # 4. Build LINK index  (one per (UE, serving_gNB) pair)
    # ------------------------------------------------------------------ #
    link_keys = []   # (ue_int, gnb_id)  — ordered
    link_idx_map = {}

    for ue_int in active_ue_ids:
        ue_id = ue_id_for_int[ue_int]
        ue_node = ue_info.get(ue_id)
        if ue_node is None:
            continue
        gnb_id = ue_node.get("serving_gnb", "none")
        if gnb_id == "none" or gnb_id not in gnb_info:
            # Fallback: pick any available gNB
            gnb_id = next(iter(gnb_info), None)
        if gnb_id is None:
            continue
        key = (ue_int, gnb_id)
        if key not in link_idx_map:
            link_idx_map[key] = len(link_keys)
            link_keys.append(key)

    if not link_keys:
        return None

    # ------------------------------------------------------------------ #
    # 5. Build feature tensors
    # ------------------------------------------------------------------ #

    # --- Flow features ---
    flow_feat_list          = []
    flow_to_queue_list      = []
    target_delay_list       = []
    target_throughput_list  = []

    for f in active_flows:
        ue_int = _flow_ue_index(f["dst"])
        if ue_int not in queue_idx_map:
            continue

        pkt_size  = float(f.get("packet_size", 0))
        interval  = float(f.get("interval",    0))
        tput      = float(f.get("throughput",  0))

        # Derived: offered load (bps) — avoids division by zero
        offered_load = (pkt_size * 8.0 / interval) if interval > 0 else 0.0

        # Delivery ratio: packets delivered to app / packets received from MAC
        # (may be absent in SC06–SC08 — defaults to 0)
        rx_from_lower = float(f.get("receivedPacketFromLowerLayer", 0))
        tx_to_upper   = float(f.get("sentPacketToUpperLayer",       0))
        delivery_ratio = (tx_to_upper / rx_from_lower) if rx_from_lower > 0 else 0.0

        flow_feat_list.append([
            pkt_size,
            interval,
            tput,
            offered_load,
            float(f.get("packet_loss",    0)),   # radio link loss ratio  (0 if absent)
            float(f.get("harqErrorRate",  0)),   # PHY HARQ error rate    (0 if absent)
            float(f.get("harqTxAttempts", 0)),   # avg HARQ rounds        (0 if absent)
            delivery_ratio,                       # end-to-end delivery    (0 if absent)
        ])
        flow_to_queue_list.append(queue_idx_map[ue_int])
        target_delay_list.append(float(f.get("delay", 0)) + float(f.get("rlcDelay", 0)))
        target_throughput_list.append(float(f.get("throughput", 0)))

    if not flow_feat_list:
        return None

    flow_feat         = np.array(flow_feat_list,         dtype=np.float32)
    flow_to_queue     = np.array(flow_to_queue_list,     dtype=np.int64)
    target_delay      = np.array(target_delay_list,      dtype=np.float32)
    target_throughput = np.array(target_throughput_list, dtype=np.float32)

    # --- Queue features ---
    queue_feat_list       = []
    queue_to_link_list    = []

    for ue_int in active_ue_ids:
        ue_id  = ue_id_for_int[ue_int]
        ue_node = ue_info.get(ue_id, {})

        ue_flows = [f for f in active_flows if _flow_ue_index(f["dst"]) == ue_int]
        qsize_bytes = _parse_qsize_bytes(ue_node.get("qsize", "10MiB"))

        # macBufferOverflow: 1 if any flow for this UE overflowed the MAC buffer
        mac_overflow = float(
            any(f.get("macBufferOverflow", 0) > 0 for f in ue_flows)
        )

        queue_feat_list.append([qsize_bytes, mac_overflow])

        # Queue → Link
        gnb_id = ue_node.get("serving_gnb", "none")
        if gnb_id == "none" or gnb_id not in gnb_info:
            gnb_id = next(iter(gnb_info), None)
        key = (ue_int, gnb_id)
        queue_to_link_list.append(link_idx_map.get(key, 0))

    queue_feat     = np.array(queue_feat_list,    dtype=np.float32)
    queue_to_link  = np.array(queue_to_link_list, dtype=np.int64)

    # --- Link features ---
    link_feat_list      = []
    link_to_queue_list  = []

    for (ue_int, gnb_id), _ in sorted(link_idx_map.items(), key=lambda x: x[1]):
        ue_id   = ue_id_for_int[ue_int]
        ue_node = ue_info.get(ue_id, {})
        gnb_node = gnb_info.get(gnb_id, {})

        sinr_dl  = float(ue_node.get("sinr_dl", 0.0))
        sinr_ul  = float(ue_node.get("sinr_ul", 0.0))
        speed    = float(ue_node.get("speed", 0.0))
        distance = _euclidean_distance(ue_node, gnb_node) if gnb_node else 0.0

        link_feat_list.append([sinr_dl, sinr_ul, distance, speed])
        # Link maps back to the queue of its UE
        link_to_queue_list.append(queue_idx_map[ue_int])

    link_feat     = np.array(link_feat_list,     dtype=np.float32)
    link_to_queue = np.array(link_to_queue_list, dtype=np.int64)

    return {
        # Features
        "flow_feat"         : flow_feat,           # [n_flows, 8]
        "queue_feat"        : queue_feat,          # [n_queues, 3]
        "link_feat"         : link_feat,           # [n_links, 4]
        # Connectivity
        "flow_to_queue"     : flow_to_queue,       # [n_flows]     int
        "queue_to_link"     : queue_to_link,       # [n_queues]    int
        "link_to_queue"     : link_to_queue,       # [n_links]     int
        # Targets
        "target_delay"      : target_delay,        # [n_flows]
        "target_throughput" : target_throughput,   # [n_flows]
        # Meta
        "timestamp"         : snapshot.get("timestamp", 0.0),
        "n_flows"           : len(flow_feat),
        "n_queues"          : len(queue_feat),
        "n_links"           : len(link_feat),
    }
