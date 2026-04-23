# OMNeT++ Features to Enrich Your Dataset

## Summary of Current State

Your `data.json` currently captures the following fields:

### Per UE (node):
`id`, `x`, `y`, `z`, `speed`, `serving_gnb`, `sinr_dl`, `sinr_ul`, `qsize`

### Per gNB (node):
`id`, `x`, `y`, `z`, `tx_power`, `scheduling_discipline`, `queue_size`

### Per Flow:
`type`, `src`, `dst`, `packet_size`, `interval`, `throughput`, `delay`, `packet_loss`,
`rlcDelay`, `macBufferOverflow`, `harqErrorRate`, `harqTxAttempts`,
`receivedPacketFromUpperLayer`, `receivedPacketFromLowerLayer`,
`sentPacketToLowerLayer`, `sentPacketToUpperLayer`

---

## Category 1 — Already in `data.json` but NOT Used by the GNN

These fields exist in your JSON right now. You only need to modify `graph_builder.py` to use them. **Zero OMNeT++ changes needed.**

### 🔵 Flow Node Enrichment (currently 3 dims → can become 8 dims)

| Field | Currently Used? | GNN Node | Why It Helps |
|---|---|---|---|
| `packet_size` | ✅ Yes | Flow | — |
| `interval` | ✅ Yes | Flow | — |
| `throughput` | ✅ Yes (as feature & target) | Flow | — |
| `packet_loss` | ❌ **No** | Flow | Measures radio link reliability — highly correlated with SINR and delay |
| `harqErrorRate` | ❌ **No** | Flow | PHY-layer retransmission indicator — directly causes latency increase |
| `harqTxAttempts` | ❌ **No** | Flow | Average HARQ rounds — tells model how many retx slots are consumed |
| `receivedPacketFromLowerLayer` | ❌ **No** | Flow | Cumulative RX count — proxy for effective link utilization |
| `sentPacketToUpperLayer` | ❌ **No** | Flow | Delivered packet count — measures successful delivery rate |

**Derived features you can compute from existing fields (no OMNeT++ change):**

| Derived Feature | Formula | Benefit |
|---|---|---|
| `offered_load` | `packet_size / interval` | Offered traffic rate in bps |
| `delivery_ratio` | `sentPacketToUpperLayer / receivedPacketFromLowerLayer` | Actual vs. received PDR |
| `harq_overhead` | `harqTxAttempts - 1` | Extra retransmission count |
| `rlc_to_e2e_ratio` | `rlcDelay / delay` | How much of E2E delay is in the RLC layer |

### 🟠 Queue Node Enrichment (currently 2 dims → can become 3 dims)

| Field | Currently Used? | GNN Node | Why It Helps |
|---|---|---|---|
| `rlcDelay` | ✅ Yes | Queue | — |
| `qsize` | ✅ Yes | Queue | — |
| `macBufferOverflow` | ❌ **No** | Queue | **Critical**: tells if the buffer is saturated — directly predicts congestion |

### 🟢 Link Node Enrichment (currently 4 dims → can become 5 dims)

| Field | Currently Used? | GNN Node | Why It Helps |
|---|---|---|---|
| `sinr_dl` | ✅ Yes | Link | — |
| `sinr_ul` | ✅ Yes | Link | — |
| `distance` | ✅ Yes (computed) | Link | — |
| `speed` | ✅ Yes | Link | — |
| `tx_power` (from gNB) | ❌ **No** | Link | Affects all UE SINRs — constant per scenario but varies across scenarios |

---

## Category 2 — Fields Collected but Removed by `add_attributes.py`

Your `add_attributes.py` script **explicitly removes** `app` and `bler`. These were likely in your original raw simulation output and can be restored.

| Removed Field | Where | GNN Node | Why It Was Removed | Should You Restore? |
|---|---|---|---|---|
| `bler` (Block Error Rate) | Flow | Flow/Link | Manually stripped | ✅ **Yes** — BLER is the PHY-layer error rate, directly related to SINR → delay → throughput |
| `app` | Flow | Flow | Manually stripped | ⚠️ Depends on content — if it's the app type/QoS class, restore it |

> [!TIP]
> To restore these, simply **remove** the `flow.pop("app", None)` and `flow.pop("bler", None)` lines in `add_attributes.py` and re-run it on your raw `network_state.json`.

---

## Category 3 — New Features to Add in OMNeT++ (DTConnector changes needed)

These require modifying your **C++ data collector** (`DTConnector.cc`) to subscribe to additional Simu5G signals and emit them to JSON. They provide the biggest improvement to model expressiveness.

### 🔵 New Flow-Level Features

| Feature | Simu5G Signal / Module | Description | Benefit |
|---|---|---|---|
| `cqi` | `NrMac::cqiReportedDl` | Channel Quality Indicator (0–15 scale) | Direct discrete proxy of link quality used by the scheduler |
| `mcs` | `NrMac::mcsComputedDl` | Modulation & Coding Scheme index | Tells which spectral efficiency the scheduler selected |
| `rb_allocated` | `NrMac::usedRbDl` | Number of Resource Blocks allocated to this UE | Scheduler fairness indicator — crucial for PF analysis |
| `scheduling_priority` | `NrMac::schedulingWeight` | PF metric value at scheduling time | The actual value that determines allocation decisions |
| `jitter` | Application layer | Variance of packet delay | Important QoS metric for real-time flows |
| `pkt_queue_delay` | `NrPdcp::pdcpDelayDl` | PDCP layer queuing delay | Separates protocol layer delays from radio delays |

### 🟠 New Queue-Level Features

| Feature | Simu5G Signal / Module | Description | Benefit |
|---|---|---|---|
| `mac_buffer_occupancy` | `NrMac::macBufferOccupancyDl` | Current MAC buffer fill level (bytes) | Real-time congestion indicator — much more informative than just overflow flag |
| `harq_round_distribution` | `NrMac::harqNack1/2/3` | % of 1st/2nd/3rd retransmissions | Distinguishes occasional vs. chronic HARQ failures |
| `scheduling_interval` | Config parameter | Time between scheduling decisions (TTI) | Determines queuing responsiveness |

### 🟢 New Link-Level Features (gNB side)

| Feature | Simu5G Signal / Module | Description | Benefit |
|---|---|---|---|
| `cell_load` | `NrMac::totalUsedRbDl / totalRbDl` | Fraction of RBs used across all UEs | Analogous to RouteNet's `link_load` — critical for congestion modeling |
| `num_active_ue` | Count from gNB | Number of UEs being served | System-level competition context |
| `avg_cell_sinr` | Mean of all UE SINRs | Average SINR across the cell | Cell-wide channel quality indicator |
| `inter_cell_interference` | PHY layer | Interference from neighboring gNBs | Especially important for multi-cell scenarios |
| `rsrp` | `NrPhy::rsrpDl` | Reference Signal Received Power (dBm) | More stable than SINR for path loss modeling |
| `rsrq` | `NrPhy::rsrqDl` | Reference Signal Received Quality | Includes interference component — complementary to RSRP |

### 🌐 New Scenario-Level / Global Features

| Feature | Source | Description | Benefit |
|---|---|---|---|
| `num_ue` | Config | Total UE count in the cell | Controls competition level |
| `gNB_bandwidth_MHz` | Config | Total bandwidth (e.g., 20 MHz) | Determines total available RBs |
| `num_prb` | Config | Number of Physical Resource Blocks | Direct resource pool size |
| `carrier_frequency_GHz` | Config | e.g., 3.5 GHz for sub-6GHz, 28 GHz for mmWave | Path loss law changes fundamentally |
| `scenario_type` | Metadata | Indoor/outdoor/highway | Propagation model identifier |

---

## Category 4 — Scenario Diversity (No Code Changes — Just Run More Simulations)

Your current dataset varies only the **queue size** across 3 scenarios. These additional axes will dramatically increase dataset diversity and model generalization:

| Axis | Current | Suggested Values | Impact |
|---|---|---|---|
| **Scheduling Policy** | PF only | PF, Round-Robin, Max-CQI | Teaches model how policy affects fairness vs. throughput |
| **Number of UEs** | 15 fixed | 5, 10, 15, 20, 30 | Models cell load scaling |
| **Mobility Model** | Mixed speeds | Pedestrian (1.5 m/s), Vehicular (30 m/s), High-speed (60 m/s) | Doppler effects and handover events |
| **TX Power** | 0.01 W fixed | 0.005, 0.01, 0.02, 0.04 W | Coverage and interference vary |
| **Bandwidth** | Fixed | 10 MHz, 20 MHz, 40 MHz, 100 MHz (FR2) | Different RB pool sizes |
| **Traffic Pattern** | CBR only | CBR, VoIP (small+periodic), Video (large+bursty) | Heterogeneous QoS classes |
| **Channel Model** | Fixed | ETSI-A, ETSI-B, 3GPP-UMa, 3GPP-UMi | Different fading profiles |

---

## Recommended Priority Order

> [!IMPORTANT]
> Do these in order — each step requires increasing effort but gives increasing benefit.

### 🟢 Step 1 — Zero Effort (just edit `graph_builder.py`)
Add to **Flow features**: `packet_loss`, `harqErrorRate`, `harqTxAttempts`, `macBufferOverflow`
Add **derived features**: `offered_load = packet_size / interval`, `delivery_ratio`

**Flow dims: 3 → 8** | **Queue dims: 2 → 3**

---

### 🟡 Step 2 — Low Effort (edit `add_attributes.py` + re-run)
Restore `bler` field from raw data → add it as a **Link feature** (BLER per UE).

**Link dims: 4 → 5**

---

### 🟠 Step 3 — Medium Effort (add signals to DTConnector.cc)
Add to the JSON collector: `cqi`, `rb_allocated`, `mac_buffer_occupancy`, `cell_load`, `num_active_ue`

**Flow dims: 8 → 10** | **Queue dims: 3 → 4** | **Link dims: 5 → 7**

---

### 🔴 Step 4 — High Effort (new simulations with varied parameters)
Run simulations with varying: scheduler type, UE count, TX power, bandwidth.
This creates **5-10× more training samples** with richer scenario diversity.

---

## Resulting Feature Dimensions After Full Enrichment

| Node Type | Current | After Step 1 | After Steps 1-3 | RouteNet-Fermi |
|---|---|---|---|---|
| **Flow** | 3 | 8 | 10 | 17 |
| **Queue** | 2 | 3 | 4 | 5 |
| **Link** | 4 | 5 | 7 | 5 |
| **Total** | 9 | 16 | 21 | 27 |
