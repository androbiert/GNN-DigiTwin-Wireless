# Feature Comparison: RouteNet-Fermi vs WirelessNet-Fermi (Our Data)

## Overview

Both models share the same **tripartite GNN architecture** (Path/Flow → Queue → Link message passing), but are designed for fundamentally different network environments:

| Property | RouteNet-Fermi (Original) | WirelessNet-Fermi (Ours) |
|---|---|---|
| **Domain** | Wired packet networks | 5G wireless (UE ↔ gNB) |
| **Simulator** | OMNeT++ (wired) | OMNeT++ (5G NR) |
| **Path concept** | Multi-hop IP paths (src→dst over routers) | Single-hop DL flows (gNB→UE) |
| **Link concept** | Wired link with fixed capacity | Wireless radio channel (SINR-based) |
| **Queue concept** | QoS queue at each router port | UE-side RLC buffer |
| **Framework** | TensorFlow | PyTorch |

---

## 1. 🔵 Flow / Path Node Features

These are the features attached to each **traffic flow** (called "path" in RouteNet, "flow" in ours).

### RouteNet-Fermi — Path Features (10 scalars + one-hot model ×7 = 17 dims total)

| Feature | Description | Unit / Type |
|---|---|---|
| `traffic` | Average bandwidth / bit-rate | bps (z-score normalized) |
| `packets` | Average packet generation rate | pkts/s (z-score normalized) |
| `model` | Traffic time-distribution type | one-hot ×7 (EXPONENTIAL, DETERMINISTIC, UNIFORM, NORMAL, ONOFF, PPBP, AR1) |
| `eq_lambda` | Equivalent arrival rate (λ) | pkts/s |
| `avg_pkts_lambda` | Average packets per λ interval | pkts |
| `exp_max_factor` | Max burst factor (for Exponential) | scalar |
| `pkts_lambda_on` | Pkt rate during ON period (OnOff) | pkts/s |
| `avg_t_off` | Average OFF duration (OnOff) | seconds |
| `avg_t_on` | Average ON duration (OnOff) | seconds |
| `ar_a` | AR(1) auto-regression coefficient | scalar [0,1] |
| `sigma` | AR(1) noise standard deviation | scalar |
| `length` | Number of hops on this path | integer (used for ragged indexing) |

> **Traffic model variety:** RouteNet-Fermi supports 7 different stochastic traffic models (Exponential, Deterministic, Uniform, Normal, OnOff, PPBP, AR1-based) — the `model` one-hot and all distribution params are needed to describe whichever model is active.

---

### WirelessNet-Fermi (Ours) — Flow Features (3 scalars = 3 dims)

| Feature | Description | Unit / Type |
|---|---|---|
| `packet_size` | Size of each data packet | bytes |
| `interval` | Inter-packet transmission interval | seconds |
| `throughput` | Observed throughput of this flow | bps (also used as target) |

> **Why so few?** In 5G downlink, all flows from a single gNB have essentially the same service model (Proportional Fair scheduler, fixed packet sizes). There is no notion of multi-hop routing or heterogeneous traffic distributions. The scheduling model is captured implicitly in the link/queue features.

---

## 2. 🟠 Queue Node Features

### RouteNet-Fermi — Queue Features (1 scalar + one-hot priority ×3 + 1 weight = 5 dims)

| Feature | Description | Unit / Type |
|---|---|---|
| `queue_size` | Buffer size of this queue | bytes (z-score normalized) |
| `priority` | QoS priority level of this queue | one-hot ×3 (up to 3 queues per link) |
| `weight` | Scheduling weight (for WFQ/DRR) | scalar [0,1] normalized |

> RouteNet models up to **3 queues per link**, each with a different priority and weight. This supports WFQ, DRR, SP, and FIFO policies simultaneously.

---

### WirelessNet-Fermi (Ours) — Queue Features (2 scalars = 2 dims)

| Feature | Description | Unit / Type |
|---|---|---|
| `rlcDelay` | RLC layer delay for this UE (max over flows) | seconds |
| `qsize_bytes` | UE-side buffer / queue size | bytes (parsed from "100KiB", "2MiB", etc.) |

> **Key difference:** We have **one queue per UE** (not per link port). There is no priority or weight because all UEs share the same Proportional Fair scheduler — queue differentiation in 5G is via radio resource management, not queue priority.

---

## 3. 🟢 Link Node Features

### RouteNet-Fermi — Link Features (1 load + one-hot policy ×4 = 5 dims)

| Feature | Description | Unit / Type |
|---|---|---|
| `capacity` | Link bandwidth | bps (used in load calculation) |
| `load` (derived) | Sum of traffic / capacity | scalar [0,∞), computed inline |
| `policy` | Scheduling policy at this output port | one-hot ×4 (WFQ, SP, DRR, FIFO) |

> The **link embedding input** is `[load, one_hot(policy)]` = 5-dim vector. Capacity itself is not directly embedded — it is used to compute `load` and during the readout (delay = occupancy / capacity).

---

### WirelessNet-Fermi (Ours) — Link Features (4 scalars = 4 dims)

| Feature | Description | Unit / Type |
|---|---|---|
| `sinr_dl` | Downlink SINR (UE measurement) | dB |
| `sinr_ul` | Uplink SINR (UE measurement) | dB |
| `distance` | 3D Euclidean distance UE ↔ serving gNB | meters |
| `speed` | UE mobility speed | m/s |

> **Why these?** These are the fundamental radio channel descriptors in 5G. SINR directly determines achievable data rate via Shannon's theorem. Distance and speed reflect path loss and Doppler effects. There is **no fixed capacity** in wireless — the effective rate changes every slot based on channel conditions.

---

## 4. 🔗 Connectivity / Edge Tensors

| Tensor | RouteNet-Fermi | WirelessNet-Fermi (Ours) |
|---|---|---|
| Path→Queue | `path_to_queue` (ragged, with position index) | `flow_to_queue` (flat int array) |
| Queue→Link | `queue_to_link` (ragged) | `queue_to_link` (flat int array) |
| Link→Path | `link_to_path` (ragged) | *(not needed — 1-hop)* |
| Queue→Path | `queue_to_path` (ragged) | *(not needed — 1-hop)* |
| Path→Link | `path_to_link` (ragged) | *(not needed — 1-hop)* |
| Link→Queue | *(not present)* | `link_to_queue` (flat int array) |

> RouteNet uses **5 ragged index tensors** to model arbitrary multi-hop path routing. Our model uses **3 simple flat arrays** because each flow takes exactly one hop (one link), making the connectivity trivial.

---

## 5. 🎯 Output Targets

| Target | RouteNet-Fermi | WirelessNet-Fermi (Ours) |
|---|---|---|
| **Primary** | `delay` — per-flow average end-to-end delay | `delay` — per-flow E2E delay |
| **Secondary** | *(none — single output)* | `throughput` — per-flow throughput |
| **Loss type** | MSE on log-transformed delay | MAPE on delay + MAPE on throughput |

---

## 6. Summary Table — All Input Dimensions

| Node Type | RouteNet-Fermi | WirelessNet-Fermi | Notes |
|---|---|---|---|
| **Flow/Path** | **17 dims** (10 scalars + one-hot×7) | **3 dims** | RF simplifies traffic model |
| **Queue** | **5 dims** (1 + one-hot×3 + 1) | **2 dims** | Single PF queue per UE |
| **Link** | **5 dims** (1 load + one-hot×4) | **4 dims** | Radio channel replaces wired capacity/policy |
| **Total** | **27 dims** | **9 dims** | |

---

## 7. What's Missing in Our Data vs RouteNet-Fermi

> [!WARNING]
> The following RouteNet-Fermi features have **no equivalent** in our current data:

| Missing Feature | Why It Matters | Possible Wireless Equivalent |
|---|---|---|
| Traffic distribution model (7 types) | Captures burstiness and stochasticity | Could add CQI variance, HARQ retx rate |
| Scheduling policy (WFQ/SP/DRR/FIFO) | Directly affects queuing delay | Could add scheduler type if scenarios vary |
| Queue priority & weight | Determines QoS differentiation | Could add QCI (QoS Class Identifier) |
| Multi-hop path length | Affects propagation delay | Fixed = 1 hop in our case ✓ |
| Link capacity (fixed) | Limits max throughput | Replaced by SINR (dynamic capacity) ✓ |

---

## 8. What Our Data Has That RouteNet-Fermi Doesn't

> [!NOTE]
> These are **wireless-specific** features unique to our dataset:

| Our Feature | Description |
|---|---|
| `sinr_dl` / `sinr_ul` | Radio channel quality (no wired equivalent) |
| `distance` (UE↔gNB) | Path loss proxy (no wired equivalent) |
| `speed` | Doppler / mobility (no wired equivalent) |
| `rlcDelay` | Protocol-layer queue delay (5G NR specific) |
| `throughput` (as target) | Dual-target prediction unique to wireless |
