# WirelessNet-Fermi v2 — Mathematical Formulation

> **For presentation use.** Every equation maps directly to the implementation in [model.py](file:///c:/Users/DELL/Desktop/GNN-DigiTwin-Wireless/wireless_gnn/model.py).

---

## 0 · Notation & Graph Definition

| Symbol | Meaning |
|--------|---------|
| $\mathcal{G} = (\mathcal{F}, \mathcal{Q}, \mathcal{L}, \mathcal{E})$ | Tripartite heterogeneous graph |
| $\mathcal{F} = \{f_1, \dots, f_F\}$ | Set of **Flow** nodes, $\lvert\mathcal{F}\rvert = F$ |
| $\mathcal{Q} = \{q_1, \dots, q_Q\}$ | Set of **Queue** nodes (one per UE), $\lvert\mathcal{Q}\rvert = Q$ |
| $\mathcal{L} = \{l_1, \dots, l_L\}$ | Set of **Link** nodes (one per UE–gNB pair), $\lvert\mathcal{L}\rvert = L$ |
| $\mathcal{N}_q^{\mathcal{F}}$ | Flows assigned to queue $q$: $\{f \in \mathcal{F} \mid \text{f2q}(f) = q\}$ |
| $D$ | Hidden dimension (default 64) |
| $H$ | Number of attention heads (default 4) |
| $d_h = D / H$ | Per-head dimension |
| $K$ | Number of message-passing iterations (default 8) |
| $T$ | Temporal attention window (default 8 snapshots) |

**Edge types (4 directed relations):**

$$
\mathcal{E} = \underbrace{(f \!\to\! q)}_{\text{f2q}} \;\cup\; \underbrace{(q \!\to\! l)}_{\text{q2l}} \;\cup\; \underbrace{(l \!\to\! q)}_{\text{l2q}} \;\cup\; \underbrace{(q \!\to\! f)}_{\text{implicit via f2q}}
$$

---

## Phase 1 · Embedding Layer

Each node type has a **2-layer MLP** that projects raw features into $\mathbb{R}^D$:

$$
\mathbf{h}_f^{(0)} = \text{ReLU}\!\Big(W_f^{(2)}\;\text{ReLU}\!\big(W_f^{(1)}\,\mathbf{x}_f + b_f^{(1)}\big) + b_f^{(2)}\Big), \quad \mathbf{x}_f \in \mathbb{R}^{8}
\tag{1.1}
$$

$$
\mathbf{h}_q^{(0)} = \text{ReLU}\!\Big(W_q^{(2)}\;\text{ReLU}\!\big(W_q^{(1)}\,\mathbf{x}_q + b_q^{(1)}\big) + b_q^{(2)}\Big), \quad \mathbf{x}_q \in \mathbb{R}^{2}
\tag{1.2}
$$

$$
\mathbf{h}_l^{(0)} = \text{ReLU}\!\Big(W_l^{(2)}\;\text{ReLU}\!\big(W_l^{(1)}\,\mathbf{x}_l + b_l^{(1)}\big) + b_l^{(2)}\Big), \quad \mathbf{x}_l \in \mathbb{R}^{4}
\tag{1.3}
$$

After embedding, all three node types live in $\mathbb{R}^D$, enabling cross-type attention.

---

## Phase 2 · Message Passing ($K$ Iterations)

For each iteration $k = 1, \dots, K$, the following four steps are applied sequentially.  
We write $\mathbf{h}^{(k)}$ for the state **after** iteration $k$.

---

### Step 1 — Flow → Queue Attention (F → Q)

> *Each queue aggregates its flows, weighted by learned criticality.*

**Multi-head scaled dot-product attention** where $\text{Query} = \text{queue}$, $\text{Key/Value} = \text{flow}$:

For each flow $j$ assigned to queue $i$ (i.e. $j \in \mathcal{N}_i^{\mathcal{F}}$), and for each head $m = 1,\dots,H$:

$$
\mathbf{q}_{i}^{(m)} = W_Q^{(m)}\,\mathbf{h}_{q_i}, \quad
\mathbf{k}_{j}^{(m)} = W_K^{(m)}\,\mathbf{h}_{f_j}, \quad
\mathbf{v}_{j}^{(m)} = W_V^{(m)}\,\mathbf{h}_{f_j}
\tag{2.1}
$$

$$
e_{ij}^{(m)} = \frac{\big(\mathbf{q}_{i}^{(m)}\big)^\top \mathbf{k}_{j}^{(m)}}{\sqrt{d_h}}
\tag{2.2}
$$

$$
\alpha_{ij}^{(m)} = \frac{\exp\!\big(e_{ij}^{(m)} - \max_{j' \in \mathcal{N}_i^{\mathcal{F}}} e_{ij'}^{(m)}\big)}{\sum_{j' \in \mathcal{N}_i^{\mathcal{F}}} \exp\!\big(e_{ij'}^{(m)} - \max_{j''} e_{ij''}^{(m)}\big)}
\quad \text{(grouped softmax)}
\tag{2.3}
$$

$$
\text{agg}_i^{(m)} = \sum_{j \in \mathcal{N}_i^{\mathcal{F}}} \alpha_{ij}^{(m)}\;\mathbf{v}_{j}^{(m)}
\tag{2.4}
$$

$$
\mathbf{m}_{q_i}^{\text{F→Q}} = W_{\text{proj}}\;\Big[\text{agg}_i^{(1)} \;\|\; \cdots \;\|\; \text{agg}_i^{(H)}\Big]
\tag{2.5}
$$

**GRU state update** (recurrent gate across iterations):

$$
\mathbf{z}_i = \sigma\!\Big(W_z\,\mathbf{m}_{q_i}^{\text{F→Q}} + U_z\,\mathbf{h}_{q_i} + b_z\Big)
\tag{2.6a}
$$

$$
\mathbf{r}_i = \sigma\!\Big(W_r\,\mathbf{m}_{q_i}^{\text{F→Q}} + U_r\,\mathbf{h}_{q_i} + b_r\Big)
\tag{2.6b}
$$

$$
\tilde{\mathbf{h}}_{q_i} = \tanh\!\Big(W_h\,\mathbf{m}_{q_i}^{\text{F→Q}} + U_h\,(\mathbf{r}_i \odot \mathbf{h}_{q_i}) + b_h\Big)
\tag{2.6c}
$$

$$
\mathbf{h}_{q_i} \;\leftarrow\; (1 - \mathbf{z}_i) \odot \mathbf{h}_{q_i} \;+\; \mathbf{z}_i \odot \tilde{\mathbf{h}}_{q_i}
\tag{2.6d}
$$

---

### Step 2a — Link Self-Attention (L → L)

> *Models inter-cell interference — each link attends to all other links.*

Standard all-pairs multi-head self-attention over $\mathcal{L}$:

$$
\mathbf{q}_i^{(m)} = W_Q^{(m)}\,\mathbf{h}_{l_i}, \quad
\mathbf{k}_j^{(m)} = W_K^{(m)}\,\mathbf{h}_{l_j}, \quad
\mathbf{v}_j^{(m)} = W_V^{(m)}\,\mathbf{h}_{l_j}
\tag{3.1}
$$

$$
\beta_{ij}^{(m)} = \text{softmax}_j\!\left(\frac{\big(\mathbf{q}_{i}^{(m)}\big)^\top \mathbf{k}_{j}^{(m)}}{\sqrt{d_h}}\right), \quad \forall\; i,j \in \{1,\dots,L\}
\tag{3.2}
$$

$$
\text{MHA}_i = W_{\text{proj}}\;\Big[\textstyle\sum_j \beta_{ij}^{(1)} \mathbf{v}_j^{(1)} \;\|\; \cdots \;\|\; \textstyle\sum_j \beta_{ij}^{(H)} \mathbf{v}_j^{(H)}\Big]
\tag{3.3}
$$

**Residual + Layer Normalization:**

$$
\mathbf{h}_{l_i} \;\leftarrow\; \text{LayerNorm}\!\Big(\mathbf{h}_{l_i} + \text{MHA}_i\Big)
\tag{3.4}
$$

> Complexity: $\mathcal{O}(L^2)$, acceptable since $L < 30$ per snapshot.

---

### Step 2b — Link → Queue Attention (L → Q)

> *Each queue absorbs radio-level information (SINR, distance, speed) from its serving link, weighted by attention.*

Architecture identical to F→Q (Step 1), but with $\text{Query} = \text{queue}$, $\text{Key/Value} = \text{link}$:

For each link $j$ mapped to queue $i$ (via $\text{l2q}(j) = i$), for each head $m$:

$$
\mathbf{q}_{i}^{(m)} = W_Q^{(m)}\,\mathbf{h}_{q_i}, \quad
\mathbf{k}_{j}^{(m)} = W_K^{(m)}\,\mathbf{h}_{l_j}, \quad
\mathbf{v}_{j}^{(m)} = W_V^{(m)}\,\mathbf{h}_{l_j}
\tag{4.1}
$$

$$
e_{ij}^{(m)} = \frac{\big(\mathbf{q}_{i}^{(m)}\big)^\top \mathbf{k}_{j}^{(m)}}{\sqrt{d_h}}
\tag{4.2}
$$

$$
\alpha_{ij}^{(m)} = \text{grouped\_softmax}\!\big(e_{ij}^{(m)},\;\text{l2q},\;Q\big)
\tag{4.3}
$$

$$
\mathbf{m}_{q_i}^{\text{L→Q}} = W_{\text{proj}}\;\Big[\textstyle\sum_{j:\,\text{l2q}(j)=i} \alpha_{ij}^{(m)}\,\mathbf{v}_j^{(m)}\;\Big]_{m=1}^{H}
\tag{4.4}
$$

**GRU state update** (same GRU formulation as Eq. 2.6, with separate parameters $W_z^{lq}, U_z^{lq}, \dots$):

$$
\mathbf{h}_{q_i} \;\leftarrow\; \text{GRU}_{\text{LQ}}\!\big(\mathbf{m}_{q_i}^{\text{L→Q}},\;\mathbf{h}_{q_i}\big)
\tag{4.5}
$$

---

### Step 2c — Queue → Link Residual (Q → L)

> *Feeds queue load (buffer fullness, congestion) back to link nodes.*

Simple linear projection with $\tanh$ gating and additive residual:

$$
\mathbf{m}_{l_j}^{\text{Q→L}} = W_{\text{ql}}\;\mathbf{h}_{q_{\text{l2q}(j)}} + b_{\text{ql}}
\tag{5.1}
$$

$$
\mathbf{h}_{l_j} \;\leftarrow\; \mathbf{h}_{l_j} + \tanh\!\big(\mathbf{m}_{l_j}^{\text{Q→L}}\big)
\tag{5.2}
$$

> The $\tanh$ bounds the message to $[-1, 1]$, preventing it from dominating the link state.

---

### Step 3 — Queue → Flow Gating (Q → F)

> *Personalised gate per flow — flows in poor conditions amplify the queue signal.*

For each flow $i$ with parent queue $q = \text{f2q}(i)$:

**Gate computation:**

$$
\mathbf{g}_i = \sigma\!\Big(W_{\text{gate}}\;\big[\mathbf{h}_{f_i} \;\|\; \mathbf{h}_{q}\big] + b_{\text{gate}}\Big) \in [0,1]^D
\tag{6.1}
$$

**Gated message:**

$$
\mathbf{m}_{f_i}^{\text{Q→F}} = \mathbf{g}_i \odot \Big(W_{\text{val}}\;\mathbf{h}_{q} + b_{\text{val}}\Big)
\tag{6.2}
$$

where $\|$ is concatenation and $\odot$ is element-wise (Hadamard) product.

**GRU state update:**

$$
\mathbf{h}_{f_i} \;\leftarrow\; \text{GRU}_{\text{QF}}\!\big(\mathbf{m}_{f_i}^{\text{Q→F}},\;\mathbf{h}_{f_i}\big)
\tag{6.3}
$$

> [!IMPORTANT]
> All three GRUs ($\text{GRU}_{\text{FQ}}$, $\text{GRU}_{\text{LQ}}$, $\text{GRU}_{\text{QF}}$) act as **recurrent memory gates** across the $K$ iterations, allowing the model to incrementally refine node states rather than overwriting them.

---

## Phase 3 · Temporal Attention (Cross-Snapshot Memory)

> *Tracks flows across handovers and mobility events using a sliding window of $T$ past snapshots.*

Let $\big\{\mathbf{H}^{(t)}\big\}_{t=1}^{T}$ be the past flow-state snapshots, each aligned to the current flow count $F$ (padded/cropped). Let $\mathbf{p}_t \in \mathbb{R}^D$ be a **learned positional embedding** for time step $t$.

**History encoding:**

$$
\tilde{\mathbf{H}}^{(t)} = \mathbf{H}^{(t)} + \mathbf{p}_t \cdot \mathbf{1}^\top, \quad t = 1,\dots,T
\tag{7.1}
$$

$$
\mathbf{M} = \text{stack}\!\big(\tilde{\mathbf{H}}^{(1)}, \dots, \tilde{\mathbf{H}}^{(T)}\big) \in \mathbb{R}^{F \times T \times D}
\tag{7.2}
$$

**Cross-attention** ($\text{Query}$ = current flow state, $\text{Key/Value}$ = temporal memory):

For each flow $i$, for each head $m$:

$$
\mathbf{q}_i^{(m)} = W_Q^{(m)}\,\mathbf{h}_{f_i}, \quad
\mathbf{k}_{i,t}^{(m)} = W_K^{(m)}\,\tilde{\mathbf{H}}_{i}^{(t)}, \quad
\mathbf{v}_{i,t}^{(m)} = W_V^{(m)}\,\tilde{\mathbf{H}}_{i}^{(t)}
\tag{7.3}
$$

$$
\gamma_{i,t}^{(m)} = \text{softmax}_t\!\left(\frac{\big(\mathbf{q}_i^{(m)}\big)^\top \mathbf{k}_{i,t}^{(m)}}{\sqrt{d_h}}\right)
\tag{7.4}
$$

$$
\text{ctx}_i = W_{\text{proj}}\;\Big[\textstyle\sum_{t=1}^{T} \gamma_{i,t}^{(1)}\,\mathbf{v}_{i,t}^{(1)} \;\|\; \cdots \;\|\; \textstyle\sum_{t=1}^{T} \gamma_{i,t}^{(H)}\,\mathbf{v}_{i,t}^{(H)}\Big]
\tag{7.5}
$$

**Residual + Layer Normalization:**

$$
\mathbf{h}_{f_i} \;\leftarrow\; \text{LayerNorm}\!\Big(\mathbf{h}_{f_i} + \text{ctx}_i\Big)
\tag{7.6}
$$

---

## Phase 4 · Cross-Flow Attention (Readout Refinement)

> *Flows sharing the same queue (same UE) attend to each other to capture co-located correlations.*

Full self-attention with a **same-queue mask**. For each head $m$:

$$
\mathbf{q}_i^{(m)} = W_Q^{(m)}\,\mathbf{h}_{f_i}, \quad
\mathbf{k}_j^{(m)} = W_K^{(m)}\,\mathbf{h}_{f_j}, \quad
\mathbf{v}_j^{(m)} = W_V^{(m)}\,\mathbf{h}_{f_j}
\tag{8.1}
$$

$$
s_{ij}^{(m)} = \frac{\big(\mathbf{q}_i^{(m)}\big)^\top \mathbf{k}_j^{(m)}}{\sqrt{d_h}}
\tag{8.2}
$$

**Same-queue masking** (blocks attention across different queues):

$$
\hat{s}_{ij}^{(m)} =
\begin{cases}
s_{ij}^{(m)} & \text{if } \text{f2q}(i) = \text{f2q}(j) \\
-\infty & \text{otherwise}
\end{cases}
\tag{8.3}
$$

$$
\alpha_{ij}^{(m)} = \text{softmax}_j\!\big(\hat{s}_{ij}^{(m)}\big)
\tag{8.4}
$$

$$
\text{out}_i = W_{\text{proj}}\;\Big[\textstyle\sum_j \alpha_{ij}^{(1)}\,\mathbf{v}_j^{(1)} \;\|\; \cdots \;\|\; \textstyle\sum_j \alpha_{ij}^{(H)}\,\mathbf{v}_j^{(H)}\Big]
\tag{8.5}
$$

**Residual + Layer Normalization:**

$$
\mathbf{h}_{f_i} \;\leftarrow\; \text{LayerNorm}\!\Big(\mathbf{h}_{f_i} + \text{out}_i\Big)
\tag{8.6}
$$

---

## Phase 5 · Readout Head

A **3-layer MLP** maps each flow's final state to a scalar prediction:

$$
\mathbf{a}_i = \text{ReLU}\!\big(W_1\,\mathbf{h}_{f_i} + b_1\big), \quad \mathbf{a}_i \in \mathbb{R}^D
\tag{9.1}
$$

$$
\mathbf{b}_i = \text{ReLU}\!\big(W_2\,\text{Dropout}(\mathbf{a}_i) + b_2\big), \quad \mathbf{b}_i \in \mathbb{R}^{D/2}
\tag{9.2}
$$

$$
\hat{y}_i = W_3\,\text{Dropout}(\mathbf{b}_i) + b_3 \;\in\; \mathbb{R}^1
\tag{9.3}
$$

$$
\hat{\mathbf{y}} = \big[\hat{y}_1,\;\dots,\;\hat{y}_F\big] \;\in\; \mathbb{R}^F
\tag{9.4}
$$

> Output: one scalar per active flow — predicted **delay** (s) or **throughput** (bps), depending on the model instance.

---

## Full Pipeline Summary

$$
\boxed{
\begin{aligned}
&\textbf{Input:} && \mathbf{x}_f \in \mathbb{R}^{8},\; \mathbf{x}_q \in \mathbb{R}^{2},\; \mathbf{x}_l \in \mathbb{R}^{4} \\[4pt]
&\textbf{Phase 1 — Embed:} && \mathbf{h}^{(0)}_f,\; \mathbf{h}^{(0)}_q,\; \mathbf{h}^{(0)}_l \in \mathbb{R}^D \quad \text{(Eqs. 1.1–1.3)} \\[4pt]
&\textbf{Phase 2 — Message Passing} && \text{for } k = 1 \dots K: \\
&\quad \text{Step 1: F→Q Attention + GRU} && \text{(Eqs. 2.1–2.6)} \\
&\quad \text{Step 2a: L→L Self-Attention} && \text{(Eqs. 3.1–3.4)} \\
&\quad \text{Step 2b: L→Q Attention + GRU} && \text{(Eqs. 4.1–4.5)} \\
&\quad \text{Step 2c: Q→L Residual} && \text{(Eqs. 5.1–5.2)} \\
&\quad \text{Step 3:\;\; Q→F Gating + GRU} && \text{(Eqs. 6.1–6.3)} \\[4pt]
&\textbf{Phase 3 — Temporal Attention:} && \text{(Eqs. 7.1–7.6)} \\[4pt]
&\textbf{Phase 4 — Cross-Flow Attention:} && \text{(Eqs. 8.1–8.6)} \\[4pt]
&\textbf{Phase 5 — Readout MLP:} && \hat{\mathbf{y}} \in \mathbb{R}^F \quad \text{(Eqs. 9.1–9.4)}
\end{aligned}
}
$$

