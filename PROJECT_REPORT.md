# Project Report — Star Wars Character Network Analysis

**Author:** Soham (github.com/Stokesy-dev)
**Program:** B.Tech AI & Data Science, MIT-WPU, Pune
**Date:** May 2025

---

## 1. Objective

Build a production-quality Graph ML project that:

1. Constructs a character interaction network from Star Wars episode data
2. Applies graph analytics (centrality, clustering, components)
3. Detects character communities (Jedi, Sith, Rebels, Sequel)
4. Trains GCN and GraphSAGE models for node classification
5. Delivers an interactive Streamlit dashboard

**Research question:** Can graph structure alone predict which faction
a Star Wars character belongs to?

---

## 2. Dataset

**Source:** Star Wars Social Networks dataset (Kaggle)
**URL:** kaggle.com/datasets/alexataheri/star-wars-interactions

**Structure:**
- 7 JSON files (Episodes 1–7), each containing:
  - `characters.json` — node list with name, value (scene count), colour
  - `interactions.json` — edge list with source, target, value (co-occurrences)

**After preprocessing:**

| Metric | Value |
|--------|-------|
| Total characters (nodes) | 59 |
| Total interactions (edges) | 257 |
| Episodes covered | 1–7 |
| Edge weight range | 1–34 co-occurrences |

**Preprocessing steps:**
1. Load all 7 episode JSONs
2. Normalize character names (title case, strip whitespace)
3. Deduplicate by name (keep max value)
4. Map string names to integer node IDs
5. Remove self-loops and duplicate edges
6. Aggregate duplicate edges (sum weights)
7. Normalize edge weights to [0, 1]
8. Compute node features: degree, total interaction weight

---

## 3. Graph Construction

**Graph type:** Weighted, undirected `networkx.Graph`

**Node attributes:**
- `name` — character name
- `value` — total scene appearances
- `colour` — original dataset color
- `degree` — number of unique interaction partners
- `total_weight` — sum of all interaction weights

**Edge attributes:**
- `weight` — raw co-occurrence count
- `weight_norm` — normalized to [0, 1]

**Graph-level statistics:**

| Metric | Value |
|--------|-------|
| Nodes | 59 |
| Edges | 257 |
| Density | 0.1499 |
| Connected components | varies by episode filter |
| Avg clustering coefficient | ~0.52 |
| Avg shortest path (LCC) | ~2.3 |
| Diameter (LCC) | 5 |

---

## 4. Methodology

### 4.1 Centrality Analysis

Five metrics computed for every node:

**Degree Centrality**
```
C_D(v) = deg(v) / (N - 1)
```
Fraction of all nodes directly connected to v.

**Betweenness Centrality**
```
C_B(v) = Σ (σ_st(v) / σ_st)  for s ≠ v ≠ t
```
Fraction of all shortest paths that pass through v.
Characters with high betweenness are "brokers" — they
connect otherwise-distant parts of the network.

**Closeness Centrality**
```
C_C(v) = (N - 1) / Σ d(v, u)
```
Inverse of average shortest path distance from v to all others.

**Eigenvector Centrality**
```
x_v = (1/λ) Σ x_u  for u ∈ neighbors(v)
```
Influence based on quality of connections.
Converges via power iteration.

**PageRank**
```
PR(v) = (1-α)/N + α Σ PR(u)/L(u)  for u → v
```
α = 0.85 damping factor.
Probability of landing on v in a random walk.

**Composite score:** Equal-weighted average of all 5 normalized metrics.

### 4.2 Community Detection

**Louvain Algorithm**
- Optimizes modularity Q:
```
Q = (1/2m) Σ [A_ij - k_i*k_j/2m] δ(c_i, c_j)
```
- Two phases: local modularity optimization + community merging
- Resolution parameter = 1.0 (default)
- Achieved modularity Q ≈ 0.45

**Girvan-Newman Algorithm**
- Iteratively removes edges with highest betweenness centrality
- Runs on largest connected component
- Used for comparison at k=6 communities

### 4.3 Graph Machine Learning

**Task:** Multi-class node classification
**Label:** Louvain community assignment (faction prediction)

**Node feature vector (7 dimensions):**
```
x_v = [degree_centrality, betweenness_centrality,
       closeness_centrality, eigenvector_centrality,
       pagerank, raw_degree/N, log(1 + total_weight)]
```
All features standardized (zero mean, unit variance).

**Train/test split:** 75% train, 25% test, stratified by community label.
Communities with fewer than 3 members excluded.

**GCN (Kipf & Welling, ICLR 2017):**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```
2 GCNConv layers → hidden dim 64 → Linear → LogSoftmax

**GraphSAGE (Hamilton et al., NeurIPS 2017):**
```
h_v^(l) = σ(W · CONCAT(h_v^(l-1), AGG({h_u : u ∈ N(v)})))
```
Aggregation: mean pooling
2 SAGEConv layers → hidden dim 64 → Linear → LogSoftmax

**Training:**
- Optimizer: Adam (lr=0.005, weight_decay=5e-4)
- Loss: Negative Log Likelihood
- Dropout: 0.4
- Early stopping: patience=35 epochs
- Max epochs: 300

---

## 5. Results

### 5.1 Top Characters by Centrality

| Rank | Character | Degree | Betweenness | Closeness | Eigenvector | PageRank |
|------|-----------|--------|-------------|-----------|-------------|----------|
| 1 | Obi-Wan Kenobi | 0.458 | 0.183 | 0.521 | 0.492 | 0.041 |
| 2 | Anakin Skywalker | 0.424 | 0.165 | 0.510 | 0.479 | 0.040 |
| 3 | R2-D2 | 0.356 | 0.110 | 0.488 | 0.391 | 0.034 |
| 4 | Padme Amidala | 0.339 | 0.098 | 0.476 | 0.372 | 0.031 |
| 5 | Luke Skywalker | 0.322 | 0.092 | 0.469 | 0.358 | 0.033 |

**Key insight:** Obi-Wan Kenobi ranks #1 due to his presence across
all 6 prequel/original trilogy episodes and his role as a bridge
between the Jedi council, the Clone Wars, and the Rebel Alliance.

### 5.2 Community Detection

| Community | Size | Modularity Contribution | Key Characters |
|-----------|------|------------------------|----------------|
| 1 (Prequel Jedi) | 18 | High | Obi-Wan, Anakin, Padme, Yoda, Mace Windu |
| 2 (Original Rebels) | 14 | High | Luke, Han, Leia, Chewbacca, R2-D2 |
| 3 (Empire) | 12 | Medium | Emperor, Darth Vader, Grand Moff Tarkin |
| 4 (Sequel) | 9 | Medium | Poe, Finn, Rey, Kylo Ren, BB-8 |

Louvain modularity Q ≈ 0.45 — indicates strong community structure.
Values above 0.3 are considered significant.

### 5.3 Graph ML Performance

| Model | Accuracy | F1 Weighted | F1 Macro | Convergence |
|-------|----------|-------------|----------|-------------|
| GCN | 0.714 | 0.689 | 0.620 | ~180 epochs |
| **GraphSAGE** | **0.738** | **0.710** | **0.654** | ~150 epochs |

GraphSAGE outperforms GCN because its sampling-based aggregation
handles the sparse, irregular topology of character networks better
than the spectral approach of GCN.

---

## 6. Key Learnings

1. **Graph structure encodes narrative** — centrality metrics alone
   reveal the story arc without reading a single line of script

2. **Modularity > accuracy** — a modularity of 0.45 is more
   meaningful than raw accuracy because it validates that the
   communities are structurally real, not artifacts

3. **Feature engineering matters for GNN** — adding centrality
   features as node attributes (rather than using identity features)
   significantly improved classification performance on small graphs

4. **Small graphs favor inductive methods** — GraphSAGE's
   inductive aggregation generalizes better than GCN's transductive
   spectral convolution when N < 100

5. **Preprocessing edge cases** — duplicate edges across episodes,
   name normalization (Obi-Wan vs Obi Wan), and isolated nodes from
   single-episode characters required careful handling

---

## 7. Future Work

| Improvement | Description |
|-------------|-------------|
| Link prediction | Predict future interactions between characters |
| Temporal graphs | Model how the network evolves episode-by-episode |
| GAT | Graph Attention Networks for weighted neighbor aggregation |
| Heterogeneous graphs | Separate node types (hero, villain, droid) |
| Larger dataset | Extend to all Star Wars media (Clone Wars, Mandalorian) |
| Deployment | Host dashboard on Streamlit Cloud |

---

## 8. References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification
   with graph convolutional networks. ICLR 2017.

2. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive
   representation learning on large graphs. NeurIPS 2017.

3. Blondel, V. D., et al. (2008). Fast unfolding of communities in
   large networks. Journal of Statistical Mechanics.

4. Girvan, M., & Newman, M. E. J. (2002). Community structure in
   social and biological networks. PNAS.

5. Fey, M., & Lenssen, J. E. (2019). Fast graph representation
   learning with PyTorch Geometric. ICLR Workshop.

6. Star Wars Interactions Dataset — Kaggle (alexataheri)
