# Recommendation Systems (GNN-based)

How graph neural networks scaled personalized recommendation to billions of items by learning directly from user–item interaction graphs.

---

## Domain Background

Personalized recommendation is the core revenue driver for e-commerce, streaming, and social platforms. The field has evolved through three phases:

1. **Matrix Factorization** (2000s): SVD, ALS, implicit-feedback MF. Simple and scalable but ignores graph structure and content.
2. **Neural Collaborative Filtering** (2017+): Replaces the inner product with an MLP. Captures non-linear interactions but still single-step.
3. **GNN-based CF** (2018+): Treats user–item interactions as a bipartite graph and learns embeddings via message passing. Dominant approach today.

---

## AI-based Approaches

### PinSage — Web-scale GCN for Pinterest (KDD 2018)

- **Source**: KDD 2018 ([paper](https://arxiv.org/abs/1806.01973)) | [Pinterest Engineering blog](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)
- **What it does**: Graph Convolutional Network deployed at Pinterest combining efficient random walks with graph convolutions to produce pin embeddings.
- **Scale**: Trained on **7.5B examples** over a graph with **3B nodes** (pins and boards) and **18B edges** — the largest deep recommender deployment at the time.
- **Key innovation**: Importance-based neighborhood sampling via random walks instead of full K-hop neighbors — enables web-scale training.
- **Why it matters**: Proof that GCNs can power recommendations at industrial scale.

### LightGCN — Simplifying GCN for Collaborative Filtering (SIGIR 2020)

- **Source**: SIGIR 2020 ([paper](https://arxiv.org/abs/2002.02126))
- **What it does**: Strips non-linear activations and feature transformations from GCN, keeping only **neighborhood aggregation** — the essential GCN component for collaborative filtering. The final embedding is a weighted sum of embeddings at each layer.
- **Key finding**: Feature-transformation and non-linear components of standard GCN hurt CF performance. Removing them makes the model **simpler, faster, and more accurate**.
- **Impact**: One of the strongest baselines in modern recommender benchmarks; the reference point against which new GNN recommenders are measured.

### GNN in Recommender Systems — Comprehensive Survey (ACM Computing Surveys, 2022)

- **Source**: ACM Computing Surveys ([survey](https://dl.acm.org/doi/10.1145/3535101))
- **Scope**: Categorizes GNN-based recommenders across collaborative filtering, sequential recommendation, social recommendation, knowledge-graph-aware, and session-based tasks.
- **Identified challenges**: Scalability to billion-item catalogs, cold-start, fairness, dynamic graph handling, and heterogeneous edge types.

### LightGCN + Knowledge-Aware Attention Sub-Network (2025)

- **Source**: Scientific Reports ([paper](https://www.nature.com/articles/s41598-025-99949-y))
- **What it does**: Combines LightGCN's simplicity with a personalized knowledge-aware attention sub-network, pulling external knowledge-graph signals into the recommender.
- **Result**: Stronger performance than LightGCN alone on benchmark datasets, particularly in cold-start scenarios where interaction data is sparse but KG context is available.

---

## Why AI?

Recommendation is inherently a graph problem (users ↔ items) at massive scale. GNNs provide the right inductive bias: the relevance of an item to a user depends on the joint neighborhood of both in the interaction graph. No classical method matches this efficiently at web scale, and industrial deployments (Pinterest, Alibaba, Meituan, Uber Eats, Spotify) have converged on GNN-based approaches.
