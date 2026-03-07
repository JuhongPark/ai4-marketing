# Deep Dive 1: GNN-based Recommender Systems

How Graph Neural Networks scaled personalized recommendation to billions of items by learning directly from the user–item interaction graph.

---

## 1. Problem Definition

**Recommendation** is the task of ranking items for a given user according to predicted relevance. Modern recommendation systems handle tens of millions of users and hundreds of millions of items with sub-100ms latency constraints.

### The Graph Structure of Recommendation

User–item interactions naturally form a **bipartite graph**:
- Users on one side, items on the other
- Edges = purchases, clicks, views, ratings
- Often augmented with user–user (social) edges and item–item (similarity, knowledge graph) edges

Classical methods (matrix factorization, neural CF) ignore multi-hop connectivity. A user's taste is influenced not only by items they directly interacted with, but by items interacted with by similar users — and by items those items' users liked — recursively. Message passing on the graph captures exactly this.

### Why GNN?

| Requirement | Why classical methods fall short | Why GNN fits |
|-------------|----------------------------------|--------------|
| Scale (billions of items) | MF requires all users × items in memory | GNN embeds nodes once, indexes them for retrieval |
| Cold start | Pure CF has no signal for new users/items | Feature-aware GNN (GraphSAGE, PinSage) generalizes via node features |
| Multi-hop signal | Neural CF is single-step | Message passing captures K-hop connectivity |
| Heterogeneous graph | MF is single-edge-type | Heterogeneous GNN (KGAT) handles multiple edge types |
| Session sequence | MF is static | SR-GNN models session as directed graph |

---

## 2. GraphSAGE — Inductive Representation Learning on Large Graphs (NeurIPS 2017)

**Paper**: Hamilton, Ying & Leskovec
**Links**: [Paper](https://arxiv.org/abs/1706.02216) | [Code](https://github.com/williamleif/GraphSAGE) | [NeurIPS PDF](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)

### Key Insight

Existing GNN approaches (GCN, 2016) are **transductive**: every node in the graph must be present during training. Any new node (a new product, a new user) requires retraining the whole model. GraphSAGE introduced the **inductive** formulation: learn a **function** that generates node embeddings by sampling and aggregating features from the node's local neighborhood.

### Architecture

```
For each node v:
  1. Sample K-hop neighborhood (fixed-size per hop)
  2. For k = 1..K:
      h_v^{k} = σ( W · AGGREGATOR(h_v^{k-1}, {h_u^{k-1} : u ∈ N(v)}) )
  3. Final embedding = h_v^{K}
```

**Aggregator variants**: Mean, LSTM over a random permutation of neighbors, or max-pooling after an MLP. All three are compared empirically.

### Results

| Dataset | Task | vs. DeepWalk/GraphSAGE-GCN |
|---------|------|----------------------------|
| Citation | Classify paper category | **+51% F1** over DeepWalk |
| Reddit | Classify post community | **+19% F1** over GraphSAGE-GCN |
| Protein-protein interaction | Classify protein function | **+66% F1** inductive generalization |

### Why It Matters for Marketing Recommendation

Catalog churn is constant in e-commerce — new products arrive daily. Transductive models (GCN, LightGCN trained from scratch) require retraining. GraphSAGE-style inductive embedding enables **online serving of new items** without retraining the full graph.

---

## 3. PinSage — Web-scale GCN at Pinterest (KDD 2018)

**Paper**: Ying, He, Chen, Eksombatchai, Hamilton & Leskovec
**Links**: [Paper](https://arxiv.org/abs/1806.01973) | [Pinterest Engineering blog](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)

### Scale Achievement

PinSage is the first GCN deployed at true web scale:
- **3 billion nodes** (pins and boards)
- **18 billion edges**
- **7.5 billion training examples**
- Trained and served at Pinterest

### Key Innovations

1. **Importance-based neighborhood**: Instead of expanding the full K-hop neighborhood (combinatorially explosive), PinSage simulates random walks starting at the target node and selects the top-`T` most-visited neighbors. Top visit-count neighbors get higher influence during aggregation.
2. **Hard negative mining via curriculum learning**: Training starts with easy negatives, progressively adds hard negatives from a candidate set of items with similar high-level features.
3. **MapReduce inference pipeline**: Billions of embeddings are materialized offline for online retrieval.

### Architecture

```
Random walk starting at pin v
  ↓ top-T visited neighbors
Weighted aggregation via MLP on (neighbor feats, walk counts)
  ↓
Concatenate with v's own features, pass through MLP
  ↓
Output: v's embedding
```

### Results (A/B Tests at Pinterest)

| Metric | Lift over prior content-based recommender |
|--------|-------------------------------------------|
| User engagement (repins) | **+150%** in hit-rate @ top-10 |
| Offline MRR | **+40% to +60%** vs. production baseline |

### Why It Matters

PinSage proved GCN could power recommendations at the scale of a major social platform, ending the "GNN is a research-only technique" narrative.

---

## 4. NGCF — Neural Graph Collaborative Filtering (SIGIR 2019)

**Paper**: Wang, He, Wang, Feng & Chua
**Links**: [Paper](https://arxiv.org/abs/1905.08108) | [Code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) | [ACM](https://dl.acm.org/doi/10.1145/3331184.3331267)

### Core Contribution

NGCF is the first paper to explicitly propose **injecting collaborative signal into embeddings via high-order connectivity propagation on the user–item bipartite graph**. Before NGCF, deep recommenders learned user/item embeddings independently and only combined them at prediction time. NGCF propagates embeddings through the graph so that a user's embedding is **directly shaped by the items they interacted with**, and vice versa.

### Architecture

```
Initial user and item embeddings e_u^{0}, e_i^{0}
  ↓
For ℓ = 1..L (propagation layers):
  e_u^{ℓ} = LeakyReLU( Σ_{i ∈ N_u} ( W1 · (e_i^{ℓ-1} ⊙ e_u^{ℓ-1}) + W2 · e_i^{ℓ-1} ) )
  (analogous for items)
  ↓
Final: e_u = concat(e_u^{0}, e_u^{1}, ..., e_u^{L})
  ↓
Prediction: ŷ(u, i) = e_u^{T} e_i
```

The interaction-encoded aggregation `e_i ⊙ e_u` is the key: it makes the message from item i to user u depend on how well i and u already align, echoing explicit collaborative filtering signal.

### Results (3 public benchmarks: Gowalla, Yelp2018, Amazon-Book)

| Metric | NGCF gain over prior SOTA (HOP-Rec, CMN) |
|--------|------------------------------------------|
| Recall@20 | **+7.8% to +14.1%** |
| NDCG@20 | **+7.2% to +12.6%** |

### Why It Matters

NGCF established the template that subsequent GNN recommenders (LightGCN, LR-GCCF, DGCF) extended and refined.

---

## 5. LightGCN — Simplifying GCN for Recommendation (SIGIR 2020)

**Paper**: He, Deng, Wang, Li, Zhang & Wang
**Links**: [Paper](https://arxiv.org/abs/2002.02126)

### The Provocation

NGCF used the full GCN pipeline: feature transformation, non-linear activation, and neighborhood aggregation. LightGCN systematically ablates each and discovers that **feature transformation and non-linear activation actively hurt** collaborative filtering performance. Only the neighborhood aggregation — the simplest component — is essential.

### Architecture

```
e_u^{ℓ+1} = Σ_{i ∈ N_u} (1 / √(|N_u| · |N_i|)) · e_i^{ℓ}
e_i^{ℓ+1} = Σ_{u ∈ N_i} (1 / √(|N_u| · |N_i|)) · e_u^{ℓ}

Final: e_u = Σ_{ℓ=0..K} α_ℓ · e_u^{ℓ}
```

No weight matrices. No ReLU. Just symmetric-normalized sum with a layer-wise weighted combination for the final embedding.

### Results (same benchmarks as NGCF)

| Metric | LightGCN vs. NGCF |
|--------|-------------------|
| Recall@20 | **+16.5%** on average |
| NDCG@20 | **+16.9%** on average |
| Training speed | Significantly faster (fewer parameters) |
| Parameters | Dramatically fewer |

### Why It Matters

LightGCN is the **strongest baseline** in modern recommender benchmarks and often the production choice for GNN-based CF — its simplicity makes deployment tractable while maintaining (or improving) accuracy. It also offers a methodological lesson: inherited "best practices" from general deep learning (activation, feature transformation) do not automatically transfer to collaborative filtering.

---

## 6. KGAT — Knowledge Graph Attention Network (KDD 2019)

**Paper**: Wang, He, Cao, Liu & Chua
**Links**: [Paper](https://arxiv.org/abs/1905.07854) | [Code](https://github.com/xiangwang1223/knowledge_graph_attention_network)

### Motivation

Pure CF treats items as opaque IDs. Real item catalogs have rich structure: a movie has genre, director, actors; a product has category, brand, ingredients. Knowledge graphs encode this structure as (entity, relation, entity) triples. KGAT fuses the user–item interaction graph with the item knowledge graph and propagates embeddings over the union — with attention over different relation types.

### Architecture

```
Combined graph:
  Users ↔ Items (interactions)
  Items ↔ Attributes ↔ Items (knowledge graph)
  ↓
For each node:
  Propagate embeddings with attention:
    α(h, r, t) = MLP(e_h, e_r, e_t)  (relation-aware importance)
    e_h ← e_h + Σ_{(h,r,t) ∈ G} α(h, r, t) · e_t
  ↓
Final user/item embeddings used for ranking
```

### Results (3 benchmarks: Amazon-Book, Last-FM, Yelp2018)

| Metric | KGAT gain over NFM, RippleNet |
|--------|-------------------------------|
| Recall@20 | **+5.2% to +12.5%** |
| NDCG@20 | **+4.8% to +11.9%** |

### Why It Matters for Marketing

Explicit knowledge-aware recommendation enables richer **explanation** of why an item was recommended ("because you liked *director X*, who directed *movie Y*"). Interpretable attention weights are increasingly important for compliance in regulated industries (finance, healthcare, pharma marketing).

---

## 7. SR-GNN — Session-based Recommendation with GNN (AAAI 2019)

**Paper**: Wu, Tang, Zhu, Wang, Xie & Tan
**Links**: [Paper](https://arxiv.org/abs/1811.00855) | [Code](https://github.com/CRIPAC-DIG/SR-GNN)

### The Session Problem

In many real platforms (news, e-commerce guest browsing), the user's identity is unknown — only the current session of clicks is visible. Classical CF cannot apply (no user embedding). RNN-based approaches (GRU4Rec) model the session as a sequence but miss complex non-linear transitions between items.

### Architecture

```
Session: [i_1, i_2, i_3, i_4]
  ↓
Construct directed session graph (nodes = unique items, edges = transitions)
  ↓
Gated Graph Neural Network (GGNN):
  Propagate item embeddings on the session graph
  ↓
Attention over session items with respect to the last item:
  (captures "global preference" + "current interest")
  ↓
Session embedding → ranking over all items
```

### Results (Yoochoose, Diginetica benchmarks)

| Metric | SR-GNN vs. GRU4Rec |
|--------|--------------------|
| Recall@20 | **+12.3%** average |
| MRR@20 | **+9.7%** average |

### Why It Matters for Marketing

Session-based recommendation is the dominant use case for **guest shoppers** and **cold-start** personalization — precisely where user-level CF cannot help. SR-GNN became the default baseline for session-based recommender research.

---

## 8. Cross-Method Comparison

| Aspect | GraphSAGE | PinSage | NGCF | LightGCN | KGAT | SR-GNN |
|--------|-----------|---------|------|----------|------|--------|
| **Year / Venue** | NeurIPS 2017 | KDD 2018 | SIGIR 2019 | SIGIR 2020 | KDD 2019 | AAAI 2019 |
| **Target task** | General node embedding | Pin recommendation | CF ranking | CF ranking | KG-aware CF | Session-based |
| **Inductive** | **Yes** | **Yes** (via features) | No | No | No | N/A (per-session) |
| **Cold start** | Strong | Strong | Weak | Weak | Moderate | Strong |
| **Key innovation** | Inductive aggregation | Random-walk sampling at web scale | High-order CF signal on bipartite graph | Simplification of GCN | Attention over KG relations | Session as graph |
| **Best when** | Evolving catalog | Billion-item catalog with features | Dense CF data | Pure CF, best accuracy per FLOP | Items have rich attributes | No user identity |
| **Production deployed** | Multiple | Pinterest | Research | Multiple | Multiple | Multiple |

### Evolution of the Field

```
Matrix Factorization (2000s)
  ↓
Neural CF (NeuMF, 2017) — introduces MLP interaction
  ↓
GCN (Kipf & Welling, 2016) — transductive node embedding
  ↓
GraphSAGE (2017) — inductive aggregation
  ↓
PinSage (2018) — web-scale production GCN
  ↓
NGCF (2019) — explicit high-order CF propagation
  KGAT (2019) — knowledge-aware recommendation
  SR-GNN (2019) — session graphs
  ↓
LightGCN (2020) — simplification, current strong baseline
  ↓
Sequential-aware GNN, Contrastive learning on graphs (2021+)
```

---

## 9. Open Problems and Future Directions

1. **Dynamic / temporal graphs**: Real user–item graphs evolve continuously. Temporal GNNs (TGN, TGAT) offer principled time handling but haven't been widely adopted in production recommenders.

2. **Cold start beyond features**: Inductive GNNs help when new items have features, but genuinely zero-signal cold start remains open. Contrastive/self-supervised GNN pretraining is a promising direction.

3. **Bias and fairness**: GNN recommenders amplify popularity bias and homophily. Debiased GNN CF (DEGNN, BiasContrastive) is active research.

4. **Scalability beyond Pinterest**: PinSage demonstrated billions of nodes, but GPT-scale item graphs (trillions of cross-shop interactions) remain beyond current methods. Distributed / sharded GNN training is an infrastructure problem as much as a modeling problem.

5. **Alignment with downstream business metrics**: Recall@K and NDCG@K don't always predict revenue. Off-policy evaluation and counterfactual ranking metrics close this gap but integrate poorly with GNN training.

6. **Privacy-preserving GNN recommendation**: Federated GNNs (so user data never leaves the device) are an open research area — critical for post-cookie personalization.

---

## 10. References

- Hamilton, Ying, Leskovec (2017). Inductive Representation Learning on Large Graphs. NeurIPS. [Paper](https://arxiv.org/abs/1706.02216) | [Code](https://github.com/williamleif/GraphSAGE)
- Ying et al. (2018). Graph Convolutional Neural Networks for Web-Scale Recommender Systems. KDD. [Paper](https://arxiv.org/abs/1806.01973)
- Wang, He, Wang, Feng, Chua (2019). Neural Graph Collaborative Filtering. SIGIR. [Paper](https://arxiv.org/abs/1905.08108) | [Code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
- He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR. [Paper](https://arxiv.org/abs/2002.02126)
- Wang, He, Cao, Liu, Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. KDD. [Paper](https://arxiv.org/abs/1905.07854) | [Code](https://github.com/xiangwang1223/knowledge_graph_attention_network)
- Wu et al. (2019). Session-based Recommendation with Graph Neural Networks. AAAI. [Paper](https://arxiv.org/abs/1811.00855) | [Code](https://github.com/CRIPAC-DIG/SR-GNN)
- Wu et al. (2022). Graph Neural Networks in Recommender Systems: A Survey. ACM Computing Surveys. [Paper](https://dl.acm.org/doi/10.1145/3535101)
