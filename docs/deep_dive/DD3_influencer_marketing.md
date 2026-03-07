# Deep Dive 3: Social Influence Prediction & Influencer Marketing

How AI predicts who will be influenced, who will influence, and which campaigns will go viral — from deep-learning influence prediction to ML-powered influencer marketing analytics.

---

## 1. Problem Definition

Influencer marketing, now a **~$24B global industry** (Statista, 2024), subsumes several related AI problems:

| Sub-problem | Question |
|-------------|----------|
| **Influence prediction** | Given a user's ego-network, will they adopt / share / purchase after neighbor activity? |
| **Influencer identification** | Which accounts are actually influential (vs. merely popular)? |
| **Campaign outcome prediction** | Will this paid influencer campaign generate engagement / conversions / sales? |
| **Creative-content scoring** | Which posts from which influencers will drive what response? |

Classical network statistics (degree, PageRank, clustering coefficient) capture only a thin slice. Deep learning, GNN, and ML on multi-modal content offer richer signal.

### Why AI?

- **High-dimensional**: Influencer profiles have follower counts, engagement history, brand fit, audience demographics, visual aesthetics, caption style — ML absorbs all.
- **Non-linear**: Influence depends on interactions among features in ways regression cannot capture.
- **Content + network**: Modern influence depends on **both** social topology and post content; joint GNN + vision/language models are essential.
- **Cold start**: New influencers and new niches require transfer learning, not retraining from zero.

---

## 2. DeepInf — Social Influence Prediction with Deep Learning (KDD 2018)

**Paper**: Qiu, Tang, Ma, Dong, Wang & Tang
**Links**: [Paper](https://arxiv.org/abs/1807.05560) | [Code](https://github.com/xptree/DeepInf)

### The Foundational Work

DeepInf is the first major deep-learning framework for social influence prediction. It showed that **learned representations outperform hand-crafted features** across Twitter, Weibo, OAG, and Digg.

### Architecture (Layer by Layer)

```
1. [Ego Network Sampling]
   Random walk with restart from user v → sample n-node local subgraph

2. [Embedding Layer]
   Map each user to D-dimensional representation

3. [Instance Normalization]
   Normalize embeddings (zero mean, unit variance per instance)

4. [Input Concatenation]
   Network embedding + action status indicator + ego indicator + vertex features

5. [GNN Layer]  ← interchangeable: GCN or GAT
   Multiple layers of neighborhood aggregation

6. [Output Layer]
   Binary prediction: will user v take the action? (yes/no)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input scope | Local ego network (not full graph) | Influence is primarily local; full graph is too expensive at web scale |
| Sampling | Random walk with restart | Captures multi-hop neighborhood with locality bias |
| GNN variant | GAT (best) vs. GCN | Attention weights capture heterogeneous edge importance |

### Why It Matters for Influencer Marketing

- Directly answers: **given exposure to an influencer, will this follower convert?**
- The ego-network approach is **scalable** to the whole audience of a major influencer.
- Establishes the template (**representation learning > feature engineering**) for all downstream influencer-marketing ML.

---

## 3. HetInf — Heterogeneous Graph Neural Network (2021)

**Paper**: Gao, Wang et al., Frontiers in Physics
**Link**: [Paper](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.787185/full)

### Motivation

DeepInf treats the graph as homogeneous (all nodes and edges are the same type). Real influencer ecosystems have:
- **Multiple node types**: users, brands, posts, hashtags, stories
- **Multiple edge types**: follow, like, share, comment, mention, tag
- **Temporal ordering**: when an action happened matters as much as whether it happened

HetInf models all three simultaneously.

### Architecture (3-Module Design)

```
Step 1: Construct heterogeneous relational network
  - User nodes + Event nodes
  - Multiple edge types (user-user, user-event)

Step 2: Sample r-heterogeneous neighbor subgraph
  - Type-aware sampling preserving node/edge diversity

Step 3: Heterogeneous GNN (3 neural modules)
  ├── NN-1: Bi-LSTM       → aggregates temporal/semantic features
  ├── NN-2: GCN            → aggregates heterogeneous node attributes
  └── NN-3: GAT            → learns interaction weights between node types
  ↓
  Influence prediction for target user
```

### Results

| Dataset | Metric | HetInf vs. DeepInf |
|---------|--------|--------------------|
| Weibo | F1 | **+17.9%** |
| Weibo | Precision | **+35.7%** |
| Digg | F1 | Significant improvement |

### Key Insight

Ignoring event-level heterogeneity (posts, stories, hashtags, product tags) causes homogeneous models to miss critical signal. On Instagram / TikTok / Xiaohongshu, **what** is posted matters as much as **who** posts it.

---

## 4. DeepPP — Personalized Propagation (2022)

**Paper**: World Wide Web Journal
**Link**: [Paper](https://arxiv.org/abs/2207.13016)

### Core Idea

Combines DeepInf's local ego-network approach with **PPNP / APPNP**-style Personalized PageRank propagation, enabling **infinite-range propagation without over-smoothing**. Originally applied to COVID-19 information diffusion; directly transferable to viral marketing content.

### Architecture

```
DeepInf (local ego-network + GAT)
  +
PPNP/APPNP propagation (Personalized PageRank transition probabilities)
  =
DeepPP — captures both local influence dynamics and long-range cascade patterns
```

### Why It Matters

Marketing content can go viral through weak ties (strangers to strangers). DeepInf's K-hop locality misses this; DeepPP's infinite-range propagation captures it.

---

## 5. ML for Influencer Campaign Success Prediction

### 5.1 Predicting the Success of Influencer Marketing Campaigns with ML (2025)

- **Source**: Springer ([chapter](https://link.springer.com/chapter/10.1007/978-981-96-3361-6_39))
- **What it does**: Analyzes historical campaign performance using ML (gradient boosting, random forest, neural networks) conditioned on influencer profile (location, cost, reach, engagement), audience demographics, and brand context. Predicts campaign success probability.
- **Why it matters**: Production-facing — the kind of tool brand marketers actually need at campaign-planning time.

### 5.2 Instagram Influencer Networks — Graph ML Approach (2024)

- **Source**: IIETA ([paper](https://www.iieta.org/journals/mmep/paper/10.18280/mmep.110806))
- **Methods**: Node2Vec + Word2Vec for influencer network embedding; downstream tasks include link prediction and community detection.
- **Result**: Graph-based embeddings recover hidden influencer sub-communities that degree / follower-count metrics miss.

### 5.3 Decoding Influencer Marketing Effectiveness on Instagram (2025)

- **Source**: Journal of Retailing and Consumer Services ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0969698925000645))
- **What it does**: Jointly models **image**, **text**, and **influencer-profile** features to predict post engagement and campaign effectiveness.
- **Significance**: Among the first papers to quantify the relative contribution of creative-content features (image aesthetics, caption sentiment) vs. influencer features (follower count, engagement rate) to outcome.

### 5.4 Novel Influence Quantification on Instagram (2024)

- **Source**: Social Network Analysis and Mining ([paper](https://link.springer.com/article/10.1007/s13278-024-01230-z))
- **What it does**: Data-science approach to influence quantification that goes beyond follower count, incorporating engagement per impression, audience overlap, and semantic relevance to target categories.
- **Why it matters**: Follower count is a notoriously bad proxy for actual influence. More granular influence metrics matter for brand-matching and budget allocation.

### 5.5 Predicting Consumer Adoption of Luxury via Instagram (2024)

- **Source**: PMC ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12603530/))
- **What it does**: Uses SVM + decision-tree ML to predict consumer luxury-product adoption from Instagram engagement patterns.
- **Result**: **~87% accuracy** in predicting adoption likelihood.

### 5.6 Virtual vs. Human Influencers — AI Comparison (2024)

- **Source**: Journal of Interactive Advertising ([paper](https://www.tandfonline.com/doi/full/10.1080/15252019.2024.2313721))
- **What it does**: Mixed-method comparison of human and virtual (AI-generated) influencers on Instagram, including ML-based engagement analysis.
- **Key finding**: Virtual influencers can match human influencers on certain metrics but underperform on trust-driven conversion — a critical finding for brands considering virtual-influencer budget.

---

## 6. MIT Sloan / Aral — Empirical Foundations

No AI-based influence model is credible without grounding against randomized-experiment results. Sinan Aral's group at MIT Sloan produced the definitive experiments.

### 6.1 Creating Social Contagion Through Viral Product Design (Management Science, 2011)

- **Link**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1564856)
- **Experiment**: 1.4M friends of 9,687 experimental users on Facebook
- **Result**: Passive-broadcast viral features generate **+246% peer influence / social contagion**; active-personalized viral features add an additional **+98%**.
- **Marketing translation**: Building viral mechanics into the product (share buttons, referral incentives) dominates paid influencer media for organic spread.

### 6.2 Identifying Influential and Susceptible Members of Social Networks (Science, 2012)

- **Link**: [Science](https://www.science.org/content/article/who-s-susceptible-peer-pressure-influencing)
- **Experiment**: 1.3M Facebook users randomized experiment
- **Findings**:
  - Younger users are **more susceptible** to peer influence
  - Men are **more influential** than women
  - Married individuals are **least susceptible** to influence
  - **Influence and susceptibility are negatively correlated** — influential people are less susceptible (and vice versa)

### 6.3 Identifying Social Influence in Networks Using Randomized Experiments (2011)

- **Link**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1907785)
- **Contribution**: Provides the methodology for separating **genuine social influence** from **homophily** (similar people happen to be connected and act similarly). Randomized exposure is the only reliable way.
- **Implication for AI**: Observational influence-prediction models cannot cleanly separate influence from homophily. AI-based influence prediction and classical causal inference are complementary, not interchangeable.

### 6.4 Exercise Contagion in a Global Social Network (Nature Communications, 2017)

- **Link**: [Nature Communications](https://www.nature.com/articles/ncomms14753)
- **Finding**: Exercise behavior propagates causally through social networks at global scale (data: multi-year global wearable data).
- **Marketing relevance**: Health, fitness, and lifestyle categories can rely on this propagation — campaigns that generate a base of visible adopters create authentic contagion.

### 6.5 Social Influence Bias — Randomized Experiment (Science, 2013)

- **Link**: [Science](https://www.science.org/doi/10.1126/science.1240466)
- **Experiment**: Randomized up/down-vote manipulation on a news-aggregation site (Muchnik, Aral, Taylor).
- **Result**: **Positive social influence (a single up-vote) increased positive ratings by +32%** and generated accumulating positive herding that raised final ratings by **+25%** on average. Negative manipulation was corrected by the crowd.
- **Implication for marketing**: Early positive signals (first reviews, first likes) have outsized effect on eventual engagement. Seeding positive signal is a legitimate and effective launch tactic.

### 6.6 The Spread of True and False News Online (Science, 2018)

- **Source**: Vosoughi, Roy & Aral — [Science](https://www.science.org/doi/10.1126/science.aap9559)
- **Data**: 126,000 news stories, 3M users on Twitter
- **Finding**: **False news spreads significantly farther, faster, deeper, and broader** than true news. Humans, not bots, are primarily responsible.
- **Implication**: Novelty and emotional arousal (fear, disgust, surprise) drive virality. Campaigns that optimize for arousal will outperform those optimizing for pure information quality — creating a tension between virality and trust that responsible marketing must navigate.

---

## 7. Cross-Method Comparison

| Aspect | DeepInf | HetInf | DeepPP | Campaign-Success ML | Aral Experiments |
|--------|---------|--------|--------|---------------------|-------------------|
| **Year** | 2018 | 2021 | 2022 | 2024–2025 | 2011–2018 |
| **Graph type** | Homogeneous | Heterogeneous | Homogeneous + PageRank | Tabular + graph | Observed + randomized |
| **Core method** | GNN (GCN/GAT) | Bi-LSTM + GCN + GAT | GAT + PPNP | Gradient boosting / RF / DL | Randomized field experiment |
| **Output** | Will user v be influenced? | Will user v be influenced (heterogeneous signal)? | Will user v be influenced (long-range)? | Will this campaign succeed? | Causal effect of influence |
| **Causal?** | No (correlational) | No | No | No | **Yes** |
| **Use case** | Model-building | Model-building | Model-building | Campaign planning | Validation & policy |

### The Right Use of Each

- **AI models (DeepInf, HetInf, DeepPP, campaign ML)**: Serve ongoing prediction for targeting and planning. Fast, scalable, but correlational.
- **Randomized experiments (Aral)**: Validate AI models, measure true causal effect, set reliable prior for Bayesian models.
- **Use together**: Aral experiments provide the causal ground truth; AI models generalize to at-scale deployment; the loop must be closed periodically to prevent model drift.

---

## 8. Open Problems and Future Directions

1. **Causal influence vs. homophily**: AI influence models still cannot cleanly separate the two without randomization. Causal-ML + observational-network data is an open frontier.

2. **Multi-modal content**: Visual and language features of posts are increasingly critical. Joint GNN + CLIP / vision-language models are emerging but not mature.

3. **Creator economy dynamics**: Virtual influencers, AI-generated avatars, AI-written captions — the creator ecosystem itself is becoming AI-driven, which changes the problem.

4. **Short-form video virality**: TikTok / Reels / Shorts exhibit different virality dynamics than static feeds. Existing influence-prediction models are optimized for timeline feeds.

5. **Cross-platform influence**: Users live on multiple platforms. Cross-platform influence models are barely explored.

6. **Compliance and disclosure**: Regulatory disclosure requirements for paid influencer posts affect engagement. Models that incorporate disclosure are missing from the literature.

7. **Incentive-compatible influencer rates**: Brands overpay for followers but underpay for genuine influence. Causal-inference-backed pricing remains open.

---

## 9. References

### Deep Learning for Influence Prediction
- Qiu et al. (2018). DeepInf: Social Influence Prediction with Deep Learning. KDD. [Paper](https://arxiv.org/abs/1807.05560) | [Code](https://github.com/xptree/DeepInf)
- Gao et al. (2021). HetInf: Heterogeneous GNN for Influence Prediction. Frontiers in Physics. [Paper](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.787185/full)
- DeepPP (2022). World Wide Web Journal. [Paper](https://arxiv.org/abs/2207.13016)

### ML for Influencer Marketing
- Predicting the Success of Influencer Marketing Campaigns Using ML (2025). Springer. [Chapter](https://link.springer.com/chapter/10.1007/978-981-96-3361-6_39)
- Exploring Instagram Influencer Networks (2024). IIETA. [Paper](https://www.iieta.org/journals/mmep/paper/10.18280/mmep.110806)
- Decoding Influencer Marketing Effectiveness on Instagram (2025). J. Retailing and Consumer Services. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0969698925000645)
- Novel Influence Quantification Model on Instagram (2024). Social Network Analysis and Mining. [Paper](https://link.springer.com/article/10.1007/s13278-024-01230-z)
- Predicting Consumer Adoption of Luxury via Instagram ML (2024). PMC. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12603530/)
- AI in Influencer Marketing: Human vs. Virtual (2024). J. Interactive Advertising. [Paper](https://www.tandfonline.com/doi/full/10.1080/15252019.2024.2313721)

### MIT Sloan / Aral — Randomized Experiments
- Aral & Walker (2011). Creating Social Contagion Through Viral Product Design. Management Science. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1564856)
- Aral & Walker (2012). Identifying Influential and Susceptible Members of Social Networks. Science. [Science](https://www.science.org/content/article/who-s-susceptible-peer-pressure-influencing)
- Aral & Walker (2011). Identifying Social Influence in Networks Using Randomized Experiments. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1907785)
- Aral & Nicolaides (2017). Exercise Contagion in a Global Social Network. Nature Communications. [Paper](https://www.nature.com/articles/ncomms14753)
- Muchnik, Aral, Taylor (2013). Social Influence Bias: A Randomized Experiment. Science. [Science](https://www.science.org/doi/10.1126/science.1240466)
- Vosoughi, Roy, Aral (2018). The Spread of True and False News Online. Science. [Paper](https://www.science.org/doi/10.1126/science.aap9559)
