# Deep Dive 2: Deep Reinforcement Learning for Viral & Paid Marketing

How deep RL agents learn to (a) pick viral campaign seeds, (b) bid in real-time ad auctions, and (c) personalize offers under budget and fairness constraints.

---

## 1. Overview

Marketing decisions are sequential, budget-constrained, and feedback-driven — a textbook match for reinforcement learning. Three problem families have seen rapid deep-RL progress:

| Problem | Classical approach | Deep RL reformulation |
|---------|--------------------|------------------------|
| **Viral / influencer seed selection** | Greedy on Monte Carlo cascades | GNN + DQN selects seeds in forward pass |
| **Real-time bidding (RTB)** | Linear bid model on predicted CTR | MDP with state = campaign budget, action = bid price |
| **Personalized treatment assignment** | A/B test, fixed allocation | Contextual bandit / offline RL |

This deep dive walks through canonical papers in each family and situates them in the marketing stack.

---

## 2. Viral Marketing: Influence Maximization via Deep RL

### 2.1 Classical Problem

**Influence Maximization** (Kempe, Kleinberg & Tardos, KDD 2003 — [paper](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf)):

> Given a network `G`, a diffusion model (Independent Cascade or Linear Threshold), and a budget `k`, find `k` seed users whose activation maximizes the expected total number of eventually activated users.

- **NP-hard** in the general case
- **Greedy gives ≥63% approximation** via submodularity, but requires expensive Monte Carlo simulation per candidate
- **Marketing translation**: Seed users = initial influencers receiving a campaign; "activated" users = users who share, purchase, or convert

### 2.2 DGN — Dual Graph Neural Network for Influence Maximization (2024)

- **Source**: Journal of Supercomputing, 2024 ([paper](https://link.springer.com/article/10.1007/s11227-024-06621-9))
- **Architecture**:
  ```
  Input Graph
    ↓
  [Dual Coupled GNNs]           ← two GNNs capturing topology + node attributes
    ↓
  Node Embeddings (context-rich)
    ↓
  [Deep Q-Network (DQN)]        ← approximates Q(state, action)
    ↓
  Sequential Seed Selection (greedy over Q-values)
  ```
- **Key result**: Matches or **surpasses SOTA** (ToupleGDD) on influence spread with **better robustness** on dense, complex networks and dramatically lower execution time than degree/K-shell/genetic/PSO baselines.
- **Why it matters**: Dual-GNN architecture learns a richer embedding than single-GNN approaches, which translates directly into better viral-campaign ROI.

### 2.3 BiGDN — Bidirectional GNN + Deep RL (2025)

- **Source**: Expert Systems with Applications, 2025 ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000065)) | [Code](https://github.com/zwl1985/BiGDN)
- **Architecture**:
  ```
  [Pre-training: influence regression]  ← warm-starts embeddings
    ↓
  [Bidirectional Neighborhood GNN]      ← captures incoming + outgoing influence asymmetry
    ↓
  [DRL Q-Network]                        ← sequential seed selection
    ↓
  BiGDNs (distilled variant)            ← knowledge distillation for inference speed
  ```
- **Result (9 real-world datasets)**:
  - Comparable to IMM (traditional SOTA) in effectiveness, with **significantly better efficiency**
  - Superior to other DRL methods on most datasets
  - Generalizes across network sizes and structures
- **Why it matters for marketing**: Directed influence is the norm (celebrity → fan, brand → customer, not symmetric). Bidirectional aggregation captures this correctly.

### 2.4 HEDRL-IM — Hypergraph DRL for Group-Level Influence (2024)

- **Source**: Information Sciences, 2024 ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025524016785)) | [Code](https://github.com/1873177187/HEDRL-IM)
- **Motivation**: Real viral marketing involves **group-level activation** — a WhatsApp group, a Slack channel, a Telegram broadcast, a family group chat. Standard graphs with pairwise edges cannot model these. Hypergraphs represent a group as a single hyperedge connecting all members.
- **Architecture**: Hypergraph Linear Threshold Model + DQN + Evolutionary Algorithm. Evolutionary search escapes local optima of DRL; evolutionary optimizes DQN weights while RL fine-tunes locally.
- **Result**: **0.3% to 30% improvement** over SOTA across 16 synthetic + 9 real-world hypergraphs.
- **Why it matters**: The dominant viral channels today (messaging apps, Discord servers, group chats) are group-based. Pairwise-edge models undersell the spread.

### 2.5 BIM-DRL — Balanced (Fair) Influence Maximization (2024)

- **Source**: Neural Networks, 2024 ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023005841))
- **Motivation**: Standard IM maximizes **total** influence, which can concentrate spread in one community while ignoring others. For brand-marketing or public-health campaigns, **balanced spread** across segments matters.
- **Reward function**: Penalizes uneven community-level spread.
- **Result**: **Superior balanced propagation accuracy** across 6 benchmarks, achieving fairness with only marginal reduction in total spread.
- **Marketing relevance**: Ensures campaigns reach under-represented segments — critical for DEI-conscious brands, regulated markets, and cross-cultural launches.

---

## 3. Real-Time Bidding (RTB) with Deep RL

Display advertising and sponsored search run billions of real-time auctions per day. For each auction, the advertiser's system must decide a bid price given limited information (user, context, remaining budget). This is a dynamic budget-constrained MDP.

### 3.1 RLB-DP — Real-Time Bidding by Reinforcement Learning (WSDM 2017)

- **Source**: WSDM 2017 ([paper](https://arxiv.org/abs/1701.02490)) | [Code](https://github.com/han-cai/rlb-dp)
- **Formulation**:
  - **State**: remaining budget, remaining auction opportunities, auction context
  - **Action**: bid price
  - **Reward**: campaign objective (clicks, conversions) on wins
  - **Transition**: auction result determines budget/opportunity update
- **Contribution**: First principled MDP formulation of RTB. Uses dynamic programming + neural value approximation to handle the scale of real auction volume.
- **Why it matters**: Linear bid-shading models (bid = CTR × eCPC) are myopic. MDP formulation optimizes **long-term budget deployment** across millions of auctions.

### 3.2 SS-RTB — Deep RL for Sponsored Search at Alibaba (KDD 2018)

- **Source**: KDD 2018 ([paper](https://arxiv.org/abs/1803.00259)) | [ACM](https://dl.acm.org/doi/10.1145/3219819.3219918)
- **Deployment**: Alibaba e-commerce sponsored search auctions
- **Key problem addressed**: "Environment changing" — state transition probabilities differ across days (weekday vs. weekend, promotion days, campaign phases). A policy trained on one day may fail on another.
- **Solution**: Robust deep RL that conditions on time-varying environment features and learns a policy that generalizes across regimes.
- **Result**: Significant lift in Alibaba's sponsored-search auction platform (exact numbers are proprietary but reported as substantial improvements over the production baseline).
- **Why it matters**: Proved that deep RL for bidding is production-ready at the scale of a major e-commerce platform.

### 3.3 Multi-Agent RL for RTB (CIKM 2018)

- **Source**: CIKM 2018 ([paper](https://dl.acm.org/doi/10.1145/3269206.3272021))
- **Formulation**: Multiple advertisers bid in the same auction, so the environment from any one advertiser's perspective is non-stationary (other bidders adapt). Multi-agent RL explicitly models this.
- **Contribution**: Cooperative / competitive multi-agent frameworks for auction bidding, capturing the game-theoretic dynamics of RTB markets.
- **Why it matters**: Single-agent RL assumes a stationary environment — unrealistic for auction markets. Multi-agent approaches are the principled frontier.

---

## 4. Contextual Bandits for Personalized Marketing

Contextual bandits are the minimal RL framework for marketing treatment assignment: **one-step** RL where the agent picks an action (offer, message, creative) conditional on user features, observes a reward (click, conversion), and updates. No long-horizon discounting; no credit assignment. But massive production impact.

### 4.1 LinUCB — Yahoo News Recommendation (2010)

- **Canonical paper**: Li, Chu, Langford, Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation" (WWW 2010)
- **Key idea**: Maintain a linear model per arm (article); select the arm with the highest **upper confidence bound** on expected reward given the context. Balances exploration vs. exploitation via UCB.
- **Yahoo deployment**: Reduced to 1.2k user features + 83 article features via PCA + user clustering (into 5 groups) for production latency.
- **Why it matters**: First at-scale production deployment of a contextual bandit in online recommendation — the reference point for all modern bandit deployments.

### 4.2 Thompson Sampling for Contextual Bandits

- **Key idea**: Sample the model parameters from their posterior distribution, then act greedily with respect to the sampled model. Probability of exploration naturally decreases as posterior tightens.
- **Yahoo deployment**: Regularized logistic regression with Gaussian posterior over weights, independent sampling per request. Served ad and news recommendations at Yahoo scale.
- **Empirical finding**: Thompson Sampling typically **matches or beats LinUCB** in real deployments, despite LinUCB's tighter theoretical regret bounds.

### 4.3 BanditLP — LinkedIn Email Marketing (2026)

- **Source**: arXiv ([paper](https://arxiv.org/abs/2601.15552))
- **What it does**: Combines contextual bandits with **constrained optimization** for large-scale email marketing. The first at-scale production deployment of this combination in email marketing.
- **Key twist**: Marketing emails are subject to cadence constraints (don't over-send to any user), budget constraints (total volume), and fairness constraints (don't starve any segment). BanditLP integrates these into the bandit optimization.
- **Why it matters**: Production bandits rarely operate unconstrained — BanditLP is the template for industrial deployments with real-world guardrails.

### 4.4 Scalable & Interpretable Contextual Bandits — Retail Prototype (2025)

- **Source**: arXiv ([paper](https://arxiv.org/html/2505.16918v1))
- **What it does**: Reviews the contextual bandit literature and presents a retail-offer prototype with emphasis on **interpretability** and scalability. Tree-based and linear reward models are compared to black-box deep models.
- **Why it matters**: Retail and CPG contexts often require explainability to merchants and brand managers. The review is a good starting point for practitioners new to the area.

### 4.5 Multi-Objective Contextual Bandits for Smart Tourism (2025)

- **Source**: Scientific Reports ([paper](https://www.nature.com/articles/s41598-025-89920-2))
- **What it does**: Extends contextual bandits to multi-objective reward (e.g., relevance, diversity, novelty, revenue) for recommendation in smart tourism.
- **Relevance**: Marketing rarely optimizes a single metric. Multi-objective bandits are the emerging frontier for recommendation and personalization.

---

## 5. Cross-Method Comparison

| Aspect | DGN / BiGDN / HEDRL-IM / BIM-DRL | RLB-DP / SS-RTB / MA-RTB | LinUCB / TS / BanditLP |
|--------|----------------------------------|--------------------------|-------------------------|
| **Problem** | Viral / influencer seed selection | Real-time auction bidding | Personalized treatment assignment |
| **RL formulation** | Sequential action (select k seeds) | MDP over budget/opportunities | Contextual bandit (one-step RL) |
| **State** | Graph + selected seeds | Budget + auction context | User features |
| **Action** | Next seed to pick | Bid price | Treatment (offer, creative) |
| **Reward** | Expected spread | Campaign objective on wins | Click / conversion |
| **Horizon** | k steps | Auction-budget length | 1 step |
| **Deployment scale** | Research (some industrial) | **Production** (Alibaba, others) | **Production** (Yahoo, LinkedIn) |

### When to Use Which

- **Start with contextual bandits** if you want fast, safe, online experimentation with low infrastructure cost.
- **Move to full RL (DQN / PPO)** if there is meaningful long-horizon credit assignment (e.g., lifetime value, subscription journeys, auction budget pacing).
- **Use GNN + DRL IM** when the targeting problem is explicitly network-based and you can simulate the cascade.

---

## 6. Open Problems and Future Directions

1. **Off-policy evaluation**: Online experiments are expensive. Offline evaluation of RL/bandit policies from logged data (doubly robust estimators, IPS, DR-IPS) is essential for iteration speed.

2. **Safety and exploration constraints**: Marketing RL agents can accidentally over-expose users (spam), violate frequency caps, or send off-brand messages during exploration. Constrained RL (CMDPs) and bandit approaches with safety layers are underdeveloped in production.

3. **Non-stationarity**: User tastes, competitor actions, and platform algorithms drift continuously. Stationary-environment RL theory breaks. Meta-RL, adaptive learning rates, and change-point detection are active directions.

4. **Counterfactual fairness**: DEI requires that treatment recommendations not discriminate. Fair RL / fair bandits is an open frontier.

5. **Multi-agent markets**: In RTB, every advertiser is learning simultaneously — the equilibrium can shift. Game-theoretic RL for ad markets is barely explored beyond simple cooperative/competitive setups.

6. **Causal RL**: RL optimizes observed reward, which may reflect selection bias. Integrating causal identification into RL training (e.g., doubly robust DQN) is the emerging frontier.

7. **LLM-integrated decision systems**: LLMs generate creatives, subject lines, and offer copy; RL/bandits decide when to deploy them. Joint optimization of content generation + targeting policy is an open research area.

---

## 7. References

### Viral / Influence Maximization
- Kempe, Kleinberg, Tardos (2003). Maximizing the Spread of Influence through a Social Network. KDD. [Paper](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf)
- Wang & Cao (2024). DGN: Dual Graph Neural Network for IM. J. Supercomputing. [Paper](https://link.springer.com/article/10.1007/s11227-024-06621-9)
- BiGDN (2025). Expert Systems with Applications. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000065) | [Code](https://github.com/zwl1985/BiGDN)
- HEDRL-IM (2024). Information Sciences. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025524016785) | [Code](https://github.com/1873177187/HEDRL-IM)
- BIM-DRL (2024). Neural Networks. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023005841)

### Real-Time Bidding
- Cai et al. (2017). Real-Time Bidding by Reinforcement Learning in Display Advertising. WSDM. [Paper](https://arxiv.org/abs/1701.02490) | [Code](https://github.com/han-cai/rlb-dp)
- Zhao et al. (2018). Deep Reinforcement Learning for Sponsored Search Real-time Bidding. KDD. [Paper](https://arxiv.org/abs/1803.00259)
- Jin et al. (2018). Real-Time Bidding with Multi-Agent Reinforcement Learning. CIKM. [Paper](https://dl.acm.org/doi/10.1145/3269206.3272021)

### Contextual Bandits
- Li, Chu, Langford, Schapire (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW. (Canonical LinUCB paper)
- BanditLP (2026). Large-Scale Stochastic Optimization for Personalized Recommendations. [Paper](https://arxiv.org/abs/2601.15552)
- Scalable and Interpretable Contextual Bandits: Literature Review and Retail Offer Prototype (2025). [Paper](https://arxiv.org/html/2505.16918v1)
- Multi-Objective Contextual Bandits for Smart Tourism (2025). Scientific Reports. [Paper](https://www.nature.com/articles/s41598-025-89920-2)
- Bandits for Recommender Systems — practitioner guide ([applyingml.com](https://applyingml.com/resources/bandits/))
