# Influencer Marketing & Viral Spread

How AI models identify influencers, predict which posts will go viral, and optimize word-of-mouth marketing.

---

## Domain Background

Word-of-mouth is consistently rated the **highest-trust marketing channel** across industries. Two formal problems dominate:

1. **Influence prediction** — Given a user, will they adopt or share this content given their neighborhood's behavior?
2. **Influence maximization (IM)** — Given a budget of `k` seed users, which users should receive the campaign to maximize total eventual adoption?

Both are NP-hard in the general case (Kempe, Kleinberg & Tardos, KDD 2003), motivating AI-based approximation.

---

## AI-based Approaches

### DeepInf — Social Influence Prediction with Deep Learning (KDD 2018)

- **Source**: KDD 2018 ([paper](https://arxiv.org/abs/1807.05560)) | [Code](https://github.com/xptree/DeepInf)
- **What it does**: End-to-end GNN (GCN or GAT) that takes a user's local ego-network and predicts whether the user will perform a specific action (e.g., share, purchase, sign up).
- **Result**: Significantly outperforms feature-engineered baselines on Weibo, Twitter, OAG, and Digg.
- **Marketing use case**: Predicting which customers in a social network will convert after seeing a campaign via an influencer neighbor — the practical core of influencer-marketing ROI estimation.

### Deep RL + GNN for Influence Maximization (Viral Marketing Seed Selection)

Viral marketing reduces to seed selection under a probabilistic cascade model. Recent deep-RL approaches treat this as sequential decision-making and dramatically outperform classical greedy.

| Model | Year | Venue | Paper |
|-------|------|-------|-------|
| **DGN** (Dual GNN + DQN) | 2024 | J. Supercomputing | [link](https://link.springer.com/article/10.1007/s11227-024-06621-9) |
| **BiGDN** (Bidirectional GNN + DRL) | 2025 | Expert Systems with Applications | [link](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000065) |
| **HEDRL-IM** (Evolutionary DRL on hypergraphs) | 2024 | Information Sciences | [link](https://www.sciencedirect.com/science/article/abs/pii/S0020025524016785) |
| **BIM-DRL** (Balanced/fair IM via DRL) | 2024 | Neural Networks | [link](https://www.sciencedirect.com/science/article/abs/pii/S0893608023005841) |

All four learn graph embeddings once and then select seeds in a forward pass, enabling **real-time viral-campaign targeting** at the scale of Facebook, TikTok, or Instagram.

### Aral et al. — Large-Scale Empirical Foundations (MIT Sloan, 2011–2017)

Sinan Aral's group at MIT Sloan produced the definitive randomized experiments on social contagion that AI models now aim to reproduce:

- **Creating Social Contagion Through Viral Product Design** (Management Science, 2011): 1.4M-user randomized experiment at a messaging app. **Passive broadcasts caused a 246% increase in peer adoption**; active-personalized messaging added another 98%. [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1564856).
- **Identifying Influential and Susceptible Members of Social Networks** (Science, 2012): 1.3M Facebook-user randomized experiment. **Younger users** are more susceptible; **men** are more influential; **married** users are least susceptible. [Paper](https://www.science.org/cms/asset/eff8876a-1565-45d1-b8d1-11ab8680834e/pap.pdf).
- **Exercise Contagion in a Global Social Network** (Nature Communications, 2017): Empirical proof that exercise behavior propagates through social networks at scale. [Paper](https://www.nature.com/articles/ncomms14753).

These experiments are the **empirical ground truth** that AI influence-prediction models must match.

### The Spread of True and False News Online (Vosoughi, Roy & Aral, Science 2018)

- **Source**: Science ([paper](https://www.science.org/doi/10.1126/science.aap9559))
- **Finding**: Across 126,000 stories and 3M users on Twitter, **false news spreads significantly farther, faster, deeper, and more broadly** than true news. Humans, not bots, are the primary accelerant.
- **Marketing implication**: Novelty and emotional arousal drive virality. Content-generation and campaign strategy should optimize for these — while respecting trust and compliance boundaries.

---

## Why AI?

Classical IM requires Monte Carlo simulation of cascades for every seed candidate — impossibly slow on modern social graphs. GNN + deep-RL methods encode the graph once, then select seeds in a forward pass. Combined with the empirical validation from randomized contagion experiments (Aral et al.), AI influence models now close the loop between **who** to target, **what** to send, and **how much** lift to expect.
