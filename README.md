# ai4-marketing

Research survey on **using AI for marketing** — how machine learning (graph neural networks, causal ML, Bayesian inference, deep RL, contextual bandits) transforms customer understanding, acquisition, retention, and marketing spend optimization.

## Motivation

Modern marketing runs at a scale and speed that classical analytics cannot handle. Millions of customer touchpoints, billions of items to recommend, thousands of campaign variants, privacy-induced signal loss, and short feedback cycles all demand models that learn from massive data and adapt in real time. **AI approaches (GNN, deep RL, Bayesian inference, causal ML)** are driving the current generation of breakthroughs in personalization, attribution, retention, and spend optimization.

## Project Structure

```
docs/
├── 01_customer_segmentation.md    # Deep learning for customer segmentation, beyond RFM
├── 02_recommendation_systems.md   # GNN-based recommenders (PinSage, LightGCN)
├── 03_influencer_marketing.md     # Influence prediction and viral marketing via GNN + DRL
├── 04_churn_prediction.md         # ML and deep learning for customer retention
└── 05_marketing_mix_modeling.md   # Bayesian MMM and multi-touch attribution
```

## Key AI Approaches by Domain

| Domain | AI Method | Key Result | Paper |
|--------|-----------|------------|-------|
| **Recommendation** | Web-scale GCN (PinSage) | Deployed at Pinterest on **3B nodes / 18B edges** | [Ying et al., KDD 2018](https://arxiv.org/abs/1806.01973) |
| **Recommendation** | LightGCN | Removes non-linearity → simpler and **more accurate** CF | [He et al., SIGIR 2020](https://arxiv.org/abs/2002.02126) |
| **Segmentation** | RFM-Net (CNN + RFM) | CNN-based customer segment classification | [RFM-Net, MDPI](https://www.mdpi.com/2076-3417/16/5/2223) |
| **Segmentation** | DL + XAI + RFM | Explainable deep segmentation for targeting | [MDPI Mathematics 2023](https://www.mdpi.com/2227-7390/11/18/3930) |
| **Churn** | CCP-Net (BiLSTM + CNN + MHSA) | **92.19% precision** on telecom churn | [Sci. Rep. 2024](https://www.nature.com/articles/s41598-024-79603-9) |
| **Attribution** | DNAMTA (LSTM + Attention) | First attention-based multi-touch attribution | [Ren et al., 2018](https://arxiv.org/abs/1809.02230) |
| **Attribution** | CAMTA (Causal Attention) | Counterfactual reasoning over customer journeys | [Kumar et al., 2020](https://arxiv.org/abs/2012.11403) |
| **MMM** | Bayesian PyMC-Marketing | Open-source Python MMM + CLV + BTYD | [PyMC Labs](https://github.com/pymc-labs/pymc-marketing) |
| **MMM** | Robyn (Meta) | Automated Bayesian MMM with Nevergrad search | [Robyn, Meta](https://facebookexperimental.github.io/Robyn/) |
| **Influence / Viral** | Deep RL + GNN for IM | Viral marketing as NP-hard seed selection | [DGN 2024](https://link.springer.com/article/10.1007/s11227-024-06621-9) |

## Why AI?

Core marketing problems are:
- **Combinatorial** (which items to recommend, which message, which seeds to pick) → **GNN + Deep RL**
- **Causal** (what drives conversion vs. what merely co-occurs with it) → **Causal ML, uplift modeling, Bayesian inference**
- **High-dimensional** (millions of users × thousands of features × long journeys) → **Deep embeddings, attention**
- **Non-stationary** (tastes, trends, competitors drift) → **Contextual bandits, online learning**
- **Accountability-demanding** (CMO needs explainability and uncertainty) → **Bayesian MMM, SHAP, XAI**

## Search Keywords

| Concept | Keywords |
|---------|----------|
| Recommenders | `graph neural network recommender`, `LightGCN`, `PinSage`, `NGCF` |
| Segmentation | `deep learning customer segmentation`, `RFM deep learning`, `customer embeddings` |
| Churn | `customer churn deep learning`, `churn prediction neural network`, `BiLSTM churn` |
| Attribution | `multi-touch attribution deep learning`, `causal attribution`, `DNAMTA`, `CAMTA` |
| MMM | `Bayesian marketing mix modeling`, `PyMC-Marketing`, `Robyn MMM`, `media mix modeling` |
| Influence / Viral | `influence maximization deep reinforcement learning`, `viral marketing GNN` |
