# Customer Segmentation

How AI models discover meaningful customer groups from transaction, behavior, and demographic data to enable targeted marketing.

---

## Domain Background

### RFM (Recency, Frequency, Monetary) Framework

- **Origin**: Direct marketing industry, 1990s
- **Core idea**: Customers differ most along three axes — how recently they bought, how often, and how much.
- **Advantages**: Simple, interpretable, widely adopted.
- **Limitations**: Captures transactional behavior only; misses content preferences, temporal dynamics, and latent customer intent.

### From RFM to Deep Learning

Modern customer data (clickstream, browsing history, social activity, in-app behavior) is too high-dimensional for classical RFM. Deep learning can absorb this complexity into **customer embeddings** — dense vectors encoding behavior, preferences, and context for downstream segmentation and targeting.

---

## AI-based Approaches

### RFM-Net — CNN for Customer Segment Classification

- **Source**: MDPI Applied Sciences ([paper](https://www.mdpi.com/2076-3417/16/5/2223))
- **What it does**: Treats RFM-derived features as structured input to a CNN for customer segment classification, combining the interpretability of RFM with the pattern recognition of deep learning.
- **Significance**: Bridges classical marketing analytics with modern DL architectures without discarding decades of RFM know-how.

### Dynamic Segmentation via LRFMS + Multivariate Time Series Clustering (2024)

- **Source**: Scientific Reports ([paper](https://www.nature.com/articles/s41598-024-68621-2))
- **What it does**: Extends RFM to **LRFMS** (Length, Recency, Frequency, Monetary, Satisfaction) and applies multivariate time series clustering so that segments **evolve over time** rather than being frozen snapshots.
- **Results**: Captures churn trajectories and lifecycle transitions that static RFM misses.

### Deep Learning + Explainable AI + RFM for Targeted Marketing (2023)

- **Source**: MDPI Mathematics ([paper](https://www.mdpi.com/2227-7390/11/18/3930))
- **What it does**: Combines a deep neural network for customer representation with explainable AI (XAI) methods so that segmentation decisions remain interpretable to marketing stakeholders.
- **Why it matters**: Marketing deployment requires explainability. Pure DL segment boundaries are often opaque to CMOs and compliance reviewers.

### Customer Segmentation Review for E-commerce Personalized Targeting (2023)

- **Source**: Information Systems and e-Business Management ([paper](https://link.springer.com/article/10.1007/s10257-023-00640-4))
- **Scope**: Comprehensive review of segmentation methods for personalized customer targeting, comparing classical clustering, deep learning, and hybrid approaches.
- **Key finding**: Method choice depends heavily on catalog size, feature richness, and the downstream marketing action — no universal winner.

### Segmenting Bank Customers via RFM + Unsupervised ML (2020)

- **Source**: arXiv ([paper](https://arxiv.org/abs/2008.08662))
- **What it does**: Benchmarks unsupervised ML methods (K-means, hierarchical, DBSCAN, Gaussian mixtures) against classical RFM for retail banking customers.
- **Finding**: Modern unsupervised ML consistently recovers more actionable segments than pure RFM once the customer base exceeds ~100k.

---

## Why AI?

Marketing-grade segmentation needs three things classical clustering cannot fully deliver: (i) **high-dimensional** feature absorption, (ii) **temporal dynamics** (segments drift as behavior evolves), and (iii) **explainability** for CMO and compliance stakeholders. Deep + XAI approaches start to deliver on all three simultaneously.
