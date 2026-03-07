# Customer Churn Prediction

How AI models predict which customers will leave and enable proactive retention.

---

## Domain Background

Churn is one of the most financially consequential marketing problems:

| Industry | Typical annual churn rate |
|----------|---------------------------|
| Telecom | **>30%** |
| SaaS (B2C subscription) | 5–15% monthly |
| Streaming media | 3–6% monthly |
| Retail (lapsed customers) | 20–40% |

Acquiring a new customer typically costs **5–25× more** than retaining an existing one, making reliable churn prediction among the highest-leverage marketing AI applications. Churn is also causal and sequential — the signals that predict churn (declining engagement, support tickets, peer churn) unfold over weeks, not a single snapshot.

---

## AI-based Approaches

### CCP-Net — Hybrid BiLSTM + CNN + Multi-Head Self-Attention (2024)

- **Source**: Scientific Reports ([paper](https://www.nature.com/articles/s41598-024-79603-9))
- **Architecture**: Hybrid neural network that fuses three complementary mechanisms:
  - **Multi-Head Self-Attention** for global dependencies across the customer timeline
  - **BiLSTM** for long-term sequential patterns
  - **CNN** for local feature extraction
- **Results**: **92.19% precision** on the Telecom benchmark — a **1–3% improvement** over prior hybrid baselines.
- **Why it matters**: Demonstrates that combining sequence, local, and global attention mechanisms outperforms any single architecture for churn on signal-rich telecom data.

### Comprehensive ML/DL Evaluation for Churn Prediction (2024)

- **Source**: MDPI Information ([paper](https://www.mdpi.com/2078-2489/16/7/537))
- **Scope**: Benchmarks classical ML (Logistic Regression, Random Forest, XGBoost) vs. deep learning (MLP, CNN, LSTM, hybrid models) across public churn datasets.
- **Finding**: No single winner — **tabular tree-based models still match or beat deep learning** on small-to-medium datasets; DL dominates only when data is large and contains sequential, textual, or multi-modal signals.
- **Practical lesson**: Evaluate RF/XGBoost first; adopt deep models only when the data shape justifies them.

### Systematic Review: 2020–2024 Trends in Churn Prediction

- **Source**: MDPI Machine Learning and Knowledge Extraction ([paper](https://www.mdpi.com/2504-4990/7/3/105))
- **Scope**: Reviews peer-reviewed churn prediction research across telecom, retail, banking, SaaS, healthcare, education, and insurance.
- **Key trend**: Explicit move from classical models toward **hybrid deep learning** plus **explainable AI (SHAP, LIME, attention visualization)**, driven by deployment requirements for interpretability and regulatory compliance.

### Prediction of Customer Churn in Telecom — ML Benchmarking (2024)

- **Source**: MDPI Algorithms ([paper](https://www.mdpi.com/1999-4893/17/6/231))
- **What it does**: Benchmarks Random Forest, SVM, Gradient Boosting, and deep learning for telecom churn.
- **Result**: **Random Forest** achieved **91.66% accuracy, 82.2% precision, 81.8% recall** — remaining competitive with more complex deep models on standard telecom features.

### Hybrid RFM + K-means + Deep Neural Network for Retail Churn (2025)

- **Source**: Expert Systems with Applications ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425020846))
- **What it does**: Combines RFM-based segmentation, K-means clustering, and a deep neural network for online-retail churn prediction.
- **Result**: **Cluster-conditioned DNN** outperforms monolithic DNN by leveraging segment-specific churn patterns — the same model trained per segment learns different decision boundaries than one universal classifier.

### Composite Deep Learning for Churn Prediction (2023)

- **Source**: Scientific Reports ([paper](https://www.nature.com/articles/s41598-023-44396-w))
- **What it does**: Composite deep model combining CNN for feature extraction with additional dense layers for classification.
- **Finding**: Confirms that feature-extraction depth matters more than classification depth for tabular churn data.

---

## Why AI?

Churn signals are subtle, multi-modal (usage, support tickets, demographics, peer effects), and time-dependent. Classical logistic regression misses non-linear interactions and sequential patterns. Modern hybrid DL captures all three simultaneously, but tree ensembles remain competitive on tabular-only data — the right choice depends on the data shape, not a universal winner. The hardest open problem remains **actionable churn**: not just predicting departure, but identifying the intervention (discount, feature unlock, human outreach) with the highest causal uplift — a problem for uplift modeling and offline RL.
