# Marketing Mix Modeling & Attribution

How AI models estimate marketing ROI, optimize spend across channels, and attribute conversions to touchpoints.

---

## Domain Background

Two classical questions every CMO must answer:

1. **Marketing Mix Modeling (MMM)**: How much did each channel (TV, paid search, display, social, email, OOH) contribute to revenue? What is the optimal future spend allocation?
2. **Multi-Touch Attribution (MTA)**: Across a long customer journey, which touchpoints deserve credit for a given conversion?

MMM operates on **aggregate** time-series data (privacy-safe, no user-level tracking required), while MTA operates on **user-level** journey data. Both are being transformed by AI, and the post-cookie / post-ATT privacy era is pushing the industry back toward aggregate MMM as user-level signal degrades.

---

## AI-based Approaches — Marketing Mix Modeling

### PyMC-Marketing — Open-source Bayesian MMM (PyMC Labs)

- **Source**: [GitHub](https://github.com/pymc-labs/pymc-marketing) | [Docs](https://www.pymc-marketing.io/)
- **What it does**: Python library implementing Bayesian MMM with **adstock** (carryover) and **saturation** (diminishing returns) effects, plus **Customer Lifetime Value (CLV)** and **Buy-Till-You-Die (BTYD)** models.
- **Underlying paper**: Jin et al. (2017) *Bayesian methods for media mix modeling with carryover and shape effects* — the canonical Google paper on Bayesian MMM.
- **Why it matters**: Bayesian priors let marketing teams inject domain knowledge (e.g., "TV adstock half-life is 2–4 weeks") and receive posterior distributions rather than brittle point estimates. Uncertainty quantification is essential for CMO-facing spend decisions.

### Robyn — Meta's Open-source MMM

- **Source**: [Robyn by Meta](https://facebookexperimental.github.io/Robyn/)
- **What it does**: Automated Bayesian MMM with hyperparameter optimization (Nevergrad), Prophet-based trend/seasonality decomposition, and ridge-regression regularization.
- **Distinguishing feature**: Explicitly designed to **reduce human bias** by automating variable selection and decay-parameter search rather than relying on analyst intuition.
- **Language**: R. PyMC-Marketing is the Python counterpart, covering the same conceptual space for Python-centric teams.

### LightweightMMM → Meridian (Google)

- **LightweightMMM**: Google's earlier open-source Bayesian MMM library, being decommissioned in favor of **Meridian**.
- **Meridian**: Google's next-generation Bayesian MMM, explicitly designed to handle the reality that privacy changes (iOS ATT, 3P cookie deprecation) are pushing the industry toward aggregate MMM.
- **Why it matters**: The largest ad platforms are publicly repositioning MMM as the primary spend-allocation tool as user-level signal degrades.

### Deep Dive Resource: `awesome-marketing-machine-learning`

- **Source**: [GitHub](https://github.com/station-10/awesome-marketing-machine-learning)
- **Content**: Curated list of libraries for MMM, MTA, causal inference, uplift modeling, and experimentation — the practical starting point for building a modern marketing data-science stack.

---

## AI-based Approaches — Multi-Touch Attribution

### DNAMTA — Deep Neural Net with Attention for MTA (2018)

- **Source**: arXiv ([paper](https://arxiv.org/abs/1809.02230))
- **Architecture**: LSTM over the customer-journey sequence + attention mechanism that produces per-touchpoint credit scores.
- **Significance**: First attention-based MTA model. Captures non-linear time dependencies that linear or heuristic attribution (first-touch, last-touch, time-decay) cannot.
- **Limitation**: Attention weights are **associational**, not causal — a touchpoint with high attention weight may simply co-occur with conversion rather than cause it.

### CAMTA — Causal Attention Model for Multi-Touch Attribution (2020)

- **Source**: arXiv ([paper](https://arxiv.org/abs/2012.11403))
- **What it does**: Combines attention-based MTA with **counterfactual causal reasoning**. Instead of predicting conversion probability, estimates the *causal effect* of each touchpoint on conversion.
- **Technique**: Adapts the counterfactual recurrent network (CRN) framework with domain-adversarial training to construct treatment-invariant representations at each time step.
- **Why it matters**: Closes the DNAMTA gap by distinguishing touchpoints that **caused** conversion from touchpoints that merely **co-occurred with** conversion — the core scientific demand of attribution.

### Amazon Ads Multi-Touch Attribution (2025)

- **Source**: arXiv ([paper](https://arxiv.org/html/2508.08209v1))
- **What it does**: Production MTA system at Amazon Ads combining causal inference with attention-based deep learning, designed for real-time auction feedback.
- **Significance**: One of the few public papers describing an at-scale production MTA system, serving as a reference architecture for deep causal MTA.

### Integrated MTA + MMM for E-commerce ROI Optimization

- **Source**: ResearchGate ([paper](https://www.researchgate.net/publication/399508893_Multi-Touch_Attribution_and_Media_Mix_Modeling_for_Marketing_ROI_Optimization_in_E-Commerce_Platforms))
- **What it does**: Combines MTA with MMM, using the user-level journey signal where available and the aggregate MMM where it is not, into a unified ROI framework.
- **Why it matters**: In a post-cookie world, neither pure MTA nor pure MMM is sufficient on its own. Integrated approaches are the emerging best practice.

---

## Why AI?

MMM and MTA are **causal, high-dimensional, and sequential**: we want to know how a change in spend *causes* a change in revenue over long customer journeys with complex carryover effects. Bayesian inference provides uncertainty quantification for CMO-facing decisions; deep attention models capture long-range sequential dependencies classical regressions miss; and causal ML closes the associational-to-causal gap in attribution. Privacy regulation is pushing the field toward aggregate MMM — which is where the open-source Bayesian stack (PyMC-Marketing, Robyn, Meridian) now sits.
