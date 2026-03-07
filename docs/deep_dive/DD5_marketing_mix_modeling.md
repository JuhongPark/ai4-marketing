# Deep Dive 5: Marketing Mix Modeling & Attribution with AI

How AI estimates the causal effect of marketing spend on business outcomes — from Bayesian MMM at the aggregate level to deep-learning causal attribution at the user-journey level.

---

## 1. Problem Definition

Two classical questions every CMO must answer:

1. **Marketing Mix Modeling (MMM)**: How much did each channel (TV, paid search, display, social, email, OOH, PR) contribute to revenue? What is the optimal future spend allocation?
2. **Multi-Touch Attribution (MTA)**: Across the long customer journey, which touchpoints deserve credit for a given conversion?

### The Privacy Shift

The industry has been undergoing a fundamental shift since 2020:

| Before | After |
|--------|-------|
| Cookie-based user tracking | iOS ATT, 3P cookie deprecation |
| User-level attribution dominant | Aggregate MMM dominant |
| Last-click attribution heuristics | Causal MMM, incrementality testing |
| Google / Meta attribution proprietary | Open-source MMM (Robyn, PyMC-Marketing, Meridian) |

Privacy regulation is **pushing the industry back toward MMM** as user-level signal degrades, forcing AI research to refocus on causal aggregate modeling.

---

## 2. Bayesian MMM — Theoretical Foundations

### 2.1 Jin et al. (2017) — The Canonical Google Paper

- **Paper**: "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects"
- **Source**: Google Research ([paper](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/))
- **Core contribution**: Formalizes MMM as a Bayesian regression model with two critical non-linearities:
  - **Carryover effect** (adstock): Advertising has **lag / delayed response** — spending $1 today affects sales over subsequent weeks. Modeled via a geometric or delayed decay function.
  - **Shape effect** (saturation): Advertising exhibits **diminishing returns** — the 100th TV GRP has less effect than the 1st. Modeled via a Hill function or similar concave saturation curve.
- **Why Bayesian?**: Priors encode domain knowledge (e.g., "TV adstock half-life is 2–4 weeks"), and posterior distributions provide **uncertainty quantification** for CMO-facing decisions.
- **Impact**: The single most influential MMM paper of the modern era. Every open-source MMM tool (Robyn, PyMC-Marketing, LightweightMMM, Meridian) traces back to this formulation.

### 2.2 Hierarchical Bayesian MMM (Google, 2017)

- **Source**: Google Research ([paper](https://research.google.com/pubs/archive/45999.pdf))
- **Contribution**: Introduces hierarchical structure to pool information across geographies, campaigns, or products — tightening posterior credible intervals dramatically.
- **Why it matters**: National-level MMM suffers from limited data (one observation per week); geo-level hierarchical models use dozens of regional observations per week.

### 2.3 New Framework for MMM (2023)

- **Source**: arXiv ([paper](https://arxiv.org/pdf/2311.05587))
- **Contribution**: Proposes an updated framework addressing MMM's historical weaknesses (collinearity between channels, identifiability under long-run effects, calibration to experimental incrementality tests).

---

## 3. Open-Source Bayesian MMM Tools

### 3.1 PyMC-Marketing — Python Bayesian Marketing Toolbox (PyMC Labs)

- **Source**: [GitHub](https://github.com/pymc-labs/pymc-marketing) | [Docs](https://www.pymc-marketing.io/)
- **What it includes**:
  - **MMM**: Adstock + saturation via PyMC, following Jin et al.
  - **Customer Lifetime Value (CLV)**: Pareto/NBD, BG/NBD, Gamma-Gamma models
  - **Buy-Till-You-Die (BTYD)**: Probabilistic customer-churn and purchase-frequency models
  - **Customer Choice Analysis (CSA)**: Discrete-choice modeling for product preference
- **Language**: Python, built on PyMC (the modern NumPyro-adjacent probabilistic programming stack).
- **Why it matters**: The most comprehensive open-source Python stack for Bayesian marketing analytics.

### 3.2 Robyn — Meta's Open-Source MMM (R)

- **Source**: [Robyn by Meta](https://facebookexperimental.github.io/Robyn/)
- **What it does**: Automated Bayesian MMM with:
  - Hyperparameter optimization via **Nevergrad** (Facebook's gradient-free optimization library)
  - **Prophet**-based trend/seasonality decomposition
  - **Ridge regression** regularization
  - Multi-objective optimization balancing model fit and business constraints
- **Distinguishing feature**: Explicitly designed to **reduce human bias** by automating variable selection and decay-parameter search.
- **Language**: R (there's no official Python equivalent from Meta).
- **Status**: Actively maintained, widely adopted across Meta's advertising partners.

### 3.3 LightweightMMM → Meridian (Google)

- **LightweightMMM**: Google's earlier open-source Bayesian MMM library, being **decommissioned** in favor of Meridian.
- **Meridian** (2024): Google's next-generation open-source MMM, now publicly available to everyone.

#### Meridian — Deep Dive

- **Source**: [About Meridian](https://developers.google.com/meridian/docs/basics/about-the-project) | [Announcement](https://blog.google/products/ads-commerce/meridian-marketing-mix-model-open-to-everyone/)
- **Core design principle**: **MMM is fundamentally a causal problem**, so Meridian is built on explicit causal-inference foundations plus hierarchical Bayesian estimation.
- **Key features**:
  - **ROI as a model parameter**: Marketers can place priors directly on channel ROI (derived from past experiments, MMM history, industry benchmarks). No manual conversion of prior knowledge to model parameters.
  - **Hierarchical geo-level model**: Uses geo-level marketing data for dramatically tighter credible intervals. Can also report regional effectiveness.
  - **Saturation via Hill function**, **adstock via geometric or binomial decay** — same Jin et al. building blocks, but refined.
  - **Search integration**: Uses Google Trends / search volume as a control variable for organic demand.
- **Why it matters**: Google's signal — that **aggregate, causal, open-source MMM is the future** — is a strategic statement about the post-cookie marketing measurement stack.

### 3.4 DeepCausalMMM — Deep Learning + Causal Inference for MMM (2025)

- **Source**: arXiv ([paper](https://arxiv.org/html/2510.13087))
- **What it does**: Deep-learning framework for MMM that integrates causal inference methods. Represents the **next step** in MMM evolution — bringing deep-learning expressiveness to what has been primarily a Bayesian / regression discipline.
- **Significance**: Signals that deep learning is starting to make principled inroads into MMM, where Bayesian methods have dominated.

---

## 4. AI for Multi-Touch Attribution

### 4.1 The MTA vs. MMM Distinction

| | MMM | MTA |
|---|---|---|
| **Data level** | Aggregate (daily/weekly spend + revenue) | User-level (individual journey) |
| **Privacy impact** | Minimal (aggregate) | Severe (requires tracking) |
| **Horizon** | Long-term (weeks/months) | Short-term (days/weeks) |
| **Channels** | All paid + organic | Digital, tracked |
| **Typical question** | How to allocate next year's budget across channels? | Which touchpoint drove this specific conversion? |

Both are active AI research areas.

### 4.2 DNAMTA — Deep Neural Net with Attention for MTA (2018)

- **Source**: Ren et al., arXiv ([paper](https://arxiv.org/abs/1809.02230))
- **Architecture**:
  ```
  Customer journey: [t_1, t_2, ..., t_n]
    ↓
  [LSTM over touchpoint sequence]
    ↓
  [Attention layer]  ← produces per-touchpoint importance scores
    ↓
  Per-touchpoint credit + conversion prediction
  ```
- **Key result**: First attention-based MTA. Captures **non-linear time dependencies** — a touchpoint that primes a later touchpoint gets credit for the combined effect.
- **Limitation**: Attention weights are **associational**, not causal. A touchpoint with high attention weight may simply co-occur with conversion rather than cause it.

### 4.3 CAMTA — Causal Attention Model for MTA (2020)

- **Source**: Kumar et al., arXiv ([paper](https://arxiv.org/abs/2012.11403))
- **What it does**: Combines attention-based MTA with **counterfactual causal reasoning** via the Counterfactual Recurrent Network (CRN) framework.
- **Key technique**: Uses domain-adversarial training to construct **treatment-invariant representations** at each time step. The representation does not leak information about which treatment (touchpoint sequence) was received, which is required for causal identification under unmeasured confounding.
- **Output**: Per-touchpoint **causal** effect on conversion — the effect that would disappear in a counterfactual world where the touchpoint did not occur.
- **Why it matters**: Closes the DNAMTA associational-to-causal gap, which is the primary scientific demand of attribution.

### 4.4 Amazon Ads Multi-Touch Attribution (2025)

- **Source**: arXiv ([paper](https://arxiv.org/html/2508.08209v1))
- **What it does**: Production MTA system at Amazon Ads combining causal inference with attention-based deep learning, designed for real-time auction feedback at the scale of Amazon Advertising.
- **Significance**: One of the few public papers describing an at-scale production MTA system, serving as a **reference architecture** for deep causal MTA at a major retailer / ad platform.

---

## 5. Integrated MMM + MTA + Incrementality

A mature marketing-measurement stack combines all three:

```
[Aggregate-level MMM (e.g., Meridian, PyMC-Marketing)]
    Long-term, cross-channel budget allocation
  ↓
[User-level MTA (e.g., DNAMTA / CAMTA)]
    Short-term within-channel optimization (where tracking exists)
  ↓
[Incrementality Tests (geo lift, PSA lift, holdout)]
    Ground-truth causal effect — calibrates MMM and MTA priors
```

### Why All Three?

- **MMM alone** is low-resolution (aggregate, slow to update) and historically plagued by identifiability issues.
- **MTA alone** is collapsing due to privacy changes, and its causal claims have been shaky.
- **Incrementality tests alone** are expensive and cannot be run on every channel × campaign combination.
- **Combined**, they calibrate each other: incrementality tests ground the priors, MTA handles granular in-channel optimization, MMM allocates across channels over time.

---

## 6. Cross-Method Comparison

| Aspect | Jin 2017 (original) | PyMC-Marketing | Robyn (Meta) | Meridian (Google) | DNAMTA | CAMTA | Amazon MTA |
|--------|----------------------|----------------|--------------|-------------------|--------|-------|------------|
| **Year** | 2017 | 2023+ | 2020+ | 2024 | 2018 | 2020 | 2025 |
| **Problem** | MMM | MMM + CLV + BTYD | MMM | MMM (causal-first) | MTA | Causal MTA | Production causal MTA |
| **Method** | Bayesian reg + carryover/shape | PyMC Bayesian | R ridge + Nevergrad | Hierarchical Bayesian | LSTM + attention | Adversarial + attention | Deep causal |
| **Causal framing** | Implicit | Implicit | Implicit | **Explicit** | No (associational) | **Yes** | **Yes** |
| **Hierarchical (geo)** | No | Optional | No | **Yes (native)** | N/A | N/A | N/A |
| **Open source** | Paper only | Yes (Python) | Yes (R) | **Yes (Python)** | Paper only | Paper only | Paper only |
| **Privacy-safe** | Yes (aggregate) | Yes | Yes | Yes | No (user-level) | No (user-level) | No (user-level) |

### Production Recommendation

- **For most teams today**: Start with **PyMC-Marketing** (Python stack) or **Meridian** (if you want Google's hierarchical geo approach) for MMM.
- **For Meta's ecosystem**: Robyn remains the de facto choice.
- **For attribution in tracked channels**: CAMTA-style causal MTA, with DNAMTA as the associational fallback when causal identification is infeasible.
- **For ground truth**: Run periodic incrementality / geo-lift experiments to calibrate MMM priors.

---

## 7. Open Problems and Future Directions

1. **Causal MMM identifiability**: MMM has historically relied on temporal variation for identification, which is weak. Hierarchical priors (Meridian) help; designed interventions (incrementality tests) help more; but fundamentally identifiable MMM remains a partially open problem.

2. **Deep learning + causal in MMM**: DeepCausalMMM is an early step. Integration of causal ML (double ML, causal forests, Bayesian double machine learning) with deep representations is the frontier.

3. **Long-run effects**: MMM typically captures short-to-medium-term effects (weeks to months). Brand-building effects play out over years. Modeling this long horizon is poorly served by current methods.

4. **Cross-platform attribution**: Users see ads on Google, Meta, Amazon, TikTok. Each platform's attribution is proprietary. Independent cross-platform MMM/MTA is an unsolved market-design problem as much as a research problem.

5. **Privacy-preserving user-level attribution**: Federated learning, differential privacy, and Apple's Private Click Measurement are early attempts. None matches the expressiveness of cookie-era MTA.

6. **Real-time MMM**: Classical MMM is trained monthly or quarterly. Real-time MMM that adapts weekly or daily is actively researched (see hierarchical online Bayesian MMM) but not widely deployed.

7. **Incrementality test automation**: Running geo lift / holdout experiments at scale across many channels and campaigns simultaneously requires new experimental-design methods. This is adjacent to the bandit / causal-RL frontier.

---

## 8. References

### Foundational MMM
- Jin et al. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Research. [Paper](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/)
- A Hierarchical Bayesian Approach to Improve Media Mix Models. Google Research. [Paper](https://research.google.com/pubs/archive/45999.pdf)
- A New Framework for Marketing Mix Modeling (2023). [Paper](https://arxiv.org/abs/2311.05587)
- Hierarchical Marketing Mix Models with Sign Constraints. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041956/)

### Open-Source MMM Tools
- PyMC-Marketing. [GitHub](https://github.com/pymc-labs/pymc-marketing) | [Docs](https://www.pymc-marketing.io/)
- Robyn (Meta). [Website](https://facebookexperimental.github.io/Robyn/)
- Meridian (Google). [About](https://developers.google.com/meridian/docs/basics/about-the-project) | [Causal Inference Intro](https://developers.google.com/meridian/docs/causal-inference/intro) | [Announcement](https://blog.google/products/ads-commerce/meridian-marketing-mix-model-open-to-everyone/)
- DeepCausalMMM (2025). [Paper](https://arxiv.org/html/2510.13087)

### Multi-Touch Attribution
- DNAMTA: Ren et al. (2018). Deep Neural Net with Attention for MTA. [Paper](https://arxiv.org/abs/1809.02230)
- CAMTA: Kumar et al. (2020). Causal Attention Model for MTA. [Paper](https://arxiv.org/abs/2012.11403)
- Amazon Ads MTA (2025). [Paper](https://arxiv.org/html/2508.08209v1)

### Practitioner Guides
- `awesome-marketing-machine-learning`. [GitHub](https://github.com/station-10/awesome-marketing-machine-learning)
- Juan Camilo Orduz on Media Effect Estimation with PyMC. [Blog](https://juanitorduz.github.io/pymc_mmm/)
