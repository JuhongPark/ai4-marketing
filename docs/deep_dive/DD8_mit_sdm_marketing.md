# Deep Dive 8: MIT SDM Research Related to AI Marketing

A survey of research from MIT's System Design and Management (SDM) program and SDM-adjacent programs relevant to AI marketing, customer analytics, personalization, and marketing-stack system design.

---

## 1. About MIT SDM

The **System Design and Management (SDM)** program is a joint offering of the MIT School of Engineering and MIT Sloan School of Management, leading to a Master of Science in Engineering and Management. SDM is not a traditional research lab but an interdisciplinary professional master's program focused on:

- **Systems thinking** and system architecture
- **Complex system design** and management
- **Socio-technical systems** modeling — including agent-based simulation
- **Engineering + management integration** for product and technology strategy

SDM does not operate as a single research group with a dedicated AI-marketing focus. Instead, its contributions come through student theses supervised by faculty across MIT, and through its teaching of **system architecture and systems thinking methodologies** — which have direct bearing on how a modern AI marketing stack is designed, deployed, and governed.

### Key SDM Faculty

| Faculty | Role | Relevance to AI Marketing |
|---------|------|---------------------------|
| **Bryan Moser** | Academic Director, Senior Lecturer | Agent-based modeling of socio-technical systems, teamwork dynamics, complex systems |
| **Bruce Cameron** | Director, System Architecture Group | System architecture, platform strategy, technology strategy |
| **Ed Crawley** | Ford Professor of Engineering | Architecture, design, and optimization of complex technical systems |
| **Michael Siegel** | Senior Lecturer | Information systems management, cybersecurity, data governance |

None of these faculty specialize in marketing per se, but each provides frameworks that apply directly to the design of production AI marketing systems.

---

## 2. Directly Related SDM / Sloan Management-Technology Theses

### 2.1 Improving Complex Sale Cycles Using ML and Predictive Analytics

- **Program**: MIT Sloan S.M. in Management of Technology (MoT) — closely related to SDM
- **Link**: [DSpace](https://dspace.mit.edu/handle/1721.1/118010)
- **Topic**: Analyzes how to shorten complex B2B customer journeys and sales cycles using machine learning and predictive analytics.
- **Scope**: Examines the benefits and challenges of applied ML / predictive analytics to improve critical stages in the sales and marketing funnel for B2B firms.

#### Connection to This Project

Maps directly onto **DD4** (churn / customer lifecycle) and **DD5** (attribution) — specifically, B2B journeys have long horizons and multiple touchpoints, making them the canonical test bed for the MTA and offline RL approaches in those chapters.

### 2.2 Improving e-Commerce Sales Using Machine Learning

- **Program**: MIT Sloan / SDM-adjacent
- **Link**: [DSpace](https://dspace.mit.edu/handle/1721.1/118511)
- **Topic**: Applies ML to the e-commerce digital sales and marketing ecosystem, covering:
  - **Search** (query understanding and ranking)
  - **Recommendation system** (product suggestion at the browse / cart stage)
  - **Product detail page** optimization
  - **Advertising ecosystem** integration
- **Scope**: Practitioner-oriented analysis of how ML improves conversion across the e-commerce funnel.

#### Connection to This Project

Maps onto **DD1** (GNN recommenders) and **DD5** (attribution) — this thesis is the "applied systems view" that complements DD1's methodological deep dive into recommendation algorithms.

### 2.3 Strategic Perspective on the Commercialization of Artificial Intelligence

- **Program**: MIT SDM
- **Source**: [SDM Research & Practice](https://sdm.mit.edu/research-practice/thesis-a-strategic-perspective-on-the-commercialization-of-artificial-intelligence/)
- **Topic**: Connects business / technology strategy literature with the evolution and adoption of AI systems, analyzing AI from the perspective of emergent system properties and commercialization pathways.
- **Why it matters for AI marketing**: The thesis treats AI as a system-level phenomenon requiring careful product, market, and organizational design — exactly the perspective needed for teams deploying Bayesian MMM, uplift modeling, or bandit systems in production.

### 2.4 Consumer Credit Risk Analysis via ML

- **Program**: MIT SDM / Sloan MoT
- **Link**: [DSpace](https://dspace.mit.edu/handle/1721.1/100614)
- **Topic**: Applications of machine learning to consumer credit risk analysis.
- **Relevance**: Credit-risk modeling shares methodological structure with churn prediction (sequential, tabular, regulated, explainability-demanding). Insights transfer to the retention-modeling domain in DD4.

---

## 3. SDM-Adjacent Programs with Marketing AI Research

SDM shares faculty, coursework, and research infrastructure with several MIT programs. The following theses / programs are not from SDM itself but come from programs that closely interact with SDM.

### 3.1 MIT Sloan Marketing Group Theses (PhD)

PhD research in the MIT Sloan Marketing Group, supervised by faculty covered in DD7 (Tucker, Eckles, Aral, Simester), covers:
- **Causal inference for marketing interventions**
- **Field experiments at scale** (ad platforms, retailers, e-commerce)
- **Privacy-preserving personalization**
- **Uplift and treatment-effect modeling**

These theses are housed at MIT Sloan rather than SDM but represent the deepest MIT research output directly addressing AI marketing.

### 3.2 MIT Operations Research Center (ORC) Theses

ORC PhD research under **Perakis** and **Bertsimas** covers:
- **Pricing and promotion optimization under demand uncertainty** (matches DD5)
- **Assortment optimization for retail** (matches DD1 recommendation / DD5 MMM)
- **Customer choice modeling** (matches DD segmentation)
- **Interpretable ML for targeting** (Optimal Classification Trees, matches DD4 uplift)

**Representative PhD dissertations**: Multiple ORC PhDs each year produce retail analytics and pricing-optimization theses with direct marketing applications.

### 3.3 MIT Initiative on the Digital Economy (IDE) Research Briefs

IDE does not confer degrees but produces research briefs and working papers directly on AI marketing topics:
- **Agentic AI in marketing** — recent MIT IDE research on how agentic AI systems transform customer service, personalization, and campaign execution. See [MIT Sloan coverage](https://mitsloan.mit.edu/ideas-made-to-matter/4-new-studies-about-agentic-ai-mit-initiative-digital-economy).
- **Scaling AI for results** — strategic analyses of how firms successfully deploy ML at enterprise scale. See [MIT Sloan Management Review coverage](https://mitsloan.mit.edu/ideas-made-to-matter/scaling-ai-results-strategies-mit-sloan-management-review).

---

## 4. SDM's Methodological Contributions to AI Marketing

While SDM has limited direct research on AI marketing topics, its **methodological toolkit** is directly applicable to the design of production AI marketing systems:

### 4.1 System Architecture Analysis (Crawley, Cameron)

SDM's core curriculum teaches system architecture methods that apply directly to AI marketing-stack design:

- **Decomposition**: Breaking the marketing system into interacting components (data pipeline → feature store → models → decision engine → experimentation → reporting). Directly maps to the architecture of a modern MLOps stack.
- **Interface analysis**: Understanding how components interact. Applies to offline / online model training, candidate generation + reranking, and how causal-ML outputs flow into decision systems.
- **Architecture evaluation**: Assessing robustness of candidate architectures. Relevant to evaluating build-vs-buy decisions (e.g., own Meridian deployment vs. vendor MMM).

**Reference**: Crawley, Cameron, Selva. *Systems Architecture: Strategy and Product Development for Complex Systems*. Pearson, 2015.

### 4.2 Agent-Based Modeling of Socio-Technical Systems (Moser)

Bryan Moser teaches agent-based modeling in SDM, which connects to:

- **Customer journey modeling**: Customers as agents with state, preferences, and decision rules; journeys emerge from agent–channel interactions.
- **Multi-agent ad auctions**: Advertisers as agents in a marketplace; bid strategies as policies; RTB equilibrium as emergent phenomenon (covered in DD2 §3).
- **Multi-touch attribution**: Customer agents and touchpoint events in a discrete-event simulation, enabling counterfactual attribution analysis.

**Reference**: Moser et al. *Agent-Based Modelling of Socio-Technical Systems*. Springer, 2012. [Link](https://link.springer.com/book/10.1007/978-94-007-4933-7)

### 4.3 Systems Thinking for Marketing Analytics Governance

SDM's systems thinking curriculum provides frameworks for understanding how decisions propagate through complex organizational systems:

- **Feedback loops and unintended consequences**: Over-targeting feeds ad fatigue, which depresses engagement, which triggers the model to target more — a classic positive-feedback trap.
- **Dynamic interactions**: Short-term optimization vs. long-term brand equity — the tension between bandit exploitation and brand-building that needs system-level treatment.
- **Holistic assessment**: MMM vs. MTA vs. incrementality are complementary tools in a holistic measurement system, not competitors.

These lenses are essential for CMOs, marketing-ops leads, and MLOps engineers deploying AI marketing systems in real organizations where the model output is only one component of the decision.

---

## 5. Summary Assessment

### What SDM Covers (Directly or Via Adjacent Programs)

| Strength | Description | Relevance |
|----------|-------------|-----------|
| **System architecture for AI marketing stacks** | Crawley-Cameron framework for decomposing and analyzing ML system components | High — cross-cutting for all DDs |
| **Agent-based modeling of customer journeys and marketplaces** | Moser's ABM methodology applies to customer journey modeling and RTB simulation | Medium-High — DD2, DD5 |
| **Commercialization strategy for AI products** | SDM thesis on AI commercialization | Medium — strategic context |
| **ML for customer journey (B2B)** | Sloan MoT thesis on predictive analytics for sales cycles | High — DD4, DD5 |
| **ML for e-commerce search / recommendation / ads** | Sloan MoT thesis on ML in e-commerce | High — DD1, DD5 |
| **ORC pricing / promotion optimization (adjacent)** | Perakis group, interpretable ML | High — DD4, DD5 |
| **Sloan Marketing PhD research (adjacent)** | Tucker, Eckles, Aral, Simester theses | **Very High** — DD3, DD4, DD5 (see DD7 for detail) |

### What SDM Does Not Cover Directly

SDM does not have deep in-house research on:
- **Graph Neural Networks** for marketing → covered by CSAIL (Barzilay, Jaakkola) + Sloan (PhD theses)
- **Deep RL for auctions or viral marketing** → covered by external research (DD2)
- **Large-scale social contagion experiments** → covered by Sloan / IDE (Aral, Eckles)
- **Bayesian MMM methodology** → covered by Google (Meridian), PyMC Labs, Meta (Robyn)
- **Core NLP / vision-language foundation models** → covered by CSAIL

### SDM's Unique Niche for AI Marketing

SDM's contribution is at the **system architecture and design level** — not in developing new AI marketing algorithms, but in providing frameworks for:

1. **Architecting resilient AI marketing systems** — stacks that are robust to data drift, privacy regulation, and organizational change.
2. **Modeling socio-technical marketing interactions** — where customers, creatives, channels, algorithms, and operators co-create outcomes.
3. **Bridging engineering and management perspectives** on how AI marketing systems integrate with organizational decision-making.

A high-performing AI marketing deployment needs both:
- The **technical methods** of DD1–DD6 (GNNs, deep RL, causal ML, Bayesian MMM)
- The **system design discipline** of SDM (decomposition, interface clarity, feedback-loop analysis, organizational fit)

The latter is harder to teach and rarely highlighted in ML papers, but it is often the difference between a model that works in a notebook and a system that drives production revenue.

---

## 6. References

### MIT SDM / Sloan Management-Technology Theses
- Improving Complex Sale Cycles Using ML and Predictive Analytics. MIT Sloan S.M. MoT, 2018. [DSpace](https://dspace.mit.edu/handle/1721.1/118010)
- Improving e-Commerce Sales Using Machine Learning. MIT Sloan. [DSpace](https://dspace.mit.edu/handle/1721.1/118511)
- A Strategic Perspective on the Commercialization of Artificial Intelligence. MIT SDM. [SDM Research](https://sdm.mit.edu/research-practice/thesis-a-strategic-perspective-on-the-commercialization-of-artificial-intelligence/)
- Applications of Machine Learning: Consumer Credit Risk Analysis. MIT. [DSpace](https://dspace.mit.edu/handle/1721.1/100614)
- Data Science and Advanced Analytics: An Integrated Framework. [DSpace](https://dspace.mit.edu/handle/1721.1/120232)

### SDM Methodological References
- Crawley, Cameron, Selva. *Systems Architecture: Strategy and Product Development for Complex Systems*. Pearson, 2015.
- Moser et al. *Agent-Based Modelling of Socio-Technical Systems*. Springer, 2012. [Link](https://link.springer.com/book/10.1007/978-94-007-4933-7)

### MIT IDE / Sloan Coverage of AI Marketing
- Scaling AI for Results: Strategies from MIT Sloan Management Review. [Article](https://mitsloan.mit.edu/ideas-made-to-matter/scaling-ai-results-strategies-mit-sloan-management-review)
- 4 New Studies About Agentic AI from MIT IDE. [Article](https://mitsloan.mit.edu/ideas-made-to-matter/4-new-studies-about-agentic-ai-mit-initiative-digital-economy)

### SDM Program Resources
- [MIT SDM Research & Practice Archive](https://sdm.mit.edu/research-practice/)
- [MIT SDM Thesis Archives (DSpace)](https://sdm.mit.edu/practice-category/research-output/student-thesis-dspace/)
- [MIT System Architecture Group](http://systemarchitect.mit.edu/people.php)

### Cross-References to Other Deep Dives
- DD1 — GNN for recommender systems (the e-commerce thesis §2.2 is an applied complement)
- DD4 — Churn prediction & uplift (the B2B thesis §2.1 and credit-risk thesis §2.4 are methodological complements)
- DD5 — MMM and attribution (ORC pricing research §3.2 is the optimization complement to Bayesian MMM)
- DD7 — MIT research landscape on AI marketing (overall MIT view, where Sloan PhD research dominates)
