# System Design and Management — AI Systems Through the SDM Lens

The third skill axis. Where **Sales** asks *"what is AI used for, and does it move revenue or capital?"* and **Productization** asks *"how do we build the shippable AI product?"*, this document asks the MIT SDM question:

> **"How do we design, operate, evolve, and retire AI systems as coherent engineered socio-technical systems — with defensible architecture, explicit lifecycle, analyzable dynamics, and accountable governance?"**

This is the lens an SDM student brings to any AI stack. It is not a restatement of ML engineering or cloud architecture. It is the discipline of treating an AI system as a *system* — with decomposition, interfaces, tradespace, emergent properties, feedback loops, technical debt, and a deliberate lifecycle.

---

## 1. The SDM Spine for AI Systems

Every subsection below slots into this spine:

```
[Context]
  Organization + customers + regulators + competitors + data + incentives
        ↓
[Architecture]
  Decomposition → Form vs. Function → Interfaces → Tradespace → Evolution
        ↓
[Lifecycle]
  Design → Develop → Deploy → Operate → Evolve → Retire
  Technical debt accrued and addressed at each stage
        ↓
[Dynamics]
  Feedback loops (good / bad) → Stock-and-flow → Delays → Emergent behavior
        ↓
[Reliability & Safety]
  SRE → STAMP/STPA → Safety cases → Incident response
        ↓
[Governance & Organization]
  Decision rights → Accountability → Team topology → Platform → Measurement
        ↓
[Realized System]
  Architecturally defensible · Operable · Evolvable · Safe · Auditable
```

DD8 (`docs/deep_dive/DD8_mit_sdm_marketing.md`) established the MIT SDM methodology at the repository's foundation — decomposition (Crawley), agent-based modeling (Moser), socio-technical systems. This document extends that foundation into **AI-specific system design and management territory** covered only in passing by DD8.

---

## 2. Sub-Skill 1: AI System Architecture (Crawley-Cameron + OPM + Tradespace)

### 2.1 Crawley-Cameron-Selva Methodology for AI

**Canonical reference**: Crawley, Cameron, Selva. *Systems Architecture: Strategy and Product Development for Complex Systems*, 2nd ed. (Pearson, 2023/2024). The 2nd edition explicitly frames methodology toward emerging technologies. Daniel Selva is the ML-and-search architect behind TAT-C ML, directly bridging tradespace exploration and AI.

**The seven-step process applied to an AI marketing stack**:

| Step | Classical systems | AI marketing stack |
|---|---|---|
| **1. Need** | Stakeholder need statement | "Increase retention uplift by X%, within budget Y, without violating brand-safety or privacy constraints Z" |
| **2. Concept** | Alternative system concepts | CF recommender vs. GNN vs. LLM-augmented hybrid vs. conversational agent |
| **3. Architecture** | Form-to-function decomposition | Functions: retrieve, rank, personalize, decide, explain, measure. Forms: vector DB, model, prompt, policy, instrument |
| **4. Decomposition** | Subsystem decomposition | Data pipeline / feature store / model training / eval / serving / decision engine / observability / governance |
| **5. Interface analysis** | Internal interfaces | Feature-store → model, model → decision engine, decision → ad platform, observability → all |
| **6. Tradespace** | Concept-performance-cost frontier | Quality × latency × cost × risk × flexibility Pareto sweep |
| **7. Evolution** | Architecture evolvability | Model deprecation, migration, safe retirement, new-capability insertion |

**The gap worth naming**: classical Crawley decomposition assumes **deterministic form-to-function mapping**. AI stacks exhibit *stochastic function* — model output is a distribution over training data × prompt × serving conditions. A fourth dimension, **epistemic uncertainty**, must be added to form / function / concept. No canonical treatise yet does this end-to-end for AI. This is an open SDM thesis opportunity.

**Pointers**:
- *Systems Architecture* 2nd ed (2023/2024) — [Pearson](https://www.pearson.com/en-us/subject-catalog/p/systems-architecture-strategy-and-product-development-for-complex-systems/P200000003288/9780137981793)
- MIT Professional Education. *AI System Architecture and LLM Applications* — [course](https://professional.mit.edu/course-catalog/ai-system-architecture-and-large-language-model-applications) — reuses form-vs-function vocabulary for LLM stacks (gateway, serving, retrieval, orchestration)
- MIT System Architecture Group — [systemarchitect.mit.edu](http://www.system-architect.org/)
- MIT Press. *Artificial Intelligence: A Systems Approach* — [book](https://direct.mit.edu/books/monograph/5793/Artificial-IntelligenceA-Systems-Approach-from)
- MIT OCW. *ESD.34 System Architecture* (Crawley) — [archive](https://ocw.mit.edu/courses/esd-34-system-architecture-january-iap-2007/)

### 2.2 Object-Process Methodology (OPM) — ISO 19450:2024

OPM's **bimodal representation** (OPD + OPL) maps naturally to AI systems: the object-stateful-process ontology directly captures *data → model → inference → decision* triples.

**2025 status**:
- OPM became **ISO 19450:2024** (January 2024 update; 158-page edition) — superseding the 2015 PAS
- **OPCloud** is the active successor to OPCAT; MIT-endorsed through Dori's SDM course
- Emerging **ISO 17649** specifies that standards should be based on an underlying OPM conceptual model — major cross-standard harmonization

**Direct AI application**:
- Kang, Shteingardt, Wang, Dori (2025). *Neuro-Conceptual Artificial Intelligence: Integrating OPM with Deep Learning to Enhance Question Answering Quality* — published case of OPM × LLM integration
- No canonical OPD reference model of an LLM pipeline (retriever → reranker → generator → guardrails) exists publicly yet — **another SDM deliverable opportunity**

**Pointers**:
- Wikipedia. *Object Process Methodology* — [wiki](https://en.wikipedia.org/wiki/Object_Process_Methodology)
- Dori. *OPM: A Holistic Systems Paradigm* (Springer) — [book](https://link.springer.com/book/10.1007/978-3-642-56209-9)
- MIT SDM blog on OPM as ISO standard — [sdm.mit.edu](https://sdm.mit.edu/the-maturation-of-model-based-systems-engineering-opm-as-the-iso-conceptual-modeling-language-standard/)
- OPCloud — [opcloud.tech](https://www.opcloud.tech/)
- ISO 19450:2024 INCOSE update slides — [omgwiki](https://www.omgwiki.org/MBSE/lib/exe/fetch.php?media=mbse:incose_mbse_iw_2021:incose_iw_2021_iso_19450_update_dov_dori_jan_30_2021.pdf)

### 2.3 Architecture Tradespace Exploration for AI

**Canonical MIT lineage**: Ross & Hastings tradespace paradigm → Selva's TAT-C → TAT-C ML. Genetic-algorithm search over enumerated concept space, Pareto-frontier analysis.

**AI-native analogs (2024-2025)**:
- **DataRobot `syftr` (2025)** — open-source Pareto-optimal GenAI workflow search; the closest AI equivalent to TAT-C
- Amazon Science. *Finding the Sweet Spot* (NeurIPS 2025 workshop) — empirical Pareto frontiers over accuracy × latency × cost for LLM deployments
- Recent arXiv: [2510.20653](https://arxiv.org/abs/2510.20653), [2507.03834](https://arxiv.org/html/2507.03834), [2510.26136](https://arxiv.org/html/2510.26136v1) — economic evaluation of LLMs

**Practitioner evaluation rituals**:
- **SEI ATAM** (Architecture Tradeoff Analysis Method) — the legacy architecture-evaluation discipline
- **SEI MLOps Evaluation Rubric** — ATAM-adjacent artifact for ML

**The AI-specific gap**: classical tradespace assumes **stable utility functions**. AI utility *drifts* with model and data updates. The frontier must be re-swept — an *evolvable tradespace* problem not addressed in canonical SDM literature.

**Pointers**:
- TAT-C ML — [selva-research.com](https://www.selva-research.com/research/tat-c-ml-tradespace-analysis-tool-for-constellations-with-machine-learning/)
- Ross & Hastings. *The Tradespace Exploration Paradigm* — [INCOSE](https://web.mit.edu/adamross/www/Ross_INCOSE05.pdf)
- Adam Ross / SEAri materials — [seari.mit.edu](https://seari.mit.edu/documents/courses/PI27s/SEAri_SC-2010-PI27s-06-1.pdf)
- DataRobot. *syftr — Pareto-Optimized AI Workflows* — [blog](https://www.datarobot.com/blog/pareto-optimized-ai-workflows-syftr/)
- Kapoor & Narayanan. *AI Agents That Matter* (arXiv 2024) — [arXiv](https://arxiv.org/abs/2407.01502)
- NVIDIA. *LLM Inference Benchmarking* — [devblog](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/)
- SEI ATAM — [sei.cmu.edu](https://www.sei.cmu.edu/library/atam-method-for-architecture-evaluation/)
- SEI MLOps Evaluation Rubric — [sei.cmu.edu](https://www.sei.cmu.edu/library/machine-learning-operations-mlops-evaluation-rubric/)

### 2.4 Cloud Reference Architectures (as prescriptive decompositions)

Four major reference architectures have converged on 5-6 pillar quality models but differ on decomposition primitives:

| Framework | Primary decomposition | AI-specific lens |
|---|---|---|
| **Azure WAF — AI Workloads** (2024-2025) | Client → AI gateway → Model serving (Foundry/AKS) → Data/grounding → Ops | Gateway pattern as first-class architectural decision |
| **AWS WAF — ML Lens + GenAI Lens + Responsible AI Lens** (re:Invent 2025) | Six pillars × six lifecycle phases (business goal → framing → data → model → deploy → monitor) | Responsible AI now a first-class pillar |
| **Google Cloud AI Adoption Framework + AI Hypercomputer** | Six themes: Learn / Lead / Access / Scale / Automate / Secure | Vertex AI RAG reference separates ingestion vs. serving subsystems |
| **NVIDIA Enterprise AI Reference Architecture** (May 2025) | Three factory tiers (RTX PRO / HGX / NVL72); hardware ratio notation (2-8-5-400) | Hardware-first architecture with explicit scaling to 32 nodes |

**AWS Responsible AI Lens ten dimensions** (re:Invent 2025): controllability, privacy, security, safety, veracity, robustness, fairness, explainability, transparency, governance.

**Pointers**:
- [Azure WAF AI Workloads](https://learn.microsoft.com/en-us/azure/well-architected/ai/) · [Application design](https://learn.microsoft.com/en-us/azure/well-architected/ai/application-design)
- [AWS WAF ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html) · [GenAI Lens](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/generative-ai-lens.html) · [Three new lenses at re:Invent 2025](https://aws.amazon.com/blogs/architecture/architecting-for-ai-excellence-aws-launches-three-well-architected-lenses-at-reinvent-2025/) · [InfoQ coverage](https://www.infoq.com/news/2025/12/aws-expands-well-architected/)
- [AWS CAF for AI/ML/GenAI](https://docs.aws.amazon.com/whitepapers/latest/aws-caf-for-ai/aws-caf-for-ai.html)
- [Google Cloud AI Adoption Framework](https://cloud.google.com/resources/cloud-ai-adoption-framework-whitepaper) · [Well-Architected AI/ML perspective](https://docs.cloud.google.com/architecture/framework/perspectives/ai-ml) · [RAG reference architecture](https://docs.cloud.google.com/architecture/rag-genai-gemini-enterprise-vertexai)
- [NVIDIA Enterprise RA whitepaper](https://docs.nvidia.com/enterprise-reference-architectures/white-paper.pdf) · [AI Factory whitepaper](https://docs.nvidia.com/ai-enterprise/planning-resource/ai-factory-white-paper/latest/index.html)

### 2.5 Architecture Decision Records (ADRs) for ML/AI

Canonical templates: **Nygard** (Title / Status / Context / Decision / Consequences), **MADR 4.0** (September 2024), **Y-Statements**. AWS Prescriptive Guidance and Azure WAF both include ADR chapters (2024-2025).

**ML-specific decision classes that must appear in ADRs**:
1. Model family and version pinning
2. Training-data lineage and refresh cadence
3. Feature store vs. pipeline recomputation
4. Online vs. batch serving topology
5. Drift- / trigger-based retraining policy
6. Human-in-the-loop and governance checkpoints
7. Prompt / template versioning for LLMs
8. Rollback semantics (weights vs. prompts vs. scaffolding)
9. Evaluation thresholds as release gates

**Reference**: Kreuzberger et al. (2022) *Machine Learning Operations (MLOps): Overview, Definition, and Architecture* — [arXiv](https://arxiv.org/abs/2205.02302) — the most-cited MLOps architecture reference.

**Gap**: no standardized ML-ADR template has emerged; this is another SDM contribution opportunity.

**Pointers**:
- [ADR community](https://adr.github.io/) · [ADR templates](https://adr.github.io/adr-templates/) · [MADR](https://adr.github.io/madr/)
- [joelparkerhenderson/architecture-decision-record](https://github.com/joelparkerhenderson/architecture-decision-record) — practical examples
- [Martin Fowler on ADR](https://martinfowler.com/bliki/ArchitectureDecisionRecord.html)
- [AWS Prescriptive Guidance ADR process](https://docs.aws.amazon.com/prescriptive-guidance/latest/architectural-decision-records/adr-process.html) · [AWS ADR best practices](https://aws.amazon.com/blogs/architecture/master-architecture-decision-records-adrs-best-practices-for-effective-decision-making/)
- [Azure WAF ADR chapter](https://learn.microsoft.com/en-us/azure/well-architected/architect-role/architecture-decision-record)
- [CMU 17-645 / MLIP book](https://mlip-cmu.github.io/book/) · [Kästner SEAI course](https://ckaestne.github.io/seai/)

### 2.6 AI System Description Languages and Metamodels

| Language | Status (2024-2026) | Fit for AI |
|---|---|---|
| **SysML v2** | OMG adopted July 2025; spec published Sep 2025 | KerML-based metamodel, textual syntax explicitly designed for AI tooling — most significant 2025 modeling-notation event |
| **OntoUML / UFO** (Guizzardi et al.) | GNN-driven stereotype prediction (Ali & Bork, CAiSE 2024); ExpO ontology-driven explainability (2024) | Ontology-first; gaining AI traction |
| **ArchiMate Next** | Snapshot 1, mid-2025 (pre-4.0) | Replaces layered structure with "Domains" to accommodate human × digital × AI components |
| **OPM / ISO 19450:2024** | Jan 2024 refresh | Bimodal; stateful-object ontology is strong fit for ML pipelines |

**The metamodel gap**: none of the above natively represent probabilistic flows, prompt templates as first-class artifacts, or the data-model co-evolution loop. SysML v2 + an AI-profile extension is the most promising path; OPM is the second-best candidate.

**Pointers**:
- [OMG SysML v2](https://www.omg.org/sysml/sysmlv2/) · [CTO.mil SysML v2 transition](https://www.cto.mil/wp-content/uploads/2025/02/SysML-v2-TransitionApproach-1.3.pdf) · [Sparx MBSE SysML v2](https://sparxsystems.com/mbse/sysml2/)
- [OntoUML wiki](https://en.wikipedia.org/wiki/OntoUML) · [Ali & Bork CAiSE 2024](https://link.springer.com/chapter/10.1007/978-3-031-61057-8_7) · [OntoUML model repo](https://github.com/OntoUML/ontouml-models)
- [ArchiMate Next preview](https://archimate.visual-paradigm.com/2026/02/10/archimate-next-a-new-era-in-enterprise-architecture-modeling/)

---

## 3. Sub-Skill 2: System Dynamics for AI (Forrester / Sterman / Meadows Applied)

AI systems are dynamic systems with **feedback loops, stocks, flows, delays, and emergent behavior**. The SDM tradition has a canonical vocabulary for analyzing these — it predates ML by decades — and the AI community has been reinventing it in isolation.

### 3.1 Foundational Canon

- **Forrester (1961)**. *Industrial Dynamics* — the origin of system dynamics as a discipline
- **Sterman (2000)**. *Business Dynamics: Systems Thinking and Modeling for a Complex World* — the canonical textbook
- **Meadows (1999)**. *Leverage Points: Places to Intervene in a System* — pointer / place-finding framework
- **Meadows (2008)**. *Thinking in Systems* — accessible introduction
- **Repenning & Sterman (2001)**. *Nobody Ever Gets Credit for Fixing Problems That Never Happened* — MIT Sloan Mgmt Review paper establishing the **"capability trap"** — directly applicable to AI organizations over-relying on model patches

**MIT System Dynamics Group**: John Sterman, Nelson Repenning, Hazhir Rahmandad. AI-specific output from the group is notably thin as of Jan 2026 — an opening for SDM thesis work.

### 3.2 Named Feedback Traps in AI Systems

The AI literature has discovered many of these piecemeal. Naming them with SD vocabulary clarifies the pattern:

| Feedback trap | Canonical reference | SD archetype |
|---|---|---|
| **Runaway / reinforcing loops** in predictive systems | Ensign et al. (FAT* 2018) *Runaway Feedback Loops in Predictive Policing* — [arXiv](https://arxiv.org/abs/1706.09847) | Reinforcing loop |
| **Performative prediction** | Perdomo et al. (ICML 2020) *Performative Prediction* — [arXiv](https://arxiv.org/abs/2002.06673) | Self-fulfilling loop |
| **Goodhart's law at scale** | Manheim & Garrabrant (2018) *Categorizing Variants of Goodhart's Law* — [arXiv](https://arxiv.org/abs/1803.04585) | KPI drift |
| **Data feedback loops** | Taori & Hashimoto (ICML 2023) *Data Feedback Loops* — [arXiv](https://arxiv.org/abs/2209.03942) | Delayed reinforcing |
| **Fairness feedback loops** | Wyllie et al. (FAccT 2024) *Fairness Feedback Loops* — [arXiv](https://arxiv.org/abs/2403.07857) | Cumulative disadvantage |
| **Degenerate loops in recommenders** | Jiang et al. (AIES 2019) *Degenerate Feedback Loops in Recommender Systems* — [arXiv](https://arxiv.org/abs/1902.10730) | Narrowing loop |
| **Algorithmic confounding / homogenization** | Chaney, Stewart, Engelhardt (RecSys 2018) — [arXiv](https://arxiv.org/abs/1710.11214) | Convergent filter bubble |
| **Popularity-bias amplification** | Mansoury et al. (CIKM 2020) — [arXiv](https://arxiv.org/abs/2007.13019) | Success-to-the-successful |
| **Long-term welfare vs. engagement** | Mladenov, Boutilier et al. (ICML 2020) — [arXiv](https://arxiv.org/abs/2008.00104) | Horizon mismatch |
| **Model collapse** (training on own output) | Shumailov et al. (Nature 2024) *AI Models Collapse When Trained on Recursively Generated Data* — [Nature](https://www.nature.com/articles/s41586-024-07566-y) · [arXiv](https://arxiv.org/abs/2305.17493) | Recursive degradation |
| **Model-collapse theory** | Dohmatob et al. (2024) *A Tale of Tails* — [arXiv](https://arxiv.org/abs/2402.07043); Gerstgrasser et al. (2024) *Is Model Collapse Inevitable?* — [arXiv](https://arxiv.org/abs/2404.01413); Alemohammad et al. (ICLR 2024) *Self-Consuming Generative Models Go MAD* — [arXiv](https://arxiv.org/abs/2307.01850) | Variance collapse |
| **Data-wall limits to growth** | Villalobos et al. (2024) *Will We Run Out of Data?* — [arXiv](https://arxiv.org/abs/2211.04325) | Limits to growth |
| **Capability trap / shifting the burden** | Repenning & Sterman (MIT SMR 2001) — [Sloan Review](https://sloanreview.mit.edu/article/nobody-ever-gets-credit-for-fixing-problems-that-never-happened-creating-and-sustaining-process-improvement/) | Shifting the burden |

### 3.3 Meadows/Sterman Archetypes Mapped to AI Organizations

- **Limits to growth** — scaling-law plateaus, data-wall, compute-capacity ceilings
- **Shifting the burden** — fine-tuning / RLHF / prompt-patches as symptomatic fixes eroding investment in data-layer and evaluation infrastructure
- **Tragedy of the commons** — shared web corpora polluted by generative output
- **Success to the successful** — Matthew effects in recommendation; foundation-model ecosystem concentration
- **Tragedy of the commons** on training-data quality — everyone benefits from clean data, nobody pays to maintain it

### 3.4 SDM Practice: Causal-Loop Diagrams for AI Systems

Canonical CLD patterns an SDM thesis could formalize (none are published end-to-end as of Jan 2026):
- *Data freshness → Model quality → Decisions → New data quality* (virtuous or vicious depending on sign)
- *Trust → Usage → Feedback → Model quality → Trust*
- *Engagement optimization → Filter bubbles → Reduced signal diversity → Model degradation*
- *Technical-debt accumulation → Slower releases → More debt* (capability trap)
- *Generative output → Web corpus pollution → Training-data degradation → Model collapse*

**Pointers**:
- [MIT System Dynamics Group](https://systemdynamics.mit.edu/) · [System Dynamics Society](https://systemdynamics.org/)
- [Sterman MIT Sloan profile](https://mitsloan.mit.edu/faculty/directory/john-d-sterman) · [Repenning profile](https://mitsloan.mit.edu/faculty/directory/nelson-p-repenning) · [Rahmandad profile](https://mitsloan.mit.edu/faculty/directory/hazhir-rahmandad)
- Forrester. *Industrial Dynamics* (MIT Press 1961 foundational) — [MIT Press](https://mitpress.mit.edu/9780262560016/industrial-dynamics/)
- Sterman. *Business Dynamics* (2000 foundational) — [McGraw-Hill](https://www.mhprofessional.com/business-dynamics-systems-thinking-and-modeling-for-a-complex-world-9780072389159-usa)
- Meadows. *Thinking in Systems* (2008) — [Chelsea Green](https://www.chelseagreen.com/product/thinking-in-systems/)
- Meadows. *Leverage Points* (1999) — [donellameadows.org](https://donellameadows.org/archives/leverage-points-places-to-intervene-in-a-system/)

---

## 4. Sub-Skill 3: Lifecycle Management and Technical Debt

### 4.1 Sculley et al. (NeurIPS 2015) — The Nine Debt Categories

**Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo, Dennison. *Hidden Technical Debt in Machine Learning Systems* — [NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)** remains the canonical reference. The nine categories:

| # | Category | One-line definition |
|---|---|---|
| 1 | **Boundary erosion / entanglement** | CACE — Changing Anything Changes Everything; no strict abstraction boundaries between features / hyperparameters / models |
| 2 | **Correction cascades** | Stacked corrective models creating dependency chains |
| 3 | **Undeclared consumers** | Downstream systems silently depending on model outputs |
| 4 | **Unstable data dependencies** | Input signals that qualitatively shift over time |
| 5 | **Underutilized data dependencies** | Legacy / bundled / epsilon features carrying risk without value |
| 6 | **Feedback loops (direct and hidden)** | Model influences its own training distribution |
| 7 | **ML-system anti-patterns / glue code** | 95% glue, 5% ML; pipeline jungles; dead experimental codepaths |
| 8 | **Configuration debt** | Config treated as afterthought despite exceeding model code in complexity |
| 9 | **Changes in the external world** | Fixed thresholds, monitoring / testing gaps, real-world drift |

**Key post-2015 finding**: *no new category has achieved canonical status replacing Sculley's taxonomy*. Subsequent work refines, operationalizes, or adds risk-oriented reframings rather than superseding:
- Tang et al. (ICSE 2020) — deep-learning program failures — [arXiv](https://arxiv.org/abs/2008.12005)
- Foidl, Felderer, Ramler (2022) — risk-oriented ML tech debt — [arXiv](https://arxiv.org/abs/2106.09524)
- Liu et al. (2021) — *Towards Understanding the Challenges Faced by ML Developers* — [arXiv](https://arxiv.org/abs/2103.14946)
- Recupito et al. (2023) — *Technical Debt in AI-Enabled Systems* — [arXiv](https://arxiv.org/abs/2304.11657)
- OBrien et al. (2022) — *23 Shades of Self-Admitted Tech Debt in ML* — [arXiv](https://arxiv.org/abs/2205.13938)
- Bogner, Verdecchia (2021) — industry survey — [arXiv](https://arxiv.org/abs/2206.00158) · [ACM systematic review](https://dl.acm.org/doi/10.1145/3453478)
- Alahdab & Calikli — ML tech-debt taxonomy — [IEEE](https://ieeexplore.ieee.org/document/9609153)

**Prompt / LLM debt** (2023-2025) is emerging as a proto-category but is not yet canonical.

### 4.2 Amershi et al. (ICSE 2019) — Engineering Workflow

**Amershi, Begel, Bird, DeLine, Gall, Kamar, Nagappan, Nushi, Zimmermann. *Software Engineering for Machine Learning: A Case Study* (ICSE-SEIP 2019)** — [Microsoft Research PDF](https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/) · [IEEE](https://ieeexplore.ieee.org/document/8804457).

Contributions:
- **9-stage workflow**: requirements → data collection → cleaning → labeling → feature engineering → training → evaluation → deployment → monitoring
- **Three axes of ML-specific difficulty**: (1) data discovery / management / versioning is much more complex than software engineering, (2) model customization and reuse require different skills than typical software teams, (3) AI components are harder to handle as distinct modules — models may be entangled and exhibit non-monotonic error behavior
- Team-integration debt — ML engineers vs. software engineers working under different cadences

**Complementary data-management canon**:
- Polyzotis et al. (SIGMOD 2017 / VLDB 2018) *Data Management Challenges in Production ML* — [VLDB PDF](https://www.vldb.org/pvldb/vol11/p2081-polyzotis.pdf)
- Schelter et al. *Deequ: Automating Large-Scale Data Quality Verification* — [VLDB PDF](https://www.vldb.org/pvldb/vol11/p1781-schelter.pdf)
- Breck et al. *Data Validation for Machine Learning* (SysML 2019) — [Google Research](https://research.google/pubs/pub47967/)
- Paleyes et al. (2020) *Challenges in Deploying Machine Learning* — [arXiv](https://arxiv.org/abs/2010.12869)

### 4.3 Standards: ISO/IEC 5338:2023 + 42001 + 23053 + 15288

- **ISO/IEC 5338:2023** *AI System Life Cycle Processes* (Dec 2023) — layers AI-specific processes over ISO/IEC/IEEE 15288 structure; introduces **continuous validation** as first-class concept. Process categories: data acquisition, data preparation, model building, V&V, deployment, operation/monitoring, continuous validation, retirement. [ISO](https://www.iso.org/standard/81118.html) · [SIG overview](https://www.softwareimprovementgroup.com/iso-5338-get-to-know-the-global-standard-on-ai-systems/)
- **ISO/IEC 42001:2023** — AI management system (see `regulatory_appendix.md`) — [ISO](https://www.iso.org/standard/81230.html)
- **ISO/IEC 23053** — AI/ML framework (complementary) — [ISO](https://www.iso.org/standard/74438.html)
- **ISO/IEC/IEEE 15288** — generic systems-engineering lifecycle (the substrate 5338 extends)

**Gap**: 5338 lacks explicit LLM / foundation-model provisions; retirement phase is thinly specified.

### 4.4 MLOps Maturity Models

| Model | Levels | Canonical property |
|---|---|---|
| **Google MLOps Maturity** (most-cited) | 0 manual → 1 ML pipeline automation → 2 CI/CD for pipelines | [Google architecture docs](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) |
| **Microsoft MLOps Maturity** | 0-4 (five levels, enterprise-granular) | [MS Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model) |
| **CD4ML** (Thoughtworks) | Three axes: code, model, data | [Martin Fowler](https://martinfowler.com/articles/cd4ml.html) |
| **CRISP-DM** (1996, foundational) | Business understanding → data → modeling → evaluation → deployment | [CRISP-DM PDF](https://www.the-modeling-agency.com/crisp-dm.pdf) |
| **TDSP** (Microsoft Team Data Science Process) | Operationalization of CRISP-DM | [MS Learn](https://learn.microsoft.com/en-us/azure/architecture/data-science-process/overview) |
| **ml-ops.org community reference** | — | [ml-ops.org](https://ml-ops.org/) |

**Gap**: maturity models rarely address **model-deprecation maturity**; most assume single-model shops, not model portfolios.

### 4.5 Model Deprecation, Migration, Retirement

Practitioner-standard but under-theorized:
- Shadow deployment → canary rollout → full
- A/A testing for parity
- Regression-contract tests for ML
- OpenAI and Anthropic model-deprecation policies are the de facto industry reference (2023-2025)
- No canonical academic paper on graceful ML retirement — **SDM thesis opportunity**

**Pointers**:
- [OpenAI deprecations](https://platform.openai.com/docs/deprecations) · [Anthropic model deprecations](https://docs.anthropic.com/en/docs/about-claude/model-deprecations)
- [Netflix ML retraining pipelines blog](https://netflixtechblog.com/machine-learning-model-retraining-pipelines-at-netflix)
- Klaise et al. (2021) *Monitoring and explainability of ML in production* — [arXiv](https://arxiv.org/abs/2103.08937)
- Breck et al. *The ML Test Score* — [Google Research](https://research.google/pubs/pub46555/)
- Zinkevich. *Rules of ML* — [Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)

### 4.6 Data-Quality and Debt Tooling

- **Deequ** (AWS) — [GitHub](https://github.com/awslabs/deequ)
- **Great Expectations** — [GitHub](https://github.com/great-expectations/great_expectations)
- **TensorFlow Data Validation** — [Docs](https://www.tensorflow.org/tfx/data_validation/get_started)
- **ML Test Score** (Breck) — closest canonical rubric
- Renggli et al. (2021) *Data Debugging survey* — [arXiv](https://arxiv.org/abs/2111.14793)

**Gap**: no SonarQube-equivalent for ML code quality; no mature tool for quantifying *correction-cascade depth* or *undeclared-consumer count*.

---

## 5. Sub-Skill 4: Platform Engineering and Team Topologies for AI

### 5.1 Team Topologies Mapped to AI Organizations

**Skelton & Pais. *Team Topologies* (IT Revolution, 2019)** — four team types:

| Team type | AI-org mapping |
|---|---|
| **Platform** | ML Platform team owning feature store, model registry, serving gateway, eval harness, prompt registry |
| **Enabling** | MLOps / ML Enablement team coaching product teams on ML best practices |
| **Complicated-Subsystem** | Foundation-model / inference-runtime / specialized-optimization team |
| **Stream-aligned** | Product squads embedded with ML engineers, owning specific model-powered features |

**Anti-pattern flagged in practice**: treating "AI CoE" as a Complicated-Subsystem team when it should be a Platform team — produces bottlenecks because CoEs are simultaneously asked to deliver research AND unblock product teams (violating cognitive-load limits from the book).

**LLM-era patterns** (2024-2026): stream-aligned **LLM squads** owning specific LLM-powered product surfaces, with a Platform team owning prompt registries, eval harnesses, and model gateways.

### 5.2 Internal Developer Platforms (IDP) for AI

- **CNCF TAG App Delivery. *Platforms for AI* whitepaper (2024)** — formalized IDP-for-AI as a discipline distinct from classic IDPs; adds GPU scheduling, model registry, feature store, eval harness as first-class platform capabilities. [CNCF](https://tag-app-delivery.cncf.io/whitepapers/platforms/)
- **Backstage** (Spotify) — de facto ML asset catalog plane (model cards, dataset lineage); [docs](https://backstage.io/docs/features/software-catalog/)
- **Port** — added AI resource primitives 2024-2025 — [port.io](https://www.getport.io/blog/ai-platform-engineering)
- **Humanitec** — AI/ML platform engineering — [humanitec.com](https://humanitec.com/blog/ai-ml-platform-engineering)
- **Thoughtworks Tech Radar Vol 30-31** (2024-2025): "LLM gateway as platform capability" → Adopt; "RAG-as-a-platform-service" → Trial — [radar](https://www.thoughtworks.com/radar)
- **CNCF Platform Engineering Maturity Model (2024)** — [CNCF](https://www.cncf.io/reports/platform-engineering-maturity-model/)

### 5.3 Canonical ML Platform Case Studies

| Platform | Firm | Key lesson |
|---|---|---|
| **Michelangelo** | Uber | Originated 2017; over-ambitious monolith, later simplified. 2024 update: ["From Predictive to Generative"](https://www.uber.com/blog/from-predictive-to-generative-ai/). [Original 2017 post](https://www.uber.com/blog/michelangelo-machine-learning-platform/) |
| **Bighead / Chronon / Zipline** | Airbnb | Modular end-to-end ML platform; reduced model dev time months→days. [Databricks session](https://www.databricks.com/session/bighead-airbnbs-end-to-end-machine-learning-platform) |
| **Pro-ML / Feathr / DARWIN** | LinkedIn | Scaling ML productivity at LinkedIn; Feathr open-sourced — [engineering blog](https://engineering.linkedin.com/blog/topic/ai) |
| **Metaflow / Maestro** | Netflix | Opinionated but Python-native; won by developer ergonomics. [Metaflow 2.0](https://netflixtechblog.com/metaflow-2-0-and-beyond) · [Maestro 2024](https://netflixtechblog.com/maestro-netflixs-workflow-orchestrator) |
| **FBLearner Flow / Ax / Looper** | Meta | Originated 2016; monolithic; evolved to decoupled event-driven architecture. [FBLearner launch](https://engineering.fb.com/2016/05/09/core-infra/introducing-fblearner-flow/) · [Looper 2022](https://engineering.fb.com/2022/09/12/production-engineering/looper-meta-ai-optimization/) |
| **Hendrix / Klio / ML Home** | Spotify | [Engineering blog](https://engineering.atspotify.com/2023/06/spotifys-new-ml-platform/) |
| **MLEnv** | Pinterest | [Medium](https://medium.com/pinterest-engineering/mlenv-standardizing-ml-at-pinterest) |
| **Merlin / Magic** | Shopify | [Engineering](https://shopify.engineering/merlin-shopifys-new-machine-learning-platform) |
| **Fabricator** | DoorDash | Declarative feature engineering — [engineering](https://doordash.engineering/2022/10/04/introducing-fabricator-a-declarative-feature-engineering-framework/) |
| **Griffin** | Instacart | Rebuilt ML platform — [tech.instacart.com](https://tech.instacart.com/griffin-how-we-rebuilt-instacarts-ml-platform) |

**Universal lesson**: every mature ML platform eventually converges on the same seven components — feature store, model registry, training orchestrator, eval service, inference gateway, observability, governance. The order of arrival and who owns what varies; the endpoints don't.

### 5.4 Conway's Law and Inverse-Conway in AI

- **Conway (1968)** — [Mel Conway](https://www.melconway.com/Home/Committees_Paper.html) (foundational)
- **Fowler on Conway's Law** — [bliki](https://martinfowler.com/bliki/ConwaysLaw.html)
- **Dehghani's Data Mesh** principles — [Fowler](https://martinfowler.com/articles/data-mesh-principles.html) — Conway-driven organizational pattern for data products

**AI-specific Conway patterns (2024-2026)**:
- Model monoliths (single foundation model served to all product teams) mirror **centralized** platform orgs
- Fine-tuned per-domain models mirror **federated** orgs
- RAG architectures encode org boundaries: *retrieval index ownership fights are always org-chart fights*
- Research-engineering handoff is where Conway hits hardest and is least studied

### 5.5 Organizational Design Patterns

**2024-2026 consensus** (McKinsey State of AI, Gartner, Forrester, BCG, Deloitte): **hub-and-spoke** — central platform + federated product teams — dominates. Pure CoE is declining; pure decentralization only works at AI-native firms.

- [McKinsey State of AI 2024/2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)
- [HBR Fountaine / McCarthy / Saleh *Building the AI-Powered Organization*](https://hbr.org/2023/07/building-the-ai-powered-organization)
- [a16z *Emerging Architectures for LLM Applications*](https://a16z.com/emerging-architectures-for-llm-applications/)
- [O'Reilly *AI Adoption in the Enterprise 2024*](https://www.oreilly.com/radar/ai-adoption-in-the-enterprise-2024/)

### 5.6 Role Taxonomy (2024-2026)

- **AI Engineer** (Shawn Wang 2023) — consolidated as distinct role; product-facing, uses models as APIs
- **ML Engineer** — training-focused
- **MLOps / Platform ML Engineer** — deployment, infrastructure, reliability
- **Applied Scientist** — research + production
- **Research Scientist** — research only
- **AI Product Manager**
- **AI / ML Architect**
- **ML Ethicist / AI Safety Lead** (emerging)
- **Data Engineer**

**Pointers**:
- [AI Engineer roadmap](https://roadmap.sh/ai-engineer) · [MLOps roadmap](https://roadmap.sh/mlops)
- [Chip Huyen — GenAI Platform 2024](https://huyenchip.com/2024/07/25/genai-platform.html)
- [Eugene Yan — LLM engineering](https://eugeneyan.com/writing/llm-engineering/)
- [Pragmatic Engineer — AI engineers](https://newsletter.pragmaticengineer.com/p/ai-engineers)
- [State of AI Report (Benaich & Hogarth annual)](https://www.stateof.ai)

### 5.7 DevEx Metrics for AI Teams

- **DORA** (Google Cloud) — [dora.dev](https://dora.dev/research/); [AI-specific DORA](https://dora.dev/ai)
- **SPACE framework** (Forsgren et al.) — [ACM Queue](https://queue.acm.org/detail.cfm?id=3454124)
- **DevEx framework** (Noda, Forsgren 2023) — [getdx.com](https://getdx.com/research/devex)
- **DORA for ML** — [Thoughtworks](https://www.thoughtworks.com/insights/blog/engineering-effectiveness/dora-for-ml)

**Proposed AI-specific metrics**: time-to-first-experiment; training-to-production cycle time; eval pass rate; model rollback frequency; feature-reuse ratio; offline/online metric parity.

**Open problems**: no industry-standard DORA-for-ML yet; eval quality is the new deployment frequency; reproducibility metrics remain informal.

---

## 6. Sub-Skill 5: SRE, Safety Engineering, STAMP/STPA

### 6.1 Leveson STAMP / STPA Applied to AI

**Nancy Leveson. *Engineering a Safer World* (MIT Press, 2012)** — foundational text for STAMP (Systems-Theoretic Accident Model and Processes) and STPA (System-Theoretic Process Analysis). The MIT PSAS group has extended STPA to ML systems, treating *unsafe control actions from learned controllers* as first-class hazards.

**Key 2022-2025 contributions**:
- Dobbe (2022) *System Safety and Artificial Intelligence* — [arXiv](https://arxiv.org/abs/2302.10588)
- Rismani et al. *From Plane Crashes to Algorithmic Harm* (CHI 2023) — [arXiv](https://arxiv.org/abs/2211.03315)
- Rismani et al. STPA for ML (FAccT 2023) — [ACM](https://dl.acm.org/doi/10.1145/3593013.3594064)
- Qi et al. (2023) *STPA for autonomous driving ML* — [arXiv](https://arxiv.org/abs/2310.03984)
- *STPA for ML Safety* (Safety Science 2024) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925753523002771)
- **Mylius (2025) *Systematic Hazard Analysis for Frontier AI using STPA*** — [arXiv 2506.01782](https://arxiv.org/html/2506.01782v1) · [blog](https://simonmylius.com/blog/stpa) — derives Unsafe Control Actions for frontier AI; finds STPA identifies causal factors missed by unstructured hazard analysis
- ***STAMP/STPA Informed Characterization of Factors Leading to Loss of Control in AI Systems*** (Dec 2025) — [arXiv 2512.17600](https://arxiv.org/abs/2512.17600)

**Pointers**:
- [MIT PSAS / Leveson group](http://sunnyday.mit.edu/) · [STAMP Workshop](https://psas.scripts.mit.edu/home/)
- [Engineering a Safer World (2012, open access)](https://mitpress.mit.edu/9780262533690/engineering-a-safer-world/)
- [STPA Handbook (Leveson & Thomas 2018)](http://sunnyday.mit.edu/STPA-Handbook.pdf)
- [STAMP-related publications](http://sunnyday.mit.edu/STAMP-publications.html)
- [UL SIS intro to STAMP / STPA / CAST](https://www.ul.com/sis/blog/introduction-to-stamp-stpa-and-cast)

**Open problem**: STPA assumes a **bounded control structure**; LLMs with open-ended action spaces break that assumption. No consensus on how to enumerate unsafe control actions when the controller is a foundation model. Training-time vs. runtime hazard distinction is unresolved.

### 6.2 SRE for AI Systems

**Canonical reference**: Google *Site Reliability Engineering* (2016) + *The Site Reliability Workbook* (2018). Classical SLI/SLO framework doesn't cleanly map to ML — quality metrics are distributional, not binary success/failure. Requires **quality SLOs** on calibration, drift, and hallucination rates.

**State of practice**:
- Google 2024-2025 writeups emphasize MLOps reliability with error budgets tied to offline eval regressions
- Meta's recommender reliability uses shadow traffic as SLI source
- SREcon 2023 and 2024 AI / ML tracks — [SREcon23](https://www.usenix.org/conference/srecon23americas) · [SREcon24](https://www.usenix.org/conference/srecon24americas)
- Microsoft Azure Well-Architected AI (2024) — [AI WAF](https://learn.microsoft.com/en-us/azure/well-architected/ai/)

**Pointers**:
- [Google SRE Book (free online, foundational)](https://sre.google/sre-book/table-of-contents/)
- [SRE Workbook (free online)](https://sre.google/workbook/table-of-contents/)
- Meta MLOps at Scale — [engineering](https://engineering.fb.com/2022/09/02/ml-applications/mlops-ml-operations-at-scale/)

**Open problems**:
- No standard for "hallucination error budget"
- Toil definition for ML ops (retraining, data-quality triage) is under-specified
- Postmortem templates for stochastic failures are immature

### 6.3 Chaos Engineering for AI

Classical **Netflix Simian Army** (2011 foundational) and *Principles of Chaos Engineering* (2017) don't cleanly extend to ML systems whose "state" is a learned distribution.

**Adaptations**:
- **PromptBench / PromptRobust** — prompt-perturbation testing analog of fault injection — [arXiv](https://arxiv.org/abs/2306.04528) · [GitHub](https://github.com/microsoft/promptbench)
- **DecodingTrust** (2023) — [arXiv](https://arxiv.org/abs/2306.11698)
- **HarmBench** (2024) — [arXiv](https://arxiv.org/abs/2402.04249)
- Input-drift chaos via synthetic distribution shift — Microsoft's Responsible AI Toolbox, IBM AI Fairness 360

**Gap**: no widely-adopted "Chaos Monkey for LLMs"; closest is red-teaming harnesses (Anthropic, OpenAI).

**Open problems**:
- Reproducibility of chaos experiments on stochastic models
- No "steady state" definition for LLMs
- Coverage metrics for prompt-space perturbation are absent

**Pointers**:
- [Principles of Chaos Engineering (2017)](https://principlesofchaos.org/)
- [Netflix Simian Army (2011, foundational)](https://netflixtechblog.com/the-netflix-simian-army-16e57fbab116)
- [Chaos Mesh](https://chaos-mesh.org/) · [Gremlin](https://www.gremlin.com/)
- [AI red-teaming taxonomy (2023)](https://arxiv.org/abs/2310.16985)

### 6.4 Incident Response and Databases

- **AI Incident Database** (Partnership on AI) — [incidentdatabase.ai](https://incidentdatabase.ai/)
- **AIAAIC** — [aiaaic.org](https://www.aiaaic.org/)
- **AVID** (AI Vulnerability Database) — [avidml.org](https://avidml.org/)
- **MITRE ATLAS** (Adversarial Threat Landscape for AI Systems) — [atlas.mitre.org](https://atlas.mitre.org/)
- **OWASP Top 10 for LLM Applications** — [genai.owasp.org](https://genai.owasp.org/llm-top-10/)
- **OECD AI Incidents Monitor** — [oecd.ai](https://oecd.ai/en/incidents)

**Public postmortem-worthy incidents**: Galactica demo pull (2022), Bard launch issues (2023), early Copilot misfires, Replika sentiment incidents.

**Open problems**: no industry-standard postmortem template for model-quality regressions; rollback granularity (weights vs. prompts vs. scaffolding) inconsistent.

### 6.5 Safety Cases and Assurance Cases

- **GSN** (Goal Structuring Notation) — [SCSC community standard](https://scsc.uk/gsn) (foundational)
- Bloomfield & Rushby (2023) *Assurance 2.0* — [arXiv](https://arxiv.org/abs/2305.10295)
- **Clymer et al. (2024) *Safety Cases for Frontier AI*** — [arXiv](https://arxiv.org/abs/2311.14522) — the canonical modern frontier-AI safety-case paper
- UK AI Safety Institute (2024) *Approach to Evaluations* — [gov.uk](https://www.gov.uk/government/publications/ai-safety-institute-approach-to-evaluations)
- FDA SaMD safety-case patterns — [FDA SaMD](https://www.fda.gov/medical-devices/software-medical-device-samd)

**Open problems**: no agreed-upon claims structure for foundation models; evidence sufficiency underspecified.

### 6.6 Resilience Engineering for AI

- Erik Hollnagel tradition — [erikhollnagel.com](https://www.erikhollnagel.com/)
- [Resilience Engineering Association](https://resilienthealthcare.net/)
- *Graceful degradation in ML systems* (2023) — [arXiv](https://arxiv.org/abs/2308.14761)

**Cross-cutting open problems (safety)**:
- No unifying ontology bridging STPA hazards, SRE SLIs, and ATLAS threats
- Error-budget semantics for stochastic quality
- Assurance-case evidence standards for foundation models are pre-paradigmatic as of early 2026

---

## 7. Sub-Skill 6: Governance as System Design Problem

Regulatory compliance is the Sales / Productization framing; the SDM framing treats governance as a **system-design concern** — decision rights, accountability graphs, measurement, evolution policy.

**Core elements**:
- **Decision rights allocation** — who can approve model releases, what evidence is required, who can override
- **Accountability graphs** — which human is responsible when which subsystem fails
- **RACI for AI** — Responsible / Accountable / Consulted / Informed for each decision class
- **Model cards** (Mitchell et al.) and **datasheets for datasets** (Gebru et al.) — standardized documentation
- **Policy as code** — expressing governance rules as executable constraints

**Cross-reference**: see `docs/skills/regulatory_appendix.md` for the regulatory-compliance frame (NIST AI RMF, EU AI Act, ISO 42001, FTC, state-law). The SDM governance frame is the **internal organizational design** that makes the regulatory frame achievable.

---

## 8. Integration — The SDM Reference Architecture for AI Marketing

Pulling the six sub-skills together into a single diagram (textual form, since OPD tools aren't rendered in markdown):

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stakeholder Context                                                 │
│  (Customers · Regulators · Investors · Brand · Competitors · Data)  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  System Architecture             │  ← §2
            │  (Decomposition / Interfaces /   │
            │   Tradespace / ADRs / OPM)       │
            └────────────────┬────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
┌─────┴──────┐   ┌───────────┴───────────┐   ┌──────┴──────┐
│ Data        │   │ Model & Decision        │   │ Serving &     │
│ Subsystem   │◀─▶│ Subsystem               │◀─▶│ Experience    │
│ (pipeline + │   │ (training / eval /      │   │ (gateway /    │
│  features)  │   │  causal / bandit / RL)  │   │  RAG / agent) │
└─────┬───────┘   └────────────┬────────────┘   └──────┬────────┘
      │                        │                       │
      │                        │                       │
      ▼                        ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Cross-Cutting                                                       │
│                                                                      │
│  · Observability  (§3.8 productization)                             │
│  · Evaluation     (§3.9 productization)                             │
│  · Guardrails     (§3.10 productization)                            │
│  · Off-Policy Eval (DD6 §4)                                         │
│  · System Dynamics monitoring (§3 here — feedback-loop detection)   │
│  · Technical-debt accounting (§4 here — Sculley categories tracked) │
│  · SRE / STAMP (§6 here — SLIs, UCAs, incident response)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  Governance & Org Design         │  ← §7 + §5 + regulatory_appendix
            │  (Team Topologies / IDP / RACI / │
            │   Model cards / Policy-as-code)  │
            └────────────────┬────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  Lifecycle Process               │  ← §4
            │  (ISO 5338 / MLOps maturity /    │
            │   Model deprecation / Retire)    │
            └──────────────────────────────────┘
```

The six sub-skills attach to the six labeled layers. An SDM student designing an AI marketing stack should be able to point to each layer and name its owner, its interfaces, its debt ledger, its reliability SLOs, its feedback loops under surveillance, and its governance checkpoints.

---

## 9. Cross-References to Existing Docs

| Concern | Also covered in |
|---|---|
| Marketing domain methods | `docs/01-05`, `docs/deep_dive/DD1-DD5` |
| Theoretical substrate (Bayesian / causal / RL / representation) | `docs/deep_dive/DD6_ai_marketing_foundations.md` |
| MIT research landscape (Tucker, Eckles, Aral, Simester, Perakis) | `docs/deep_dive/DD7_mit_marketing_research.md` |
| Original SDM baseline (Crawley, Cameron, Moser) | `docs/deep_dive/DD8_mit_sdm_marketing.md` |
| Selling AI | `docs/skills/sales_skill.md` §2 |
| Investment AI | `docs/skills/sales_skill.md` §3 |
| Product-implementation concerns | `docs/skills/productization_skill.md` |
| Regulatory compliance | `docs/skills/regulatory_appendix.md` |
| Agentic topology and Dao 2025 | `docs/frontier/agentic_ai_marketing.md` + DD8 §2 |

---

## 10. Open Problems Worth an SDM Thesis

Concrete gaps in the 2024-2026 literature that an SDM-thesis-scope project could fill:

1. **End-to-end Crawley decomposition of a production AI marketing stack** — no canonical reference exists
2. **Canonical OPD (Object-Process Diagram) of an LLM-based commerce pipeline** — no reference model published
3. **Epistemic uncertainty as a fourth architectural dimension** alongside form / function / concept
4. **Evolvable tradespace methodology** when utility functions drift with model / data updates
5. **Standardized ML-ADR template** with the nine ML-specific decision classes
6. **Unified causal-loop diagram** spanning Shumailov model collapse, Ensign runaway loops, Repenning-Sterman capability trap, Meadows archetypes
7. **Formal bridge between AI-safety feedback-loop literature and MIT System Dynamics tradition**
8. **Tenth debt category** (prompt / LLM debt) achieving canonical post-Sculley status — no paper yet
9. **Model-deprecation-maturity model** — MLOps maturity models treat this thinly
10. **Graceful ML retirement methodology** — no canonical academic paper
11. **Correction-cascade-depth and undeclared-consumer-count metrics**
12. **DORA-for-ML benchmark** with validated eval-quality metric
13. **Assurance-case claims structure for foundation models**
14. **Error-budget semantics for stochastic quality** (hallucination budget)
15. **Unifying ontology spanning STPA UCAs + SRE SLIs + MITRE ATLAS threats**
16. **SysML v2 + AI-profile extension** natively representing probabilistic flows, prompt templates, data-model co-evolution

Each item is small enough to scope to a semester of SDM research and large enough to be a genuine contribution.

---

## 11. Reading Paths for SDM Students

### Path SDM-A: "I need to defend the architecture of an AI marketing stack"
1. §2 (architecture) — Crawley seven-step applied
2. §2.4 (cloud reference architectures)
3. §2.5 (ADRs) — capture your decisions
4. §8 (integrated diagram)
5. DD8 §4 for the methodological foundation

### Path SDM-B: "I need to analyze why a production AI system is degrading"
1. §3 (system dynamics) — identify the feedback trap by name
2. §4.1 (Sculley's nine debt categories) — diagnose the accumulated debt
3. §6 (STAMP/STPA) — enumerate unsafe control actions
4. §3.4 — draw the causal-loop diagram

### Path SDM-C: "I'm designing the org that will build and run this"
1. §5.1 (Team Topologies mapping)
2. §5.3 (platform case studies) — pick your reference
3. §5.4 (Conway's Law)
4. §5.5-5.7 (org patterns, roles, DevEx)
5. §7 (governance)

### Path SDM-D: "I'm picking a thesis topic"
1. §10 — 16 concrete open problems, each SDM-scopable
2. Cross-reference DD8 for SDM methodology inheritance
3. Frame the thesis question as one of the SDM-spine layers
4. Present at CODE@MIT / MIT SDM Symposium

---

## 12. Bibliographic Posture

Every claim in this document is either:
- Cited to a published paper, standard, or vendor reference architecture
- Cited to a practitioner writeup with a named author or named firm
- Flagged as *open problem* or *gap* when no citation exists

Foundational-but-older sources (Sculley 2015, Crawley 2015, Leveson 2012, Sterman 2000, Meadows 2008, Forrester 1961, Conway 1968, CRISP-DM 1996) are explicitly labeled. Everything dated 2024-2026 reflects the current state at the repository's authoring (April 2026; knowledge cutoff January 2026).

See `docs/CATALOG.md` for paper-level lookup of every reference used here.
