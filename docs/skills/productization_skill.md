# Productization Skill — AI Product Implementation

Scope: **building the product itself**. Everything that happens between "research paper" and "customer-facing system running in production." Explicitly excludes go-to-market, fundraising, and buyer psychology (those belong to the Sales skill).

The productization skill is **three concentric layers**:

1. **Model engineering** — architectures, training recipes, evaluation regimes
2. **System engineering** — MLOps / LLMOps, RAG pipelines, agent orchestration, serving, observability, off-policy evaluation
3. **Product engineering** — PRD authoring, customer-research synthesis, PMF methodology, system architecture, commercialization framing *as it relates to product behavior*

Where the Sales skill asks *"will this convince a buyer or an investor?"*, Productization asks *"will this survive a production load, an on-call page, a team handoff, a data drift, a regulatory audit, a privacy review, and a model change?"*

---

## 1. Skill Subdivisions

```
PRODUCTIZATION SKILL
├── 2.1 Model Engineering
│     ├── Recommendation models (GNN, LLM recommenders, hybrid)
│     ├── Decision models (RL, bandits, uplift)
│     ├── Causal and Bayesian models (MMM, double ML, causal attention)
│     ├── Generative models (LLM, diffusion, multi-modal)
│     └── Representation learning (embeddings, transformers, GNNs)
│
├── 2.2 System Engineering
│     ├── MLOps / LLMOps (training, eval, serving, monitoring)
│     ├── RAG pipeline engineering
│     ├── Agent orchestration and multi-agent topology
│     ├── Off-policy evaluation and safe deployment
│     ├── Data pipelines, feature stores, catalog freshness
│     └── Privacy and governance infrastructure
│
└── 2.3 Product Engineering
      ├── PRD authoring and spec
      ├── Customer-research synthesis (JTBD extraction, interview mining)
      ├── PMF methodology for AI products
      ├── System architecture (Crawley-Cameron decomposition)
      └── Commercialization-as-product-design (not pricing / GTM)
```

Each layer has its own failure modes, review rituals, and success metrics. A mature AI marketing stack is one in which all three layers are explicitly staffed, not implicitly collapsed.

---

## 2. Model Engineering

### 2.1 Recommendation Models

**Repo coverage**:
- `docs/deep_dive/DD1_gnn_recommender_systems.md` — GraphSAGE (2017), PinSage (2018, 3B nodes / 18B edges at Pinterest), NGCF (2019), LightGCN (2020), KGAT (2019), SR-GNN (2019), 2022 ACM survey, 2025 LightGCN + KG attention
- `docs/frontier/llm_enhanced_recommenders.md` — P5 (2022), TALLRec (2023, LoRA on LLaMA-7B), LLaRA (2024, hybrid alignment), E4SRec (2023), Lost in Sequence (2025, attention-failure analysis), When Transformers Meet Recommenders (2025), Instance-Wise LoRA (2024)

**Productization-lens reading**:
- **LightGCN is the default baseline** for GNN recommenders — remove the non-linear transform, stack pure neighbor aggregation. Engineering simplicity *is* the win; don't start somewhere fancier.
- **PinSage is the scalability proof** — prove you can do neighbor sampling + minibatch GCN at 3B-node scale, and you know the ceiling.
- **TALLRec (LoRA on LLaMA-7B) is the production-affordability proof** — 1× RTX 3090 is the hardware budget, not 4× A5000. This is the engineering decision that moves LLM recommenders from research demo to shippable product.
- **Lost in Sequence (2025)** is the sobering Productization reading — attention analysis shows LLM-SRec attends to 40% of recent items, TALLRec 9.6%, LLaRA 0.3%. Using an LLM does not automatically mean better sequential behavior. Architecture diagnostics matter.

### 2.2 Decision Models — RL, Bandits, Uplift

**Repo coverage**:
- `docs/deep_dive/DD2_deep_rl_viral_marketing.md` — DGN, BiGDN, HEDRL-IM (hypergraph), BIM-DRL (fairness), RLB-DP (2017 WSDM), SS-RTB (2018 KDD, Alibaba), Multi-Agent RL for RTB (2018 CIKM), LinUCB (2010 WWW), Thompson Sampling, BanditLP (2026), Multi-Objective Bandits for Smart Tourism (2025)
- `docs/deep_dive/DD4_customer_churn_cascade.md` §7 — uplift modeling, CausalML, Gutierrez & Gerardy (2017)
- `docs/deep_dive/DD6_ai_marketing_foundations.md` §2.3 — offline RL; Levine, Kumar, Tucker & Fu (2020) tutorial

**Productization-lens reading**:
- **HEDRL-IM (hypergraph IM, 2024)** matters engineering-wise because group chats (WhatsApp, Discord, Slack) are hyperedges. Pairwise graph abstractions break; the data model in your feature store needs to represent hyperedges natively.
- **BanditLP (LinkedIn 2026)** is the canonical template for production bandits with **cadence + budget + fairness constraints** — the abstraction layer you need to build even before your first bandit ships.
- **Offline RL (Levine et al. 2020)** is the regime every production marketing RL lives in. Online exploration against real customers is rarely acceptable. Build your RL stack assuming logged data and off-policy estimation, not simulated rollout.
- **Uplift modeling + CausalML** is not optional for retention decisions — churn prediction alone cannot drive retention budget allocation.

### 2.3 Causal and Bayesian Models

**Repo coverage**:
- `docs/deep_dive/DD5_marketing_mix_modeling.md` — Jin (2017 Google), Google Meridian (2024 open source), Robyn (Meta), PyMC-Marketing (PyMC Labs), DeepCausalMMM (2025), DNAMTA (2018), CAMTA (2020), Amazon Ads MTA (2025)
- `docs/deep_dive/DD6_ai_marketing_foundations.md` — Chernozhukov et al. double ML (2018), DoubleML library, Bayesian / causal cross-mapping

**Productization-lens reading**:
- **Jin 2017's adstock and shape priors** are the engineering contract — any MMM codebase's default priors must reflect domain knowledge or the posterior is useless. Meridian, PyMC-Marketing, and Robyn all ship opinionated priors; modify them consciously.
- **Meridian is the privacy-aware default** post-cookie. Build the aggregate-data pipeline first; do not bolt it on.
- **Double ML** gives you valid inference for treatment effects when flexible ML estimates nuisance functions. The DoubleML library (Python + R) is the production bridge between modern ML and valid causal inference.

### 2.4 Generative Models

**Repo coverage**:
- `docs/frontier/generative_ai_and_llms_in_marketing.md` — Stable Diffusion CTR pipeline (ACM WebConf 2024), SD 3.5 (Oct 2024), Retail-GPT (2024), Graph-Enhanced RAG (2025), Amazon QAC RAG; GenAI marketing stats (76% use for content, +800% LLM-referral YoY)
- `docs/frontier/agentic_ai_marketing.md` — agentic adoption stats, multi-agent role specialization

**Productization-lens reading**:
- The Stable Diffusion + CTR predictor pipeline (ACM 2024) is the **closed-loop creative generation** template. Generate → filter via learned CTR / brand-safety predictor → serve top. Do not skip the filter; raw diffusion outputs have unacceptable brand-safety variance.
- **SD 3.5 Medium (2.5B params)** runs on consumer hardware, SD 3.5 Large (8.1B) targets enterprise. Your serving topology depends on this choice.
- **LLM referral traffic is growing 800% YoY (Semrush)**. SEO becomes LLM-seeing, not just search-ranking. Product content needs to be parseable by LLMs as well as humans.

### 2.5 Representation Learning (Cross-Cutting)

**Repo coverage**:
- `docs/deep_dive/DD6_ai_marketing_foundations.md` §2.4 — representation learning as the universal substrate
- `docs/deep_dive/DD3_influencer_marketing.md` — DeepInf, HetInf, DeepPP (ego-network GNN embeddings)
- `docs/deep_dive/DD4_customer_churn_cascade.md` §2 — CCP-Net (BiLSTM + CNN + Multi-Head Self-Attention)

**Productization-lens reading**: every other layer depends on this one. Bayesian inference on raw pixels is hopeless; on a 256-dim embedding it's tractable. Invest in shared representation infrastructure (a feature store, an embedding service, a pretrained-encoder library) before specializing by domain.

---

## 3. System Engineering

This layer is where the repo has its **largest gap vs. industry state-of-the-art**. The existing DDs describe algorithms; production systems live or die on plumbing.

### 3.1 MLOps / LLMOps

**New to repo via this skill axis**:

| Signal | Value | Source |
|---|---|---|
| Engineers shipping AI-incorporated products | **56%** (2025, up from 42% YoY) | Avnet 2025 engineering survey |
| Engineers citing data quality as top design challenge | **46%** | same |
| Engineers citing continuous learning / maintenance as top operational challenge | **54%** | same |
| LLM models unchanged 6+ months → error-rate increase | **+35%** (input-data drift) | ZenML 1,200 deployments analysis 2025 |
| Engineers deploying RAG in production or within 12 months | **70%** | RAGFlow 2025 |
| AI agent market size (2024 → 2030) | **$5.4B → $50.3B** | AIAccelerator Institute |

**Productization-lens reading**: the winners are not those with model access — they are those with **evaluation practices, operational discipline, and infrastructure**. Concrete implications:

- **Context engineering** emerged as a distinct discipline in 2025. Prompt libraries are necessary but insufficient.
- **MCP (Model Context Protocol)** stabilized as integration infrastructure. Production agents talk to tools through MCP, not ad-hoc.
- **Evaluation and guardrail maturation** — offline evals + granular node-level evals + automated log evals + CI/CD gates. Without CI/CD for models, you do not have LLMOps, you have LLM-ops-theater.

**New references**:
- ZenML Blog (2025). *What 1,200 Production Deployments Reveal About LLMOps in 2025* ([link](https://www.zenml.io/blog/what-1200-production-deployments-reveal-about-llmops-in-2025))
- Avnet. *AI Adoption in Engineering: 10 Key Trends* ([link](https://www.rdworldonline.com/avnet-study-engineers-are-shipping-more-ai-products-but-confidence-in-them-remains-uneven/))
- AI Accelerator Institute. *LLMOps Tools of Choice 2025* ([link](https://www.aiacceleratorinstitute.com/llmops-tools-of-choice-2025/))
- AI Accelerator Institute. *What Is LLMOps? Complete 2025 Industry Guide* ([link](https://www.aiacceleratorinstitute.com/your-guide-to-llmops/))
- Braintrust. *The 5 Best LLMOps Platforms in 2025* ([link](https://www.braintrust.dev/articles/best-llmops-platforms-2025))
- OneReach. *LLMOps for AI Agents: Monitoring, Testing & Iteration in Production* ([link](https://onereach.ai/blog/llmops-for-ai-agents-in-production/))
- Orq.ai. *LLM Product Development in 2025: The Ultimate Guide* ([link](https://orq.ai/blog/llm-product-development))
- ZenML LLMOps Database — OpenAI Forward Deployed Engineering case study ([link](https://www.zenml.io/llmops-database/forward-deployed-engineering-bringing-enterprise-llm-applications-to-production))
- ACM Computing Surveys (2025). *A Survey of AIOps in the Era of Large Language Models* ([link](https://dl.acm.org/doi/10.1145/3746635))

### 3.2 RAG Pipeline Engineering

**Repo coverage (partial)**:
- `docs/frontier/generative_ai_and_llms_in_marketing.md` §5 — Retail-GPT, Graph-Enhanced RAG, Amazon QAC RAG
- `docs/frontier/agentic_ai_marketing.md` §2.3 — RAG commerce agents

**New to repo — production RAG engineering discipline**:

Production RAG in 2025 is a multi-component system; evaluation must be **component-wise**, not end-to-end-only:

| Component | Typical metric | Failure mode |
|---|---|---|
| Retrieval | Precision@k, Recall@k | Irrelevant context → LLM grounded in noise |
| Context sufficiency | "Does the retrieved context contain the answer?" | LLM forced to hallucinate |
| Faithfulness | Grounded-answer rate | LLM ignores retrieved context |
| Answer relevance | Task-conditional quality | Right facts, wrong question |
| Hallucination rate | Faithfulness complement | User-visible fabrication |

**Production practice**: standardize offline test runs, granular node-level evals, automated log evals, CI/CD gates. Domain-specific RAG is reported to deliver **78% accuracy lift** over vanilla LLM on domain queries (Pinecone, 2025).

**New references**:
- arXiv (July 2025). *A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions* ([link](https://arxiv.org/html/2507.18910v1))
- RAGFlow. *From RAG to Context — A 2025 Year-End Review of RAG* ([link](https://ragflow.io/blog/rag-review-2025-from-rag-to-context))
- Future AGI. *RAG Evaluation Metrics Guide 2025* ([link](https://futureagi.com/blogs/rag-evaluation-metrics-2025))
- Pinecone. *RAG Evaluation: Don't Let Customers Tell You First* ([link](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/))
- Braintrust. *The 5 Best RAG Evaluation Tools in 2025* ([link](https://www.braintrust.dev/articles/best-rag-evaluation-tools))
- Maxim AI. *RAG Evaluation: A Complete Guide for 2025* ([link](https://www.getmaxim.ai/articles/rag-evaluation-a-complete-guide-for-2025/))
- Orq.ai. *Mastering RAG Evaluation: Best Practices & Tools for 2025* ([link](https://orq.ai/blog/rag-evaluation))
- Dextra Labs. *RAG Pipeline, Use Cases & Diagrams 2025* ([link](https://dextralabs.com/blog/rag-pipeline-explained-diagram-implementation/))

### 3.3 Agent Orchestration and Multi-Agent Topology

**Repo coverage**:
- `docs/frontier/agentic_ai_marketing.md` §3 — Dao (2025) MIT SDM thesis on coordination topology and adversarial resilience
- `docs/deep_dive/DD8_mit_sdm_marketing.md` §2.1, §4 — SDM system architecture methodology

**Productization-lens reading — Dao's topology result is the engineering contract**:

| Topology | Adversarial resilience | When to use |
|---|---|---|
| **Group Chat** | Highly vulnerable (errors broadcast) | Never for brand-safety-critical agents |
| **Reflexion** | Self-critique loops | When agents are smart enough to critique |
| **Crowdsourcing** | Independent voting | When you can afford k× LLM calls |
| **Blackboard** | Shared state with access control | When agents need persistent shared memory |

**Reported 11% of organizations in production use**; **38% piloting**, **30% exploring**. Gartner predicts >40% of agentic AI projects will fail by 2027 due to legacy-system integration. Responsible deployment patterns (§4 of frontier/agentic_ai_marketing):

1. Human-in-the-loop on anything externally visible
2. Read-only agents before write agents
3. Bounded budgets (token, spend, action count) per agent per window
4. Observability on agent decision traces
5. Rollback policies for agent-driven campaigns
6. Topology-aware coordination (Reflexion / Crowdsourcing / Blackboard, not Group Chat)

### 3.4 Off-Policy Evaluation and Safe Deployment

**Repo coverage**:
- `docs/deep_dive/DD6_ai_marketing_foundations.md` §4 — OPE as the quiet core of every system; IPS, Doubly Robust, SNIPS, Fitted-Q, model-based OPE
- `docs/deep_dive/DD2_deep_rl_viral_marketing.md` §6 — OPE as open problem for RL in marketing

**Productization-lens reading**: every production marketing ML team does OPE under the hood — they just may not call it that. Naming it, owning it, and instrumenting it is the move from unconscious practice to engineered practice. Doubly Robust + fitted-Q + periodic incrementality calibration is the reference stack.

### 3.5 Data Pipelines, Feature Stores, Catalog Freshness

**Engineering-critical but under-researched in academic literature**:

- **Catalog freshness** is the binding constraint for LLM recommenders (see `frontier/llm_enhanced_recommenders.md` §7.2). Product catalogs change daily; LLMs retrained monthly miss fresh items. RAG at inference time is the current pattern.
- **Feature store** is the shared substrate across recommendation, churn, MMM, and uplift. Centralizing it breaks per-team silos but requires versioning and backfill discipline.
- **Embedding services** amortize representation learning across consumers.

No single academic source dominates; reference architectures from Uber (Michelangelo), Airbnb (Zipline), LinkedIn (Feathr), Meta (FBLearner Flow) are the canon.

### 3.6 Privacy and Governance Infrastructure

**Repo coverage**:
- `docs/deep_dive/DD7_mit_marketing_research.md` §2.1 — Tucker (2014) showing privacy UX and personalization are not opposed
- `docs/deep_dive/DD5_marketing_mix_modeling.md` — Meridian as privacy-aware MMM
- `docs/frontier/generative_ai_and_llms_in_marketing.md` §4 — only 37% of marketers have AI governance clauses in vendor contracts (MarTech 2026)

**Productization-lens reading**: privacy and governance are system-level, not feature-level. They live in:
- Data-minimization at ingest (feature store contracts)
- Differential privacy on aggregations (for MMM-style aggregate reporting)
- Consent propagation through the pipeline
- Audit trails on agent decisions (see §3.3)
- Regulation-aware planning (GDPR, CCPA, DSA — open problem)

---

## 4. Product Engineering

### 4.1 PRD Authoring and Customer-Research Synthesis

**New to repo via this skill axis**:

AI-assisted product management is now commodity infrastructure for engineering-organization productivity — and particularly so for AI-product teams:

| Signal | Value |
|---|---|
| AI synthesizing 50 customer interviews | ~1 hour (vs. ~4 hours human) |
| Time savings on research synthesis via AI | ~75% (4hr → 20min) |
| Claude noted as strong for strategic / nuanced PRD output | multiple surveys |
| PM-workflow compression | Execution layer (docs, synthesis) compressed; strategic output expanded |

**Productization-lens reading**: the AI-product PM doesn't write fewer documents, but spends a different mix. JTBD extraction from interview transcripts, PRD first-draft generation, and test-plan synthesis are delegated. What remains is the *strategic* work — framing, prioritization, sequencing — which AI cannot meaningfully automate.

**New references**:
- Institute of Product Management (2025). *How to Use AI in Product Management: The Complete Guide* ([link](https://www.institutepm.com/knowledge-hub/ai-use-in-product-management))
- ChatPRD (2026). *10 Best AI Tools for Product Managers* ([link](https://www.chatprd.ai/learn/best-ai-tools-for-product-managers))
- Productboard. *AI in Product Management: 6 Prompts* ([link](https://www.productboard.com/blog/6-ai-prompt-templates-for-product-managers/))
- Dust. *AI for Product Management: How to Actually Use It* ([link](https://dust.tt/blog/ai-for-product-management))
- MindStudio. *10 AI Agents for Product Managers* ([link](https://www.mindstudio.ai/blog/ai-agents-for-product-managers))
- Ainna. *How Should Product Managers Use AI in 2026? The Methodology-Driven Guide* ([link](https://ainna.ai/resources/faq/ai-product-management))
- Alloy.app. *Cursor for Product Managers: Complete AI-Powered Product Management Guide* ([link](https://alloy.app/library/cursor-for-pms-complete-guide-ai-product-management))
- Cognitive Future. *Best AI Tools for Product Managers in 2026: Full Guide and Reviews* ([link](https://cognitivefuture.ai/best-ai-tools-for-product-managers/))

### 4.2 PMF Methodology for AI Products

**Sales-overlapping territory**: the productization view of PMF is *"does the product do the job?"* separate from *"can we sell it?"* Both matter, and the skills are complementary.

**Repo coverage**:
- `docs/deep_dive/DD8_mit_sdm_marketing.md` §2.3 — Strategic Perspective on the Commercialization of AI (MIT SDM thesis)

**Productization-lens reading (cross-link with sales_skill.md §3.4)**:
- 2025 playbook: *micro-test → traction proof → MVP-lite → GTM validation → real usage*
- The micro-test is a product-engineering instrument — not a sales tactic. Instrument product telemetry to answer: *does the user do the job the product claims to enable?* If not, no amount of sales fixes the gap.
- AI-native teams reach PMF with smaller teams via **product-internal automation** — LLM-in-the-loop PM, LLM-assisted eng, LLM-assisted design. This changes the staffing model, not just the tool chain.

### 4.3 System Architecture (Crawley-Cameron)

**Repo coverage**:
- `docs/deep_dive/DD8_mit_sdm_marketing.md` §4.1 — Crawley, Cameron, Selva (2015) Systems Architecture textbook; decomposition, interface analysis, architecture evaluation
- `docs/deep_dive/DD8_mit_sdm_marketing.md` §4.2 — Moser agent-based modeling for socio-technical systems

**Productization-lens reading**: decomposition (data pipeline → feature store → models → decision engine → experimentation → reporting) is the common skeleton of every AI marketing stack. The SDM discipline is teaching yourself to see the skeleton when it's hidden inside a 200-notebook Jupyter mess. Interface clarity and feedback-loop analysis matter most when the model works in a notebook but fails at the team-handoff boundary.

### 4.4 Commercialization as Product Design

**Narrow scope here**: how the product *behaves* to make adoption plausible — not pricing, not GTM (those are in Sales).

Examples:
- Meridian (open source 2024) is a commercialization-design choice by Google: the product is harder to resist adopting when it is a well-maintained OSS library.
- PyMC-Marketing's docs are a commercialization-design choice by PyMC Labs: the OSS-first funnel compresses the buyer path.
- LangChain / LlamaIndex's ecosystem strategy is commercialization-as-architecture: win the integration graph, win the category.

This is *not* the investment narrative (which is Sales §3.4). It is the product-internal decisions that *enable* that narrative.

---

## 5. Cross-Layer Integration — A Unified Productization Stack

Borrowing the DD6 unified-framework picture and reading it through the productization lens:

```
[Product Layer]
  PRD + JTBD + PMF instrumentation + system architecture
    ↓
[System Layer]
  Feature store + Embedding service + MLOps + LLMOps + RAG + Agent orchestration + OPE
    ↓
[Model Layer]
  Representation (GNN / Transformer / Diffusion) + Decision (RL / Bandit / Uplift)
    + Causal (Double ML / Causal Attention) + Bayesian (MMM / Hierarchical)
    ↓
[Output]
  A system the engineer on call can debug at 2 AM
  A product the PM can extend without a rewrite
  A stack the CTO can present as "production maturity" to an investor
```

The last three output bullets are the productization acceptance criteria. A paper is not a product until all three hold.

---

## 6. Open Problems — Productization Axis

1. **Catalog-freshness RAG vs. fine-tune tradeoff**. When does retraining an LLM beat RAG on freshness? Cost models are anecdotal.

2. **Multi-agent OPE**. Single-agent OPE (DR-IPS, fitted-Q) does not extend cleanly to coordinated agents. A multi-agent marketing stack today cannot be safely evaluated from logs alone.

3. **Evaluation at generative scale**. Millions of LLM-generated creatives, RAG answers, or agent decisions per day cannot be human-reviewed. Learned evaluators (quality predictors, brand-safety classifiers) are the bottleneck, and their own evaluation is open.

4. **Agent cost-aware planning**. Naive agent loops burn tokens. Cost-aware planning (when to call the LLM, when to skip) is unsolved at scale.

5. **Regulatory compliance automation**. Agents that verify their own actions comply with GDPR / CCPA / DSA / emerging AI-disclosure rules. Policy-aware planning is unsolved.

6. **LLM + causal ML integration**. LLMs generate content, classical causal ML assigns treatments. Integrating causal uplift directly with LLM content generation is an open frontier (DD6 Gap 5).

7. **Production-grade joint content + targeting optimization**. Two systems today; one problem operationally. Bandits over generated creatives is emerging but not solved.

8. **Platform dependency and model-provider risk**. When a production product depends on a closed LLM that may change weights or deprecate, what's the mitigation stack? Model-agnostic product design is an open discipline.

9. **Observability for agent decisions**. When a multi-agent system makes a bad recommendation, how do you trace through the coordinated decision to find the root cause? Current tools are prompt-trace at best.

10. **Counterfactually fair uplift / bandit algorithms**. Fairness under causal constraints remains theoretical; production implementations are sparse (DD6 Gap 2).

---

## 7. Back-References into Existing Deep Dives

| Sub-skill | Primary DD | Supporting DD |
|---|---|---|
| Recommendation models | DD1, frontier/llm_enhanced_recommenders | `docs/02` |
| Decision models | DD2, DD4 §7 | DD6 §2.3 |
| Causal / Bayesian | DD5, DD6 | — |
| Generative | frontier/generative_ai_and_llms_in_marketing | — |
| Representation | DD6 §2.4 | DD1, DD3, DD4 |
| MLOps / LLMOps | — | New refs in §3.1 |
| RAG engineering | frontier/generative §5, frontier/agentic §2.3 | New refs in §3.2 |
| Agent topology | frontier/agentic §3 | DD8 §2.1 (SDM) |
| Off-policy evaluation | DD6 §4 | DD2 §6 |
| Data / feature / catalog | (adjacent) | frontier/llm_enhanced §7.2 |
| Privacy / governance | DD7 §2.1, DD5, frontier/generative §4 | — |
| Product engineering | DD8 §2.3, §4.1 | New refs in §4 |

---

## 8. Further Reading

- `docs/skills/sales_skill.md` — the complementary lens
- `docs/deep_dive/DD6_ai_marketing_foundations.md` — the theoretical spine for the model layer
- `docs/deep_dive/DD8_mit_sdm_marketing.md` — the system-architecture spine for the system layer
- `docs/CATALOG.md` — paper-level index; new references from this doc are back-listed under *LLMOps & RAG Engineering*, *AI Product Management*, and *AI Commercialization* sections.
