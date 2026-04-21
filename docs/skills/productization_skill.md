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

### 2.6 Foundation Model Selection for Marketing Workloads

A new productization sub-skill: choosing the right foundation model per workload and managing multi-model stacks. The market rebalanced in 2024-2026 and engineering decisions now carry strategic weight.

**Market state (2025 mid-year, Menlo Ventures)**:
- **Anthropic: 32% enterprise AI market share**, ahead of OpenAI and Google (each ~20%)
- Model API spending **more than doubled** from $3.5B to $8B of generative AI spend in the period covered
- Enterprises are building **vendor-agnostic architectures** to avoid lock-in and capture rapid-innovation gains across providers

**Selection dimensions that actually drive production decisions**:

| Dimension | Why it matters |
|---|---|
| **Multimodal coverage** | Ad-creative QA and conversational-commerce often need image + text reasoning (Claude Vision, GPT-4V/5, Gemini 1.5+, Pixtral) |
| **Context window** | Long-context personalization and catalog-grounded reasoning; 200K+ tokens becomes table stakes |
| **Cost tiering** | Opus/Sonnet/Haiku-style families let you route small tasks to cheap models; this is now the primary serving-architecture decision |
| **Compliance** | SOC 2 Type II, GDPR, HIPAA — for enterprise marketing, these are filter gates, not tiebreakers |
| **Open weights availability** | Self-hosting LLaMA-3 / Mixtral / Qwen / DeepSeek for data-sovereignty or cost at high utilization |
| **Fine-tuning / LoRA support** | Brand-voice specialization; TALLRec-style adaptation (see frontier/llm_enhanced_recommenders) |

**Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5 reference** (per system prompt for this repo, Jan 2026 cutoff): canonical Anthropic family as of repo authoring. For builds-against-the-API decisions, see [claude-api skill](https://docs.anthropic.com) and current Anthropic API docs.

**Key references**:
- Menlo Ventures. *2025 Mid-Year LLM Market Update: Foundation Model Landscape + Economics* ([link](https://menlovc.com/perspective/2025-mid-year-llm-market-update/))
- Sebastian Raschka. *The State of LLMs 2025: Progress and Predictions* ([link](https://magazine.sebastianraschka.com/p/state-of-llms-2025))
- Helicone. *Complete LLM Model Comparison Guide 2025* ([link](https://www.helicone.ai/blog/the-complete-llm-model-comparison-guide))
- LLM Stats. *AI Leaderboard 2026 — Compare Top AI Models & Rankings* ([link](https://llm-stats.com))
- LXT. *LLM Benchmarks in 2026* ([link](https://www.lxt.ai/blog/llm-benchmarks/))
- Empler. *The Ultimate Guide to the Latest LLMs (2025)* ([link](https://www.empler.ai/blog/the-ultimate-guide-to-the-latest-llms-a-detailed-comparison-for-2025))
- SnapLogic. *The Great LLM Race: Enterprise AI* ([link](https://www.snaplogic.com/blog/great-llm-race-enterprise-ai))
- ArtificialAnalysis. *Throughput / price leaderboard* ([link](https://artificialanalysis.ai/)) — continuously updated

**Productization-lens reading**: in 2026, the right answer is almost never "one model." It is a model router (Haiku-class for classification and lightweight summarization, Sonnet-class for most conversational work, Opus-class for complex reasoning / strategy), backed by Bedrock / Azure / Vertex for managed paths and vLLM / SGLang for self-host. See §3.2 for the serving-layer implications.

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

### 3.1.1 Vector Database Layer (Foundation for RAG and Retrieval)

The retrieval substrate underneath every RAG pipeline. For a marketing-scale workload (tens of millions of items, sub-100ms P95 retrieval, hybrid dense+sparse+metadata filter, minute-cadence catalog freshness):

| Dimension | Pinecone | Weaviate | Qdrant | Milvus | Chroma | pgvector | OpenSearch | Elastic | Atlas Vector | Redis |
|---|---|---|---|---|---|---|---|---|---|---|
| Deploy | SaaS | SaaS+OSS | SaaS+OSS | SaaS+OSS | OSS/Cloud | Self-host | AWS/OSS | SaaS+OSS | SaaS | SaaS+OSS |
| Hybrid (BM25+dense) | Sparse-dense | Native | Native+payload | Native | Weak | Manual | Strong | **Strongest** | Moderate | Moderate |
| Metadata filter perf | **Excellent** (serverless) | Good | Excellent (quantized) | Good | Poor at scale | Good (IVFFlat/HNSW) | Excellent | Excellent | Good | Moderate |
| Incremental updates | Streaming | Streaming | Streaming | Batch-heavy | Limited | Live | Near real-time | Near real-time | Live | Live |
| Cost / query at 50M | Low (serverless) | Medium | **Low (binary quant)** | Low self-host | N/A prod | Lowest if on PG | Medium-high | High | Medium | High RAM |
| Ops simplicity | **Highest** (managed) | Medium | Medium | Low (operator) | Dev-grade | **Highest** (existing PG) | Medium | Medium | High (Atlas) | Medium |

**When to pick which (marketing workload)**:
- **Pinecone serverless**: lowest ops burden; pay-per-query beats provisioned for spiky campaign bursts. 2024 serverless redesign decoupled storage/compute.
- **Qdrant**: best cost-per-query at scale due to binary/scalar quantization and Rust core; strongest for dense + payload filtering.
- **pgvector + pgvectorscale** (Timescale): wins when catalog already lives in Postgres (most CDP/CRM stacks). StreamingDiskANN (2024) closed the gap with dedicated DBs to ~100M.
- **OpenSearch / Elastic**: wins on true hybrid (BM25 + ELSER/SPLADE + dense). Best where lexical matters (brand names, SKUs, exact-match titles).
- **Milvus**: highest raw throughput at 100M+ but operationally heaviest. Pick only for pure-scale workloads.
- **Chroma / Redis**: dev and cache layer respectively; not primary prod at tens of millions.

**Key references**:
- [Pinecone Serverless architecture (2024)](https://www.pinecone.io/blog/serverless/)
- [Qdrant Binary Quantization](https://qdrant.tech/articles/binary-quantization/)
- [Qdrant Benchmarks](https://qdrant.tech/benchmarks/)
- [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained)
- [Milvus 2.4 / 2.5 GPU indexes](https://milvus.io/blog/milvus-2-4-release.md)
- [pgvector (github)](https://github.com/pgvector/pgvector)
- [Timescale — pgvectorscale benchmark 2024](https://www.timescale.com/blog/pgvector-vs-pinecone/)
- [OpenSearch neural + hybrid](https://opensearch.org/blog/semantic-search-solutions/)
- [Elastic ELSER v2 learned sparse encoder](https://www.elastic.co/search-labs/blog/elser-v2-elastic-learned-sparse-encoder)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/developer/products/atlas/atlas-vector-search-overview/)
- [Redis Vector Search](https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/)
- [ANN-Benchmarks](https://ann-benchmarks.com/) — ongoing
- [DiskANN / SPFresh fresh-index updates (arXiv 2024)](https://arxiv.org/abs/2401.02116)
- [HN: Why pgvector Is Enough (2024 discussion)](https://news.ycombinator.com/item?id=39549831)

**Open problems**: (a) hybrid reranking latency budget — dense + SPLADE + reranker still ~150-250ms, pushing sub-100ms P95 onto the ANN layer only; (b) multi-tenant filter pushdown cost on serverless tiers is opaque; (c) graph-aware retrieval (for recommendation-style queries) is not standardized across vendors.

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

#### 3.3.1 Orchestration Framework Comparison (2024-2026)

The 4-agent marketing stack (Lead Qualification + Creative Generation + Brand Guardian + Campaign Orchestrator) has measurable framework trade-offs:

| Framework | Topology | Tool reliability | Observability | State / memory | Prod track record | Cost-aware |
|---|---|---|---|---|---|---|
| **LangGraph** (+ LangSmith) | Graph (stateful DAG) — supports all 4 Dao topologies | High; MCP first-class | LangSmith native (best-in-class) | Checkpointing, time-travel, durable | Klarna, Uber, Replit, LinkedIn | Manual via routing nodes |
| **CrewAI** | Role-based Group Chat / Crew | Medium; function-call wrappers | Basic; OTel via plugins | Task-output passing, limited persistence | ~60% of Fortune 500 claimed; fast POC | Weak |
| **AutoGen / AG2** → MS Agent Framework | Conversational Group Chat (vulnerable per Dao) → graph+conversational hybrid | Medium → High | OTel; Studio UI | In-memory → session state + middleware | Microsoft Research; sunsetting into **MS Agent Framework 1.0 (GA April 2026)** | Poor (5-6× tokens of LangGraph) → improving |
| **DSPy** | Module composition (not orchestration) | N/A (signatures) | MLflow integration | N/A | Databricks, JetBlue, VMware | Optimizer-driven |
| **LlamaIndex** | Event-driven Workflows | Medium (RAG-centric) | LlamaTrace, OTel | Good for retrieval state | Strong in RAG-heavy apps | Weak |
| **Haystack 2.x** | Pipeline (DAG, Blackboard-like) | High; explicit routing | Native tracing; Enterprise Platform | Pipeline state | Airbus, NVIDIA; regulated sectors | Manual |
| **Pydantic AI 1.0** (Sep 2025) | Single-agent + graph primitives | **Very high** (type-safe) | Logfire native (best type-level) | Durable execution added late 2025 | Bedrock AgentCore; growing | Logfire cost tracking |
| **OpenAI Agents SDK** (Mar 2025) | Handoffs (lightweight Group Chat) | High; guardrails built-in | Built-in tracing | Thread-based | OpenAI-native shops | Tied to OpenAI routing |

**When each wins for a marketing stack**:
- **LangGraph + LangSmith**: default for 4-agent marketing. Explicit graph maps cleanly onto Dao's Reflexion / Blackboard topologies. Brand Guardian becomes a verification node. Best observability for CRM / ad-API writes.
- **Pydantic AI**: wins when Creative Generation outputs must be schema-validated (ad-copy structs, UTM params, brand-constrained JSON). Pair with LangGraph.
- **CrewAI**: wins only for rapid POC — its role metaphor ("marketer, copywriter, editor") accelerates stakeholder buy-in, but Group Chat topology is the most prompt-injection-vulnerable per Dao 2025.
- **Microsoft Agent Framework**: wins when the stack lives in Dynamics 365 / M365 Copilot; 1.0 GA (April 2026) gives SLAs and .NET parity.
- **Haystack**: wins for Brand Guardian specifically — pipeline model supports Blackboard-style verification with explicit routing, popular in regulated industries.

**DSPy — the right abstraction at a different layer**: DSPy is production-ready as a **prompt compiler**, not an orchestrator. Databricks case studies (MLflow integration, JetBlue) show 2× deployment speedup. Signatures + MIPROv2 optimization eliminate brittle prompt-chaining. Production pattern: **DSPy compiles the prompts; LangGraph orchestrates the agents**. Treat it as the "TensorFlow of prompts" sitting under the orchestrator.

**Key references**:
- Dao 2025 MIT SDM thesis — [DSpace PDF](https://dspace.mit.edu/bitstream/handle/1721.1/162506/dao-daoluc-sm-sdm-2025-thesis.pdf) — already in DD8
- [SDM CAS 2025 Best Paper](https://sdm.mit.edu/research-practice/dao-best-paper-cas-2025/)
- [AgentDojo benchmark (arXiv 2024)](https://arxiv.org/abs/2406.13352)
- [Multi-Agent Design: Prompts + Topologies (arXiv 2025)](https://arxiv.org/html/2502.02533v1)
- [Multi-Agent Defense Pipeline vs Prompt Injection (arXiv 2025)](https://arxiv.org/html/2509.14285)
- [Prompt Injections to Protocol Exploits (arXiv 2025)](https://arxiv.org/html/2506.23260v1)
- [Budget-Aware Tool-Use Scaling (arXiv 2025)](https://arxiv.org/abs/2511.17006)
- [CostBench (arXiv 2025)](https://arxiv.org/html/2511.02734v1)
- [Prompts as Code — DSPy case studies (arXiv 2025)](https://arxiv.org/html/2507.03620v1)
- [Pydantic AI docs](https://ai.pydantic.dev/)
- [MS Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [MS Agent Framework launch](https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/)
- [LangGraph production](https://www.langchain.com/langgraph)
- [DSPy MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/)
- [Haystack 2.x](https://haystack.deepset.ai/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)

**Open problems beyond Dao's topology result**:
1. **Topology-injection coupling**: no framework auto-selects topology based on threat model
2. **Cost-aware planning immature**: only BATS / TALE research; no framework natively implements Pareto-optimal token budgets for marketing KPIs (CAC, CPM)
3. **CRM write idempotency**: ad-platform APIs (Meta, Google Ads) and CRM writes lack universal replay / rollback semantics across frameworks
4. **Evaluation gap**: AgentDojo covers generic prompt injection; no marketing-specific benchmark for brand-safety or hallucinated claims
5. **Framework consolidation**: MS Agent Framework merger (Oct 2025 — Apr 2026) signals maturation; expect 2026-2027 consolidation around **LangGraph, MS Agent Framework, Pydantic AI** as the production triad

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
- Regulation-aware planning (GDPR, CCPA, DSA, NIST AI RMF, EU AI Act, FTC Operation AI Comply, ISO 42001, COPPA 2025 amendments, Colorado AI Act Feb 2026, California AB 2013)

For the regulatory frame in full, see the **cross-axis appendix** (`docs/skills/regulatory_appendix.md`).

### 3.7 LLM Serving and Inference Infrastructure

The layer between a trained / loaded model and customer-facing latency. For marketing-scale QPS on LLaMA-3-70B / Mixtral-8×22B or the Anthropic / OpenAI hosted families:

| Dimension | vLLM | TGI | TensorRT-LLM | SGLang | Ollama | Bedrock | Azure OpenAI | Vertex |
|---|---|---|---|---|---|---|---|---|
| Continuous batching | **Yes (originator)** | Yes | Yes | Yes | No | Managed | Managed | Managed |
| PagedAttention / KV reuse | **Yes (native)** | Yes | Yes | **RadixAttention (best for prefix reuse)** | Limited | Opaque | Opaque | Opaque |
| Speculative decoding | Yes (Medusa / EAGLE) | Yes | **Yes (best-in-class)** | Yes | No | Model-dep | Yes (GPT-4o) | Yes |
| Structured output / JSON | Guided decoding | Yes | Yes | **Strongest (RadixAttn + regex)** | Weak | Yes (tool use) | Yes | Yes |
| Throughput rank (H100, 70B) | 2 | 3 | **1** (when tuned) | 2 (prefix-heavy) | N/A | N/A | N/A | N/A |
| Multi-tenant isolation | Manual | Manual | Manual | Manual | No | **Strong** | **Strong** | **Strong** |
| Cost / 1M tokens (70B self-host H100) | ~$0.20-0.40 | similar | **lowest** | similar | N/A | $2.65 (Llama-3-70B) | N/A direct | ~$2.00 |
| Ops burden | Medium | Medium | **High** (engine build) | Medium | Trivial dev | None | None | None |

**When to pick which**:
- **vLLM**: default self-hosted choice 2024-2026; widest model coverage, solid throughput, community velocity
- **TensorRT-LLM**: NVIDIA fleet + peak tokens/sec (>1.5-2× vLLM on tuned workloads). Engine compilation and versioning are painful
- **SGLang**: wins for **agentic marketing workloads with heavy prefix reuse** (same system prompt + persona across millions of personalized generations) thanks to RadixAttention; strongest constrained decoding
- **TGI**: still competitive; tightly integrated with HF ecosystem; lagged vLLM on features since mid-2024
- **Managed (Bedrock / Azure / Vertex)**: compliance, regional residency, or unpredictable traffic. Break-even vs self-host ~30-50% sustained GPU utilization for 70B-class

**Key references**:
- [vLLM repo](https://github.com/vllm-project/vllm)
- [PagedAttention paper (arXiv 2023)](https://arxiv.org/abs/2309.06180)
- [SGLang Llama-3 serving (lmsys 2024)](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
- [SGLang / RadixAttention paper (arXiv)](https://arxiv.org/abs/2312.07104)
- [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/)
- [TGI repo](https://github.com/huggingface/text-generation-inference)
- [Anyscale — continuous batching explainer](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Medusa speculative decoding (arXiv 2024)](https://arxiv.org/abs/2401.09670)
- [EAGLE-2 speculative decoding (arXiv 2024)](https://arxiv.org/abs/2406.16858)
- [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
- [Azure OpenAI Provisioned Throughput](https://learn.microsoft.com/azure/ai-services/openai/concepts/provisioned-throughput)
- [Vertex GenAI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [ArtificialAnalysis leaderboard](https://artificialanalysis.ai/)
- [SGLang repo](https://github.com/sgl-project/sglang)

**Open problems**: (a) KV-cache reuse across requests in multi-tenant environments is a security / isolation concern; (b) speculative decoding gains collapse on high-entropy marketing copy; (c) no clean standard for hybrid routing (small-model-first cascades); (d) per-tenant fine-tune serving (LoRA swapping) still custom work outside S-LoRA / vLLM multi-LoRA.

**Cross-cutting observation with §3.1.1**: the productization bottleneck for AI marketing is rarely raw model quality — it is the **retrieval-generation coupling** (vector DB + KV-cache reuse) that determines per-impression cost. Stacks that co-design these (e.g., SGLang RadixAttention + Qdrant payload filters, or Bedrock Knowledge Bases) will dominate unit economics in 2026.

### 3.8 Observability and Tracing for LLM / Agent Systems

| Tool | Traces | Prompt / Response Log | Drift | Cost / Token | Feedback |
|---|---|---|---|---|---|
| **Arize AX** | Yes (OTel) | Yes | **Strong** (embedding + tabular) | Yes | Yes (annotations) |
| **WhyLabs** | Partial | Yes | Strong (whylogs profiles) | Limited | Via integrations |
| **Evidently** | Partial | Yes | Strong (text drift reports) | Limited | Custom |
| **Fiddler** | Yes | Yes | **Strong (LLM + ML unified)** | Yes | Yes |
| **Langfuse** | Yes (OTel) | Yes | Emerging | **Yes (native)** | Yes (scores API) |
| **Phoenix (OSS)** | Yes (OTel / OpenInference) | Yes | Yes (embedding) | Yes | Yes |
| **LangSmith** | Yes | Yes | Limited | Yes | **Yes (native)** |
| **Traceloop / OpenLLMetry** | Yes (OTel standard) | Yes | Via exports | Yes | Via Traceloop |
| **Helicone** | Yes (proxy + async) | Yes | Limited | **Yes (strength)** | Yes |
| **W&B Weave** | Yes | Yes | Via W&B Models | Yes | Yes |

**Selection criteria**:
- OpenTelemetry-compatible (OpenInference-aligned) is the 2025 default to avoid vendor lock-in: **Phoenix, Langfuse, Traceloop** lead
- Marketing creative pipelines need **multimodal traces** (image + text) — Arize AX and Langfuse added image logging 2024-2025; Helicone lags
- Cost governance at scale: **Helicone, Langfuse** for per-user / per-feature token attribution
- Drift for marketing: embedding drift on customer intent is the real signal; Arize, Phoenix, WhyLabs, Evidently cover this best
- Self-host: Langfuse (MIT-ish), Phoenix (ELv2), Helicone (Apache) are genuinely self-hostable; LangSmith and Arize AX are SaaS-primary

**Key references**:
- [Arize AX docs](https://arize.com/docs/ax)
- [Arize Phoenix (OSS)](https://docs.arize.com/phoenix)
- [Langfuse docs](https://langfuse.com/docs)
- [LangSmith docs](https://docs.smith.langchain.com/)
- [Helicone docs](https://docs.helicone.ai/)
- [Traceloop / OpenLLMetry](https://www.traceloop.com/docs/openllmetry)
- [W&B Weave](https://weave-docs.wandb.ai/)
- [WhyLabs LLM monitoring](https://docs.whylabs.ai/docs/large-language-model-monitoring/)
- [Evidently docs](https://docs.evidentlyai.com/)
- [Fiddler LLM monitoring](https://docs.fiddler.ai/product-guide/llm-monitoring)
- [OpenTelemetry GenAI conventions](https://opentelemetry.io/blog/2024/otel-generative-ai/)
- [OpenInference spec](https://github.com/Arize-ai/openinference)
- [LLM observability survey (arXiv 2024)](https://arxiv.org/abs/2404.00903)
- [Thoughtworks Tech Radar — Langfuse](https://www.thoughtworks.com/radar/tools/langfuse)

**Open problems**: multi-agent trace causality (whose fault when Agent B hallucinates from Agent A's retrieval?); no standard for tool-call-graph visualization; drift detection on generated-creative quality lacks ground truth.

### 3.9 Evaluation Frameworks for RAG and Agents

| Tool | Retrieval | Faithfulness | Hallucination | Task Scorers | Red-Team | CI/CD |
|---|---|---|---|---|---|---|
| **Ragas** | **Strong** (context P/R, sufficiency) | Yes | Yes | Custom | No | Yes |
| **TruLens** | Yes (triad) | Yes (groundedness) | Yes | Yes | No | Yes |
| **DeepEval** | Yes | Yes | Yes (G-Eval) | **Rich library** | Partial | **Pytest-native** |
| **OpenAI Evals** | Custom | Custom | Custom | Template-based | Limited | Yes |
| **Promptfoo** | Limited | Via LLM-as-judge | Yes | Yes | **Strong (red-team)** | Yes (YAML) |
| **LangSmith Evaluate** | Yes | Yes | Yes | Yes | Limited | Yes |
| **Phoenix Evals** | Yes | Yes | Yes | Pre-built | Limited | Yes |
| **Giskard** | Yes | Yes | Yes | Yes | **Strong (scan)** | Yes |

**Selection criteria**:
- **RAG default 2025**: Ragas + Phoenix Evals; Ragas v0.2 added async + multimodal
- **Agent trajectories** need tool-call correctness / plan adherence — DeepEval and LangSmith lead; Ragas weak here
- **Red-teaming marketing outputs** (brand harm, competitor disparagement): Promptfoo `redteam` and Giskard scan are the only structured options
- **LLM-as-judge bias**: position, verbosity, self-preference are well-documented — pair with human spot-checks and use multiple judges (Prometheus-2, G-Eval)
- **CI/CD gating**: Promptfoo and DeepEval integrate cleanest with GitHub Actions; treat eval thresholds as release gates

**Key references**:
- [Ragas docs](https://docs.ragas.io/) · [Ragas paper (arXiv 2023)](https://arxiv.org/abs/2309.15217)
- [TruLens](https://www.trulens.org/)
- [DeepEval / Confident AI](https://docs.confident-ai.com/)
- [OpenAI Evals repo](https://github.com/openai/evals)
- [Promptfoo docs](https://www.promptfoo.dev/docs/)
- [LangSmith Evaluate](https://docs.smith.langchain.com/evaluation)
- [Arize Phoenix Evals](https://arize.com/docs/phoenix/evaluation/llm-evals)
- [Giskard docs](https://docs.giskard.ai/)
- [LLM-as-Judge / MT-Bench (arXiv 2023, foundational)](https://arxiv.org/abs/2306.05685)
- [Prometheus 2 (arXiv 2024)](https://arxiv.org/abs/2405.01535)
- [G-Eval (arXiv 2023, foundational)](https://arxiv.org/abs/2303.16634)
- [RAG evaluation survey (arXiv 2024)](https://arxiv.org/abs/2410.05983)
- [Eugene Yan — LLM evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- [Hamel Husain — Evals](https://hamel.dev/blog/posts/evals/)

**Open problems**: multi-agent off-policy evaluation — no accepted metric for counterfactual agent trajectories; judge-model drift when upgrading scorer LLM invalidates historical scores; evaluating creative "brand fit" remains subjective.

### 3.10 Guardrails and Safety

| Tool | Prompt Injection | PII | Topical Drift | Brand Voice | Jailbreak |
|---|---|---|---|---|---|
| **NeMo Guardrails** | Yes (Colang rails) | Via integrations | **Strong** (dialog rails) | Yes (Colang flows) | Yes |
| **Guardrails AI** | Yes (validators) | Yes (Presidio) | Yes | Custom validators | Yes |
| **Lakera Guard** | **Yes (flagship)** | Yes | Partial | Limited | **Yes (flagship)** |
| **Protect AI Rebuff** | **Yes (flagship)** | No | No | No | Yes |
| **Azure Content Safety** | Yes (Prompt Shields) | Yes | Yes (groundedness) | Limited | Yes |
| **Bedrock Guardrails** | Yes | Yes | Yes (denied topics) | Yes (word filters) | Yes |

**Selection criteria**:
- **Cloud-native marketing stacks**: start with Bedrock or Azure guardrails (policy-as-config, audit logs) and layer **Lakera** for prompt-injection depth
- **Brand-voice enforcement is underserved** — most tools ship banned-word lists; real brand consistency needs fine-tuned classifiers or Guardrails AI custom validators with golden examples
- **NeMo Guardrails Colang 2.0 (2024)** is richest for conversational flow control; steep learning curve
- **PII**: Microsoft Presidio is the de facto backend; most guardrail tools wrap it
- **Latency**: inline guardrails add 100-500ms; async / sampled checks reduce but weaken enforcement

**Key references**:
- [NeMo Guardrails docs](https://docs.nvidia.com/nemo/guardrails/) · [repo](https://github.com/NVIDIA/NeMo-Guardrails)
- [Guardrails AI docs](https://www.guardrailsai.com/docs) · [Guardrails Hub](https://hub.guardrailsai.com/)
- [Lakera Guard](https://www.lakera.ai/lakera-guard)
- [Protect AI Rebuff](https://github.com/protectai/rebuff)
- [Azure Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/)
- [AWS Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [OWASP Top 10 for LLM Apps](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Prompt injection taxonomy (arXiv 2024)](https://arxiv.org/abs/2403.06833)
- [HarmBench (arXiv 2024)](https://arxiv.org/abs/2402.04249)
- [Lakera Gandalf red-team playground](https://gandalf.lakera.ai/)
- [JailbreakBench (arXiv 2023, foundational)](https://arxiv.org/abs/2310.03684)
- [Microsoft Presidio](https://github.com/microsoft/presidio)
- [Prompt-injection defense survey (arXiv 2024)](https://arxiv.org/abs/2406.05498)

**Open problems**: red-team-vs-defense arms race — new jailbreaks (many-shot, multilingual, encoded) break defenses within weeks; no standard for measuring brand-voice compliance; guardrails can't detect subtle marketing harms (dog-whistles, culturally-specific offense) without localized training data; multimodal guardrails (image + text creatives) are immature.

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

### 4.5 Human-AI Interaction (HAI) and UX for AI Marketing Products

AI-marketing products have distinctive UX obligations — disclosure, confidence display, graceful degradation, feedback capture — that traditional UX disciplines did not anticipate. Two widely-adopted frameworks anchor this:

**Google People + AI (PAIR) Guidebook**:
- Covers the mixed technical / UX concerns specific to AI products; 2024-2025 updates integrated GenAI-specific UX patterns (confidence display for non-deterministic outputs, explain-ability in chat form factors, repair flows after a bad answer)
- [PAIR Guidebook home](https://pair.withgoogle.com/guidebook/) · [Chapter index](https://pair.withgoogle.com/guidebook/chapters) · [GenAI update blog post](https://medium.com/people-ai-research/updating-the-people-ai-guidebook-in-the-age-of-generative-ai-cace6c846db4)
- [Google Design — People + AI Research library](https://design.google/library/people-ai-research)

**Microsoft HAX Toolkit (Human-AI eXperience)**:
- **18 Guidelines for Human-AI Interaction** — evidence-based best practices derived from Microsoft Research validation studies
- **HAX Workbook** — facilitates cross-functional alignment (engineering + design + product) on AI-UX decisions
- [HAX Toolkit home](https://www.microsoft.com/en-us/haxtoolkit/) · [Microsoft Research project page](https://www.microsoft.com/en-us/research/project/hax-toolkit/) · [Guidelines page](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/) · [Workbook](https://www.microsoft.com/en-us/haxtoolkit/workbook/)
- [Practitioner synthesis guide (Medium, 2025)](https://medium.com/@farzinraisstousi/the-ultimate-guide-to-ai-design-frameworks-in-2025-choosing-the-right-approach-for-your-team-5ae06336dc58)

**Marketing-UX specific concerns that neither framework covers completely**:
- Disclosure of AI-generated ad creative (EU AI Act transparency obligations become binding Aug 2026)
- Brand-voice consistency UX signals (when an LLM outputs mid-brand, the end user doesn't see the drift — the brand team does, weeks later)
- Confidence-display patterns for recommendations (when to show a probability, when to show none, when to show alternatives)
- Feedback-capture UX for thumbs-up/down on AI decisions — necessary for the feedback loop that LLMOps (§3.1) and Evaluation (§3.9) depend on

### 4.6 AI Product Analytics and Telemetry

**PostHog vs. Amplitude vs. traditional analytics** for AI products:

| Signal | PostHog | Amplitude | Mixpanel | Heap |
|---|---|---|---|---|
| LLM analytics (tokens, cost, latency, trace) | **Native (2024-2025)** | Absent | Absent | Absent |
| Error tracking | **Native** | Absent | Absent | Absent |
| Experimentation / feature flags | Native | Separate product | Separate product | Separate |
| Pricing model | Usage-based OSS + cloud | Seat-based | Seat-based | Event-based |
| Marketing analytics (attribution) | Good | Strong | Strong | Strong |

**PostHog's 2025 positioning**: "one platform for everything" — LLM tracking (tokens, cost, latency, full-trace), product analytics (funnels, cohorts, retention), feature flags, experiments, session replay. The pitch: **stop buying Langfuse + Amplitude + Sentry + LaunchDarkly**, consolidate.

**Marketing-specific telemetry patterns** that differ from typical product telemetry:
- Prompt-level analytics: same prompt yields different responses — "usage" is richer than event count
- Cost telemetry per user / session / feature: marketing uses often drive token burn disproportionately (long conversations, high-intent personalization)
- Feedback loops: explicit thumbs-up/down + implicit signals (session length, follow-up purchase) feed §3.8 observability and §3.9 eval

**Key references**:
- [PostHog LLM Analytics docs](https://posthog.com/docs/llm-analytics) · [LLM Observability handbook](https://posthog.com/handbook/growth/use-case-selling/ai-llm-observability)
- [PostHog vs Amplitude comparison](https://posthog.com/blog/posthog-vs-amplitude)
- [PostHog getting started with LLM analytics](https://posthog.com/docs/llm-analytics/start-here)
- [PostHog AI engineering docs](https://posthog.com/docs/ai-engineering)
- [LiteLLM + PostHog integration](https://docs.litellm.ai/docs/observability/posthog_integration)
- [Comprehensive Guide — Top Open-Source LLM Observability Tools in 2025 (Dev.to)](https://dev.to/practicaldeveloper/comprehensive-guide-top-open-source-llm-observability-tools-in-2025-1kl1)
- [LLM metrics template — PostHog](https://posthog.com/templates/ai-analytics)

### 4.7 Feature Flagging and AI Rollout

Feature flagging has become a strategic layer for AI products specifically:

**The landscape (2026)**:

| Platform | Focus | AI-era signal |
|---|---|---|
| **Statsig** | Data-first experimentation + flags | **Acquired by OpenAI for $1.1B (2026)** — signals that feature flags are strategic to AI-era product monetization |
| **LaunchDarkly** | Workflow-first, enterprise compliance | Still leader on audit / approval rigor; experimentation is a recent add-on |
| **GrowthBook** | Open-source experimentation + flags | Warehouse-native (BigQuery / Snowflake / Redshift) |
| **PostHog** | All-in-one (see §4.6) | Strong flags + experimentation in the same product |
| **Kameleoon, Flagsmith, Split** | Specialty / mid-market | — |

**Marketing-AI specific use cases**:
- **Progressive rollout of creative-generation agents** (5% of traffic → 25% → 100%) with KPI gates
- **A/B testing generated creatives** at scale against CTR predictors
- **Per-tenant model routing** — route Brand A to Claude Sonnet, Brand B to GPT-4o based on flag
- **Safety kill-switches** — flip a flag when guardrails (§3.10) detect out-of-policy outputs

**Productization-lens reading**: OpenAI's Statsig acquisition is strategically interesting — it moves the feature-flag layer *inside* the AI-product company, suggesting that experimentation + routing is core AI-product infrastructure, not a separate vendor category.

**Key references**:
- [Statsig vs LaunchDarkly comparison](https://www.statsig.com/vs/launchdarkly)
- [Top Feature Flag Tools Compared (Startupik)](https://startupik.com/top-feature-flag-tools-compared-launchdarkly-vs-statsig-vs-growthbook/)
- [Kameleoon — 10 top feature flag management tools 2026](https://www.kameleoon.com/blog/top-feature-flag-management-tools)
- [LaunchDarkly vs PostHog](https://www.statsig.com/perspectives/feature-flags-experimentation-pricing-comparison)
- [Flagsmith — Statsig Alternatives](https://www.flagsmith.com/blog/statsig-alternatives)

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
