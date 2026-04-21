# ai4-marketing Research Index

Quick navigation across every document in this project. For paper-level lookup, see `docs/CATALOG.md`.

---

## Document Map

### Root

| File | Purpose |
|------|---------|
| `README.md` | Project overview, key approaches table, search keywords |
| `INDEX.md` | This document — topic-based navigation |
| `docs/CATALOG.md` | All cited papers / tools, grouped by method or domain |

### Domain Surveys (`docs/`)

Short, breadth-first surveys of each marketing AI domain. Read these first for context.

| File | Topic | Primary AI Methods |
|------|-------|---------------------|
| `docs/01_customer_segmentation.md` | Customer segmentation and targeting | Deep clustering, RFM-Net, LRFMS, XAI |
| `docs/02_recommendation_systems.md` | GNN-based recommenders | PinSage, LightGCN, NGCF, KGAT, SR-GNN |
| `docs/03_influencer_marketing.md` | Influence prediction and viral spread | DeepInf, Deep RL for IM, Aral experiments |
| `docs/04_churn_prediction.md` | Customer churn prediction and retention | CCP-Net, Random Forest, hybrid DL, RFM |
| `docs/05_marketing_mix_modeling.md` | MMM and multi-touch attribution | PyMC-Marketing, Robyn, Meridian, DNAMTA, CAMTA |

### Deep Dives (`docs/deep_dive/`)

Long-form treatments: architectures, quantitative results, open problems. Read these after the domain surveys.

| File | Topic | Core References |
|------|-------|------------------|
| `docs/deep_dive/DD1_gnn_recommender_systems.md` | GNN for recommendation | GraphSAGE (2017), PinSage (2018), NGCF (2019), LightGCN (2020), KGAT (2019), SR-GNN (2019) |
| `docs/deep_dive/DD2_deep_rl_viral_marketing.md` | Deep RL for viral, paid, personalized marketing | DGN, BiGDN, HEDRL-IM, BIM-DRL; RLB-DP, SS-RTB; LinUCB, Thompson, BanditLP |
| `docs/deep_dive/DD3_influencer_marketing.md` | Social influence and influencer marketing | DeepInf, HetInf, DeepPP; Aral/MIT Sloan experiments; ML for Instagram |
| `docs/deep_dive/DD4_customer_churn_cascade.md` | Customer churn prediction, contagion, uplift | CCP-Net, classical ML; Ferreira-Telang churn contagion; CausalML uplift |
| `docs/deep_dive/DD5_marketing_mix_modeling.md` | Bayesian MMM and causal attribution | Jin 2017, PyMC-Marketing, Robyn, Meridian; DNAMTA, CAMTA, Amazon MTA |
| `docs/deep_dive/DD6_ai_marketing_foundations.md` | Theoretical foundations synthesis | Bayesian + causal + RL + representation learning; off-policy evaluation |
| `docs/deep_dive/DD7_mit_marketing_research.md` | MIT research landscape on AI marketing | Tucker, Eckles, Aral, Simester, Perakis, Bertsimas, Roy, Pentland, Barzilay, Jaakkola |
| `docs/deep_dive/DD8_mit_sdm_marketing.md` | MIT SDM program research on AI marketing | Dao (2025) multi-agent thesis, MoT theses, SDM system architecture methodology |

### Frontier Research (`docs/frontier/`)

Post-2023 topics not yet covered in the main deep dives. Read these for the current edge of the field.

| File | Topic | Core References |
|------|-------|------------------|
| `docs/frontier/generative_ai_and_llms_in_marketing.md` | LLMs and diffusion models for creative, personalization, conversational commerce | Stable Diffusion CTR pipeline, Retail-GPT, Amazon RAG, consumer sentiment research |
| `docs/frontier/llm_enhanced_recommenders.md` | LLM-based recommenders | P5 (2022), TALLRec (2023), LLaRA (2024), E4SRec, LLM-SRec, instance-wise LoRA |
| `docs/frontier/agentic_ai_marketing.md` | Autonomous agents, multi-agent coordination, RAG commerce | Deloitte/Gartner agentic stats, Dao (2025) topology analysis, RAG for commerce |

### Skills Axis (`docs/skills/`)

Cross-cutting lens that re-reads the repo by **organizational capability** rather than by marketing domain or AI method. Read these when you want to see the same research through a selling / investment / productization / systems-design lens.

**Three-lens framework**: Sales (selling + investment) · Productization (model + system + product engineering) · System Design & Management (architecture + dynamics + lifecycle + platform + SRE + governance).

| File | Topic | Core Coverage |
|------|-------|----------------|
| `docs/skills/README.md` | Three-lens skills framework | Scope, boundary rules, triangulation heuristics, reading paths |
| `docs/skills/sales_skill.md` | AI for selling + investment | §2 Selling: targeting, recommendation, retention, MMM, viral, sales enablement, academic GenAI research (Kapoor/Kumar, Hermann/Puntoni, V.Kumar/Kotler, Columbia, Stanford, Wharton), vertical sub-skills (SaaS/PLG, Fintech, Retail/CPG, Healthcare/HCP, Enterprise B2B/ABM). §3 Investment: VC screening, due diligence, IR, CFO, commercialization narrative, PE AI, quant hedge funds, public-equity AI, valuation models, accelerator AI |
| `docs/skills/productization_skill.md` | AI product implementation | Model engineering (GNN / RL / Bayesian / LLM / Diffusion / foundation-model selection) + System engineering (MLOps / LLMOps / vector DBs / RAG / agents / orchestration / LLM serving / OPE / observability / evaluation / guardrails) + Product engineering (PRD / PMF / architecture / HAI UX / product telemetry / feature flagging) |
| `docs/skills/system_design_and_management.md` | **AI systems as engineered socio-technical systems (SDM lens)** | §2 Architecture (Crawley-Cameron / OPM ISO 19450:2024 / tradespace / cloud reference architectures / ADRs / SysML v2) · §3 System dynamics (Forrester / Sterman / Meadows; runaway loops, performative prediction, Goodhart, model collapse, capability trap) · §4 Lifecycle and technical debt (Sculley 2015 nine categories, Amershi 2019, ISO 5338, MLOps maturity) · §5 Platform engineering and Team Topologies (Skelton-Pais, CNCF IDP for AI, canonical ML-platform case studies) · §6 SRE, safety, STAMP (Leveson, Mylius 2025) · §7 Governance as system design · §10 sixteen open SDM thesis problems |
| `docs/skills/regulatory_appendix.md` | Cross-axis regulatory frame | NIST AI RMF + 2024 GenAI Profile, EU AI Act 2025-2027 timeline, FTC Operation AI Comply + COPPA 2025, ISO 42001, US state-law fragmentation, sector-specific nuances, 2026 operating checklist |
| `docs/skills/RESEARCH_PLAN.md` | First-round follow-up research plan | Coverage-gap analysis for skills-axis extensions (Tier 1 executed) |
| `docs/skills/SDM_LENS_RESEARCH_PLAN.md` | SDM-lens research plan | 15 themes across architecture / dynamics / lifecycle / platform / SRE / governance; Tier 1 executed into `system_design_and_management.md` |

---

## Topic Index (Keyword-Based Discovery)

### By AI Method

| Method | Primary documents |
|--------|---------------------|
| **Graph Neural Networks (CF)** | DD1, docs/02, frontier/llm_enhanced_recommenders (hybrid) |
| **Deep Reinforcement Learning** | DD2, DD6 (theory) |
| **Contextual Bandits** | DD2, DD6 |
| **Bayesian Inference / Bayesian MMM** | DD5, DD6, docs/05 |
| **Causal Inference / Double ML** | DD4 (uplift), DD5 (attribution), DD6 (theory) |
| **Uplift Modeling** | DD4 §7, DD6 §4 |
| **Offline RL / Off-Policy Evaluation** | DD6 |
| **Large Language Models** | frontier/generative_ai_and_llms_in_marketing, frontier/llm_enhanced_recommenders, frontier/agentic_ai_marketing |
| **Diffusion Models** | frontier/generative_ai_and_llms_in_marketing |
| **Retrieval-Augmented Generation (RAG)** | frontier/generative_ai_and_llms_in_marketing §5, frontier/agentic_ai_marketing §2 |
| **Deep Learning (general)** | DD4 (churn), docs/01 (segmentation) |
| **Hybrid ML + classical models** | DD4, DD5, frontier/llm_enhanced_recommenders |

### By Marketing Domain

| Domain | Documents |
|--------|-----------|
| **Recommendation** | DD1, docs/02, frontier/llm_enhanced_recommenders |
| **Customer Segmentation** | docs/01 |
| **Influencer Marketing / Viral** | DD2, DD3, docs/03 |
| **Customer Churn / Retention** | DD4, docs/04 |
| **Marketing Mix Modeling** | DD5, docs/05 |
| **Multi-Touch Attribution** | DD5, docs/05 |
| **Real-Time Bidding (Programmatic)** | DD2 §3 |
| **Personalized Offers / Treatments** | DD2 §4 (bandits), DD4 §7 (uplift) |
| **Creative / Copy Generation** | frontier/generative_ai_and_llms_in_marketing |
| **Conversational Commerce** | frontier/generative_ai_and_llms_in_marketing §5, frontier/agentic_ai_marketing |
| **Agentic Marketing Workflows** | frontier/agentic_ai_marketing |

### By Research Institution

| Institution | Primary document |
|-------------|---------------------|
| **MIT Sloan (Marketing Group)** | DD7 §2 |
| **MIT ORC (Operations Research Center)** | DD7 §3 |
| **MIT IDE** | DD7 §4 |
| **MIT IDSS** | DD7 §5 |
| **MIT Media Lab** | DD7 §6 |
| **MIT CSAIL** | DD7 §7 |
| **MIT SDM Program** | DD8 |
| **Google Research (MMM)** | DD5 §2, §3.3 (Meridian) |
| **Meta (Robyn MMM)** | DD5 §3.2 |
| **Amazon Science (RAG, MTA)** | DD5 §4.3, frontier/generative_ai_and_llms_in_marketing §5 |
| **Stanford (GNN)** | DD1 (Leskovec group) |
| **Pinterest (PinSage)** | DD1 §3 |

### By Challenge / Problem Type

| Problem | Where to look |
|---------|---------------|
| **Cold start** | DD1 §2 (GraphSAGE), frontier/llm_enhanced_recommenders |
| **Long customer journey / lifetime** | DD4 (retention), DD5 (attribution), frontier/agentic_ai_marketing |
| **Privacy / post-cookie** | DD5 (Meridian, MMM), DD7 (Tucker), frontier/generative_ai_and_llms_in_marketing |
| **Fairness / balanced spread** | DD2 §2.5 (BIM-DRL) |
| **Scalability to web scale** | DD1 §3 (PinSage), frontier/llm_enhanced_recommenders §7 |
| **Interpretability / XAI for marketing** | DD6 gap 2, docs/01 |
| **Causal identification** | DD4 §7, DD5 §4, DD6 |
| **Off-policy evaluation** | DD6 §4 |
| **Brand safety under AI generation** | frontier/generative_ai_and_llms_in_marketing §4, frontier/agentic_ai_marketing §4 |

### By Skill (Sales · Productization · SDM)

| Skill | Where to look | When to enter here |
|-------|---------------|---------------------|
| **Sales — Selling** | skills/sales_skill §2 | "Will this move revenue?" |
| **Sales — Academic GenAI research** | skills/sales_skill §2.8 | "What does peer-reviewed marketing science say?" |
| **Sales — Vertical sub-skills** | skills/sales_skill §2.9 | "How does AI selling look in SaaS / fintech / retail / healthcare / B2B?" |
| **Sales — Investment** | skills/sales_skill §3 | "Will this move a funding round or investor decision?" |
| **Sales — Private equity** | skills/sales_skill §3.6 | "How do PE firms deploy AI across portfolios?" |
| **Sales — Hedge fund / public equity** | skills/sales_skill §3.7 | "How do quant shops and sell-side analysts use AI?" |
| **Sales — Valuation models** | skills/sales_skill §3.8 | "Can an LLM help with DCF / comparables / M&A?" |
| **Productization — Model engineering** | skills/productization_skill §2 | "How do we build and evaluate the model?" |
| **Productization — Foundation-model selection** | skills/productization_skill §2.6 | "Which base model and why?" |
| **Productization — Vector DBs** | skills/productization_skill §3.1.1 | "Which retrieval substrate at marketing scale?" |
| **Productization — Orchestration frameworks** | skills/productization_skill §3.3.1 | "LangGraph vs CrewAI vs AutoGen vs DSPy vs Pydantic AI?" |
| **Productization — LLM serving** | skills/productization_skill §3.7 | "vLLM vs TGI vs TensorRT-LLM vs Bedrock at scale?" |
| **Productization — Observability** | skills/productization_skill §3.8 | "Arize / Langfuse / Phoenix / LangSmith — which stack?" |
| **Productization — Evaluation** | skills/productization_skill §3.9 | "Ragas / TruLens / DeepEval / Promptfoo?" |
| **Productization — Guardrails** | skills/productization_skill §3.10 | "NeMo / Guardrails AI / Lakera / Bedrock Guardrails?" |
| **Productization — HAI / UX** | skills/productization_skill §4.5 | "Google PAIR / Microsoft HAX — how to apply?" |
| **Productization — Product telemetry** | skills/productization_skill §4.6 | "PostHog vs Amplitude for AI products?" |
| **Productization — Feature flagging** | skills/productization_skill §4.7 | "Statsig (now OpenAI-owned) vs LaunchDarkly?" |
| **SDM — System architecture** | skills/system_design_and_management §2 | "Decompose the AI stack à la Crawley-Cameron. OPM of an LLM pipeline. Tradespace over quality × latency × cost × risk. What ADRs do I need?" |
| **SDM — System dynamics** | skills/system_design_and_management §3 | "Which feedback loops are destabilizing my AI system? Is this runaway-loop / performative-prediction / Goodhart / capability-trap / model-collapse?" |
| **SDM — Technical debt & lifecycle** | skills/system_design_and_management §4 | "Which of Sculley's nine categories am I accruing? Where on the MLOps maturity ladder am I? Do I have a retirement plan?" |
| **SDM — Platform & team topology** | skills/system_design_and_management §5 | "Are we Team Topologies-aligned? Which Uber / Airbnb / Netflix / Meta platform pattern applies? Conway's Law consequences?" |
| **SDM — SRE, STAMP, safety** | skills/system_design_and_management §6 | "What are the Unsafe Control Actions? What does the incident-response runbook look like? Do we have a safety case?" |
| **SDM — Governance as system design** | skills/system_design_and_management §7 | "Decision rights / accountability graph / RACI for AI / model cards / policy-as-code" |
| **SDM — Open thesis topics** | skills/system_design_and_management §10 | "Which 16 gaps in 2024-2026 literature can I scope for a thesis?" |
| **Cross-axis — Regulatory** | skills/regulatory_appendix | "NIST / EU / FTC / ISO / state-law obligations" |

Each skill doc back-links into the relevant DD and frontier docs, so readers entering from the skills axis never lose access to methodological depth.

---

## How to Find What You Need

**Looking for a specific paper by title or author?**
→ `docs/CATALOG.md` — search by title or author; each entry lists the documents where it is discussed.

**Looking for a method (e.g., "contextual bandits", "Bayesian MMM")?**
→ Use the "By AI Method" table above.

**Looking for research in a specific marketing domain (e.g., churn, attribution)?**
→ Use the "By Marketing Domain" table above.

**Looking for a skill lens (selling, investment, productization)?**
→ `docs/skills/` — three documents re-read the repo through the Sales and Productization lenses, and introduce research not covered in the domain axis (VC screening, due diligence, IR, LLMOps, RAG engineering, AI product management).

**Looking for MIT-specific research?**
→ DD7 (faculty landscape) and DD8 (SDM theses and adjacent programs).

**Looking for the current state of the art (2024-2026)?**
→ `docs/frontier/` — three documents cover generative AI, LLM recommenders, and agentic marketing.

**Looking for theoretical foundations?**
→ DD6 — synthesizes Bayesian, causal, RL, and representation-learning frameworks.

**Looking for a strong baseline to beat?**
→ DD1 for recommendation (LightGCN), DD4 for churn (Random Forest), DD5 for MMM (Bayesian adstock+saturation).

---

## Reading Paths

### Path A: "I'm new to AI marketing and want the full landscape"

1. `README.md`
2. Each `docs/01_*` through `docs/05_*` for breadth
3. `docs/deep_dive/DD6_ai_marketing_foundations.md` for the theoretical synthesis
4. `docs/frontier/*` for 2024-2026 developments
5. `docs/CATALOG.md` to build a targeted reading list

### Path B: "I need to decide whether to use method X for problem Y"

1. Use "By AI Method" or "By Marketing Domain" table to find the right DD
2. Read the relevant DD's cross-method comparison section
3. Check "Open Problems" section of that DD for known limitations
4. Look up candidate papers in `docs/CATALOG.md` for details

### Path C: "I'm a PhD student / researcher surveying the area"

1. `docs/CATALOG.md` — scan all paper entries
2. DD6 for the unifying theoretical view
3. Each DD1-DD5 for methodology depth
4. DD7 and DD8 for MIT-specific research groups
5. `docs/frontier/*` for current research gaps
6. Cross-reference with curated GitHub lists linked in DD1, DD2, DD4, DD5, frontier

### Path D: "I'm a practitioner building an AI marketing stack"

1. `README.md` key approaches table — pick the relevant cell
2. Read the referenced paper directly via `docs/CATALOG.md`
3. Read the relevant DD section for production considerations
4. Check `docs/frontier/agentic_ai_marketing.md` §4 for deployment pitfalls
5. Use open-source tools linked in DD5 (PyMC-Marketing, Robyn, Meridian), DD4 (CausalML), frontier/llm_enhanced_recommenders (LLM4Rec curated lists)

### Path E: "I need to ship revenue this quarter"

1. `docs/skills/sales_skill.md` §2 — selling sub-skills, each with back-link to the relevant DD
2. From each sub-skill, follow the DD link for methodological depth
3. `docs/frontier/generative_ai_and_llms_in_marketing.md` §4 for GenAI personalization ROI
4. `docs/frontier/agentic_ai_marketing.md` §2 for autonomous campaign execution

### Path F: "We are fundraising — how does AI change the investor dynamic?"

1. `docs/skills/sales_skill.md` §3 — investment sub-skills (VC screening, due diligence, IR, pitch, CFO)
2. Cross-reference with `docs/skills/productization_skill.md` — investors now expect production maturity (§3)
3. DD8 §2.3 — *A Strategic Perspective on the Commercialization of AI* (MIT SDM)

### Path G: "I'm building the product and need to ship"

1. `docs/skills/productization_skill.md` — all three layers in order
2. `docs/deep_dive/DD6_ai_marketing_foundations.md` — cross-method theoretical substrate
3. `docs/deep_dive/DD8_mit_sdm_marketing.md` — system architecture discipline
4. `docs/frontier/*` — current implementation edge

### Path SDM: "I am an MIT SDM student designing an AI stack"

1. `docs/deep_dive/DD8_mit_sdm_marketing.md` — SDM methodological baseline
2. `docs/skills/system_design_and_management.md` — end-to-end SDM lens on AI marketing
3. §11 within that doc — Reading Paths SDM-A (defend architecture) / SDM-B (diagnose degradation) / SDM-C (design the org) / SDM-D (pick a thesis topic)
4. `docs/skills/regulatory_appendix.md` — governance as compliance (§7 of SDM doc is governance as system design — they are complementary)

---

## Maintenance Notes

This index should be updated whenever a new document is added to `docs/`. Each new document should:

1. Be listed in the Document Map section above.
2. Have its key methods added to the "By AI Method" table.
3. Have its domain added to the "By Marketing Domain" table if new.
4. Be tagged as Sales-relevant, Productization-relevant, or both (see `docs/skills/`).
5. Have its cited papers added to `docs/CATALOG.md` with a "Discussed in" back-reference.

Keeping `INDEX.md` (topic navigation), `docs/CATALOG.md` (paper-level lookup), and `docs/skills/*` (capability lens) mutually consistent is the core discipline for keeping the project searchable as it grows.
