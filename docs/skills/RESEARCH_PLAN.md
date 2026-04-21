# Follow-Up Research Plan for the Skills Axis

Scope: deepen both Sales (selling + investment) and Productization skill documents beyond their first pass. The initial revision established the framework and seeded each sub-skill with production-oriented references. This plan enumerates the under-explored extensions and the research moves required to close them.

---

## 1. Coverage Gaps Identified in the First Pass

| Dimension | First-pass strength | First-pass gap |
|---|---|---|
| Sales — Selling | Heavy on industry-analyst stats, strong method linkage to DDs | Sparse on peer-reviewed marketing-science research beyond MIT |
| Sales — Selling | Covers B2B / e-commerce broadly | Thin on vertical specificity (SaaS, fintech, retail, healthcare, enterprise) |
| Sales — Investment | Covers VC screening, DD, IR, PMF | No coverage of private equity, hedge funds, M&A quant, valuation models, public-equity AI, accelerator-stage AI |
| Productization — Model | Covers all major architectures via existing DDs | Thin on foundation-model selection, MoE, state-space, multimodal trade-offs |
| Productization — System | Covers LLMOps and RAG concepts | Thin on concrete vendor / OSS comparison for vector DBs, serving, orchestration, observability |
| Productization — System | Mentions guardrails | No detailed coverage of guardrail stacks (NeMo, Guardrails AI, Lakera) and evaluation frameworks (Ragas, TruLens, Langfuse, Phoenix) |
| Productization — Product | Covers PRD + PMF | Thin on AI-specific UX, human-AI interaction design, accessibility, AI ethics frameworks (NIST AI RMF, EU AI Act) |
| Cross-axis | Clear dual-listing rules | Does not yet surface the compliance / regulatory layer that bridges both skills |

---

## 2. Research Themes and Questions

### Theme A — Academic Marketing Research for AI Selling

**Questions**:
- Which peer-reviewed studies in *Journal of Marketing Research*, *Marketing Science*, *Journal of Consumer Research*, *International Journal of Research in Marketing* from 2023-2026 are directly relevant to AI-driven selling, beyond the MIT faculty covered in DD7?
- What are the Stanford, Wharton, Kellogg, Chicago Booth, HBS, INSEAD equivalents of Tucker / Eckles / Aral / Simester?
- Are there seminal post-ChatGPT experiments on generative-AI personalization effects in marketing contexts?

**Success signal**: new named faculty with specific 2023+ publications added to the sales_skill doc.

### Theme B — Industry-Vertical AI Selling

**Questions**:
- SaaS: product-led growth, free-trial conversion AI, usage-based pricing ML
- Fintech / payments: risk-adjusted lead scoring, underwriting-aware sales
- Retail / CPG: shelf-space optimization, SKU recommendation, in-store AI
- Healthcare / life sciences: HCP targeting, field-force AI, patient-access programs
- Enterprise B2B: ABM, intent data, committee-aware selling

**Success signal**: a vertical-lens sub-section in sales_skill §2 or a new §2.8 summarizing vertical patterns.

### Theme C — Private Equity and Hedge Fund AI

**Questions**:
- How do PE firms use AI for portfolio-company operational DD vs. deal DD?
- Which quant hedge funds (Renaissance, Two Sigma, D.E. Shaw, Citadel) publicly describe their ML stacks?
- What is the current state of alt-data pipelines (satellite, credit-card panel, web-scraped) in investment workflows?
- LBO modeling automation?

**Success signal**: new Investment sub-section (sales_skill §3.6 or split) covering PE / hedge fund / public-equity quant.

### Theme D — AI Valuation and M&A Models

**Questions**:
- DCF + LLM augmentation: can LLMs read 10-K / S-1 and generate defensible DCF inputs?
- Comparable-company selection automated via embedding similarity?
- M&A post-merger integration ML?
- SPAC and IPO pricing with AI?

**Success signal**: a §3.7 sub-section with concrete citations.

### Theme E — Accelerator and Early-Stage AI

**Questions**:
- How does Y Combinator, Techstars, 500 Global, or equivalent use AI internally?
- How do founders use AI during the accelerator cycle (YC W25 / S25 cohort practices)?
- Grant / non-dilutive funding AI (SBIR, Horizon Europe)?

**Success signal**: covered as a short sub-section; possibly merged with §3.4 commercialization.

### Theme F — Foundation-Model Productization

**Questions**:
- Selection criteria: Claude Opus 4.7 vs. Sonnet 4.6 vs. Haiku 4.5 vs. GPT-5 vs. Gemini 2.x vs. LLaMA-3 vs. Mixtral vs. DBRX for marketing workloads.
- Mixture-of-Experts (MoE) and state-space (Mamba) trade-offs vs. dense transformers.
- Multimodal foundation models for ad creative + understanding (Gemini 1.5, Claude Vision, GPT-4V/5, Pixtral).
- Domain-adapted marketing LMs (retail-specific, ads-specific).

**Success signal**: a §2.6 or new sub-section in productization_skill on foundation-model selection for marketing.

### Theme G — Vector Database Engineering

**Questions**:
- Head-to-head: Pinecone vs. Weaviate vs. Qdrant vs. Milvus vs. Chroma vs. pgvector vs. OpenSearch vs. Elastic vs. Mongo Atlas Vector for marketing-scale workloads.
- Hybrid retrieval (dense + sparse + metadata filters) engineering patterns.
- Operational concerns: index rebuild, incremental updates, cost-per-query at catalog-freshness cadence.

**Success signal**: a concrete comparison table in productization_skill §3.2 or a new §3.2.x.

### Theme H — Orchestration Frameworks

**Questions**:
- LangChain vs. LlamaIndex vs. Semantic Kernel vs. CrewAI vs. AutoGen vs. DSPy vs. Haystack: trade-offs for marketing agent stacks.
- DSPy's "programmatic prompts" vs. traditional prompt-chaining — is it finally the right abstraction?
- Multi-agent topology implementations (CrewAI, AutoGen Roles, LangGraph) matched to Dao (2025) topology analysis.

**Success signal**: a sub-section with concrete framework comparison and back-link to Dao.

### Theme I — Model Serving and Inference Infrastructure

**Questions**:
- vLLM vs. TGI vs. TensorRT-LLM vs. SGLang vs. Ollama (dev) vs. Bedrock / Vertex / Azure OpenAI (managed).
- Continuous batching, speculative decoding, KV-cache reuse — operational effects at marketing-scale QPS.
- Cost tiering (Opus/Sonnet/Haiku equivalents) as a serving-architecture decision.

**Success signal**: added to productization_skill §3.1 or §3.2.

### Theme J — Observability, Evaluation, and Guardrails

**Questions**:
- Arize vs. WhyLabs vs. Evidently vs. Fiddler vs. Langfuse vs. Phoenix (Arize OSS) vs. LangSmith vs. Traceloop — what does each layer cover, what are the overlaps?
- Ragas vs. TruLens vs. OpenAI Evals vs. DeepEval — evaluation frameworks for RAG and agent quality.
- NVIDIA NeMo Guardrails vs. Guardrails AI vs. Lakera Guard vs. PromptGuard for marketing content safety.
- Incident frameworks for AI: postmortems, red-teaming cadence, MLRisk registers.

**Success signal**: a new §3.7 (Observability & Evaluation) and §3.8 (Guardrails).

### Theme K — AI Product UX and Human-AI Interaction

**Questions**:
- Google's People + AI Guidebook (2025 update), Microsoft HAX Toolkit, Apple Human Interface Guidelines for ML — what do they prescribe for marketing UX?
- Confidence-display patterns (when to show a probability, when not).
- Graceful-degradation patterns when an agent fails.
- Feedback-capture UX (thumbs-up/down, why-bad, repair flows).

**Success signal**: a §4.x sub-section on UX for AI marketing products.

### Theme L — AI Ethics, Risk, and Regulatory Frameworks

**Questions**:
- NIST AI Risk Management Framework (2023, 2024 GenAI profile) applicability to marketing.
- EU AI Act (in force 2025) obligations for marketing AI: tiered risk categories, transparency, disclosure.
- FTC guidance on algorithmic pricing / targeting (2024-2026).
- ISO 42001 (AI management system) adoption.

**Success signal**: a cross-axis sub-section or a shared appendix covering regulatory requirements both skills must respect.

### Theme M — AI Product Analytics and Telemetry

**Questions**:
- Amplitude / Mixpanel / Heap / PostHog AI-feature analytics patterns.
- Prompt-level analytics: what does "usage analytics" mean when the same prompt yields different responses?
- Cost / token telemetry aggregation (cost per user, per session, per feature).
- Feature flagging for AI rollouts (LaunchDarkly, Statsig, Split, GrowthBook).

**Success signal**: a §4.x sub-section or expansion in §3.1 (MLOps) referencing telemetry.

---

## 3. Prioritization

Organize the 13 themes above into three tiers based on **information density the repo currently lacks** and **practitioner pain likely to surface in real work**.

**Tier 1 (execute in this round)**:
- Theme C: PE / hedge fund / public-equity AI (largest gap in Sales-Investment)
- Theme G + H + I: Vector DB + Orchestration + Serving (the operational stack reality)
- Theme J: Observability / Eval / Guardrails (the production-safety stack)
- Theme L: AI Ethics / Regulatory (shared across Sales and Productization)

**Tier 2 (next round if time permits)**:
- Theme A: Academic marketing research beyond MIT
- Theme B: Industry-vertical AI selling
- Theme D: AI valuation and M&A models
- Theme F: Foundation-model productization selection
- Theme K: AI Product UX

**Tier 3 (as-needed)**:
- Theme E: Accelerator / grant AI
- Theme M: Product analytics and telemetry

Tier 1 is the immediate plan; it is broad enough to substantively upgrade both skill docs in one pass.

---

## 4. Execution Method

1. **Parallel web research** on Tier 1 themes using delegated agents, one per theme, with a clear "return under 400 words" cap.
2. **Vendor / OSS comparison tables** — cross-verified where possible against both vendor docs and a neutral practitioner source.
3. **Link validation** on every new citation before it is added.
4. **Cross-reference discipline** — every new reference appears in `docs/CATALOG.md` with a "Discussed in" back-link and in the relevant skill doc.
5. **Open problems update** — extended open-problem lists in both skill docs reflecting genuinely new gaps, not restatements of method gaps in DD6.
6. **Final validation pass** — all internal links, INDEX.md entries, and CATALOG entries checked for consistency.

---

## 5. Output Artifacts

After this plan is executed, the repo will have:

- `docs/skills/sales_skill.md` expanded with PE / hedge fund / M&A / valuation sub-sections and an ethics/regulatory reference frame
- `docs/skills/productization_skill.md` expanded with vector DB / orchestration / serving / observability / guardrails / UX sub-sections
- `docs/CATALOG.md` extended with 6-8 new sections corresponding to Tier 1 research
- `INDEX.md` updated to surface the new sub-skill buckets
- Open-problem lists in both docs refreshed to reflect genuinely new gaps

The framework itself (Sales vs. Productization, selling + investment vs. implementation) is not re-opened — the first-pass framing remains load-bearing.

---

## 6. Out-of-Scope for This Round

- Changes to the original DDs (DD1-DD8) unless required for consistency.
- Reworking the frontier docs (generative / llm_enhanced / agentic) — they already serve their purpose.
- Adding code examples or tutorials — research-synthesis repo, not an implementation repo.
- Translating any content to Korean — repo convention is English-only (see CLAUDE.md).
