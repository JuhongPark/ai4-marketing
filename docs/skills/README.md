# Skills-Axis Framework for AI Marketing

A cross-cutting lens over this repository. Where the domain-axis asks *"what marketing problem does this solve?"* (segmentation, recommendation, churn, MMM, viral), the **skills axis** asks *"which organizational capability does this AI serve?"*

Every AI marketing system lives inside a business. That business needs **three** distinct skill stacks:

| Skill | Scope | Who owns it | Outcome |
|---|---|---|---|
| **Sales skill** | Selling to customers + Raising / managing investment | CMO, CRO, CFO, IR, founders | Revenue, pipeline, funding rounds, investor engagement |
| **Productization skill** | Building and shipping the product that AI runs on | CTO, VP Engineering, ML lead, Product, PM | Deployable systems, PMF, robust infrastructure, maintainable stacks |
| **System Design & Management (SDM) skill** | Designing, operating, evolving, retiring AI systems as coherent socio-technical systems | AI / ML Architect, Platform lead, SRE, Governance / Risk, Systems Engineer | Architectural defensibility, operability, evolvability, safety, auditability |

These three skills are **complementary**, not competing. The same LightGCN paper informs the Productization skill (how to architect a GNN recommender), the Sales skill (how the recommender drives revenue per visit), and the SDM skill (how the recommender fits into an architecture with defined interfaces, feedback-loop exposure, technical-debt exposure, and an incident-response plan). The three skill documents re-read the same research through different operating lenses.

**The SDM axis is the one MIT SDM students recognize as their native vocabulary** — decomposition, interfaces, tradespace, lifecycle, feedback loops, capability traps, STAMP, Team Topologies, Conway's Law. It is not a restatement of Productization; it is the *system-level* framing of the *engineering-and-management* intersection that defines the SDM discipline.

---

## Framework Definitions

### Sales Skill — Selling + Investment

**Selling (customer-facing)** covers every AI system that directly or indirectly moves a customer closer to purchase, renewal, or advocacy:

- Targeting and segmentation for outreach
- Recommendation and personalization for conversion
- Pricing, promotion, and creative optimization
- Retention, winback, and expansion
- Marketing-mix budget allocation and attribution
- Viral, influencer, and word-of-mouth amplification
- Sales enablement, pipeline forecasting, and conversational commerce

**Investment (capital-facing)** covers every AI system used to raise, deploy, or defend capital for the business — a dimension largely absent from the domain surveys but unavoidable for any AI-native company:

- Venture capital deal sourcing, screening, and decisioning (external investor view)
- Due diligence automation (investor-side and counterparty-side)
- Investor relations, earnings preparation, and targeting (issuer-side)
- Pitch/deal-memo assistance, fundraising narrative, competitive positioning
- AI-augmented CFO workflows (forecasting, scenario modeling)

Rule of thumb: *if the primary outcome is revenue or capital, it is a Sales-skill artifact, even if the method (RL, GNN, LLM) is identical to a Productization artifact.*

### Productization Skill — Product Implementation Only

**Productization** is strictly bounded to the act of **building the product itself** — the engineering, science, and design that turns research into a shippable system. It intentionally excludes go-to-market, pricing, and fundraising (those are Sales-skill concerns even when the buyer is a CTO).

Productization spans three layers:

1. **Model engineering**: architectures, training recipes, evaluation regimes (GNN, deep RL, Bayesian inference, causal ML, LLM fine-tuning, diffusion)
2. **System engineering**: MLOps / LLMOps, RAG pipelines, agent orchestration, serving, observability, data pipelines, off-policy evaluation
3. **Product engineering**: PRD authoring, customer-interview synthesis, PMF methodology, system architecture, commercialization strategy as it relates to *what the product does*, not *how it is sold*

Rule of thumb: *if you can do it inside engineering / product without a sales, marketing, finance, or IR stakeholder, it is Productization.*

### System Design & Management (SDM) Skill — AI Systems as Engineered Socio-Technical Systems

**SDM** asks the question neither Sales nor Productization frames as primary: *"how do we design, operate, evolve, and retire this AI system as a coherent engineered socio-technical system, with defensible architecture, explicit lifecycle, analyzable dynamics, and accountable governance?"*

SDM spans six sub-skills:

1. **System architecture** — Crawley-Cameron decomposition, OPM (ISO 19450:2024), tradespace exploration, cloud reference architectures, Architecture Decision Records
2. **System dynamics** — Forrester / Sterman / Meadows feedback-loop analysis applied to AI (runaway loops, performative prediction, Goodhart, capability trap, model collapse)
3. **Lifecycle management and technical debt** — Sculley et al. 2015 nine debt categories, ISO/IEC 5338, MLOps maturity, model deprecation
4. **Platform engineering and Team Topologies** — Skelton & Pais four team types, CNCF IDP for AI, canonical ML-platform case studies (Uber / Airbnb / LinkedIn / Netflix / Meta), Conway's Law
5. **SRE, safety engineering, STAMP/STPA** — Leveson system-theoretic process analysis applied to AI, Google SRE for ML, chaos engineering for AI, safety cases, incident response
6. **Governance as system design** — decision rights, accountability graphs, RACI for AI, model cards, policy-as-code (complementary to the regulatory frame in `regulatory_appendix.md`)

Rule of thumb: *if a question starts with "how do we treat this AI as a coherent system with an architecture, a lifecycle, a reliability envelope, and an evolution plan," it is an SDM concern.*

The SDM skill document (`system_design_and_management.md`) builds on the methodological foundation established in `docs/deep_dive/DD8_mit_sdm_marketing.md`. DD8 covered the MIT SDM program's *general* methodology; the SDM skill doc applies that methodology specifically to AI systems in marketing contexts, closing sixteen concrete open problems identified as thesis-scope opportunities.

---

## Boundary Rules and Overlaps

Some artifacts legitimately live in multiple skills. The project treats these as **multi-listed**, not double-counted:

| Artifact | Sales lens | Productization lens | SDM lens |
|---|---|---|---|
| LightGCN (DD1) | "How a better recommender lifts revenue per session" | "How to architect a production web-scale GNN recommender" | "How to decompose the ranking subsystem and surface its feedback exposure" |
| Retail-GPT (frontier/generative, frontier/agentic) | "Conversational commerce as sales channel" | "How to engineer a RAG + tool-use commerce agent" | "Reference-architecture decomposition; incident-response runbook for agent hallucination" |
| PyMC-Marketing / Robyn / Meridian (DD5) | "Budget allocation and CMO narrative" | "How to build and evaluate a Bayesian MMM stack" | "Lifecycle: retraining cadence, data-pipeline debt, posterior-drift monitoring" |
| Bandit systems (DD2 §4) | "Real-time offer selection that lifts conversion" | "How to engineer and evaluate a contextual-bandit serving layer" | "System dynamics: reward hacking, Goodhart, capability trap on exploration budget" |
| SDM system architecture (DD8) | Minimal | "Decomposition, interfaces, and governance for AI marketing stacks" | **Primary — methodological foundation extended in `system_design_and_management.md`** |
| Aral viral experiments (DD3, DD7) | "How viral seeding amplifies sales" | Minimal (methodology, not product) | "Socio-technical design of the experiment, measurement under interference" |
| Catherine Tucker privacy research (DD7) | "Privacy UX affects ad engagement" | Minimal | "Governance-as-system-design: user-agency UX as architectural decision" |
| Dao multi-agent thesis (DD8, frontier/agentic) | Minimal | "Coordination topology for multi-agent marketing systems" | **Primary — STAMP-adjacent cascading-failure analysis for multi-agent LLM systems** |
| Sculley et al. 2015 *Hidden Technical Debt in ML* | Minimal | "Debt-reduction as engineering discipline" | **Primary — the canonical SDM-flavored ML paper; nine debt categories** |
| Skelton & Pais *Team Topologies* | Minimal | "Platform-team engineering" | **Primary — AI-org design pattern** |
| Shumailov et al. *Model Collapse* (Nature 2024) | Minimal | "Training-data provenance" | **Primary — dynamic-system failure archetype** |
| Leveson STAMP / STPA | Minimal | Minimal (method, not product) | **Primary — safety-engineering method** |

When in doubt, triangulate:
- *"If I delete the sales team, does this still matter?"* → **Productization or SDM**
- *"If I delete the engineering team, does this still matter?"* → **Sales**
- *"Does the question start with how the system decomposes, evolves, fails, or is governed as a whole?"* → **SDM**
- *"Does the question start with which model / pipeline / tool to ship?"* → **Productization**
- *"Does the question start with whose revenue or capital moves?"* → **Sales**

---

## Mapping Principles (Index Convention)

The two skill documents each mirror the structure:

1. **Skill subdivisions** (2-3 top-level buckets, e.g., Selling vs. Investment for Sales; Model vs. System vs. Product for Productization)
2. **Pre-existing repo coverage**: which DDs and frontier docs already address each bucket, and how to read them with the skill lens
3. **Newly surveyed research**: what was missing from the repo and has been added — with full citations
4. **Open problems**: skill-specific gaps, not method-specific gaps (those stay in DD6)
5. **References**: split into "already in CATALOG" and "new to CATALOG"

New citations added via the skills axis are back-referenced in `docs/CATALOG.md` so paper-level lookup stays authoritative regardless of which axis the reader entered from.

---

## Reading Paths Through the Skills Axis

### Path S: "I need AI to drive revenue this quarter"
1. `docs/skills/sales_skill.md` — Selling subdivisions (targeting, personalization, retention, MMM)
2. From each subdivision, follow the back-link into the relevant DD for methodological depth
3. `docs/frontier/generative_ai_and_llms_in_marketing.md` §4 for GenAI personalization ROI
4. `docs/frontier/agentic_ai_marketing.md` §2 for autonomous campaign execution

### Path I: "We are raising a round — how does AI change the investor dynamic?"
1. `docs/skills/sales_skill.md` — Investment subdivisions (deal sourcing, due diligence, IR, CFO)
2. Cross-reference the VC-side research with the issuer-side research — both use the same underlying NLP / RAG stacks
3. Revisit `docs/skills/productization_skill.md` — the technical bar has shifted; investors expect production maturity

### Path P: "I am building the product and need to ship"
1. `docs/skills/productization_skill.md` — all three layers
2. `docs/deep_dive/DD6_ai_marketing_foundations.md` for the cross-method theoretical substrate
3. `docs/deep_dive/DD8_mit_sdm_marketing.md` for system architecture discipline
4. `docs/frontier/*` — current state of the implementation edge

### Path SDM: "I am an MIT SDM student and want to treat this AI stack as a system"
1. `docs/deep_dive/DD8_mit_sdm_marketing.md` — the SDM methodological baseline (Crawley-Cameron, Moser ABM)
2. `docs/skills/system_design_and_management.md` §1-§8 — the SDM lens applied end-to-end to AI marketing
3. §10 of that doc — sixteen open problems scoped for SDM theses
4. §11 Reading Paths SDM-A / SDM-B / SDM-C / SDM-D for specific use cases
5. Cross-reference `docs/skills/regulatory_appendix.md` for governance-as-compliance (complementary to §7 of the SDM doc)

---

## Maintenance Notes

When a new document is added to `docs/`, its author should:

1. Tag the document as Sales-relevant, Productization-relevant, or both in the document map in `INDEX.md`.
2. If it introduces a new sub-skill bucket, extend the relevant skill document.
3. Add new citations to `docs/CATALOG.md` with a "Discussed in" column that includes the skill doc.

The skills axis and the domain axis must stay mutually consistent — a change on one side triggers a check on the other.
