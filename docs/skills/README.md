# Skills-Axis Framework for AI Marketing

A cross-cutting lens over this repository. Where the domain-axis asks *"what marketing problem does this solve?"* (segmentation, recommendation, churn, MMM, viral), the **skills axis** asks *"which organizational capability does this AI serve?"*

Every AI marketing system lives inside a business. That business needs two distinct skill stacks:

| Skill | Scope | Who owns it | Outcome |
|---|---|---|---|
| **Sales skill** | Selling to customers + Raising / managing investment | CMO, CRO, CFO, IR, founders | Revenue, pipeline, funding rounds, investor engagement |
| **Productization skill** | Building and shipping the product that AI runs on | CTO, VP Engineering, ML lead, Product, PM | Deployable systems, PMF, robust infrastructure, maintainable stacks |

These two skills are **complementary**, not competing. The same LightGCN paper informs the Productization skill (how to architect a GNN recommender) and the Sales skill (how the recommender drives revenue per visit). The two skill documents re-read the same research through different operating lenses.

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

---

## Boundary Rules and Overlaps

Some artifacts legitimately live in both skills. The project treats these as **dual-listed**, not double-counted:

| Artifact | Sales lens | Productization lens |
|---|---|---|
| LightGCN (DD1) | "How a better recommender lifts revenue per session" | "How to architect a production web-scale GNN recommender" |
| Retail-GPT (frontier/generative, frontier/agentic) | "Conversational commerce as sales channel" | "How to engineer a RAG + tool-use commerce agent" |
| PyMC-Marketing / Robyn / Meridian (DD5) | "Budget allocation and CMO narrative" | "How to build and evaluate a Bayesian MMM stack" |
| Bandit systems (DD2 §4) | "Real-time offer selection that lifts conversion" | "How to engineer and evaluate a contextual-bandit serving layer" |
| SDM system architecture (DD8) | Minimal | "Decomposition, interfaces, and governance for AI marketing stacks" |
| Aral viral experiments (DD3, DD7) | "How viral seeding amplifies sales" | Minimal (methodology is causal-inference, but artifact is not a product) |
| Catherine Tucker privacy research (DD7) | "Privacy UX affects ad engagement — relevant to personalization strategy" | Minimal |
| Dao multi-agent thesis (DD8, frontier/agentic) | Minimal | "Coordination topology for multi-agent marketing systems" |

When in doubt: ask *"if I delete the sales team, does this still matter?"* If yes, it is Productization. Ask *"if I delete the engineering team, does this still matter?"* If yes, it is Sales.

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

---

## Maintenance Notes

When a new document is added to `docs/`, its author should:

1. Tag the document as Sales-relevant, Productization-relevant, or both in the document map in `INDEX.md`.
2. If it introduces a new sub-skill bucket, extend the relevant skill document.
3. Add new citations to `docs/CATALOG.md` with a "Discussed in" column that includes the skill doc.

The skills axis and the domain axis must stay mutually consistent — a change on one side triggers a check on the other.
