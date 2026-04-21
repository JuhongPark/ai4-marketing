# SDM-Lens Research Plan — AI Systems Design and Management

This plan refocuses the repository through the lens an MIT SDM student actually brings to AI: **AI systems as engineered socio-technical systems requiring architectural decomposition, lifecycle management, dynamic analysis, and governance** — not as clever models that happen to run in production.

The earlier skills axis (Sales + Productization) answered *"what is AI used for, and how do we build it?"* This plan adds the missing axis: *"how do we design, operate, evolve, and retire AI systems as coherent systems?"*

---

## 1. Why This Axis

Existing repo coverage (DD1-DD8, frontier, two skills docs, regulatory appendix) describes:
- Algorithms (GNN, RL, Bayesian, LLM)
- Marketing problems (segmentation, MMM, viral)
- Infrastructure (vector DB, serving, orchestration, observability)
- Business (sales, investment, productization)
- Regulatory (NIST, EU AI Act, FTC, ISO 42001)

What it does not yet treat as primary:
1. **System architecture** — how the system *decomposes*, which subsystems carry which concerns, how interfaces are specified, what the architecture trade-space looks like
2. **System dynamics** — feedback loops in AI systems that make short-term wins become long-term problems (model-retraining traps, user-feedback corruption, engagement bait, reward hacking at organizational scale)
3. **Lifecycle management** — AI systems as objects with birth, maturity, senescence, retirement; the economics and engineering of each phase
4. **Socio-technical coupling** — humans, incentives, data, models, and processes as interlocking subsystems, not as "AI + people around it"
5. **Failure-mode analysis** — complex-system failures (cascading, latent, emergent), not just bug fixing
6. **Technical debt and architecture erosion** — Sculley et al. NeurIPS 2015 and its decade of follow-on
7. **Platform and organizational design** — AI as platform product, Team Topologies, Conway's Law for AI orgs
8. **Portfolio management** — allocating scarce attention across a portfolio of AI initiatives

These are the concerns that DD8 (MIT SDM) already flagged as methodological and that belong in every working AI marketing stack.

---

## 2. Theme Inventory (15 themes, tiered)

### Tier 1 — execute in this round

| # | Theme | Why it matters | Canonical reference |
|---|---|---|---|
| 1 | **AI System Architecture** | The SDM signature — decomposition, interfaces, tradespace for AI stacks | Crawley / Cameron / Selva (2015); Dori OPM; MIT System Architecture Group |
| 3 | **System Dynamics for AI** | Feedback loops in AI are the single most under-analyzed source of long-run failure | Forrester; Sterman *Business Dynamics* (2000); Meadows *Thinking in Systems* (2008); recent AI-specific stock-and-flow work |
| 5 | **Lifecycle Management and Technical Debt** | AI systems decay; debt accumulates faster than classical software; retirement is rarely planned | Sculley et al. NeurIPS 2015 *Hidden Technical Debt in ML Systems*; ISO/IEC 5338; CRISP-DM / TDSP |
| 7 | **Platform Engineering + Team Topologies** | AI organizations succeed or fail on team design + platform abstractions | Skelton & Pais *Team Topologies* (2019); IDP / Backstage; Conway's Law |
| 8 | **SRE, Safety Engineering, STAMP** | AI outages and failures require reliability discipline — not the same as classical SRE | Leveson *Engineering a Safer World* (2012) / STAMP / STPA; Google SRE book; chaos engineering for AI |

### Tier 2 — next round

| # | Theme | Why it matters |
|---|---|---|
| 2 | **AI as Socio-Technical System** | Organizational readiness, role design, work redesign when AI enters |
| 4 | **Complex-System Failure Modes and Cascades** | Dao (2025) is already in DD8; extend with STAMP and MASA |
| 6 | **AI Governance as System Design Problem** | Decision rights, accountability graphs, RACI for AI — distinct from regulatory compliance |
| 10 | **Enterprise Architecture for AI** | TOGAF, Data Mesh, API strategy, integration with legacy systems |
| 14 | **Human Factors and Cognitive Engineering** | Automation bias, trust calibration, deskilling, cognitive load |

### Tier 3 — boundary / as-needed

| # | Theme | Why it matters |
|---|---|---|
| 9 | **Capacity Planning and AI Unit Economics** | GPU forecasting, COGS, build-vs-buy — already brushed in §3.7 of productization |
| 11 | **AI Portfolio Management** | Stage-gate for AI projects, kill criteria, resource allocation |
| 12 | **Standards and Interoperability** | ONNX, MLflow, OpenTelemetry GenAI, MCP — brushed in §3.1 |
| 13 | **AI Safety Engineering for Autonomous Systems** | Emerging — mostly outside marketing's direct scope |
| 15 | **AI System Retirement and Migration** | Rarely discussed explicitly |

---

## 3. The SDM Conceptual Spine

Every Tier 1 research output should ultimately slot into this SDM spine:

```
[Context]
  Organization + customers + regulators + competitors + data + incentives (socio-technical context)
        ↓
[Architecture]
  Decomposition into subsystems (data / model / decision / serving / observability / governance)
  Interfaces (human-AI, AI-AI, AI-data, AI-external)
  Tradespace (cost × latency × quality × risk × flexibility)
        ↓
[Lifecycle]
  Design → Develop → Deploy → Operate → Evolve → Retire
  Technical debt at each stage
  Portfolio-level sequencing
        ↓
[Dynamics]
  Feedback loops (good and bad)
  Stock-and-flow (data freshness, model quality, trust, debt)
  Emergent properties (drift, bias amplification, reward hacking)
        ↓
[Governance]
  Decision rights + accountability + measurement + evolution policy
  Safety cases and failure-mode analysis
        ↓
[Realized System]
  A system the SDM student can defend as well-architected, operable, evolvable, and safe
```

This is the picture that the new `system_design_and_management.md` document will argue from. Each sub-skill is a slice of the picture.

---

## 4. Research Moves for Tier 1

### 4.1 AI System Architecture (Theme 1)

Questions:
- How does Crawley-Cameron decomposition apply to an AI marketing stack? What subsystems? What interfaces?
- What is the canonical OPM diagram for an AI marketing system?
- How do architecture trade-space tools (Tradespace Analysis Tool for Constellations / TAT-C, cost-capability frontiers) translate to AI?
- What are the dominant reference architectures for production AI marketing (MS AI Reference Architecture, AWS ML Well-Architected Framework, Google AI Adoption Framework)?
- What does an "architecture decision record" for AI look like?

Delegate to agent; return decomposition pattern, reference architectures, tradespace methods.

### 4.2 System Dynamics for AI (Theme 3)

Questions:
- What are the canonical stock-and-flow models for AI systems? (data quality → model quality → engagement → data quality; trust → usage → training data → model quality → trust)
- What feedback-trap patterns dominate in AI marketing? (Goodhart's law on KPIs, reward hacking, model-training loops where the model optimizes for its own measurement proxy)
- Recent papers that apply Forrester system dynamics to ML systems?
- Sterman's "shifting the burden" and "limits to growth" applied to AI org decisions?

Delegate to agent; return stock-and-flow patterns, named feedback traps, academic references.

### 4.3 Lifecycle Management and Technical Debt (Theme 5)

Questions:
- Deep-dive Sculley et al. NeurIPS 2015 *Hidden Technical Debt in Machine Learning Systems* — all 9 debt categories and what has been added in the decade since.
- ISO/IEC 5338 (AI system lifecycle processes) — scope, phases, deliverables.
- CRISP-DM vs. TDSP vs. modern MLOps maturity models (Google's ML Ops Maturity Model, Microsoft ML Ops Maturity).
- Model deprecation / replacement / migration patterns.
- Decade-long follow-up papers: Amershi et al. *Software Engineering for Machine Learning: A Case Study* (ICSE 2019), Polyzotis et al. *Data Management Challenges in Production ML* (SIGMOD 2018), *Hidden Technical Debt in AI* follow-ups.

Delegate to agent; return debt taxonomy + lifecycle standards + maturity models.

### 4.4 Platform Engineering and Team Topologies (Theme 7)

Questions:
- Skelton & Pais *Team Topologies* (2019) — four team types (Stream-aligned, Enabling, Complicated-Subsystem, Platform) mapped to AI organizations.
- Internal developer platforms (IDP) for AI: Backstage, Port, Humanitec patterns for ML.
- Platform-as-product thinking applied to AI platforms.
- Conway's Law in AI orgs — how the team structure shapes the model / pipeline architecture.
- Real-world case studies: Uber Michelangelo (and its cautionary tale), Airbnb Bighead, LinkedIn Pro-ML, Netflix Metaflow, Meta FBLearner, Spotify Hendrix/Klio.

Delegate to agent; return team-pattern mapping + platform case studies.

### 4.5 SRE, Safety Engineering, STAMP (Theme 8)

Questions:
- Google SRE book applied to AI — what's the same, what differs?
- Leveson STAMP/STPA — how does it adapt to AI systems whose state is a learned distribution?
- Chaos engineering for AI (Gremlin, Chaos Mesh for ML, prompt-perturbation testing).
- Incident response for AI: runbooks, rollback, canary patterns.
- Safety cases / assurance cases for AI systems.
- Recent academic papers on AI system reliability engineering.

Delegate to agent; return SRE-for-AI patterns + STAMP-for-AI + chaos-eng-for-AI + incident patterns.

---

## 5. Execution Order

1. **This plan document** (you are reading it) — seeded with Tier 1 themes
2. **Five parallel research agents**, one per Tier 1 theme, briefed with the §4 questions
3. **Direct web searches** for specific canonical references (Sculley 2015, Leveson STAMP, Team Topologies, Sterman) to double-source the agents
4. **Synthesis** into `docs/skills/system_design_and_management.md` — the third skill-axis document
5. **Refactor** `docs/skills/README.md` from two-lens (Sales + Productization) to three-lens (+ System Design & Management)
6. **Cross-link injection** — short callouts in sales_skill.md and productization_skill.md pointing to SDM sections where architectural / dynamics / lifecycle concerns apply
7. **Catalog and INDEX updates** — new sections for system architecture, system dynamics, technical debt, lifecycle, platform engineering, SRE/STAMP
8. **Validation** — link checks, cross-reference consistency
9. **Commit** — English-only conventional commit, subject-only, author Juhong Park

---

## 6. Success Criteria

The SDM-lens expansion is complete when an MIT SDM student reading the repo can answer, without leaving it:

1. "What is the system-architecture decomposition of a production AI marketing stack?"
2. "Which feedback loops in AI marketing most commonly drive long-run failure, and what system-dynamics patterns name them?"
3. "What technical-debt categories from Sculley 2015 apply here, and what has been added since?"
4. "How do Team Topologies map to an AI marketing organization?"
5. "What does an incident / safety case / STPA analysis for an AI marketing agent look like?"
6. "Which existing DDs are the methodological sources for each of the above?"

If those six questions have concrete, referenced answers in `system_design_and_management.md`, the round is a success.

---

## 7. Out of Scope

- Rewriting DD8 — DD8 remains the historical SDM-at-MIT reference section; the new doc extends into AI-specific territory
- Creating new frontier docs — the frontier directory stays as-is
- Translating any content to Korean — repo convention is English-only
- Speculative frameworks without published grounding — every claim must cite

---

## 8. Tier 2-3 Deferred

Tier 2 themes (socio-technical, complex-system cascades beyond Dao, governance-as-design, enterprise architecture, cognitive engineering) and Tier 3 themes (capacity economics, portfolio management, standards, safety engineering beyond STAMP, retirement) are captured here as the backlog. They will be executed only when the Tier 1 synthesis exposes concrete gaps that can't be closed without them.
