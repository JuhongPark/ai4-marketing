# Healthcare / Bio AI — Special-Topic Research Plan

This plan extends the repo with a **special-topic survey** of healthcare and bio AI, adjacent to the main marketing-science axis. The repo's existing coverage treats healthcare only as a vertical sub-skill inside `sales_skill.md §2.9` (HCP targeting). This plan adds a first-class treatment of healthcare AI as a socio-technical domain, keeps the marketing-science lens where it applies, and uses the three-lens framework (Sales · Productization · System Design & Management) to re-read the domain.

---

## 1. Why a Special Topic

Healthcare / bio AI sits at the intersection of **all three skill lenses** but under constraints the consumer-AI stack does not face:

- **Sales lens** — revenue model is payer / prescriber mediated, not direct-to-consumer. HCP (healthcare-professional) targeting, medical-science-liaison (MSL) enablement, DTC (direct-to-consumer drug) advertising, and patient-acquisition marketing for hospitals are all regulated.
- **Productization lens** — HIPAA, FDA Software-as-a-Medical-Device (SaMD), EU MDR / IVDR, and ISO/IEC 62304 reshape every layer from data ingestion to observability. De-identification, Business Associate Agreements (BAAs), audit logs, and on-prem / sovereign-cloud requirements are table stakes.
- **SDM lens** — stakeholder topology is the most complex in any AI domain (patient, provider, payer, pharma, regulator, platform vendor, IRB). Feedback loops have direct safety consequences. Lifecycle includes clinical validation and post-market surveillance, not just A/B testing.

The domain is also **the leading edge of AI regulation** — FDA's AI/ML Action Plan, Predetermined Change Control Plans (PCCP), EU AI Act Annex III (high-risk), and WHO guidance on large multimodal models are defining the governance playbook every other industry will eventually adopt.

---

## 2. Scope

### In-Scope

1. **Pharma commercial AI (the marketing-adjacent core)** — HCP targeting, omnichannel orchestration, MLR (medical/legal/review) content AI, DTC advertising, MSL enablement, patient-adherence marketing
2. **Provider / hospital AI (the buyer-side of commercial pharma)** — patient acquisition marketing for health systems, patient engagement chatbots, health-system CRM, hospital operations AI
3. **Drug discovery and bio foundation models (upstream context)** — AlphaFold 3, ESM-3, Evo-2, RoseTTAFold, OpenFold; AI-first biotech (Insilico, Recursion, Xaira, Absci, Isomorphic, Generate, Inceptive); clinical-trial AI; real-world evidence (RWE) AI
4. **Clinical AI (patient-facing and provider-facing)** — medical imaging SaMD, EHR NLP, ambient scribes, medical LLMs (Med-PaLM 2, MedGemini, Glass Health, OpenEvidence, Hippocratic AI), symptom triage, digital therapeutics
5. **Regulatory and ethical landscape** — HIPAA, FDA SaMD / GMLP / PCCP, EU MDR / IVDR, EU AI Act Annex III, WHO 2024 LMM guidance, NIST AI RMF for health, ISO 81001 / 13485, IEC 62304
6. **SDM lens on health AI** — stakeholder architecture, feedback loops (underdiagnosis, disparity amplification), case studies (IBM Watson Health, Epic sepsis model, Babylon Health), STAMP / STPA applied to medical AI, platform engineering under HIPAA
7. **Three-lens synthesis** — how Sales, Productization, and SDM concerns differ from consumer-AI marketing

### Out-of-Scope (or brief mention only)

- Consumer wellness / fitness app marketing (covered adequately by `docs/frontier/generative_ai_and_llms_in_marketing.md`)
- Standalone telemedicine business models
- Pure biological foundation models with no commercial or marketing connection
- Korean-language content (repo convention is English-only)
- Implementation tutorials or code examples (research-synthesis repo, not implementation repo)

---

## 3. Research Themes

### Theme 1 — Pharma Commercial AI

**Questions**
- State of the art in HCP targeting and next-best-action (NBA) beyond the IQVIA / Veeva / Salesforce LSC4CE platforms already mentioned in `sales_skill.md §2.9`
- Omnichannel orchestration and trigger-based engagement — which vendors, which architectures, which evaluation metrics
- MLR-compliant content generation — how pharma uses LLMs under Promotional Regulatory Compliance (PRC), with human-in-the-loop review
- DTC pharmaceutical advertising AI — creative optimization, channel mix, regulatory constraints (FDA OPDP for US, ABPI for UK)
- Patient adherence marketing — where digital marketing crosses into clinical
- Key players beyond the existing trio: Aktana, Komodo Health, Swoop, Doceree, ZS Associates, Eversana, BCG X, Reverba, PharmaLex

**Success signal**: a sub-section that names specific 2024-2026 platforms, published outcome metrics, and production architectures.

### Theme 2 — Provider / Hospital AI

**Questions**
- Patient-acquisition marketing for health systems — SEO, programmatic, intent data for care-seeking, reputation management
- Patient engagement at scale — chatbots (Hyro, Conversa), appointment optimization, digital front door
- Health-system CRM — Salesforce Health Cloud, Innovaccer, Epic (native), Oracle Health (Cerner), Mercury Healthcare, Actium
- Hospital operational AI — scheduling, capacity, bed management, OR utilization
- Clinical decision support (context only; full deep-dive is out of scope)
- Care-gap closure and population-health marketing

**Success signal**: a sub-section mapping the provider-AI stack from marketing front-door through engagement to operations, with vendor examples.

### Theme 3 — Drug Discovery and Bio Foundation Models

**Questions**
- Biological foundation models: AlphaFold 3 (DeepMind, 2024), ESM-3 (EvolutionaryScale, 2024), Evo / Evo-2 (Arc Institute, 2024-2025), RoseTTAFold All-Atom, OpenFold, AlphaMissense
- AI-first biotech platforms: Insilico Medicine, Recursion Pharmaceuticals, Absci, Xaira Therapeutics, Isomorphic Labs, Generate Biomedicines, Inceptive, Valence (now Recursion), BenevolentAI
- Clinical trial AI — patient recruitment ML, synthetic control arms, adaptive trials, real-world evidence (RWE) AI
- MIT-specific research: Regina Barzilay and Tommi Jaakkola (CSAIL, MIT Jameel Clinic), Manolis Kellis (CSAIL, computational biology), Jim Collins (MIT IMES, Antibiotics AI Project), Caroline Uhler (MIT IDSS, biology + causal inference), MIT Broad Institute collaboration

**Success signal**: a sub-section that maps the current bio-AI platform landscape and its commercialization pathway (why these discoveries eventually require commercial / marketing AI to reach patients).

### Theme 4 — Clinical AI and Medical LLMs

**Questions**
- Medical imaging AI — FDA-cleared SaMD landscape (Aidoc, Viz.ai, Paige, PathAI, HeartFlow, Zebra Medical / Nanox AI, Rad AI), 2024-2026 device authorizations
- EHR NLP and ambient scribes — Nuance DAX Copilot (Microsoft), Abridge, Suki, Augmedix, DeepScribe, Microsoft Dragon Copilot, Amazon HealthScribe
- Medical LLMs — Med-PaLM 2 (Google), MedGemini (Google), Glass Health, OpenEvidence, Hippocratic AI, Amie (Google multimodal diagnostic), John Snow Labs Medical LLM
- Symptom triage and patient-facing chat — Ada, K Health, Buoy, with safety-concern case studies (Babylon Health collapse)
- Digital therapeutics (DTx) — Akili (AKL-T01), Big Health, Pear Therapeutics (2023 bankruptcy) as a cautionary case

**Success signal**: a sub-section with specific devices, FDA clearance numbers where known, and published outcome evidence.

### Theme 5 — Healthcare AI Regulation

**Questions**
- HIPAA updates 2024-2026 and their intersection with AI (reproductive-health privacy rule 2024, proposed security rule update 2024-2025)
- FDA SaMD framework, Good Machine Learning Practice (GMLP) guiding principles (FDA / Health Canada / MHRA 2021 and subsequent updates), Predetermined Change Control Plan (PCCP, 2024 final guidance)
- FDA's January 2025 draft guidance on AI-enabled device software functions, and the list of authorized AI/ML-enabled medical devices (FDA public list, ~950+ as of 2024)
- EU MDR and IVDR applicability to AI medical devices, notified-body pathways
- EU AI Act Annex III — health as high-risk category; conformance requirements effective 2026-2027
- WHO guidance on AI for health (2021 ethics/governance, 2024 LMM guidance)
- NIST AI RMF and its 2024 GenAI Profile applied to health AI
- ISO/IEC 81001-5-1 (health software security), ISO 13485 (QMS), IEC 62304 (medical device software)

**Success signal**: a regulatory sub-section that could serve as the compliance checklist for a healthcare-AI product launch, with back-link to `skills/regulatory_appendix.md`.

### Theme 6 — SDM Lens on Healthcare AI

**Questions**
- Socio-technical stakeholder topology: patient, provider, payer, pharma, regulator, platform vendor, IRB / ethics board, hospital CIO, chief medical officer — decision rights and accountability graph
- Feedback loops unique to health: underdiagnosis reinforcement, disparity amplification (race/gender/insurance), reimbursement-driven model bias
- Case studies in system failure:
  - IBM Watson Health cancer-recommendation scandal (2018-2020) — hallucinated treatment recommendations, scope-overreach, eventual divestiture to Francisco Partners as Merative (2022)
  - Epic sepsis model (2021 JAMA Internal Medicine study) — widely deployed model with sensitivity 33%, far below claimed performance
  - Babylon Health collapse (2023) — diagnostic AI at consumer scale, regulatory and clinical governance failure
- STAMP / STPA applied to clinical AI (unsafe control actions in a model-in-the-loop clinical workflow)
- Platform engineering under HIPAA: de-identification engineering, federated learning for multi-hospital data, sovereign-cloud patterns (AWS HealthLake, Google Healthcare API, Azure Health Data Services)
- Lifecycle specifics: FDA PCCP as a first-class lifecycle artifact, post-market surveillance, model card + algorithmic accountability requirements

**Success signal**: a sub-section that could be handed to an MIT SDM student as the architecture-and-governance overview of healthcare AI.

### Theme 7 — Three-Lens Application

**Sales**
- How HCP-targeting AI expands the salable market (named 2025 metrics already in `sales_skill.md §2.9`)
- Patient-acquisition AI for providers as a distinct commercial motion
- Payer-access marketing (market access and prior-authorization AI) as the third commercial motion

**Productization**
- Foundation-model selection under PHI constraint (closed-model cloud vs. open-model on-prem)
- Vector DB and RAG engineering under HIPAA (de-identified vs. BAA-covered managed services)
- MLOps under post-market surveillance (PCCP as MLOps policy artifact)
- Observability and evaluation specific to clinical AI (calibration, subgroup performance, drift detection on clinical endpoints)

**SDM**
- Architecture tradespace under safety constraints: quality × latency × cost × *safety* × regulatory burden
- System dynamics: performative-prediction risk in diagnosis, Goodhart's law on reimbursement-driven endpoints
- Governance: FDA PCCP as an architecture decision record, IRB / ethics board as part of the decision-rights graph

---

## 4. Execution Method

1. **Six parallel research agents**, one per Theme 1-6, briefed with the §3 questions and a ~600-word response cap.
2. **Double-source** every named metric, device clearance, and outcome claim through a second citation where possible.
3. **Cross-reference discipline** — every new paper / tool / platform added to `docs/CATALOG.md` with a "Discussed in" back-link.
4. **Synthesis pass** — integrate agent outputs into `docs/special/healthcare_bio_ai.md` using the three-lens framework as the connective tissue.
5. **Index update** — new Special Topics row in `INDEX.md` document map; healthcare entries in the By Skill table.
6. **Validation** — all internal links, INDEX entries, CATALOG entries checked for consistency.
7. **Commit** — English-only conventional commit, subject-only, author Juhong Park.

---

## 5. Deliverables

- `docs/special/HEALTHCARE_AI_RESEARCH_PLAN.md` (this document)
- `docs/special/healthcare_bio_ai.md` — primary survey document integrating all six themes
- Possible split into sub-documents if length exceeds ~700 lines:
  - `healthcare_commercial_ai.md` (Themes 1-2)
  - `drug_discovery_and_clinical_ai.md` (Themes 3-4)
  - `healthcare_ai_regulation_and_sdm.md` (Themes 5-6 and Theme 7 synthesis)
- `docs/CATALOG.md` — new section "Healthcare / Bio AI" with ~20-40 new references
- `INDEX.md` — new Special Topics section with back-links
- `README.md` — short Project Structure update mentioning special topics

---

## 6. Success Criteria

The special-topic extension is complete when a reader can answer, without leaving the repo:

1. "What is the 2024-2026 state of the art in pharma commercial AI, and which platforms dominate?"
2. "How do patient-acquisition AI and health-system CRM differ from consumer marketing AI?"
3. "Which biological foundation models exist, and which AI-first biotech platforms depend on them?"
4. "What is the current FDA-cleared medical AI landscape, and what are the leading ambient-scribe and medical-LLM products?"
5. "What regulatory stack does a healthcare-AI product launch need to clear (US + EU + WHO + ISO)?"
6. "What does an SDM architecture, dynamics, and governance analysis of a healthcare-AI stack look like, with named failure case studies?"
7. "How do the Sales, Productization, and SDM lenses shift when applied to healthcare rather than consumer marketing?"

---

## 7. Out-of-Scope for This Round

- Rewriting existing DDs or frontier docs unless required for cross-link consistency
- Creating implementation code or production configs
- Korean-language content (see `CLAUDE.md` — repo is English-only)
- Deep-dive into clinical decision support algorithms beyond vendor / device landscape
- MIT-internal course material or proprietary research
