# Healthcare and Bio AI — Special-Topic Overview

**Scope**: A cross-repo special-topic survey of AI applied to healthcare and life sciences, adjacent to the main marketing-science axis. This document is the entry point to three companion documents that together cover the commercial, scientific, and governance layers of the domain. It is explicitly **not** a marketing-only treatment — healthcare AI is a socio-technical system whose commercial layer is inseparable from its regulatory, clinical, and scientific substrates.

**Position in the repo**: the marketing-science core (`docs/01-05`, `docs/deep_dive/`, `docs/frontier/`) treats healthcare only as one vertical inside `sales_skill.md §2.9`. This special topic expands the vertical into a first-class socio-technical treatment, then re-reads it through the three skill lenses (Sales · Productization · System Design & Management).

---

## 1. Why Healthcare AI Is Different

Healthcare / bio AI sits at the intersection of **all three skill lenses** under constraints the consumer-AI stack does not face:

| Dimension | Consumer marketing AI | Healthcare AI |
|---|---|---|
| Buyer / payer | Customer buys, advertiser pays | Prescriber selects, payer reimburses, patient consumes |
| Regulation | FTC, GDPR, EU AI Act (limited-risk) | FDA SaMD, EU MDR, EU AI Act (high-risk), HIPAA, state laws, ISO 13485, IEC 62304 |
| Data | First-party + consent-based | PHI under HIPAA; BAAs; de-identification; IRB |
| Failure mode | Lost revenue | Patient harm, litigation, regulatory action |
| Feedback latency | Minutes to weeks | Months to years; post-market surveillance required |
| Model lifecycle | Continuous retraining | Pre-authorized modification envelope (FDA PCCP) |
| Stakeholder graph | 3-5 nodes | 8-10 nodes (patient, clinician, payer, pharma, regulator, platform vendor, IRB, CIO, CMIO, researcher) |

The domain is also **the leading edge of AI regulation** — FDA's PCCP (Dec 2024), EU AI Act Annex III high-risk coupling with MDR/IVDR (effective Aug 2026), and WHO's Large Multi-Modal Models guidance (Jan 2024) are defining governance playbooks other industries will eventually adopt.

---

## 2. Document Map

This special topic splits into three companion documents. Read this overview first for orientation; then enter the companion most relevant to your question.

| Document | Coverage | When to enter |
|---|---|---|
| [`healthcare_commercial_ai.md`](healthcare_commercial_ai.md) | Pharma commercial AI (HCP targeting, omnichannel, MLR, DTC, adherence, market access) + provider / hospital AI (patient acquisition, engagement, CRM, operations, population health) | "How does AI move revenue and patient volume in pharma and hospitals?" |
| [`drug_discovery_and_clinical_ai.md`](drug_discovery_and_clinical_ai.md) | Biological foundation models (AlphaFold 3, ESM-3, Evo-2, Boltz, RoseTTAFold), AI-first biotech platforms, clinical-trial AI, RWE AI, medical imaging SaMD, ambient clinical scribes, medical LLMs, symptom triage, digital therapeutics | "What is the 2024-2026 scientific and clinical edge?" |
| [`healthcare_ai_regulation_and_sdm.md`](healthcare_ai_regulation_and_sdm.md) | HIPAA + FDA + EU MDR/IVDR + EU AI Act + WHO + NIST + ISO + state laws + CMS / payer rules; SDM-lens stakeholder topology, feedback loops, STAMP/STPA, case-study failures (Watson, Epic Sepsis, Babylon, Change Healthcare, Apple Watch AFib), platform engineering under HIPAA, Team Topologies for healthcare-AI orgs | "What must a healthcare-AI product clear before launch, and what is the system-design picture?" |
| [`HEALTHCARE_AI_RESEARCH_PLAN.md`](HEALTHCARE_AI_RESEARCH_PLAN.md) | The research plan that produced this set | Background / process trail |

---

## 3. Three-Lens Synthesis

The three skill lenses in `docs/skills/` re-read the same subject matter through different operating capabilities. Applied to healthcare AI, each lens reveals a different commercial geometry.

### 3.1 Sales Lens

Healthcare AI has **three distinct commercial motions**, each with its own buyer and its own AI stack:

1. **HCP-facing commercial** — pharma rep enablement and omnichannel engagement. Dominated by IQVIA (IQVIA.ai, unveiled at NVIDIA GTC March 2026, 150+ agents, 19 of top 20 pharmas), Veeva (AI Agents on Amazon Bedrock with Anthropic LLMs, GA Dec 2025), Salesforce Life Sciences Cloud for Customer Engagement + Agentforce 360 (GA Oct 2025). A second tier of AI-native specialists (Aktana — acquired by PharmaForceIQ Q1 2026; Komodo Health's Prism / Sentinel / MapAI; ZS ZAIDYN — Everest Group 2025 Leader; Swoop; Doceree; Eversana) supplies the decision layer. See `healthcare_commercial_ai.md §1-3`.

2. **Patient-facing commercial for providers** — health-system digital front door and CRM. Four-way competition: Salesforce Agentforce for Health (UChicago Medicine early adopter, June 2025), Innovaccer (Best in KLAS 2026 across three categories; $275M Series F Jan 2025 at ~$3.45B with Kaiser Permanente and Banner Health as strategic investors), Epic Cheers (bundled with Cosmos), Oracle Health (voice-first agentic EHR, Aug 2025). Qventus ($105M Series D from KKR Jan 2025), Hyro (45+ health systems incl. Sutter Health, Tampa General), LeanTaaS iQueue (Best in KLAS 2025 Capacity Optimization) hold the operations seams. See `healthcare_commercial_ai.md §4-6`.

3. **Payer-access commercial** — prior-authorization automation and HEOR AI. Cohere Health (12M+ PA requests/year, 98% provider satisfaction on Cohere Align, September 2025 ZignaAI acquisition) leads; CMS-4201-F (applicable Jan 1, 2024) bars AI as the **sole** basis for MA coverage denial, making this the most regulated commercial motion. See `healthcare_commercial_ai.md §7`.

The 2025 sales-metric baseline (IQVIA 2025: 20-50% rep efficiency, 5-10% revenue lift; Aktana case: 14% sales lift; ZAIDYN: 10 hrs/week saved per field rep; AllazoHealth: 12.2% adherence lift; Cohere: ~85% auto-approval) extends the [`sales_skill.md §2.9`](../skills/sales_skill.md) healthcare vertical with specific 2026 platforms.

### 3.2 Productization Lens

Productizing healthcare AI diverges from the consumer stack (see [`productization_skill.md`](../skills/productization_skill.md)) at four layers:

- **Model selection under PHI constraint** — open-model on-prem (Llama, Mistral variants, MedGemma, Meditron) vs. BAA-covered managed models (AWS Bedrock Claude, Azure OpenAI, Google Vertex AI Med-PaLM / MedGemini). A BAA is necessary but not sufficient — prompt logs, system prompts, and fine-tuning artifacts extend the PHI boundary and must be governed.
- **RAG and vector DBs under HIPAA** — requires de-identified corpora or BAA-governed managed services (AWS HealthLake, Google Cloud Healthcare API, Azure Health Data Services); re-identification risk under LLM membership inference makes Safe Harbor de-identification operationally brittle; serious vendors use Expert Determination + synthetic-data augmentation.
- **MLOps under post-market surveillance** — FDA's Predetermined Change Control Plan (PCCP, finalized 3 Dec 2024) reframes retraining as a **pre-authorized regulatory envelope** comprising a Description of Modifications + Modification Protocol + Impact Assessment, not an ops-only concern. ONC HTI-1 Final Rule (effective 31 Dec 2024) mandates 31 "source attributes" for Predictive Decision Support Interventions — a de-facto model-card requirement.
- **Evaluation and observability for clinical endpoints** — calibration, subgroup performance, drift on clinical outcomes replace consumer metrics. OpenAI HealthBench (May 2025, 48,562 rubric items across 5,000 physician-co-authored conversations) is the most significant 2025 benchmark development. See `drug_discovery_and_clinical_ai.md §5`.

### 3.3 SDM Lens

The SDM lens (see [`system_design_and_management.md`](../skills/system_design_and_management.md)) on healthcare AI produces the densest socio-technical picture in the repo:

- **Architecture tradespace** gains a hard **safety** axis alongside cost × latency × quality × risk × flexibility — FDA severity / harm classification (class I/II/III), EU MDR Rule 11 auto-classification of diagnostic/therapeutic software as Class IIa minimum, and the life-critical-endpoint class of failure.
- **System dynamics** exposes feedback loops unique to health: **Obermeyer et al.** (*Science* 2019) on cost-as-proxy-for-need producing racial under-enrollment (fix raised Black-patient high-risk enrollment from 17.7% to 46.5%); **Seyyed-Kalantari et al.** (*Nature Medicine* 2021) on chest-X-ray classifiers under-flagging Black/Hispanic/female/Medicaid patients; performative prediction (Perdomo et al., ICML 2020) applied to sepsis prediction erasing its own ground truth; Goodhart on HEDIS/Star measures.
- **Technical debt** inherits the full Sculley 2015 taxonomy plus a healthcare-specific layer: PCCP drift vs. authorized envelope, MDR post-market surveillance gaps (JAMA Health Forum 2025: only ~5% of FDA-cleared AI devices have any MAUDE adverse-event report), bias-audit debt for Section 1557 nondiscrimination.
- **Platform engineering and Team Topologies** map cleanly: stream-aligned clinical-product squads; a HIPAA-compliant ML platform team; a clinical-informatics / AI-safety enabling team; a regulatory-affairs complicated-subsystem team.
- **STAMP / STPA** treats the controller as {model + clinician + EHR + workflow policy}, making UCAs (Unsafe Control Actions) enumerable: missed alert, unsafe recommendation, wrong timing (post-deterioration), wrong duration (alert fatigue). See `healthcare_ai_regulation_and_sdm.md §5-6`.
- **Governance as system design** — five regulatory frames stack (HIPAA + FDA + EU MDR/IVDR + EU AI Act high-risk + state laws) plus CMS payer rules; a single EU AI-Act + MDR integrated conformity assessment through a dual-accredited notified body is now possible.

Canonical failure case studies: **IBM Watson for Oncology** (2018 STAT News investigation; IBM sold Watson Health to Francisco Partners for rebranding as Merative, 21 Jan 2022); **Epic Sepsis Model v1** (Wong et al., *JAMA Internal Medicine* 2021: sensitivity 33% vs. claimed AUC 0.76-0.83); **Babylon Health** (administration 11 Sep 2023, assets sold for £500k to eMed); **Apple Watch AFib** base-rate failure (sensitivity ~95%, PPV ~16% at 1% population prevalence); **Change Healthcare ransomware** (12-21 Feb 2024, ~100M records, one-in-three US claim flows halted — platform concentration risk). See `healthcare_ai_regulation_and_sdm.md §3`.

---

## 4. What Was Already in the Repo, What Is New

### Already present

- `sales_skill.md §2.9` — Healthcare / Life Sciences (HCP Targeting) — IQVIA, Veeva, Salesforce LSC4CE, Swoop, Doceree; 2025 market metrics.
- `regulatory_appendix.md §7` — sector-specific note that FDA regulates AI claims that cross into clinical decision support, and LLM patient-education content tied to disease claims is scrutinized.

### New in this special topic

- The complete pharma commercial AI vendor landscape beyond HCP targeting — MLR content AI (Veeva Quick Check Agent, Eversana Orchestrate MLR), DTC advertising under the September 2025 FDA OPDP enforcement wave, patient adherence (AllazoHealth, Phreesia, Accredo), payer AI (Cohere Health).
- Provider / hospital AI — a domain the repo previously did not treat at all: digital front door, health-system CRM, hospital operations, care-gap closure.
- Bio foundation models (AlphaFold 3, ESM-3, Evo / Evo-2, Boltz-1 / Boltz-2 — the last two from MIT Jameel Clinic) and AI-first biotech (Insilico Medicine rentosertib Phase 2a readout, Recursion-Exscientia, Xaira, Isomorphic, Generate, Absci, Inceptive).
- Clinical AI product landscape — the FDA-cleared SaMD list (~1,450 devices by Dec 2025), ambient scribes (Microsoft Dragon Copilot, Abridge $5.3B, Suki, Epic AI Charting Feb 2026), medical LLMs (Med-PaLM 2, Med-Gemini, AMIE, HealthBench, OpenEvidence $12B Jan 2026, Hippocratic AI), DTx (Akili EndeavorRx / EndeavorOTC, Big Health CMS DMHT codes Jan 2025, Pear Therapeutics bankruptcy April 2023).
- The healthcare-AI regulatory stack as a coherent story: HIPAA 2024-2026, FDA PCCP + Jan 2025 draft guidance + 6 Jan 2026 revised CDS guidance, EU AI Act × MDR/IVDR coupling, WHO LMM guidance, NIST + ISO + state laws + CMS-4201-F.
- SDM-lens treatment of healthcare AI as a socio-technical system — stakeholder topology, named feedback loops from peer-reviewed literature, failure case studies, platform engineering under HIPAA, Team Topologies mapping, STAMP/STPA for clinical AI.

### Intentionally out of scope

- Deep-dive into clinical decision-support algorithms beyond vendor / device landscape
- Consumer wellness / fitness app marketing (covered in `frontier/generative_ai_and_llms_in_marketing.md`)
- Standalone telemedicine business models
- Pure biological foundation models with no commercial or marketing connection
- Implementation code or production configs
- Korean-language content (repo is English-only)

---

## 5. How to Find the Detail

- For specific vendors and platforms — enter `healthcare_commercial_ai.md` (HCP + provider) or `drug_discovery_and_clinical_ai.md` (bio + clinical).
- For specific regulations or standards — enter `healthcare_ai_regulation_and_sdm.md §1-4`.
- For socio-technical analysis and case studies — enter `healthcare_ai_regulation_and_sdm.md §5-8`.
- For the full paper / vendor / model catalog — see `docs/CATALOG.md` § Healthcare / Bio AI.

---

## 6. Open Questions

The special-topic treatment surfaces questions the marketing-science core cannot answer alone:

1. **Commercialization pathway from bio-AI discovery to prescriber uplift** — Insilico's rentosertib is the first AI-discovered + AI-designed drug to post a positive Phase 2a readout (*Nature Medicine*, June 2025). What is the shortest line from a bio-AI discovery to an HCP-targeting campaign? The repo covers both ends but not the connective tissue.
2. **Governance of generative AI in MLR** — Veeva Quick Check / Eversana Orchestrate MLR are production; FDA and EMA have not issued definitive guidance on LLM-generated promotional content. The compliance layer is evolving under vendor-defined patterns.
3. **Payer-AI causality and Section 1557** — CMS-4201-F bars AI-only denials; the naviHealth / UnitedHealth and Humana class actions are still shaping the *how much* AI can inform coverage decisions. Causal-inference methods from `DD5` (Bayesian MMM, uplift) may transfer here.
4. **FDA PCCP as an architecture decision record** — is PCCP the healthcare-specific form of an ADR? If yes, the SDM lens should treat it as a canonical example of *governance as system design* (§7 of the SDM doc).
5. **STAMP/STPA for healthcare AI specifically** — Mylius (June 2025) and subsequent arXiv work treat frontier AI; a peer-reviewed clinical-AI-specific STPA paper at the same quality level is not cleanly identifiable in 2024-2026 — a thesis-scale gap.

These open questions are natural follow-ups when this special topic is extended.

---

## 7. Maintenance Notes

When updating this overview:

1. Keep the table in §2 synchronized with the companion documents' actual scope.
2. Keep the three-lens §3 summary cross-linked to `sales_skill.md §2.9`, `productization_skill.md`, `system_design_and_management.md`, and `regulatory_appendix.md`.
3. Any new healthcare-AI vendor, model, or regulation added to one of the companions should be reflected in `docs/CATALOG.md` and referenced here only if it meaningfully changes the three-lens picture.
4. Treat all vendor-published performance metrics as vendor-attributed unless double-sourced by a peer-reviewed journal or independent analyst (KLAS, HIMSS, Rock Health, Everest Group).
5. Update INDEX.md when companion documents are added or split further.
