# Regulatory and AI-Governance Appendix

A cross-axis appendix referenced from both `sales_skill.md` (all sub-skills that touch customer data or capital decisions) and `productization_skill.md` §3.6 (Privacy and Governance Infrastructure). The frameworks below apply whether you are selling, investing, or building.

---

## 1. Risk-Tier Summary

The table below maps common AI-marketing activities to the four principal regulatory frames relevant in 2026.

| Activity | EU AI Act Tier | NIST AI RMF Function | FTC Priority | Key Obligation |
|---|---|---|---|---|
| Subliminal / manipulative personalization | **Prohibited (Art. 5)** | GOVERN + MAP | Section 5 "unfair" | Banned Feb 2025 |
| Emotional-state inference (ads) | Prohibited in workplace / education; high-risk elsewhere | MEASURE (bias, efficacy) | Dark-patterns enforcement | Disclosure + purpose limit |
| Social scoring for service access | **Prohibited** | GOVERN | — | Banned Feb 2025 |
| Automated personalized pricing | Limited-risk; high-risk if affecting "essential services" | MEASURE (disparate impact) | Algorithmic pricing investigations | Substantiation + non-discrimination |
| Generative ad creative (synthetic media) | Limited-risk (transparency) | MAP (provenance) + MANAGE | Deceptive AI claims, endorsement guides | **Label AI-generated content (Aug 2026)** |
| Chatbots / conversational commerce | Limited-risk | MANAGE (incident response) | CAN-SPAM, endorsement, dark patterns | Disclose non-human identity |
| Behavioral targeting of minors | **Prohibited** (profiling-based) | GOVERN + MAP | COPPA 2025 amendments | Parental consent; no targeted ads |
| Biometric categorization for ad targeting | **Prohibited** (race / religion / orientation inference) | MAP | Section 5 | Banned Feb 2025 |
| Healthcare LLM marketing claims | High-risk if clinical decision support | MEASURE (accuracy claims) | FDA + FTC joint oversight | Substantiation; disease-claim rules |
| Credit / insurance ad eligibility filtering | **High-risk** (Annex III) | Full NIST lifecycle | ECOA, Reg B | Conformity assessment + logging |

---

## 2. NIST AI RMF and 2024 Generative AI Profile — What It Prescribes

The four NIST functions map to marketing-AI artifacts as:

- **GOVERN**: AI policy, procurement rules for martech vendors, RACI for campaign launches
- **MAP**: per-campaign context documentation (audience, data lineage, model card for personalization engines)
- **MEASURE**: A/B testing with fairness metrics, hallucination rates for generative copy, uplift + disparate-impact testing
- **MANAGE**: incident runbook (synthetic media misuse, prompt injection on chat agents), rollback, human review thresholds

The **2024 GenAI Profile (NIST AI 600-1)** adds twelve GenAI-specific risks — confabulation, data privacy, IP infringement, obscene / violent content, value-chain opacity — all directly relevant to creative generation pipelines.

---

## 3. EU AI Act — 2025-2027 Obligation Timeline

- **Feb 2025**: Prohibited-practice rules went into force — subliminal manipulation, social scoring, biometric categorization for sensitive inference are banned
- **Aug 2025**: General-Purpose AI (GPAI) obligations for foundation-model providers — documentation, training-data summary, downstream-deployer support
- **Aug 2026**: Transparency obligations for generative / manipulated content — AI-generated text, images, audio, video must be labeled
- **Aug 2026-2027**: High-risk AI system conformity assessments for Annex III use cases
- **Extraterritoriality (Art. 2)**: US-hosted marketing AI that serves EU users is in scope

For credit, insurance, and service-access ad-eligibility filtering, the full high-risk obligation stack applies: conformity assessment, fundamental-rights impact assessment, risk-management system, data governance, technical documentation, record-keeping, transparency to deployers, human oversight, accuracy / robustness / cybersecurity.

---

## 4. FTC Priorities (2023-2026)

The current FTC enforcement stance is that AI claims are claims — they must be substantiated.

- **Operation AI Comply (Sep 2024)**: first wave of enforcement against deceptive AI claims and AI-enabled schemes. Operative precedent for any "AI-powered" marketing claim
- **COPPA 2025 amendments**: tighten targeted advertising to children under 13; new data minimization rule
- **Algorithmic pricing**: FTC investigates personalized / AI-driven pricing under unfair-practice doctrine
- **Substantiation**: AI efficacy claims ("30% more conversion," "AI-generated better than human") require evidence
- **CAN-SPAM + AI**: AI-generated commercial email still subject to CAN-SPAM identification and opt-out rules
- **Dark patterns**: AI-accelerated personalization that steers users toward choices they would not otherwise make is under scrutiny

---

## 5. ISO 42001 and Management System Certification

ISO / IEC 42001:2023 defines an AI management system. In 2026 it is becoming an **RFP gate** for B2B AI vendors selling into enterprise, healthcare, and financial services. Treat it as certification readiness work — similar to SOC 2 Type II but scoped to AI lifecycle governance.

---

## 6. US State-Law Fragmentation (Live in 2026)

No federal preemption is in sight. State-level obligations that affect marketing AI:

- **Colorado AI Act** (effective Feb 2026) — algorithmic discrimination impact assessments
- **California AB 2013** — training-data transparency for generative models
- **Texas TRAIGA** — risk-based governance requirements
- Expect additional state-level moves through 2026-2027

---

## 7. Sector-Specific Nuances

- **Healthcare marketing**: FDA regulates AI claims that cross into clinical-decision-support territory; LLM-generated patient-education content tied to specific disease claims is particularly scrutinized
- **Financial services**: CFPB adverse-action guidance (2023+) applies to AI-driven credit denials; Reg Z / BSA / AML overlays apply to AI-augmented lending marketing
- **Children's marketing**: COPPA 2025 tightens, but **LLM-driven lookalike modeling can re-identify minors' cohorts** — audit tooling lags the underlying risk
- **Capital-markets communication**: Regulation FD, MAR (EU), emerging AI-disclosure rules around earnings-prep LLMs

---

## 8. Operating Checklist (2026 Minimum Bar)

A marketing team running AI in 2026 should be able to affirmatively answer:

1. Have we **inventoried every marketing model** against EU AI Act Annex III and Art. 5 prohibitions? Any model touching minors, biometrics, or emotion inference is flagged?
2. Is **AI-content labeling** (C2PA or equivalent provenance) deployed before the August 2026 EU GPAI transparency deadline?
3. Do we maintain an **AI incident log** (NIST MANAGE) with ≥6-month retention for synthetic-media misuse, prompt injection, chatbot failures?
4. Do we have **substantiation files** for every AI claim ("AI-powered," "personalized," efficacy numbers)? FTC has fined firms for unbacked AI claims since 2024
5. Do our **vendor DPAs and model cards** cover every martech tool using foundation models? Disclosure of training-data lineage included?
6. Do we run **bias / disparate-impact testing** quarterly on targeting and pricing models with documented thresholds?
7. Are we pursuing **ISO 42001 readiness** if we sell B2B to regulated buyers?
8. Is our **child-directed content** under 13 reviewed under COPPA 2025 amendments (no targeted ads, new data minimization)?

---

## 9. Open Regulatory Problems

- **Cross-border arbitrage**: US-hosted personalization serving EU users triggers AI Act extraterritoriality (Art. 2); enforcement coordination with FTC / state AGs unresolved
- **GPAI upstream liability**: when a marketer fine-tunes OpenAI / Anthropic for campaign copy, allocation of transparency / documentation duty under AI Act Art. 53 vs. deployer obligations is contested
- **State-law fragmentation** in the US — no federal preemption near-term
- **"Manipulation" threshold ambiguity**: Art. 5(1)(a) bans manipulative AI but offers no quantitative test; persuasion-vs-manipulation line unsettled for personalization
- **Substantiation of emergent-behavior claims**: FTC expects evidence, but foundation-model outputs are non-deterministic; no settled methodology
- **Children's data in shared-audience platforms**: COPPA 2025 tightens, but lookalike modeling can re-identify minors' cohorts; audit tooling lags

---

## 10. References

### Core Framework Documents
- NIST AI RMF 1.0 — [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
- NIST AI 600-1 GenAI Profile — [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- EU AI Act (Regulation 2024/1689) — [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- EU AI Act implementation timeline — [artificialintelligenceact.eu](https://artificialintelligenceact.eu/implementation-timeline/)
- ISO / IEC 42001:2023 — [iso.org](https://www.iso.org/standard/81230.html)
- EC AI Act portal — [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- NIST AI RMF Roadmap — [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework/roadmap-nist-artificial-intelligence-risk-management-framework-ai)
- GDPR Art. 22 (automated decisions) — [gdpr.eu](https://gdpr.eu/recital-71-profiling/)
- WEF AI Governance Alliance — [weforum.org](https://www.weforum.org/publications/ai-governance-alliance/)

### US Enforcement and Guidance
- FTC. *AI Companies: Uphold Your Privacy, Confidentiality Commitments* (2024) — [ftc.gov](https://www.ftc.gov/business-guidance/blog/2024/06/ai-companies-uphold-your-privacy-confidentiality-commitments)
- FTC. *Operation AI Comply — Crackdown on Deceptive AI Claims* (Sep 2024) — [ftc.gov](https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes)
- FTC. *COPPA rule* — [ftc.gov](https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa)
- CFPB. *Credit Denials and AI* — [consumerfinance.gov](https://www.consumerfinance.gov/about-us/newsroom/cfpb-issues-guidance-credit-denials-lenders-using-artificial-intelligence/)
- FTC. *FCRA / Consumer Reports Guidance* — [ftc.gov](https://www.ftc.gov/business-guidance/resources/using-consumer-reports-what-landlords-need-know)

### Sector-Specific
- FDA. *AI/ML-Enabled Medical Devices* — [fda.gov](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)

---

## 11. How to Use This Appendix

- **Sales teams** entering EU or high-risk markets: §1 risk-tier table + §3 EU timeline + §8 checklist. Feed the outputs into MMM/uplift decisions and investor Q&A.
- **Product teams** building AI: §2 NIST mapping + §3 EU timeline + §5 ISO 42001 + §8 checklist. Bake these into the backlog, not into a one-off compliance sprint.
- **Executives** preparing for fundraising or M&A: §4 FTC posture + §6 state-law fragmentation + §8 checklist are the likely diligence questions.
- **Legal / compliance**: §7 sector-specific nuances and §9 open problems flag the unresolved interpretations you should track actively.

Treat this appendix as the shared reference any new AI-marketing decision should be checked against before shipping. Keep it updated as major guidance (FTC enforcement actions, EU AI Act secondary legislation, state-law rollouts) changes.
