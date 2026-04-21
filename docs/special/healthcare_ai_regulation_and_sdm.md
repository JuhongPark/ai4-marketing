# Healthcare AI Regulation and System Design & Management

**Scope**: The five-frame regulatory stack a healthcare-AI product must clear (HIPAA + FDA + EU MDR/IVDR + EU AI Act + WHO/NIST/ISO) plus state law and CMS payer rules (Part A); the SDM / socio-technical lens on healthcare AI — stakeholder topology, feedback loops, case-study failures, STAMP/STPA, platform engineering, Team Topologies, lifecycle artifacts (Part B). Paired with [`healthcare_commercial_ai.md`](healthcare_commercial_ai.md) and [`drug_discovery_and_clinical_ai.md`](drug_discovery_and_clinical_ai.md). See [`healthcare_bio_ai.md`](healthcare_bio_ai.md) for the overview.

**Relation to existing repo**: extends [`regulatory_appendix.md`](../skills/regulatory_appendix.md) (the consumer-marketing-centered cross-axis appendix) and [`system_design_and_management.md`](../skills/system_design_and_management.md) (the SDM skill lens) with healthcare-specific frames, case studies, and artifacts.

---

## Part A — Regulatory Stack

### 1. HIPAA (US)

Two 2024-2026 updates materially affect healthcare AI:

- **HIPAA Privacy Rule to Support Reproductive Health Care Privacy** — HHS OCR final rule **22 April 2024 (89 FR 32976)**, effective 25 June 2024 with compliance by 23 December 2024 and §164.520 by 16 February 2026. The U.S. District Court for the Northern District of Texas **vacated most of the rule nationally on 18 June 2025**; enforceability uncertain in 2026.
- **HIPAA Security Rule NPRM** — **27 December 2024 (90 FR 898)**, comment period closed 7 March 2025. First major Security Rule update since 2013. Removes the "addressable" designation and explicitly requires encryption at rest and in transit, MFA, anti-malware, network segmentation, semi-annual vulnerability scans, annual penetration testing, and that **AI systems that create, receive, maintain, transmit, or interact with ePHI appear on a technology-asset inventory and be risk-assessed accordingly**.

Parallel to this, **Section 1557 ACA** final rule (**6 May 2024**) extended nondiscrimination obligations to "patient care decision support tools" effective **1 May 2025**, making covered entities — **not tool vendors** — liable for biased AI outputs.

For LLM training data, the 2012 HHS OCR de-identification guidance still governs. **Safe Harbor** (18-identifier strip) is operationally brittle against modern re-identification — combination attacks using ZIP-3, birth year, and rare diagnoses remain feasible, and LLM membership inference raises the residual risk further. Serious vendors use **Expert Determination** with documented statistical re-identification risk analyses, synthetic-data augmentation, and limited-data-set + DUA combinations.

### 2. FDA AI/ML Framework

The device-side stack has moved from the 2019 *AI/ML SaMD Action Plan* and the 2021 joint **FDA / Health Canada / MHRA GMLP guiding principles** (10 principles, updated with the **IMDRF GMLP final document January 2025**) into operational guidance:

- **Predetermined Change Control Plan (PCCP)** — *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions*, **finalized 3 December 2024** (Federal Register 2024-28361). Scope expanded beyond machine learning to all AI functions. Structure: **Description of Modifications + Modification Protocol + Impact Assessment**. Enables pre-authorized post-market modifications — a first-class regulatory lifecycle artifact distinct from ops-only retraining.
- **AI-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations** — **draft guidance 7 January 2025**, comments closed 7 April 2025. Applies Total Product Life Cycle lens covering model description, data lineage, bias analysis, human-AI workflow, transparency, monitoring. Finalization expected during 2026; not yet issued as of April 2026.
- **Public list of AI/ML-enabled medical devices** — 1,016 authorizations by mid-2024, ~1,250 by July 2025, 1,451 by the December 2025 refresh (single-year record ~295 in 2025). Still dominated by radiology.
- **Digital Health Advisory Committee** — inaugural meeting 20-21 November 2024 on generative-AI device safety; second meeting 6 November 2025 on generative-AI-enabled digital mental-health devices.
- **Clinical Decision Support** — **21st Century Cures §3060** exemption narrowed by FDA's 2022 final CDS guidance and further revised **6 January 2026**.

### 3. EU MDR / IVDR and EU AI Act

Classification and coupling:

- **MDR 2017/745 Rule 11** (Annex VIII) classifies diagnostic/therapeutic SaMD as **Class IIa minimum** — Class IIb if serious deterioration possible, Class III if death or irreversible harm — which pushes almost all clinical AI into notified-body conformity assessment. **MDCG 2019-11 Rev.1** (June 2025) clarifies qualification and classification including Medical Device AI.
- **IVDR 2017/746** imposes equivalent tightening on AI-enabled diagnostics.
- **EU AI Act (Regulation 2024/1689)** entered into force **1 August 2024**. Under **Article 6(1)**, any AI system that is a safety component of — or is itself — a product requiring third-party conformity assessment under MDR/IVDR is **automatically high-risk**. Full high-risk obligations (risk management, data governance, technical documentation, logging, human oversight, robustness, post-market monitoring) apply from **2 August 2026**; longer runway (to 2 August 2027) possible for most Annex I systems with further delays under the Digital Omnibus.
- A single **integrated conformity assessment** through a notified body accredited under both MDR/IVDR and the AI Act is permitted.

EMA: **Reflection Paper on the Use of AI in the Medicinal Product Lifecycle** finalized **9 September 2024** (CHMP/CVMP); replaces draft terminology ("high risk" → "high patient risk" / "high regulatory impact") and underpins risk-based expectations across discovery, trials, manufacturing, pharmacovigilance.

### 4. WHO, NIST, ISO, State Law, CMS

- **WHO** — 2021 *Ethics and Governance of AI for Health* (six principles: autonomy, safety, transparency, responsibility, equity, sustainability). **18 January 2024 Large Multi-Modal Models guidance** (40+ recommendations across diagnosis, patient use, clerical tasks, education, research). **GI-AI4H** (launched July 2023) continues quarterly governance, converging on normative standards for evaluation and federated-data discovery.
- **NIST AI RMF 1.0** (Jan 2023) supplies the Govern-Map-Measure-Manage structure; **NIST AI 600-1 Generative AI Profile** (26 July 2024) adds 12 GenAI-specific risks (hallucination, data leakage, harmful bias, CBRN uplift) that healthcare vendors use to scaffold LLM evaluations.
- **ISO/IEC 42001:2023** — first certifiable AI Management System standard. **ISO 13485** — medical-device QMS backbone. **IEC 62304** — medical device software lifecycle (safety classes A/B/C). **IEC 81001-5-1:2021** — health-software cybersecurity lifecycle. **IEC TR 60601-4-5:2021** — IEC 62443-aligned security capabilities for medical electrical equipment including AI-enabled connected devices.
- **US state law fragmentation**:
  - **California AB 2013** *Generative AI: Training Data Transparency Act* — signed 28 September 2024, effective **1 January 2026** — developer disclosure of training-data provenance.
  - **California AB 3030** *Physician-AI Disclosure* — effective **1 January 2025** — health facilities must mark AI-generated patient communications unless reviewed by a licensed provider.
  - **Utah AI Policy Act (SB 149)** — effective **1 May 2024**, with 2025 amendments covering mental-health chatbots.
  - **Colorado SB 24-205** — high-risk AI including health; effective date extended from 1 Feb 2026 to **30 June 2026**.
  - **Washington My Health My Data Act** — enforceable **31 March 2024** (small businesses 30 June 2024); covers AI-derived health inferences.
- **CMS-4201-F** — published April 2023, applicable **1 January 2024**. Medicare Advantage organizations cannot use AI/algorithmic predictions as the **sole** basis for coverage termination. OCM FAQ (6 February 2024) clarified determinations must reflect each patient's circumstances — history, clinician notes, physician recommendations — not population-level data alone. Response to the **naviHealth / UnitedHealth and Humana class-action litigation**.

---

## Part B — SDM / Socio-Technical Lens

### 5. Stakeholder Topology and Decision-Rights Graph

Healthcare AI has arguably the densest stakeholder mesh in any AI domain. A Crawley-Cameron object-process decomposition makes the pathology visible: the "system" is not the model, it is a graph whose nodes are {patient, clinician, hospital CIO / CMO / CMIO, payer (insurer, PBM, CMS), pharma / device manufacturer, platform vendor (Epic / Oracle Health / cloud), IRB, regulator (FDA / EMA / MHRA / PMDA / NMPA)} and whose edges are decision rights, data flows, and money flows.

A correct RACI for a deployed model is non-trivial because:
- **Accountable** legally collapses onto the physician and institution under medical malpractice.
- **Responsible** for model behavior is split across vendor and platform.
- **Informed** often excludes the patient entirely.

Structural conflicts of interest:
- Vendors are paid for adoption, not outcomes.
- Payers capture savings from tools that providers paid for.
- Governance committees routinely omit community advocates, safety-net providers, and pediatric populations (documented in *Nature Medicine* 2025 and *npj Digital Medicine* 2025-2026 governance-maturity and safety-net papers).

### 6. Feedback Loops Unique to Healthcare AI

Sterman-style loop analysis reveals reinforcing feedbacks that no other AI domain combines.

| Loop | Canonical reference | Mechanism |
|---|---|---|
| **Underdiagnosis reinforcement** | Obermeyer, Powers, Vogeli, Mullainathan — *Science* 2019 | Algorithm used on ~200M Americans used healthcare **cost** as a proxy for **need**. Under-spending on Black patients encoded as under-need. Remediation raised Black-patient high-risk enrollment from **17.7% to 46.5%**. |
| **Disparity amplification in imaging** | Seyyed-Kalantari et al. — *Nature Medicine* 2021 | Chest X-ray classifiers selectively under-flag Black, Hispanic, female, under-20, Medicaid patients as "no finding". Intersectional subgroups worst. |
| **Performative prediction** | Perdomo, Zrnic, Mendler-Dünner, Hardt — ICML 2020; Hardt & Mendler-Dünner arXiv 2310.16608 | A sepsis prediction that triggers an antibiotic bundle erases the outcome the model was trained on; a mortality score that de-prioritizes care confirms itself. |
| **Reimbursement-driven endpoint bias** | Parikh & Helmchen — *npj Digital Medicine* 2022 "Paying for AI" | Model developers optimize for **billable events**, not patient-centered outcomes. The meta-loop. |
| **Goodhart on quality metrics** | HEDIS, CMS Star measures | Once a quality metric is tied to reimbursement, distributions shift (risk-avoiding admissions, sepsis-alert silencing). |

### 7. Case Studies in System Failure

Four (plus one) cases map cleanly to Leveson's "system accident" typology.

- **IBM Watson for Oncology** — STAT News obtained internal IBM slides (25 July 2018) showing "unsafe and incorrect" recommendations. Root cause: training on synthetic MSK hypothetical cases and a narrow specialist expert panel, not guideline or patient data — a **scope / requirements failure**. IBM sold Watson Health data / analytics assets to **Francisco Partners on 21 January 2022** (completed 30 June 2022); the rebranded entity is **Merative**.
- **Epic Sepsis Model v1** — Wong et al. *JAMA Internal Medicine* 2021: AUC **0.63**, sensitivity **33%** at Michigan Medicine vs. Epic's claimed AUC 0.76-0.83. A **validation-transfer failure** across hundreds of US hospitals. Epic released a revised gradient-boosted v2 requiring local re-tuning (evidenced through third-party validation papers; no canonical Epic statement of revision).
- **Babylon Health** — once valued $4.2B post-SPAC (October 2021) → UK administration **11 September 2023**, assets to **eMed for £500k**. MHRA raised device-classification concerns from 2020 onward. **Governance failure** at corporate scale.
- **Apple Watch AFib** — meta-analyses show ~95% sensitivity / specificity, but at 1% population prevalence the positive predictive value collapses to **~16%** (AHA *Circulation* editorial) — a **base-rate / low-prevalence amplification failure**.
- **Change Healthcare ransomware** (12-21 February 2024) — not AI, but Optum's ransomware-induced outage exposed ~100M records and halted one-in-three US claim flows. A **Meadows "leverage point"** demonstration of platform concentration risk.

### 8. STAMP / STPA Applied to Clinical AI

Leveson's STAMP treats the clinical AI workflow as a **hierarchical control structure**: the controller is not the model alone but the composite **{model + clinician + EHR + workflow policy}**, and the controlled process is the patient's trajectory.

Unsafe Control Actions (UCAs) become enumerable:

1. The model issues a recommendation that is **not provided but should be** (missed-alert failure).
2. The model issues a recommendation that is **provided but unsafe** (hallucination / wrong recommendation).
3. **Timing is wrong** — alert fires post-deterioration, no longer actionable.
4. **Duration is wrong** — alert suppressed after first dismissal (classic sepsis-alert fatigue).

Current canonical references (frontier AI rather than clinical-specific):
- **Mylius** — *Systematic Hazard Analysis for Frontier AI using STPA*, arXiv:2506.01782 (June 2025).
- *STAMP/STPA-Informed Characterization of Factors Leading to Loss of Control in AI Systems* — arXiv:2512.17600 (2025).
- *From Silos to Systems: Process-Oriented Hazard Analysis for AI Systems* — arXiv:2410.22526 (2024).
- Qi et al. *STPA for Learning-Enabled Systems: A Survey and a New Practice* — arXiv:2302.10588 (2023).

**Open gap**: a peer-reviewed clinical-AI-specific STPA paper at the same quality level as Mylius 2025 is not cleanly identifiable in 2024-2026 indexing. Practitioners today borrow the frontier/autonomous-systems STPA templates — a thesis-scale gap.

### 9. Platform Engineering Under HIPAA

Maps to the Team Topologies **platform team** archetype.

- **De-identification** bifurcates into Safe Harbor (18-identifier strip) vs. Expert Determination (statistical risk opinion). LLM membership-inference work makes residual risk operationally material for fine-tuning corpora.
- **Federated learning** dominates the multi-hospital pattern: **NVIDIA FLARE** (production scale), **Owkin Substra** (now Linux Foundation AI/Data), **Flower**. arXiv:2511.00037 (Nov 2025) is the current benchmark.
- **Sovereign-hosting** is consolidated around **AWS HealthLake**, **Google Cloud Healthcare API**, and **Azure Health Data Services**. HIPAA-compliant generative-AI hosting requires a **BAA with AWS Bedrock, Azure OpenAI Service, or Google Vertex AI** — necessary but not sufficient because prompt logs, system prompts, and fine-tuning artifacts extend the PHI boundary.
- **Audit-trail engineering** is becoming a first-class requirement under both FDA 510(k) TPLC guidance and **EU AI Act Art. 12** logging. Litigation discovery extends retention horizons beyond classical observability.

### 10. Lifecycle-Specific Artifacts

- **FDA PCCP** (finalized 3 December 2024) is **not an MLOps retraining policy** — it is a pre-authorized envelope of future modifications. Retraining becomes a first-class regulatory artifact — a fundamental divergence from classical MLOps where retraining is an ops concern.
- **ONC HTI-1 Final Rule** (effective 31 December 2024) mandates **31 "source attributes"** for Predictive Decision Support Interventions in certified EHRs — a de-facto model-card requirement. The HTI-5 2025 proposed rollback under the incoming administration creates regulatory whiplash worth tracking.
- **Post-market surveillance gap**: *JAMA Health Forum* 2025 found only **~5% of FDA-cleared AI devices have any MAUDE adverse-event report** — systemic under-reporting.
- **NYC Local Law 144** (AEDT bias audits) — employment-focused, but its annual-independent-audit pattern is the template healthcare auditors are borrowing.

### 11. Team Topologies for Healthcare-AI Orgs

A healthcare-AI organization resolved through Skelton-Pais decomposes cleanly into four team types:

| Team type | Role in healthcare AI |
|---|---|
| **Stream-aligned** | Clinical-product squads owning a single care pathway — radiology AI, ambient-scribe (Abridge / Dragon Copilot), sepsis monitoring, EHR-integrated CDS. Each a flow of change tied to one specialty workflow. |
| **Platform** | HIPAA-compliant ML platform team — de-identification pipelines, BAA-governed model gateways, FHIR ingestion, sovereign-cloud tenancy, audit-trail store, feature store. Makes compliance the path of least resistance. |
| **Enabling** | Clinical-informatics and AI-safety team — model-card review, clinical-evidence / REMARK / TRIPOD+AI review, health-equity review, physician-in-the-loop training. Exists to raise the other teams' capability and then disband per topology. |
| **Complicated-subsystem** | Regulatory affairs and validation engineering — FDA / CE / PMDA liaison, PCCP authorship, 510(k) / De Novo submissions, MDR Annex IX technical-file authorship. Leveson safety-case owners; Sculley hidden-debt firewalls. |

**Interaction modes**: X-as-a-Service for the platform; Facilitating for the enabling team; Collaboration boundary around the regulatory complicated-subsystem — where Team Topologies integrates directly with Leveson's control structure.

---

## 12. Synthesis — Five Takeaways

1. **Healthcare AI is a five-frame regulatory stack** — HIPAA, FDA, EU MDR/IVDR + AI Act, WHO/NIST/ISO, and state / CMS — stacked not additive. A single product must clear all five.
2. **PCCP reframes MLOps as governance** — retraining is no longer an ops concern but a pre-authorized modification envelope. Healthcare teams need regulatory-affairs in the MLOps loop.
3. **Feedback loops in health are safety events, not optimization targets** — Obermeyer 2019, Seyyed-Kalantari 2021, and performative prediction turn fairness from a metric into a system-dynamics property.
4. **The EHR and the hyperscaler absorb most of the surface** — Epic AI Charting (Feb 2026), Microsoft Dragon Copilot, Oracle Health Clinical AI Agent, Salesforce Agentforce for Health, and hyperscaler BAAs (Bedrock, Vertex, Azure OpenAI) are the long-term beneficiaries of consolidation. Venture-backed AI specialists live or die on whether they integrate faster than the platform extends.
5. **STAMP / STPA for clinical AI is a thesis-scale gap** — the safety-engineering framework exists (Leveson; Mylius 2025 for frontier AI), but a peer-reviewed clinical-AI-specific STPA methodology at the same quality level is not cleanly identifiable in 2024-2026. An MIT SDM student has room to occupy.

---

## 13. References

### HIPAA and HHS OCR
- *HIPAA Privacy Rule to Support Reproductive Health Care Privacy* — Federal Register (26 Apr 2024) — [federalregister.gov](https://www.federalregister.gov/documents/2024/04/26/2024-08503/hipaa-privacy-rule-to-support-reproductive-health-care-privacy)
- HHS fact sheet — [hhs.gov](https://www.hhs.gov/hipaa/for-professionals/special-topics/reproductive-health/final-rule-fact-sheet/index.html)
- HIPAA Security Rule NPRM — Federal Register (6 Jan 2025) — [federalregister.gov](https://www.federalregister.gov/documents/2025/01/06/2024-30983/hipaa-security-rule-to-strengthen-the-cybersecurity-of-electronic-protected-health-information)
- HHS OCR NPRM fact sheet — [hhs.gov](https://www.hhs.gov/hipaa/for-professionals/security/hipaa-security-rule-nprm/factsheet/index.html)
- *Nondiscrimination in Health Programs and Activities* (Section 1557 final rule, 6 May 2024) — [federalregister.gov](https://www.federalregister.gov/documents/2024/05/06/2024-08711/nondiscrimination-in-health-programs-and-activities)
- HHS OCR de-identification guidance — [hhs.gov](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)

### FDA
- FDA. *PCCP Final Guidance Dec 2024* — [fda.gov](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence)
- Federal Register 2024-28361 — [federalregister.gov](https://www.federalregister.gov/documents/2024/12/04/2024-28361/marketing-submission-recommendations-for-a-predetermined-change-control-plan-for-artificial)
- FDA. *AI-Enabled Device Software Functions Draft Guidance Jan 2025* — [fda.gov](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)
- Federal Register notice for Jan 2025 draft — [federalregister.gov](https://www.federalregister.gov/documents/2025/01/07/2024-31543/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)
- FDA. *AI/ML-Enabled Medical Devices list* — [fda.gov](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices)
- FDA / Health Canada / MHRA GMLP guiding principles (2021) — [fda.gov](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
- FDA Transparency Principles for ML-Enabled Medical Devices — [fda.gov](https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles)
- FDA Digital Health Advisory Committee — [fda.gov](https://www.fda.gov/medical-devices/digital-health-center-excellence/fda-digital-health-advisory-committee)
- DHAC Nov 2024 meeting — [fda.gov](https://www.fda.gov/advisory-committees/advisory-committee-calendar/november-20-21-2024-digital-health-advisory-committee-meeting-announcement-11202024)
- DHAC Nov 2025 GenAI mental-health meeting — [fda.gov](https://www.fda.gov/advisory-committees/advisory-committee-calendar/november-6-2025-digital-health-advisory-committee-meeting-announcement-11062025)
- FDA CDS FAQs §3060 — [fda.gov](https://www.fda.gov/medical-devices/software-medical-device-samd/clinical-decision-support-software-frequently-asked-questions-faqs)

### EU
- EU AI Act Article 6 — [artificialintelligenceact.eu](https://artificialintelligenceact.eu/article/6/)
- Regulation (EU) 2017/745 MDR — [eur-lex.europa.eu](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02017R0745-20240709)
- Regulation (EU) 2017/746 IVDR — [eur-lex.europa.eu](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02017R0746-20240709)
- MDCG 2019-11 Rev.1 (June 2025) — [health.ec.europa.eu](https://health.ec.europa.eu/latest-updates/update-mdcg-2019-11-rev1-qualification-and-classification-software-regulation-eu-2017745-and-2025-06-17_en)
- EMA Reflection Paper (9 Sep 2024) — [ema.europa.eu](https://www.ema.europa.eu/en/documents/scientific-guideline/reflection-paper-use-artificial-intelligence-ai-medicinal-product-lifecycle_en.pdf)
- EMA AI scientific-guideline page — [ema.europa.eu](https://www.ema.europa.eu/en/use-artificial-intelligence-ai-medicinal-product-lifecycle-scientific-guideline)
- Reed Smith analysis of AI Act + medical devices — [reedsmith.com](https://www.reedsmith.com/en/perspectives/2024/10/the-eu-ai-act-and-medical-devices-navigating-high-risk-compliance)

### WHO
- WHO 2021 *Ethics and Governance of AI for Health* — [who.int](https://www.who.int/publications/i/item/9789240029200)
- WHO Jan 2024 *Large Multi-Modal Models* — [who.int](https://www.who.int/publications/i/item/9789240084759)
- WHO news release (18 Jan 2024) — [who.int](https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models)
- WHO Global Initiative on AI for Health (GI-AI4H) — [who.int](https://www.who.int/initiatives/global-initiative-on-ai-for-health)

### NIST and ISO/IEC
- NIST AI RMF 1.0 — [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
- NIST AI 600-1 GenAI Profile — [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- ISO/IEC 42001:2023 — [iso.org](https://www.iso.org/standard/42001)
- IEC 62304 — [iso.org](https://www.iso.org/standard/38421.html)
- IEC TR 60601-4-5:2021 — [webstore.iec.ch](https://webstore.iec.ch/en/publication/64703)

### State / CMS
- California AB 2013 — [leginfo.legislature.ca.gov](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240AB2013)
- California AB 3030 — [leginfo.legislature.ca.gov](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB3030)
- Utah SB 149 — [le.utah.gov](https://le.utah.gov/~2024/bills/static/SB0149.html)
- Colorado SB 24-205 — [leg.colorado.gov](https://leg.colorado.gov/bills/sb24-205)
- Washington My Health My Data Act (RCW 19.373) — [app.leg.wa.gov](https://app.leg.wa.gov/RCW/default.aspx?cite=19.373&full=true)
- CMS-4201-F — [cms.gov](https://www.cms.gov/newsroom/fact-sheets/2024-medicare-advantage-and-part-d-final-rule-cms-4201-f)
- NYC Local Law 144 AEDT — [nyc.gov](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page)

### Bias, feedback loops, case studies
- Obermeyer, Powers, Vogeli, Mullainathan. *Dissecting racial bias in an algorithm used to manage the health of populations* — *Science* 366, 447-453 (2019) — [science.org](https://www.science.org/doi/10.1126/science.aax2342)
- Seyyed-Kalantari et al. *Underdiagnosis bias of AI applied to chest radiographs* — *Nature Medicine* 27 (2021) — [nature.com](https://www.nature.com/articles/s41591-021-01595-0)
- Perdomo, Zrnic, Mendler-Dünner, Hardt. *Performative Prediction* — ICML 2020 — [proceedings.mlr.press](https://proceedings.mlr.press/v119/perdomo20a.html)
- Hardt & Mendler-Dünner. *Performative Prediction: Past and Future* — [arxiv.org](https://arxiv.org/abs/2310.16608)
- Ross & Swetlitz. *IBM Watson recommended unsafe and incorrect treatments* — *STAT News* (25 July 2018) — [statnews.com](https://www.statnews.com/2018/07/25/ibm-watson-recommended-unsafe-incorrect-treatments/)
- IBM. *Francisco Partners to acquire Watson Health assets* — [newsroom.ibm.com](https://newsroom.ibm.com/2022-01-21-Francisco-Partners-to-Acquire-IBMs-Healthcare-Data-and-Analytics-Assets)
- Ross. *Merative rebranding of IBM Watson Health assets* — *STAT News* (6 July 2022) — [statnews.com](https://www.statnews.com/2022/07/06/ibm-watson-health-merative-data/)
- Wong et al. *External Validation of a Widely Implemented Proprietary Sepsis Prediction Model* — *JAMA Internal Medicine* 181(8) (2021) — [jamanetwork.com](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307)
- Habib, Lin, Grant. *The Epic Sepsis Model Falls Short* editorial — *JAMA Internal Medicine* (2021) — [jamanetwork.com](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781313)
- Wikipedia. *2024 Change Healthcare ransomware attack* — [wikipedia.org](https://en.wikipedia.org/wiki/2024_Change_Healthcare_ransomware_attack)
- Lomas. *MHRA concerns about Babylon Health* — *TechCrunch* (5 March 2021) — [techcrunch.com](https://techcrunch.com/2021/03/05/uks-mhra-says-it-has-concerns-about-babylon-health-and-flags-legal-gap-around-triage-chatbots/)
- Sifted. *The rise and fall of Babylon* — [sifted.eu](https://sifted.eu/articles/the-rise-and-fall-of-babylon)
- Perino et al. *Apple Watch AFib meta-analysis* — [pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC11780081/)
- *Smartwatch-Detected AFib — the value in positive predictive value* — *Circulation* editorial — [ahajournals.org](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.122.062292)

### STAMP / STPA
- Mylius. *Systematic Hazard Analysis for Frontier AI using STPA* — arXiv:2506.01782 (June 2025) — [arxiv.org](https://arxiv.org/abs/2506.01782)
- *STAMP/STPA-Informed Characterization of Factors Leading to Loss of Control in AI Systems* — arXiv:2512.17600 (2025) — [arxiv.org](https://arxiv.org/abs/2512.17600)
- *From Silos to Systems: Process-Oriented Hazard Analysis for AI Systems* — arXiv:2410.22526 (2024) — [arxiv.org](https://arxiv.org/html/2410.22526)
- Qi et al. *STPA for Learning-Enabled Systems: A Survey and A New Practice* — arXiv:2302.10588 (2023) — [arxiv.org](https://arxiv.org/pdf/2302.10588)

### Platform, technical debt, governance
- Sculley et al. *Hidden Technical Debt in Machine Learning Systems* — NeurIPS 2015 — [papers.nips.cc](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
- *Benchmarking Federated Learning Frameworks for Medical Imaging* — arXiv:2511.00037 (Nov 2025) — [arxiv.org](https://arxiv.org/abs/2511.00037)
- *JAMA Health Forum — Benefit-Risk Reporting for FDA-Cleared AI-Enabled Medical Devices* (2025) — [jamanetwork.com](https://jamanetwork.com/journals/jama-health-forum/fullarticle/2839236)
- *A general framework for governing marketed AI/ML medical devices* — *npj Digital Medicine* (2025) — [nature.com](https://www.nature.com/articles/s41746-025-01717-9)
- *The urgency of centering safety-net organizations in AI governance* — *npj Digital Medicine* (2025) — [nature.com](https://www.nature.com/articles/s41746-025-01479-4)
- *International partnership for governing generative AI models in medicine* — *Nature Medicine* (2025) — [nature.com](https://www.nature.com/articles/s41591-025-03787-4)
- Parikh & Helmchen. *Paying for artificial intelligence in medicine* — *npj Digital Medicine* (2022) — [nature.com](https://www.nature.com/articles/s41746-022-00609-6)
- *A reimbursement framework for artificial intelligence in healthcare* — *npj Digital Medicine* (2022) — [nature.com](https://www.nature.com/articles/s41746-022-00621-w)
- *A future role for health applications of LLMs depends on regulators enforcing safety standards* — *Lancet Digital Health* (2024) — [thelancet.com](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(24)00124-9/fulltext)
- ONC HTI-1 Final Rule — [healthit.gov](https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program)

### Flagged / partially sourceable
- Peer-reviewed clinical-AI-specific STPA paper at Mylius 2025 quality — not cleanly identifiable; practitioners borrow frontier-AI STPA templates (flagged as thesis-scale gap).
- "Epic's revised approach to sepsis model" — v2 is evidenced through third-party validation papers; no canonical Epic statement of revision verified.
- HHS "AI Accountability Act" — not a cleanly identifiable enacted US statute as of April 2026; the landscape is Congressional discussion + HTI-1/HTI-5 rulemaking.
- Change Healthcare data-breach totals — "~100M individuals" is consistent across 2024-2025 reporting but revised multiple times by UnitedHealth / OCR; treat as point-in-time.
- HIPAA reproductive-health rule enforceability — most vacated by N.D. Tex. order 18 Jun 2025; appellate trajectory unsettled in April 2026.
- Colorado SB 205 effective date — extended to 30 Jun 2026; further action plausible during the 2026 session.
- "FDA 950+ AI/ML devices by late 2024" — rolling figure; FDA does not publish a "950" checkpoint. Authoritative end-of-2024 count is 1,016 per a PMC taxonomy paper (PMC12219150).
