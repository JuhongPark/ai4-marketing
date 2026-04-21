# Drug Discovery and Clinical AI

**Scope**: Biological foundation models, AI-first biotech, clinical-trial AI, real-world evidence AI (Part A); medical imaging SaMD, ambient clinical scribes, medical LLMs, symptom triage, digital therapeutics (Part B). Paired with [`healthcare_commercial_ai.md`](healthcare_commercial_ai.md) (commercial layer) and [`healthcare_ai_regulation_and_sdm.md`](healthcare_ai_regulation_and_sdm.md) (governance layer). See [`healthcare_bio_ai.md`](healthcare_bio_ai.md) for the overview.

**Why these together**: the bio-AI discovery pipeline is the **upstream** of the commercial pharma stack, and the clinical-AI product landscape is the **downstream** patient-and-provider surface. Both are scientific / regulatory territory that precedes or undergirds the marketing motions covered in `healthcare_commercial_ai.md`.

---

## Part A — Drug Discovery and Biological Foundation Models

### 1. Biological Foundation Models

A dense wave of 2024-2026 releases turned structural biology into a foundation-model problem.

| Model | Publication | Contribution |
|---|---|---|
| **AlphaFold 3** | Abramson et al., *Nature*, May 2024 | Unified protein + DNA + RNA + ion + ligand prediction via diffusion. ~76% of predicted binding poses within 2 Å RMSD on PoseBusters (vs. 38% for DiffDock); lDDT ≈ 0.79 on protein-nucleic-acid complexes. |
| **ESM-3** | EvolutionaryScale, June 2024 | 98B-parameter multimodal protein LM over 2.78B proteins. Generated *esmGFP*, a fluorescent protein only 58% identical to any natural GFP. |
| **ESMFold** | Meta FAIR, *Science* 2023 | ~60× faster than AlphaFold 2 with high pLDDT on well-folded singletons; underpins the 617M-structure **ESM Metagenomic Atlas**. |
| **RoseTTAFold All-Atom** | Krishna et al., *Science* March 2024 | Extends RFdiffusion with unified atomic representations for covalent modifications and metals. |
| **AlphaMissense** | Cheng et al., *Science* 2023 | Classifies 89% of possible human missense variants as likely pathogenic / benign at 90% ClinVar-calibrated precision. |
| **Evo** | Arc Institute / Stanford / Together, Feb 2024 | 7B StripedHyena on prokaryotic DNA, 131k-token context. |
| **Evo-2** | Arc Institute, Feb 2025 preprint (*Nature* forthcoming) | 40B parameters, 1 Mb context, 9.3T nucleotides across the tree of life. Predicts BRCA1 pathogenicity zero-shot. |
| **Boltz-1** | Wohlwend, Corso et al., **MIT Jameel Clinic**, Nov 2024 | First fully open **MIT-licensed** model matching AlphaFold 3-level accuracy on biomolecular complexes. |
| **Boltz-2** | MIT + Recursion, June 2025 | Adds binding-affinity prediction within ~20s on a single GPU; outperformed CASP16 affinity-challenge winners. |
| **OpenFold3** | Consortium, preview Oct 2025 | Pursues a fully open AF3-equivalent trained on ~13M synthetic structures. |

**MIT significance**: Boltz-1 / Boltz-2 / BoltzGen (Oct 2025) sequence is led by **Regina Barzilay** (2025 IEEE Frances E. Allen Medal; TIME100 AI 2025) and **Tommi Jaakkola** at the **MIT Jameel Clinic** (co-founded 2018 by MIT and Community Jameel). These are the MIT anchors for drug-discovery AI.

### 2. AI-First Biotech Platforms

The defining 2025 event: **Insilico Medicine's Phase 2a readout of rentosertib** (INS018_055), a generative-AI-discovered TNIK inhibitor for idiopathic pulmonary fibrosis. The 60 mg QD arm showed **+98.4 mL mean FVC change vs. −20.3 mL on placebo** in the 71-patient GENESIS-IPF trial (NCT05938920), published in *Nature Medicine* June 2025 — **the first clinical proof-of-concept for end-to-end AI-designed drug development**.

| Platform | Status |
|---|---|
| **Recursion Pharmaceuticals** | Closed Exscientia merger November 2024 (~$850M pro-forma cash, ~$100M annual synergies). Q1 2025: deprioritized 3 clinical programs to focus on a CDK7 inhibitor and REC-4881 rare-disease readouts in H2 2025. |
| **Xaira Therapeutics** | Launched April 2024 with **>$1B** from ARCH / Foresite. Built on Baker-lab RFdiffusion / RFantibody; Marc Tessier-Lavigne as CEO. |
| **Isomorphic Labs** | Novartis deal (~$37.5M upfront, up to ~$1.2B milestones) + Lilly deal (~$45M upfront, up to ~$1.7B milestones), both Jan 2024. **$600M raise March 2025** (Thrive Capital lead). Hassabis targets first human trials by end of 2026. |
| **Absci** | *ABS-201* (androgenetic alopecia) advanced to Phase 1/2a HEADLINE. Origin-1 for de-novo full-length mAb design against "zero-prior" epitopes. Almirall deal up to ~$650M. |
| **Generate Biomedicines** | Plans two Phase 3 trials (SOLAIRIA-1/-2) of *GB-0895*, a long-acting AI-engineered anti-TSLP mAb for severe asthma. |
| **Inceptive** | ~$100M (NVentures / a16z) for AI-designed mRNA. |
| **BenevolentAI** | Cut ~30% of staff and announced Euronext delisting plans across 2024-2025 — the most-visible cautionary case. |
| **Valence Labs** (Recursion) | Released MolGPS (3B-parameter GNN) and TxPert for perturbation response. |
| **Atomic AI** | ~$42M total funding (ATOM-1 RNA model). |

### 3. Clinical-Trial AI

Patient-recruitment NLP consolidated in 2025: **Tempus AI acquired Deep 6 AI (March 2025)**, adding 30M+ patient records and 750+ provider sites to its trial-matching footprint. **Mendel.ai** reports 24-50% recruitment lifts in published oncology trials. **Trialjectory** (rebranded Leal Health) runs predictive oncology journey matching.

On synthetic controls, **Unlearn.AI's PROCOVA** digital-twin method received EMA qualification for primary analysis in Phase 2/3 continuous-outcome trials (not yet formally FDA-qualified). **Medidata's Synthetic Control Arm** remains the longest-running regulatory-grade external control (Medicenna Phase 3 glioblastoma precedent).

**Medable** rolled out generative-AI eCOA generation and oncology / cell-gene therapy modules; **Science 37** continues its FDA-inspected direct-to-patient Metasite model.

### 4. Real-World Evidence (RWE) AI

- **Flatiron Health** (Roche) — canonical oncology EHR RWE network, 5M+ patient journeys across 22+ tumor types, LLMs integrated for EHR abstraction.
- **Tempus AI** — IPO'd on Nasdaq (TEM) **14 June 2024 at $37/share**, raising ~$410.7M at ~$6B market cap; the March 2025 Deep 6 AI acquisition tightens the trial-matching-to-RWE loop.
- **Komodo Health MapLab** — drove 31 RWE studies at ISPOR 2025 against 330M+ patient journeys linking claims + labs + genomics.
- **Truveta Live Link** — prospective EHR-linked RWE.
- **ConcertAI** — oncology RWE platform for pharma strategy and payer evidence.

### 5. MIT-Specific Research Connections

The **MIT Abdul Latif Jameel Clinic for Machine Learning in Health** (co-founded 2018 by MIT and Community Jameel) is the nerve center.

| Faculty | Lab / Group | Focus |
|---|---|---|
| **Regina Barzilay** | Jameel Clinic AI Faculty Lead; CSAIL | Drug discovery, clinical NLP, breast cancer screening. 2025 IEEE Frances E. Allen Medal; TIME100 AI 2025. Led Boltz-1 / Boltz-2 / BoltzGen. |
| **Tommi Jaakkola** | CSAIL | Diffusion / flow-based generative models for molecules; molecular-dynamics emulation. Co-authored Boltz-1 / Boltz-2. |
| **Jim Collins** | IMES, Broad, Wyss | Antibiotics AI Project — halicin lineage. August 2025: generative designs *NG1* and *DN1* active against drug-resistant *N. gonorrhoeae* and MRSA. |
| **Manolis Kellis** | CSAIL / Broad | Computational biology, genomics, regulatory-genomics foundation models. |
| **Caroline Uhler** | IDSS / Broad / Schmidt Center | Causal inference × multimodal biology. *Image2Reg* in *Cell Systems* 2025. |

Shared compute and data with the broader **Broad Institute** ecosystem.

---

## Part B — Clinical AI and Medical LLMs

### 6. Medical Imaging AI (FDA-Cleared SaMD)

The FDA public list of AI/ML-enabled medical devices crossed **1,016 by mid-2024**, reached **~1,250 by July 2025**, and **1,451 by the December 2025 refresh** — with 295 devices cleared in 2025 alone, the largest annual total. Radiology dominates (~76% of authorizations).

| Vendor | 2024-2026 milestones |
|---|---|
| **Aidoc** | ~31 FDA clearances cumulative. October 2025: Breakthrough Device Designation. January 2026: comprehensive CT triage solution consolidating 11 new + 3 prior indications. |
| **Viz.ai** | Viz LVO, **Viz ICH Plus cleared 8 Feb 2024**, Viz ANEURYSM, Viz AAA (first FDA-cleared AAA triage algorithm, March 2023). |
| **Paige** | Paige Prostate — first FDA-authorized AI product in digital pathology (De Novo DEN200080, 2021). 527-slide, 16-pathologist validation: sensitivity 89.5% → 96.8%, 70% reduction in false negatives. |
| **PathAI** | **AISight Dx cleared June 2025 for primary diagnosis**. AIM-MASH (December 2025) became first AI-powered pathology tool qualified by FDA for MASH clinical trials. |
| **HeartFlow** | FFRCT moved to **Category I CPT code 75580** on 1 Jan 2024; CMS payment $997 → $1,017 in 2025. New Category I CPT for AI-enabled plaque analysis effective Jan 2026 — the first durable reimbursement rail for cardiology AI. |
| **Nanox.AI** | HealthCCSng V2.0 510(k) cleared August 2024 for opportunistic coronary artery calcium scoring. |
| **Annalise.ai** | FDA triage clearance for limited chest findings (vs. 124 findings on CE-marked EU product). *Lancet Digital Health* multi-reader study across 102 chest findings. |
| **Rad AI** | Dominant radiology-report-generation niche; published outcome studies sparse. |

### 7. EHR NLP and Ambient Clinical Scribes

The fastest-growing segment of clinical AI.

| Product | Status |
|---|---|
| **Microsoft Dragon Copilot** | Merger of DAX Copilot + Dragon Medical One, March 2025; **GA May 2025** in US and Canada, European expansion late 2025 / early 2026. **Stanford prospective QI study** (48 physicians, Oct 2023-Jan 2024, *JAMIA* 2024): −24.42 points task load, −1.94 burnout (p<0.001). |
| **Abridge** | **$250M Series D Feb 2025 at $2.75B**; **$300M Series E June 2025 at $5.3B** (a16z + Khosla). ~$100M ARR by May 2025; deep Epic integration; 150+ health systems, 55 specialties, 28 languages. Named customers include Duke, Johns Hopkins, Mayo Clinic, UNC Health, Memorial Sloan Kettering, UI Health. |
| **Suki AI** | $70M Oct 2024; 400+ systems. |
| **Commure** | Acquired **Augmedix for $139M (2024)**. |
| **DeepScribe** | AWS HealthScribe partnership; **led 2025 KLAS Spotlight Report for Ambient AI**. |
| **Epic AI Charting** | Native launch **February 2026** (part of Art / Penny / Emmie). Epic reports 85% of its customers are live with generative AI across these copilots — a direct threat to venture-backed scribes. |

### 8. Medical Large Language Models

Benchmark leapfrogging defined the 2023-2026 race.

| Model / benchmark | Year | Result |
|---|---|---|
| **Med-PaLM** (Singhal et al., *Nature*) | 2023 | First LLM above ~60% USMLE pass: **67.2% on MedQA**. |
| **Med-PaLM 2** (*Nature Medicine*, Jan 2025 peer-review) | 2024-2025 | **86.5% on MedQA**. Preferred over physicians on 8 of 9 clinical axes. |
| **Med-Gemini** (Google, April 2024) | 2024 | **91.1% on MedQA**. SOTA on 10 of 14 medical benchmarks. |
| **AMIE** (Tu et al., *Nature*, April 2025) | 2025 | Remote OSCE with 159 cases, 20 PCPs. Outperformed primary-care clinicians on 30 of 32 specialist-rated and 25 of 26 patient-rated axes. |
| **John Snow Labs Medical LLM** | 2024 | **87.35 on the Open Medical LLM leaderboard** (May 2024), outperforming GPT-4 and Med-PaLM 2 on MedQA, PubMedQA, clinical MMLU. |
| **OpenAI HealthBench** (arXiv:2505.08775, May 2025) | 2025 | 262 physicians across 60 countries; 5,000 conversations, 48,562 rubric items. OpenAI **o3 scored 60% overall, 32% on HealthBench Hard**. For the first time physicians no longer meaningfully improved model responses. |

On the commercial side:

- **OpenEvidence** — **$210M Series B July 2025 ($3.5B)**; **$200M Series C October 2025 ($6B)**; **$250M Series D January 2026 ($12B)**. ~$700M cumulative.
- **Hippocratic AI** — $500M Series A valuation March 2024 → **$1.64B Series B (**$141M**) January 2025** → **$3.5B Series C (**$126M**)**. Claims 115M+ patient interactions. Explicitly refuses diagnostic use cases — constraining agents to non-diagnostic care-coordination (appointment prep, post-discharge calls, chronic check-ins) is a regulatory strategy as much as a clinical one, navigating around the SaMD line.
- **Glass Health** — clinical reasoning and scribing.
- **Microsoft BiomedCLIP** — widely used multimodal biomedical foundation model.

### 9. Symptom Triage and Patient-Facing Chat

The highest-risk frontier. Canonical benchmarks:

- **Gilbert et al.** *BMJ Open* 2020: only Ada, Babylon, Symptomate within one SD of GPs on safety. Ada top-3 accuracy **70.5%** vs. GPs' **82.1%**.
- **JMIR AI 2024** clinical-vignette study documented further gaps between mass-market symptom checkers and physicians.

**Babylon Health** is the cautionary case: $4.2B post-SPAC peak valuation (October 2021) → announced US exit early August 2023 → **Chapter 7 filed August 15, 2023** → **UK administration September 11, 2023**, assets sold to **eMed for £500,000**. A *Lancet* triage-accuracy study had been publicly disputed by biomedical researchers.

### 10. Digital Therapeutics (DTx)

Clear bifurcation between regulatory wins and commercial collapse.

| Product | Status |
|---|---|
| **Akili EndeavorRx** (AKL-T01) | De Novo DEN200026, cleared 15 June 2020 — first video-game-based prescription therapeutic. Based on *Lancet Digital Health* RCT in 600+ children. June 2024: **EndeavorOTC authorized over-the-counter for adults with ADHD**. |
| **Mahana Parallel (Mahana IBS)** | FDA De Novo 25 Nov 2020 (K211372) — first prescription DTx for IBS. 558-patient RCT in *Gut* (66% symptom reduction at 1 year). |
| **Big Health** (Sleepio, Daylight) | 80+ peer-reviewed papers. **1 Jan 2025 CMS Physician Fee Schedule** added three new **Digital Mental Health Treatment CPT codes** — establishing the first Medicare reimbursement rail for DMHTs. |
| **Germany DiGA** | 374,000+ prescriptions since the 2019 Digital Healthcare Act. 2026 scope widening to include certain Class IIb devices. |
| **Pear Therapeutics** (reSET, reSET-O, Somryst) | **Chapter 11 filed 7 April 2023**. Assets sold at auction May 2023 for **$6.05M** against $32M in debt. **PursueCare acquired reSET / reSET-O** late 2023 and relaunched them in 2024. The category's cautionary tale — payer-coverage bottleneck nearly killed DTx before CMS DMHT codes revived it. |

---

## 11. Open Problems

1. **From AlphaFold 3 / ESM-3 to approved drugs** — Insilico's rentosertib is the first AI-end-to-end Phase 2a positive, but the causal chain from structure prediction to clinical endpoint is still empirical; there is no foundation-model guarantee that a predicted structure translates into drug efficacy.
2. **Open vs. closed biological models** — ESM-3 and AlphaFold 3 are commercial or partially open, while Boltz-1/2 are MIT-licensed. The open-vs-closed tension of general-purpose foundation models is replaying in biology, with direct economic consequences for downstream biotech.
3. **Synthetic control arms and regulatory acceptance** — EMA has qualified Unlearn's PROCOVA; FDA has not issued an equivalent qualification. Harmonization will shape the next generation of trial designs.
4. **Ambient-scribe commodification** — Epic's native AI Charting (Feb 2026) and Microsoft Dragon Copilot consolidation threaten venture-backed scribes. The long-term question is whether any independent scribe survives absorption into the EHR or hyperscaler.
5. **HealthBench and the physician-ceiling effect** — for the first time, physicians do not meaningfully improve model responses on OpenAI's HealthBench. If this generalizes, the clinical-decision-support governance model (physician-in-the-loop review as safety net) has to be rethought.
6. **Patient-facing chat regulatory line** — when does a symptom checker cross into SaMD? FDA guidance remains in draft; Babylon's collapse was as much a governance failure as a product failure.
7. **DTx reimbursement durability** — CMS DMHT codes (Jan 2025) revived the category, but private-payer adoption is uneven and commercial scale remains unproven.

---

## 12. References

### Biological foundation models
- Abramson et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3* — *Nature* 2024 — [nature.com](https://www.nature.com/articles/s41586-024-07487-w)
- Isomorphic Labs. *AlphaFold 3 overview* — [isomorphiclabs.com](https://www.isomorphiclabs.com/articles/alphafold-3-predicts-the-structure-and-interactions-of-all-of-lifes-molecules)
- EvolutionaryScale. *ESM-3 release* — [evolutionaryscale.ai](https://www.evolutionaryscale.ai/blog/esm3-release)
- EvolutionaryScale. *Simulating 500 million years of evolution with a language model* (ESM-3) — [biorxiv.org](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v2.full)
- Meta FAIR. *ESMFold / ESM Metagenomic Atlas* — [github.com](https://github.com/facebookresearch/esm)
- Arc Institute. *Evo DNA foundation model* — [arcinstitute.org](https://arcinstitute.org/news/evo)
- Arc Institute. *Evo-2 preprint* — [biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)
- Arc Institute. *Evo 2 tool page* — [arcinstitute.org](https://arcinstitute.org/tools/evo)
- Krishna et al. *RoseTTAFold All-Atom* — *Science* 2024 — [science.org](https://www.science.org/doi/10.1126/science.adl2528)
- Cheng et al. *AlphaMissense* — *Science* 2023 — [science.org](https://www.science.org/doi/10.1126/science.adg7492)
- Wohlwend, Corso et al. *Boltz-1 preprint* — [biorxiv.org](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v1)
- MIT News. *Boltz-1 open-source model* — [news.mit.edu](https://news.mit.edu/2024/researchers-introduce-boltz-1-open-source-model-predicting-biomolecular-structures-1217)
- *Boltz-2 preprint* — [biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
- Recursion IR. *Boltz-2 release with MIT* — [ir.recursion.com](https://ir.recursion.com/news-releases/news-release-details/mit-and-recursion-release-boltz-2-next-generation-ai-model)
- OpenFold Consortium. *OpenFold3 preview* — [businesswire.com](https://www.businesswire.com/news/home/20251028507233/en/OpenFold-Consortium-Releases-Preview-of-OpenFold3)

### AI-first biotech
- Insilico Medicine. *Rentosertib Phase 2a Nature Medicine* — [insilico.com](https://insilico.com/news/tnrecuxsc1-insilico-announces-nature-medicine-publi)
- PubMed. *Generative AI TNIK inhibitor for IPF Phase 2a* — [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/40461817/)
- Recursion IR. *Exscientia merger completion* — [ir.recursion.com](https://ir.recursion.com/news-releases/news-release-details/recursion-and-exscientia-two-leaders-ai-drug-discovery-space)
- Xaira Therapeutics launch — [businesswire.com](https://www.businesswire.com/news/home/20240423707240/en/Xaira-Therapeutics-Launches)
- Isomorphic Labs. *Novartis / Lilly deals 2024* — [isomorphiclabs.com](https://www.isomorphiclabs.com/articles/isomorphic-labs-kicks-off-2024-with-two-pharmaceutical-collaborations)
- Fortune. *Isomorphic Labs first human trials* — [fortune.com](https://fortune.com/2025/07/06/deepmind-isomorphic-labs-cure-all-diseases-ai-now-first-human-trials/)
- Absci. *Q3 2025 financials* — [investors.absci.com](https://investors.absci.com/news-releases/news-release-details/absci-reports-business-updates-and-third-quarter-2025-financial/)
- Generate Biomedicines. *SOLAIRIA Phase 3 GB-0895* — [generatebiomedicines.com](https://generatebiomedicines.com/media-center/generatebiomedicines-to-initiate-global-phase-3-studies-of-gb-0895-a-long-acting-anti-tslp-antibody-for-severe-asthma-engineered-with-ai)
- Nasdaq. *Inceptive $100M Series A* — [nasdaq.com](https://www.nasdaq.com/articles/biotech-startup-inceptive-secures-$100-mln-funding)
- BusinessCloud. *BenevolentAI restructuring* — [businesscloud.co.uk](https://businesscloud.co.uk/news/more-layoffs-as-benevolentai-plans-to-delist-return-to-original-mission/)
- Valence Labs. *MolGPS* — [valencelabs.com](https://www.valencelabs.com/introducing-molgps-a-foundational-gnn-for-molecular-property-prediction/)

### Clinical trials and RWE
- Tempus AI IPO — [tempus.com](https://www.tempus.com/news/pr/tempus-announces-pricing-of-initial-public-offering/)
- Tempus AI acquires Deep 6 AI — [tempus.com](https://www.tempus.com/news/pr/tempus-announces-acquisition-of-deep-6-ai/)
- Mendel.ai — [mendel.ai](https://www.mendel.ai/)
- Unlearn.AI FDA/EMA framework — [unlearn.ai](https://www.unlearn.ai/blog/how-unlearn-boosts-trial-power-using-the-fdas-ai-framework)
- Medidata Synthetic Control Arm — [medidata.com](https://www.medidata.com/en/clinical-trial-products/medidata-ai/real-world-data/synthetic-control-arm/)
- Medable — [medable.com](https://www.medable.com/)
- Flatiron Health / Roche acquisition — [flatiron.com](https://resources.flatiron.com/press/press-release/roche-completes-acquisition-of-flatiron-health)
- Komodo Health ISPOR 2025 — [komodohealth.com](https://www.komodohealth.com/press/rwe-platform-drives-patient-journey-research-studies-ispor-2025/)
- Truveta Live Link — [truveta.com](https://www.truveta.com/blog/news/truveta-live-link-real-world-evidence/)

### MIT Jameel Clinic and related
- Regina Barzilay — *2025 IEEE Frances E. Allen Medal* — [jclinic.mit.edu](https://jclinic.mit.edu/2025-ieee-regina-barzilay/)
- Tommi Jaakkola — [eecs.mit.edu](https://www.eecs.mit.edu/people/tommi-jaakkola/)
- Jim Collins — *NG1 / DN1 MIT News* — [news.mit.edu](https://news.mit.edu/2025/using-generative-ai-researchers-design-compounds-kill-drug-resistant-bacteria-0814)
- Caroline Uhler — [idss.mit.edu](https://idss.mit.edu/staff/caroline-uhler/)
- Manolis Kellis Lab — [compbio.mit.edu](https://compbio.mit.edu/)
- MIT Jameel Clinic home — [jclinic.mit.edu](https://jclinic.mit.edu/)

### FDA and regulatory
- FDA. *AI/ML-Enabled Medical Devices* — [fda.gov](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices)
- FDA. *Paige Prostate DEN200080* — [accessdata.fda.gov](https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN200080.pdf)
- FDA. *EndeavorRx DEN200026* — [accessdata.fda.gov](https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN200026.pdf)
- FDA. *Mahana IBS K211372* — [accessdata.fda.gov](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K211372.pdf)
- The Imaging Wire. *FDA AI approvals past 1k* (Dec 2025) — [theimagingwire.com](https://theimagingwire.com/2025/12/10/ai-enabled-medical-devices-granted-fda-marketing-authorization/)
- Innolitics. *2025 AI/ML 510(k) Review* — [innolitics.com](https://innolitics.com/articles/year-in-review-ai-ml-medical-device-k-clearances/)

### Medical imaging and scribes
- Aidoc. *Jan 2026 comprehensive CT foundation model* — [aidoc.com](https://www.aidoc.com/about/news/aidoc-secures-fda-clearance-for-healthcares-first-comprehensive-foundation-model-ai/)
- Viz.ai. *Viz AAA first AAA AI clearance* — [viz.ai](https://www.viz.ai/news/viz-ai-is-first-to-receive-fda-510-k-clearance-for-ai-algorithm-for-abdominal-aortic-aneurysm)
- Viz.ai. *Viz ICH Plus Feb 2024* — [viz.ai](https://www.viz.ai/news/viz-ai-receives-fda-510k-clearance-for-artificial-intelligence-algorithm-for-the-quantification-of-intracerebral-hemorrhage)
- Paige. *First digital pathology AI clearance* — [businesswire.com](https://www.businesswire.com/news/home/20210922005369/en/)
- PathAI. *AISight Dx June 2025 FDA clearance* — [pathai.com](https://www.pathai.com/news/pathai-receives-fda-clearance-for-aisight-dx-platform-for-primary-diagnosis)
- HeartFlow. *Category I CPT Jan 2024* — [heartflow.com](https://www.heartflow.com/press-release/Heartflow-announces-new-reimbursement-code-and-increased-payment-for-ffrct/)
- HeartFlow. *AI plaque analysis CPT* — [heartflow.com](https://www.heartflow.com/newsroom/heartflow-announces-new-category-i-cpt-code/)
- Nanox. *HealthCCSng V2.0 clearance* — [investors.nanox.vision](https://investors.nanox.vision/news-releases/news-release-details/nanox-receives-fda-clearance-healthccsng-v20-upgraded-version)
- Microsoft. *Dragon Copilot March 2025* — [news.microsoft.com](https://news.microsoft.com/source/2025/03/03/microsoft-dragon-copilot-provides-the-healthcare-industrys-first-unified-voice-ai-assistant-that-enables-clinicians-to-streamline-clinical-documentation-surface-information-and-automate-task/)
- Ma et al. *Ambient AI scribes physician burnout* — *JAMIA* 2024-2025 — [pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC11756571/)
- Fierce Healthcare. *Abridge $300M Series E June 2025* — [fiercehealthcare.com](https://www.fiercehealthcare.com/ai-and-machine-learning/ambient-ai-startup-abridge-scores-300m-series-e-backed-a16z-and-khosla)
- HIT Consultant. *Abridge $250M Series D* — [hitconsultant.net](https://hitconsultant.net/2025/02/17/abridge-raises-250m-to-advance-ai-powered-clinical-documentation/)
- Mobihealthnews. *Suki $70M* — [mobihealthnews.com](https://www.mobihealthnews.com/news/suki-secures-70m-enhance-its-ai-ambient-scribe-offerings)
- Healthcare Dive. *Commure acquires Augmedix $139M* — [healthcaredive.com](https://www.healthcaredive.com/news/commure-acquire-augmedix-ai-documentation/721978/)
- Healthcare Dive. *Epic AI Charting Feb 2026* — [healthcaredive.com](https://www.healthcaredive.com/news/epic-rolls-out-ai-charting-art-notetaking-documentation-scribe/811462/)

### Medical LLMs and benchmarks
- Singhal et al. *Med-PaLM* — *Nature* 2023 — [nature.com](https://www.nature.com/articles/s41586-023-06291-2)
- Singhal et al. *Med-PaLM 2* — *Nature Medicine* Jan 2025 — [nature.com](https://www.nature.com/articles/s41591-024-03423-7)
- Saab et al. *Med-Gemini* — [arxiv.org](https://arxiv.org/abs/2404.18416)
- Tu et al. *AMIE Nature April 2025* — [nature.com](https://www.nature.com/articles/s41586-025-08866-7)
- Arora et al. *HealthBench* — [arxiv.org](https://arxiv.org/abs/2505.08775)
- John Snow Labs. *Medical LLM leaderboard May 2024* — [johnsnowlabs.com](https://www.johnsnowlabs.com/john-snow-labs-achieves-new-state-of-the-art-medical-llm-accuracy-benchmarks-outperforming-gpt-4-med-palm2-and-hundreds-of-others/)
- Crunchbase. *OpenEvidence Series D $12B Jan 2026* — [news.crunchbase.com](https://news.crunchbase.com/venture/openevidence-ai-doctors-doubles-valuation-seriesd/)
- Hippocratic AI Series C $126M — [hippocraticai.com](https://hippocraticai.com/hippocratic-ai-announces-series-c-funding-126-million/)

### Symptom triage and DTx
- Gilbert et al. *BMJ Open symptom-checker comparison* — [pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC7745523/)
- TechCrunch. *Babylon collapse* — [techcrunch.com](https://techcrunch.com/2023/08/31/the-fall-of-babylon-failed-tele-health-startup-once-valued-at-nearly-2b-goes-bankrupt-and-sold-for-parts/)
- Akili. *EndeavorOTC FDA authorization* — [businesswire.com](https://www.businesswire.com/news/home/20240618578815/en/)
- Stern et al. *Germany DiGA reimbursement* — *npj Digital Medicine* 2024 — [nature.com](https://www.nature.com/articles/s41746-024-01137-1)
