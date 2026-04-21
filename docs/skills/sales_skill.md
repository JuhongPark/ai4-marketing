# Sales Skill — AI for Selling and Investment

Scope: **selling** (moving customers toward purchase, renewal, advocacy) **+ investment** (raising, deploying, defending capital). Both sub-skills share a single organizational logic: *use AI to surface and act on signals that another human — a buyer or an investor — will otherwise miss, mis-time, or misprice.*

This document re-reads existing DDs through the selling lens, and introduces a body of investment-focused research that the domain-axis surveys did not cover.

---

## 1. Skill Subdivisions

```
SALES SKILL
├── 1.1 Selling (customer-facing)
│     ├── Targeting & segmentation
│     ├── Recommendation & personalization
│     ├── Conversion, pricing, creative
│     ├── Retention, winback, expansion
│     ├── Marketing-mix allocation & attribution
│     ├── Viral / influencer amplification
│     └── Sales enablement & conversational commerce
│
└── 1.2 Investment (capital-facing)
      ├── VC deal sourcing & screening (external investor view)
      ├── Due diligence automation (both sides)
      ├── Investor relations & earnings (issuer side)
      ├── Pitch, fundraising narrative, commercialization framing
      └── AI-augmented CFO workflows
```

The two halves are joined by a common thread: **both are ultimately about convincing a human decision-maker (buyer, investor) in a high-stakes, low-information regime**. Many of the underlying AI methods (RAG, LLM summarization, embedding retrieval, uplift modeling) transfer across the halves with minimal translation.

---

## 2. Selling Sub-Skills

### 2.1 Targeting and Segmentation

**What it does**: Pick the right customers to address with the right intensity.

**Repo coverage** (read these first):
- `docs/01_customer_segmentation.md` — RFM-Net, LRFMS, XAI segmentation
- `docs/deep_dive/DD4_customer_churn_cascade.md` §7 — uplift modeling (who responds vs. who would churn anyway)
- `docs/deep_dive/DD7_mit_marketing_research.md` §2.4 — Simester on ML-targeting robustness (Management Science 2020)

**Sales-lens reading**: Segmentation is a ranking-for-outreach problem. The question is never *"which cluster is this customer in?"* — it is *"what is the expected incremental revenue from spending the next marketing dollar on this customer?"* That is uplift, not classification. The Simester paper is the empirical benchmark for making that call robustly in retail contexts.

**KPI map**:
- Targeting precision → cost per acquisition (CPA)
- Uplift lift → marginal ROI of outreach
- XAI / OCT interpretability → CMO confidence, regulatory audit

### 2.2 Recommendation and Personalization

**Repo coverage**:
- `docs/02_recommendation_systems.md` — GNN-based recommenders (landing page)
- `docs/deep_dive/DD1_gnn_recommender_systems.md` — PinSage, LightGCN, KGAT, SR-GNN
- `docs/frontier/llm_enhanced_recommenders.md` — P5, TALLRec, LLaRA, LLM-SRec

**Sales-lens reading**: Every recommender decision is a direct lever on revenue-per-session. LightGCN and PinSage are not just modeling wins — PinSage's 3B-node / 18B-edge Pinterest deployment is a direct source of homepage revenue. The LLM-recommender wave adds cold-start coverage (zero- and few-shot), which directly attacks the first-session conversion problem where classical CF fails.

**Sales rule of thumb**: use GNN for warm-session revenue; use LLM recommenders for new-customer / cold-start revenue; fuse the two when the funnel mixes both (most e-commerce).

### 2.3 Pricing, Promotion, and Creative

**Repo coverage**:
- `docs/deep_dive/DD5_marketing_mix_modeling.md` — Bayesian MMM, Robyn, Meridian
- `docs/deep_dive/DD7_mit_marketing_research.md` §3.1 — Perakis (3-10% profit lift with Oracle RGBU promotion optimization)
- `docs/frontier/generative_ai_and_llms_in_marketing.md` §3 — Stable Diffusion for CTR-optimized creative (−78% design cost, +60% engagement)

**Sales-lens reading**: Diffusion-generated creative combined with a CTR predictor is a closed-loop **creative optimization** system — not a technology demo. Perakis' retail analytics work is the optimization counterpart: once the creative is generated, *which price, which promotion, which assortment* maximizes profit? The two research strands belong in the same operating cadence.

### 2.4 Retention, Winback, and Expansion

**Repo coverage**:
- `docs/04_churn_prediction.md` — ML and deep learning for churn
- `docs/deep_dive/DD4_customer_churn_cascade.md` — CCP-Net, contagion, uplift
- `docs/deep_dive/DD2_deep_rl_viral_marketing.md` §4.3 — BanditLP (LinkedIn email marketing with cadence/fairness/budget constraints)

**Sales-lens reading**: Churn prediction alone is useless for sales — prediction without intervention selection wastes retention budget on customers who would have stayed anyway. **Uplift modeling** (DD4 §7, Gutierrez & Gerardy 2017, CausalML) is the sales-critical complement. BanditLP (LinkedIn, 2026) shows how to operationalize this under real cadence and fairness constraints at scale.

**KPI map**:
- Uplift-ranked retention spend → preserved revenue per retention dollar
- Cadence constraints → avoided unsubscribe cost
- Segment fairness → regulatory and DEI defensibility

### 2.5 Marketing Mix Modeling and Attribution

**Repo coverage**:
- `docs/05_marketing_mix_modeling.md` — PyMC-Marketing, Robyn, Meridian, DNAMTA, CAMTA
- `docs/deep_dive/DD5_marketing_mix_modeling.md` — Jin 2017 (Google Bayesian MMM foundation), DeepCausalMMM (2025), Amazon Ads MTA (2025)

**Sales-lens reading**: MMM is the CFO's and CMO's shared language for "how much and where." The Jin (2017) priors on adstock and shape enable domain-informed Bayesian inference — small data, high-stakes, requires priors. Meridian (Google, open-sourced 2024) is the post-cookie answer: aggregate-level causal attribution that respects modern privacy while still informing allocation.

**Sales workflow**: MMM → incrementality test → re-prior → MMM rerun. The quarterly loop is the substrate for every budget decision.

### 2.6 Viral and Influencer Amplification

**Repo coverage**:
- `docs/03_influencer_marketing.md` — DeepInf, Aral experiments, Instagram ML
- `docs/deep_dive/DD2_deep_rl_viral_marketing.md` §2 — DGN, BiGDN, HEDRL-IM (hypergraph), BIM-DRL (fair spread)
- `docs/deep_dive/DD3_influencer_marketing.md` — DeepInf, HetInf, DeepPP; Aral's randomized-experiment program
- `docs/deep_dive/DD7_mit_marketing_research.md` §2.3 — Aral's Management Science / Science / Nature Communications findings

**Sales-lens reading**: Viral seed selection is the only place marketing gets **network-structure leverage** — the right 1,000 influencers beat the wrong 100,000 every time. HEDRL-IM matters sales-operationally because the dominant viral channels now are **group-based** (WhatsApp, Discord, Slack, Telegram) and pairwise-edge IM models undersell the spread. BIM-DRL matters when brand-equity or compliance demand balanced spread — DEI-conscious consumer brands, regulated markets, cross-cultural launches.

### 2.7 Sales Enablement and Conversational Commerce

**Repo coverage (partial)**:
- `docs/frontier/generative_ai_and_llms_in_marketing.md` §5 — Retail-GPT, Graph-Enhanced RAG, Amazon QAC RAG
- `docs/frontier/agentic_ai_marketing.md` §2 — multi-agent specialized roles (Lead Qualification, Campaign Orchestrator)

### 2.8 Academic Marketing Research on Generative AI (beyond MIT)

The repo's academic spine is DD7 (MIT Sloan / ORC). The skills-axis expansion adds peer-reviewed GenAI marketing research from other institutions, emphasizing field experiments over theoretical treatments.

| Paper / Source | Institution / Authors | Key finding | Link |
|---|---|---|---|
| Kapoor & Kumar. *Frontiers: Generative AI and Personalized Video Advertisements* (2025) | Marketing Science 44(4) | Mobile-ad WhatsApp field experiment; GenAI personalized video ads increase engagement by **+6-9pp** over image ads and generic videos | [INFORMS](https://pubsonline.informs.org/doi/10.1287/mksc.2023.0494) |
| Hermann & Puntoni. *Generative AI in Marketing and Principles for Ethical Design and Deployment* (2025) | Sage / J. of Marketing | Ethical-design principles synthesizing 2023-2025 GenAI marketing literature | [Sage](https://journals.sagepub.com/doi/10.1177/07439156241309874) |
| V. Kumar, Kotler, Gupta, Rajan. *Generative AI in Marketing: Promises, Perils, and Public Policy Implications* (2025) | Sage / JAMS-adjacent | Policy-focused synthesis from canonical marketing scholars | [Sage](https://journals.sagepub.com/doi/10.1177/07439156241286499) |
| *How Generative AI Is Shaping the Future of Marketing* (2024-2025) | JAMS (Springer) | Framework paper on GenAI's impact across marketing functions | [Springer](https://link.springer.com/article/10.1007/s11747-024-01064-3) |
| Columbia Business School. *GenAI field experiments at a cross-border retailer* (2025) | Columbia Marketing | Six-month large-scale RCT: GenAI integration across 7 consumer workflows yields **0% to 16.3% sales lift** | See 2025 CBS AI/ML/Business Analytics Conference proceedings |
| Stanford GSB. *AI and Marketing: New Methods and New Risks Conference* | Stanford (Athey group) | Causal-ML for marketing intervention evaluation; adaptation of NLP-causal methods | [Stanford GSB](https://www.gsb.stanford.edu/events/ai-marketing-new-methods-new-risks-conference) |
| AI at Wharton. *Generative AI Adoption Study* | Wharton | **AI adoption in marketing/sales tripled 20% (2023) → 62% (2024)** | [BusinessWire](https://www.businesswire.com/news/home/20241028137398/en/Generative-AI-Adoption-Surges-New-Study-by-AI-at-Wharton-Reveals-Doubling-of-AI-Use-Across-Key-Business-Functions) |
| Wharton Executive Education. *AI in Marketing: Creating Customer Value in an AI-Driven Enterprise* (2025) | Wharton | Program curriculum as market signal for executive skills demand | [Wharton](https://news.wharton.upenn.edu/press-releases/2025/10/wharton-executive-education-launches-ai-in-marketing-creating-customer-value-in-an-ai-driven-enterprise/) |
| MIT Sloan Management Review. *Gain Consumer Insight With Generative AI* (2024-2025) | MIT SMR | Practitioner framework for LLM-based consumer research | [SMR](https://sloanreview.mit.edu/article/gain-consumer-insight-with-generative-ai/) |

**Sales-lens reading**: the Kapoor & Kumar (2025) result is the cleanest production-grade causal evidence that GenAI personalized video outperforms image and generic video at ~7pp engagement, under real ad-platform conditions. It is the reference study to cite when defending creative investment in GenAI video. Columbia's 0-16.3% lift range is a useful counter-anchor: **GenAI returns vary widely by workflow**; the story is not uniform.

### 2.9 Vertical Sub-Skills — Selling AI in Specific Industries

Where the repo's DDs are domain-agnostic, actual production AI selling differs meaningfully by vertical. A short tour:

**SaaS and Product-Led Growth (PLG)**:
- 69% of bootstrapped and equity-backed SaaS teams use AI to eliminate bottlenecks (Freemius 2025 Micro-SaaS state report)
- Usage-based pricing is becoming the norm; hybrid models outperform flat subscriptions
- AI copilots shipping with flexible paywalls that flex by use-case / growth stage / team size
- OpenAI's **$1.1B acquisition of Statsig** (2026) signals that feature-flagging + experimentation is strategic to AI-era product monetization
- Sources: [Freemius Micro-SaaS 2025](https://freemius.com/blog/state-of-micro-saas-2025/); [Bessemer AI Pricing Playbook](https://www.bvp.com/atlas/the-ai-pricing-and-monetization-playbook); [Metronome AI Pricing Field Report 2025](https://metronome.com/blog/ai-pricing-in-practice-2025-field-report-from-leading-saas-teams); [L.E.K. — How AI Is Changing SaaS Pricing](https://www.lek.com/insights/tmt/global/ar/how-ai-changing-saas-pricing); [Lago — 6 Proven Pricing Models for AI SaaS](https://getlago.com/blog/6-proven-pricing-models-for-ai-saas); [getmonetizely 2026 Guide to SaaS, AI, Agentic Pricing Models](https://www.getmonetizely.com/blogs/the-2026-guide-to-saas-ai-and-agentic-pricing-models)

**Fintech / Lending / Payments**:
- AI credit-underwriting market: **$1.6B (2024) → $2.17B (2025), ~35% CAGR** (Marqeta / codewave)
- **83% of lenders plan to increase GenAI budgets in 2026**
- Platforms: Upstart, SoFi, Zest AI, finbots.ai, Liberis
- MNT-Halan automated **>50% of loan approvals** and achieved a **60% approval rate on previously unscoreable users** — AI directly expands addressable market, the clearest demonstration of AI-as-sales-growth in any vertical
- Sources: [Zest AI](https://www.zest.ai/); [V7 — AI Commercial Loan Underwriting 2025](https://www.v7labs.com/blog/ai-commercial-loan-underwriting); [WEF — AI Credit Scoring and Financial Inclusion (Oct 2025)](https://www.weforum.org/stories/2025/10/how-responsibly-deploying-ai-credit-scoring-models-can-progress-financial-inclusion/); [Codewave — AI in Fintech 2025](https://codewave.com/insights/fintech-ai-companies-use-cases/); [Marqeta — AI in Payments and Fintech 2025](https://www.marqeta.com/blog/ai-in-payments-and-fintech-enhancing-human-decision-making-and-innovation); [MDPI FinTech — Credit Sales and Risk Scoring](https://www.mdpi.com/2674-1032/4/3/31)

**Retail / CPG / FMCG**:
- **80%+ of retailers plan increased AI shelf-management investment over the next two years** (Pygmalios 2025)
- **65% use AI for product-placement optimization**
- Global AI in retail market: **$5.5B (2022) → $55B (2030)** projection; AI retail supply-chain spend exceeds **$10B in 2025**
- Agentic store-planning stacks: Demand Agent + Space Agent + Allocation Agent + Pricing Agent + Compliance Agent (couture.ai 2026)
- Sources: [Intelligence Node — Digital Shelf Optimization](https://www.intelligencenode.com/blog/digital-shelf-optimization-google-ai/); [ParallelDots — Retail Shelf-Scanning AI](https://www.paralleldots.com/resources/blog/shelf-optimization-ai-ml-scan); [Pygmalios — Best AI Analytics for FMCG Shelf Performance 2025](https://www.pygmalios.com/insights/best-ai-analytics-fmcg-shelf); [Couture.ai — Smarter Store Planning 2026](https://couture.ai/blog/smarter-store-planning-why-sku-level-decisions-matter-in-2026/); [Toolio — AI Retail Planning](https://www.toolio.com/post/how-ai-improves-every-stage-of-retail-planning); [CommerceIQ — Digital Shelf Optimization](https://www.commerceiq.ai/digital-shelf-optimization); [DeltaSales — AI for CPG Brands](https://deltasalesapp.com/blog/ai-for-cpg-brands-to-boost-sales)

**Healthcare / Life Sciences (HCP Targeting)**:
- AI-powered HCP targeting: **20-50% representative-efficiency improvement**, **5-10% HCP-satisfaction uplift**, **3-5% prescriber-number increase**, **5-10% incremental revenue** (IQVIA 2025)
- Late-2025 platform launches: **Veeva AI Agents** (four agents — Free Text, Voice, Pre-call, Media — using Anthropic LLMs on Amazon Bedrock); **Salesforce Life Sciences Cloud for Customer Engagement (LSC4CE)** with IQVIA (Sep 2025)
- First-party data is disproportionately valuable because HCP digital signals are scarcer than consumer signals
- Sources: [Doceree — AI in Healthcare Marketing 2025](https://blog.doceree.com/ai-in-healthcare-marketing-and-its-impact-in-2025); [Salesforce — HCP Targeting 2025](https://www.salesforce.com/life-sciences/cloud/hcp-targeting/); [Talking Medicines — 5 AI Trends for Healthcare Marketing 2025](https://talkingmedicines.com/2025/01/5-ai-trends-driving-healthcare-marketing-in-2025/); [IQVIA — Harnessing AI for Global HCP Targeting](https://www.iqvia.com/blogs/2025/07/harnessing-ai-for-effective-global-hcp-targeting-and-engagement); [BCG X — Healthcare Commercial AI](https://www.bcg.com/x/product-library/healthcare-commercial-ai); [Agilisium — How AI Is Transforming HCP Engagement](https://www.agilisium.com/blogs/beyond-traditional-sales-how-ai-agents-are-transforming-hcp-engagement-in-life-sciences); [IntuitionLabs — HCP Engagement Solutions](https://intuitionlabs.ai/articles/hcp-engagement-solutions); [Pharmaphorum — AI-Powered Innovations in HCP Engagement](https://pharmaphorum.com/sales-marketing/life-sciences-marketing-ahead-unveiling-role-ai-powered-innovations-hcp-engagement); [Swoop — Predictive AI Targeting](https://swoop.com/blog/swoop-launches-predictive-ai-targeting-the-next-generation-of-healthcare-marketing/); [L7 Creative — AI Transforms Pharma Marketing 2025](https://www.l7creative.com/ai-marketing/ai-transforms-pharma-marketing-2025/)

**Enterprise B2B / ABM (Account-Based Marketing)**:
- **70% of B2B marketers** use ABM; implementers report **208% revenue lift** from marketing; **87% of ROI-measuring marketers** say ABM outperforms every other investment
- **98% of B2B marketers** consider intent data essential; **91% use intent data and scoring to prioritize ABM accounts**
- Global ABM market: **$1.2B (2024), 12.4% CAGR through 2030**
- Sources: [ABM Agency — Intent Data for Smarter ABM 2025](https://abmagency.com/intent-data-the-secret-weapon-for-smarter-abm-in-2025/); [Factors.ai — Buyer Intent Data 2025](https://www.factors.ai/blog/buyer-intent-for-abm); [AriseGTM — AI-Enabled ABM 2025](https://arisegtm.com/blog/ai-enabled-abm-2025); [ZoomInfo Pipeline — ABM Software 2026](https://pipeline.zoominfo.com/marketing/abm-software); [AI-Ark — Best B2B Intent Data Providers 2025](https://ai-ark.com/blog/b2b-intent-data-providers/); [Informa TechTarget ABM press](https://www.businesswire.com/news/home/20251106291963/en/Propelled-by-Proprietary-Intent-Data-Informa-TechTarget-Continues-Account-Based-Marketing-ABM-Leadership)

**Sales-lens cross-vertical reading**: fintech and healthcare show the clearest causal evidence of **AI expanding the salable market** (underwriting previously unscoreable users; targeting HCPs at scale). SaaS shows AI changing the **pricing architecture** (usage-based over flat). Retail/CPG shows AI becoming the **merchandising substrate**. Enterprise B2B shows AI fused with **intent data** as the core of demand generation. These are not minor variations — they are fundamentally different monetization architectures depending on the vertical.

**New to repo via this skill axis** — production-side sales enablement and B2B selling:

| Signal | Value | Source |
|---|---|---|
| AI adoption in demand gen (US firms) | **87%** | Kixie / B2B sales 2025 |
| Positive impact from at least one AI initiative | **65%** | same |
| Predicted AI-CRM standard practice (B2B top teams) | **>90% by 2027** | Salesforce State of Sales |
| Forecast-accuracy lift from AI-driven predictive analytics | **+43%** | same |
| Sales-enablement time saved per rep | **12-15 hours / week** | same |
| Win-rate lift from early AI deployments | **+30%** | Bain & Company 2025 |
| Sales professionals using AI daily | **56%** | LinkedIn 2025 |
| Customer journeys occurring via AI conversational interfaces (2028) | **70%** | HelloRep 2025 forecast |

**Sales-lens reading**: Conversational commerce is no longer an FAQ-bot story. HBS research (2025) shows AI chatbots raise customer sentiment by 1.63 points and cut response time 22%. AI-driven proactive chats recover 35% of abandoned carts. Microsoft Advertising reports 73% higher CTRs on conversational-AI ad placements. The sales function now has a copilot layer in the same way engineering has Copilot / Cursor.

**New references** (added to CATALOG):
- Kixie. *AI Sales Enablement Trends in 2025* ([link](https://www.kixie.com/sales-blog/ai-sales-enablement-trends-in-2025-you-need-to-know/))
- Martal Group. *AI-Enhanced Pipeline Management in 2025* ([link](https://martal.ca/pipeline-management-lb/))
- Forecastio. *AI Sales Forecasting for B2B* ([link](https://forecastio.ai/blog/ai-sales-forecasting))
- HelloRep. *The Future of AI in Ecommerce: 40+ Statistics on Conversational AI Agents for 2025* ([link](https://www.hellorep.ai/blog/the-future-of-ai-in-ecommerce-40-statistics-on-conversational-ai-agents-for-2025))
- Cirrus Insight. *AI in Sales 2025: Statistics, Trends & Generative AI Insights* ([link](https://www.cirrusinsight.com/blog/ai-in-sales))
- Microsoft Advertising. *Conversational AI Redefines Audience Engagement — 73% Higher CTRs* (August 2025) ([link](https://about.ads.microsoft.com/en/blog/post/august-2025/73-higher-ctrs-why-advertisers-need-to-pay-attention-to-conversational-ai))

---

## 3. Investment Sub-Skills

This half of the skill is **new research** — the existing DDs briefly touch fundraising (DD8 §2.3 on AI commercialization) but do not survey the VC-side, IR-side, or CFO-side AI stacks. What follows is the net-new contribution of the skills axis.

### 3.1 VC Deal Sourcing and Screening

**Market scale (2025)**:
- AI / ML startups captured **~61% of global VC deal value ($258.7B of $427.1B)** in 2025, up from 30% in 2022 (OECD 2025 report)
- Median VC review of a deck is under 3 minutes — algorithmic screening is not a convenience, it is the gate

**What AI does at the VC screening stage**:

| Function | AI approach | Operating impact |
|---|---|---|
| **Deal sourcing** | Index >30M companies + hiring signals + early traction → surface pre-seed / stealth startups before they fundraise | Catches deals before formal processes |
| **Screening** | XGBoost / gradient-boost models on founder / team / market features | Reported **+25% hit rate over median VC** (Hustle Commons) |
| **Portfolio fit** | Embedding similarity against mandate and existing portfolio | De-risks thesis drift |
| **Bias mitigation** | Algorithmic scoring on representative datasets | Hedges founder-demographic bias of all-human decisions |
| **Founder-market fit evaluation** | LLM synthesis of founder interviews, past ventures, network | Preserved as human judgment domain (HBR 2025) |

**Sales-lens reading for the founder**: if an XGBoost model meaningfully beats median VC hit rate, the founder's own pitch material should be engineered to **pass the algorithmic first-pass filter** — structured founder traits, demonstrated market, traction data in machine-parseable form. The pitch is no longer aimed at a human first.

**New references**:
- CFA Institute (2025). *AI in Venture Capital: Separating Signal from Noise* ([link](https://blogs.cfainstitute.org/investor/2025/12/10/ai-in-venture-capital-separating-signal-from-noise/))
- OECD (2025). *Venture Capital Investments in Artificial Intelligence through 2025* — full report ([link](https://www.oecd.org/en/publications/venture-capital-investments-in-artificial-intelligence-through-2025_a13752f5-en/full-report.html))
- Harvard Business Review (2025). *How Generative AI Is Reshaping Venture Capital* ([link](https://hbr.org/2025/11/how-generative-ai-is-reshaping-venture-capital))
- Affinity (2026). *10 AI Tools for Venture Capital Firms in 2026* ([link](https://www.affinity.co/guides/vc-ai-tools))
- Qubit Capital. *AI Startup Screening: How AI Tools Streamline Investor Decisions* ([link](https://qubit.capital/blog/ai-startup-screening))
- Crunchbase News. *3 Ways AI Is Transforming Venture Capital Investment* ([link](https://news.crunchbase.com/ai/transforming-vc-investment-nikkhoo-navigate/))
- BestBrokers. *Venture Capital 2025 Recap: AI Startups Capture Over Half of Global Funding* ([link](https://www.bestbrokers.com/forex-brokers/the-state-of-ai-venture-capital-in-2025-ai-boom-slows-with-fewer-startups-but-bigger-bets/))
- Equisy (2025). *How AI Is Reshaping Investor-Startup Matching in 2025/26* ([link](https://equisy.io/how-ai-is-reshaping-investor-startup-matching-in-2025-26/))

### 3.2 Due Diligence Automation

**Technology pattern**: NLP / RAG over legal, financial, and regulatory documents; extraction → classification → risk scoring → drafting.

**Key industry data points**:
- Vantager (AI due-diligence platform) went public July 2025 with USD 100B of client AUM
- Y Combinator's Request for Startups 2025 explicitly invites *"start a law firm, staff it with AI agents, and compete with existing law firms"*
- Gartner (2025): >40% of enterprise apps will embed task-specific AI agents by end of 2026, up from <5% at end of 2025
- Harvey, LEGALFLY, Spellbook, Sirion represent production-grade AI due-diligence stacks for contract analysis, M&A review, compliance

**Dual-lens observation**: the investor runs AI DD on the startup; the startup runs AI DD on its own contracts before the investor asks. The same toolchain serves both sides of the table. LLM + RAG + structured extraction over legal corpora is the common substrate.

**New references**:
- Sirion. *AI Due Diligence: The Hidden Accelerant in Modern Deal-Making* ([link](https://www.sirion.ai/library/contract-ai/ai-due-diligence/))
- The Hedge Fund Journal. *Applying AI to Investment, Operational and Legal Due Diligence (Vantager)* ([link](https://thehedgefundjournal.com/vantager-ai-investment-operational-legal-due-diligence/))
- Harvey. *AI Platform for Legal and Professional Services* ([link](https://www.harvey.ai))
- LegalFly. *Top 7 AI Tools for M&A Due Diligence 2026* ([link](https://www.legalfly.com/post/the-top-7-ai-tools-for-m-a-due-diligence))
- Spellbook. *AI Due Diligence: Faster M&A and Legal Review* ([link](https://www.spellbook.legal/learn/ai-due-diligence))
- Angel School. *How AI Is Transforming Deal Sourcing & Startup Due Diligence in 2025* ([link](https://www.angelschool.vc/blog/how-artificial-intelligence-is-transforming-deal-sourcing-and-startup-due-diligence))
- IMAA Institute. *M&A Due Diligence AI Tools* ([link](https://imaa-institute.org/blog/ai-for-due-diligence/))

### 3.3 Investor Relations and Earnings

**Gartner (Jan 2026) survey of 146 CFOs (Oct-Dec 2025)**: 35%+ reported increases in volume, frequency, and time-sensitivity of investor communications in 2025 vs. 2024. AI research tools used by institutional investors mean CFOs must spend **more** time, not less, on IR — the question asymmetry has shifted.

**Production IR-AI stacks**:

| Tool | Function |
|---|---|
| **ChatFin, AlphaSense** | "Red-team" bots: read a draft earnings script, generate tough analyst questions the CFO will face |
| **Irwin** | Investor targeting: analyze institutional holdings and trading behaviors; surface high-compatibility funds that own peers but not you |
| **Q4** | IR Ops platform: website + events + digital investor-behavior tracking |
| **Workiva** | Assured integrated reporting: ESG + financial consistency enforcement |
| **BlueFlame AI** | Fundraising and IR workflow automation |

**Predictive AI in stock market (segment): $0.84B (2025) → $1.82B (2030)** (ChatFin / industry data).

**New references**:
- Gartner (Jan 2026). *Gartner Says AI Is Leading CFOs to Spend More Time on Investor Relations* ([link](https://www.gartner.com/en/newsroom/press-releases/2026-01-21-gartner-says-ai-is-leading-cfos-to-spend-more-time-on-investor-relations))
- IR Impact (March 2026). *AI-Driven Investor Relations: From Static Targeting to Agentic Workflows* ([link](https://www.ir-impact.com/2026/03/ai-driven-investor-relations-from-static-targeting-to-agentic-workflows/))
- ChatFin. *Top 10 Best AI Tools for Investor Relations* ([link](https://chatfin.ai/blog/top-10-best-ai-tools-for-investor-relations-ir/))
- V7 Labs. *How AI Can Fix Investor Reporting and Investor Relations* ([link](https://www.v7labs.com/blog/ai-investor-reporting))
- BlueFlame AI. *AI for Investor Relations & Fundraising* ([link](https://www.blueflame.ai/solutions/investor-relations))
- Nasdaq. *Maximizing Investor Engagement with Intelligent IR Platforms* ([link](https://www.nasdaq.com/articles/ir-intelligence/investor-relations-platform))
- CFOTech. *CFOs Step Up Investor Relations as AI Tools Reshape IR* ([link](https://cfotech.ca/story/cfos-step-up-investor-relations-as-ai-tools-reshape-ir))

### 3.4 Pitch, Fundraising Narrative, and Commercialization

**Key research nucleus** (already partially in repo, now read through investment lens):
- `docs/deep_dive/DD8_mit_sdm_marketing.md` §2.3 — *A Strategic Perspective on the Commercialization of AI* (MIT SDM)

**Bessemer / VC-side framing on AI-PMF (2025)**:
- 2025 playbook: *micro-test → traction proof → MVP-lite → GTM validation → real usage* (not MVP → iterate → win)
- AI Supernova cohort: ~$40M ARR in year 1, ~$125M ARR in year 2
- Roughly 15 AI companies with ARR > $100M globally; three with >$1B (OpenAI ~$10B, Anthropic ~$4B, Scale AI ~$1.5B)
- AI-native startups reach PMF with smaller teams via higher automation

**Sales-lens reading for the founder**: the commercialization narrative investors expect in 2026 is no longer "we built the best model." It is "we compress the execution layer and expand strategic throughput." The shift from MVP-to-iterate to micro-test-to-traction is a **capital-efficiency** argument, which is the core thing an investor underwrites.

**New references**:
- Bessemer Venture Partners. *Mastering Product-Market Fit: A Detailed Playbook for AI Founders* ([link](https://www.bvp.com/atlas/mastering-product-market-fit-a-detailed-playbook-for-ai-founders))
- Bessemer Venture Partners. *The State of AI 2025* ([link](https://www.bvp.com/atlas/the-state-of-ai-2025))
- The VC Corner. *OpenAI's Product Lead on Redefining Product-Market Fit for AI Startups (2025 Guide)* ([link](https://www.thevccorner.com/p/ai-product-market-fit-framework-openai))
- AInvest. *AI Startups and the Evolution of Product-Market Fit in 2025* ([link](https://www.ainvest.com/news/ai-startups-evolution-product-market-fit-2025-2511/))
- TechCrunch (Nov 2025). *How AI Startups Should Be Thinking About Product-Market Fit* ([link](https://techcrunch.com/2025/11/11/how-ai-startups-should-be-thinking-about-product-market-fit/))
- 36Kr EU. *The First Year of AI Unicorns' Commercialization* ([link](https://eu.36kr.com/en/p/3530194899410052))
- World Economic Forum (April 2025). *How Founders Are Shaping the Future of Startups with AI* ([link](https://www.weforum.org/stories/2025/04/how-founders-are-shaping-the-future-of-entrepreneurship-with-ai/))

### 3.5 AI-Augmented CFO Workflows

Adjacent to IR but narrower: forecasting, scenario modeling, variance analysis.

**Pattern**: LLM-on-top-of-ERP reads transactions and contracts, produces variance narratives the controller used to write by hand. RAG grounded on the close ledger; evaluation via finance-team QA. Often built on the same platforms as IR (Workiva, BlueFlame) but used by the FP&A function, not IR.

**Open problem**: no widely-cited academic synthesis yet; the literature is almost entirely industry-analyst reports. See DD5 (MMM) for the closest academic parallel — Bayesian hierarchical models for marketing spend translate to scenario modeling for CFO budgets with minimal change.

### 3.6 Private Equity AI

PE firms now apply AI at three distinct points: **deal sourcing**, **operational DD**, and **portfolio value creation**. The first two overlap with §3.1 and §3.2 but have PE-specific tooling; the third is new territory.

**Who's doing what (2024-2026)**:

| Firm | Known AI activity | Source |
|---|---|---|
| **Blackstone** | Data-science team (led by Matt Katz / Jon Gray commentary) applies ML across 250+ portfolio companies for pricing, procurement, labor forecasting. Disclosed AI-driven EBITDA uplift themes in 2024-2025 earnings | Blackstone 2024 Annual Letter; FT, Bloomberg coverage |
| **KKR** | Partnered with **Palantir Foundry** for portfolio operations analytics; Henry McVey's macro team uses LLMs for thematic screening; "Capstone" operational group integrates AI into value-creation playbooks | FT, Bloomberg 2024-2025 |
| **EQT** | **Motherbrain** (operational since ~2016, continuously updated through 2024-2025): screens 50M+ companies, ~15 deals publicly credited by 2024 | [EQT ThinQ](https://eqtgroup.com/thinq) |
| **Bain Capital** | Announced **OpenAI partnership (Feb 2024)** for services layer spanning diligence + portfolio ops | [OpenAI Blog](https://openai.com/blog) |
| **Vista Equity Partners, Thoma Bravo** | Disclosed internal GenAI copilots for software portfolio companies (code migration, SG&A reduction); **10-30% productivity gains** reported in press | Various industry coverage |
| **Mid-market** | **Keystone** (Bain & Co. adjacent), **AlphaSense Generative Search**, **Grata**, **SourceScrub** + **DealCloud** dominate deal sourcing; **Rogo AI**, **Mosaic.tech** enter LBO-modeling assistance | PitchBook, Dealroom 2024-2025 |

**Sales-lens reading for LPs**: when a GP claims AI-driven value-creation, the credible markers are (a) portfolio-company named, (b) quantified EBITDA impact, (c) the model or vendor disclosed. Anything less is narrative.

**New references**:
- McKinsey. *Generative AI in Private Equity: A New Frontier for Value Creation* (2024) — mckinsey.com insights
- Bain & Co. *Global Private Equity Report 2024/2025* — AI chapter
- PitchBook. *How AI Is Rewiring Private Equity Deal Sourcing* (2024)
- Preqin. *AI Adoption in Alternatives 2024*
- HBR. *How Private Equity Firms Are Creating Value With AI* (2024)
- Gompers, Kaplan, Mukharlyamov. *What Do Private Equity Firms (Say They) Do?* — SSRN updated AI addendum (2024)

**Open problems**: (i) attribution — did AI drive portfolio EBITDA or is it a narrative layer? (ii) data contamination in DD LLMs trained on seller-provided CIMs. (iii) fund-of-funds AI screening is nascent; no dominant platform.

### 3.7 Quant Hedge Fund and Public-Equity AI

The hedge-fund side is mostly opaque, but the public record is richer than it was pre-2023.

**Quant Hedge Fund AI (publicly described)**:

| Firm | What's public | Academic signal |
|---|---|---|
| **Two Sigma** | Venn platform; NeurIPS / ICML submissions on transformers for time-series and RL for execution; discloses satellite, credit-card, shipping data usage | [Two Sigma Insights](https://twosigma.com/insights), arXiv 2024-2025 |
| **Citadel / Citadel Securities** | Aggressive ML-PhD recruiting; Ken Griffin discusses LLM integration for research productivity; market-making is heavily ML-driven (microstructure) | Bloomberg, Risk.net 2024 |
| **Renaissance Technologies** | Famously opaque; ex-RenTech researcher papers (2024) suggest ensemble-kernel + HMM hybrids | arXiv 2024 |
| **Man AHL** | Most academically open: transformer backbones for futures trading (2024-2025); documented LLM-for-earnings-calls pipeline | [Man Institute](https://man.com/ri) |
| **Jane Street** | OCaml throughout; 2024 conference talks reveal differentiable programming + deep RL for options market-making | ICFP 2024 tech talks |
| **D.E. Shaw** | "Pattern" lab; published graph-neural-network work on cross-asset prediction | — |
| **Hudson River Trading** | Low-latency ML inference on FPGAs/GPUs; opaque on alpha stack | Job posts 2024 |

**Alt-data vendors (the input layer for all of the above)**: Orbital Insight, RS Metrics (satellite); Yipit Data, Second Measure (credit-card / transaction panels); Thinknum, Similarweb (web-scraped).

**Public-Equity AI Tools (issuer + analyst side)**:

| Tool | Function | 2024-2025 signal |
|---|---|---|
| **Hebbia** | "Matrix" product — document-grid QA over 10-Ks, S-1s; used by BlackRock, Centerview, Charlesbank | $130M Series B (a16z, 2024) |
| **Rogo AI** | Banker-facing LLM for IB / PE workflows; integrated with CapIQ, PitchBook | ~$50M+ raised by 2025 |
| **AlphaSense + Tegus** | Sell-side research search + expert-call transcripts; ~$930M acquisition (June 2024) | Reuters, WSJ |
| **Kensho** (S&P Global) | Kensho Scribe and Kensho LLM-ready data across Capital IQ | S&P subsidiary since 2018 |
| **Daloopa** | Filing-structuring for DCF / model-building | Private |
| **FactSet GenAI Assistant** | Announced 2024; native embed in FactSet workstation | factset.com |

**Academic papers worth citing**:
- Cao, Jiang, Wang, Yang. *From Man vs. Machine to Man + Machine*. RFS (2024)
- Lopez-Lira & Tang. *Can ChatGPT Forecast Stock Price Movements?* SSRN / JFE 2024
- *GPT's Idea of Stock Factors*. SSRN 2024
- Bloomberg. *BloombergGPT* (arXiv 2303.17564, 2023 — flagged as foundational but pre-2024)
- *FinGPT / FinLLM* (ongoing arXiv series, 2024)
- Hoberg-Phillips TNIC database — text-based industry similarity, updated through 2024
- BIS Working Paper 1194, *AI in Asset Management* (2024)
- CFA Institute Research Foundation, *AI in Investment Management* (2024)

**Sales-lens reading**: for founders raising from hedge-fund-backed or PE-backed strategics, the investor is no longer just reading your S-1 — a Hebbia / Rogo / AlphaSense pipeline is reading your filings, your competitor filings, and every analyst comment. **Write for the retrieval layer**, not only the human reader.

### 3.8 AI Valuation and M&A Models

**Embedding-based comparable-company selection**: Hoberg-Phillips TNIC, extended by commercial vendors (Daloopa, Finsemble) using embedding similarity rather than GICS. This displaces rigid sector-code filters with semantic similarity.

**DCF-augmenting LLMs**: Bloomberg BloombergGPT (2023, foundational), FactSet GenAI Assistant (2024), domain-specific FinLLM variants. LLMs extract revenue drivers, segment disclosures, and assumption rationales from 10-Ks / S-1s — but hallucination in financial citations remains unsolved, and audit trails from LLM outputs back to source filings are inconsistent.

**M&A DD automation**: already covered in §3.2; the new angle for M&A specifically is post-merger integration (PMI) automation — LLMs synthesize employee-handbook conflicts, contract-portfolio overlaps, and system-of-record redundancies.

**SPAC / IPO pricing ML**: academic modest lift over bookbuilding; no dominant production tool; small, non-stationary samples are the blocker.

**Open problems**:
- Hallucination in financial citations with audit-trail uncertainty
- Embedding-based comparables lack regulatory / accounting treatment (which matters for valuation multiples)
- Finance-LLM benchmarks (FinBench, FinEval) are early-stage and gameable
- No dominant production SPAC / IPO pricing tool

### 3.9 Accelerator and Early-Stage AI

Adjacent to §3.4 (commercialization narrative). Covers the founder's AI-augmented experience inside an accelerator cycle:
- **Y Combinator W25 / S25 cohorts**: founders using LLMs for customer discovery, deck iteration, product demos; YC's 2025 RFS (Request for Startups) explicitly invites AI-native law firms and vertical AI agents
- **Grant / non-dilutive funding AI**: SBIR / Horizon Europe proposal drafting via LLM; currently informal, no established vendor category yet

This sub-skill is lightweight in 2026 but likely to grow.

---

## 4. Cross-Sub-Skill Synergies

The selling and investment halves share more substrate than they appear to:

| Underlying technology | Selling use | Investment use |
|---|---|---|
| **RAG** | Retail-GPT, catalog-grounded chat | Due-diligence doc review, earnings prep |
| **Embedding retrieval** | User-item matching for recommendation | Investor-company matching for IR targeting |
| **LLM summarization** | Customer review mining, ad copy | Deal-memo drafting, earnings Q&A |
| **Uplift / causal ML** | Retention decisions | Capital-allocation decisions (which business unit funding moves the needle) |
| **Bayesian hierarchical models** | MMM for channel allocation | Scenario modeling for CFO budgets |
| **Multi-agent orchestration** | Campaign orchestrator agent | IR workflow agent (Irwin-style) |
| **Bandit / OPE** | Personalization under cadence caps | A/B testing of investor communication formats (emerging) |

**Strategic implication**: a company that builds one well-operated AI stack for selling can amortize most of the infrastructure into its investment function. The opposite is rare — teams that start with CFO-tooling usually lack the user-facing-serving infrastructure to move into sales.

---

## 5. KPI Summary by Sub-Skill

| Sub-skill | Leading KPI | Lagging KPI | Governance concern |
|---|---|---|---|
| Targeting | Uplift / incremental response | CPA | Fair ML, regulatory targeting rules |
| Recommendation | Recall@k, NDCG | Revenue per session, retention | Filter bubbles, price discrimination |
| Pricing / creative | CTR, CVR | Profit lift | Brand safety, IP provenance |
| Retention | Uplift on treated segment | Net revenue retention | Unsubscribe cost |
| MMM | Posterior mROAS | Total revenue attributed | Privacy (post-cookie) |
| Viral | Expected spread | Attributed revenue | Fairness of spread, disclosure |
| Sales enablement | Rep time saved | Win rate, quota attainment | Data-leak / PII in logs |
| VC screening | Hit rate vs. baseline | Portfolio IRR | Bias, model-washing |
| DD automation | Doc throughput, extraction accuracy | Cycle time to close | Attorney-client privilege |
| IR | Analyst question coverage | Institutional ownership mix | Regulation FD, selective disclosure |
| CFO | Forecast MAPE | Budget adherence | Auditability |

---

## 6. Open Problems — Sales Axis

1. **Joint causal attribution across selling and investment activity**. A Series-B announcement drives both customer trust (sales) and investor confidence (capital). Classical MMM and IR tools treat these as separate; no unified attribution exists.

2. **LLM-assisted narrative vs. measurement drift**. The same LLM that writes the earnings Q&A script can be asked to write the MMM narrative. If both sides of the business run on the same model, narrative drift becomes systemic risk.

3. **Causal evaluation of AI-native DD outputs**. When Harvey extracts and summarizes a data room, what's the actual decision-quality lift vs. attorney-only review? No rigorous RCTs yet.

4. **Founder-facing algorithmic parity**. If VC screening uses XGBoost on parseable founder features, the founder who doesn't produce machine-parseable materials is effectively filtered out regardless of merit. This is a market-access asymmetry without precedent.

5. **IR adversarial robustness**. As both sell-side analysts and CFOs run AI tools on each other's outputs, the feedback loop is unstable — red-team bots train against defense bots train against red-team bots. Steady-state is an open question.

6. **Regulatory boundaries for generative financial communication**. Drafting an earnings Q&A with an LLM intersects Regulation FD, MAR (in Europe), and emerging AI-disclosure rules. The legal substrate has not caught up.

7. **Cross-skill measurement attribution**. If an MMM study argues that the Super Bowl ad lifted brand, and the same ad appears in the S-1 narrative that prices the IPO, which effect is measured where?

---

## 7. Back-References into Existing Deep Dives

| Sub-skill | Primary DD | Supporting DD |
|---|---|---|
| Targeting & segmentation | `docs/01`, DD4 §7 | DD7 §2.4 (Simester) |
| Recommendation | DD1, frontier/llm_enhanced_recommenders | `docs/02` |
| Pricing / creative | DD5, frontier/generative_ai_and_llms_in_marketing §3 | DD7 §3.1 (Perakis) |
| Retention | DD4 | DD2 §4 (BanditLP) |
| MMM | DD5 | DD6 §2.1, §2.2 |
| Viral / influencer | DD2 §2, DD3 | DD7 §2.3 (Aral) |
| Sales enablement | frontier/generative, frontier/agentic | New refs in §2.7 |
| VC screening | — | New refs in §3.1 |
| DD automation | — | New refs in §3.2 |
| IR / CFO | — | New refs in §3.3, §3.5 |
| Pitch / commercialization | DD8 §2.3 | New refs in §3.4 |

---

## 8. Further Reading (Meta-Level)

- `docs/skills/productization_skill.md` — the Productization lens; nearly every tool in §2-3 above lives inside a stack engineered per that document.
- `docs/skills/system_design_and_management.md` — the SDM lens; whenever a Sales decision has architectural, lifecycle, or feedback-loop implications (almost always), the SDM doc is the right companion. Examples: §2 targeting + §6.1 STAMP for customer-impact hazards; §2.5 MMM + §3 system dynamics for budget feedback loops; §3.1 VC screening + §2 architecture for what investors will probe.
- `docs/deep_dive/DD6_ai_marketing_foundations.md` — cross-method theoretical substrate; Bayesian + causal + RL + representation mapping.
- `docs/skills/regulatory_appendix.md` — regulatory obligations that cut across every sub-skill.
- `docs/CATALOG.md` — paper-level index; all new references added via this skill doc are back-listed there.
- `INDEX.md` — topic and domain navigation; each skill sub-section has a direct mapping in the domain axis.
