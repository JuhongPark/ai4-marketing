# Deep Dive 7: MIT Research Landscape — AI for Marketing

A survey of MIT research relevant to AI marketing, organized by lab, lead researcher, and their publications most relevant to this project's scope.

---

## 1. Overview

MIT is one of the deepest concentrations of AI marketing research in the world, spanning:

- **MIT Sloan School of Management — Marketing Group**: Field-experimental and causal-inference work on digital advertising, social contagion, personalization, and direct marketing.
- **MIT Initiative on the Digital Economy (IDE)**: Applied research at the intersection of digital technology, data, and business.
- **MIT Operations Research Center (ORC)**: Optimization + ML for pricing, promotions, assortment, and retail analytics.
- **MIT IDSS (Institute for Data, Systems, and Society)**: Decision systems, causal inference, experimentation.
- **MIT Media Lab**: Social media, social physics, misinformation, public-discourse analysis.
- **MIT CSAIL**: NLP, deep learning, probabilistic models — the AI infrastructure underlying marketing applications.

### MIT Research Map for AI Marketing

| Research Area | MIT Lab / Group | Lead Faculty | Related DD |
|---------------|-----------------|--------------|-------------|
| Digital advertising, privacy, personalization | Sloan Marketing | Catherine Tucker | DD3, DD5 |
| Causal inference in networks, experimentation | Sloan / IDSS / IDE | Dean Eckles | DD3, DD4, DD6 |
| Social contagion, viral marketing, influence | Sloan / IDE | Sinan Aral | DD2, DD3 |
| Direct marketing, ML targeting, field experiments | Sloan | Duncan Simester | DD2, DD4 |
| Pricing, promotion, assortment optimization | ORC / Sloan | Georgia Perakis | DD5 |
| Optimization, classification trees, analytics | ORC / Sloan | Dimitris Bertsimas | DD4, DD6 |
| Social media, misinformation, public discourse | Media Lab / CCC | Deb Roy | DD3 |
| Social Physics, idea flow, behavioral data | Media Lab | Alex Pentland | DD3 |
| NLP, deep learning, causal ML | CSAIL | Regina Barzilay, Tommi Jaakkola | DD1, DD6 |

---

## 2. MIT Sloan Marketing Group

### 2.1 Catherine Tucker — Digital Advertising and Privacy

**Position**: Sloan Distinguished Professor of Management, MIT Sloan
**Affiliations**: NBER (Codirector, Program on Digital Economics & AI)
**Links**: [MIT Sloan Faculty Page](https://mitsloan.mit.edu/faculty/directory/catherine-tucker) | [Personal Page](https://mitmgmtfaculty.mit.edu/cetucker/)

#### Research Focus

Catherine Tucker's research sits at the intersection of **marketing, technology economics, and regulation**. She studies how digital data and ML improve firm performance, and the regulatory challenges that follow — privacy, algorithmic bias, digital advertising effectiveness.

#### Key Research Themes

- **Targeted online advertising effectiveness**: How does personalization affect click-through and conversion?
- **Privacy and advertising**: What happens to advertising when privacy regulation changes?
- **Algorithmic bias in ad delivery**: Do ad-targeting algorithms differentially serve different populations?
- **Digital health marketing**: Application of digital advertising to healthcare contexts

#### Landmark Paper: "Social Networks, Personalized Advertising, and Privacy Controls" (2014)

- **Source**: Journal of Marketing Research, 2014 ([paper](https://journals.sagepub.com/doi/abs/10.1509/jmr.10.0355)) | [DSpace](https://dspace.mit.edu/bitstream/handle/1721.1/99170/Tucker_Social%20networks.pdf)
- **Data**: Randomized field experiment on a social networking website
- **Key finding**: After a privacy policy change giving users **more control** over personal information, users became **twice as likely to click** on personalized ads. Granting users agency over their data increases — not decreases — engagement with targeted advertising.
- **Implication for AI**: Privacy and personalization are not fundamentally opposed; the UX framing matters enormously. AI systems that respect user agency may outperform those that maximize data collection.

#### Policy Voice

Tucker has testified to Congress on digital privacy and algorithms, and presented research to the OECD, World Bank, IMF, and European Court of Justice — making her one of MIT's most policy-influential marketing-AI scholars.

---

### 2.2 Dean Eckles — Causal Inference in Social Networks

**Position**: William F. Pounds Professor of Management, Professor of Marketing, MIT Sloan
**Affiliations**: Associate Director of IDSS (Institute for Data, Systems, and Society); MIT IDE
**Links**: [MIT Sloan Faculty Page](https://mitsloan.mit.edu/faculty/directory/dean-eckles) | [Personal Site](https://www.deaneckles.com/) | [IDE Profile](https://ide.mit.edu/people/dean-eckles/)

#### Research Focus

Eckles' research examines how people interact **with and through** communication technologies, and how these technologies mediate and direct **social influence**. This work forces him to develop new tools for:

- **Causal inference under network interference** — when your treatment affects not only you but your neighbors
- **Design of field experiments** — especially in settings with spillovers
- **Social influence** and contagion in networks

#### Professional Background

Before MIT, Eckles was a scientist at **Facebook and Nokia**. At Facebook, he worked on News Feed, messaging, advertising, tools for randomized experiments, and survey methods — bringing direct industry experience in at-scale ad experimentation.

#### Leadership and Community

Eckles leads the **analytics research area at MIT IDE**, and organizes the annual **Conference on Digital Experimentation (CODE@MIT)** — the central venue for digital experimentation research in North America.

#### Connection to This Project

- **DD3 (Social Influence / Influencer Marketing)**: Eckles' methods for causal inference under network interference are the theoretical foundation for correctly interpreting the AI influence models discussed there.
- **DD4 (Churn and uplift)**: His causal-inference framework underlies uplift modeling and treatment-effect estimation.
- **DD6 (Theoretical Foundations)**: Eckles' work maps directly onto the "causal inference" foundational framework.

---

### 2.3 Sinan Aral — Social Contagion and Viral Marketing

**Position**: David Austin Professor of Management, MIT Sloan
**Affiliations**: Director, MIT Initiative on the Digital Economy (IDE)
**Links**: [MIT Sloan](https://mitsloan.mit.edu/faculty/directory/sinan-aral)

Aral is one of the most-cited scholars on social contagion and viral marketing. His randomized-experiment research is covered in detail in DD3 §6 (see that file for the Management Science, Science, and Nature Communications papers). A brief summary of contributions:

| Year | Venue | Paper | Key Finding |
|------|-------|-------|-------------|
| 2011 | Management Science | Creating Social Contagion Through Viral Product Design | Passive broadcasts generate +246% peer influence; active-personalized adds +98% |
| 2011 | SSRN / working | Identifying Social Influence in Networks Using Randomized Experiments | Methodology to distinguish influence from homophily |
| 2012 | Science | Identifying Influential and Susceptible Members of Social Networks | Younger more susceptible, men more influential, married least susceptible |
| 2013 | Science | Social Influence Bias: A Randomized Experiment | +32% positive rating increase from a single upvote |
| 2017 | Nature Communications | Exercise Contagion in a Global Social Network | Exercise propagates causally through social networks at global scale |
| 2018 | Science | The Spread of True and False News Online | False news spreads farther, faster, deeper than true news — humans, not bots, cause this |

### 2.4 Duncan Simester — Direct Marketing and ML Targeting

**Position**: NTU Chair in Management Science, MIT Sloan
**Links**: [MIT Sloan Faculty](https://mitsloan.mit.edu/faculty/directory/duncan-simester) | [Personal](https://mitmgmtfaculty.mit.edu/dsimester/about/)

#### Research Focus

Simester's research combines **economics + AI + field experiments** to improve marketing and strategy. His studies rely heavily on **industry participation** — large-scale field experiments conducted with cooperating firms, particularly in direct marketing, catalog marketing, and e-commerce targeting.

#### Key Publication: "Targeting Prospective Customers: Robustness of Machine-Learning Methods to Typical Data Challenges" (Management Science, 2020)

- **Source**: Management Science, Vol. 66, No. 6 ([paper](https://pubsonline.informs.org/doi/10.1287/mnsc.2019.3308)) | [DSpace PDF](https://dspace.mit.edu/bitstream/handle/1721.1/130508/Targeting%20Prospects%20submission%203.pdf)
- **What it does**: Evaluates **seven widely used ML methods** for prospecting / new-customer targeting using results from two large-scale field experiments. Tests robustness to data challenges: missing treatment assignment, unbalanced data, confounding, and sample selection.
- **Why it matters**: Simester's robustness analysis is the **empirical benchmark** for practitioners choosing between uplift trees, random forests, gradient boosting, and deep learning for targeting. The paper's conclusions on what works under real-world data challenges are directly actionable.

#### Additional Work

- **Transfer Learning for Targeted Marketing** (Ibragimov, Simester, Timoshenko): Bayesian matrix factorization approach for transfer learning across firms. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5146292)
- Large-scale validation at a **luxury fashion retailer** — demonstrating substantial improvements in predictive accuracy and customer response via ML-based targeting.

#### Connection to This Project

- **DD4 (Churn and uplift)**: Simester's ML-targeting-under-data-challenges work is the empirical foundation for uplift modeling in retail contexts.
- **DD2 (Deep RL and bandits)**: His field experiments generate the kind of logged data needed for off-policy evaluation of bandit and RL policies.

---

## 3. MIT Operations Research Center (ORC)

### 3.1 Georgia Perakis — Pricing, Promotion, and Retail Analytics

**Position**: William F. Pounds Professor of Operations Research & Statistics and Operations Management
**Role**: Codirector of the Operations Research Center (ORC)
**Links**: [ORC](https://orc.mit.edu/faculty_person/georgia-perakis/) | [MIT Sloan](https://mitsloan.mit.edu/faculty/directory/georgia-perakis) | [Personal](https://mitmgmtfaculty.mit.edu/gperakis/)

#### Research Focus

Perakis teaches and researches at the **intersection of optimization and machine learning**, applied to **pricing, revenue management, retail assortment, and promotion optimization**.

#### Retail Analytics Contributions

Perakis' group has produced multiple at-scale retail analytics methods in collaboration with industry:

- **Promotion and markdown optimization** — models analyzing price effects, promotion effects, and consumer behavior to improve retailer profit. A method developed by Perakis and collaborators with **Oracle RGBU** reported **3–10% profit lift** across retailer partners.
- **Price effects + assortment optimization** — integrated optimization frameworks for retailers, deployed via Oracle Retail Assortment & Space Optimization, Oracle Retail Promotion & Markdown Optimization, and Oracle Retail Offer Optimization products.

#### Selected Publication: "Using Business Analytics to Upgrade Sales Promotions" (2021)

- **Source**: Interfaces / INFORMS Journal on Applied Analytics ([paper](https://journals.sagepub.com/doi/10.1177/2694105820210103006))
- **Authors**: Baardman, Cohen, Panchamgam, Perakis
- **Scope**: Case-study-driven treatment of how business analytics and ML improve sales-promotion decisions.

#### Connection to This Project

- **DD5 (MMM and Attribution)**: Perakis' optimization-based retail analytics complement the Bayesian MMM stack — where MMM answers "how much to spend on each channel," Perakis' work answers "how to price and promote once the budget is set."
- **DD4 (Churn and uplift)**: Her optimization-under-uncertainty frameworks transfer directly to retention / CRM optimization.

### 3.2 Dimitris Bertsimas — Optimization and Interpretable ML

**Position**: Boeing Leaders for Global Operations Professor of Management, Sloan Distinguished Professor, MIT Sloan / ORC
**Key contribution**: **Optimal Classification Trees (OCT)** and optimization-based ML methods that produce interpretable models comparable to Random Forest / gradient boosting.
**Relevance**: Marketing applications include interpretable churn modeling, segmentation, and targeting — particularly important for regulated industries.

---

## 4. MIT Initiative on the Digital Economy (IDE)

**Director**: Sinan Aral (Sloan, see §2.3)
**Analytics lead**: Dean Eckles (Sloan, see §2.2)

IDE is MIT's hub for research on the digital economy, spanning:

- **Digital experimentation** — via CODE@MIT, the annual Conference on Digital Experimentation
- **Social contagion and virality** (Aral's research program)
- **AI in business** — ongoing programs on generative AI in marketing, productivity, and decision-making
- **Future of work** — adjacent to marketing via platform economics

### CODE@MIT — Conference on Digital Experimentation

An annual academic + industry conference organized by Eckles. Covers:
- Randomized experimental design under network interference
- Causal inference methods for digital platforms
- Field experiments in advertising, recommendations, marketplaces
- Applied off-policy evaluation

This is the central North American venue for digital-experimentation research — directly relevant to every chapter of this project.

---

## 5. MIT IDSS — Institute for Data, Systems, and Society

**Associate director (causal inference)**: Dean Eckles (also MIT Sloan Marketing)

IDSS is the MIT unit where the **statistical and causal methodology** of AI marketing lives. Eckles' joint appointment between IDSS and Sloan Marketing makes him the direct bridge between rigorous causal-inference theory and applied marketing analytics — exactly the intersection that defines modern AI marketing.

**Scope for AI marketing**:
- Causal inference theory (fits DD6)
- Experimentation under interference in social networks (fits DD3, DD4)
- Decision theory and off-policy evaluation (fits DD2 offline RL)

IDSS is complementary to the Sloan marketing group — one supplies marketing questions, the other supplies causal methodology.

---

## 6. MIT Media Lab

### 6.1 Deb Roy — Social Media, Public Discourse, Misinformation

**Position**: Professor, MIT Media Lab
**Labs**: Laboratory for Social Machines (LSM, 2014–2020, Twitter-funded); Center for Constructive Communication (CCC, current)

#### Notable Work

- **Laboratory for Social Machines (LSM)**: $10M Twitter-funded initiative (2014) with full Twitter data access. Produced large-scale studies of public discourse and misinformation spread.
- **The Spread of True and False News Online** (Vosoughi, Roy & Aral, Science 2018) — see DD3 §6. Landmark study combining Roy's data access with Aral's causal methodology.
- **Center for Constructive Communication (CCC)**: Designing human-AI systems for dialogue, listening, and deliberation. Implications for brand communication, customer-service AI, and public-facing marketing.

#### Why It Matters for AI Marketing

Roy's work on the dynamics of public discourse and misinformation is essential for brand safety, content strategy, and crisis-communication planning.

### 6.2 Alex "Sandy" Pentland — Social Physics

**Position**: Toshiba Professor, MIT Media Lab (emeritus as of recent transitions; still MIT-affiliated)
**Lab**: Human Dynamics Laboratory

#### Social Physics Framework

Pentland studies how **idea flow** propagates through social networks and transforms into behaviors. His book *Social Physics* (Penguin, 2014) codifies the framework.

#### Key Concepts Relevant to Marketing

- **Exploring**: Exposure to novel ideas (via weak ties)
- **Engagement**: Face-to-face social interaction and its role in behavior change
- **Network productivity**: Predicting outcomes from information-exchange patterns **alone**, independent of content

#### Notable Findings

- **eToro investment network**: Members with access to diverse strategies earned **30% higher returns**.
- **Face-to-face interaction**: Increasing face-to-face time produced measurable productivity gains in banks, military units, and IT consulting.

#### Why It Matters for AI Marketing

Pentland's framework offers a social-physics lens on virality that complements the AI cascade models in DD2 and DD3 — specifically, his emphasis on **weak ties as idea-propagation paths** informs how viral-marketing seed selection should treat network structure.

---

## 7. MIT CSAIL — Computer Science and Artificial Intelligence Laboratory

### 7.1 Regina Barzilay — NLP and Deep Learning

**Position**: School of Engineering Distinguished Professor, EECS / CSAIL
**Relevance to AI marketing**: NLP methods for content, caption, and review analysis; deep learning architectures.
**Applications**: Product description generation, review mining for customer insight, automated content moderation.

### 7.2 Tommi Jaakkola — Probabilistic Models and Causal Inference

**Position**: Professor, EECS / CSAIL
**Relevance to AI marketing**: Theoretical foundations of message passing, graphical models, causal inference. His work on **message passing limitations** is directly relevant to understanding the reach of GNN models covered in DD1.
**Cross-reference**: See DD6 §2.2 for the causal inference framework Jaakkola's methods support.

---

## 8. Research Output Mapping to This Project

| Deep Dive | Most Directly Relevant MIT Work |
|-----------|----------------------------------|
| **DD1** (GNN Recommenders) | CSAIL foundations (Jaakkola, Barzilay); theoretical grounding for propagation |
| **DD2** (Deep RL Marketing) | Aral randomized experiments (validation); IDE CODE@MIT venue; Simester field experiments (logged data for OPE) |
| **DD3** (Influencer Marketing) | Aral (viral marketing, contagion); Eckles (causal inference in networks); Tucker (privacy + targeting); Roy/Pentland (social media dynamics) |
| **DD4** (Churn and Retention) | Simester (ML targeting under data challenges); Eckles (causal inference / uplift); Perakis (optimization under uncertainty) |
| **DD5** (MMM and Attribution) | Tucker (advertising effectiveness); Perakis (pricing and promotion optimization); Eckles (digital experimentation for calibration) |
| **DD6** (Theoretical Foundations) | Bertsimas (optimization-based ML); Jaakkola (probabilistic models, causal methods); Eckles (causal inference theory) |

---

## 9. Distinctive Strengths of MIT's AI Marketing Research

| Strength | Description |
|----------|-------------|
| **Theory-experiments integration** | Causal-inference theory (Eckles, Tucker) and at-scale field experiments (Aral, Simester) conducted at the same institution |
| **Optimization + ML integration** | ORC (Perakis, Bertsimas) brings classical optimization expertise to ML marketing applications — a rare combination |
| **Industry partnerships** | Simester (retail), Perakis (Oracle RGBU), Tucker (ad platforms), Aral (Facebook, other platforms) — MIT work is grounded in real business data |
| **Policy voice** | Tucker testifies to Congress; MIT work shapes digital-privacy and antitrust policy, which in turn shapes the future of AI marketing |
| **Venue building** | CODE@MIT (digital experimentation), NBER Digital Economics + AI program (Tucker), MIT IDE — MIT hosts the central venues for the field |

### Implications for This Project

1. **Causal inference is the MIT methodological signature** — any AI marketing system that ignores causal identification is missing MIT's most important contribution.
2. **Field experiments are essential** — MIT's empirical tradition argues against relying on observational ML alone.
3. **Industry grounding matters** — MIT's at-scale industry collaborations provide the data scale that makes research reproducible.
4. **Policy and ethics are first-class** — Tucker's privacy + ad research, and Aral's work on misinformation, demand that responsible AI marketing consider downstream social effects.

---

## 10. Key References

### MIT Sloan Marketing
- Catherine Tucker. [MIT Sloan Profile](https://mitsloan.mit.edu/faculty/directory/catherine-tucker) | [Personal](https://mitmgmtfaculty.mit.edu/cetucker/)
- Tucker (2014). Social Networks, Personalized Advertising, and Privacy Controls. JMR. [Paper](https://journals.sagepub.com/doi/abs/10.1509/jmr.10.0355) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1694319)
- Dean Eckles. [MIT Sloan Profile](https://mitsloan.mit.edu/faculty/directory/dean-eckles) | [Personal](https://www.deaneckles.com/) | [IDE](https://ide.mit.edu/people/dean-eckles/)
- Sinan Aral. [MIT Sloan Profile](https://mitsloan.mit.edu/faculty/directory/sinan-aral) (full publication list in DD3 §6)
- Duncan Simester. [MIT Sloan Profile](https://mitsloan.mit.edu/faculty/directory/duncan-simester) | [Personal](https://mitmgmtfaculty.mit.edu/dsimester/about/)
- Simester et al. (2020). Targeting Prospective Customers. Management Science. [Paper](https://pubsonline.informs.org/doi/10.1287/mnsc.2019.3308)
- Ibragimov, Simester, Timoshenko. Transfer Learning for Targeted Marketing. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5146292)

### MIT ORC
- Georgia Perakis. [ORC Profile](https://orc.mit.edu/faculty_person/georgia-perakis/) | [MIT Sloan](https://mitsloan.mit.edu/faculty/directory/georgia-perakis)
- Baardman, Cohen, Panchamgam, Perakis (2021). Using Business Analytics to Upgrade Sales Promotions. [Paper](https://journals.sagepub.com/doi/10.1177/2694105820210103006)

### MIT IDE and CODE@MIT
- [MIT IDE Website](https://ide.mit.edu/)
- [CODE@MIT — Conference on Digital Experimentation](https://ide.mit.edu/events/2023-conference-on-digital-experimentation-mit-codemit/)

### MIT Media Lab
- Deb Roy. [MIT Media Lab](https://www.media.mit.edu/people/dkroy/)
- Alex Pentland. *Social Physics: How Good Ideas Spread*. Penguin, 2014.
- Vosoughi, Roy, Aral (2018). The Spread of True and False News Online. Science. [Paper](https://www.science.org/doi/10.1126/science.aap9559)

### MIT CSAIL / EECS (cross-reference to DD6)
- Regina Barzilay. [CSAIL Profile](https://www.csail.mit.edu/person/regina-barzilay)
- Tommi Jaakkola. [CSAIL Profile](https://www.csail.mit.edu/person/tommi-jaakkola)
