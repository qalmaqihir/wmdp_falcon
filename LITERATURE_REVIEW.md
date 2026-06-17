# Literature Review: LLM Safety Evals, Benchmarks & Unlearning
**Project:** Falcon3 Biosecurity Capability Evaluation  
**Date:** 2026-05-22  
**Scope:** Biosecurity benchmarks · Cybersecurity benchmarks · Machine unlearning · Representation learning

---

## Table of Contents
1. [Biosecurity Benchmarks & Datasets](#1-biosecurity-benchmarks--datasets)
2. [Cybersecurity Benchmarks & Datasets](#2-cybersecurity-benchmarks--datasets)
3. [Machine Unlearning Methods](#3-machine-unlearning-methods)
4. [Representation Learning & Mechanistic Interpretability](#4-representation-learning--mechanistic-interpretability)
5. [Synthesis: Priority Papers for Falcon3 Work](#5-synthesis-priority-papers-for-falcon3-work)
6. [Conceptual Notes](#6-conceptual-notes)
7. [Full Citation Index](#7-full-citation-index)

---

## 1. Biosecurity Benchmarks & Datasets

### 1.1 WMDP — The Core Benchmark
**Li et al. (2024). "The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning." ICML 2024 (PMLR Vol. 235)**  
arXiv: https://arxiv.org/abs/2403.03218

**What it is:** 3,668 multiple-choice questions (MCQ) proxy-measuring hazardous knowledge in three domains:
- `wmdp-bio`: 1,273 questions (biosecurity)
- `wmdp-chem`: 585 questions (chemical weapons)
- `wmdp-cyber`: 1,810 questions (offensive cyber)

**Key design choices:**
- Questions are *proxies* — they measure knowledge correlating with hazardous capability without containing actual dangerous information
- Expert-filtered to exclude export-controlled or directly actionable content
- Hosted on HuggingFace: `cais/wmdp` — available in Inspect Evals as `wmdp_bio`, `wmdp_chem`, `wmdp_cyber`

**Published results on WMDP-bio:**
| Model | WMDP-bio | MMLU |
|-------|----------|------|
| GPT-4 | ~74% | ~86% |
| Llama 2 70B | ~46% | ~68% |
| Falcon 40B | ~41% | ~55% |
| Random baseline | 25% | 25% |

**Also introduces RMU** (see Section 3.1) — reduces Llama 2 WMDP-bio from 46% → ~25% while preserving MMLU.

**Limitations:**
- Proxy measurement: high score ≠ confirmed dangerous capability
- MCQ format: doesn't test open-ended synthesis or agentic task completion
- No Falcon3 results exist — this is the research gap we fill

**Why it matters for our work:** Primary benchmark. Everything we run produces data that didn't exist before.

---

### 1.2 LAB-Bench — Biology Research Capabilities
**FutureHouse (2024). "LAB-Bench: Measuring Capabilities of Language Models for Biology Research." arXiv:2407.10362**

**What it is:** 2,400+ MCQ questions evaluating practical biology research skills:
- `LitQA2`: Literature recall and synthesis
- `ProtocolQA`: Protocol planning and troubleshooting
- `FigQA`: Figure interpretation (images — skip for text-only models)
- `DbQA`: Database navigation (BLAST, UniProt, etc.)
- `SeqQA`: DNA/protein sequence manipulation

**Key finding:** Frontier LLMs beat human researchers on LitQA2 (literature recall) but lag on complex multi-step reasoning. Gap between benchmark performance and real-world utility.

**Public subset:** `futurehouse/lab-bench` on HuggingFace. Works with Inspect AI.

**Limitations:**
- Text-only tasks (FigQA requires vision)
- Public subset easier than full version (researcher calibration not public)
- Doesn't test actual wetlab capability

**Why it matters:** Broader biology capability measure beyond WMDP's hazard-specific focus. Day 3 target for Falcon3.

---

### 1.3 VCT — Virology Capabilities Test
**Anonymous (2025). "Virology Capabilities Test (VCT): A Multimodal Virology Q&A Benchmark." arXiv:2504.16137**

**What it is:** 322 multimodal questions (text + images) at PhD level:
- Fundamental virology (can memorize)
- Tacit knowledge (lab intuition — hard to memorize)
- Visual interpretation (microscopy, gel results)

**Alarming findings:**
| Model | Score | Beats what % of expert virologists |
|-------|-------|-------------------------------------|
| OpenAI o3 | 43.8% | 94% |
| GPT-4o | ~38% | 53% |
| Gemini 1.5 Pro | ~35% | 67% |
| Average PhD virologist | ~22% | — |

**Implication:** Frontier models *already outperform most expert virologists* on virology knowledge. This is not a future risk — it's present.

**Limitations:** Multimodal (can't fully test with text-only Falcon3). The text-only subset is more accessible.

**Why it matters:** Establishes the upper bound of existing biosecurity risk. Falcon3 7B will score much lower — showing where open-weight models sit on the threat spectrum.

---

### 1.4 ABC-Bench — Agentic Biosecurity
**Anonymous (2025). "ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity." NeurIPS 2025 Workshop**

**What it is:** End-to-end agentic tasks, not just knowledge MCQ:
- Liquid handling robot programming
- DNA fragment design for synthesis
- Synthesis screening evasion strategies

**Key result:** Grok 3 achieves 53% (vs. 24% for PhD biologists on non-specialized tasks). GPT-4o-mini-high code ran *successfully on a physical robot* — physical world validation.

**Why this changes the picture:** WMDP measures knowledge. ABC-Bench measures the *ability to act* on that knowledge. A model scoring 35% on WMDP but 0% on ABC-Bench is much less dangerous than one that translates knowledge to action.

**Why it matters:** Represents future evaluation direction. For now, Falcon3 evaluation starts with WMDP but ABC-Bench shows where the field is heading.

---

### 1.5 BioLP-Bench — Protocol Understanding
**Anonymous (2024). "BioLP-bench: Measuring understanding of biological lab protocols by large language models." bioRxiv:2024.08.21.608694**

**What it is:** Models receive a lab protocol containing one critical error and several non-critical errors. Task: identify which error causes experiment failure.

**Key finding:** State-of-the-art LLMs perform poorly compared to human experts. Most models fail to identify critical mistakes. This suggests depth of understanding is shallower than WMDP scores imply.

**Uses Inspect AI** — directly compatible with our setup. GitHub: `baceolus/BioLP-bench`

**Why it matters:** Failure-detection capability. A model that understands protocols deeply enough to know *what goes wrong* is more dangerous than one that just knows facts.

---

### 1.6 Novice Uplift Papers — Quantifying Real-World Risk
**Two key papers (2025):**

**(a) "LLM Novice Uplift on Dual-Use, In Silico Biology Tasks." arXiv:2602.23329**  
**(b) "Measuring Mid-2025 LLM-Assistance on Novice Performance in Biology." arXiv:2602.16703**

**Core finding:** LLMs raise novice performance to *expert baseline* on:
- Virology troubleshooting ("Magnification" stage of bioweapon development)
- Molecular cloning workflow design ("Formulation" stage)
- Not just abstract knowledge — actual bottleneck-removal in dual-use pipelines

**Trajectory:** Top model performance 4x+ improvement from 2023→2025. Several models now match expert performance on LAB-Bench biology and GPQA subsets.

**Why this matters for Falcon3:** We're measuring a moving target. 35% on WMDP-bio from Falcon3-7B needs context — what does that level of capability enable for a determined novice with LLM assistance?

---

### 1.7 HarmBench — Refusal Evaluation with Biosecurity Subset
**Mazeika et al. (2024). "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." arXiv:2402.04249**

**What it is:** 400 harmful behaviors × 18 red-teaming methods × 33 models/defenses. Includes explicit chemical/biological threat categories.

**Key finding:** No uniformly effective attack or defense. Robustness independent of model size. Large models not automatically safer.

**Why it matters:** Complements WMDP (knowledge-based) with refusal-based evaluation. Falcon3 might score 35% on WMDP but refuse 95% of HarmBench requests — important distinction between *has knowledge* vs. *will apply it*.

---

## 2. Cybersecurity Benchmarks & Datasets

*Note: Cyber domain is parallel to bio domain in the WMDP framework. These benchmarks show how the field measures offensive capability in LLMs — methodology transfers directly.*

---

### 2.1 CyberSecEval Series (Meta)
**Bhatt et al. — Three iterations 2023–2024**

**(a) Purple Llama CyberSecEval (2023).** arXiv:2312.04724 — Original benchmark: vulnerable code generation, exploit suggestion.

**(b) CyberSecEval 2 (2024).** arXiv:2404.13161 — Adds: automatic exploit generation, insecure code outputs, prompt injection susceptibility.

**(c) CyberSecEval 3 (2024).** arXiv:2408.01605 — Adds: automated social engineering, autonomous cyber operations, scaling manual offensive ops. Evaluates 8 risk categories.

**Design pattern:** Each version adds more *agentic* capability evaluation, moving from knowledge → code → autonomous operation. Mirrors the bio domain trajectory (WMDP → ABC-Bench).

**Why it matters for bio evals:** Methodology template. Cyber domain moved faster; bio evaluations are following the same capability escalation path.

---

### 2.2 NYU CTF Bench
**Topala et al. (2024). "NYU CTF Bench." arXiv:2406.05590. NeurIPS 2024.**

**What it is:** Open-source CTF benchmark from picoCTF (high-school/undergrad level). Multi-step reasoning + tool use + vulnerability exploitation. 5 LLMs evaluated.

**GitHub:** `NYU-LLM-CTF` — open and usable.

**Why it matters:** Shows LLMs can reason through multi-step offensive security tasks. Not just MCQ — actual task completion.

---

### 2.3 InterCode-CTF
**Palisade Research (2024). "Hacking CTFs with Plain Agents." arXiv:2412.02776**

**What it is:** 91 Docker-based CTF challenges across crypto, forensics, reverse engineering. Modeled as POMDP. Recent work: 95% success using ReAct prompting.

**Alarming result:** 95% success on high-school difficulty CTF. Current LLMs *already exceed baseline offensive security capability at this level.*

**Why it matters:** Direct parallel to VCT result in bio domain — LLMs already competent at accessible-level offensive tasks.

---

### 2.4 SecEval / SECURE
**Anonymous (2024). "SECURE: Benchmarking Large Language Models for Cybersecurity Advisory." arXiv:2405.20441**

**What it is:** ~2,100 MCQ across 8 security domains (Software, Application, System, Web, Crypto, Memory Safety, Network, Pentesting). Generated via 6-stage GPT-4 pipeline from OWASP/MITRE/CWE sources.

**Why it matters:** Knowledge-level baseline parallel to WMDP. MCQ format directly comparable. Shows how cybersecurity knowledge is distributed across model families.

---

### 2.5 Key Pattern Across Cyber Benchmarks
The cyber domain shows a clear progression:
1. **Knowledge MCQ** (WMDP-cyber, SECURE) → measures: does model know offensive concepts?
2. **Code generation** (CyberSecEval 1-2) → measures: can model write exploits?
3. **Agentic task completion** (NYU CTF, InterCode) → measures: can model autonomously hack?
4. **Real-world impact** (CyberSecEval 3) → measures: can model scale operations?

Bio domain is at stage 1-2. ABC-Bench pushes toward stage 3. Expect rapid progression.

---

## 3. Machine Unlearning Methods

**The core problem:** A model that scores 35% on WMDP-bio has absorbed hazardous knowledge during pretraining. Safety fine-tuning adds a *refusal layer* — but knowledge persists. Unlearning attempts to remove the knowledge itself.

**Key distinction: unlearning ≠ refusal training**
- **Refusal training**: Model learns to decline dangerous requests → behavior-level → circumventable
- **Unlearning**: Model's underlying knowledge/capability is degraded → knowledge-level → more robust (if complete)

---

### 3.1 RMU — Representation Misdirection for Unlearning
**Li et al. (2024). Introduced in the WMDP paper. arXiv:2403.03218**

**The method:** Two-component loss function:
1. **Forget loss**: On hazardous data — maximize activation norm AND redirect hidden states toward a random vector (misdirect, don't just suppress)
2. **Retain loss**: On benign data — preserve activations close to original model

```
L_total = L_forget + α * L_retain
L_forget = -||h_forget - r||²   # push activations toward random vector r
L_retain = ||h_retain - h_orig||²  # keep benign activations stable
```

**Results on Llama 2-7B:**
| Metric | Before RMU | After RMU |
|--------|-----------|-----------|
| WMDP-bio | 46% | ~25% (random) |
| MMLU | 46% | 45% (preserved) |
| MT-Bench | 6.2 | 6.1 (preserved) |

**How it relates to RepE:** RMU is inspired by Representation Engineering (Zou et al.). Instead of zeroing out a direction (which is unstable), it misdirects — flooding with noise, making representations less usable.

**Limitations:**
- Knowledge not truly removed — activations redirected/obfuscated
- Sophisticated extraction attacks (fine-tuning on few hazardous examples) may recover knowledge
- Shallow intervention: operates on specific layers, may miss distributed encoding

**Why it matters for Falcon3:** This is the method to reproduce. Run WMDP-bio → apply RMU → re-run WMDP-bio. Show the delta.

---

### 3.2 Gradient Ascent (GA) — Naive Baseline
**Formalized in multiple papers; recent analysis: arXiv:2502.19301**

**The method:** Reverse the gradient on forget set examples — increase loss on hazardous content (model less likely to produce correct answers).

```
L_GA = -L_CE(model(x_forget), y_forget)  # ascend loss
```

Often paired with retain loss: `L_total = L_GA + λ * L_CE(model(x_retain), y_retain)`

**Problems:**
- Catastrophic collapse: unbounded gradient growth
- Instability: difficult hyperparameter tuning (λ)
- Over-forgetting: damages benign capabilities
- Cross-entropy loss structurally unsuited for ascent

**Why it matters:** Important baseline — all other methods compare against GA. Understanding *why* GA fails informs why better methods work.

---

### 3.3 Negative Preference Optimization (NPO)
**Anonymous (2024). "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning." arXiv:2404.05868**

**The method:** Adapts DPO (Direct Preference Optimization) framework. Uses *only negative* samples (no positive counterpart needed). Bounds the forgetting loss exponentially — slows progression toward catastrophic collapse.

**Key result:** First method to effectively unlearn 50%+ of training data. Surpasses GA, KL divergence, and earlier methods on large forget sets.

**Limitations:** Reference model bias (subsequent work identifies this). May not achieve complete removal. Some later papers suggest original NPO was overcomplicated.

**Why it matters:** Represents principled improvement over GA. More stable. Relevant if we want to evaluate unlearning at scale on Falcon3.

---

### 3.4 TOFU — Evaluation Framework for Unlearning
**Touvron et al. (2024). "TOFU: A Task of Fictitious Unlearning for LLMs." arXiv:2401.06121**

**What it is:** Benchmark with 200 *fictitious* authors × 20 QA pairs each. Because data is synthetic (not in pretraining), unlearning is ground-truth verifiable. Tests at 1%, 5%, 10% forget set sizes.

**Why fictitious?** Real unlearning benchmarks suffer from: (a) uncertainty about what's in pretraining, (b) inability to verify removal. Fictional data solves both.

**Metrics established:** Forget quality (did we remove it?), retain quality (did benign capabilities survive?), model utility (overall degradation?).

**Key finding:** Most baseline methods show poor unlearning — strong motivation for better approaches.

**Why it matters:** If we apply RMU to Falcon3, TOFU-style evaluation tells us if it *actually worked* vs. just changed output behavior.

---

### 3.5 "Who's Harry Potter?" — Pioneer Unlearning
**Eldan & Russinovich (2023). "Who's Harry Potter? Approximate Unlearning in LLMs." arXiv:2310.02238**

**The method:** Three-stage:
1. Use reinforced model to identify Harry Potter-specific tokens
2. Replace idiosyncratic expressions with generic counterparts in training data
3. Fine-tune on the "genericized" version

**Result:** Erased Llama 2-7B's Harry Potter knowledge in ~1 GPU hour. MMLU preserved.

**Limitations:** Designed for copyrighted content (structured narrative) — may not transfer to distributed scientific knowledge. Manual token identification required. Unclear if truly removed vs. output-suppressed.

**Why it matters:** Proof of concept for efficient domain-specific unlearning. First widely known demonstration that LLM unlearning is computationally feasible.

---

### 3.6 MEMIT & ROME — Knowledge Editing as Unlearning
**Meng et al. (2022, 2023). ROME: arXiv:2202.05629. MEMIT: arXiv:2210.07229**

**What they are:** Methods for *editing* factual associations in pretrained models.
- **ROME**: Uses causal tracing to locate where a fact is stored → edits that specific MLP
- **MEMIT**: Batch extension of ROME — edit thousands of facts at once

**As unlearning:** Negate the edit (set target to empty/wrong answer). Recent work (2025) shows MEMIT competitive with dedicated unlearning methods when using *query merging* — combining edits efficiently.

**Why it matters:** Shows knowledge is *localizable* in specific layers and positions. If hazardous knowledge localizes similarly to factual knowledge (it does, partially — see ROME causal tracing), targeted removal is feasible.

---

### 3.7 Task Vectors — Parameter-Space Unlearning
**Ilharco et al. (2023). "Editing Models with Task Arithmetic." arXiv:2212.04089**  
**Extension: "Per-parameter Task Arithmetic for Unlearning." arXiv:2601.22030 (2025)**

**The method:**
1. Fine-tune on forget set → get fine-tuned model
2. Task vector = (fine-tuned weights) - (original weights)
3. Subtract task vector from original: `θ_unlearned = θ_orig - α * τ`

**Simple, efficient, composable.** Works as a baseline. Per-parameter variant adds granular control over which parameters adjust.

**Limitations:** Over-forgetting (removes benign uses of same parameters). No principled parameter selection without per-parameter variant.

**Why it matters:** Efficient unlearning baseline. Good sanity check: if task vector subtraction reduces WMDP while preserving MMLU, that's evidence hazardous knowledge is somewhat separable.

---

### 3.8 SISA Training — Architectural Approach
**Bourtoule et al. (2021). "Machine Unlearning." IEEE S&P. arXiv:1912.03817**

**The method:** Pre-training design for efficient unlearning:
- Shard data into S disjoint subsets
- Train isolated sub-models per shard
- Slice each shard into R slices with checkpoints
- When unlearning: retrain only affected shard from last safe checkpoint

**Why it matters (even though impractical for Falcon3):** Shows that *training-time decisions* can enable efficient post-hoc unlearning. Motivates future foundation model design. Falcon3 can't use SISA retroactively, but future safety-designed models could.

---

### 3.9 2025 Unlearning Survey
**Geng et al. (2025). "A Comprehensive Survey of Machine Unlearning in Large Language Models." arXiv:2503.01854**

**Taxonomy:**
- **Training-time**: SISA, data curation (prevent learning)
- **Post-training**: Gradient ascent, GA+KL, NPO, RMU, task vectors, knowledge editing
- **Inference-time**: Prompt engineering, activation steering, output filtering

**Key insight from survey:** "Unlearning" in LLMs is largely *approximate* — current methods degrade performance on hazardous tasks without guaranteeing knowledge removal. True unlearning remains an open problem.

---

## 4. Representation Learning & Mechanistic Interpretability

*These methods study the internal structure of LLMs — where knowledge is encoded, how it's organized, and how to manipulate it. Directly relevant to understanding why RMU works and how to build better unlearning.*

---

### 4.1 Representation Engineering (RepE)
**Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405**

**Core idea:** High-level concepts (truthfulness, honesty, safety, harm) are encoded as *directions* in the residual stream. Reading or writing these directions enables concept-level control without model modification.

**Method — Linear Artificial Tomography (LAT):**
1. Create paired inputs: `(honest statement, deceptive equivalent)`
2. Extract difference in hidden states: `d = h_honest - h_deceptive`
3. Average across many pairs → direction vector for "honesty"
4. Add/subtract vector at inference time → steer behavior

**Applications:** Truthfulness, honesty, jailbreak resistance, hallucination reduction. Demonstrated on Llama 2.

**Limitations:** Requires paired examples. Concept directions layer-dependent (no universal layer). May not generalize across domains (a "safety" vector learned from one distribution may not transfer).

**Why it matters:** Foundation for RMU. Validates that representation-level intervention is feasible for high-level concepts. If "biosecurity knowledge" has a direction, we can misdirect it (= RMU).

---

### 4.2 Linear Representation Hypothesis
**Foundational claims + recent extensions. Key: arXiv:2502.09674 "The Hidden Dimensions of LLM Alignment" (2025)**

**The hypothesis:** High-level concepts (gender, language, sentiment, safety, truth) are *linearly* encoded in activation space as directions. Behavior follows: steer the direction, steer the behavior.

**Empirically validated on:**
- Political ideology, sentiment, humor (original RepE paper)
- Safety and harmfulness (multiple papers)
- Multi-dimensional safety analysis (dominant + non-dominant safety features)

**Safety vulnerability:** Non-dominant safety features can be manipulated. Trigger-token removal can reduce refusal rate. Linear safety assumptions create exploitable structure.

**Why it matters:** If hazardous knowledge is linearly encoded, targeted removal is theoretically achievable. LEACE (Section 4.4) gives the optimal linear removal. But if encoding is nonlinear (superposition), linear methods are incomplete — motivating RMU's misdirection approach.

---

### 4.3 Sparse Autoencoders (SAEs) for Mechanistic Interpretability
**Bricken et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic Transformer Circuits.**  
**Cunningham et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600**

**The problem:** Neurons are *polysemantic* — each neuron fires on multiple unrelated concepts (superposition). Individual neurons aren't interpretable.

**SAE solution:** Train autoencoder with L1 sparsity penalty to reconstruct hidden states from a larger sparse basis. Learned features tend to be *monosemantic* — one concept per feature.

**Anthropic result:** Extracted ~100k interpretable features from Claude 3 Sonnet. Demonstrated "Golden Gate Bridge" feature — activating it made Claude believe it *was* the Golden Gate Bridge. Feature steering works.

**Safety warning (2025 work):**  
- "Use Sparse Autoencoders to Discover Unknown Concepts" (arXiv:2506.23845): warn against acting on discovered features without validation
- "When the Coffee Feature Activates on Coffins" (arXiv:2601.03047): features may label-match without representing concepts; steering may not mean what you think

**Why it matters for Falcon3:** SAEs could identify *which specific features* encode biosecurity knowledge. If those features can be removed (zero the feature vector), targeted unlearning becomes possible. But validation required — feature ≠ concept.

---

### 4.4 LEACE — Optimal Linear Concept Erasure
**Belrose et al. (2023). "LEACE: Perfect linear concept erasure in closed form." NeurIPS 2023. arXiv:2306.03819**

**What it is:** Closed-form solution for erasing a concept from representations while minimizing representational damage. Uses least-squares projection. After LEACE, no linear classifier can detect the concept from the representation.

**Applications:** Bias removal (gender, race), fairness. Proven optimal for the linear case.

**Limitations:** Only removes the *linear component* of concept encoding. Nonlinear representations persist. Concept may survive in non-dominant directions (see Section 4.2 safety vulnerabilities).

**Why it matters:** Establishes theoretical limit of linear erasure. Shows that hazardous knowledge with nonlinear encoding cannot be fully removed by projection methods → motivates RMU's nonlinear misdirection.

---

### 4.5 Causal Tracing & ROME — Knowledge Localization
**Meng et al. (2022). "Locating and Editing Factual Associations in GPT." arXiv:2202.05629 (ROME). NeurIPS 2022.**

**Causal tracing method:**
1. Run model on clean input → cache all activations
2. Corrupt input with Gaussian noise → observe degraded output
3. Restore activations layer-by-layer from cache → find which restoration recovers output

**Key finding:** Factual associations concentrated in **mid-layer MLPs** at the *subject token position*. Restoring a single layer's MLP activations sufficient to recover factual recall.

**Implications:**
- Knowledge is localizable (not distributed uniformly)
- MLP layers act as "key-value memory"
- Layer-specific edits (ROME/MEMIT) work because knowledge is concentrated

**Limitation (important):** Recent work suggests localization may be an artifact of the corruption methodology — not necessarily how the model naturally processes information. Treat as a useful tool, not ground truth.

**Why it matters:** If biosecurity knowledge localizes similarly to factual knowledge, RMU-style layer-specific intervention is well-targeted. If not, broader intervention needed.

---

### 4.6 Activation Steering & Safety Vulnerabilities
**"The Rogue Scalpel: Activation Steering Compromises LLM Safety." arXiv:2509.22067 (2025)**  
**"Analysing the Safety Pitfalls of Steering Vectors." arXiv:2603.24543 (2025)**  
**"SafeSteer: Interpretable Safety Steering with Refusal-Evasion in LLMs." arXiv:2506.04250**

**Finding:** Adding vectors to residual stream at inference time breaks safety alignment:
- Random vector addition: increases harmful compliance 0–27%
- Combining 20 random vectors: creates universal jailbreak
- SAE feature steering on *benign* features: comparable harmful effect

**Quantitative impact:**
- Up to 57% increase in attack success rate
- Up to 50% decrease in refusal rate

**Why this matters for unlearning research:** If safety alignment is this mechanistically fragile, refusal-based safety (behavior modification) is insufficient. Knowledge-level unlearning is necessary for robust safety — but even RMU can potentially be reversed by fine-tuning. Comprehensive safety requires both.

---

### 4.7 Refusal Direction Mechanistics
**"Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction." arXiv:2509.15202 (2025)**  
**"Robust LLM Safeguarding via Refusal Feature Adversarial Training." arXiv:2409.20089**

**Finding:** Refusal is encoded as a specific *direction* in the residual stream — identifiable and ablatable. Adversaries can ablate this direction to jailbreak models. Defense: **DeepRefusal** probabilistically ablates refusal direction across layers *during training*, making the direction less consistent → harder to attack.

**Key insight:** Refusal and knowledge are mechanistically distinct. A model can be robustly refusing but still have the knowledge. A model can have the knowledge removed but still refuse — because they're separate mechanisms.

**Why it matters:** Clarifies what RMU achieves: it degrades the *knowledge*, not just the *refusal behavior*. That's stronger, but still not complete if fine-tuning on few examples recovers knowledge.

---

### 4.8 Probing Classifiers — Promises and Limits
**"The Geometry of Harmfulness in LLMs through Subconcept Probing." arXiv:2507.21141**  
**"False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize." arXiv:2509.03888**

**Probing classifiers:** Train lightweight linear classifiers on hidden states to detect whether model "knows" a concept. Useful for:
- Identifying which layers contain hazardous knowledge
- Measuring unlearning effectiveness at representation level

**Critical limitation:** Probing-based safety detection shows poor out-of-distribution generalization. In-domain performance looks good; OOD fails. Cannot reliably use probes as safety detectors.

**Why it matters:** Probes useful for *research* (localization, analysis) but not for *deployment* (safety verification). WMDP accuracy remains the right downstream metric.

---

## 5. Synthesis: Priority Papers for Falcon3 Work

### Tier 1 — Must Read (Direct Application)
| Paper | Why |
|-------|-----|
| WMDP (Li et al., 2024) | Primary benchmark; defines the task |
| RMU (same paper) | Primary unlearning method to reproduce |
| LAB-Bench (FutureHouse, 2024) | Day 3 evaluation target |
| VCT (2025) | Establishes frontier threat ceiling |
| Novice Uplift papers (2025) | Real-world risk quantification |
| NPO (2024) | Unlearning comparison baseline |
| TOFU (2024) | How to evaluate unlearning rigorously |

### Tier 2 — Important Context
| Paper | Why |
|-------|-----|
| ABC-Bench (2025) | Future eval direction; agentic capabilities |
| HarmBench (2024) | Refusal vs. knowledge distinction |
| RepE (Zou et al., 2023) | Theoretical foundation for RMU |
| Causal Tracing/ROME (2022) | Knowledge localization theory |
| SAEs (Anthropic, 2024) | Feature-level understanding |
| Linear Rep. Hypothesis (2025) | Why representation manipulation works |
| Who's Harry Potter? (2023) | Unlearning precedent, efficiency |
| Activation Steering Safety (2025) | Why refusal-only is insufficient |

### Tier 3 — Background
| Paper | Why |
|-------|-----|
| CyberSecEval 1/2/3 (2023-24) | Parallel cyber domain methodology |
| NYU CTF Bench / InterCode (2024) | Agentic offensive capability pattern |
| SISA (2021) | Training-time unlearning foundation |
| LLMU (2024) | Alternative unlearning framework |
| LEACE (2023) | Theoretical limits of linear erasure |
| Task Vectors (2023) | Simple unlearning baseline |
| Gradient Ascent analysis (2025) | Why GA fails; what to avoid |
| Refusal Direction (2025) | Mechanistic safety |
| Unlearning Survey (2025) | Field taxonomy |

---

## 6. Conceptual Notes

### On What WMDP Actually Measures
WMDP is a *proxy*. High WMDP score means: "this model likely contains sufficient biology knowledge to provide meaningful uplift to someone attempting dual-use biology research." It does NOT mean: "this model will help someone make a bioweapon." That distinction is critical for:
- Writing up results responsibly
- Interpreting what 35% vs. 74% means in practice
- Contextualizing with the VCT/novice uplift literature

### On the Knowledge vs. Refusal Distinction
Running Falcon3 through WMDP tells us about *latent knowledge*. WMDP uses direct MCQ — not asking for harmful instructions. A model that scores 35% knows enough to answer 35% of proxy-hazardous biology questions correctly. Whether it would *apply* that knowledge if prompted differently is a separate question (HarmBench territory).

### On What "Unlearning" Can Achieve (and Can't)
Current best methods (RMU, NPO) degrade performance to random-chance on hazardous benchmarks while preserving general capability. But:
1. Fine-tuning on a small number of hazardous examples can partially recover knowledge
2. The knowledge is redirected/obfuscated, not truly deleted
3. True unlearning (verifiable removal with zero recovery path) remains unsolved

For our purposes: unlearning is a meaningful harm reduction measure, not a security guarantee.

### On Architecture and Knowledge Storage
- ROME/causal tracing: factual knowledge concentrated in mid-layer MLPs
- SAEs: concepts distributed as sparse features across layers
- RepE: high-level concepts encoded as directions in residual stream

These are complementary views, not contradictory. Different levels of abstraction. Hazardous knowledge likely has:
- Factual component (localized, editable via ROME/MEMIT)
- Reasoning component (distributed, harder to target)
- Tacit/procedural component (may not be linearly separable)

RMU targets the representation-level encoding and shows empirical degradation. Whether it removes all three components is unknown.

### On Falcon-H1 vs. Falcon3 (Architecture Hypothesis)
Falcon3 is a transformer. Falcon-H1 is a hybrid SSM-Transformer (Mamba-style). SSMs have different information routing — the "where is knowledge stored" question may have a different answer. If RMU is calibrated for transformer MLP layers, it may need modification for SSM blocks. This is a genuine research contribution if tested.

---

## 7. Full Citation Index

### Biosecurity
```
Li, Y., et al. (2024). The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning. 
  ICML 2024. arXiv:2403.03218. https://arxiv.org/abs/2403.03218

FutureHouse (2024). LAB-Bench: Measuring Capabilities of Language Models for Biology Research.
  arXiv:2407.10362. https://arxiv.org/abs/2407.10362

Anonymous (2025). Virology Capabilities Test (VCT): A Multimodal Virology Q&A Benchmark.
  arXiv:2504.16137. https://arxiv.org/abs/2504.16137

Anonymous (2025). ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity.
  NeurIPS 2025. https://openreview.net/pdf/efa6989a1bbafaf92bb9ce187b701c826ecffed5.pdf

Anonymous (2024). BioLP-bench: Measuring understanding of biological lab protocols by LLMs.
  bioRxiv:2024.08.21.608694. https://www.biorxiv.org/content/10.1101/2024.08.21.608694v3

Anonymous (2025). LLM Novice Uplift on Dual-Use, In Silico Biology Tasks.
  arXiv:2602.23329. https://arxiv.org/pdf/2602.23329

Anonymous (2025). Measuring Mid-2025 LLM-Assistance on Novice Performance in Biology.
  arXiv:2602.16703. https://arxiv.org/pdf/2602.16703

Mazeika, M., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated 
  Red Teaming and Robust Refusal. arXiv:2402.04249. https://arxiv.org/abs/2402.04249
```

### Cybersecurity
```
Bhatt, M., et al. (2023). Purple Llama CyberSecEval: A Secure Coding Benchmark for LLMs.
  arXiv:2312.04724. https://arxiv.org/abs/2312.04724

Bhatt, M., et al. (2024). CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for LLMs.
  arXiv:2404.13161. https://arxiv.org/abs/2404.13161

Bhatt, M., et al. (2024). CyberSecEval 3: Advancing the Evaluation of Cybersecurity Risks 
  and Capabilities in LLMs. arXiv:2408.01605. https://arxiv.org/abs/2408.01605

Topala, I., et al. (2024). NYU CTF Bench: A Scalable Open-Source Benchmark for Evaluating 
  LLMs in Offensive Security. arXiv:2406.05590. NeurIPS 2024.

Palisade Research (2024). Hacking CTFs with Plain Agents. arXiv:2412.02776.

Anonymous (2024). SECURE: Benchmarking LLMs for Cybersecurity Advisory. arXiv:2405.20441.
```

### Machine Unlearning
```
Li, Y., et al. (2024). [RMU] The WMDP Benchmark. arXiv:2403.03218.

Anonymous (2024). Negative Preference Optimization: From Catastrophic Collapse to Effective 
  Unlearning. arXiv:2404.05868.

Touvron, L., et al. (2024). TOFU: A Task of Fictitious Unlearning for LLMs. arXiv:2401.06121.

Anonymous (2024). Large Language Model Unlearning (LLMU). OpenReview NeurIPS 2024.

Eldan, R. & Russinovich, M. (2023). Who's Harry Potter? Approximate Unlearning in LLMs.
  arXiv:2310.02238. https://arxiv.org/abs/2310.02238

Bourtoule, L., et al. (2021). Machine Unlearning (SISA). IEEE S&P. arXiv:1912.03817.

Meng, K., et al. (2022). Locating and Editing Factual Associations in Pre-Trained LMs (ROME).
  NeurIPS 2022. arXiv:2202.05629.

Meng, K., et al. (2022). Mass Editing Memory in a Transformer (MEMIT). arXiv:2210.07229.

Ilharco, G., et al. (2023). Editing Models with Task Arithmetic. arXiv:2212.04089.

Anonymous (2025). Per-parameter Task Arithmetic for Unlearning in LLMs. arXiv:2601.22030.

Anonymous (2025). Rethinking LLM Unlearning Objectives: A Gradient Perspective. 
  ICLR 2025. arXiv:2502.19301.

Geng, X., et al. (2025). A Comprehensive Survey of Machine Unlearning in LLMs. arXiv:2503.01854.
```

### Representation Learning
```
Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.
  arXiv:2310.01405. https://arxiv.org/abs/2310.01405

Anonymous (2025). The Hidden Dimensions of LLM Alignment: Multi-Dimensional Safety Analysis.
  arXiv:2502.09674.

Bricken, T., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from 
  Claude 3 Sonnet. Transformer Circuits. https://transformer-circuits.pub/2024/scaling-monosemanticity/

Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in LLMs.
  arXiv:2309.08600.

Belrose, N., et al. (2023). LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.
  arXiv:2306.03819.

Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT (ROME). 
  NeurIPS 2022. arXiv:2202.05629.

Anonymous (2025). The Rogue Scalpel: Activation Steering Compromises LLM Safety.
  arXiv:2509.22067.

Anonymous (2025). Analysing the Safety Pitfalls of Steering Vectors. arXiv:2603.24543.

Anonymous (2025). SafeSteer: Interpretable Safety Steering with Refusal-Evasion in LLMs.
  arXiv:2506.04250.

Anonymous (2025). Beyond Surface Alignment: Rebuilding LLM Safety via Probabilistically 
  Ablating Refusal Direction. arXiv:2509.15202.

Anonymous (2024). Robust LLM Safeguarding via Refusal Feature Adversarial Training. 
  arXiv:2409.20089.

Anonymous (2025). The Geometry of Harmfulness in LLMs through Subconcept Probing. 
  arXiv:2507.21141.

Anonymous (2025). False Sense of Security: Why Probing-based Malicious Input Detection 
  Fails to Generalize. arXiv:2509.03888.

Anonymous (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717.

Anonymous (2025). When the Coffee Feature Activates on Coffins: Analysis of Feature 
  Extraction and Steering. arXiv:2601.03047.
```

---

*Last updated: 2026-05-22 | Session: Falcon Day 1*
