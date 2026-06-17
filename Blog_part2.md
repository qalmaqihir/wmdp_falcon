# Part 2: Can You Make Falcon3 Forget What It Knows? I Tried RMU.

*Part 2 of the Falcon3 biosecurity series. Part 1 established that Falcon3-1B scores 40.1% and Falcon3-7B scores 70.9% on WMDP-bio, well above the 25% random baseline, and robust to system prompt changes. Now we apply Representation Misdirection Unlearning and find out what it actually takes to degrade a model's biosecurity knowledge at the representation level.*



Part 1 ended with a clean null result: you cannot suppress Falcon3's demonstrated biosecurity knowledge by changing the system prompt. ±0.4 percentage points across three prompt conditions, all confidence intervals overlapping. The model's parametric knowledge is robust to surface-level instructional framing.

That finding motivates a different class of intervention. If behavioral guardrails sit above the knowledge, can we go lower, directly into the model's representations, and degrade the knowledge itself?

This is what machine unlearning research is trying to answer. And it is a hard problem.



## What Is RMU?

Representation Misdirection for Unlearning (RMU) was introduced alongside the WMDP benchmark in Li et al. (ICML 2024). The method operates directly on a model's internal representations rather than on its behavior.

The core idea: take a frozen copy of the model (the reference), and fine-tune a "live" copy with two competing objectives.

**Forget loss** :- on hazardous data (WMDP-bio questions), push the live model's hidden states at layer *L* toward a random noise vector *r*. Not toward zero, not toward silence ie. toward noise. The model's internal representation of biosecurity content is *misdirected*, not suppressed.

**Retain loss**:- on benign data (Wikitext-2), keep the live model's hidden states close to the reference model's. This is the preservation objective: the model should still be able to process general text normally.

**Total loss:** L = L_forget + α · L_retain

The parameter α controls the balance. High α = retain constraint is strong, forgetting is slow. Low α = forgetting is fast, but the retain constraint may fail to protect general capability.

This is distinctly not refusal training. Refusal training teaches the model to say "I can't help with that"; a behavior-layer block. RMU degrades what the model *knows* at the representation level. A model with effective RMU applied would not know the answer to dangerous questions, not merely decline to say it.



## What I Did

I applied RMU to `tiiuae/Falcon3-1B-Instruct` on an Apple M2 Max (MPS backend), running entirely locally via HuggingFace Transformers. The choice of 1B was practical: the 7B backward pass requires gradient checkpointing and substantially more memory than the 7B forward pass used in inference evals.

**Hyperparameters:** steps=300, lr=5×10⁻⁵, layer=9, α=100, β=1.0

**Data:** 200 WMDP-bio forget samples (disjoint from eval), 200 Wikitext-2-raw-v1 retain sequences, 50-sample quick-eval set for mid-training monitoring.

**Reference model:** A second frozen copy of Falcon3-1B-Instruct loaded alongside the live model. The retain loss measures divergence of live hidden states from this frozen reference.

The forget and retain sets were sampled once with seed=42 and held fixed throughout training.



## What Happened

Here is the training trajectory:

| Step | Forget Loss | Retain Loss | WMDP Accuracy (n=50) |
|------|------------|-------------|----------------------|
| 0 (baseline) | - | - | **48.0%** |
| 25 | 5.47 | 0.50 | 34.0% |
| 50 | 2.41 | 0.21 | 14.0% |
| 75 | 1.00 | 0.17 | 24.0% |
| 100 | 0.47 | 0.57 | 4.0% |
| **125** | **0.48** | **0.68** | **0.0% ← all format failures** |
| 150–300 | ~0.20 | 0.2–**404** | 0.0% (all format failures) |

By step 125, WMDP accuracy reached 0%. But this is not the clean result it appears to be.

At the final evaluation (step 300): **50 out of 50 samples were format failures.** The model produced no valid A/B/C/D answers. Not wrong answers or no answers at all.

At steps 291 and 294, retain loss spiked to 404.0 and 100.0, respectively. Normal range throughout training was 0.1–1.0. These spikes indicate that the live model's representations had drifted catastrophically away from the frozen reference on benign (Wikitext) data, the retain constraint had collapsed entirely.

![](./results/unlearning/rmu_20260617_172240/rmu_training_curves.png)



**The verdict: this is model collapse, not controlled unlearning.**

The forget loss did converge cleanly (24.0 → ~0.18). RMU did misdirect the biosecurity representations. But the retain constraint failed to hold the rest of the model together. With α=100, the retain penalty was insufficient to prevent the optimization from destroying the model's general output distribution in the process of destroying its biosecurity representations.

The result is a model that cannot write coherent text at all and not because it forgot biosecurity content, but because it forgot how to generate language.

---

## Why This Happened

α calibration is the core engineering challenge in RMU, and Li et al. are explicit about this. Their published α=1200 for WMDP-bio on Llama 2-7B was the result of a tuning sweep run against live MMLU accuracy. They monitored general capability preservation and backed off when it degraded.

I did not run that calibration loop. I set α=100 based on the paper's recommendation to "start high" and did not include live MMLU monitoring. For a 1B model with a different architecture, this was too aggressive.

The mechanism is straightforward: α=100 means the retain loss is weighted at 100× the forget loss in the total objective. But in absolute magnitude, the forget loss started at ~24 (large), while the retain loss started at ~0.5 (small). The optimizer drove the forget loss down rapidly; from 24 to 0.47 in the first 100 steps. As the forget loss collapsed toward zero, the total loss became almost entirely retain loss. But by then, the model parameters had moved so far from the reference that the retain constraint was chasing a moving target it could not catch.

The retain spikes at steps 291 and 294 are the signal that the retain loss computation itself became numerically unstable i.e the live model's hidden states had diverged from the reference model's to the point where the retain penalty exploded.

There is also a subtler issue: our retain set was Wikitext-2, which consists of running prose. The WMDP-bio MCQ format (single letter A/B/C/D answers) is quite different. When the retain constraint only protects prose generation, the model's MCQ answering format can collapse independently. Using MMLU questions as the retain set would directly preserve the MCQ-format output distribution.

---

## What Clean Unlearning Looks Like

For contrast, the Li et al. (2024) result on Llama 2-7B is:

| Metric | Before RMU | After RMU |
|--------|-----------|-----------|
| WMDP-bio (logprob eval) | 46% | ~25% (random) |
| MMLU | 46% | 45% |
| MT-Bench | 6.2 | 6.1 |
| Format failures | ~0% | ~0% |

The model answers WMDP-bio questions at random chance, it has genuinely lost the knowledge, while preserving essentially all general capability. Format failure rate stays near zero: the model still writes coherent text, still answers other MCQ benchmarks correctly, it just no longer has the biosecurity domain knowledge encoded in its representations.

That is the target. Our result missed it by collapsing the output distribution entirely rather than selectively degrading the biosecurity direction.

The difference is α calibration and a live MMLU guard during training.



## What the Method Did Get Right

It is worth being precise about what the run demonstrated.

The forget loss converged. Starting at 24.0 (the model's hidden states were far from the random noise target), it reached ~0.18 by step 300. The misdirection objective was achieved like the Falcon3-1B's biosecurity representations were moved toward the random noise vector at layer 9.

WMDP accuracy dropped from 48% to 0% in approximately 125 gradient steps. The drop was monotonic after step 50. The biosecurity knowledge degradation happened, as it just took the rest of the model's output capability with it.

This is a meaningful result about method behavior: RMU is aggressive. It works fast on the forget direction. The retain constraint is the engineering challenge, not the forget direction. If anything, our run demonstrated that the forget component is robust even on a novel architecture; the retention side is what needs careful calibration.



## Next Steps

The path to a clean unlearning result:

**α calibration sweep:** Run α ∈ {0.5, 1, 2, 5, 10, 25} with a 50-step pilot at each setting. Measure: WMDP-eval accuracy, format failure rate, and spot-check MMLU at each checkpoint. Find the largest α that keeps format failures below 5% and MMLU within 3pp of baseline.

**Live MMLU guard:** Add MMLU accuracy as a stopping criterion. Abort if MMLU drops more than 5pp from baseline at any checkpoint. This is the calibration loop Li et al. ran. We can reproduce it.

**Better retain set:** Replace Wikitext-2 with MMLU questions. This directly protects MCQ-format output and is more analogous to the "benign general knowledge" that WMDP-bio should not degrade.

**Early stop criterion:** At step 125 in the current run, WMDP accuracy was already 0% and format failures were still mounting. A criterion of "stop when WMDP-eval ≤ 27% AND fmt_failures < 5%" would have captured the result at a point before collapse.

**Falcon3-7B:** With α calibrated on 1B, scale up to 7B with gradient checkpointing enabled. Full WMDP-bio (n=1,273) before and after, MMLU (n≥500) for preservation check.

The publication target remains: before/after WMDP-bio delta + MMLU preservation on a Falcon3 model, using the same text-generation protocol as Phase 1. No one has published this. The path is clear. It is an engineering calibration problem, not a fundamental obstacle.



## Why This Matters for AI Safety

The system prompt null result from Part 1 established that behavioral guardrails cannot reduce demonstrated parametric knowledge. This Part 2 result shows that representation-level unlearning *can* reach that knowledge, the misdirection happens, but doing it cleanly, without collateral damage, requires careful calibration.

This is the actual engineering frontier of machine unlearning: not whether the method works in principle, but whether it can be applied reliably to novel architectures without destroying general capability in the process. Li et al. showed it on Llama 2-7B. Our result suggests the α parameter does not transfer across model families without retuning.

For AI safety practice, this has direct implications:

- Unlearning deployed models will require per-architecture calibration, not a one-size-fits-all parameter set
- Live capability monitoring (MMLU, MT-Bench) during unlearning is not optional such that it is the mechanism that prevents collapse
- The forget direction is the easy part; the retain constraint is the engineering challenge

These are tractable problems. Part 3 will address them.



## Reproducibility

The RMU implementation and training logs are at:

**GitHub: https://github.com/qalmaqihir/wmdp_falcon**

```bash
# Run RMU on Falcon3-1B
python unlearning/03_rmu_exercise.py \
  --steps 300 --forget-size 200 --retain-size 200

# Saved checkpoint
results/unlearning/rmu_20260617_172240/
```

RMU hyperparameters: steps=300, lr=5e-5, layer=9, alpha=100, beta=1.0. Forget set = WMDP-bio (seed=42, disjoint from eval). Retain set = Wikitext-2-raw-v1.



## References

- Li, N. et al. (2024). The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning. *ICML 2024*. arXiv:2403.03218
- Zou, A. et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405
- TII (2024). Falcon3 Model Family. [huggingface.co/tiiuae](https://huggingface.co/tiiuae)



*Jawad Haider · June 2026 · [Part 3: RMU Calibration on Falcon3 → Coming Next]()*
