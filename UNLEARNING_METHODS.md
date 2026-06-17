# LLM Unlearning Methods — Reference Guide
**Project:** Falcon3 Biosecurity Evaluation · AI Safety  
**Last updated:** 2026-06-09

---

## The Core Problem

Goal: remove specific knowledge (e.g. biosecurity) from a trained LLM **without** retraining from scratch and **without** destroying general capability.

Two measurement axes:
- **Forget quality** — does model score near-random on forget domain?
- **Retain quality** — does general capability (MMLU, MT-Bench) stay intact?

---

## Method Taxonomy

| Method | Category | Core Idea | Forget Quality | Retain Quality | Complexity |
|--------|----------|-----------|---------------|----------------|------------|
| **Gradient Ascent (GA)** | Weight-space | Maximize loss on forget data | Medium | Poor (catastrophic) | Low |
| **Gradient Difference (GD)** | Weight-space | GA on forget + GD on retain simultaneously | Better | Better | Low |
| **Task Vector (TV)** | Weight-space | Fine-tune on forget → subtract that delta from weights | Medium | Good | Low |
| **NPO / SimNPO** | Preference-opt | Treat forget outputs as "negative" preferences | Good | Good | Medium |
| **RMU** | Representation | Steer hidden states of forget data → random direction | Best (WMDP-tuned) | Best (WMDP-tuned) | Medium |
| **LUNE (LoRA)** | PEFT | LoRA fine-tuning with negative examples | Good | Good | Low-medium |

---

## Method 1 — Gradient Ascent (GA)

**Category:** Weight-space  
**Complexity:** ⭐ Lowest — good strawman baseline

### Idea
Reverse gradient descent: maximize loss on forget data instead of minimizing it. Model "unlearns" by being pushed away from the forget distribution.

### Loss
```python
loss = -model_loss(forget_batch)
loss.backward()
optimizer.step()
```

### Problem
No constraint on retain set → model degrades rapidly (catastrophic forgetting of general capability). Useful only as comparison baseline, not production unlearning.

### When to use
Always implement first. Takes 30 minutes. Gives lower bound on unlearning quality and upper bound on capability damage.

---

## Method 2 — Gradient Difference (GD)

**Category:** Weight-space  
**Complexity:** ⭐ Low — one extra loss term vs GA

### Idea
GA on forget set + gradient descent on retain set simultaneously. The retain loss acts as regularizer preventing catastrophic degradation.

### Loss
```python
loss = -alpha * loss(forget_batch) + beta * loss(retain_batch)
# alpha typically >> beta to prioritize forgetting
```

### Key hyperparameters
| Param | Typical range | Notes |
|-------|--------------|-------|
| `alpha` | 1.0–10.0 | Forget weight |
| `beta` | 0.1–1.0 | Retain weight |
| LR | 1e-5 – 1e-4 | Lower than standard fine-tuning |
| Steps | 100–500 | Monitor MMLU to stop early |

### Notes
- Better retain quality than pure GA
- Still operates purely in weight space → can miss deeper representations
- Good second baseline before RMU

---

## Method 3 — Task Vectors (TV)

**Category:** Weight-space arithmetic  
**Complexity:** ⭐⭐ Medium — requires two fine-tuning runs

### Idea
1. Fine-tune model on forget data → get `θ_forget`
2. Compute task vector: `τ = θ_forget - θ_pretrained`
3. Apply: `θ_unlearned = θ_pretrained - λ * τ`

The `τ` vector encodes "forget domain knowledge direction" in weight space. Subtracting it removes that knowledge.

### Code sketch
```python
# Step 1: fine-tune on forget data normally
# Step 2: compute delta
task_vector = {k: finetuned[k] - pretrained[k] for k in pretrained}
# Step 3: subtract
unlearned = {k: pretrained[k] - lambda_ * task_vector[k] for k in pretrained}
```

### Key hyperparameter
- `λ` (negation scale) — larger = more aggressive forgetting, smaller = safer retain

### Notes
- No training at unlearning time (fast)
- Quality varies by domain
- Works well when forget fine-tuning was clean

---

## Method 4 — NPO / SimNPO (Negative Preference Optimization)

**Category:** Preference optimization  
**Complexity:** ⭐⭐ Medium

### Idea
Frame unlearning as preference optimization. Model learns to **prefer** safe/generic outputs over forget-domain outputs. Borrows from RLHF/DPO.

**NPO loss:**
```
L_NPO = -E[log σ(β * log π_θ(y_safe|x) - β * log π_θ(y_forget|x))]
```

**SimNPO** (simpler): removes reference model, uses length-normalized reward.

### Notes
- More stable than GA — preference framing provides implicit regularization
- Needs "preferred" outputs constructed for retain domain
- SimNPO is current SOTA in the gradient-ascent family (2024–2025)
- Requires pairing forget inputs with safe alternative outputs

---

## Method 5 — RMU (Representation Misdirection for Unlearning) ⭐

**Category:** Representation engineering  
**Complexity:** ⭐⭐ Medium  
**Paper:** Li et al., 2024 — [WMDP Benchmark](https://arxiv.org/pdf/2403.03218)  
**Code:** [centerforaisafety/wmdp](https://github.com/centerforaisafety/wmdp)

### Core Idea
At a specific transformer layer `L`, push hidden states of forget-data toward a **fixed random vector** `u`, while keeping retain-data hidden states unchanged.

This destroys coherent intermediate representations for the forget domain → downstream layers cannot reconstruct the knowledge.

### Loss Function
```
L_total = alpha * L_forget + beta * L_retain

L_forget = || h_L(x_forget) - c * u ||²
L_retain = || h_L(x_retain) - h_L_frozen(x_retain) ||²
```

Where:
- `h_L` = hidden state at layer `L` of the **updated** model
- `h_L_frozen` = hidden state at layer `L` of a **frozen reference copy** of original model
- `u` = fixed random unit vector (sampled once, frozen throughout training)
- `c` = scaling coefficient (hyperparameter)

### PyTorch pseudocode
```python
import torch
import torch.nn.functional as F

# Setup
random_vector = torch.randn(hidden_size).to(device)
random_vector = random_vector / random_vector.norm()  # unit vector

# Forward pass (forget batch)
hidden_forget = model(forget_input, output_hidden_states=True).hidden_states[L]
target = c * random_vector.unsqueeze(0).expand_as(hidden_forget)
L_forget = F.mse_loss(hidden_forget, target)

# Forward pass (retain batch) — frozen reference model
with torch.no_grad():
    hidden_retain_ref = frozen_model(retain_input, output_hidden_states=True).hidden_states[L]
hidden_retain = model(retain_input, output_hidden_states=True).hidden_states[L]
L_retain = F.mse_loss(hidden_retain, hidden_retain_ref)

# Combined loss
loss = alpha * L_forget + beta * L_retain
loss.backward()
optimizer.step()
```

### Key Hyperparameters
| Param | WMDP paper default | Notes for Falcon3-1B |
|-------|--------------------|----------------------|
| Layer `L` | 7 (for 7B) | Try layers 4–6 for 1B (middle third) |
| `c` (scaling) | 20 | Adaptive variant: `c = norm(h_L)` |
| `alpha` (forget weight) | 1000 | High — forces misdirection |
| `beta` (retain weight) | 1 | Keep low |
| LR | 5e-5 | Adam |
| Steps | 150–300 | Stop when MMLU drops >2% |
| Batch size | 4 | Small — gradient stability |

### WMDP Benchmark Results (Zephyr-7B)
| Metric | Before RMU | After RMU |
|--------|-----------|----------|
| WMDP-Bio | ~55% | 31.2% |
| WMDP-Cyber | ~52% | 28.2% |
| MMLU | ~59% | ~58% |

Near-random (25%) on WMDP, negligible MMLU drop.

### Adaptive RMU (2025 improvement)
Instead of fixed `c`, scale by L2 norm of original hidden state:
```python
c = original_hidden_norm * scale_factor
```
Prevents fixed coefficient being too weak/strong across different layers.

---

## Method 6 — LUNE (LoRA + Negative Examples)

**Category:** PEFT  
**Complexity:** ⭐⭐ Medium  
**Paper:** [LUNE, 2024](https://arxiv.org/html/2512.07375v1)

### Idea
Use LoRA (parameter-efficient fine-tuning) with negative examples. Only trains small adapter matrices → low memory, fast training. Negative examples are constructed to steer model away from forget domain.

### Advantage over full fine-tuning
- 10–100x fewer trainable parameters
- Feasible on M2 Mac even for 7B
- LoRA can be detached/reattached

### Notes
- Good fit for resource-constrained environments (M2 Mac)
- Quality slightly below full-parameter RMU but much more practical
- Can combine LoRA with RMU loss for best-of-both

---

## For This Project (Falcon3 Pipeline)

### Recommended sequence
```
1. Baseline WMDP-bio (already done for some models)
2. Implement GA — 30 min, gives worst-case comparison
3. Implement GD — 30 min, adds retain regularization
4. Implement RMU — 2-3 hours, the main method
5. Eval each: WMDP-bio accuracy + MMLU subset
```

### Data
- **Forget:** WMDP-bio corpus (released by CAS alongside benchmark)
- **Retain:** Wikitext-103 or general Wikipedia slice

### Compute note (M2 Max)
- Falcon3-1B: RMU feasible, ~1–2 hrs training
- Falcon3-3B: borderline, try with gradient checkpointing
- Falcon3-7B: likely OOM on backward pass — use for eval only

### Expected result table
| Model | Method | WMDP-Bio | MMLU | Delta WMDP |
|-------|--------|----------|------|------------|
| Falcon3-1B | Baseline | ~?% | ~49% | — |
| Falcon3-1B | GA | ~?% | ~?% | Δ |
| Falcon3-1B | GD | ~?% | ~?% | Δ |
| Falcon3-1B | RMU | ~?% | ~?% | Δ |

---

## Key Limitations to Document

1. **WMDP is a proxy** — MCQ format doesn't test open-ended synthesis or agentic capability
2. **RMU is shallow** — [Alignment Forum analysis](https://www.alignmentforum.org/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow) shows representations partially recoverable with fine-tuning
3. **Quantization effects** — Ollama Q4 may compress out/in different knowledge than full-precision RMU targets
4. **Hyperparameter transfer** — WMDP paper tuned for Llama 2-7B; Falcon3 architecture differs

---

## References

| Paper | Link |
|-------|------|
| WMDP Benchmark + RMU (Li et al., 2024) | https://arxiv.org/pdf/2403.03218 |
| Official RMU code | https://github.com/centerforaisafety/wmdp |
| Survey: Machine Unlearning in LLMs (2025) | https://arxiv.org/html/2503.01854v2 |
| Rethinking Machine Unlearning for LLMs | https://arxiv.org/pdf/2402.08787 |
| Adaptive RMU — SemEval 2025 | https://arxiv.org/html/2506.16548v1 |
| LUNE: LoRA-based unlearning | https://arxiv.org/html/2512.07375v1 |
| RMU is mostly shallow — Alignment Forum | https://www.alignmentforum.org/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow |
| Unlearning blog overview | https://tuananhbui89.github.io/blog/2025/unlearn-llms/ |
