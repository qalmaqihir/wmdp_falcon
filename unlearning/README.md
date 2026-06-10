# Unlearning Module — Falcon3 on WMDP-bio

Implements machine unlearning baselines on Falcon3-1B-Instruct.
Goal: demonstrate the full measure → unlearn → re-measure pipeline.

## File Map

| File | Role |
|------|------|
| `config.py` | All hyperparameter constants |
| `utils.py` | Shared utilities — do not modify |
| `00_architecture_exploration.py` | Educational — run before Exercise 01 |
| `01_ga_gd_exercise.py` | **Exercise 01** — implement 3 TODOs (GA + GD) |
| `02_rmu_exploration.py` | Educational — run before Exercise 03 |
| `03_rmu_exercise.py` | **Exercise 03** — implement 4 TODOs (RMU) |
| `04_quantization_exploration.py` | Educational — run before Exercise 05 |
| `05_quantized_rmu_exercise.py` | **Exercise 05** — implement 2 TODOs (LoRA-RMU) |

## Setup

```bash
cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
source venv/bin/activate          # venv is at project ROOT, not inside falcon_eval_wmdp/
cd falcon_eval_wmdp
source ../.env                    # loads HF_TOKEN (optional — only for gated wmdp-corpora)
```

## Step 1 — Architecture Exploration (read-only, ~2 min)

```bash
python unlearning/00_architecture_exploration.py
```

Outputs: layer count, hidden size, layer access path (`model.model.layers[L]`), hidden state shapes.
No training. Not needed for GA/GD — but builds intuition for RMU (Exercise 02).

## Step 2 — Implement the TODOs in `01_ga_gd_exercise.py`

Open the file. Find the three `raise NotImplementedError` blocks:

- **TODO 1** (`ga_loss`): 3 lines — forward pass, flip sign, return.
- **TODO 2** (`gd_loss`): 5 lines — call `ga_loss` + retain CE + combine.
- **TODO 3** (`train_unlearning`): ~25 lines — training loop with cycle iterators.

Read the docstring above each TODO carefully before coding.

## Step 3 — Smoke Test

```bash
# Verifies all code paths. ~2 min. No meaningful unlearning at 5 steps.
python unlearning/01_ga_gd_exercise.py --method ga --steps 5 --forget-size 20 --eval-samples 20
```

Expected output: no errors, baseline eval runs, 5 training steps with logged loss, post-eval runs.

## Step 4 — Full GA Run

```bash
python unlearning/01_ga_gd_exercise.py --method ga --steps 100 --forget-size 200
```

~10 min on M2 Max. **Expected observation:** WMDP-bio accuracy drops, but the model starts producing
low-quality output in general (random tokens, repetition). This is catastrophic forgetting.

## Step 5 — Full GD Run

```bash
python unlearning/01_ga_gd_exercise.py --method gd --steps 200 --forget-size 200 --retain-size 200
```

~20 min on M2 Max. **Expected observation:** WMDP-bio accuracy drops similarly, but general output
quality is preserved because the retain regularizer counteracts the GA term.

## Step 6 — Benchmark-Grade Final Eval (optional)

```bash
python unlearning/01_ga_gd_exercise.py --method gd --steps 200 --full-eval
```

Adds full 1273-sample eval at the end (~15–20 min extra). Gives a number comparable to the
existing baseline from `experiments/run_wmdp_bio.py`.

## Expected Results Table (fill in as you run)

| Model | Method | Steps | WMDP-bio Pre | WMDP-bio Post | Delta |
|-------|--------|-------|-------------|--------------|-------|
| Falcon3-1B | Baseline | — | ?% | — | — |
| Falcon3-1B | GA | 100 | ?% | ?% | Δ |
| Falcon3-1B | GD | 200 | ?% | ?% | Δ |
| Falcon3-1B | RMU | 300 | ?% | ?% | Δ |
| Falcon3-1B | RMU + LoRA (bf16) | 300 | ?% | ?% | Δ |

---

## Exercise 02 — RMU Exploration (read-only, ~5 min)

```bash
python unlearning/02_rmu_exploration.py
```

Seven sections with no training, no TODOs:

1. Architecture recap — verifies layer count and hidden size for Falcon3-1B
2. Forward hooks live demo — registers `register_forward_hook` on layer 9, runs one forward pass, inspects shape
3. Random misdirection vector — builds `c` (unit norm, seed 42), shows norm = 1.0
4. Forget loss preview — computes MSE between hidden state and `alpha * c`
5. Retain loss preview — computes MSE between live and frozen hidden states
6. Two-pass dry run — full forget + retain forward pass, no `optimizer.step()`
7. Why MSE not CE — explains representation-space loss vs token-prediction loss

All sample texts use `"<Dummy Text for forget data>"` — no sensitive content anywhere in this file.

---

## Exercise 03 — Implement RMU

### Step A — Read the exploration first

Run `02_rmu_exploration.py` above. Understand hook registration, the random vector, and the two-pass structure before touching the TODOs.

### Step B — Implement the 4 TODOs in `03_rmu_exercise.py`

- **TODO 1** (`make_hidden_state_hook`): ~3 lines — closure that writes `output[0]` into `storage[key]`.
- **TODO 2** (`rmu_forget_loss`): ~3 lines — MSE between hidden and `alpha * random_vec` (expanded).
- **TODO 3** (`rmu_retain_loss`): ~1 line — MSE between live and frozen hidden states.
- **TODO 4** (`train_rmu`): ~45 lines — two-pass loop: forget forward → retain frozen (no_grad) → retain live → combined loss → backward → step.

Read the docstring above each `raise NotImplementedError` before coding.

**Key architecture facts:**
- Hook target (no PEFT): `model.model.layers[RMU_LAYER]`
- Frozen model: loaded with `load_model_and_tokenizer(frozen=True)` — all params frozen, eval mode
- Random vector: `build_rmu_random_vector(hidden_size, device, seed=42)` — seeded, unit norm
- Loss: `loss_f + BETA_RMU * loss_r` — no CE anywhere

### Step C — Smoke Test

```bash
python unlearning/03_rmu_exercise.py --steps 5 --forget-size 20 --retain-size 20 --eval-samples 20
```

Expected: no errors, baseline eval, 5 steps with forget/retain loss logged, post-eval.

### Step D — Full RMU Run

```bash
python unlearning/03_rmu_exercise.py --steps 300 --forget-size 200 --retain-size 200
```

~30 min on M2 Max. **Expected:** WMDP accuracy drops while retain quality holds better than GA.

### Step E — Full Eval

```bash
python unlearning/03_rmu_exercise.py --steps 300 --full-eval
```

Adds the 1273-sample WMDP eval at the end.

---

## Exercise 04 — Quantization Exploration (read-only, ~3 min)

```bash
python unlearning/04_quantization_exploration.py
```

Seven sections, no training, no TODOs:

1. Memory math — parameter footprint at fp32 / bf16 / int8 / int4
2. What is quantization — NF4 vs INT8 vs INT4 comparison
3. Why it matters for RMU — what changes and what stays the same
4. Per-method quantization impact — GA, GD, RMU, Task Vectors
5. Device matrix — what runs on CUDA / MPS / CPU
6. LoRA mechanics live demo — wraps model, prints trainable params (~2M of ~1B), verifies hook still works through PEFT wrapper
7. What stays the same in Exercise 05 vs Exercise 03

**Key insight:** On M2 MPS, `bitsandbytes` INT4/INT8 is unavailable. Exercise 05 uses bf16 + LoRA (full-float base, adapter-only training). The training loop is identical to QLoRA — only the base dtype differs.

---

## Exercise 05 — Implement LoRA-RMU

### Step A — Read the exploration first

Run `04_quantization_exploration.py` above. Understand the PEFT layer path shift and why optimizer scope changes.

### Step B — Implement the 2 TODOs in `05_quantized_rmu_exercise.py`

- **TODO 1** (`register_rmu_hooks_for_lora_model`): ~10 lines — register hooks on both the PEFT-wrapped live model and the frozen reference model. Layer paths differ:
  - Live (PEFT):   `peft_model.base_model.model.model.layers[layer_idx]`
  - Frozen (plain): `frozen_model.model.layers[layer_idx]`
- **TODO 2** (`train_qlora_rmu`): ~30 lines — same loop as `train_rmu` in Exercise 03, but:
  - Optimizer targets only `(p for p in peft_model.parameters() if p.requires_grad)` (LoRA params only)
  - Use `peft_model` for live forward; `frozen_model` for retain reference (no_grad)
  - Loss math is identical to Exercise 03

### Step C — Smoke Test

```bash
python unlearning/05_quantized_rmu_exercise.py --backend bf16-lora --steps 5 --forget-size 20
```

### Step D — Full LoRA-RMU Run

```bash
python unlearning/05_quantized_rmu_exercise.py --backend bf16-lora --steps 300 --forget-size 200 --retain-size 200
```

Adapter checkpoint saved to `results/unlearning/` (~10 MB instead of ~2 GB).

### CUDA-only backends (if running on GPU server)

```bash
python unlearning/05_quantized_rmu_exercise.py --backend 4bit-lora --steps 300
python unlearning/05_quantized_rmu_exercise.py --backend 8bit-lora --steps 300
```

These exit cleanly on MPS with an explanatory message — no crash.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `cais/wmdp-corpora` gated error | Normal — falls back automatically to MCQ text. Or `huggingface-cli login` |
| MPS OOM | Reduce `--forget-size` to 50 or lower. `BATCH_SIZE=2` is already conservative |
| `NotImplementedError` | You haven't filled in a TODO yet |
| Loss doesn't change | Check `optimizer.zero_grad()` placement — must be before `loss.backward()` each step |
| Post-eval same as baseline | 5–10 steps is too few. Use ≥50 steps |
| Model output garbage after GA | Expected — this IS the catastrophic forgetting lesson |
| Hook captures wrong layer | Verify `model.model.layers[L]` (no PEFT) vs `peft_model.base_model.model.model.layers[L]` (with PEFT) |
| Gradient not flowing through hook | Do NOT detach `output[0]` in the hook — gradient must flow back through the live model |
| `bitsandbytes` error on MPS | Expected — use `--backend bf16-lora` on M2 Mac |
| LoRA adapter not saving | Check `peft_model.save_pretrained(path)` not `model.save_pretrained(path)` |
