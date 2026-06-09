# Unlearning Module — Falcon3 on WMDP-bio

Implements machine unlearning baselines on Falcon3-1B-Instruct.
Goal: demonstrate the full measure → unlearn → re-measure pipeline.

## File Map

| File | Role |
|------|------|
| `config.py` | All hyperparameter constants |
| `utils.py` | Shared utilities — do not modify |
| `00_architecture_exploration.py` | Educational — run before Exercise 01 |
| `01_ga_gd_exercise.py` | **YOUR EXERCISE** — implement 3 TODOs |

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

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `cais/wmdp-corpora` gated error | Normal — falls back automatically to MCQ text. Or `huggingface-cli login` |
| MPS OOM | Reduce `--forget-size` to 50 or lower. BATCH_SIZE=2 is already conservative |
| `NotImplementedError` | You haven't filled in a TODO yet |
| Loss doesn't change | Check `optimizer.zero_grad()` placement — must be AFTER `optimizer.step()` |
| Post-eval same as baseline | 5–10 steps is too few. Use ≥50 steps with `--steps 50` |
| Model output is garbage after GA | Expected — this IS the catastrophic forgetting lesson |

## Expected Results Table (fill in as you run)

| Model | Method | Steps | WMDP-bio Pre | WMDP-bio Post | Delta |
|-------|--------|-------|-------------|--------------|-------|
| Falcon3-1B | Baseline | — | ?% | — | — |
| Falcon3-1B | GA | 100 | ?% | ?% | Δ |
| Falcon3-1B | GD | 200 | ?% | ?% | Δ |
| Falcon3-1B | RMU | — | ?% | ?% | Δ |

## Next: Exercise 02 — RMU

RMU (Representation Misdirection for Unlearning) is architecturally-aware:
it hooks into hidden states at layer L instead of operating on the final loss.

After completing this exercise, you will have the training loop mechanics
(cycles, backward, step, eval) needed to implement RMU's two-loss structure.
