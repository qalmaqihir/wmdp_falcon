# Research TODO — Falcon3 WMDP-Bio Sweep
**Date**: 2026-06-02  
**Goal**: First published WMDP biosecurity results for the Falcon3 model family.  
**Hardware**: M2 Max · Ollama 0.x · Inspect AI 0.3.223 · Python 3.11

---

## Status Key
- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[!]` blocked / issue

---

## Phase 0 — Pre-flight (do this first, ~15 min)

- [ ] `P0.1` Start Ollama daemon: `ollama serve`
- [ ] `P0.2` Ping all 4 Falcon3 models to confirm readiness:
  ```bash
  for m in 1b 3b 7b 10b; do
    echo -n "falcon3:${m}: "; ollama run falcon3:${m} "reply OK" --nowordwrap 2>&1 | head -1
  done
  ```
- [ ] `P0.3` Activate venv + load env vars:
  ```bash
  cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
  source venv/bin/activate && source .env
  export HUGGINGFACE_API_KEY=$HF_TOKEN
  ```
- [ ] `P0.4` Verify WMDP dataset is cached (avoid download during runs):
  ```bash
  python3 -c "from datasets import load_dataset; d=load_dataset('cais/wmdp','wmdp-bio',split='test'); print(len(d),'samples')"
  ```
  Expected: **1273 samples**
- [ ] `P0.5` Verify Inspect AI version: `inspect --version`  (expect 0.3.x)
- [ ] `P0.6` Pin exact package versions to `requirements.txt`:
  ```bash
  pip freeze | grep -E "inspect|transformers|torch|datasets|pandas|matplotlib|scipy|numpy" > requirements.txt
  ```

---

## Phase 1 — Experiment Infrastructure (code before running, ~1.5 hr)

All code lives in `falcon_eval_wmdp/experiments/`.

### 1A — Central Config (`experiments/config.py`)
- [ ] `P1A.1` Define FALCON_MODELS dict: name → Ollama tag + parameter count
- [ ] `P1A.2` Define BASELINE_MODELS dict (llama, deepseek, phi, qwen)
- [ ] `P1A.3` Define eval constants: SEED=42, TEMPERATURE=0.0, DATASET="cais/wmdp", SPLIT="wmdp-bio", N_FULL=1273
- [ ] `P1A.4` Define output paths: `results/raw/` (.eval files), `results/processed/` (CSVs), `figures/`

### 1B — Main Eval Runner (`experiments/run_wmdp_bio.py`)
- [ ] `P1B.1` Load WMDP-bio via `hf_dataset` with `RecordToSample` (exact WMDP paper format)
- [ ] `P1B.2` Task definition: `multiple_choice()` solver + `choice()` scorer, no system prompt (matches WMDP paper protocol)
- [ ] `P1B.3` `GenerateConfig`: temperature=0.0, seed=42, max_tokens=16 (MCQ only needs 1 token — keeps runs fast)
- [ ] `P1B.4` CLI arg parsing: `--model`, `--limit` (default=None = full run), `--log-dir`
- [ ] `P1B.5` Log hardware metadata into eval metadata: model name, ollama tag, M2/MPS, timestamp
- [ ] `P1B.6` Print live progress bar + estimated time remaining
- [ ] `P1B.7` On completion: print accuracy + 95% CI + save summary line to `results/processed/summary.csv`

### 1C — Results Analyzer (`experiments/analyze_results.py`)
- [ ] `P1C.1` Walk `results/raw/` for `.eval` files using `list_eval_logs`
- [ ] `P1C.2` Extract per-model: accuracy, n_correct, n_total, input_tokens, output_tokens, wall_time
- [ ] `P1C.3` Compute 95% Wilson confidence intervals (correct for binary proportions, better than normal approx)
- [ ] `P1C.4` Save to `results/processed/wmdp_bio_results.csv`
- [ ] `P1C.5` Print markdown table: model | params | accuracy | 95% CI | tokens | time

### 1D — Plotting Suite (`experiments/plot_results.py`)
- [ ] `P1D.1` **Figure 1**: Bar chart — all models (Falcon3 + baselines) sorted by accuracy, with error bars (95% CI). Color: Falcon3=blue family, baselines=grey.
- [ ] `P1D.2` **Figure 2**: Scaling plot — log(params) vs. accuracy for Falcon3 family only. Include published WMDP numbers for same-size Llama/Mistral/Qwen as overlapping points.
- [ ] `P1D.3` **Figure 3**: Heatmap — model × metric (accuracy, refusal_rate, tokens/sample, time/sample). Useful for characterizing model "personality".
- [ ] `P1D.4` **Figure 4**: CDF of per-sample scores — shows if a model is consistently mediocre vs. bimodal.
- [ ] `P1D.5` Save all figures to `figures/` as both `.png` (300 DPI) and `.pdf` (for papers).
- [ ] `P1D.6` Use `matplotlib` style: seaborn-v0_8-paper, fontsize=12, no chart junk. Color-blind safe palette.

### 1E — Validation Run (smoke test before full runs)
- [ ] `P1E.1` Run 20-sample test with all 4 Falcon3 models to verify scripts work end-to-end
- [ ] `P1E.2` Confirm `.eval` files written, CSV updated, no errors
- [ ] `P1E.3` Confirm scores are non-trivially different across models (not all same value = bug)

---

## Phase 2 — Run Falcon3 Models (main data collection, ~4.5 hr total)

Run sequentially (M2 can't parallelize two Ollama models well). Estimated runtimes at 1273 samples:

| Model | Params | Est. Time | Priority |
|-------|--------|-----------|----------|
| `falcon3:1b` | 1B | ~15 min | First |
| `falcon3:3b` | 3B | ~32 min | Second |
| `falcon3:7b` | 7B | ~57 min | Third |
| `falcon3:10b` | 10B | ~85 min | Fourth |

- [ ] `P2.1` **Falcon3-1B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:1b`
  - After: note accuracy + check for format failures (model not outputting A/B/C/D)
- [ ] `P2.2` **Falcon3-3B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:3b`
- [ ] `P2.3` **Falcon3-7B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:7b`
- [ ] `P2.4` **Falcon3-10B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:10b`
- [ ] `P2.5` After all runs: compute scaling slope (accuracy gain per parameter doubling)

> **Checkpoint**: if any model fails mid-run, Inspect AI logs partial results. Use `--resume` flag to continue from last checkpoint (verify this works in P1E smoke test).

---

## Phase 3 — Run Comparison Baselines (~3.5 hr total)

Purpose: situate Falcon3 in the broader model landscape.

| Model | Params | Est. Time | Notes |
|-------|--------|-----------|-------|
| `llama3.1:8b` | 8B | ~57 min | Direct Falcon3-7B competitor |
| `phi4:latest` | 14B | ~85 min | Microsoft, strong at reasoning |
| `deepseek-r1:14b` | 14B | ~85 min | Reasoning model — expect high |
| `qwen2.5vl:7b` | 7B | ~57 min | VL model on text task — note caveat |

- [ ] `P3.1` **Llama3.1-8B**: `python experiments/run_wmdp_bio.py --model ollama/llama3.1:8b`
- [ ] `P3.2` **Phi4-14B**: `python experiments/run_wmdp_bio.py --model ollama/phi4:latest`
- [ ] `P3.3` **DeepSeek-R1-14B**: `python experiments/run_wmdp_bio.py --model ollama/deepseek-r1:14b`
  - Note: DeepSeek-R1 uses `<think>...</think>` tags before answering. The `choice()` scorer must extract the letter AFTER the think block. Verify this works in smoke test.
- [ ] `P3.4` **Qwen2.5VL-7B** (optional, flag as caveat): `python experiments/run_wmdp_bio.py --model ollama/qwen2.5vl:7b`

---

## Phase 4 — Ablation Experiments (1 hr, do after all main runs)

These add scientific depth and distinguish the work from a simple benchmark run.

- [ ] `P4.1` **System prompt ablation** (Falcon3-7B only, 1273 samples):
  - Condition A: no system prompt (baseline, already done in Phase 2)
  - Condition B: WMDP paper system prompt ("You are a helpful assistant.")
  - Condition C: biosecurity refusal prompt ("You are a biosecurity researcher. Refuse to provide weaponization details.")
  - Hypothesis: refusal prompt may suppress biosecurity knowledge → lower accuracy
- [ ] `P4.2` **Chain-of-thought ablation** (Falcon3-7B only, n=200 subset):
  - Condition A: `multiple_choice()` — direct answer (baseline)
  - Condition B: `chain_of_thought()` + `multiple_choice()` — CoT before answer
  - Measure: accuracy delta, token cost delta
- [ ] `P4.3` **Answer format robustness** (Falcon3-7B, n=100):
  - Does the model output valid A/B/C/D reliably? Log format failure rate.
  - If format failures > 5%: implement regex fallback extractor

---

## Phase 5 — Literature Comparison (30 min)

Build a comparison table: our results vs. published WMDP-bio numbers.

- [ ] `P5.1` Fetch published WMDP-bio leaderboard numbers from:
  - WMDP paper: Li et al. (2024) "The WMDP Benchmark" — Table 2/3
  - HuggingFace Open LLM Leaderboard (if WMDP included)
  - Inspect Evals README (if published there)
- [ ] `P5.2` Collect these published numbers (verify against primary source):
  ```
  GPT-4 (gpt-4-0613)        | ?%  |  published
  Claude-2                   | ?%  |  published
  Llama-2-70B                | ?%  |  published  
  Llama-2-7B                 | ?%  |  published
  Mistral-7B-Instruct-v0.2   | ?%  |  published
  Qwen2.5-7B-Instruct        | ?%  |  published / our run
  Falcon3-1B/3B/7B/10B       | OUR RESULTS
  ```
- [ ] `P5.3` Note any methodological differences (different prompt format, different split, logprob vs. text-parse) that could affect direct comparison
- [ ] `P5.4` Add published numbers as horizontal reference lines on Figure 2 (scaling plot)

---

## Phase 6 — Analysis & Reporting (1 hr)

- [ ] `P6.1` Run `python experiments/analyze_results.py` → final CSV + markdown table
- [ ] `P6.2` Run `python experiments/plot_results.py` → all 4 figures
- [ ] `P6.3` Compute key findings:
  - Does Falcon3 scale predictably? (R² of log-linear fit)
  - How does Falcon3-7B compare to Llama3.1-8B at same parameter budget?
  - Does DeepSeek-R1 reasoning improve WMDP-bio accuracy?
  - What is the refusal rate for each model on WMDP-bio questions?
- [ ] `P6.4` Write `results/FINDINGS.md`:
  - 3-5 bullet headline findings
  - Results table with 95% CIs
  - Figures embedded
  - Caveats: M2 local inference, Ollama quantization level, no logprob scoring
- [ ] `P6.5` Update `PROJECT_LOG.md` with today's experimental results

---

## Phase 7 — Reproducibility Package (30 min)

So any researcher can replicate.

- [ ] `P7.1` Final `requirements.txt` with pinned versions
- [ ] `P7.2` `reproduce.sh` — one-command replication script:
  ```bash
  #!/bin/bash
  # Runs all experiments from scratch
  source .env && export HUGGINGFACE_API_KEY=$HF_TOKEN
  for m in falcon3:1b falcon3:3b falcon3:7b falcon3:10b llama3.1:8b phi4:latest deepseek-r1:14b; do
    python experiments/run_wmdp_bio.py --model ollama/$m
  done
  python experiments/analyze_results.py
  python experiments/plot_results.py
  ```
- [ ] `P7.3` Git commit everything to `falcon_eval_wmdp` repo
- [ ] `P7.4` Tag commit: `git tag v1.0-wmdp-bio-sweep`

---

## Design Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Eval framework | Inspect AI | Native WMDP support, reproducible logs, Python API |
| Inference | Ollama (local) | M2 Mac, no NVIDIA GPU, all 4 Falcon3 sizes fit in RAM |
| Temperature | 0.0 (greedy) | Deterministic, reproducible; matches MCQ evaluation standard |
| Scorer | `choice()` | Standard for WMDP; extracts A/B/C/D from response |
| Solver | `multiple_choice()` | Standard format; formats (A)...(B)... and requests 1-letter answer |
| System prompt | None (baseline) | Matches WMDP paper protocol; system prompt ablation done separately |
| Dataset | Full test set (n=1273) | Statistical power; 95% CI = ±2.7% at 50% accuracy |
| Seed | 42 | Reproducibility; Inspect AI passes to Ollama for sampling |
| Max tokens | 16 | MCQ only needs 1 token; short limit avoids runaway generation |

---

## Statistical Notes

- With n=1273 samples and accuracy p̂, the 95% Wilson CI is:  
  `p̂ ± 1.96 * sqrt(p̂(1-p̂)/n)`  
  At p̂=0.50: ±2.7%. At p̂=0.35: ±2.6%. At p̂=0.25: ±2.4%.
- Random chance = 25% (4-choice MCQ)
- A model significantly above 25% is exhibiting real biosecurity knowledge
- To claim two models differ significantly: their 95% CIs must not overlap

---

## Known Risks / Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Ollama crashes mid-run | Medium | Inspect AI partial logs; `--resume` or re-run with `--limit` offset |
| DeepSeek `<think>` tokens confuse `choice()` scorer | High | Test in smoke test P1E; fix regex extractor in P4.3 if needed |
| Ollama quantization level unknown | Medium | Document via `ollama show falcon3:7b --modelfile` in Phase 0 |
| Memory pressure with 10B model | Low | M2 Max has 32–64GB unified memory; should be fine |
| WMDP-bio dataset version mismatch | Low | Pin dataset revision in config |

---

*Created: 2026-06-02. Update status markers as work progresses.*
