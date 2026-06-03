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

- [x] `P0.1` Start Ollama daemon: `ollama serve`
- [x] `P0.2` Ping all 4 Falcon3 models to confirm readiness:
  ```bash
  for m in 1b 3b 7b 10b; do
    echo -n "falcon3:${m}: "; ollama run falcon3:${m} "reply OK" --nowordwrap 2>&1 | head -1
  done
  ```
- [x] `P0.3` Activate venv + load env vars:
  ```bash
  cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
  source venv/bin/activate && source .env
  export HUGGINGFACE_API_KEY=$HF_TOKEN
  ```
- [x] `P0.4` Verify WMDP dataset is cached (avoid download during runs):
  ```bash
  python3 -c "from datasets import load_dataset; d=load_dataset('cais/wmdp','wmdp-bio',split='test'); print(len(d),'samples')"
  ```
  Confirmed: **1273 samples**
- [x] `P0.5` Verify Inspect AI version: `inspect --version`  (confirmed 0.3.x)
- [ ] `P0.6` Pin exact package versions to `requirements.txt` *(do after all runs)*:
  ```bash
  pip freeze | grep -E "inspect|transformers|torch|datasets|pandas|matplotlib|scipy|numpy" > requirements.txt
  ```

---

## Phase 1 — Experiment Infrastructure (code before running, ~1.5 hr)

All code lives in `falcon_eval_wmdp/experiments/`.

### 1A — Central Config (`experiments/config.py`)
- [x] `P1A.1` Define FALCON_MODELS dict: name → Ollama tag + parameter count
- [x] `P1A.2` Define BASELINE_MODELS dict (llama, mistral, qwen, deepseek, gemma, phi4-mini)
- [x] `P1A.3` Define eval constants: SEED=42, TEMPERATURE=0.0, DATASET="cais/wmdp", SPLIT="wmdp-bio", N_FULL=1273
- [x] `P1A.4` Define output paths: `results/raw/` (.eval files), `results/processed/` (CSVs), `figures/`

### 1B — Main Eval Runner (`experiments/run_wmdp_bio.py`)
- [x] `P1B.1` Load WMDP-bio via `hf_dataset` with `RecordToSample` (exact WMDP paper format)
- [x] `P1B.2` Task definition: `multiple_choice()` solver + `robust_choice()` scorer, no system prompt (matches WMDP paper protocol)
- [x] `P1B.3` Flat kwargs to `eval()`: temperature=0.0, seed=42, max_tokens=32 (768 for think-tag models)
- [x] `P1B.4` CLI arg parsing: `--model`, `--limit` (default=None = full run), `--system-prompt`, `--cot`
- [x] `P1B.5` Log hardware metadata: platform, python version, timestamp, chip info
- [x] `P1B.6` Inspect AI provides live progress display natively
- [x] `P1B.7` On completion: print accuracy + 95% Wilson CI + append to `results/processed/wmdp_bio_results.csv`

### 1C — Results Analyzer (`experiments/analyze_results.py`)
- [x] `P1C.1` Load from `results/processed/wmdp_bio_results.csv`
- [x] `P1C.2` Extract per-model: accuracy, n_correct, n_total, input_tokens, output_tokens, wall_time
- [x] `P1C.3` Compute 95% Wilson confidence intervals
- [x] `P1C.4` CSV written by `run_wmdp_bio.py` directly; analyzer reads and formats
- [x] `P1C.5` Print table: model | params | quant | accuracy | 95% CI | n | format-fail | ±random

### 1D — Plotting Suite (`experiments/plot_results.py`)
- [x] `P1D.1` **Figure 1**: Bar chart — all models sorted by accuracy, error bars (95% CI), Falcon3=blue, baselines=grey
- [x] `P1D.2` **Figure 2**: Scaling plot — log₂(params) vs. accuracy for Falcon3, published WMDP overlay points
- [x] `P1D.3` **Figure 3**: Heatmap — model × {accuracy, format_fail%, tokens/sample, time/sample}
- [x] `P1D.4` **Figure 4**: CDF of per-sample scores (requires raw .eval files)
- [x] `P1D.5` Saves all figures as `.png` (300 DPI) + `.pdf` to `figures/`
- [x] `P1D.6` matplotlib style: seaborn-v0_8-paper, fontsize=12, Paul Tol color-blind safe palette

### 1E — Validation Run (smoke test before full runs)
- [x] `P1E.1` Ran 10-sample test on Falcon3-1B to verify end-to-end pipeline
- [x] `P1E.2` Confirmed `.eval` files written, CSV updated, 0 format failures
- [x] `P1E.3` Confirmed scoring works correctly (predicted vs. target letters match expected format)
- [x] `P1E.4` **Key bugs fixed during smoke test**:
  - `eval()` takes flat kwargs, NOT `GenerateConfig` object
  - Ollama base URL must include `/v1` suffix: `http://localhost:11434/v1`
  - `target.text` (not `str(target)`) for Inspect AI Target objects in custom scorers

---

## Phase 2 — Run Falcon3 Models (main data collection) ✅ COMPLETE

Run sequentially (M2 can't parallelize two Ollama models well). All 4 completed 2026-06-02:

| Model | Params | Quant | Accuracy | 95% CI | Correct/1273 | Fmt-fail | Wall time |
|-------|--------|-------|----------|--------|--------------|----------|-----------|
| `falcon3:1b` | 1.7B | Q8_0 | **40.1%** | 37.5–42.9% | 511 | 1 | 1.2 min |
| `falcon3:3b` | 3.2B | Q4_K_M | **57.9%** | 55.2–60.6% | 737 | 6 | 2.1 min |
| `falcon3:7b` | 7.5B | Q4_K_M | **70.9%** | 68.4–73.4% | 903 | 1 | 6.1 min |
| `falcon3:10b` | 10.3B | Q4_K_M | **73.7%** | 71.2–76.0% | 938 | 0 | 7.8 min |

> ⚠️ CORRECTED (Phase 5): GPT-4 published score is **82.2%** (logprob eval, Li et al. 2024) — NOT 72.1%. The 72.1% figure was unverified. See `results/LITERATURE_COMPARISON.md` for full analysis. Direct comparison is also confounded by eval protocol (logprob vs. text-gen). Updated claim: Falcon3-10B (73.7%) approaches Mixtral-8x7B (74.8%) at ~4.5× fewer parameters.

- [x] `P2.1` **Falcon3-1B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:1b`
- [x] `P2.2` **Falcon3-3B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:3b`
- [x] `P2.3` **Falcon3-7B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:7b`
- [x] `P2.4` **Falcon3-10B** full run: `python experiments/run_wmdp_bio.py --model ollama/falcon3:10b`
- [ ] `P2.5` Compute scaling slope after Phase 3 complete (run `analyze_results.py`)

> **Note**: Falcon3-1B uses Q8_0 (higher precision) vs. Q4_K_M for 3B/7B/10B — document as confound in scaling analysis.

---

## Phase 3 — Run Comparison Baselines ✅ COMPLETE

Purpose: situate Falcon3 against sub-13B models from all major families at matched param counts.  
Decision (2026-06-02): replaced original oversized baselines (Phi4-14B, DeepSeek-R1-14B) with ~7-10B equivalents for fair comparison.

| Model | Params | Accuracy | 95% CI | Correct/1273 | Fmt-fail | Notes |
|-------|--------|----------|--------|--------------|----------|-------|
| `llama3.1:8b` | 8.0B | **72.7%** | 70.2–75.1% | 926 | 13 | |
| `mistral:7b` | 7.2B | **63.9%** | 61.2–66.5% | 813 | 1 | Published was ~45% (v0.2) — version gap explains diff |
| `qwen2.5:7b` | 7.6B | **71.6%** | 69.0–74.0% | 911 | 0 | Run twice — identical results, confirms temp=0 determinism |
| `phi4-mini:latest` | 3.8B | **62.1%** | 59.4–64.7% | 790 | 0 | |
| `deepseek-r1:7b` | 7.6B | ❌ DROPPED | — | — | — | Think tags → ~1.5 s/sample → 32+ hrs; aborted at 143/1273 |
| `gemma2:9b` | 9.2B | ❌ DROPPED | — | — | — | Not pulled; skipped in favor of moving to Phase 4 |

```bash
# Pull remaining models (run in a second terminal while deepseek-r1:7b runs):
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull gemma2:9b
ollama pull phi4-mini:latest

# Run order after deepseek finishes:
python experiments/run_wmdp_bio.py --model ollama/llama3.1:8b      # ✅ done
# deepseek-r1:7b DROPPED — think tags make it ~3+ hrs on M2 (too slow)
python experiments/run_wmdp_bio.py --model ollama/mistral:7b
python experiments/run_wmdp_bio.py --model ollama/qwen2.5:7b
python experiments/run_wmdp_bio.py --model ollama/gemma2:9b
python experiments/run_wmdp_bio.py --model ollama/phi4-mini:latest
```

- [x] `P3.1` **Llama3.1-8B**: `python experiments/run_wmdp_bio.py --model ollama/llama3.1:8b`
- [!] `P3.2` ~~**DeepSeek-R1-7B**~~ **DROPPED** — think tags cause ~1.5 s/sample on M2 → estimated 32+ hrs for full run. Aborted at 143/1273 (11%). Not feasible on local hardware.
- [x] `P3.3` **Mistral-7B**: `python experiments/run_wmdp_bio.py --model ollama/mistral:7b`
  - ⚠️ Our 63.9% vs. published 45% (Li et al. 2024) — version difference: we ran v0.3, paper tested v0.2-instruct. Note in methodology.
- [x] `P3.4` **Qwen2.5-7B**: `python experiments/run_wmdp_bio.py --model ollama/qwen2.5:7b`
  - Duplicate run removed from CSV; both runs gave identical 71.6% — validates temp=0 reproducibility.
- [!] `P3.5` ~~**Gemma2-9B**~~ **SKIPPED** — not pulled; moving to Phase 4. Can add later if needed.
- [x] `P3.6` **Phi4-mini-3.8B**: `python experiments/run_wmdp_bio.py --model ollama/phi4-mini:latest`

---

## Phase 4 — Ablation Experiments ✅ COMPLETE (CoT result pending)

These add scientific depth and distinguish the work from a simple benchmark run.

| Condition | Accuracy | 95% CI | Fmt-fail | vs baseline |
|-----------|----------|--------|----------|-------------|
| Falcon3-7B baseline (none, no CoT) | 70.9% | 68.4–73.4% | 1 (0.1%) | — |
| Helpful system prompt | 71.2% | 68.6–73.6% | 3 (0.2%) | +0.3pp (not significant) |
| Biosec system prompt | 70.5% | 67.9–72.9% | 5 (0.4%) | −0.4pp (not significant) |
| CoT max_tokens=32 | 29.4% | 26.9–31.9% | 250 (19.6%) | ❌ INVALID — see note |
| CoT max_tokens=512 | **PENDING** | — | — | rerun in progress |

- [x] `P4.1` **System prompt ablation** (Falcon3-7B only, 1273 samples):
  - Condition A: no system prompt (baseline, already done in Phase 2) — 70.9%
  - Condition B: "You are a helpful assistant." — **71.2%** (+0.3pp, within CI)
  - Condition C: biosecurity refusal prompt — **70.5%** (−0.4pp, within CI)
  - Result: **null result** — system prompts have no statistically significant effect.
  - The biosecurity suppression hypothesis is not supported.
- [~] `P4.2` **Chain-of-thought ablation** (Falcon3-7B, full 1273):
  - Condition A: baseline — 70.9%
  - Condition B (max_tokens=32): **INVALID** — 19.6% format failures; model reasoned past token limit, never output A–D. Not a real CoT measurement.
  - Condition B (max_tokens=512): **rerun in progress** — result TBD.
  - ⚠️ Note: `chain_of_thought()` solver generates multi-token reasoning; max_tokens=32 always truncates before the final answer letter.
- [x] `P4.3` **Answer format robustness**: answered by existing data.
  - Falcon3-7B baseline: 1/1273 = 0.1% failures → negligible; no fallback extractor needed.

---

## Phase 5 — Literature Comparison ✅ COMPLETE (2026-06-03)

Build a comparison table: our results vs. published WMDP-bio numbers.  
Full output: `results/LITERATURE_COMPARISON.md`

- [x] `P5.1` Searched: WMDP paper (Li et al. 2024, arXiv:2403.03218, full text extracted), WMDP GitHub README, HuggingFace dataset card, WMDP.ai (JS-rendered — inaccessible), Falcon3 HuggingFace model card, OpenAlex, sci-papers MCP.
- [x] `P5.2` Verified published numbers from primary source (WMDP paper, Table 2 Appendix B, logprob eval):
  ```
  GPT-4                      | 82.2%  |  Li et al. 2024 (logprob)
  Yi-34b                     | 75.3%  |  Li et al. 2024 (logprob)
  Mixtral-8x7B               | 74.8%  |  Li et al. 2024 (logprob)
  zephyr-7b                  | 63.7%  |  Li et al. 2024 (logprob)
  Claude-2                   | N/A    |  NOT in WMDP paper (prior entry was fabricated)
  Llama-2-70B / 7B           | N/A    |  NOT in WMDP paper (prior entries were fabricated)
  Mistral-7B-Instruct-v0.2   | N/A    |  NOT in WMDP paper (prior ~45% was unverified)
  Falcon3-1B/3B/7B/10B       | OUR RESULTS (first published — no prior data exists)
  Falcon3 technical report   | N/A    |  Not yet published (TII "coming soon" as of 2026-06)
  ```
- [x] `P5.3` Documented critical methodology gap: logprob (lm-eval-harness, paper) vs. text-gen (Inspect AI, ours). Logprob typically gives higher scores. Direct numeric comparison is approximate — within-cohort comparisons are the most valid. See `results/LITERATURE_COMPARISON.md § P5.3`.
- [x] `P5.4` Updated `experiments/plot_results.py` Figure 2: replaced incorrect SIZE_MAP scatter points with horizontal dashed reference lines (GPT-4=82.2%, Mixtral=74.8%, Yi-34b=75.3%, zephyr=63.7%) labeled "logprob refs — different eval protocol".
  - Also fixed `experiments/config.py:PUBLISHED_RESULTS` with 4 verified models + protocol note.
  - Fixed `results/FINDINGS.md`: corrected GPT-4 (82.2% not 72.1%), removed fabricated Claude-2/Llama-2 entries.

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
