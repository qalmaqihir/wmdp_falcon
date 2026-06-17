# Plan: Comprehensive `inspect-ai` Tutorial with Falcon Models

## Context

User wants strong grip on the **inspect-ai** framework (v0.3.223 installed, UK AISI's eval library). The placeholder file [inspect-ai-tutorial.py](inspect-ai-tutorial.py) is empty. Existing project shows: 15 prior `.eval` log files in `logs/` (all wmdp-bio runs from 2026-05-20/23, ollama provider), Falcon3-1B-Instruct + Falcon3-7B-Instruct cached locally, FINAL_PLAN.md already gestures at the four building blocks (Task / Dataset / Solver / Scorer) at lines 70–122. Goal: a single comprehensive tutorial covering **every** important inspect-ai concept exposed in the API surface map, anchored to Falcon models, so the user has a working reference for the biosecurity research project.

Decisions from clarifying questions:
- **Provider**: both — ollama_chat/falcon3:1b as default, with cell showing hf/ provider swap.
- **Tools/agents**: full coverage including @tool, use_tools, basic_agent, react.
- **Layout**: code in `inspect-ai-tutorial.py` + narrative companion in `INSPECT_AI_TUTORIAL.md`.

## Files to create

1. **[inspect-ai-tutorial.py](inspect-ai-tutorial.py)** — runnable cell-based tutorial (`# %%` separators).
2. **[INSPECT_AI_TUTORIAL.md](INSPECT_AI_TUTORIAL.md)** — narrative explanation paired part-by-part with the code cells. Each section names the corresponding cell heading in the .py so cross-reference is mechanical.

## Tutorial structure (8 parts, ~24 cells)

Each part = 2–4 cells in the .py + one ## section in the .md. Concepts in **bold** are the public API symbols being introduced.

### Part 1 — Foundations: Task, Eval, CLI
- 1A: Imports + **`@task`** decorator + **`Task(dataset, solver, scorer)`** — minimal wmdp_bio reproduction using `inspect_evals.wmdp.wmdp_bio`.
- 1B: **`eval(task, model="ollama_chat/falcon3:1b", limit=10, log_dir="logs/")`** programmatic run, inspect returned `EvalLog`.
- 1C: Same via CLI: `!inspect eval inspect_evals/wmdp_bio --model ollama_chat/falcon3:1b --limit 10`. Note **`inspect view`** for browsing.
- 1D: Provider swap — `model="hf/tiiuae/Falcon3-1B-Instruct"` with `model_args={"device": "mps", "torch_dtype": "float16"}`.

### Part 2 — Dataset Deep Dive
- 2A: **`Sample`** + **`MemoryDataset`** — handcraft 3 biosecurity-flavored MCQ samples.
- 2B: **`hf_dataset`** + **`FieldSpec`** + **`RecordToSample`** — load `cais/wmdp` directly with custom field mapping.
- 2C: **`csv_dataset`** / **`json_dataset`** — load from a tiny local file written ad-hoc in the cell.

### Part 3 — Solver Deep Dive
- 3A: **`generate`** baseline + **`system_message`** — Falcon as biosecurity-aware assistant.
- 3B: **`chain_of_thought`** + **`prompt_template`** — show effect on a hard MCQ.
- 3C: **`multiple_choice`** solver + **`MultipleChoiceTemplate`** — the canonical MCQ scaffold.
- 3D: **`chain(...)`** composition + **`fork(...)`** parallel branches + **`self_critique`** — three-step pipeline.

### Part 4 — Scorer Deep Dive
- 4A: Built-ins tour — **`choice`**, **`exact`**, **`match`**, **`includes`**, **`pattern`**, **`f1`** — applied to small toy outputs.
- 4B: **`model_graded_fact`** + **`model_graded_qa`** — use Falcon3-7B-Instruct as the grader for 1B's answers.
- 4C: Custom **`@scorer`** + custom **`@metric`** + score values **`CORRECT`**/**`INCORRECT`**/**`PARTIAL`** — write a "contains-numeric-citation" scorer.
- 4D: Score reducers — **`pass_at`**, **`mean_score`**, **`at_least`** — used with epochs in Part 8.

### Part 5 — Model Configuration
- 5A: **`get_model`** + **`GenerateConfig`** (temperature, max_tokens, top_p, stop_seqs, logprobs, seed) — direct model.generate() call.
- 5B: **`ChatMessageSystem`** / **`ChatMessageUser`** / **`ChatMessageAssistant`** — manual chat construction + `model.generate(messages)`.
- 5C: Multi-model comparison via list `eval(task, model=["ollama_chat/falcon3:1b", "ollama_chat/falcon3:3b"])`.
- 5D: **`eval_set`** — batched runs over (task × model × config) grid.

### Part 6 — Tools & Agents
- 6A: **`@tool`** decorator — define a toy `lookup_pathogen_info(name: str)` returning canned safe data. **`ToolDef`** intro.
- 6B: **`use_tools`** solver — wire the tool to a Task, observe tool calls.
- 6C: **`basic_agent`** — a Falcon-powered loop with the toy tool + **`message_limit`** safety.
- 6D: **`react`** agent — ReAct-style reasoning agent. Brief mention of **`bridge`**, **`handoff`**, **`as_solver`**.

### Part 7 — Log Analysis
- 7A: **`read_eval_log`** on one of the existing `logs/*.eval` files. Walk `EvalLog.results`, `.samples`, `.stats`.
- 7B: **`list_eval_logs`** + **`read_eval_log_sample_summaries`** — aggregate across all 15 prior runs.
- 7C: **`Transcript`** + **`transcript`** — programmatic transcript export. Note **`bundle_log_dir`** for sharing.

### Part 8 — Advanced Patterns
- 8A: **`Epochs(n=3, reducer="pass_at_3")`** — repeated trials. Wire to Part 4D reducers.
- 8B: **`token_limit`** / **`time_limit`** / **`message_limit`** / **`sample_limits`** — eval-level guardrails.
- 8C: Custom task from scratch — `biosec_refusal_task()` with handcrafted samples + custom @scorer measuring refusal rate.
- 8D: **`approval`** / **`human_approver`** intro (brief, no real prompt) + closing pointer to sandboxes (**`sandboxenv`**) for code-execution evals.

## Companion `.md` structure

Mirror the 8 parts. Each section:
- **What it covers** — concept(s) introduced.
- **Why it matters** — when you'd reach for this in real evals.
- **Code reference** — link to the matching `# %%` cell in `inspect-ai-tutorial.py` (e.g. `Cell 3C`).
- **Gotchas** — known footguns (e.g. `choice` scorer expects A/B/C/D labels; `model_graded_*` consumes extra tokens; ollama needs `ollama pull falcon3:1b` first).

Cross-references in the .md use the format `inspect-ai-tutorial.py § Cell NX` so navigation is mechanical.

## Key design choices

- **Default model `ollama_chat/falcon3:1b`** — matches existing logs, fastest cycle on M2. `falcon3:3b` referenced for multi-model comparison cells.
- **One `# %%` block per concept** — keeps each cell runnable in isolation in VSCode interactive mode.
- **Reuse prior logs** — Part 7 reads from existing `logs/*.eval` rather than generating new ones, saving runtime.
- **`limit=5` or `limit=10`** on every full-task eval cell to keep iteration fast; documented at the top of the .md.
- **No custom dataset download** — Part 2's hf_dataset cell uses `cais/wmdp` (already cached implicitly via inspect_evals). Toy local datasets are written inline in cells.
- **Tools section uses a stub tool**, not real network — keeps the tutorial offline-safe and biosecurity-appropriate (no real pathogen lookup).
- **Custom @scorer in Part 8C** is biosecurity-themed (refusal detection) — ties tutorial back to the user's project domain.

## Verification

1. Activate venv: `source venv/bin/activate`.
2. Confirm install: `python -c "import inspect_ai, inspect_evals; print(inspect_ai.__version__, inspect_evals.__version__)"` → expects `0.3.223 0.12.0`.
3. Confirm ollama running + model pulled: `ollama list | grep falcon3` (need at least `falcon3:1b`; `falcon3:3b` for multi-model cell).
4. Run cells top-to-bottom in VSCode interactive mode or `jupyter`. Each Part-1 cell completes in <2 min on M2. Parts 2–5 each <3 min with `limit=10`.
5. Part 6 tools cells: assert a tool call appears in the eval log (transcript shows `tool_call` event).
6. Part 7 log analysis: must successfully read existing `logs/2026-05-20T10-54-52*.eval` and print non-empty `results`.
7. Sanity check: `inspect view --log-dir logs/` opens the UI to browse generated tutorial runs alongside prior ones.

## Out of scope

- Real sandboxes (Docker/podman) — mentioned in Part 8D only.
- Real-world tool integration (web_search, computer use) — only @tool/basic_agent/react covered.
- MCP servers, deep agents, citations, multimodal content — referenced in the .md "next steps" list, no code cells.
- W&B / external logging integrations.
- Performance tuning / batch size optimization (covered in lm-eval tutorial already).
