# inspect-ai Comprehensive Tutorial — Narrative Guide

Companion to [inspect-ai-tutorial.py](inspect-ai-tutorial.py).
Every section maps to a `# %%` cell in that file (e.g. **Cell 1A**).
Run with: `source venv/bin/activate`, then open the .py in VSCode Interactive mode.

**Prerequisites**
- `ollama serve` running in a terminal
- `ollama pull falcon3:1b` (and `falcon3:3b`, `falcon3:7b` for multi-model cells)
- `pip install "lm_eval[hf]"` already done (venv confirmed)
- All `limit=5` / `limit=10` guards in cells — remove for full benchmark runs

---

## Part 1 — Foundations: Task, Eval, CLI

### What it covers
The four core concepts that every inspect-ai eval is built from:

| Concept | Role | Key types |
|---------|------|-----------|
| **Task** | Container tying everything together | `Task`, `@task` |
| **Dataset** | Questions + expected answers | `Sample`, `MemoryDataset`, `hf_dataset` |
| **Solver** | How to get a model response | `generate`, `multiple_choice`, `chain_of_thought` |
| **Scorer** | How to judge the response | `choice`, `exact`, `model_graded_fact` |

### Cell 1A — Hello World
`eval(wmdp_bio(), model="ollama/falcon3:1b", limit=5)` is the minimum viable eval.
Returns `list[EvalLog]` — one entry per model (here, just one).

### Cell 1B — EvalLog anatomy
```
EvalLog
  .eval      → EvalSpec  (task name, model, config)
  .status    → "success" | "error" | "cancelled"
  .results   → EvalResults
                .scores  → list[EvalScore]
                              .name     scorer name
                              .metrics  dict[str, EvalMetric]
  .stats     → EvalStats  (token usage per model)
  .samples   → list[EvalSample]
                .id / .input / .target / .scores / .transcript
```
`log.results.scores[0].metrics["accuracy"].value` is the number you care about.

### Cell 1C — CLI
```bash
inspect eval inspect_evals/wmdp_bio --model ollama/falcon3:1b --limit 5 --log-dir logs/
inspect view --log-dir logs/          # opens web UI at http://localhost:7575
```
The web UI lets you browse every sample, model input/output, and score.

### Cell 1D — Provider swap
Only the model string changes. `hf/` uses the local HuggingFace transformers stack:
```python
get_model("hf/tiiuae/Falcon3-1B-Instruct", device="mps", dtype="float16")
```
**Gotcha:** `hf/` is significantly slower than `ollama/` on M2 for inference — use ollama for fast iteration, hf/ when you need exact logits or reproducibility.

---

## Part 2 — Dataset Deep Dive

### What it covers
Every way to load data: in-memory, HuggingFace Hub, CSV, JSON.

### Cell 2A — Sample + MemoryDataset
`Sample` is the atomic unit. Key fields:

| Field | Type | Notes |
|-------|------|-------|
| `input` | `str \| list[ChatMessage]` | The question / prompt |
| `target` | `str \| list[str]` | Expected answer. For MCQ: `"B"`, `"C"`, etc. |
| `choices` | `list[str] \| None` | MCQ options; letter assigned by position |
| `id` | `str \| int \| None` | Used in transcripts and logs |
| `metadata` | `dict \| None` | Arbitrary key-value pairs; survives into EvalSample |

`MemoryDataset(samples, name="...")` wraps a list for use in Task.

### Cell 2B — hf_dataset + RecordToSample
`RecordToSample` is a `Callable[[dict], Sample]` that maps raw HF fields to Sample fields.
Use it whenever field names differ from inspect-ai defaults (`input`, `target`, `choices`).

```python
# cais/wmdp stores answer as int 0-3; convert to letter
target=chr(ord("A") + record["answer"])
```

`FieldSpec` is the simpler alternative when field names just need renaming (no logic).

**Gotcha:** `choice()` scorer expects `"A"` / `"B"` / `"C"` / `"D"` strings as target, not integers.

### Cell 2C — csv_dataset / json_dataset
Both expect rows with at minimum `input` and `target` columns. Extra columns become metadata if `FieldSpec(metadata=["col1", "col2"])` is used.

---

## Part 3 — Solver Deep Dive

### What it covers
Solvers are the compute step: they take a `TaskState` and return a modified `TaskState`.
They compose: `solver=[s1, s2, s3]` is a sequential pipeline.

### Cell 3A — generate + system_message
- `system_message(text)` prepends a system turn to the conversation.
- `generate()` calls the model and stores the completion in `state.output`.
Together they form the simplest possible solver chain.

### Cell 3B — chain_of_thought + prompt_template
- `chain_of_thought()` wraps the current prompt in a "Think step by step..." scaffold before calling generate.
- `prompt_template(template_str)` substitutes `{prompt}` with the current question text.
Use CoT when accuracy matters and latency is acceptable. Expect 2–4× more tokens.

### Cell 3C — multiple_choice
`multiple_choice()` is the canonical MCQ solver. It:
1. Formats `Sample.choices` as `(A) ... (B) ... (C) ... (D) ...`
2. Calls `generate()`
3. Expects `choice()` as the scorer

**Gotcha:** `multiple_choice()` already calls `generate()` internally. Don't add a separate `generate()` after it.

### Cell 3D — chain, fork, self_critique
- `chain([s1, s2])` = `[s1, s2]` passed as a list — same thing, explicit name.
- `fork([s1, s2])` runs both branches on the same initial state in parallel; results merge.
- `self_critique(critique_template, completion_template)` runs three model calls: initial answer → critique → revised answer. Useful for calibration studies.

**Gotcha:** `self_critique` triples token consumption. Use `limit=2` when testing.

---

## Part 4 — Scorer Deep Dive

### What it covers
How to judge model outputs — from simple string matching to LLM-as-judge to custom logic.

### Cell 4A — Built-in scorers

| Scorer | When to use |
|--------|-------------|
| `choice()` | MCQ with A/B/C/D targets |
| `exact()` | Short answers requiring exact string equality |
| `match()` | Case-insensitive; ignores punctuation |
| `includes()` | Target is a substring of the output |
| `pattern(regex)` | Target is a regex pattern |
| `f1()` | Token-overlap F1; standard for extractive QA |

### Cell 4B — model_graded_fact / model_graded_qa
Use an LLM to judge free-text answers. `model_graded_fact` is for factual correctness; `model_graded_qa` is for answer quality.

```python
scorer=model_graded_fact(model="ollama/falcon3:7b")
```

**Gotcha:** This doubles your inference cost (two models run). Use `limit=10` when testing. The grader model's own biases affect scores — use the strongest available model as grader.

### Cell 4C — Custom @scorer + @metric
```python
@scorer(metrics=[accuracy(), refusal_rate()])
def my_scorer():
    async def score(state: TaskState, target) -> Score:
        ...
        return Score(value=CORRECT, explanation="...", metadata={...})
    return score
```

Score constants: `CORRECT="C"`, `INCORRECT="I"`, `PARTIAL="P"`, `NOANSWER="N"`.
`metadata` dict survives into `EvalSample.scores[scorer_name].metadata` — useful for post-hoc filtering.

### Cell 4D — Score reducers
Used with `Epochs` (Part 8A):

| Reducer | Formula |
|---------|---------|
| `mean_score()` | mean(scores across epochs) |
| `pass_at(k)` | 1.0 if any of k runs = CORRECT |
| `at_least(n, k)` | 1.0 if ≥ n of k runs = CORRECT |

---

## Part 5 — Model Configuration

### What it covers
Direct model access, generation parameters, multi-model runs.

### Cell 5A — get_model + GenerateConfig
`get_model(model_string)` returns a `Model` object you can call directly outside of any eval.
Useful for prototyping prompts before committing to a full eval run.

```python
config = GenerateConfig(
    temperature=0.0,    # 0 = greedy (deterministic)
    max_tokens=256,
    top_p=0.9,
    seed=42,
)
output = asyncio.run(model.generate(messages, config=config))
```

`output.completion` — the text response.
`output.stop_reason` — `"stop"` | `"length"` | `"tool_calls"`.
`output.usage` — `ModelUsage(input_tokens, output_tokens, total_tokens)`.

### Cell 5B — ChatMessage types
Build conversation history manually for few-shot or multi-turn scenarios:

```python
ChatMessageSystem(content="...")     # system turn
ChatMessageUser(content="...")       # user turn
ChatMessageAssistant(content="...")  # injected assistant turn (few-shot)
ChatMessageTool(...)                 # tool result turn
```

### Cell 5C — Multi-model comparison
```python
eval(task, model=["ollama/falcon3:1b", "ollama/falcon3:3b"])
```
Returns `list[EvalLog]` with one entry per model. Useful for side-by-side benchmarking.

### Cell 5D — eval_set
```python
success, logs = eval_set(tasks=[t1, t2], model=["model1"], limit=3)
```
Best for automated grids (multiple tasks × multiple models × multiple configs). Returns `(bool, list[EvalLog])`.

---

## Part 6 — Tools & Agents

### What it covers
Giving the model callable tools, and running autonomous agentic loops.

### Cell 6A — @tool decorator
```python
@tool
def my_tool():
    async def execute(param: str) -> str:
        """Docstring becomes the model-visible description."""
        return "result"
    return execute
```
The inner function must be `async`. Type annotations become the JSON schema. The docstring's `Args:` section documents each parameter for the model.

**Gotcha:** The outer function is the factory; the inner is the implementation. Always follow this two-level pattern.

### Cell 6B — use_tools
```python
solver=[use_tools(my_tool()), generate()]
```
`use_tools` registers tools so the model can call them. `generate()` then runs the model — if it emits a tool call, inspect-ai executes the tool and loops.

### Cell 6C — basic_agent
```python
basic_agent(
    init=[system_message("...")],
    tools=[my_tool()],
    message_limit=6,      # safety cutoff
)
```
Autonomous loop: generate → (tool call → execute tool → generate) × N → submit.
`message_limit` prevents runaway loops. **Always set it** in evaluation contexts.

### Cell 6D — react agent
ReAct-style (Reason + Act). Provides a richer prompting scaffold than `basic_agent`.
```python
from inspect_ai.agent import react, as_solver
agent = react(tools=[my_tool()])
solver = as_solver(agent)   # makes it usable in Task.solver
```

Other agent utilities:
- `handoff(other_agent)` — delegate to a sub-agent and return its result
- `as_tool(agent)` — expose an agent as a callable tool for another agent
- `bridge(fn)` — wrap a sync function as a solver

---

## Part 7 — Log Analysis

### What it covers
Reading, aggregating, and exporting eval results from `.eval` files.

### Cell 7A — read_eval_log
`.eval` files are ZIP archives with structured JSON inside. `read_eval_log(path)` decompresses automatically.

Key navigation pattern:
```python
log = read_eval_log("logs/my_run.eval")
log.results.scores[0].metrics["accuracy"].value   # top-level metric
log.samples[i].scores["choice"].value             # per-sample score
log.stats.model_usage["ollama/..."].input_tokens
```

### Cell 7B — list_eval_logs
```python
infos = list_eval_logs("logs/")   # returns list[EvalLogInfo]
```
`EvalLogInfo` has `.name` (path), `.task`, `.model`, `.eval_id`. Use it to loop over all runs without loading full samples.

### Cell 7C — Transcript
`sample.transcript.events` is the full audit trail: model calls, tool calls, scores.
Event types: `"model"`, `"tool"`, `"score"`, `"error"`, `"info"`.
Add `--log-level debug` to `inspect eval` CLI to capture full transcripts.

```python
from inspect_ai.log import bundle_log_dir
bundle_log_dir("logs/", "logs/bundle.html")   # portable single-file viewer
```

---

## Part 8 — Advanced Patterns

### What it covers
Repeated trials, resource limits, custom biosecurity tasks, approval gates, sandboxes.

### Cell 8A — Epochs + reducers
```python
epochs=Epochs(3, reducer=[mean_score(), pass_at(2)])
```
Runs each sample 3 times. `pass_at(2)` scores 1.0 if at least one of 3 runs is correct.
Standard for stochastic tasks (code generation, open-ended reasoning) where single-run scores are noisy.

### Cell 8B — Limits
```python
with token_limit(8000):
    with time_limit(120):
        logs = eval(...)
```
Limits abort individual samples, not the whole eval. Aborted samples show `"error"` status in the transcript and score `NOANSWER`. The eval itself still completes and reports metrics over non-aborted samples.

Available limits: `token_limit(n)`, `time_limit(seconds)`, `message_limit(n)`.

### Cell 8C — Custom biosecurity task
Full end-to-end pattern for a novel eval:
1. Handcraft `MemoryDataset` with `target="REFUSED"` or `target="ANSWERED"`.
2. Define `@metric` for custom aggregation.
3. Define `@scorer` with async scoring logic.
4. Wrap in `@task` and `eval()`.

This is the template for any bespoke AI safety measurement.

### Cell 8D — Approval + Sandboxes
**Approval** gates dangerous tool calls for human review:
```python
eval(task, approval=[ApprovalPolicy(approver=human_approver(), tools=["bash"])])
```
Use `auto_approver()` in automated test pipelines; `human_approver()` in production safety evals.

**Sandboxes** isolate model-generated code in Docker containers:
```python
sandbox("docker")  # passed as Task.sandbox
```
Required for code-execution benchmarks (HumanEval, SWE-bench, etc.).
See: https://inspect.aisi.org.uk/sandboxing.html

---

## Quick Reference

```python
# Minimum eval
from inspect_ai import eval, task, Task
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice
from inspect_ai.dataset import Sample, MemoryDataset

@task
def my_task():
    return Task(
        dataset=MemoryDataset([Sample(input="Q?", choices=["A","B","C","D"], target="A")]),
        solver=multiple_choice(),
        scorer=choice(),
    )

eval(my_task(), model="ollama/falcon3:1b", limit=5, log_dir="logs/")
```

```bash
# CLI
inspect eval inspect_evals/wmdp_bio --model ollama/falcon3:1b --limit 20 --log-dir logs/
inspect view --log-dir logs/
```

## Things not covered here (next steps)

- **MCP servers** (`mcp_tools`, `mcp_server_stdio`) — connect tools to external services
- **Multimodal** (`ContentImage`, `ContentAudio`) — vision/audio benchmarks
- **Deep agents** (`deepagent`, `subagent`, `handoff`) — hierarchical multi-agent pipelines
- **Citations** (`Citation`, `UrlCitation`) — grounding evals
- **W&B / HF Hub logging** — result tracking at scale
- **Custom sandbox environments** — Docker, EC2, k8s
- **Compaction** (`CompactionStrategy`) — managing long context in agent runs
