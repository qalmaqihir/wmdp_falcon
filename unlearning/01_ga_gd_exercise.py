"""
01_ga_gd_exercise.py — Gradient Ascent & Gradient Difference Unlearning
=========================================================================
EXERCISE: Implement GA and GD unlearning on Falcon3-1B-Instruct.

LEARNING OBJECTIVES:
    1. Understand that GA/GD are architecture-INDEPENDENT: they only use
       the model's output cross-entropy loss — no internal state hooks needed.
    2. Implement GA (3 lines of logic) and see catastrophic forgetting in action.
    3. Implement GD (2 more lines) and see how a retain regularizer saves utility.
    4. Build familiarity with the training loop pattern you will reuse in RMU.

THEORY REFERENCE:
    See UNLEARNING_METHODS.md (project root) for math, intuition, and
    expected results for each method before reading this code.

BEFORE YOU START:
    Run the architecture exploration script first:
        python unlearning/00_architecture_exploration.py
    It shows Falcon3-1B's layer structure — not needed for GA/GD today,
    but builds the mental model you need for Exercise 02 (RMU).

YOUR TASKS:
    ┌─────────────────────────────────────────────────────────────┐
    │  TODO 1 (Section 2): Implement ga_loss()                    │
    │  TODO 2 (Section 3): Implement gd_loss()                    │
    │  TODO 3 (Section 4): Implement train_unlearning()           │
    └─────────────────────────────────────────────────────────────┘
    All other code is provided. Do NOT modify utils.py or config.py.

SETUP:
    cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
    source venv/bin/activate
    cd falcon_eval_wmdp
    export HUGGINGFACE_API_KEY=$HF_TOKEN   # optional — only needed for gated corpus

USAGE EXAMPLES:
    # Smoke test first (verifies all code paths, ~2 min):
    python unlearning/01_ga_gd_exercise.py --method ga --steps 5 --forget-size 20 --eval-samples 20

    # Full GA run — watch catastrophic forgetting (~10 min on M2):
    python unlearning/01_ga_gd_exercise.py --method ga --steps 100 --forget-size 200

    # Full GD run — watch the retain regularizer preserve utility (~20 min on M2):
    python unlearning/01_ga_gd_exercise.py --method gd --steps 200 --forget-size 200 --retain-size 200

    # Full GD + benchmark-grade final eval (adds ~15–20 min):
    python unlearning/01_ga_gd_exercise.py --method gd --steps 200 --full-eval

EXPECTED RESULTS (Falcon3-1B):
    Baseline WMDP-bio:   ~30-35%  (above 25% random — model has biosecurity knowledge)
    After GA  (100 steps): WMDP-bio → random, general output quality DEGRADES  ← lesson
    After GD  (200 steps): WMDP-bio drops, general capability PRESERVED        ← lesson

ARCHITECTURE-INDEPENDENCE NOTE:
    This file works identically on Falcon3, LLaMA, Mistral, GPT-2, or any
    AutoModelForCausalLM. No architecture-specific code anywhere.
    The only place architecture matters is in the optimizer choice — AdamW
    is universal — and in device placement, which utils.py handles.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Path bootstrap — makes both `experiments.config` and `unlearning.*` importable
# regardless of the working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.config import (
    HF_MODEL_ID,
    LR_GA, STEPS_GA,
    LR_GD, STEPS_GD, ALPHA_GD, BETA_GD,
    MAX_SEQ_LEN, BATCH_SIZE, EVAL_EVERY, EVAL_N_SAMPLES,
    UNLEARNING_RESULTS_DIR,
)
from unlearning.utils import (
    get_device,
    load_model_and_tokenizer,
    build_forget_loader,
    build_retain_loader,
    quick_wmdp_eval,
    full_wmdp_eval,
    save_checkpoint,
)


# ===========================================================================
# SECTION 0 — SETUP  (provided — do not modify)
# ===========================================================================

def setup(args) -> tuple:
    """Load model, build data loaders, initialise optimizer.

    Returns:
        (model, tokenizer, forget_loader, retain_loader, optimizer, device)
    """
    print("\n" + "=" * 60)
    print(f"  Exercise 01 — Unlearning method: {args.method.upper()}")
    print("=" * 60)
    print()
    print("  REMINDER: Run 00_architecture_exploration.py first if you")
    print("  haven't already — it explains why this file needs no")
    print("  architecture-specific code (unlike RMU in Exercise 02).")
    print()

    device = get_device()

    # Load Falcon3-1B in bf16. ~2 GB RAM, ~60 s on first load.
    model, tokenizer = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=False)

    # Forget set: biosecurity text (wmdp-corpora or WMDP MCQ fallback).
    forget_loader = build_forget_loader(
        tokenizer,
        n_samples=args.forget_size,
        max_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    # Retain set: general Wikitext-2 text (only used for GD).
    retain_loader = build_retain_loader(
        tokenizer,
        n_samples=args.retain_size,
        max_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    # AdamW is the standard optimizer for transformer fine-tuning.
    # GA/GD use the same optimizer as normal training — the difference
    # is only in the SIGN and COMPOSITION of the loss, not the optimizer.
    lr = args.lr if args.lr is not None else (LR_GA if args.method == "ga" else LR_GD)
    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"\n[setup] method={args.method}  steps={args.steps}  lr={lr:.2e}")
    print(f"[setup] forget_size={args.forget_size}  retain_size={args.retain_size}")
    print(f"[setup] alpha={args.alpha}  beta={args.beta}  (GD only)\n")

    return model, tokenizer, forget_loader, retain_loader, optimizer, device


# ===========================================================================
# SECTION 0b — DATA INSPECTION  (diagnostic helper)
# ===========================================================================

def print_loader_samples(loader, tokenizer, label: str, n: int = 4) -> None:
    """Decode and print n batches from a DataLoader for sanity-checking."""
    print(f"\n{'─' * 60}")
    print(f"  DATA SAMPLE — {label}")
    print(f"{'─' * 60}")
    for i, batch in enumerate(loader):
        if i >= n:
            break
        ids = batch["input_ids"][0]  # first item in batch, shape (seq_len,)
        # Strip padding (token id 0 or pad_token_id)
        pad_id = tokenizer.pad_token_id or 0
        ids_clean = ids[ids != pad_id]
        text = tokenizer.decode(ids_clean, skip_special_tokens=True)
        print(f"\n  [sample {i+1}]  tokens={len(ids_clean)}")
        # Print first 400 chars so terminal stays readable
        preview = text[:400].replace("\n", " ↵ ")
        print(f"  {preview}{'...' if len(text) > 400 else ''}")
    print(f"{'─' * 60}\n")


# ===========================================================================
# SECTION 1 — BASELINE EVAL  (provided — do not modify)
# ===========================================================================

def run_baseline(model, tokenizer, device, n_samples: int) -> dict:
    """Measure WMDP-bio accuracy before any unlearning."""
    print("\n[baseline] Measuring pre-unlearning WMDP-bio accuracy ...")
    result = quick_wmdp_eval(model, tokenizer, device, n_samples=n_samples)
    print(f"[baseline] WMDP-bio accuracy (n={n_samples}): {result['accuracy']:.1%}")
    print(f"           ({result['n_correct']}/{result['n_total']} correct, 25% = random chance)")
    return result


# ===========================================================================
# SECTION 2 — TODO 1: Gradient Ascent Loss
# ===========================================================================

def ga_loss(model: torch.nn.Module, forget_batch: dict) -> torch.Tensor:
    """
    Compute the Gradient Ascent (GA) loss on a batch of forget data.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  THEORY:                                                     │
    │    Normal training: minimize CE loss → model learns text.    │
    │    GA unlearning:   MAXIMIZE CE loss → model unlearns text.  │
    │    Maximizing loss = minimizing NEGATIVE loss.               │
    │    So: L_GA = -CE(model(x_forget), x_forget)                 │
    │                                                              │
    │  STEPS:                                                      │
    │    (1) Extract input_ids and labels from forget_batch.       │
    │        The batch is a dict with keys:                        │
    │            "input_ids"      — token ids, shape (B, seq_len)  │
    │            "labels"         — same as input_ids here         │
    │            "attention_mask" — 1 for real tokens              │
    │                                                              │
    │    (2) Run a forward pass:                                   │
    │            outputs = model(input_ids=...,                    │
    │                           attention_mask=...,                │
    │                           labels=...)                        │
    │        `outputs.loss` is the mean cross-entropy over tokens. │
    │                                                              │
    │    (3) FLIP THE SIGN. We want to MAXIMIZE this loss,         │
    │        which is equivalent to minimizing -loss.              │
    │                                                              │
    │    (4) Return the negated loss (a scalar tensor).            │
    │                                                              │
    │  HINT:                                                       │
    │    The PyTorch optimizer always MINIMIZES. So if you return  │
    │    `-outputs.loss`, calling .backward() + optimizer.step()   │
    │    will push the model AWAY from the forget text. ✓          │
    │                                                              │
    │    Architecture note: `outputs.loss` is identical for       │
    │    Falcon3, LLaMA, Mistral, GPT-2, etc. No model-specific   │
    │    code needed here.                                         │
    │                                                              │
    │  EXPECTED VALUE:                                             │
    │    Before unlearning: CE ≈ 2–4, so ga_loss ≈ -2 to -4.     │
    │    As training proceeds: CE grows, ga_loss becomes more      │
    │    negative. Watch the logged loss in the training loop.    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    """
    # ########## Complete the Implementation ##########
    try:
        
        # Step 1: extract tensors from the batch dict
        input_ids = forget_batch["input_ids"]
        labels = forget_batch["labels"]
        attention_mask = forget_batch["attention_mask"]

        # Step 2: forward pass through the model with labels
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Step 3: get the standard language modeling loss
        loss = outputs.loss

        # Step 4: negate and return (maximise = minimise negative)
        return -loss
    except Exception as e:
        print(f"Exception: {str(e)}\n")
        raise NotImplementedError(
            "TODO 1: implement ga_loss(). ~3 lines. See docstring above."
        )
    # ##################################################


# ===========================================================================
# SECTION 3 — TODO 2: Gradient Difference Loss
# ===========================================================================

def gd_loss(
    model: torch.nn.Module,
    forget_batch: dict,
    retain_batch: dict,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute the Gradient Difference (GD) loss.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  THEORY:                                                     │
    │    GD = GA (forget) + retain regularizer.                    │
    │    L_GD = -alpha * CE(forget) + beta * CE(retain)            │
    │                                                              │
    │    The forget term pushes model away from biosecurity text.  │
    │    The retain term PULLS model back toward general text.     │
    │    The retain loss acts as a regularizer that prevents       │
    │    catastrophic forgetting of general language capability.   │
    │                                                              │
    │  STEPS:                                                      │
    │    (1) Compute the forget term.                              │
    │        Option A (recommended): call ga_loss(model, forget_batch)   │
    │            This already returns -CE(forget) (negated). ✓     │
    │            To apply alpha: forget_term = alpha * ga_loss(...) │
    │        Option B: inline the logic from ga_loss.              │
    │                                                              │
    │    (2) Compute the retain term.                              │
    │        Standard FORWARD (not ascent) on retain_batch.        │
    │            outputs = model(input_ids=..., labels=..., ...)   │
    │            retain_term = outputs.loss   # standard CE        │
    │        Note: this is POSITIVE — we want to MINIMIZE it.      │
    │                                                              │
    │    (3) Combine:                                              │
    │            total = forget_term + beta * retain_term          │
    │        forget_term is already NEGATIVE (from ga_loss).       │
    │        retain_term is POSITIVE.                              │
    │        The optimizer minimizes total, so:                    │
    │            - forget part: pushes loss up on forget data ✓    │
    │            - retain part: keeps loss low on retain data ✓    │
    │                                                              │
    │    (4) Return total.                                         │
    │                                                              │
    │  HINT — sign trap:                                           │
    │    ga_loss() already returns -CE(forget).                    │
    │    Do NOT negate again. The formula is:                      │
    │        total = alpha * ga_loss(forget) + beta * retain_loss  │
    │    NOT:                                                      │
    │        total = -alpha * ga_loss(forget) + beta * retain_loss │
    │                                                              │
    │  EXPECTED BEHAVIOUR:                                         │
    │    GD loss should oscillate (forget term pulls negative,     │
    │    retain term pulls positive). It should NOT grow           │
    │    monotonically negative like pure GA does.                 │
    │    This is the signal that the regularizer is working. ✓     │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    """
    # ########## Complete the Implementation ##########
    try:
        
        # Step 1: compute the forget term (use ga_loss or inline)
        # forget_term = alpha * ga_loss(model, forget_batch)
        
        forget_term = alpha * ga_loss(model, forget_batch)

        # Step 2: compute the retain term (standard CE, no negation)
        # retain_outputs = model(...)
        # retain_term = retain_outputs.loss
        
        input_ids_retain = retain_batch["input_ids"]
        labels_retain = retain_batch["labels"]
        attention_mask_retain = retain_batch["attention_mask"]
        
        retain_outptus = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain, labels=labels_retain)
        retain_term = retain_outptus.loss

        # Step 3: combine and return
        # return forget_term + beta * retain_term
        
        return forget_term + beta*retain_term
    
    except Exception as e:
        print(f"Exception: {str(e)}\n")
        raise NotImplementedError(
            "TODO 2: implement gd_loss(). ~5 lines. See docstring above."
        )
    # ##################################################


# ===========================================================================
# SECTION 4 — TODO 3: Training Loop
# ===========================================================================

def train_unlearning(
    method: str,
    model: torch.nn.Module,
    forget_loader,
    retain_loader,
    optimizer,
    steps: int,
    tokenizer,
    device: torch.device,
    alpha: float,
    beta: float,
) -> list[tuple]:
    """
    Training loop for GA or GD unlearning.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  STEPS:                                                      │
    │    (1) Set model to training mode: model.train()             │
    │                                                              │
    │    (2) Create infinite iterators over the loaders.           │
    │        DataLoaders are finite (they stop after one pass).    │
    │        Use itertools.cycle to loop them indefinitely:        │
    │            from itertools import cycle                       │
    │            forget_iter = cycle(forget_loader)                │
    │            retain_iter = cycle(retain_loader)                │
    │                                                              │
    │    (3) Loop for `steps` iterations:                          │
    │        a) Get next forget batch: batch_f = next(forget_iter) │
    │        b) Move tensors to device:                            │
    │               batch_f = {k: v.to(device) for k, v in ...}   │
    │        c) If method == "ga":                                 │
    │               loss = ga_loss(model, batch_f)                 │
    │           If method == "gd":                                 │
    │               batch_r = next(retain_iter)                    │
    │               batch_r = {k: v.to(device) for k, v in ...}   │
    │               loss = gd_loss(model, batch_f, batch_r,        │
    │                              alpha=alpha, beta=beta)         │
    │                                                              │
    │        d) Backward + step + zero_grad:                       │
    │               loss.backward()                                │
    │               optimizer.step()                               │
    │               optimizer.zero_grad()                          │
    │                                                              │
    │        e) Print progress every step:                         │
    │               print(f"Step {step+1}/{steps}  loss={loss.item():.4f}")  │
    │                                                              │
    │        f) Every EVAL_EVERY steps, run quick eval:            │
    │               model.eval()                                   │
    │               r = quick_wmdp_eval(model, tokenizer, device,  │
    │                                   n_samples=EVAL_N_SAMPLES)  │
    │               model.train()                                  │
    │                                                              │
    │    (4) Collect history. Return a list of tuples:             │
    │            [(step, loss_value, wmdp_acc_or_None), ...]       │
    │        Append (step+1, loss.item(), None) on normal steps.   │
    │        Append (step+1, loss.item(), r['accuracy']) on eval.  │
    │                                                              │
    │  NOTES:                                                      │
    │    - `loss.item()` detaches the tensor to a Python float.    │
    │      Call it BEFORE .backward() or after — both work.        │
    │    - zero_grad() placement: call it at the TOP of the loop   │
    │      (before backward) OR at the BOTTOM (after step). Both   │
    │      work. NEVER call it between backward() and step() —     │
    │      that wipes the gradients you just computed.             │
    │    - Remember: for GA, retain_iter is created but never used.│
    │      That's fine — it just cycles in the background.         │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        method:        "ga" or "gd"
        model:         The model being unlearned.
        forget_loader: DataLoader of forget (biosecurity) text.
        retain_loader: DataLoader of retain (general) text.
        optimizer:     AdamW optimizer.
        steps:         Number of gradient steps.
        tokenizer:     For eval.
        device:        Target device.
        alpha:         Weight on forget term (GD only).
        beta:          Weight on retain term (GD only).

    Returns:
        List of (step, loss_value, wmdp_acc_or_None) tuples.
    """
    # ########## Complete the Implementation ##########

    # Step 1: set model to training mode
    model.train()

    # Step 2: create infinite iterators
    from itertools import cycle
    forget_iter = cycle(forget_loader)
    retain_iter = cycle(retain_loader)
    
    # forget_iter = cycle(forget_loader)
    # retain_iter = cycle(retain_loader)

    # Step 3: training loop
    # history = []
    # for step in range(steps):
    #     ... (see docstring above for sub-steps)
    #     history.append(...)
    try:
        history = []
        for step in range(steps):
            batch_f = next(forget_iter)
            batch_f ={k:v.to(device) for k,v in batch_f.items()}
            if method == "ga":
                loss = ga_loss(model, batch_f)
            if method == "gd":
                batch_r = next(retain_iter)
                batch_r = {k:v.to(device) for k, v in batch_r.items()}
                loss = gd_loss(model, batch_f, batch_r, alpha=alpha, beta=beta)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"[Training] Step {step+1}/{steps}  loss={loss.item():.4f}")
            
            if (step + 1) % EVAL_EVERY == 0:
                model.eval()
                r = quick_wmdp_eval(model, tokenizer, device, n_samples=EVAL_N_SAMPLES)
                model.train()
                print(f"[Eval] Step {step+1}/{steps}  loss={loss.item():.4f} accuracy={r['accuracy']:.4f}\n")
                history.append((step + 1, loss.item(), r['accuracy']))
            else:
                history.append((step + 1, loss.item(), None))
            
        # Step 4: return history
        return history
    except Exception as e:
        print(f"Exception: {str(e)}")
        raise NotImplementedError(
            "TODO 3: implement train_unlearning(). ~25 lines. See docstring above."
        )
    # ##################################################


# ===========================================================================
# SECTION 5 — POST-UNLEARNING EVAL + SAVE  (provided — do not modify)
# ===========================================================================

def post_unlearning_eval_and_save(
    model,
    tokenizer,
    device,
    pre_result: dict,
    history: list,
    args,
) -> None:
    """Measure accuracy after unlearning, print delta, optionally run full eval."""
    print("\n[post-eval] Measuring post-unlearning WMDP-bio accuracy ...")
    post_result = quick_wmdp_eval(model, tokenizer, device, n_samples=args.eval_samples)

    pre_acc  = pre_result["accuracy"]
    post_acc = post_result["accuracy"]
    delta    = post_acc - pre_acc

    print(f"\n{'=' * 60}")
    print(f"  UNLEARNING RESULTS — method={args.method.upper()}")
    print(f"{'=' * 60}")
    print(f"  Pre-unlearning  WMDP-bio: {pre_acc:.1%}")
    print(f"  Post-unlearning WMDP-bio: {post_acc:.1%}")
    print(f"  Delta                   : {delta:+.1%}")
    print(f"  Random chance           : 25.0%")
    if delta < -0.03:
        print(f"  → Forgetting observed ✓  (WMDP-bio accuracy decreased)")
    else:
        print(f"  → Minimal forgetting (try more steps or higher learning rate)")
    print(f"{'=' * 60}\n")

    # Optional: full benchmark-grade eval (1273 samples, ~15–20 min on M2).
    full_result = None
    if args.full_eval:
        print("[full-eval] Running full WMDP-bio eval ...")
        full_result = full_wmdp_eval(model, tokenizer, device)

    # Build metadata dict for checkpoint.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "method":           args.method,
        "steps":            args.steps,
        "lr":               args.lr,
        "alpha":            args.alpha,
        "beta":             args.beta,
        "forget_size":      args.forget_size,
        "retain_size":      args.retain_size,
        "pre_wmdp_acc":     pre_acc,
        "post_wmdp_quick":  post_acc,
        "delta_wmdp":       delta,
        "post_wmdp_full":   full_result["accuracy"] if full_result else None,
        "timestamp":        timestamp,
        "model_id":         HF_MODEL_ID,
    }

    # Log the mid-training eval trajectory.
    eval_points = [(s, l, a) for (s, l, a) in history if a is not None]
    if eval_points:
        print("  Mid-training WMDP-bio trajectory:")
        for (step, loss, acc) in eval_points:
            print(f"    step {step:>4}  loss={loss:.4f}  wmdp={acc:.1%}")

    # Save checkpoint + metadata.
    out_dir = UNLEARNING_RESULTS_DIR / f"{args.method}_{timestamp}"
    save_checkpoint(model, tokenizer, out_dir, metadata)
    print(f"\n[saved] Checkpoint + metadata at: {out_dir}")

    # Save full training history as CSV for plotting.
    history_path = out_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "wmdp_acc"])
        for step, loss_val, acc in history:
            writer.writerow([step, loss_val, "" if acc is None else acc])
    print(f"[saved] Training history at: {history_path}")


# ===========================================================================
# SECTION 6 — CLI + MAIN  (provided — do not modify)
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GA/GD Unlearning Exercise — Falcon3-1B on WMDP-bio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method", choices=["ga", "gd"], default="ga",
        help="Unlearning method: gradient ascent (ga) or gradient difference (gd).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Number of gradient steps. Defaults: GA=100, GD=200.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate. Defaults: GA=5e-6, GD=1e-5.",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA_GD,
        help="Weight on forget term (GD only). Higher = more aggressive forgetting.",
    )
    parser.add_argument(
        "--beta", type=float, default=BETA_GD,
        help="Weight on retain term (GD only). Higher = stronger utility preservation.",
    )
    parser.add_argument(
        "--forget-size", type=int, default=200,
        help="Number of forget sequences to load.",
    )
    parser.add_argument(
        "--retain-size", type=int, default=200,
        help="Number of retain sequences to load (used by GD only).",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=EVAL_N_SAMPLES,
        help="WMDP-bio samples for quick pre/post eval.",
    )
    parser.add_argument(
        "--full-eval", action="store_true",
        help="After training, run full WMDP-bio eval on all 1273 samples (~20 min).",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip pre-unlearning eval (saves ~30 s during development).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve default steps.
    if args.steps is None:
        args.steps = STEPS_GA if args.method == "ga" else STEPS_GD

    # Section 0: setup.
    model, tokenizer, forget_loader, retain_loader, optimizer, device = setup(args)

    # Section 0b: data sanity check — decode 2 samples from each loader.
    print_loader_samples(forget_loader, tokenizer, "FORGET (biosecurity)")
    print_loader_samples(retain_loader, tokenizer, "RETAIN (Wikitext-2)")

    # Section 1: baseline eval.
    if args.skip_baseline:
        pre_result = {"accuracy": float("nan"), "n_correct": 0, "n_total": 0}
        print("[baseline] Skipped (--skip-baseline).")
    else:
        pre_result = run_baseline(model, tokenizer, device, n_samples=args.eval_samples)

    # Sections 2–4: run unlearning (student implements these).
    print(f"\n[train] Starting {args.method.upper()} unlearning  ({args.steps} steps) ...")
    t0 = time.perf_counter()

    history = train_unlearning(
        method=args.method,
        model=model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        optimizer=optimizer,
        steps=args.steps,
        tokenizer=tokenizer,
        device=device,
        alpha=args.alpha,
        beta=args.beta,
    )

    elapsed = time.perf_counter() - t0
    print(f"[train] Done in {elapsed/60:.1f} min ({elapsed/args.steps:.1f} s/step).")

    # Section 5: post-unlearning eval + save.
    post_unlearning_eval_and_save(model, tokenizer, device, pre_result, history, args)


if __name__ == "__main__":
    main()
