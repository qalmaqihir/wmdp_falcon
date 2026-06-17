"""
05_quantized_rmu_exercise.py — LoRA + RMU on Falcon3-1B
=========================================================
EXERCISE: Apply RMU on top of a LoRA-wrapped Falcon3-1B model.

LEARNING OBJECTIVES:
    1. Understand how PEFT's LoRA wrapper changes the layer path for hooks.
    2. Implement hook registration on a PEFT-wrapped model (layer path shifts
       by one .base_model level compared to the bare model in Exercise 03).
    3. Assemble the LoRA-RMU training loop — same MSE loss math, smaller
       optimizer scope (LoRA params only).

THEORY REFERENCE:
    See UNLEARNING_METHODS.md (project root) for RMU math.
    See 04_quantization_exploration.py for LoRA mechanics and device matrix.

BEFORE YOU START:
    Run the exploration scripts first:
        python unlearning/02_rmu_exploration.py        # RMU primitives
        python unlearning/03_rmu_exercise.py ...       # plain RMU (complete it first)
        python unlearning/04_quantization_exploration.py  # LoRA mechanics

YOUR TASKS:
    ┌─────────────────────────────────────────────────────────────┐
    │  TODO 1 (Section 3): register_rmu_hooks_for_lora_model()   │
    │  TODO 2 (Section 5): train_qlora_rmu()                     │
    └─────────────────────────────────────────────────────────────┘
    All other code is provided. Do NOT modify utils.py or config.py.

SETUP:
    pip install peft     # required for this exercise
    cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
    source venv/bin/activate
    cd falcon_eval_wmdp

USAGE EXAMPLES:
    # Smoke test (bf16+LoRA, ~2 min — fewer trainable params = faster):
    python unlearning/05_quantized_rmu_exercise.py --backend bf16-lora --steps 5 --forget-size 20 --eval-samples 20

    # Full LoRA+RMU run (~15 min — much faster than full-model RMU):
    python unlearning/05_quantized_rmu_exercise.py --backend bf16-lora --steps 300

    # Reference: 4-bit / 8-bit backends (REQUIRES CUDA — will exit on MPS):
    python unlearning/05_quantized_rmu_exercise.py --backend 4bit-lora --steps 5
    python unlearning/05_quantized_rmu_exercise.py --backend 8bit-lora --steps 5

EXPECTED RESULTS (bf16+LoRA, Falcon3-1B, M2 MPS):
    print_trainable_parameters() → ~2M trainable / ~1B total (~0.2%)
    Baseline WMDP eval: ~30–35%
    After 300 steps: WMDP eval should drop toward 25%
    Memory peak: ~4 GB (LoRA adapter only, no second full-model copy)
    Adapter save: ~10 MB (vs ~2 GB for full model checkpoint)
"""

import argparse
import sys
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.config import (
    HF_MODEL_ID,
    RMU_LAYER, LR_RMU, STEPS_RMU, ALPHA_RMU, BETA_RMU,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
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

# ---------------------------------------------------------------------------
# Loss primitives — identical to 03_rmu_exercise.py.
# Inlined here because Python module names cannot start with a digit.
# If you update the logic in 03_rmu_exercise.py, mirror the change here.
# ---------------------------------------------------------------------------

def build_rmu_random_vector(hidden_size: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randn(hidden_size, generator=gen)
    c = c / c.norm()
    return c.to(device)


def rmu_forget_loss(hidden: torch.Tensor, random_vec: torch.Tensor, alpha: float) -> torch.Tensor:
    import torch.nn.functional as F
    rv = random_vec.to(hidden.dtype)
    target = (alpha * rv).expand_as(hidden)
    return F.mse_loss(hidden, target)


def rmu_retain_loss(hidden_live: torch.Tensor, hidden_frozen: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    return F.mse_loss(hidden_live, hidden_frozen)


# ===========================================================================
# SECTION 0 — CUDA-only guard  (provided — do not modify)
# ===========================================================================

def _require_cuda(backend: str, device: torch.device) -> None:
    """Exit with a clear message if a CUDA-only backend is requested on MPS/CPU."""
    if device.type != "cuda":
        print(f"\n[backend] '{backend}' requires CUDA (bitsandbytes CUDA kernels).")
        print(f"  Current device: {device}")
        print(f"  On M2 Mac, use: --backend bf16-lora  (runs on MPS in full float)")
        print(f"  To use 4-bit/8-bit, run on a CUDA GPU (Linux/Windows).")
        print(f"\n  Reference code for {backend} is included below for study purposes.")
        sys.exit(0)


# ===========================================================================
# SECTION 1 — Quantized backbone builders  (provided — reference code only)
# ===========================================================================

def _build_4bit_lora_model(device: torch.device):
    """
    Load Falcon3-1B in 4-bit NF4 and wrap with LoRA.

    REQUIRES CUDA — bitsandbytes NF4 kernels are not implemented for MPS.
    This function is provided as reference. It will not be called on MPS.

    Config reference:
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    """
    _require_cuda("4bit-lora", device)  # exits on non-CUDA

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model, tokenizer


def _build_8bit_lora_model(device: torch.device):
    """
    Load Falcon3-1B in 8-bit INT8 and wrap with LoRA.

    REQUIRES CUDA — bitsandbytes INT8 kernels are not implemented for MPS.
    """
    _require_cuda("8bit-lora", device)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model, tokenizer


def _build_bf16_lora_model(device: torch.device):
    """
    Load Falcon3-1B in bf16 and wrap with LoRA. Runs on MPS.

    This is the M2 Mac path. The base model weights are full bf16 float
    (not quantized). LoRA adapters add ~2M trainable params on top.
    Training loop is IDENTICAL to QLoRA — only the base dtype differs.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("[error] peft not installed. Run: pip install peft")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=False)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)
    return peft_model, tokenizer


# ===========================================================================
# SECTION 2 — Build LoRA model (dispatcher)  (provided — do not modify)
# ===========================================================================

def build_lora_model(backend: str, device: torch.device):
    """
    Dispatch to the correct builder based on --backend arg.

    Returns:
        (peft_model, tokenizer)
        peft_model.base_model.model.model.layers[L] is the hook target.
    """
    builders = {
        "bf16-lora": _build_bf16_lora_model,
        "4bit-lora":  _build_4bit_lora_model,
        "8bit-lora":  _build_8bit_lora_model,
    }
    if backend not in builders:
        raise ValueError(f"Unknown backend '{backend}'. Choose: {list(builders)}")

    print(f"[backend] Building {backend} model ...")
    peft_model, tokenizer = builders[backend](device)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in peft_model.parameters())
    print(f"[backend] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.0f}M total ({trainable/total*100:.3f}%)")

    return peft_model, tokenizer


# ===========================================================================
# SECTION 3 — TODO 1: Hook registration for PEFT model
# ===========================================================================

def register_rmu_hooks_for_lora_model(
    peft_model,
    frozen_model,
    layer_idx: int,
):
    """
    Register RMU forward hooks on both the PEFT-wrapped live model and the
    frozen reference model. Return the storage dicts and hook handles.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  EDUCATIONAL POINT:                                          │
    │    PEFT wraps the model like this:                           │
    │        PeftModelForCausalLM                                  │
    │          └── base_model : PeftModel                          │
    │               └── model : FalconForCausalLM (original model) │
    │                    └── model : FalconModel                   │
    │                         └── layers : ModuleList             │
    │                              └── [L] : LlamaDecoderLayer    │
    │                                                              │
    │    So the path is:                                           │
    │        peft_model.base_model.model.model.layers[layer_idx]  │
    │    (one extra .base_model compared to plain model.model)     │
    │                                                              │
    │    For the frozen reference model (NOT wrapped by PEFT):     │
    │        frozen_model.model.layers[layer_idx]                  │
    │    (same path as in Exercise 03)                             │
    │                                                              │
    │  STEPS (~10 lines):                                          │
    │    1. Create live_store = {} and frozen_store = {}           │
    │    2. Define a hook function that stores output[0] in        │
    │       storage[key] — same as make_hidden_state_hook in       │
    │       Exercise 03. You can inline it or import it.           │
    │    3. Register hook on peft_model.base_model.model.model     │
    │            .layers[layer_idx]                                │
    │       Store the handle as live_handle.                       │
    │    4. Register hook on frozen_model.model.layers[layer_idx]  │
    │       Store the handle as frozen_handle.                     │
    │    5. Return (live_store, frozen_store,                      │
    │               live_handle, frozen_handle)                    │
    │                                                              │
    │  CRITICAL: do NOT detach output[0] in the live hook.         │
    │    Detach in the frozen hook is fine — the frozen model runs  │
    │    under torch.no_grad() so gradients are already off.       │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        peft_model:   PEFT-wrapped live model (from build_lora_model).
        frozen_model: Bare frozen reference model (from load_model_and_tokenizer).
        layer_idx:    Which layer to hook (RMU_LAYER from config).

    Returns:
        (live_store, frozen_store, live_handle, frozen_handle)
    """
    # ########## Complete the Implementation ##########

    raise NotImplementedError(
        "TODO 1: implement register_rmu_hooks_for_lora_model(). ~10 lines. See docstring."
    )

    # ##################################################


# ===========================================================================
# SECTION 4 — SETUP  (provided — do not modify)
# ===========================================================================

def setup(args) -> tuple:
    """Load models, build data loaders, init optimizer."""
    print("\n" + "=" * 60)
    print(f"  Exercise 05 — LoRA+RMU Unlearning  backend={args.backend}")
    print("=" * 60)
    print()
    print("  REMINDER: Run 04_quantization_exploration.py first —")
    print("  it explains the layer path shift introduced by PEFT.")
    print()

    device = get_device()

    # Live model — PEFT-wrapped (LoRA adapters are trainable).
    peft_model, tokenizer = build_lora_model(args.backend, device)

    # Frozen reference model — bare (no PEFT). Already frozen by load_model_and_tokenizer.
    print("[setup] Loading frozen reference model (bare, no PEFT) ...")
    frozen_model, _ = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=True)

    # Forget set: domain-specific text loaded via build_forget_loader().
    # NOTE: actual text content handled entirely by utils.py — no CBRN text here.
    forget_loader = build_forget_loader(
        tokenizer, n_samples=args.forget_size, max_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE,
    )
    retain_loader = build_retain_loader(
        tokenizer, n_samples=args.retain_size, max_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE,
    )

    lr = args.lr if args.lr is not None else LR_RMU
    # Optimizer covers only trainable params (LoRA adapters).
    # PEFT already set requires_grad=False for base weights — just filter.
    optimizer = AdamW(
        (p for p in peft_model.parameters() if p.requires_grad),
        lr=lr,
    )

    print(f"\n[setup] steps={args.steps}  lr={lr:.2e}  layer={args.layer}")
    print(f"[setup] alpha={args.alpha}  beta={args.beta}")
    print(f"[setup] forget_size={args.forget_size}  retain_size={args.retain_size}\n")

    return peft_model, frozen_model, tokenizer, forget_loader, retain_loader, optimizer, device


# ===========================================================================
# SECTION 5 — TODO 2: LoRA-RMU Training Loop
# ===========================================================================

def train_qlora_rmu(
    peft_model,
    frozen_model,
    forget_loader,
    retain_loader,
    optimizer,
    steps: int,
    tokenizer,
    device: torch.device,
    layer_idx: int,
    alpha: float,
    beta: float,
    eval_samples: int,
) -> list[tuple]:
    """
    RMU training loop for a PEFT-wrapped (LoRA) model.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  This is the same algorithm as train_rmu in Exercise 03.     │
    │  The differences are:                                        │
    │    1. Use register_rmu_hooks_for_lora_model() instead of     │
    │       manually hooking (TODO 1 above already handles paths). │
    │    2. Model is peft_model — call peft_model.train() /        │
    │       peft_model.eval() to switch modes.                     │
    │    3. Optimizer already covers only LoRA params — no change  │
    │       to the backward / step calls.                          │
    │    4. Eval: pass peft_model to quick_wmdp_eval (it is a      │
    │       valid nn.Module with a forward() method).              │
    │                                                              │
    │  STEPS (~30 lines):                                          │
    │    Build random_vec = build_rmu_random_vector(...).          │
    │    Register hooks via register_rmu_hooks_for_lora_model().   │
    │    Create infinite iterators over loaders.                   │
    │    Loop for steps:                                           │
    │      Pass 1: peft_model(**forget_batch)  → h_forget          │
    │      Pass 2 (no_grad): frozen_model(**retain_batch)          │
    │              → h_frozen_retain                               │
    │      Pass 3: peft_model(**retain_batch)  → h_retain          │
    │      loss_f = rmu_forget_loss(h_forget, random_vec, alpha)   │
    │      loss_r = rmu_retain_loss(h_retain, h_frozen_retain)     │
    │      total  = loss_f + beta * loss_r                         │
    │      optimizer.zero_grad()                                   │
    │      total.backward()                                        │
    │      optimizer.step()                                        │
    │      Print step, loss_f, loss_r, total.                      │
    │      Every EVAL_EVERY: quick_wmdp_eval, peft_model.train().  │
    │    Remove hooks in finally.                                  │
    │    Return history.                                           │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        peft_model:    PEFT-wrapped live model (LoRA adapters trainable).
        frozen_model:  Bare frozen reference model (no PEFT, no grad).
        forget_loader: DataLoader of forget text.
        retain_loader: DataLoader of retain (general) text.
        optimizer:     AdamW over LoRA params only.
        steps:         Gradient steps.
        tokenizer:     For eval.
        device:        Target device.
        layer_idx:     Layer to hook.
        alpha:         Misdirection magnitude.
        beta:          Weight on retain loss.
        eval_samples:  Samples for quick_wmdp_eval.

    Returns:
        List of (step, total_loss, wmdp_acc_or_None) tuples.
    """
    # ########## Complete the Implementation ##########

    raise NotImplementedError(
        "TODO 2: implement train_qlora_rmu(). ~30 lines. See docstring above."
    )

    # ##################################################


# ===========================================================================
# SECTION 6 — POST-EVAL + SAVE  (provided — do not modify)
# ===========================================================================

def run_baseline(peft_model, tokenizer, device, n_samples: int) -> dict:
    """Measure WMDP eval accuracy before any unlearning."""
    print("\n[baseline] Measuring pre-unlearning WMDP eval accuracy ...")
    result = quick_wmdp_eval(peft_model, tokenizer, device, n_samples=n_samples)
    print(f"[baseline] WMDP eval accuracy (n={n_samples}): {result['accuracy']:.1%}")
    return result


def post_unlearning_eval_and_save(
    peft_model,
    tokenizer,
    device,
    pre_result: dict,
    history: list,
    args,
) -> None:
    """Measure post-unlearning accuracy, save LoRA adapter."""
    print("\n[post-eval] Measuring post-unlearning WMDP eval accuracy ...")
    post_result = quick_wmdp_eval(peft_model, tokenizer, device, n_samples=args.eval_samples)

    pre_acc  = pre_result["accuracy"]
    post_acc = post_result["accuracy"]
    delta    = post_acc - pre_acc

    print(f"\n{'=' * 60}")
    print(f"  UNLEARNING RESULTS — LoRA+RMU  backend={args.backend}")
    print(f"{'=' * 60}")
    print(f"  Pre-unlearning  WMDP eval: {pre_acc:.1%}")
    print(f"  Post-unlearning WMDP eval: {post_acc:.1%}")
    print(f"  Delta                    : {delta:+.1%}")
    print(f"  Random chance            : 25.0%")
    if delta < -0.03:
        print(f"  → Forgetting observed ✓")
    else:
        print(f"  → Minimal forgetting (try more steps or higher alpha/lr)")
    print(f"{'=' * 60}\n")

    full_result = None
    if args.full_eval:
        full_result = full_wmdp_eval(peft_model, tokenizer, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "method":          f"lora_rmu_{args.backend}",
        "steps":           args.steps,
        "lr":              args.lr,
        "alpha":           args.alpha,
        "beta":            args.beta,
        "layer":           args.layer,
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "forget_size":     args.forget_size,
        "retain_size":     args.retain_size,
        "pre_wmdp_acc":    pre_acc,
        "post_wmdp_quick": post_acc,
        "delta_wmdp":      delta,
        "post_wmdp_full":  full_result["accuracy"] if full_result else None,
        "timestamp":       timestamp,
        "model_id":        HF_MODEL_ID,
    }

    eval_points = [(s, l, a) for (s, l, a) in history if a is not None]
    if eval_points:
        print("  Mid-training WMDP eval trajectory:")
        for (step, loss, acc) in eval_points:
            print(f"    step {step:>4}  loss={loss:.4f}  wmdp={acc:.1%}")

    # Save LoRA adapter only — ~10 MB instead of ~2 GB.
    out_dir = UNLEARNING_RESULTS_DIR / f"lora_rmu_{args.backend}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    print(f"[checkpoint] Saving LoRA adapter to {out_dir} ...")
    peft_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[saved] LoRA adapter + metadata at: {out_dir}")
    print(f"[saved] Adapter size is ~10 MB (vs ~2 GB for full model).")


# ===========================================================================
# SECTION 7 — CLI + MAIN  (provided — do not modify)
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA+RMU Unlearning Exercise — Falcon3-1B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=["bf16-lora", "4bit-lora", "8bit-lora"], default="bf16-lora",
        help="Model backend. Only bf16-lora runs on MPS (M2 Mac). Others require CUDA.",
    )
    parser.add_argument("--steps",        type=int,   default=None,      help="Gradient steps.")
    parser.add_argument("--lr",           type=float, default=None,      help="Learning rate.")
    parser.add_argument("--alpha",        type=float, default=ALPHA_RMU,  help="Misdirection magnitude.")
    parser.add_argument("--beta",         type=float, default=BETA_RMU,   help="Weight on retain loss.")
    parser.add_argument("--layer",        type=int,   default=RMU_LAYER,  help="Layer index to hook.")
    parser.add_argument("--forget-size",  type=int,   default=200,        help="Forget sequences.")
    parser.add_argument("--retain-size",  type=int,   default=200,        help="Retain sequences.")
    parser.add_argument("--eval-samples", type=int,   default=EVAL_N_SAMPLES, help="WMDP eval samples.")
    parser.add_argument("--full-eval",    action="store_true",            help="Run full 1273-sample eval.")
    parser.add_argument("--skip-baseline", action="store_true",           help="Skip pre-unlearning eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.steps is None:
        args.steps = STEPS_RMU

    peft_model, frozen_model, tokenizer, forget_loader, retain_loader, optimizer, device = setup(args)

    if args.skip_baseline:
        pre_result = {"accuracy": float("nan"), "n_correct": 0, "n_total": 0}
        print("[baseline] Skipped (--skip-baseline).")
    else:
        pre_result = run_baseline(peft_model, tokenizer, device, n_samples=args.eval_samples)

    print(f"\n[train] Starting LoRA+RMU ({args.steps} steps, layer={args.layer}) ...")
    t0 = time.perf_counter()

    history = train_qlora_rmu(
        peft_model=peft_model,
        frozen_model=frozen_model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        optimizer=optimizer,
        steps=args.steps,
        tokenizer=tokenizer,
        device=device,
        layer_idx=args.layer,
        alpha=args.alpha,
        beta=args.beta,
        eval_samples=args.eval_samples,
    )

    elapsed = time.perf_counter() - t0
    print(f"[train] Done in {elapsed/60:.1f} min ({elapsed/args.steps:.1f} s/step).")

    post_unlearning_eval_and_save(peft_model, tokenizer, device, pre_result, history, args)


if __name__ == "__main__":
    main()
