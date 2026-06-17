"""
03_rmu_exercise.py — Representation Misdirection for Unlearning (RMU)
======================================================================
EXERCISE: Implement RMU on Falcon3-1B-Instruct.

LEARNING OBJECTIVES:
    1. Understand why RMU is architecture-DEPENDENT (hooks into hidden states
       at a specific layer — unlike GA/GD which only use outputs.loss).
    2. Implement a forward hook that captures hidden states with gradient.
    3. Implement the forget and retain MSE losses that operate in representation space.
    4. Assemble the full two-pass training loop with a frozen reference model.

THEORY REFERENCE:
    See UNLEARNING_METHODS.md (project root) for math and intuition.
    See 02_rmu_exploration.py for live demos of every primitive used here.

BEFORE YOU START:
    Run the exploration scripts first:
        python unlearning/00_architecture_exploration.py   # layer path, hidden_size
        python unlearning/02_rmu_exploration.py            # hooks, vectors, dry run

YOUR TASKS:
    ┌─────────────────────────────────────────────────────────────┐
    │  TODO 1 (Section 2): Implement make_hidden_state_hook()     │
    │  TODO 2 (Section 3): Implement rmu_forget_loss()            │
    │  TODO 3 (Section 4): Implement rmu_retain_loss()            │
    │  TODO 4 (Section 6): Implement train_rmu()                  │
    └─────────────────────────────────────────────────────────────┘
    All other code is provided. Do NOT modify utils.py or config.py.

SETUP:
    cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
    source venv/bin/activate
    cd falcon_eval_wmdp

USAGE EXAMPLES:
    # Smoke test first (verifies all code paths, ~3 min):
    python unlearning/03_rmu_exercise.py --steps 5 --forget-size 20 --retain-size 20 --eval-samples 20

    # Full RMU run (~30 min on M2 Max):
    python unlearning/03_rmu_exercise.py --steps 300 --forget-size 200 --retain-size 200

    # Full RMU + benchmark-grade final eval (adds ~15–20 min):
    python unlearning/03_rmu_exercise.py --steps 300 --full-eval

EXPECTED RESULTS (Falcon3-1B):
    Baseline WMDP eval : ~30–35%  (above 25% random)
    After RMU (300 steps): WMDP eval should drop toward 25%; general
                            capability should stay roughly preserved.

ARCHITECTURE NOTE:
    Unlike GA/GD (Exercise 01), this file IS architecture-dependent.
    It accesses model.model.layers[L] directly. If you switch to a
    different model family (e.g., old GPT-2 style), the layer path changes.
    00_architecture_exploration.py and 02_rmu_exploration.py cover why.
"""

import argparse
import sys
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.config import (
    HF_MODEL_ID,
    RMU_LAYER, LR_RMU, STEPS_RMU, ALPHA_RMU, BETA_RMU,
    MAX_SEQ_LEN, BATCH_SIZE, EVAL_EVERY, EVAL_N_SAMPLES,
    UNLEARNING_RESULTS_DIR,
)
from unlearning.utils import (
    get_device,
    load_model_and_tokenizer,
    build_retain_loader,
    full_wmdp_eval,
    save_checkpoint,
)
from unlearning.rmu_utils import (
    make_disjoint_splits,
    build_forget_loader_from_samples,
    wmdp_eval_on_samples,
)


# ===========================================================================
# SECTION 0 — SETUP  (provided — do not modify)
# ===========================================================================

def setup(args, forget_samples: list) -> tuple:
    """Load model + frozen reference model, build data loaders, init optimizer."""
    print("\n" + "=" * 60)
    print("  Exercise 03 — RMU Unlearning")
    print("=" * 60)
    print()
    print("  REMINDER: Run 02_rmu_exploration.py first if you haven't —")
    print("  it shows every primitive this file uses.")
    print()

    device = get_device()

    # Live model — parameters will be updated.
    print("[setup] Loading live model ...")
    model, tokenizer = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=False)

    # Frozen reference model — weights never change; provides the retain anchor.
    print("[setup] Loading frozen reference model ...")
    frozen_model, _ = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=True)

    # Forget set: pre-selected samples from make_disjoint_splits (disjoint from eval).
    # NOTE: no CBRN text in this file — content handled by rmu_utils.
    forget_loader = build_forget_loader_from_samples(
        forget_samples,
        tokenizer,
        max_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    retain_loader = build_retain_loader(
        tokenizer,
        n_samples=args.retain_size,
        max_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    lr = args.lr if args.lr is not None else LR_RMU
    # Optimizer wraps only live model params (frozen_model params have requires_grad=False).
    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"\n[setup] steps={args.steps}  lr={lr:.2e}  layer={args.layer}")
    print(f"[setup] alpha={args.alpha}  beta={args.beta}")
    print(f"[setup] forget_size={args.forget_size}  retain_size={args.retain_size}\n")

    return model, frozen_model, tokenizer, forget_loader, retain_loader, optimizer, device


# ===========================================================================
# SECTION 1 — BASELINE EVAL  (provided — do not modify)
# ===========================================================================

def run_baseline(model, tokenizer, device, eval_samples_list: list) -> dict:
    """Measure WMDP eval accuracy before any unlearning."""
    print("\n[baseline] Measuring pre-unlearning WMDP eval accuracy ...")
    result = wmdp_eval_on_samples(model, tokenizer, device, eval_samples_list)
    print(f"[baseline] WMDP eval accuracy (n={len(eval_samples_list)}): {result['accuracy']:.1%}")
    print(f"           ({result['n_correct']}/{result['n_total']} correct, 25% = random chance)")
    return result


# ===========================================================================
# SECTION 2 — TODO 1: Forward Hook
# ===========================================================================

def make_hidden_state_hook(storage: dict, key: str) -> Callable:
    """
    Return a forward hook function that stores the hidden state in storage[key].

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  BACKGROUND:                                                 │
    │    A forward hook has signature:                             │
    │        fn(module, input, output) -> None                     │
    │    It is called automatically after the module's forward().  │
    │    `output` is a tuple. For LlamaDecoderLayer:               │
    │        output[0]  → hidden_state  (B, seq_len, hidden_size)  │
    │        output[1+] → optional attention weights               │
    │                                                              │
    │  WHAT TO DO:                                                 │
    │    Return a closure that, when called as a hook, stores      │
    │    output[0] into storage[key].                              │
    │                                                              │
    │    CRITICAL: do NOT detach or clone output[0].               │
    │    Detaching would break the gradient path on the live model.│
    │    The forget loss needs gradients to flow back through h    │
    │    all the way to the model's weights. Detaching severs that.│
    │                                                              │
    │    The frozen model's hook CAN store a detached copy —       │
    │    but that detachment is handled inside train_rmu (TODO 4)  │
    │    with torch.no_grad(), not here.                           │
    │                                                              │
    │  USAGE (in train_rmu):                                       │
    │    live_store   = {}                                         │
    │    frozen_store = {}                                         │
    │    h1 = model.model.layers[L].register_forward_hook(        │
    │             make_hidden_state_hook(live_store, "h"))         │
    │    h2 = frozen_model.model.layers[L].register_forward_hook( │
    │             make_hidden_state_hook(frozen_store, "h"))       │
    │    ...                                                       │
    │    h1.remove(); h2.remove()                                  │
    │                                                              │
    │  EXPECTED (~3 lines):                                        │
    │    def hook(module, input, output):                          │
    │        storage[key] = output[0]                              │
    │    return hook                                               │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        storage: Dict to write into — keys let the caller distinguish
                 live vs frozen, forget vs retain.
        key:     Dict key to store the hidden state under.

    Returns:
        A forward hook function (module, input, output) -> None.
    """
    # ########## Complete the Implementation ##########
    try:
        def hook(module, _input, output):
            storage[key] = output[0]
        return hook
    except Exception as e:
        print(f"Error Occured while creating the forward hook...\n{str(e)}\n")

    # raise NotImplementedError(
    #     "TODO 1: implement make_hidden_state_hook(). ~3 lines. See docstring above."
    # )

    # ##################################################


# ===========================================================================
# SECTION 3 — TODO 2: Forget Loss
# ===========================================================================

def rmu_forget_loss(
    hidden: torch.Tensor,
    random_vec: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Compute the RMU forget loss: MSE between hidden state and alpha * random_vec.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  FORMULA:                                                    │
    │    target = (alpha * random_vec).expand_as(hidden)           │
    │    loss_f = F.mse_loss(hidden, target)                       │
    │                                                              │
    │  SHAPES:                                                     │
    │    hidden     : (batch, seq_len, hidden_size)                │
    │    random_vec : (hidden_size,)   ← 1D unit vector            │
    │    target     : (batch, seq_len, hidden_size) after expand   │
    │                                                              │
    │  DTYPE NOTE:                                                 │
    │    hidden is bfloat16 (model runs in bf16).                  │
    │    Cast random_vec to hidden.dtype before scaling.           │
    │    F.mse_loss operates in the same dtype — both inputs must  │
    │    match. Consider calling .float() on both if you see NaN.  │
    │                                                              │
    │  WHY EXPAND, NOT BROADCAST:                                  │
    │    random_vec has shape (hidden_size,).                       │
    │    Multiplying alpha * random_vec gives (hidden_size,).      │
    │    .expand_as(hidden) makes it (batch, seq_len, hidden_size) │
    │    without copying memory — just changing the stride.        │
    │    This is equivalent to "apply the same target to every     │
    │    token position in every sequence in the batch."           │
    │                                                              │
    │  EXPECTED (~3 lines):                                        │
    │    rv = random_vec.to(hidden.dtype)                          │
    │    target = (alpha * rv).expand_as(hidden)                   │
    │    return F.mse_loss(hidden, target)                         │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        hidden:     Hidden state from model.model.layers[L], shape (B, T, H).
        random_vec: Fixed unit vector c, shape (H,).
        alpha:      Misdirection magnitude (ALPHA_RMU from config).

    Returns:
        Scalar MSE loss tensor with gradient.
    """
    # ########## Complete the Implementation ##########
    rv = random_vec.to(hidden.dtype)
    target = (alpha*rv).expand_as(hidden)
    return F.mse_loss(hidden, target)

    # If you hit NaNs once you start running real steps (bf16 squares can overflow more easily than you'd think at higher alpha), swap the last line for:
    # return F.mse_loss(hidden.float(), target.float())

    # raise NotImplementedError(
    #     "TODO 2: implement rmu_forget_loss(). ~3 lines. See docstring above."
    # )

    # ##################################################


# ===========================================================================
# SECTION 4 — TODO 3: Retain Loss
# ===========================================================================

def rmu_retain_loss(
    hidden_live: torch.Tensor,
    hidden_frozen: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the RMU retain loss: MSE between live and frozen hidden states.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  FORMULA:                                                    │
    │    loss_r = F.mse_loss(hidden_live, hidden_frozen)           │
    │                                                              │
    │  PRE-CONDITIONS (guaranteed by train_rmu, TODO 4):           │
    │    hidden_frozen is already DETACHED (captured under         │
    │    torch.no_grad() in the frozen model's forward pass).      │
    │    You do NOT need to call .detach() here.                   │
    │                                                              │
    │  DTYPE NOTE:                                                 │
    │    Both tensors are bfloat16. Cast to float() if you see NaN │
    │    (bfloat16 MSE can overflow for large hidden states).      │
    │    For safety: F.mse_loss(hidden_live.float(),               │
    │                           hidden_frozen.float())             │
    │                                                              │
    │  EXPECTED (~1 line):                                         │
    │    return F.mse_loss(hidden_live, hidden_frozen)             │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        hidden_live:   Hidden state from live model on retain batch, (B, T, H).
        hidden_frozen: Hidden state from frozen model on retain batch, (B, T, H).
                       Already detached — no gradient flows through it.

    Returns:
        Scalar MSE loss tensor (gradient flows only through hidden_live).
    """
    # ########## Complete the Implementation ##########
    return F.mse_loss(hidden_live, hidden_frozen)

    # raise NotImplementedError(
    #     "TODO 3: implement rmu_retain_loss(). ~1 line. See docstring above."
    # )

    # ##################################################


# ===========================================================================
# SECTION 5 — PROVIDED: build_rmu_random_vector  (do not modify)
# ===========================================================================

def build_rmu_random_vector(
    hidden_size: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build the fixed unit-norm random misdirection vector c.

    Seeded for reproducibility. Same c is used for every step and every batch.
    Not a TODO — provided so the exercise focuses on the loss + loop.

    Returns:
        Tensor of shape (hidden_size,), unit norm, on the given device.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randn(hidden_size, generator=gen)
    c = c / c.norm()
    return c.to(device)


# ===========================================================================
# SECTION 6 — TODO 4: Training Loop
# ===========================================================================

def train_rmu(
    model: torch.nn.Module,
    frozen_model: torch.nn.Module,
    forget_loader,
    retain_loader,
    optimizer,
    steps: int,
    tokenizer,
    device: torch.device,
    layer_idx: int,
    alpha: float,
    beta: float,
    eval_samples_list: list,
) -> list[tuple]:
    """
    RMU training loop with two-pass hidden-state loss.

    ┌──────────────────────────────────────────────────────────────┐
    │             ###### Complete the Implementation ######        │
    │                                                              │
    │  ALGORITHM (per step):                                       │
    │    1. Get forget batch and retain batch.                     │
    │    2. PASS 1 — live model forward on forget batch:           │
    │           model(**forget_batch)                              │
    │       → live_store["h"] now holds h_forget (with gradient).  │
    │                                                              │
    │    3. PASS 2 — frozen model forward on retain batch          │
    │       (inside torch.no_grad()):                              │
    │           frozen_model(**retain_batch)                       │
    │       → frozen_store["h"] holds h_frozen_retain (detached).  │
    │                                                              │
    │    4. PASS 3 — live model forward on retain batch:           │
    │           model(**retain_batch)                              │
    │       → live_store["h"] now holds h_retain (with gradient).  │
    │         (live_store["h"] is OVERWRITTEN — that's fine,       │
    │          h_forget reference saved in step 2 is still valid.) │
    │                                                              │
    │    5. Compute losses:                                        │
    │           loss_f = rmu_forget_loss(h_forget, random_vec, alpha)  │
    │           loss_r = rmu_retain_loss(h_retain, h_frozen_retain)    │
    │           total  = loss_f + beta * loss_r                    │
    │                                                              │
    │    6. Backward + step + zero_grad.                           │
    │                                                              │
    │    7. Every EVAL_EVERY steps: quick_wmdp_eval, model.train().│
    │                                                              │
    │  HOOK LIFECYCLE:                                             │
    │    Register hooks ONCE before the loop (not inside the loop).│
    │    Use try/finally to remove them even if an exception fires. │
    │    live_store overwrite between forget and retain is fine.   │
    │                                                              │
    │  IMPORTANT — save h_forget before PASS 3 overwrites it:     │
    │    h_forget = live_store["h"]    # save reference             │
    │    # ... PASS 3 runs, live_store["h"] = h_retain ...         │
    │    h_retain = live_store["h"]    # now this is h_retain       │
    │    # use h_forget (saved earlier) and h_retain (just set)    │
    │                                                              │
    │  EXPECTED (~45 lines):                                       │
    │    Build random_vec via build_rmu_random_vector().           │
    │    Register hooks. Infinite cycle loaders.                   │
    │    Loop for steps:                                           │
    │      Pass 1 (forget) → h_forget                              │
    │      Pass 2 (frozen retain) under no_grad → h_frozen_retain  │
    │      Pass 3 (live retain) → h_retain                         │
    │      Losses → backward → step → zero_grad                   │
    │      Print step, loss_f, loss_r, total.                      │
    │      Every EVAL_EVERY: quick eval.                           │
    │    Remove hooks in finally.                                  │
    │    Return history list.                                      │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Args:
        model:         Live model (will be updated).
        frozen_model:  Frozen reference model (never updated).
        forget_loader: DataLoader of forget text.
        retain_loader: DataLoader of retain (general) text.
        optimizer:     AdamW wrapping live model params.
        steps:         Gradient steps to run.
        tokenizer:     For eval calls.
        device:        Target device.
        layer_idx:     Which layer to hook (RMU_LAYER from config).
        alpha:         Misdirection magnitude.
        beta:          Weight on retain loss.
        eval_samples:  Samples for quick_wmdp_eval.

    Returns:
        List of (step, total_loss, wmdp_acc_or_None) tuples.
    """
    # ########## Complete the Implementation ##########

    # Step 1: build the fixed random misdirection vector
    # random_vec = build_rmu_random_vector(model.config.hidden_size, device)
    random_vec = build_rmu_random_vector(model.config.hidden_size, device)

    # Step 2: register hooks on live and frozen model
    # live_store   = {}
    # frozen_store = {}
    # live_handle   = model.model.layers[layer_idx].register_forward_hook(
    #                     make_hidden_state_hook(live_store, "h"))
    # frozen_handle = frozen_model.model.layers[layer_idx].register_forward_hook(
    #                     make_hidden_state_hook(frozen_store, "h"))
    
    live_store, frozen_store = {}, {}
    live_handle=model.model.layers[layer_idx].register_forward_hook(make_hidden_state_hook(live_store, "h"))
    frozen_handle= frozen_model.model.layers[layer_idx].register_forward_hook(make_hidden_state_hook(frozen_store,"h"))
    

    # Step 3: infinite iterators over loaders
    # forget_iter = cycle(forget_loader)
    # retain_iter = cycle(retain_loader)
    
    forget_iter = cycle(forget_loader)
    retain_iter=cycle(retain_loader)
    

    # Step 4: training loop
    # history = []
    # model.train()
    # try:
    #     for step in range(steps):
    #         forget_batch = {k: v.to(device) for k, v in next(forget_iter).items()}
    #         retain_batch = {k: v.to(device) for k, v in next(retain_iter).items()}
    #
    #         # Pass 1: forget forward
    #         _ = model(**forget_batch)
    #         h_forget = live_store["h"]               # save before overwrite
    #
    #         # Pass 2: frozen retain (no_grad → h_frozen_retain is detached)
    #         with torch.no_grad():
    #             _ = frozen_model(**retain_batch)
    #         h_frozen_retain = frozen_store["h"]
    #
    #         # Pass 3: live retain
    #         _ = model(**retain_batch)
    #         h_retain = live_store["h"]               # now h_retain
    #
    #         # Losses
    #         loss_f = rmu_forget_loss(h_forget, random_vec, alpha)
    #         loss_r = rmu_retain_loss(h_retain, h_frozen_retain)
    #         total  = loss_f + beta * loss_r
    #
    #         # Backward + step
    #         optimizer.zero_grad()
    #         total.backward()
    #         optimizer.step()
    #
    #         # Logging
    #         wmdp_acc = None
    #         if (step + 1) % EVAL_EVERY == 0:
    #             r = wmdp_eval_on_samples(model, tokenizer, device, eval_samples_list)
    #             wmdp_acc = r["accuracy"]
    #
    #         print(f"Step {step+1}/{steps}  loss_f={loss_f.item():.4f}  "
    #               f"loss_r={loss_r.item():.6f}  total={total.item():.4f}")
    #         history.append((step + 1, total.item(), wmdp_acc))
    #
    # finally:
    #     live_handle.remove()
    #     frozen_handle.remove()
    #
    # return history
    history = []
    model.train()
    try:
        for step in range(steps):
            forget_batch={k:v.to(device) for k, v in next(forget_iter).items()}
            retain_batch={k:v.to(device) for k,v in next(retain_iter).items()}
            # Pass 1; live model on forget text. live_store["h"] gets gradient
            _ =model(**forget_batch)
            h_forget = live_store["h"] # saved for ref
            
            # Pass 2: frozen model on retain text, no_grad(). Detached anchor
            with torch.no_grad():
                _ = frozen_model(**retain_batch)
            h_frozen_retrain = frozen_store["h"]
            
            # Pass 3: Live model on retain text,  overwrites live_stroe["h"]; but h_forget above already poiitns to the old tensor, so it;s safe 
            _ = model(**retain_batch)
            h_retain = live_store["h"]
            
            loss_f = rmu_forget_loss(h_forget, random_vec, alpha)
            loss_r = rmu_retain_loss(h_retain, h_frozen_retrain)
            total = loss_f + beta * loss_r
            
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            
            
            wmdp_acc = None
            if (step+1) % EVAL_EVERY == 0:
                r = wmdp_eval_on_samples(model, tokenizer, device, eval_samples_list)
                wmdp_acc = r["accuracy"]
                
            print(f"Step {step+1}/{steps}: Loss_f={loss_f.item():.5f}   "
                  f"loss_r={loss_r.item():.5f} total={total.item():.5f}"
                  + (f"   wmdp_acc={wmdp_acc:.2%}" if wmdp_acc is not None else ""))
            history.append((step+1, loss_f.item(), loss_r.item(), total.item(), wmdp_acc))
    finally:
        live_handle.remove()
        frozen_handle.remove()
    
    return history

    # raise NotImplementedError(
    #     "TODO 4: implement train_rmu(). ~45 lines. See docstring above."
    # )

    # ##################################################


# ===========================================================================
# SECTION 7a — PLOT TRAINING HISTORY
# ===========================================================================

def plot_history(
    history: list[tuple],
    pre_acc: float,
    post_acc: float,
    save_path: Path,
) -> None:
    """Plot RMU training curves and save to disk.

    Args:
        history:   List of (step, loss_f, loss_r, total, wmdp_acc_or_None).
        pre_acc:   Baseline WMDP accuracy (before unlearning).
        post_acc:  Post-unlearning WMDP accuracy.
        save_path: Path to write the PNG.
    """
    steps   = [h[0] for h in history]
    loss_f  = [h[1] for h in history]
    loss_r  = [h[2] for h in history]
    total   = [h[3] for h in history]

    eval_steps = [h[0] for h in history if h[4] is not None]
    eval_accs  = [h[4] for h in history if h[4] is not None]

    n_panels = 3 if eval_steps else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    fig.suptitle("RMU Training Curves", fontsize=13, fontweight="bold")

    # Panel 1: forget loss
    ax = axes[0]
    ax.plot(steps, loss_f, color="tab:red", linewidth=1.2, label="loss_f (forget)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Forget Loss (MSE → random vector)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: retain loss + total
    ax = axes[1]
    ax.plot(steps, total,  color="tab:blue",  linewidth=1.4, label="total")
    ax.plot(steps, loss_r, color="tab:green", linewidth=1.0, linestyle="--", label="loss_r (retain)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Retain + Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3 (optional): WMDP eval trajectory
    if eval_steps:
        ax = axes[2]
        ax.plot(eval_steps, [a * 100 for a in eval_accs],
                color="tab:purple", marker="o", linewidth=1.2, label="WMDP acc (mid-train)")
        ax.axhline(pre_acc  * 100, color="gray",    linestyle="--", linewidth=1.0, label=f"Baseline {pre_acc:.1%}")
        ax.axhline(post_acc * 100, color="tab:red", linestyle=":",  linewidth=1.0, label=f"Post-RMU {post_acc:.1%}")
        ax.axhline(25.0, color="black", linestyle="-.", linewidth=0.8, label="Random 25%")
        ax.set_xlabel("Step")
        ax.set_ylabel("WMDP Accuracy (%)")
        ax.set_title("WMDP Eval Accuracy Trajectory")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# SECTION 7 — POST-UNLEARNING EVAL + SAVE  (provided — do not modify)
# ===========================================================================

def post_unlearning_eval_and_save(
    model,
    tokenizer,
    device,
    pre_result: dict,
    history: list,
    args,
    eval_samples_list: list,
) -> None:
    """Measure accuracy after unlearning, print delta, optionally run full eval."""
    print("\n[post-eval] Measuring post-unlearning WMDP eval accuracy ...")
    post_result = wmdp_eval_on_samples(model, tokenizer, device, eval_samples_list)

    pre_acc  = pre_result["accuracy"]
    post_acc = post_result["accuracy"]
    delta    = post_acc - pre_acc

    print(f"\n{'=' * 60}")
    print(f"  UNLEARNING RESULTS — RMU")
    print(f"{'=' * 60}")
    print(f"  Pre-unlearning  WMDP eval: {pre_acc:.1%}")
    print(f"  Post-unlearning WMDP eval: {post_acc:.1%}")
    print(f"  Delta                    : {delta:+.1%}")
    print(f"  Random chance            : 25.0%")
    if delta < -0.03:
        print(f"  → Forgetting observed ✓  (WMDP eval accuracy decreased)")
    else:
        print(f"  → Minimal forgetting (try more steps or higher alpha/lr)")
    print(f"{'=' * 60}\n")

    full_result = None
    if args.full_eval:
        print("[full-eval] Running full WMDP eval ...")
        full_result = full_wmdp_eval(model, tokenizer, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "method":          "rmu",
        "steps":           args.steps,
        "lr":              args.lr,
        "alpha":           args.alpha,
        "beta":            args.beta,
        "layer":           args.layer,
        "forget_size":     args.forget_size,
        "retain_size":     args.retain_size,
        "pre_wmdp_acc":    pre_acc,
        "post_wmdp_quick": post_acc,
        "delta_wmdp":      delta,
        "post_wmdp_full":  full_result["accuracy"] if full_result else None,
        "timestamp":       timestamp,
        "model_id":        HF_MODEL_ID,
    }

    eval_points = [(s, lf, lr, lt, a) for (s, lf, lr, lt, a) in history if a is not None]
    if eval_points:
        print("  Mid-training WMDP eval trajectory:")
        for (step, lf, lr, lt, acc) in eval_points:
            print(f"    step {step:>4}  loss_f={lf:.4f}  loss_r={lr:.4f}  total={lt:.4f}  wmdp={acc:.1%}")

    out_dir = UNLEARNING_RESULTS_DIR / f"rmu_{timestamp}"
    save_checkpoint(model, tokenizer, out_dir, metadata)
    print(f"\n[saved] Checkpoint + metadata at: {out_dir}")

    plot_path = out_dir / "rmu_training_curves.png"
    plot_history(history, pre_result["accuracy"], post_acc, plot_path)
    print(f"[saved] Training curves at: {plot_path}")


# ===========================================================================
# SECTION 8 — CLI + MAIN  (provided — do not modify)
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RMU Unlearning Exercise — Falcon3-1B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",       type=int,   default=None,     help="Gradient steps. Default: STEPS_RMU from config.")
    parser.add_argument("--lr",          type=float, default=None,     help="Learning rate. Default: LR_RMU from config.")
    parser.add_argument("--alpha",       type=float, default=ALPHA_RMU, help="Misdirection magnitude.")
    parser.add_argument("--beta",        type=float, default=BETA_RMU,  help="Weight on retain loss.")
    parser.add_argument("--layer",       type=int,   default=RMU_LAYER, help="Layer index to hook.")
    parser.add_argument("--forget-size", type=int,   default=200,       help="Number of forget sequences.")
    parser.add_argument("--retain-size", type=int,   default=200,       help="Number of retain sequences.")
    parser.add_argument("--eval-samples",type=int,   default=EVAL_N_SAMPLES, help="Samples for quick WMDP eval.")
    parser.add_argument("--full-eval",   action="store_true",           help="Run full 1273-sample eval after training.")
    parser.add_argument("--skip-baseline", action="store_true",         help="Skip pre-unlearning eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.steps is None:
        args.steps = STEPS_RMU

    # Partition WMDP-bio into disjoint forget-train and eval sets before loading models.
    # This prevents the eval from measuring direct disruption rather than generalized forgetting.
    forget_samples_list, eval_samples_list = make_disjoint_splits(
        forget_size=args.forget_size,
        eval_size=args.eval_samples,
        seed=42,
    )

    model, frozen_model, tokenizer, forget_loader, retain_loader, optimizer, device = setup(
        args, forget_samples_list
    )

    if args.skip_baseline:
        pre_result = {"accuracy": float("nan"), "n_correct": 0, "n_total": 0}
        print("[baseline] Skipped (--skip-baseline).")
    else:
        pre_result = run_baseline(model, tokenizer, device, eval_samples_list)

    print(f"\n[train] Starting RMU unlearning ({args.steps} steps, layer={args.layer}) ...")
    t0 = time.perf_counter()

    history = train_rmu(
        model=model,
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
        eval_samples_list=eval_samples_list,
    )

    elapsed = time.perf_counter() - t0
    print(f"[train] Done in {elapsed/60:.1f} min ({elapsed/args.steps:.1f} s/step).")

    post_unlearning_eval_and_save(model, tokenizer, device, pre_result, history, args, eval_samples_list)


if __name__ == "__main__":
    main()
