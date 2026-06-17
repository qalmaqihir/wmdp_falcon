"""
02_rmu_exploration.py — RMU Primitives: Hooks, Random Vectors, Losses
======================================================================
Run this BEFORE 03_rmu_exercise.py. No training, no TODOs, no argparse.

Every primitive that RMU uses is demonstrated live:
  1. Architecture recap (layer path, hidden_size)
  2. Forward hooks — register, capture, print shapes, remove
  3. Random misdirection vector — build, normalize, scale by alpha
  4. Forget loss preview — MSE(hidden, alpha*c) for several alpha values
  5. Retain loss preview — MSE(h_live, h_frozen) starts at ~0
  6. Full two-pass dry run — one complete RMU step without optimizer.step()
  7. Why MSE not CE? — conceptual explanation

WHY THIS MATTERS:
  GA/GD only need outputs.loss — identical code on any HF causal LM.
  RMU needs: (a) a forward hook on model.model.layers[L], (b) a fixed
  random vector c, (c) a frozen reference model. This file shows each
  piece working before you put them together in 03_rmu_exercise.py.

USAGE:
  cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
  source venv/bin/activate && cd falcon_eval_wmdp
  python unlearning/02_rmu_exploration.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.config import (
    HF_MODEL_ID,
    RMU_LAYER,
    ALPHA_RMU,
)
from unlearning.utils import (
    get_device,
    load_model_and_tokenizer,
    print_architecture_summary,
)


SEP_THICK = "=" * 64
SEP_THIN  = "-" * 64

def _load_real_samples() -> tuple[str, str]:
    """Fetch one real forget sample and one real retain sample from the actual datasets.

    Mirrors the two-tier strategy in build_forget_loader():
      - Forget: cais/wmdp-corpora bio-remove-corpus (canonical WMDP paper corpus).
                Falls back to cais/wmdp wmdp-bio MCQ formatted as plain text
                if the primary corpus is gated or unavailable.
      - Retain: wikitext/wikitext-2-raw-v1 (same as build_retain_loader()).

    Returns:
        (forget_text, retain_text) — two non-empty strings ready for tokenization.
    """
    from datasets import load_dataset

    print("[samples] Loading real forget sample ...")
    forget_text: str | None = None
    try:
        ds = load_dataset(
            "cais/wmdp-corpora", "bio-remove-corpus", split="train",
            trust_remote_code=False,
        )
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) > 100:
                forget_text = text[:600]
                break
        if forget_text:
            print(f"[samples] forget — wmdp-corpora: {len(forget_text)} chars")
    except Exception as e:
        if any(kw in str(e).lower() for kw in ["gated", "401", "403", "unauthorized", "restricted"]):
            print(f"[samples] wmdp-corpora gated — falling back to wmdp-bio MCQ.")
        else:
            print(f"[samples] wmdp-corpora unavailable ({type(e).__name__}) — using MCQ fallback.")

    if forget_text is None:
        ds = load_dataset("cais/wmdp", "wmdp-bio", split="test", trust_remote_code=False)
        row = list(ds)[0]
        choices = row["choices"]
        letters = ["A", "B", "C", "D"]
        ans_letter = letters[row["answer"]]
        ans_text = choices[row["answer"]]
        forget_text = (
            f"Question: {row['question']}\n"
            f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
            f"Answer: {ans_letter}) {ans_text}"
        )
        print(f"[samples] forget — wmdp-bio MCQ fallback: {len(forget_text)} chars")

    print("[samples] Loading real retain sample ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=False)
    retain_text: str | None = None
    for row in ds:
        text = row.get("text", "").strip()
        if len(text) > 100:
            retain_text = text[:600]
            break
    if retain_text is None:
        retain_text = "The history of science spans many centuries and civilizations."
    print(f"[samples] retain — wikitext-2: {len(retain_text)} chars")

    return forget_text, retain_text


# ===========================================================================
# Section 1 — Architecture recap
# ===========================================================================

def section_architecture_recap(model) -> None:
    """Re-print the key RMU-relevant facts from 00_architecture_exploration.py."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 1: Architecture Recap (for RMU)")
    print(SEP_THICK)
    n = model.config.num_hidden_layers
    h = model.config.hidden_size
    mid = n // 2
    print(f"  num_hidden_layers : {n}")
    print(f"  hidden_size       : {h}")
    print(f"  Layer access path : model.model.layers[L]  where L ∈ [0, {n-1}]")
    print(f"  Recommended RMU L : {mid}  (middle layer — empirically effective)")
    print()
    print("  GA/GD only use outputs.loss → architecture-independent.")
    print("  RMU uses model.model.layers[L].output[0] → architecture-dependent.")
    print("  The two facts you need: the path above, and hidden_size for c's shape.")
    print(SEP_THIN)


# ===========================================================================
# Section 2 — Forward hooks
# ===========================================================================

def section_forward_hooks(model, tokenizer, device, forget_sample: str) -> None:
    """Demo: register a hook, run a forward pass, print the hidden state, remove."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 2: Forward Hooks — Live Demo")
    print(SEP_THICK)

    # ── Sample input — real biosecurity forget text ────────────────────────
    sample = forget_sample
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── The hook captures a reference to output[0] ────────────────────────
    # output is a tuple emitted by LlamaDecoderLayer:
    #   output[0]: hidden_state  (batch, seq_len, hidden_size)
    #   output[1]: optional attention weights (if output_attentions=True)
    captured = {}

    def my_hook(module, input, output):
        # output[0] is the hidden state tensor with gradient connection.
        captured["h"] = output[0]

    print(f"  Registering hook on model.model.layers[{RMU_LAYER}] ...")
    handle = model.model.layers[RMU_LAYER].register_forward_hook(my_hook)

    print(f"  Running forward pass on real forget sample ...")
    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    h = captured["h"]
    print()
    print(f"  Captured hidden state:")
    print(f"    shape       : {tuple(h.shape)}  (batch=1, seq={h.shape[0]}, hidden={h.shape[1]})")
    print(f"    dtype       : {h.dtype}")
    print(f"    mean        : {h.float().mean().item():.4f}")
    print(f"    std         : {h.float().std().item():.4f}")
    print(f"    L2 norm     : {h.float().norm().item():.2f}")
    print()
    print(f"  This tensor is the raw hidden state at layer {RMU_LAYER}.")
    print(f"  RMU's forget loss pushes this toward alpha * c.")
    print(f"  RMU's retain loss keeps this close to the frozen model's version.")

    print()
    print("  Removing hook ...")
    handle.remove()
    print("  Hook removed. A new forward pass will NOT update captured['h'].")

    print()
    print("  HOOK LIFECYCLE SUMMARY:")
    print("    1. handle = layer.register_forward_hook(fn)   → hook active")
    print("    2. model(**batch)                              → fn() called, stores h")
    print("    3. handle.remove()                            → hook inactive")
    print("    In train_rmu: register ONCE before the loop, remove in finally block.")
    print(SEP_THIN)


# ===========================================================================
# Section 3 — Random misdirection vector
# ===========================================================================

def section_random_vector(hidden_size: int, device: torch.device) -> torch.Tensor:
    """Demo: build the fixed random unit vector c and show scaling by alpha."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 3: Random Misdirection Vector")
    print(SEP_THICK)

    gen = torch.Generator(device=device).manual_seed(42)
    c = torch.randn(hidden_size, generator=gen, device="mps")#device) #Manually putting mps otherwise throws error; RuntimeError: Expected a 'mps' device type for generator but found 'cpu' 
    c = c / c.norm()

    print(f"  c = torch.randn({hidden_size})  (seeded for reproducibility)")
    print(f"  c = c / c.norm()           → unit vector")
    print()
    print(f"  c.shape : {tuple(c.shape)}")
    print(f"  c.norm  : {c.norm().item():.6f}  (should be 1.0)")
    print(f"  c.mean  : {c.float().mean().item():.4f}")
    print(f"  c.std   : {c.float().std().item():.4f}")
    print()
    print("  alpha * c  (misdirection target) for several alpha values:")
    print(f"  {'alpha':>10}  {'magnitude (||alpha*c||)':>26}  {'interpretation'}")
    print(f"  {'-'*10}  {'-'*26}  {'-'*28}")
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        target = alpha * c
        print(f"  {alpha:>10.1f}  {target.norm().item():>26.2f}  "
              f"{'← paper uses ~1500 for 7B' if alpha == 1000 else ''}")
    print()
    print(f"  Current ALPHA_RMU = {ALPHA_RMU} (Falcon3-1B, conservative start).")
    print(f"  The model is trained to map forget hidden states toward alpha*c.")
    print(f"  c is FIXED throughout training — same vector for every step and batch.")
    print(f"  Larger alpha → stronger misdirection but harder for retain to resist.")
    print(SEP_THIN)
    return c


# ===========================================================================
# Section 4 — Forget loss preview
# ===========================================================================

def section_forget_loss_preview(
    model, tokenizer, device, c: torch.Tensor, forget_sample: str
) -> None:
    """Demo: compute MSE between hidden state and alpha*c for several alpha values."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 4: Forget Loss Preview (no training)")
    print(SEP_THICK)

    # Real biosecurity forget text — same source as build_forget_loader().
    sample = forget_sample
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    captured = {}

    def hook_fn(module, input, output):
        captured["h"] = output[0]

    handle = model.model.layers[RMU_LAYER].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    h = captured["h"]
    print(f"  Hidden state shape: {tuple(h.shape)}")
    print()
    print(f"  Forget loss = MSE(hidden, alpha * c.expand_as(hidden))")
    print(f"  {'alpha':>10}  {'||alpha*c||':>14}  {'loss_f':>10}  note")
    print(f"  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*30}")

    # c is shape (hidden_size,); expand to (batch, seq_len, hidden_size)
    c_f = c.to(h.dtype)
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        target = (alpha * c_f).expand_as(h)
        loss_f = F.mse_loss(h.float(), target.float())
        note = "← current ALPHA_RMU" if abs(alpha - ALPHA_RMU) < 0.1 else ""
        print(f"  {alpha:>10.1f}  {(alpha * c_f).norm().item():>14.2f}  {loss_f.item():>10.4f}  {note}")

    print()
    print("  Observations:")
    print("  - Larger alpha → larger MSE target → larger loss_f → stronger gradient.")
    print("  - At alpha=100 (ALPHA_RMU default), loss is large enough to train against.")
    print("  - In TODO 2 of 03_rmu_exercise.py you implement this exact computation.")
    print(SEP_THIN)


# ===========================================================================
# Section 5 — Retain loss preview
# ===========================================================================

def section_retain_loss_preview(
    model, frozen_model, tokenizer, device, retain_sample: str
) -> None:
    """Demo: MSE(h_live, h_frozen) starts at ~0 when both models are identical."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 5: Retain Loss Preview (no training)")
    print(SEP_THICK)

    # Real Wikitext-2 retain text — same source as build_retain_loader().
    inputs = tokenizer(retain_sample, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    live_cap    = {}
    frozen_cap  = {}

    def live_hook(module, input, output):
        live_cap["h"] = output[0]

    def frozen_hook(module, input, output):
        frozen_cap["h"] = output[0]

    h_live_handle   = model.model.layers[RMU_LAYER].register_forward_hook(live_hook)
    h_frozen_handle = frozen_model.model.layers[RMU_LAYER].register_forward_hook(frozen_hook)

    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    with torch.no_grad():
        _ = frozen_model(**inputs)

    h_live_handle.remove()
    h_frozen_handle.remove()

    h_live   = live_cap["h"]
    h_frozen = frozen_cap["h"]

    loss_r = F.mse_loss(h_live.float(), h_frozen.float())

    print(f"  Live   model hidden state shape : {tuple(h_live.shape)}")
    print(f"  Frozen model hidden state shape : {tuple(h_frozen.shape)}")
    print()
    print(f"  retain_loss = MSE(h_live, h_frozen) = {loss_r.item():.8f}")
    print()
    print("  This is nearly zero because both models start with identical weights.")
    print("  As RMU trains, the live model's hidden states on retain text")
    print("  will drift from the frozen model. The retain loss penalizes that drift.")
    print()
    print("  Key insight: frozen_model acts as an anchor.")
    print("  It never updates → it always represents the 'before-unlearning' baseline.")
    print("  retain_loss ≈ 0 means the live model still behaves like the original.")
    print(SEP_THIN)


# ===========================================================================
# Section 6 — Two-pass dry run
# ===========================================================================

def section_two_pass_dry_run(
    model, frozen_model, tokenizer, device, c: torch.Tensor,
    forget_sample: str, retain_sample: str,
) -> None:
    """Demo: walk through exactly one RMU step without optimizer.step()."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 6: Two-Pass Dry Run (full step, no optimizer.step)")
    print(SEP_THICK)

    # Real samples — same sources as build_forget_loader() / build_retain_loader().

    f_inputs = tokenizer(forget_sample, return_tensors="pt", truncation=True, max_length=64)
    r_inputs = tokenizer(retain_sample, return_tensors="pt", truncation=True, max_length=64)
    f_inputs = {k: v.to(device) for k, v in f_inputs.items()}
    r_inputs = {k: v.to(device) for k, v in r_inputs.items()}

    live_store   = {}
    frozen_store = {}

    def live_hook(module, input, output):
        live_store["h"] = output[0]       # gradient flows back through this

    def frozen_hook(module, input, output):
        frozen_store["h"] = output[0]

    live_handle   = model.model.layers[RMU_LAYER].register_forward_hook(live_hook)
    frozen_handle = frozen_model.model.layers[RMU_LAYER].register_forward_hook(frozen_hook)

    print("  ── PASS 1: forget forward (live model) ──")
    model.train()
    _ = model(**f_inputs)
    h_forget = live_store["h"]
    print(f"    h_forget.shape       : {tuple(h_forget.shape)}")
    print(f"    h_forget.requires_grad: {h_forget.requires_grad}  ← must be True for backward()")

    print()
    print("  ── PASS 2: retain forward (frozen model, no_grad) ──")
    with torch.no_grad():
        _ = frozen_model(**r_inputs)
    h_frozen_retain = frozen_store["h"].detach()
    print(f"    h_frozen_retain.shape : {tuple(h_frozen_retain.shape)}")
    print(f"    h_frozen_retain.requires_grad: {h_frozen_retain.requires_grad}  ← must be False")

    print()
    print("  ── PASS 3: retain forward (live model) ──")
    _ = model(**r_inputs)
    h_live_retain = live_store["h"]
    print(f"    h_live_retain.shape  : {tuple(h_live_retain.shape)}")
    print(f"    h_live_retain.requires_grad: {h_live_retain.requires_grad}")

    print()
    print("  ── LOSS COMPUTATION ──")
    c_f = c.to(h_forget.dtype)
    target_forget = (ALPHA_RMU * c_f).expand_as(h_forget)
    loss_f = F.mse_loss(h_forget, target_forget)

    loss_r = F.mse_loss(h_live_retain, h_frozen_retain.to(h_live_retain.dtype))

    from unlearning.config import BETA_RMU
    total_loss = loss_f + BETA_RMU * loss_r

    print(f"    loss_f (forget MSE)  : {loss_f.item():.4f}")
    print(f"    loss_r (retain MSE)  : {loss_r.item():.8f}")
    print(f"    beta                 : {BETA_RMU}")
    print(f"    total = loss_f + beta*loss_r : {total_loss.item():.4f}")

    print()
    print("  ── BACKWARD (no optimizer.step) ──")
    total_loss.backward()
    print("    total_loss.backward() completed — gradients populated.")
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    if grad_norms:
        print(f"    Params with gradients : {len(grad_norms)}")
        print(f"    Max grad norm         : {max(grad_norms):.6f}")
        print(f"    Mean grad norm        : {sum(grad_norms)/len(grad_norms):.6f}")

    live_handle.remove()
    frozen_handle.remove()

    print()
    print("  This is what TODO 4 in 03_rmu_exercise.py will implement.")
    print("  The only difference: TODO 4 adds optimizer.step() + zero_grad() + the loop.")
    print(SEP_THIN)


# ===========================================================================
# Section 7 — Why MSE not CE?
# ===========================================================================

def section_why_mse() -> None:
    """Short conceptual explanation of why RMU uses MSE on hidden states."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 7: Why MSE and Not Cross-Entropy?")
    print(SEP_THICK)
    explanation = """
  RMU operates on continuous hidden-state VECTORS, not discrete tokens.

  Cross-entropy (CE) loss is defined over a probability distribution over
  the vocabulary — it measures how well the model assigns probability to
  the correct next token. It requires a discrete target (a token index).

  The misdirection target alpha*c is a continuous vector in R^hidden_size.
  There is no "correct token" — we want to push the hidden state to a
  specific point in representation space. MSE is the natural distance
  metric for continuous vectors in Euclidean space:

      loss_f = MSE(h, alpha*c)
             = (1/N) * sum_i (h_i - alpha*c_i)^2

  This gradient of MSE(h, alpha*c) w.r.t. h points directly from h toward
  alpha*c — exactly what we want. CE would require discretizing the target,
  losing the directional information in the representation space.

  The retain loss uses the same MSE reasoning:
      loss_r = MSE(h_live, h_frozen)
  We want h_live to STAY CLOSE to h_frozen — again a continuous-space
  distance, perfect for MSE.

  Summary:
    GA / GD  → output space (token probs)   → CE loss
    RMU      → representation space (hidden states) → MSE loss
"""
    print(explanation)
    print(SEP_THIN)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print(f"\n{SEP_THICK}")
    print("  02_rmu_exploration.py — RMU Primitives Demo")
    print(f"{SEP_THICK}")
    print("  No training. No TODOs. Read-only exploration.")
    print("  Run 03_rmu_exercise.py after this to implement RMU yourself.")
    print()

    device = get_device()

    # ── Load real samples from the actual datasets ───────────────────────────
    forget_sample, retain_sample = _load_real_samples()

    # ── Load live model ──────────────────────────────────────────────────────
    print("[load] Loading live model (not frozen) ...")
    model, tokenizer = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=False)
    hidden_size = model.config.hidden_size

    # ── Section 1: architecture recap ───────────────────────────────────────
    section_architecture_recap(model)

    # ── Section 2: forward hooks ─────────────────────────────────────────────
    section_forward_hooks(model, tokenizer, device, forget_sample)

    # ── Section 3: random misdirection vector ───────────────────────────────
    c = section_random_vector(hidden_size, device)

    # ── Section 4: forget loss preview ──────────────────────────────────────
    section_forget_loss_preview(model, tokenizer, device, c, forget_sample)

    # ── Load frozen model for retain sections ───────────────────────────────
    print("\n[load] Loading frozen reference model ...")
    frozen_model, _ = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=True)

    # ── Section 5: retain loss preview ──────────────────────────────────────
    section_retain_loss_preview(model, frozen_model, tokenizer, device, retain_sample)

    # ── Section 6: two-pass dry run ─────────────────────────────────────────
    section_two_pass_dry_run(model, frozen_model, tokenizer, device, c,
                             forget_sample, retain_sample)

    # ── Section 7: why MSE ───────────────────────────────────────────────────
    section_why_mse()

    print(f"\n{SEP_THICK}")
    print("  EXPLORATION COMPLETE")
    print(f"{SEP_THICK}")
    print("  Next: python unlearning/03_rmu_exercise.py --steps 5 --forget-size 20")
    print()


if __name__ == "__main__":
    main()
