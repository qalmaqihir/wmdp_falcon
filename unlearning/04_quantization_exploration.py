"""
04_quantization_exploration.py — Quantization + LoRA for Unlearning
====================================================================
Run this BEFORE 05_quantized_rmu_exercise.py. No training, no TODOs, no argparse.

Sections:
  1. Memory math — parameter footprint in fp32 / bf16 / int8 / int4
  2. What is quantization? — NF4 vs INT8 vs INT4, brief comparison
  3. Why this matters for RMU — what changes and what stays the same
  4. Per-method impact of quantization — GA, GD, RMU, Task Vectors
  5. Device matrix — what runs where (CUDA / MPS / CPU)
  6. LoRA mechanics — wrap model, print trainable params, run forward pass
  7. What stays the same in Exercise 05 vs Exercise 03

WHY THIS MATTERS:
  Once weights are quantized (int4 or int8), they are frozen by construction.
  You cannot pass gradients through them with a standard optimizer.
  The solution: add LoRA (Low-Rank Adaptation) adapters — small trainable
  matrices that sit alongside the frozen quantized layers. Only LoRA params
  update; the quantized base weights never change.
  On M2 MPS, 4-bit/8-bit quantization via bitsandbytes is NOT available.
  Exercise 05 therefore uses bf16 + LoRA (full float, adapter-only training).

USAGE:
  cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
  source venv/bin/activate && cd falcon_eval_wmdp
  python unlearning/04_quantization_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.config import (
    HF_MODEL_ID,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    RMU_LAYER,
)
from unlearning.utils import (
    get_device,
    load_model_and_tokenizer,
    print_architecture_summary,
)


SEP_THICK = "=" * 64
SEP_THIN  = "-" * 64


# ===========================================================================
# Section 1 — Memory math
# ===========================================================================

def section_memory_math(model) -> None:
    """Compute and print parameter footprint in various dtypes."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 1: Memory Math — Falcon3-1B Parameter Footprint")
    print(SEP_THICK)

    n_params = sum(p.numel() for p in model.parameters())

    dtype_bytes = {
        "fp32":  4,
        "bf16":  2,
        "int8":  1,
        "int4":  0.5,  # 4 bits = 0.5 bytes
    }

    print(f"  Total parameters: {n_params / 1e9:.3f}B  ({n_params:,})")
    print()
    print(f"  {'dtype':<8}  {'bytes/param':>12}  {'total RAM':>12}  note")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*32}")
    for dtype, bpp in dtype_bytes.items():
        gb = n_params * bpp / 1e9
        notes = {
            "fp32":  "standard PyTorch default",
            "bf16":  "← Falcon3-1B loaded here (bfloat16)",
            "int8":  "bitsandbytes INT8, CUDA or CPU only",
            "int4":  "bitsandbytes NF4, CUDA only (MPS ✗)",
        }
        print(f"  {dtype:<8}  {bpp:>12.1f}  {gb:>11.2f}G  {notes[dtype]}")

    print()
    print("  Note: in practice, optimizer states (AdamW) add ~2x the model RAM")
    print("  during training. With LoRA, only ~2M params have optimizer states.")
    print(SEP_THIN)


# ===========================================================================
# Section 2 — What is quantization?
# ===========================================================================

def section_what_is_quantization() -> None:
    """Brief comparison of quantization approaches."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 2: What Is Quantization?")
    print(SEP_THICK)

    print("""
  Quantization maps floating-point weights to lower-precision integers.
  The forward pass de-quantizes on the fly → hidden states are still float.
  This means forward hooks on hidden states work IDENTICALLY to bf16 models.

  Three common schemes:
  ┌────────────┬──────────────────────────────────────────────────────┐
  │ NF4        │ 4-bit NormalFloat — used in QLoRA (Dettmers 2023).   │
  │            │ Designed for normally-distributed weights.            │
  │            │ Requires bitsandbytes + CUDA. MPS not supported.     │
  ├────────────┼──────────────────────────────────────────────────────┤
  │ INT8       │ 8-bit integer — simpler, broader hardware support.   │
  │            │ bitsandbytes supports CPU (slow) + CUDA.             │
  │            │ MPS not supported.                                   │
  ├────────────┼──────────────────────────────────────────────────────┤
  │ INT4       │ Generic 4-bit (not NF4). Also CUDA-only.             │
  │            │ Lower quality than NF4 for LLMs.                     │
  └────────────┴──────────────────────────────────────────────────────┘

  Weight quantization: compress stored weights. Forward pass dequantizes.
  Activation quantization: also compress activations at inference time.
    → Much harder to train with. Not relevant for these exercises.
""")
    print(SEP_THIN)


# ===========================================================================
# Section 3 — Why this matters for RMU
# ===========================================================================

def section_why_matters_for_rmu() -> None:
    """Explain what changes when the base model is quantized."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 3: Why Quantization Matters for RMU")
    print(SEP_THICK)

    print("""
  WHAT STAYS THE SAME after quantization:
    ✓  Hidden states are still float tensors (dequant happens inside forward).
    ✓  Forward hook code (make_hidden_state_hook) is IDENTICAL.
    ✓  Forget loss (rmu_forget_loss) is IDENTICAL.
    ✓  Retain loss (rmu_retain_loss) is IDENTICAL.
    ✓  Frozen reference model concept is IDENTICAL
       (quantized backbone is already frozen by construction).

  WHAT CHANGES:
    ✗  Base model weights are frozen (quantized = not differentiable).
    ✗  Standard optimizer cannot update them.
    ✓  Solution: add LoRA adapters to selected layers.
       LoRA params are fp32 / bf16 — fully differentiable.
       Optimizer updates ONLY LoRA params (~2M vs ~1B for full model).

  Net effect: same RMU algorithm, different set of trainable parameters.
  Hook captures the same hidden states; loss math is identical.
  Only the optimizer scope changes: model.parameters() → LoRA params only.
  PEFT's get_peft_model() handles this automatically (requires_grad=False
  for base weights, requires_grad=True for LoRA params).
""")
    print(SEP_THIN)


# ===========================================================================
# Section 4 — Per-method quantization impact
# ===========================================================================

def section_per_method_impact() -> None:
    """Show what changes for each unlearning method when weights are quantized."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 4: Per-Method Impact of Quantization")
    print(SEP_THICK)

    print("""
  Method         │ What changes with quantized base weights
  ───────────────┼──────────────────────────────────────────────────────────
  GA             │ Needs LoRA. CE loss math is otherwise IDENTICAL.
                 │ Optimizer targets LoRA params instead of all params.
  ───────────────┼──────────────────────────────────────────────────────────
  GD             │ Same as GA above. Retain CE loss also unchanged.
  ───────────────┼──────────────────────────────────────────────────────────
  RMU            │ Needs LoRA. Hook code UNCHANGED. Forget/retain MSE
                 │ losses UNCHANGED. Frozen ref model is the quantized
                 │ base — already frozen by construction (no extra work).
  ───────────────┼──────────────────────────────────────────────────────────
  Task Vectors   │ INCOMPATIBLE. Task vectors subtract weight tensors in
                 │ float space (θ_finetuned - θ_pretrained). Quantized
                 │ weights are integers — subtraction loses meaning.
                 │ Would require dequantizing both checkpoints first.

  Summary: GA, GD, RMU all work with LoRA adapters on quantized bases.
  Task vectors require float-space weight arithmetic → dequant needed.
""")
    print(SEP_THIN)


# ===========================================================================
# Section 5 — Device matrix
# ===========================================================================

def section_device_matrix() -> None:
    """Print which quantization modes work on which backends."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 5: Device Matrix")
    print(SEP_THICK)

    print("""
  Backend             │ INT8        │ NF4 (4-bit) │ bf16 + LoRA
  ────────────────────┼─────────────┼─────────────┼────────────────
  CUDA (Linux/Win)    │  ✓          │  ✓          │  ✓
  MPS  (M2 Mac)       │  ✗          │  ✗          │  ✓  ← Exercise 05
  CPU  + bitsandbytes │  ~slow      │  ✗          │  ✓

  Why MPS doesn't support INT8/NF4:
    bitsandbytes uses CUDA-specific kernels (custom CUDA extensions).
    Apple MPS uses Metal shaders — no bitsandbytes MPS backend exists yet.

  Exercise 05 strategy:
    Use bf16 (full float) + LoRA adapters on MPS.
    This gives the LoRA adapter experience without 4-bit compression.
    The training loop is IDENTICAL to QLoRA — only the base dtype differs.
    Reference code for 4-bit/8-bit is included but wrapped in cuda-only guards.
""")
    print(SEP_THIN)


# ===========================================================================
# Section 6 — LoRA mechanics
# ===========================================================================

def section_lora_mechanics(model, tokenizer, device) -> None:
    """Wrap model with LoRA, print trainable params, verify hidden states work."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 6: LoRA Mechanics — Live Demo")
    print(SEP_THICK)

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("  peft not installed. Install with:")
        print("    pip install peft")
        print("  Skipping LoRA demo.")
        print(SEP_THIN)
        return

    print(f"  LoRA config:")
    print(f"    r              = {LORA_R}   (rank — controls adapter capacity)")
    print(f"    lora_alpha     = {LORA_ALPHA}")
    print(f"    lora_dropout   = {LORA_DROPOUT}")
    print(f"    target_modules = {LORA_TARGET_MODULES}")
    print()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)

    # Count trainable vs total params.
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in peft_model.parameters())
    print(f"  Trainable parameters : {trainable / 1e6:.2f}M  ({trainable / total * 100:.3f}%)")
    print(f"  Total parameters     : {total / 1e6:.0f}M")
    print()
    print("  This is the key benefit of LoRA: instead of updating 1B params,")
    print("  the optimizer only tracks gradient state for ~2M adapter params.")

    print()
    print("  ── Verifying hidden states still work through PEFT wrapper ──")
    # NOTE: using neutral placeholder text — real forget data from build_forget_loader().
    sample = "<Dummy Text for forget data>"  # safe placeholder; no CBRN content here
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    captured = {}

    def hook_fn(module, input, output):
        captured["h"] = output[0]

    # PEFT wraps the model: peft_model.base_model.model.model.layers[L]
    # (one extra .base_model wrapper from PEFT's PeftModel → PeftModelForCausalLM).
    layer = peft_model.base_model.model.model.layers[RMU_LAYER]
    handle = layer.register_forward_hook(hook_fn)

    peft_model.eval()
    with torch.no_grad():
        _ = peft_model(**inputs)
    handle.remove()

    h = captured["h"]
    print(f"  Hidden state shape  : {tuple(h.shape)}")
    print(f"  Hidden state dtype  : {h.dtype}")
    print(f"  Hidden state norm   : {h.float().norm().item():.2f}")
    print()
    print("  Hook works through the PEFT wrapper. ✓")
    print()
    print("  IMPORTANT layer path difference:")
    print("    Without PEFT: model.model.layers[L]")
    print("    With PEFT   : peft_model.base_model.model.model.layers[L]")
    print("  This is what TODO 1 in 05_quantized_rmu_exercise.py addresses.")
    print(SEP_THIN)


# ===========================================================================
# Section 7 — What stays the same in Exercise 05
# ===========================================================================

def section_what_stays_the_same() -> None:
    """Bullet recap of unchanged code from Exercise 03 to Exercise 05."""
    print(f"\n{SEP_THICK}")
    print("  SECTION 7: What Stays the Same in Exercise 05 vs Exercise 03")
    print(SEP_THICK)

    print("""
  UNCHANGED (import directly from 03_rmu_exercise):
    ✓  rmu_forget_loss()   — MSE loss math is identical
    ✓  rmu_retain_loss()   — MSE loss math is identical
    ✓  build_rmu_random_vector() — seeded unit vector unchanged
    ✓  quick_wmdp_eval() / full_wmdp_eval() — eval is model-agnostic

  CHANGED in Exercise 05:
    •  Layer path for hooks:
           model.model.layers[L]
       →   peft_model.base_model.model.model.layers[L]

    •  Optimizer scope:
           AdamW(model.parameters(), lr=...)
       →   AdamW(
               (p for p in peft_model.parameters() if p.requires_grad),
               lr=...
           )
       (PEFT already sets requires_grad=False for base weights,
        requires_grad=True for LoRA params — just filter on it.)

    •  Model loading:
           load_model_and_tokenizer()  →  load_model_and_tokenizer()
           + get_peft_model(model, lora_config)

    •  Checkpoint saving:
           model.save_pretrained()   →  peft_model.save_pretrained()
       LoRA adapter is ~10 MB instead of ~2 GB. Much more convenient.
""")
    print(SEP_THIN)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print(f"\n{SEP_THICK}")
    print("  04_quantization_exploration.py — Quantization + LoRA Demo")
    print(f"{SEP_THICK}")
    print("  No training. No TODOs. Read-only exploration.")
    print("  Run 05_quantized_rmu_exercise.py after this.")
    print()

    device = get_device()

    print("[load] Loading model for memory math and LoRA demo ...")
    model, tokenizer = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=False)

    section_memory_math(model)
    section_what_is_quantization()
    section_why_matters_for_rmu()
    section_per_method_impact()
    section_device_matrix()
    section_lora_mechanics(model, tokenizer, device)
    section_what_stays_the_same()

    print(f"\n{SEP_THICK}")
    print("  EXPLORATION COMPLETE")
    print(f"{SEP_THICK}")
    print("  Next: python unlearning/05_quantized_rmu_exercise.py --backend bf16-lora --steps 5 --forget-size 20")
    print()


if __name__ == "__main__":
    main()
