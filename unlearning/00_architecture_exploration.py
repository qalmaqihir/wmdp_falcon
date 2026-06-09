"""
00_architecture_exploration.py — Understand Falcon3-1B before implementing unlearning.
========================================================================================
Run this FIRST before the GA/GD exercise. No training. Just loads the model,
prints its structure, and runs a single forward pass to show hidden state shapes.

WHY THIS MATTERS:
    GA and GD are architecture-independent — they only need `outputs.loss`
    from a standard forward pass. You don't need to know anything about
    Falcon3's internals to implement them.

    RMU (Exercise 02) is architecture-DEPENDENT. It hooks into hidden states
    at a specific transformer layer L. You need to know:
        - How many layers the model has (to pick L sensibly)
        - The hidden_size (to create the random misdirection vector)
        - The attribute path to reach layer L (differs across architectures)

    This script answers all three questions for Falcon3-1B and explains why
    they differ across model families. Read the output carefully before Exercise 02.

Usage:
    cd "/Users/jawadhaider/Study/Technical AI Safety Project/Falcon Day 1"
    source venv/bin/activate
    cd falcon_eval_wmdp
    python unlearning/00_architecture_exploration.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.utils import get_device, load_model_and_tokenizer, print_architecture_summary
from unlearning.config import HF_MODEL_ID


def explore_layer_internals(model) -> None:
    """Print the internal structure of a single transformer block."""
    sep = "-" * 60
    print(f"\n{sep}")
    print("  Decoder block structure (layer 0 children)")
    print(sep)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer0 = model.model.layers[0]
        for name, mod in layer0.named_children():
            # Count parameters
            n_params = sum(p.numel() for p in mod.parameters()) / 1e6
            print(f"  {name:<20} {type(mod).__name__:<30} ({n_params:.1f}M params)")
        print()
        print("  NOTE for RMU:")
        print("    The full block output = hidden state after all sub-modules.")
        print("    We will hook into the OUTPUT of model.model.layers[L] to get")
        print(f"    a tensor of shape (batch, seq_len, {model.config.hidden_size}).")
        print("    This is the vector we will 'misdirect' toward a random direction.")
    else:
        print("  Could not find model.model.layers — see model structure above.")
    print(sep)


def explore_hidden_states(model, tokenizer, device) -> None:
    """Run a forward pass with output_hidden_states=True and print shapes."""
    sep = "-" * 60
    print(f"\n{sep}")
    print("  Hidden state shapes (from a real forward pass)")
    print(sep)

    model.eval()
    with torch.no_grad():
        sample_text = "SARS-CoV-2 spike protein binds to the ACE2 receptor on human cells."
        inputs = tokenizer(sample_text, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (num_layers + 1) tensors.
    # hidden_states[0] = embedding output (before layer 0).
    # hidden_states[i] = output of layer i-1 (after layer i-1).
    # hidden_states[-1] = output of the last layer.
    hs = outputs.hidden_states
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2

    print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"  hidden_states is a tuple of {len(hs)} tensors (embedding + {n_layers} layers)")
    print(f"  Each tensor shape: (batch=1, seq_len, hidden_size={model.config.hidden_size})")
    print()
    print(f"  Layer  0 (embedding output): {tuple(hs[0].shape)}")
    print(f"  Layer  {mid} (middle):          {tuple(hs[mid].shape)}")
    print(f"  Layer {n_layers} (final):          {tuple(hs[n_layers].shape)}")
    print()
    print("  RMU accesses hidden_states[L] via a forward hook (not direct indexing).")
    print("  But the shape it expects = the shape shown above.")
    print(f"\n  Recommended RMU target layer: L = {mid} (index {mid} of {n_layers} layers)")
    print(f"  Random vector for RMU will have shape: (hidden_size={model.config.hidden_size},)")
    print(sep)


def print_architecture_comparison() -> None:
    """Print a comparison table of layer access paths across common model families."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Layer Access Paths — Architecture Comparison")
    print("  (Critical for RMU; not needed for GA/GD)")
    print(sep)

    rows = [
        ("Model family",          "Layer access path",            "Hidden size"),
        ("─" * 20,                "─" * 30,                       "─" * 12),
        ("Falcon3-1B/3B/7B/10B",  "model.model.layers[L]",        "2048/3072/4096/4096"),
        ("LLaMA-2/3, Mistral",    "model.model.layers[L]",        "4096 (7B)"),
        ("Old Falcon (1B/7B/40B)","model.transformer.h[L]",       "2048/4544/8192"),
        ("GPT-2",                 "model.transformer.h[L]",       "768 (base)"),
        ("Phi-3/4",               "model.model.layers[L]",        "3072 (mini)"),
        ("Gemma-2",               "model.model.layers[L]",        "2304 (2B)"),
    ]
    for row in rows:
        print(f"  {row[0]:<28}  {row[1]:<32}  {row[2]}")

    print()
    print("  KEY LESSON:")
    print("  Falcon3 uses the same LLaMA-style path as most modern open models.")
    print("  Old Falcon (pre-2024) used model.transformer.h — different layout.")
    print("  ALWAYS verify with print(model) or print_architecture_summary() before")
    print("  implementing RMU on a new model family.")
    print(sep)


def print_ga_gd_independence() -> None:
    """Explain why GA/GD don't need any of the above information."""
    sep = "-" * 60
    print(f"\n{sep}")
    print("  Why GA and GD are architecture-INDEPENDENT")
    print(sep)
    print()
    print("  GA loss:  loss = -model(input_ids, labels=input_ids).loss")
    print()
    print("  That's it. `outputs.loss` is the standard cross-entropy averaged")
    print("  over all tokens. It's computed in the language model head (lm_head),")
    print("  which is architecture-agnostic. Whether the backbone is Falcon3,")
    print("  LLaMA, Mistral, or GPT-2, the loss interface is identical.")
    print()
    print("  GD adds one line: retain_loss = model(retain_ids, labels=retain_ids).loss")
    print("  Same interface. Still architecture-independent.")
    print()
    print("  RMU breaks this pattern: instead of using the final loss, it computes")
    print("  a CUSTOM loss on INTERMEDIATE hidden states. That requires knowing")
    print("  the exact attribute path and hidden_size — hence architecture-dependent.")
    print(sep)


def main() -> None:
    print("\n" + "=" * 60)
    print("  Falcon3-1B Architecture Exploration")
    print("  Run this before Exercise 01 (GA/GD) and Exercise 02 (RMU)")
    print("=" * 60)

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(HF_MODEL_ID, device, frozen=True)

    # 1. High-level structural summary.
    print_architecture_summary(model)

    # 2. Internal decoder block.
    explore_layer_internals(model)

    # 3. Actual hidden state shapes from a forward pass.
    explore_hidden_states(model, tokenizer, device)

    # 4. Cross-architecture comparison table.
    print_architecture_comparison()

    # 5. Why GA/GD don't need any of this.
    print_ga_gd_independence()

    print("\nDone. You now know:")
    print(f"  - Falcon3-1B has {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}")
    print(f"  - Layer access path for RMU: model.model.layers[L]")
    print(f"  - Recommended RMU layer: L = {model.config.num_hidden_layers // 2}")
    print(f"  - GA/GD require none of the above — just outputs.loss")
    print("\nNext step: python unlearning/01_ga_gd_exercise.py --method ga --steps 5 --forget-size 20\n")


if __name__ == "__main__":
    main()
