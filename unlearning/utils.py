"""
utils.py — Shared utilities for the unlearning module.

Fully implemented. Do not edit for the exercises — just import what you need.

Functions:
    get_device()                  → torch.device (MPS or CPU)
    load_model_and_tokenizer()    → (model, tokenizer)
    print_architecture_summary()  → prints model structure details
    build_forget_loader()         → DataLoader of forget (biosecurity) text
    build_retain_loader()         → DataLoader of retain (general) text
    quick_wmdp_eval()             → fast accuracy check on N WMDP-bio samples
    full_wmdp_eval()              → full accuracy check on all 1273 samples
    save_checkpoint()             → save model + metadata JSON to disk
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

# Make experiments.config importable when this file is run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from unlearning.config import (
    HF_MODEL_ID,
    FORGET_DATASET_PRIMARY,
    FORGET_DATASET_FALLBACK,
    RETAIN_DATASET,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EVAL_N_SAMPLES,
)


# ===========================================================================
# Device
# ===========================================================================

def get_device() -> torch.device:
    """Return MPS if available (M-series Mac), else CPU.

    Also enables the MPS fallback so ops not yet implemented on MPS
    silently fall back to CPU rather than crashing.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        print("[device] Using MPS (M-series GPU)")
        return torch.device("mps")
    print("[device] MPS unavailable — using CPU")
    return torch.device("cpu")


# ===========================================================================
# Model loading
# ===========================================================================

def load_model_and_tokenizer(
    model_id: str = HF_MODEL_ID,
    device: Optional[torch.device] = None,
    frozen: bool = False,
):
    """Load an AutoModelForCausalLM + AutoTokenizer.

    Args:
        model_id:  HuggingFace model ID or local checkpoint path.
        device:    Target device. If None, calls get_device().
        frozen:    If True, freeze all parameters and set eval mode.
                   Used for the reference model copy in RMU (next exercise).

    Returns:
        (model, tokenizer) both moved to device.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = get_device()

    print(f"[model] Loading {model_id} ...")
    t0 = time.perf_counter()

    # Load in bfloat16 to save RAM (~2 GB for 1B vs ~4 GB float32).
    # MPS supports bfloat16 in PyTorch 2.x.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    if frozen:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    else:
        model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Falcon3/Llama tokenizers have no pad token by default.
    # Set it to EOS so we can batch sequences of different lengths.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    elapsed = time.perf_counter() - t0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] Loaded {params_m:.0f}M params in {elapsed:.1f}s  dtype={model.dtype}  device={device}")

    return model, tokenizer


# ===========================================================================
# Architecture summary
# ===========================================================================

def print_architecture_summary(model) -> None:
    """Print key structural facts about the model.

    Used by both 00_architecture_exploration.py and 01_ga_gd_exercise.py.
    The information printed here becomes important for RMU (Exercise 02),
    where we need to hook into specific hidden states at a specific layer.
    """
    cfg = model.config
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  Architecture Summary: {type(model).__name__}")
    print(sep)
    print(f"  model_type         : {getattr(cfg, 'model_type', 'unknown')}")
    print(f"  architectures      : {getattr(cfg, 'architectures', ['unknown'])}")
    print(f"  num_hidden_layers  : {cfg.num_hidden_layers}")
    print(f"  hidden_size        : {cfg.hidden_size}")
    print(f"  intermediate_size  : {getattr(cfg, 'intermediate_size', 'N/A')}")
    print(f"  num_attention_heads: {cfg.num_attention_heads}")
    print(f"  vocab_size         : {cfg.vocab_size}")
    print(f"  torch_dtype        : {model.dtype}")

    # Detect which attribute path to use for layer access.
    print(f"\n  Layer access paths (relevant for RMU):")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        print(f"    model.model.layers  →  {n} decoder blocks  ✓  (LLaMA-style)")
        print(f"    For RMU: use model.model.layers[L] where L ∈ [0, {n-1}]")
        mid = n // 2
        print(f"    Suggested mid-layer for RMU: L = {mid}  (layer index {mid} of {n})")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        print(f"    model.transformer.h  →  {n} blocks  ✓  (old Falcon/GPT-2 style)")
    else:
        print(f"    Layer path unknown — inspect model manually (see 00_architecture_exploration.py)")

    print(f"\n  Top-level children:")
    for name, mod in model.named_children():
        print(f"    {name}: {type(mod).__name__}")

    print(sep + "\n")


# ===========================================================================
# Dataset helpers — internal
# ===========================================================================

class _TokenizedDataset(Dataset):
    """Simple dataset that holds pre-tokenized sequences."""

    def __init__(self, input_ids_list: list[torch.Tensor], pad_token_id: int):
        self.data = input_ids_list
        self.pad_id = pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        ids = self.data[idx]
        # Mask pad positions in labels with -100 (ignore_index in CE loss) so
        # padding tokens don't contribute gradients to the unlearning objective.
        labels = ids.clone()
        labels[labels == self.pad_id] = -100
        attention_mask = (ids != self.pad_id).long()
        return {
            "input_ids":      ids,
            "labels":         labels,
            "attention_mask": attention_mask,
        }


def _tokenize_texts(
    texts: list[str],
    tokenizer,
    max_len: int,
    n_samples: int,
) -> list[torch.Tensor]:
    """Tokenize a list of strings into fixed-length tensors."""
    result = []
    for text in texts:
        if len(result) >= n_samples:
            break
        if not text.strip():
            continue
        ids = tokenizer.encode(
            text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).squeeze(0)
        # Skip very short sequences (less signal than noise for unlearning).
        if ids.size(0) < 8:
            continue
        # Pad to max_len so batches are uniform.
        if ids.size(0) < max_len:
            pad_len = max_len - ids.size(0)
            ids = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)])
        result.append(ids)
    return result


# ===========================================================================
# Data loaders
# ===========================================================================

def build_forget_loader(
    tokenizer,
    n_samples: int = 200,
    max_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build a DataLoader of biosecurity forget text.

    Tries cais/wmdp-corpora (bio-remove-corpus) first — the canonical
    forget corpus used in the WMDP paper. If the dataset is gated on
    HuggingFace (requires access request), automatically falls back to
    formatting the WMDP-bio MCQ test split as plain text.

    Returns:
        DataLoader yielding dicts with keys input_ids, labels, attention_mask.
    """
    from datasets import load_dataset

    texts = None
    path, name, split = FORGET_DATASET_PRIMARY
    try:
        print(f"[forget] Loading {path}/{name} (split={split}) ...")
        ds = load_dataset(path, name, split=split, trust_remote_code=False)
        texts = [s["text"] for s in ds if s.get("text", "").strip()]
        print(f"[forget] Loaded {len(texts)} documents from wmdp-corpora (canonical source).")
    except Exception as e:
        if any(kw in str(e).lower() for kw in ["gated", "401", "403", "unauthorized", "restricted", "access"]):
            print(f"[forget] cais/wmdp-corpora is gated: {e}")
            print(f"[forget] Falling back to WMDP-bio MCQ formatted as text.")
        else:
            print(f"[forget] Failed to load wmdp-corpora ({e}). Falling back.")

    if texts is None:
        path_fb, name_fb, split_fb = FORGET_DATASET_FALLBACK
        ds = load_dataset(path_fb, name_fb, split=split_fb, trust_remote_code=False)
        letters = ["A", "B", "C", "D"]
        texts = []
        for s in ds:
            q = s["question"]
            choices = s["choices"]
            ans_letter = letters[s["answer"]]
            ans_text = choices[s["answer"]]
            formatted = (
                f"Question: {q}\n"
                f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
                f"Answer: {ans_letter}) {ans_text}\n\n"
            )
            texts.append(formatted)
        print(f"[forget] Loaded {len(texts)} MCQ samples (fallback source).")

    # Edge case: corpus loaded but all rows were empty after strip().
    if not texts:
        texts = None  # force fallback path below if this triggers again

    ids_list = _tokenize_texts(texts or [], tokenizer, max_len, n_samples)
    print(f"[forget] Tokenized {len(ids_list)} sequences (max_len={max_len}).")
    return DataLoader(
        _TokenizedDataset(ids_list, pad_token_id=tokenizer.pad_token_id),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )


def build_retain_loader(
    tokenizer,
    n_samples: int = 200,
    max_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build a DataLoader of general retain text (Wikitext-2).

    The retain loader is used in Gradient Difference (GD) to prevent
    the model from forgetting general language capability while it
    forgets the biosecurity domain.

    Returns:
        DataLoader yielding dicts with keys input_ids, labels, attention_mask.
    """
    from datasets import load_dataset

    path, name, split = RETAIN_DATASET
    print(f"[retain] Loading {path}/{name} (split={split}) ...")
    ds = load_dataset(path, name, split=split, trust_remote_code=False)
    texts = [s["text"] for s in ds if s.get("text", "").strip()]
    print(f"[retain] Loaded {len(texts)} documents from Wikitext-2.")

    ids_list = _tokenize_texts(texts, tokenizer, max_len, n_samples)
    print(f"[retain] Tokenized {len(ids_list)} sequences (max_len={max_len}).")
    return DataLoader(
        _TokenizedDataset(ids_list, pad_token_id=tokenizer.pad_token_id),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )


# ===========================================================================
# Evaluation
# ===========================================================================

def _run_wmdp_eval(
    model,
    tokenizer,
    device: torch.device,
    n_samples: int,
    desc: str = "eval",
) -> dict:
    """Internal: run WMDP-bio MCQ eval on the first n_samples questions.

    Does NOT use Inspect AI — runs the HF model directly with greedy
    1-token generation. Faster but lower fidelity than the full pipeline.
    Suitable for mid-training monitoring and before/after comparison.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test", trust_remote_code=False)
    samples = list(ds)[:n_samples]

    letters = ["A", "B", "C", "D"]
    n_correct = 0
    t0 = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for s in samples:
            q       = s["question"]
            choices = s["choices"]
            target  = letters[s["answer"]]

            prompt = (
                f"Question: {q}\n"
                f"A) {choices[0]}\nB) {choices[1]}\n"
                f"C) {choices[2]}\nD) {choices[3]}\n"
                f"Answer:"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1]
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Slice only the newly generated token(s) — out contains input + generated ids.
            gen_token = tokenizer.decode(out[0, input_len:]).strip()

            # Match first A/B/C/D character in the generated token.
            m = re.search(r"[A-D]", gen_token.upper())
            predicted = m.group(0) if m else "?"
            if predicted == target:
                n_correct += 1

    elapsed = time.perf_counter() - t0
    accuracy = n_correct / n_samples if n_samples > 0 else 0.0

    return {
        "accuracy":    round(accuracy, 4),
        "n_correct":   n_correct,
        "n_total":     n_samples,
        "eval_time_s": round(elapsed, 1),
    }


def quick_wmdp_eval(
    model,
    tokenizer,
    device: torch.device,
    n_samples: int = EVAL_N_SAMPLES,
) -> dict:
    """Fast WMDP-bio accuracy check (default: 50 samples, ~30 s on MPS).

    Use this during training (every EVAL_EVERY steps) to monitor forgetting
    without stalling the training loop.

    Returns:
        dict with keys: accuracy, n_correct, n_total, eval_time_s
    """
    result = _run_wmdp_eval(model, tokenizer, device, n_samples, desc="quick-eval")
    print(
        f"  [quick-eval] {n_samples} samples  "
        f"accuracy={result['accuracy']:.1%}  "
        f"({result['n_correct']}/{result['n_total']})  "
        f"time={result['eval_time_s']}s"
    )
    return result


def full_wmdp_eval(
    model,
    tokenizer,
    device: torch.device,
) -> dict:
    """Full WMDP-bio accuracy check on all 1273 samples.

    Use this after training for benchmark-grade post-unlearning numbers.
    Runs the HF model directly (not via Inspect AI).
    Comparable to the Inspect AI pipeline but ~3x faster due to no overhead.
    Expected runtime: ~15–20 min on M2 Max MPS for Falcon3-1B.

    Returns:
        dict with keys: accuracy, n_correct, n_total, eval_time_s
    """
    print("[full-eval] Running full WMDP-bio eval (1273 samples) ...")
    result = _run_wmdp_eval(model, tokenizer, device, n_samples=1273, desc="full-eval")
    print(
        f"  [full-eval] FINAL: accuracy={result['accuracy']:.1%}  "
        f"({result['n_correct']}/{result['n_total']})  "
        f"time={result['eval_time_s']/60:.1f} min"
    )
    return result


# ===========================================================================
# Checkpointing
# ===========================================================================

def save_checkpoint(
    model,
    tokenizer,
    out_dir: Path,
    metadata: dict,
) -> None:
    """Save model weights + tokenizer + metadata JSON to out_dir.

    Saved model can be loaded later with:
        model, tok = load_model_and_tokenizer(str(out_dir), device)

    Args:
        model:     The (possibly unlearned) model.
        tokenizer: Matching tokenizer.
        out_dir:   Output directory (created if missing).
        metadata:  Arbitrary dict saved alongside as metadata.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[checkpoint] Saving to {out_dir} ...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[checkpoint] Saved model, tokenizer, and {meta_path.name}.")
