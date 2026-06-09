"""
Unlearning module configuration.

All constants for GA, GD, and future RMU methods live here.
Mirrors experiments/config.py style — change here only, never inline.

Reference: UNLEARNING_METHODS.md in project root for theory behind each value.
"""
import sys
from pathlib import Path

# Expose experiments.config for import
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import PROJECT_ROOT, SEED  # noqa: E402

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
HF_MODEL_ID = "tiiuae/Falcon3-1B-Instruct"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# Forget set: two-tier. Primary is the canonical corpus from the WMDP paper.
# It may be gated on HF (requires access request). utils.py falls back
# automatically to WMDP-bio MCQ questions formatted as plain text.
FORGET_DATASET_PRIMARY  = ("cais/wmdp-corpora", "bio-remove-corpus", "train")
FORGET_DATASET_FALLBACK = ("cais/wmdp",          "wmdp-bio",          "test")

# Retain set: Wikitext-2, small and general, no auth required.
RETAIN_DATASET = ("wikitext", "wikitext-2-raw-v1", "train")

# ---------------------------------------------------------------------------
# Sequence length and batching
# ---------------------------------------------------------------------------
MAX_SEQ_LEN  = 512  # truncate/pad all sequences to this length
BATCH_SIZE   = 2    # M2 Mac safe: backward pass + optimizer state fits in RAM
EVAL_EVERY   = 25   # run quick WMDP eval every N training steps
EVAL_N_SAMPLES = 50 # samples for quick mid-run eval (~30 s on MPS)

# ---------------------------------------------------------------------------
# GA hyperparameters
# Source: UNLEARNING_METHODS.md — range from 1e-5 to 1e-4 for standard fine-tuning,
# but GA is aggressive so we start lower to avoid immediate catastrophic degradation.
# ---------------------------------------------------------------------------
LR_GA    = 5e-6
STEPS_GA = 100

# ---------------------------------------------------------------------------
# GD hyperparameters
# alpha: weight on the forget (GA) term.
# beta:  weight on the retain (CE) term.
# Both default to 1.0; tune alpha up / beta down to increase forgetting aggressiveness.
# ---------------------------------------------------------------------------
LR_GD    = 1e-5
STEPS_GD = 200
ALPHA_GD = 1.0
BETA_GD  = 1.0

# ---------------------------------------------------------------------------
# RMU hyperparameters
# Source: Li et al. (2024) — WMDP paper. Layer L = mid-network; alpha large
# to give a strong misdirection signal; beta balances the retain anchor.
# Paper uses alpha~1500 for 7B models; start lower for 1B (smaller hidden_size).
# ---------------------------------------------------------------------------
RMU_LAYER  = 9      # Falcon3-1B has 18 layers → middle = 9
LR_RMU     = 5e-5   # higher than GD; MSE gradients are smaller per-param than CE
STEPS_RMU  = 300
ALPHA_RMU  = 100.0  # misdirection magnitude (random vector scale)
BETA_RMU   = 1.0    # weight on retain loss

# ---------------------------------------------------------------------------
# LoRA / QLoRA hyperparameters (Exercise 05)
# ---------------------------------------------------------------------------
LORA_R               = 8
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.0
LORA_TARGET_MODULES  = ["q_proj", "v_proj"]  # LLaMA-style attention projections

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
UNLEARNING_RESULTS_DIR = PROJECT_ROOT / "results" / "unlearning"
