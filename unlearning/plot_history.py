"""
plot_history.py — Plot GA / GD unlearning training curves.

Reads history.csv files saved by 01_ga_gd_exercise.py and produces
two-panel figures: loss over steps and WMDP-bio accuracy over steps.

USAGE:
    # Single run:
    python unlearning/plot_history.py results/unlearning/ga_20240614_120000

    # Compare GA vs GD side-by-side:
    python unlearning/plot_history.py results/unlearning/ga_TIMESTAMP results/unlearning/gd_TIMESTAMP

    # Auto-discover all runs:
    python unlearning/plot_history.py --all

    # Save to custom path:
    python unlearning/plot_history.py --all --out figures/unlearning_curves.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "unlearning"
FIGURES_DIR  = PROJECT_ROOT / "figures"

# Color palette: GA = warm red, GD = cool blue, extras cycle onward.
_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """Return (metadata_dict, rows) where rows are dicts with keys step/loss/wmdp_acc."""
    meta_path    = run_dir / "metadata.json"
    history_path = run_dir / "history.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"No history.csv in {run_dir}. Run 01_ga_gd_exercise.py first.")

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    rows = []
    with open(history_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "step":     int(row["step"]),
                "loss":     float(row["loss"]),
                "wmdp_acc": float(row["wmdp_acc"]) if row["wmdp_acc"] else None,
            })

    return meta, rows


def _label_for(meta: dict, run_dir: Path) -> str:
    method = meta.get("method", run_dir.name.split("_")[0]).upper()
    steps  = meta.get("steps", "?")
    lr     = meta.get("lr")
    lr_str = f"  lr={lr:.0e}" if lr else ""
    return f"{method}  ({steps} steps{lr_str})"


def _discover_runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    dirs = sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "history.csv").exists()],
        key=lambda d: d.stat().st_mtime,
    )
    return dirs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_runs(run_dirs: list[Path], out_path: Path | None = None) -> None:
    if not run_dirs:
        print("[plot] No runs found. Train a model first.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_loss, ax_acc = axes

    for i, run_dir in enumerate(run_dirs):
        try:
            meta, rows = _load_run(run_dir)
        except FileNotFoundError as e:
            print(f"[plot] Skipping {run_dir.name}: {e}")
            continue

        color = _PALETTE[i % len(_PALETTE)]
        label = _label_for(meta, run_dir)

        steps     = [r["step"]     for r in rows]
        losses    = [r["loss"]     for r in rows]
        eval_rows = [r for r in rows if r["wmdp_acc"] is not None]
        eval_steps = [r["step"]     for r in eval_rows]
        eval_accs  = [r["wmdp_acc"] for r in eval_rows]

        # --- Loss panel ---
        ax_loss.plot(steps, losses, color=color, lw=1.4, label=label, alpha=0.9)

        # Smooth trend (window = 10% of steps, min 5)
        w = max(5, len(losses) // 10)
        if len(losses) >= w:
            kernel = np.ones(w) / w
            smooth = np.convolve(losses, kernel, mode="valid")
            x_smooth = steps[w - 1:]
            ax_loss.plot(x_smooth, smooth, color=color, lw=2.2, ls="--", alpha=0.6)

        # --- Accuracy panel ---
        if eval_steps:
            pre_acc = meta.get("pre_wmdp_acc")
            # Prepend baseline at step 0 if available
            if pre_acc is not None:
                plot_steps = [0] + eval_steps
                plot_accs  = [pre_acc] + eval_accs
            else:
                plot_steps = eval_steps
                plot_accs  = eval_accs

            ax_acc.plot(plot_steps, plot_accs, color=color, lw=2.0,
                        marker="o", markersize=5, label=label)

    # --- Loss panel formatting ---
    ax_loss.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.axhline(0, color="gray", lw=0.8, ls=":")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Accuracy panel formatting ---
    ax_acc.set_title("WMDP-bio Accuracy During Unlearning", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("WMDP-bio Accuracy")
    ax_acc.axhline(0.25, color="gray", lw=1.2, ls="--", label="Random chance (25%)")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("GA / GD Unlearning — Falcon3-1B-Instruct", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path is None:
        tags = set()
        for d in run_dirs:
            if (d / "history.csv").exists():
                try:
                    meta, _ = _load_run(d)
                    tags.add(meta.get("method", d.name.split("_")[0]).upper())
                except Exception:
                    pass
        method_tags = "_".join(sorted(tags))
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"unlearning_{method_tags or 'curves'}.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")

    # Also save PDF alongside PNG
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[plot] Saved → {pdf_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GA/GD unlearning training curves from history.csv files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dirs", nargs="*", type=Path,
        help="One or more run directories containing history.csv. "
             "Relative paths resolved from project root.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=f"Auto-discover all runs in {RESULTS_DIR}.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path. Default: figures/unlearning_<methods>.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        run_dirs = _discover_runs()
        if not run_dirs:
            print(f"[plot] No runs found in {RESULTS_DIR}.")
            sys.exit(1)
        print(f"[plot] Found {len(run_dirs)} run(s):")
        for d in run_dirs:
            print(f"       {d.name}")
    elif args.run_dirs:
        run_dirs = []
        for p in args.run_dirs:
            resolved = p if p.is_absolute() else PROJECT_ROOT / p
            run_dirs.append(resolved)
    else:
        # Default: show the most recent run
        run_dirs = _discover_runs()[-1:]
        if not run_dirs:
            print(f"[plot] No runs found. Pass a directory or use --all.")
            sys.exit(1)
        print(f"[plot] No args — plotting most recent run: {run_dirs[0].name}")

    plot_runs(run_dirs, out_path=args.out)


if __name__ == "__main__":
    main()
