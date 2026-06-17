"""
plot_results.py — Generate all result figures from summary.csv.

Outputs (in figures/):
  fig1_bar_all_models.png/.pdf          — All models, accuracy + error bars
  fig2_scaling_falcon3.png/.pdf         — Log-param scaling for Falcon3 family
  fig3_metric_heatmap.png/.pdf          — Model × metric heatmap
  fig4_cdf_comparison.png/.pdf          — Per-sample score CDF (requires .eval logs)

Usage:
    python experiments/plot_results.py
    python experiments/plot_results.py --skip-cdf    # skip CDF (needs raw logs)
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive; safe in headless/script mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from experiments.config import (
    ALL_MODELS,
    FALCON_MODELS,
    FIGURES_DIR,
    PUBLISHED_RESULTS,
    RANDOM_CHANCE,
    RESULTS_PROCESSED_DIR,
    RESULTS_RAW_DIR,
    SUMMARY_CSV,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
FONTSIZE = 12
plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE + 1,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "legend.fontsize": FONTSIZE - 1,
    "figure.dpi": 150,
})

# Color-blind safe palette (Paul Tol's bright scheme)
FALCON_BLUE  = "#4477AA"
BASELINE_GREY = "#BBBBBB"
PUBLISHED_RED = "#EE6677"
RANDOM_COLOR  = "#999933"

FALCON_SHADES = ["#4477AA", "#66AABB", "#88CCEE", "#AADDCC"]  # 1B→10B

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows() -> list[dict]:
    if not SUMMARY_CSV.exists():
        print(f"No results yet. Run experiments first. (Expected: {SUMMARY_CSV})")
        return []
    rows = []
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def baseline_rows(rows: list[dict]) -> list[dict]:
    """Filter to no-system-prompt, no-CoT rows only."""
    return [
        r for r in rows
        if r.get("system_prompt", "none") == "none"
        and str(r.get("use_cot", "False")).lower() == "false"
    ]


def falcon_rows(rows: list[dict]) -> list[dict]:
    names = set(FALCON_MODELS.keys())
    return sorted(
        [r for r in rows if r.get("display_name", "") in names],
        key=lambda r: float(r.get("params_b", 0)),
    )


def other_rows(rows: list[dict]) -> list[dict]:
    names = set(FALCON_MODELS.keys())
    return sorted(
        [r for r in rows if r.get("display_name", "") not in names],
        key=lambda r: -float(r.get("accuracy", 0)),
    )


# ---------------------------------------------------------------------------
# Figure 1 — Bar chart: all models, sorted by accuracy
# ---------------------------------------------------------------------------

def fig1_bar_all_models(rows: list[dict]) -> None:
    data = baseline_rows(rows)
    if not data:
        print("Fig 1: no data yet.")
        return

    falcon_names = set(FALCON_MODELS.keys())
    data_sorted = sorted(data, key=lambda r: float(r.get("accuracy", 0)), reverse=True)

    names  = [r["display_name"] for r in data_sorted]
    accs   = [float(r["accuracy"]) * 100 for r in data_sorted]
    ci_los = [float(r["accuracy"]) * 100 - float(r["ci_lo"]) * 100 for r in data_sorted]
    ci_his = [float(r["ci_hi"]) * 100 - float(r["accuracy"]) * 100 for r in data_sorted]
    colors = [FALCON_BLUE if n in falcon_names else BASELINE_GREY for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, accs, color=colors, width=0.6, zorder=3)
    ax.errorbar(x, accs, yerr=[ci_los, ci_his], fmt="none", color="black",
                capsize=4, linewidth=1.5, zorder=4)

    # Random chance baseline
    ax.axhline(RANDOM_CHANCE * 100, color=RANDOM_COLOR, linestyle="--",
               linewidth=1.5, label=f"Random chance ({RANDOM_CHANCE:.0%})", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("WMDP-Bio Accuracy (%)")
    ax.set_title("WMDP-Bio Accuracy — Falcon3 Family vs. Baselines\n"
                 "(error bars = 95% Wilson CI)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    falcon_patch = mpatches.Patch(color=FALCON_BLUE, label="Falcon3 (our models)")
    baseline_patch = mpatches.Patch(color=BASELINE_GREY, label="Baselines (our runs)")
    ax.legend(handles=[falcon_patch, baseline_patch,
                       plt.Line2D([0], [0], color=RANDOM_COLOR, linestyle="--",
                                  label=f"Random chance ({RANDOM_CHANCE:.0%})")],
              loc="upper right")

    # Annotate values on bars
    for bar_, acc in zip(bars, accs):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + 1.2,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig1_bar_all_models")


# ---------------------------------------------------------------------------
# Figure 2 — Scaling: log(params) vs. accuracy for Falcon3 + published overlay
# ---------------------------------------------------------------------------

def fig2_scaling_falcon3(rows: list[dict]) -> None:
    f_rows = [r for r in baseline_rows(rows) if r.get("display_name", "") in FALCON_MODELS]
    if not f_rows:
        print("Fig 2: no Falcon3 data yet.")
        return

    f_sorted = sorted(f_rows, key=lambda r: float(r["params_b"]))
    xs = [math.log2(float(r["params_b"])) for r in f_sorted]
    ys = [float(r["accuracy"]) * 100 for r in f_sorted]
    yerr_lo = [float(r["accuracy"]) * 100 - float(r["ci_lo"]) * 100 for r in f_sorted]
    yerr_hi = [float(r["ci_hi"]) * 100 - float(r["accuracy"]) * 100 for r in f_sorted]
    labels = [r["display_name"] for r in f_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Falcon3 line + points
    ax.plot(xs, ys, "-o", color=FALCON_BLUE, linewidth=2, markersize=8,
            label="Falcon3 (our runs)", zorder=4)
    ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="none",
                color=FALCON_BLUE, capsize=4, linewidth=1.5, zorder=5)
    for x_, y_, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x_, y_), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color=FALCON_BLUE)

    # Linear fit (log scale)
    if len(xs) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(min(xs) - 0.3, max(xs) + 0.3, 100)
        ax.plot(x_fit, m * x_fit + b, "--", color=FALCON_BLUE, alpha=0.5,
                label=f"Linear fit: {m:+.1f}%/log₂-param  R²={_r2(xs,ys,m,b):.2f}")

    # Published reference lines (horizontal) — Li et al. 2024, logprob eval.
    # Plotted as dashed lines with labels rather than scatter points because:
    # (a) eval protocols differ (logprob vs text-gen), and
    # (b) GPT-4/Yi-34b/Mixtral parameter counts are not directly comparable.
    pub_line_styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (name, (acc, src)) in enumerate(PUBLISHED_RESULTS.items()):
        ls = pub_line_styles[i % len(pub_line_styles)]
        ax.axhline(acc, color=PUBLISHED_RED, linestyle=ls, linewidth=1.2, alpha=0.7,
                   label=f"{name}: {acc:.1f}% ({src})", zorder=2)

    # Random chance
    ax.axhline(RANDOM_CHANCE * 100, color=RANDOM_COLOR, linestyle=":",
               label=f"Random chance ({RANDOM_CHANCE:.0%})")

    # X axis labels: show actual param counts
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{2**x:.1f}B" for x in xs], rotation=15)
    ax.set_xlabel("Parameter count (log₂ scale)")
    ax.set_ylabel("WMDP-Bio Accuracy (%)")
    ax.set_title("WMDP-Bio Scaling: Falcon3 Family\n"
                 "(dashed = published logprob refs, Li et al. 2024 — different eval protocol)")
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "fig2_scaling_falcon3")


def _r2(xs, ys, m, b):
    ys_pred = [m * x + b for x in xs]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, ys_pred))
    ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Figure 3 — Metric heatmap: model × metric
# ---------------------------------------------------------------------------

def fig3_metric_heatmap(rows: list[dict]) -> None:
    data = baseline_rows(rows)
    if not data:
        print("Fig 3: no data yet.")
        return

    metrics = ["accuracy", "format_failures_pct", "tokens_per_sample", "time_per_sample_s"]
    metric_labels = ["Accuracy (%)", "Format Fail (%)", "Tokens/sample", "Time/sample (s)"]

    # Build matrix
    model_names = [r["display_name"] for r in data]
    matrix = []
    for r in data:
        n = int(r.get("n_total", 1)) or 1
        wall = float(r.get("wall_time_s", 0))
        tok_in = int(r.get("input_tokens", 0))
        tok_out = int(r.get("output_tokens", 0))
        row = [
            float(r["accuracy"]) * 100,
            int(r.get("format_failures", 0)) / n * 100,
            (tok_in + tok_out) / n,
            wall / n,
        ]
        matrix.append(row)

    mat = np.array(matrix)
    # Normalize each column to [0,1] for color mapping
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    col_range = np.where(col_max - col_min > 0, col_max - col_min, 1)
    mat_norm = (mat - col_min) / col_range

    fig, ax = plt.subplots(figsize=(8, max(4, len(model_names) * 0.6)))
    im = ax.imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title("Model Comparison Heatmap\n(green=better, normalized per metric)")

    # Annotate cells with actual values
    fmt_strs = ["{:.1f}%", "{:.1f}%", "{:.0f}", "{:.1f}s"]
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            ax.text(j, i, fmt_strs[j].format(val),
                    ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < mat_norm[i, j] < 0.7 else "white")

    # Colorbar annotation
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized score (per metric)", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig3_metric_heatmap")


# ---------------------------------------------------------------------------
# Figure 4 — CDF of per-sample scores (requires raw .eval logs)
# ---------------------------------------------------------------------------

def fig4_cdf_comparison(rows: list[dict]) -> None:
    """Load per-sample scores from .eval files and plot CDFs."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        print("Fig 4: inspect_ai not importable. Skipping CDF.")
        return

    data = baseline_rows(rows)
    if not data:
        print("Fig 4: no data. Skipping CDF.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    falcon_names = set(FALCON_MODELS.keys())
    plotted = False

    for r in data:
        log_path = r.get("log_path", "")
        if not log_path or not Path(log_path).exists():
            continue
        try:
            log = read_eval_log(log_path)
        except Exception as e:
            print(f"  Could not read {log_path}: {e}")
            continue

        scores_numeric = []
        for s in log.samples:
            if s.scores:
                v = list(s.scores.values())[0].value
                scores_numeric.append(1.0 if str(v).upper() == "C" else 0.0)

        if not scores_numeric:
            continue

        xs = np.sort(scores_numeric)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        name = r["display_name"]
        color = FALCON_BLUE if name in falcon_names else BASELINE_GREY
        ax.step(xs, ys, label=name, color=color, linewidth=2)
        plotted = True

    if not plotted:
        print("Fig 4: no .eval files readable. Skipping CDF.")
        plt.close(fig)
        return

    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.set_xlabel("Per-sample score (0=incorrect, 1=correct)")
    ax.set_ylabel("Cumulative fraction of samples")
    ax.set_title("CDF of Per-Sample WMDP-Bio Scores")
    ax.legend(loc="center left")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "fig4_cdf_comparison")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cdf", action="store_true", help="Skip Figure 4 (CDF)")
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        return

    print("Generating Figure 1 — bar chart …")
    fig1_bar_all_models(rows)

    print("Generating Figure 2 — scaling plot …")
    fig2_scaling_falcon3(rows)

    print("Generating Figure 3 — metric heatmap …")
    fig3_metric_heatmap(rows)

    if not args.skip_cdf:
        print("Generating Figure 4 — CDF …")
        fig4_cdf_comparison(rows)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
