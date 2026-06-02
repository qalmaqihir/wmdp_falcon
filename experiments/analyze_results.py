"""
analyze_results.py — Parse all .eval logs and produce the results table.

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --csv-only   # skip .eval scan, use summary.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    ALL_MODELS,
    FALCON_MODELS,
    PUBLISHED_RESULTS,
    RANDOM_CHANCE,
    RESULTS_PROCESSED_DIR,
    RESULTS_RAW_DIR,
    SUMMARY_CSV,
)


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_summary_csv() -> list[dict]:
    if not SUMMARY_CSV.exists():
        return []
    rows = []
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def print_results_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found. Run experiments first.")
        return

    # Filter to baseline system prompt (apples-to-apples)
    baseline = [r for r in rows if r.get("system_prompt", "none") == "none" and r.get("use_cot", "False") == "False"]

    if not baseline:
        baseline = rows  # fallback: show everything

    # Sort: Falcon3 first by params, then baselines
    falcon_names = set(FALCON_MODELS.keys())
    falcon_rows = sorted(
        [r for r in baseline if r.get("display_name", "") in falcon_names],
        key=lambda r: float(r.get("params_b", 0)),
    )
    other_rows = sorted(
        [r for r in baseline if r.get("display_name", "") not in falcon_names],
        key=lambda r: -float(r.get("accuracy", 0)),
    )
    ordered = falcon_rows + other_rows

    print(f"\n{'Model':<22} {'Params':>7} {'Quant':<8} {'Acc':>7} {'95% CI':>15} {'n':>6} {'Fmt-fail':>9} {'±rand':>7}")
    print("-" * 84)
    for r in ordered:
        acc = float(r.get("accuracy", 0))
        ci_lo = float(r.get("ci_lo", 0))
        ci_hi = float(r.get("ci_hi", 0))
        n = int(r.get("n_total", 0))
        ff = int(r.get("format_failures", 0))
        delta = acc - RANDOM_CHANCE
        marker = " *" if ci_lo > RANDOM_CHANCE else "  "  # * = sig. above random
        print(
            f"{r.get('display_name','?'):<22} "
            f"{r.get('params_b','?'):>6}B "
            f"{r.get('quant','?'):<8} "
            f"{acc:>6.1%} "
            f"[{ci_lo:.1%}–{ci_hi:.1%}] "
            f"{n:>6} "
            f"{ff:>8} "
            f"{delta:>+6.1%}{marker}"
        )

    print("\n* = 95% CI lower bound > 25% random chance")

    # Published comparison
    print(f"\n{'--- Published WMDP-bio numbers (Li et al. 2024) ---':}")
    print(f"{'Model':<35} {'Acc':>7} {'Source'}")
    print("-" * 60)
    for name, (acc, src) in sorted(PUBLISHED_RESULTS.items(), key=lambda x: -x[1][0]):
        print(f"{name:<35} {acc:>6.1f}%  {src}")
    print()


def print_scaling_stats(rows: list[dict]) -> None:
    """Print Falcon3 scaling statistics."""
    falcon_names = set(FALCON_MODELS.keys())
    falcon_rows = sorted(
        [r for r in rows
         if r.get("display_name", "") in falcon_names
         and r.get("system_prompt", "none") == "none"
         and r.get("use_cot", "False") == "False"],
        key=lambda r: float(r.get("params_b", 0)),
    )

    if len(falcon_rows) < 2:
        print("Need ≥2 Falcon3 results for scaling analysis.")
        return

    import math
    xs = [math.log2(float(r["params_b"])) for r in falcon_rows]
    ys = [float(r["accuracy"]) * 100 for r in falcon_rows]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n--- Falcon3 Scaling Analysis ---")
    print(f"Slope:  {slope:+.2f}% accuracy per log2-parameter doubling")
    print(f"R²:     {r2:.3f}")
    print(f"Model:  accuracy(%) = {slope:.2f} * log2(params_B) + {intercept:.2f}")

    for r in falcon_rows:
        x = math.log2(float(r["params_b"]))
        pred = slope * x + intercept
        actual = float(r["accuracy"]) * 100
        print(f"  {r['display_name']:<15}  actual={actual:.1f}%  predicted={pred:.1f}%  residual={actual-pred:+.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    rows = load_summary_csv()
    print_results_table(rows)
    print_scaling_stats(rows)


if __name__ == "__main__":
    main()
