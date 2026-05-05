#!/usr/bin/env python3
"""
analyze_phase1.py — summarize Phase 1 benchmark results.

Loads the four phase 1 JSONs (math500/aime2024 × eager/flash), merges them
per dataset, prints accuracy / wall-time / peak-GPU tables, and produces
two PDF plots (one per dataset) showing accuracy vs. cache size with
FA2-compatible methods as solid lines and attention-only methods dashed.

Usage:
    python scripts/analyze_phase1.py
    python scripts/analyze_phase1.py --results-dir /data/user_data/skolawol/kvcache/results/phase1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_RESULTS = Path("/data/user_data/skolawol/kvcache/results/phase1")
PLOTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "phase1_plots"

# Method classification — drives line style and grouping
FA2_METHODS = {
    "hs_variance", "hs_variance_detrend", "band_adaptive_hs",
    "kv_val", "kv_key", "lag_kv_key", "lag_kv",
}
EAGER_METHODS = {"h2o", "thinKV", "raas", "attn_hs_product", "hybrid_seg_hs"}

# Plot order (top to bottom in legend)
PLOT_ORDER = [
    "none",
    "hs_variance_detrend", "hs_variance", "band_adaptive_hs",
    "kv_val", "kv_key", "lag_kv_key", "lag_kv",
    "thinKV", "h2o", "raas",
    "hybrid_seg_hs", "attn_hs_product",
]

COLORS = {
    "none": "black",
    "hs_variance_detrend": "tab:red",
    "hs_variance": "tab:orange",
    "band_adaptive_hs": "gold",
    "kv_val": "tab:blue",
    "kv_key": "tab:cyan",
    "lag_kv_key": "tab:purple",
    "lag_kv": "indigo",
    "thinKV": "tab:green",
    "h2o": "tab:gray",
    "raas": "tab:brown",
    "hybrid_seg_hs": "tab:pink",
    "attn_hs_product": "olive",
}


def load_dataset(results_dir: Path, dataset: str) -> dict:
    """Merge eager + flash JSONs for one dataset.  Returns {method: {cs: result}}."""
    merged = {}
    for variant in ("eager", "flash"):
        path = results_dir / f"benchmark_{dataset}_{variant}.json"
        if not path.exists():
            print(f"  WARN: {path} missing")
            continue
        with open(path) as f:
            data = json.load(f)
        for method, sizes in data["results"].items():
            # If the same method appears in both files, the eager run wins
            # (only "none" should overlap, and they're identical).
            if method not in merged:
                merged[method] = sizes
    return merged


def print_table(merged: dict, dataset: str, field: str, fmt: str):
    cs_sorted = sorted({cs for s in merged.values() for cs in s}, key=int)
    print(f"\n{dataset} — {field}:")
    header = "  " + "method".ljust(22) + "FA2".ljust(6) + "".join(cs.rjust(10) for cs in cs_sorted)
    print(header)
    print("  " + "-" * (22 + 6 + 10 * len(cs_sorted)))
    for m in PLOT_ORDER:
        if m not in merged:
            continue
        fa2 = "✓" if m in FA2_METHODS or m == "none" else "✗"
        line = "  " + m.ljust(22) + fa2.ljust(6)
        for cs in cs_sorted:
            val = merged[m].get(cs, {}).get(field)
            line += (fmt.format(val) if val is not None else "-").rjust(10)
        print(line)


def plot_accuracy(merged: dict, dataset: str, ceiling_method: str = "none"):
    cs_sorted = sorted({cs for s in merged.values() for cs in s}, key=int)
    cs_int = [int(cs) for cs in cs_sorted]

    ceiling = merged.get(ceiling_method, {}).get(cs_sorted[0], {}).get("accuracy")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in PLOT_ORDER:
        if m not in merged or m == "none":
            continue
        ys = [merged[m].get(cs, {}).get("accuracy") for cs in cs_sorted]
        if all(y is None for y in ys):
            continue
        # gap-tolerant: drop None pairs
        xs_y = [(x, y) for x, y in zip(cs_int, ys) if y is not None]
        xs, ys_clean = zip(*xs_y)
        style = "-" if m in FA2_METHODS else "--"
        ax.plot(xs, [100 * y for y in ys_clean],
                style, marker="o", color=COLORS.get(m, "gray"),
                label=m, linewidth=1.8)

    if ceiling is not None:
        ax.axhline(100 * ceiling, color="black", linestyle=":", linewidth=1,
                   label=f"none (ceiling={100*ceiling:.0f}%)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(cs_int, [str(c) for c in cs_int])
    ax.set_xlabel("KV cache budget (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Phase 1 — {dataset}\n(solid = FA2-compatible, dashed = attention-required)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"accuracy_{dataset}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = p.parse_args()

    for dataset in ("math500", "aime2024"):
        merged = load_dataset(args.results_dir, dataset)
        if not merged:
            continue
        print_table(merged, dataset, "accuracy", "{:>8.1%}")
        print_table(merged, dataset, "mean_wall_time_s", "{:>8.1f}s")
        print_table(merged, dataset, "mean_peak_gpu_mb", "{:>7.0f}MB")
        plot_accuracy(merged, dataset)


if __name__ == "__main__":
    main()
