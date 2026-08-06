#!/usr/bin/env python3
"""Regenerate the accuracy figures for the camera-ready.

Fixes three defects in the figures built on Aug 3:
  1. The MATH-500 H2O curve plotted the retracted run (1/5/49/67); the paper's
     text and tables now carry the clean n=50 rerun (0/22/46/64).
  2. Legends used internal method keys (hs_variance_detrend) in a published
     figure; they now use the paper's names (EpiKV-Flat).
  3. EpiKV-Seg was absent. It is plotted from the same source the paper's
     table uses (the H100 run: 34% at 1024, 56% at 2048), not from the local
     n=100 file, so figure and table cannot disagree.

Usage: python scripts/plot_paper_figures.py [--out ../EpiKV-overleaf/figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PHASE1 = REPO / "results" / "phase1"
MODAL = REPO.parent / "results_modal"

FA2 = {"hs_variance_detrend", "hs_variance", "band_adaptive_hs",
       "kv_val", "kv_key", "lag_kv_key", "lag_kv", "kv_seg_hs"}

# Paper names. Order controls legend order.
NAME = [
    ("hs_variance_detrend", "EpiKV-Flat"),
    ("kv_seg_hs",           "EpiKV-Seg"),
    ("hs_variance",         "HS-variance"),
    ("band_adaptive_hs",    "Band-adaptive"),
    ("kv_val",              "KV-val"),
    ("kv_key",              "KV-key"),
    ("lag_kv_key",          "Lag-KV-key"),
    ("lag_kv",              "Lag-KV"),
    ("thinKV",              "ThinKV"),
    ("h2o",                 "H2O"),
    ("raas",                "RaaS"),
    ("attn_hs_product",     r"Attn$\times$HS"),
]
COLOR = {
    "hs_variance_detrend": "tab:red", "kv_seg_hs": "firebrick",
    "hs_variance": "tab:orange", "band_adaptive_hs": "gold",
    "kv_val": "tab:blue", "kv_key": "tab:cyan",
    "lag_kv_key": "tab:purple", "lag_kv": "indigo",
    "thinKV": "tab:green", "h2o": "tab:gray", "raas": "tab:brown",
"attn_hs_product": "olive",
}


def load_merged(dataset):
    merged = {}
    for variant in ("eager", "flash"):
        p = PHASE1 / f"benchmark_{dataset}_{variant}.json"
        if not p.exists():
            continue
        for m, sizes in json.load(open(p))["results"].items():
            merged.setdefault(m, sizes)
    return merged


def series(merged, m):
    return {int(cs): 100 * v["accuracy"] for cs, v in merged.get(m, {}).items()
            if v.get("accuracy") is not None}


def plot(dataset, out_dir):
    merged = load_merged(dataset)
    data = {m: series(merged, m) for m, _ in NAME}

    if dataset == "math500":
        # Defect 1: clean-rerun H2O (n=50), the numbers the paper reports.
        h2o = json.load(open(MODAL / "e13_h2o.json"))["results"]["h2o"]
        data["h2o"] = {int(cs): 100 * v["accuracy"] for cs, v in h2o.items()}
        # Defect 3: EpiKV-Seg from the table's own source.
        seg = json.load(open(MODAL / "e1_kvseghs_tight.json"))["results"]["kv_seg_hs"]
        data["kv_seg_hs"] = {int(cs): 100 * v["accuracy"] for cs, v in seg.items()}
    else:
        data.pop("kv_seg_hs", None)   # no AIME run for Seg yet

    ceiling = merged.get("none", {})
    ceil_val = 100 * next(iter(ceiling.values()))["accuracy"] if ceiling else None

    # Sized for the width it is actually printed at (half of a two-column
    # page), not for a full-width figure that then gets scaled down.
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 8})
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    for m, label in NAME:
        pts = data.get(m)
        if not pts:
            continue
        xs = sorted(pts)
        ours = m in ("hs_variance_detrend", "kv_seg_hs")
        ax.plot(xs, [pts[x] for x in xs],
                "-" if m in FA2 else "--",
                marker="o", color=COLOR[m], label=label,
                linewidth=2.4 if ours else 1.0,
                alpha=1.0 if ours else 0.55,
                markersize=5 if ours else 3,
                zorder=3 if ours else 2)
    if ceil_val is not None:
        ax.axhline(ceil_val, color="black", linestyle=":", linewidth=1,
                   label=f"none ({ceil_val:.0f}%)")

    ax.set_xscale("log", base=2)
    xs_all = sorted({x for pts in data.values() for x in (pts or {})})
    ax.set_xticks(xs_all, [str(x) for x in xs_all])
    ax.set_xlabel("KV cache budget (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    out = out_dir / f"accuracy_{dataset}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO.parent / "EpiKV-overleaf" / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plot("math500", a.out)
    plot("aime2024", a.out)
