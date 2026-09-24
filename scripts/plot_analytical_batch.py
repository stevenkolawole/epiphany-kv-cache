#!/usr/bin/env python3
"""Appendix figure: maximum concurrent requests on one 80 GB GPU against context length, with
and without a fixed cache budget. Pure KV arithmetic; reads reports/analytical_batch_80gb.json
(scripts_phase2/analytical_batch_throughput.py). Drawn with figstyle at 0.85 text width.

  python scripts/plot_analytical_batch.py --out ../EpiKV-overleaf/figures/analytical_batch_80gb.pdf
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "reports" / "analytical_batch_80gb.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.load(open(a.input))
    xs = sorted(int(n) for n in d["no_eviction_batch"])
    none = [d["no_eviction_batch"][str(n)] for n in xs]
    fs.use()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(fs.TEXT_W * 0.85, 2.2))
    ax.plot(xs, none, marker="o", ms=3, lw=1.2, color=fs.GREY)
    ax.annotate("no eviction", (xs[1], none[1]), xytext=(6, 4), textcoords="offset points", color=fs.GREY, fontsize=7)
    styles = {"2048": ("-", "s"), "4096": ("--", "^")}
    for K, (ls, mk) in styles.items():
        ys = [d["eviction_batch"][K][str(n)] for n in xs]
        ax.plot(xs, ys, ls=ls, marker=mk, ms=3, lw=1.2, color=fs.BLUE)
        ax.annotate(f"eviction, $K$ = {int(K):,}", (xs[0], ys[0]), xytext=(4, 4), textcoords="offset points",
                    ha="left", color=fs.BLUE, fontsize=7)
    # the one number: the gap at 32k tokens
    n0 = 32768
    lo, hi = d["no_eviction_batch"][str(n0)], d["eviction_batch"]["2048"][str(n0)]
    ax.annotate("", (n0, hi), (n0, lo), arrowprops=dict(arrowstyle="<->", color=fs.BLUE, lw=0.9))
    ax.annotate(f"{hi / lo:.0f}$\\times$", (n0, (hi + lo) / 2), xytext=(6, 0), textcoords="offset points",
                va="center", color=fs.BLUE, fontsize=8, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$2^{{{n.bit_length() - 1}}}$" for n in xs])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("max concurrent requests")
    ax.set_ylim(0, max(d["eviction_batch"]["2048"].values()) * 1.15)
    ax.grid(True, axis="y", color=fs.GREY_LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
