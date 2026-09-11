#!/usr/bin/env python3
"""The trade figure: accuracy against decode throughput, one point per method,
kernel-coded, both models, K=1024 (filled) and K=2048 (hollow).

    python scripts/plot_pareto.py --out ../EpiKV-overleaf/figures/pareto.pdf
    python scripts/plot_pareto.py --timing results/timing_llama results/timing_qwen ...

Accuracy comes from the logical MATH grids (results/llama_logical,
results/qwen_logical). Throughput is generated tokens / wall time per
process; by default it is read from the same grid cells (mixed load, two to
three processes per card, so absolute values are provisional), or from a
dedicated equal-load timing run when --timing dirs are given.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fs.use()
ROOT = Path(__file__).resolve().parent.parent / "results"

# method -> (label, kernel)
METHODS = {
    "kv_seg_hs": ("EpiKV-Seg", "fa2"),
    "hs_variance": ("HS-variance", "fa2"),
    "kv_key": ("Key-var", "fa2"),
    "kv_val": ("Val-var", "fa2"),
    "lag_kv": ("Lag-KV", "fa2"),
    "lag_kv_key": ("Lag-KV-key", "fa2"),
    "raas": ("RaaS", "eager"),
    "h2o": ("H2O", "eager"),
    "r_kv": ("R-KV", "eager"),
    "longflow": ("LongFlow", "eager"),
    "thinkv_faithful": ("ThinKV", "eager"),
}
LABELLED = {"kv_seg_hs", "raas", "h2o", "r_kv", "hs_variance", "thinkv_faithful"}


def load(d):
    """{(method, K): (acc, tok/s)} plus none."""
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        stem = os.path.basename(f)[:-5]
        if "tau128" in stem:
            continue
        j = json.load(open(f))
        for m, per in j["results"].items():
            for K, r in per.items():
                pp = [p for p in r["per_problem"] if p.get("wall_time_s")]
                tps = sum(p["n_tokens_generated"] for p in pp) / sum(p["wall_time_s"] for p in pp)
                out[(m, int(K))] = (r["accuracy"] * 100, tps)
    return out


def panel(ax, data, timing, title):
    none = data.get(("none", 1024))
    if none:
        ax.axvline(none[0], color=fs.GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.text(none[0], ax.get_ylim()[1] if False else 0, "", fontsize=6)
    for m, (label, kernel) in METHODS.items():
        col = fs.BLUE if kernel == "fa2" else fs.RED
        pts = []
        for K, filled in ((1024, True), (2048, False)):
            if (m, K) not in data:
                continue
            acc = data[(m, K)][0]
            tps = (timing.get((m, K)) or data[(m, K)])[1]
            pts.append((acc, tps, filled))
        if not pts:
            continue
        if len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=col, lw=0.6, alpha=0.5, zorder=1)
        for acc, tps, filled in pts:
            big = m == "kv_seg_hs"
            ax.scatter([acc], [tps], s=34 if big else 18, marker="s" if big else "o",
                       facecolor=col if filled else "white", edgecolor=col, linewidth=1.0, zorder=3)
        if m in LABELLED:
            acc, tps, _ = pts[0]
            ax.annotate(label, (acc, tps), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=col, fontweight="bold" if m == "kv_seg_hs" else "normal")
    if none:
        ax.text(none[0], ax.get_ylim()[0], "", fontsize=6)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("MATH-500 accuracy (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing", nargs="*", default=[], help="equal-load timing dirs: llama then qwen")
    a = ap.parse_args()
    llama = load(ROOT / "llama_logical")
    qwen = load(ROOT / "qwen_logical")
    t_l = load(a.timing[0]) if len(a.timing) > 0 else {}
    t_q = load(a.timing[1]) if len(a.timing) > 1 else {}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.15), sharey=False,
                                 gridspec_kw={"wspace": 0.32})
    panel(ax, llama, t_l, "Llama-8B")
    panel(bx, qwen, t_q, "Qwen-7B")
    ax.set_ylabel("decode tokens / s per process")
    for axis, data in ((ax, llama), (bx, qwen)):
        none = data.get(("none", 1024))
        if none:
            axis.text(none[0], axis.get_ylim()[1] * 0.98, "no eviction", fontsize=6.5, color=fs.GREY,
                      ha="right", va="top", rotation=90)
    # kernel legend, direct
    ax.text(0.02, 0.04, "filled = K 1024, hollow = K 2048", transform=ax.transAxes, fontsize=6, color=fs.INK)
    bx.text(0.02, 0.10, "blue: attention-free (FlashAttention)", transform=bx.transAxes, fontsize=6, color=fs.BLUE)
    bx.text(0.02, 0.04, "red: attention-based (eager kernel)", transform=bx.transAxes, fontsize=6, color=fs.RED)
    fs.panel_label(ax, "a")
    fs.panel_label(bx, "b")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)
    for name, data in (("Llama", llama), ("Qwen", qwen)):
        for K in (1024, 2048):
            row = sorted(((m, v) for (m, k), v in data.items() if k == K), key=lambda x: -x[1][0])
            print(name, K, "  ".join(f"{m} {v[0]:.0f}%/{v[1]:.0f}t/s" for m, v in row))


if __name__ == "__main__":
    main()
