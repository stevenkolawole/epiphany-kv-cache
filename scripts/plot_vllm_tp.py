#!/usr/bin/env python3
"""Figure for EpiKV 4.5: decode throughput and peak KV per request against
concurrency, EpiKV-Seg eviction inside vLLM vs. the same engine without
eviction. Reads results/vllm_port/tp/tp_B<B>_K<K>.json.

    python scripts/plot_vllm_tp.py --out ../EpiKV-overleaf/figures/vllm_tp.pdf
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fs.use()
D = Path(__file__).resolve().parent.parent / "results" / "vllm_port" / "tp"


def load():
    rows = {}
    for f in glob.glob(str(D / "tp_B*_K*.json")):
        d = json.load(open(f))
        rows[(d["meta"]["batch"], d["meta"]["cache_size"])] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load()
    Bs = sorted({b for b, _ in rows})
    none = {b: rows[(b, 1000000)] for b in Bs if (b, 1000000) in rows}
    seg = {b: rows[(b, 1024)] for b in Bs if (b, 1024) in rows}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 1.9),
                                 gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.38})
    # (a) throughput
    xs = sorted(none); ax.plot(xs, [none[b]["tokens_per_s"] for b in xs], marker="o", color=fs.GREY, lw=1.2, ms=3.5)
    xs = sorted(seg); ax.plot(xs, [seg[b]["tokens_per_s"] for b in xs], marker="s", color=fs.BLUE, lw=1.4, ms=3.5)
    # labels in the empty lower-right region, as a direct legend
    ax.plot([90, 150], [190, 190], color=fs.GREY, lw=1.2); ax.plot([115], [190], marker="o", color=fs.GREY, ms=3.5)
    ax.text(180, 190, "no eviction", color=fs.GREY, fontsize=7, va="center")
    ax.plot([90, 150], [120, 120], color=fs.BLUE, lw=1.4); ax.plot([115], [120], marker="s", color=fs.BLUE, ms=3.5)
    ax.text(180, 120, "EpiKV-Seg, $K$=1024", color=fs.BLUE, fontsize=7, va="center")
    if 256 in none and 256 in seg:
        ax.annotate("KV pool full:\ncapped engine ahead", xy=(256, seg[256]["tokens_per_s"]),
                    xytext=(256, seg[256]["tokens_per_s"] * 0.25), fontsize=6.5, color=fs.BLUE,
                    ha="center", va="top", arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.BLUE))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [b for b in Bs if b in (1, 8, 32, 128, 512)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(b) for b in ticks])
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("decode tokens / s")
    fs.panel_label(ax, "a")
    # (b) peak KV per request
    xs = sorted(none); bx.plot(xs, [none[b]["peak_kv_tokens_per_req"] for b in xs], marker="o", color=fs.GREY, lw=1.2, ms=3.5)
    xs = sorted(seg); bx.plot(xs, [seg[b]["peak_kv_tokens_per_req"] for b in xs], marker="s", color=fs.BLUE, lw=1.4, ms=3.5)
    bx.axhline(1024, color=fs.BLUE, lw=0.5, ls=(0, (3, 2)))
    bx.text(Bs[0], 1024 * 1.08, "$K$", color=fs.BLUE, fontsize=7, va="bottom")
    bx.set_xscale("log", base=2)
    bx.set_xticks(ticks)
    bx.set_xticklabels([str(b) for b in ticks])
    bx.text(1.2, 3850, "no eviction", color=fs.GREY, fontsize=7, va="top")
    bx.text(1.2, 1150, "EpiKV-Seg", color=fs.BLUE, fontsize=7, va="bottom")
    if 1024 in none:
        bx.text(1024, 1350, "both engines\npool-bound", color=fs.GREY, fontsize=6.5, ha="right", va="bottom")
    bx.set_xlabel("concurrent requests")
    bx.set_ylabel("peak KV tokens per request")
    bx.set_ylim(0, 4800)
    fs.panel_label(bx, "b")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
