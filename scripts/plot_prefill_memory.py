#!/usr/bin/env python3
"""Figure 1 (right): peak GPU memory of one forward pass against context length, attention
scoring (eager, output_attentions=True) against the FlashAttention pass our score runs on.
Reads reports/prefill_memory_80gb.json (bench_prefill_memory.py). Drawn at the width it is
printed at (0.35 text width) with figstyle.

  python scripts/plot_prefill_memory.py --out ../EpiKV-overleaf/figures/prefill_memory.pdf
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
    ap.add_argument("--input", type=Path, default=ROOT / "reports" / "prefill_memory_80gb.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.load(open(a.input))
    gpu = d.get("gpu_gb", 80.0)
    res = d["results"]
    fs.use()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(fs.TEXT_W * 0.35, 1.55))
    # "flash" is a FlashAttention-2 forward pass that returns the hidden states, which is what EpiKV reads;
    # "eager_attn" returns the attention weights, which is what an attention-based policy reads.
    series = {"eager_attn": (fs.RED, "o", "attention-based\nscoring"), "flash": (fs.BLUE, "s", "EpiKV")}
    for mode, (c, mk, label) in series.items():
        rows = sorted(res[mode].values(), key=lambda r: r["seq_len"])
        ok = [r for r in rows if not r["oom"]]
        xs = [r["seq_len"] for r in ok]
        ys = [r["peak_gpu_mb"] / 1024 for r in ok]
        ax.plot(xs, ys, marker=mk, ms=2.6, lw=1.2, color=c)
        oom = [r["seq_len"] for r in rows if r["oom"]]
        if oom:
            ax.scatter([min(oom)], [gpu], marker="x", s=28, color=c, lw=1.2, zorder=5, clip_on=False)
            ax.annotate("OOM", (min(oom), gpu), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=6.5, color=c)
        if mode == "flash":      # under the flat part of the curve, clear of both lines
            ax.annotate(label, (2 ** 13, 19), xytext=(2, -4), textcoords="offset points", ha="center", va="top",
                        fontsize=6.5, color=c)
        else:                    # above the last measured point, to its left
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(-3, 4), textcoords="offset points", ha="right", va="bottom",
                        fontsize=6.5, color=c)
    ax.axhline(gpu, color=fs.GREY, ls=":", lw=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2 ** k for k in (10, 12, 14, 16)])
    ax.set_xticklabels([f"$2^{{{k}}}$" for k in (10, 12, 14, 16)])
    ax.set_ylim(0, gpu * 1.16)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak memory (GB)")
    ax.grid(True, axis="y", color=fs.GREY_LIGHT, lw=0.5)
    ax.set_axisbelow(True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
