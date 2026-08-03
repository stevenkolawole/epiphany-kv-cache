#!/usr/bin/env python3
"""
plot_prefill_memory.py

Figure: peak GPU memory of a single forward pass vs. context length, eager
(output_attentions=True) vs. FlashAttention. Reads the JSON from
bench_prefill_memory.py. OOM points are drawn as an X at the GPU ceiling.

    python scripts/plot_prefill_memory.py \
        --input reports/prefill_memory_80gb.json --output reports/prefill_memory.pdf
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("reports/prefill_memory_80gb.json"))
    p.add_argument("--output", type=Path, default=Path("reports/prefill_memory.pdf"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.input.read_text())
    gpu_gb = data.get("gpu_gb", 80.0)
    res = data["results"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    style = {"eager_attn": ("tab:red", "o", "attention scoring (eager)"),
             "flash": ("tab:green", "s", "ours (FlashAttention)")}

    for mode, (colour, marker, label) in style.items():
        if mode not in res:
            continue
        rows = sorted(res[mode].values(), key=lambda r: r["seq_len"])
        xs_ok = [r["seq_len"] for r in rows if not r["oom"]]
        ys_ok = [r["peak_gpu_mb"] / 1024 for r in rows if not r["oom"]]
        ax.plot(xs_ok, ys_ok, marker=marker, color=colour, lw=1.4, label=label)
        # First OOM point: mark at the GPU ceiling.
        oom = [r["seq_len"] for r in rows if r["oom"]]
        if oom:
            ax.scatter([min(oom)], [gpu_gb], marker="x", s=70, color=colour, zorder=5)
            ax.annotate("OOM", (min(oom), gpu_gb),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color=colour)

    ax.axhline(gpu_gb, color="0.4", ls=":", lw=1)
    ax.text(ax.get_xlim()[0], gpu_gb, f" {gpu_gb:.0f} GB GPU", va="bottom",
            fontsize=7, color="0.3")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
