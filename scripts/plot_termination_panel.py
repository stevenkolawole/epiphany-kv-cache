#!/usr/bin/env python3
"""Figure 1 (right), alternative version: where the errors go under eviction. For each policy,
terminated wrong answers extend left of a common zero and cap hits (loops) extend right, at
K=1024 on MATH-500 (n=495), both models stacked; the dashed line is the loop count without
eviction. Numbers are parsed from the generated Table 3 rows (tables/tab_termination_rows.tex),
so the figure and the table cannot disagree.

  python3 scripts/plot_termination_panel.py --out ../EpiKV-overleaf/figures/termination_panel.pdf
"""
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROWS = Path(__file__).resolve().parent.parent.parent / "EpiKV-overleaf" / "tables" / "tab_termination_rows.tex"
SHOW = [("none", "no eviction"), ("RaaS", "RaaS"), ("H2O", "H2O"),
        ("\\methodseg", r"EpiKV$^{-\mathrm{Q}}$"), ("\\methodq", "EpiKV")]


def parse():
    out = {}
    for line in ROWS.read_text().splitlines():
        line = line.replace("\\rowcolor{epirow}", "").strip()
        if not line or line.startswith("%") or "&" not in line:
            continue
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        out[cells[0]] = [int(x) for x in cells[1:]]   # errors, cap, wrong (Llama) | errors, cap, wrong (Qwen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = parse()
    fs.use()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(fs.TEXT_W * 0.37, 1.95), sharex=True)
    for r, (ax, model) in enumerate(zip(axs, ("Llama-8B", "Qwen-7B"))):
        off = 0 if r == 0 else 3
        for i, (key, lab) in enumerate(SHOW[::-1]):
            _, cap, wrong = d[key][off:off + 3]
            ax.barh(i, -wrong, color=fs.GREY_LIGHT, height=0.66, lw=0)
            ax.barh(i, cap, color=(fs.GREY_MID if key == "none" else fs.RED), height=0.66, lw=0)
            ax.text(-wrong - 4, i, str(wrong), ha="right", va="center", fontsize=6, color=fs.INK)
            ax.text(cap + 7, i, str(cap), ha="left", va="center", fontsize=6, color=fs.INK)
        ax.axvline(0, color=fs.INK, lw=0.6)
        ax.axvline(d["none"][off + 1], color=fs.INK, lw=0.6, ls=(0, (2, 2)))
        ax.set_yticks(range(len(SHOW)))
        ax.set_yticklabels([lab for _, lab in SHOW[::-1]], fontsize=6.5)
        for t in ax.get_yticklabels():
            if t.get_text() == "EpiKV":
                t.set_color(fs.BLUE_DARK); t.set_fontweight("bold")
        ax.tick_params(axis="y", length=0, pad=1)
        ax.set_xticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.set_xlim(-100, 150); ax.set_ylim(-0.55, len(SHOW) - 0.45)
        ax.text(-0.02, 0.985, model, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=fs.INK, fontweight="bold")
    axs[0].text(-6, len(SHOW) - 0.35, "wrong answers", ha="right", va="bottom", fontsize=6.5, color=fs.GREY)
    axs[0].text(6, len(SHOW) - 0.35, "loops (cap hits)", ha="left", va="bottom", fontsize=6.5, color=fs.RED)
    fig.subplots_adjust(hspace=0.22)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
