#!/usr/bin/env python3
"""
Figure 1a and the appendix exhibit: what the score keeps, beside what H2O keeps.

Input: JSON sidecars written by scripts/dump_retained.py --json_out. Each has
per-generated-token text, the epiphany score at scoring time (NaN for H2O,
which has no such score), and the final kept / recency flags.

Design, agreed 2026-09-10:
  * three categories only: retained by score, retained by the recency window,
    evicted. The recency window must be visibly distinct or a reader assumes
    the answer block survived by position.
  * a score strip s(t) aligned over the tokens, so the reader sees that the
    highlighted tokens are the high-scoring ones and retention is not recency.
  * EpiKV-Seg on the left, H2O on the right, same problem, same budget. The
    two traces are identical up to the first eviction and diverge after; the
    caption says so. Problem 3 was chosen as the first problem in MATH-500
    whose trace exceeds the budget and still terminates; no shopping.
  * a single generator for both crops: --lines N gives the Figure 1a excerpt,
    --lines 0 gives the full-trace appendix figure.

Tokens are laid out in rows of fixed width; each row carries its own score
strip directly above it, so alignment is exact without pixel arithmetic.
Colours are a categorical palette that separates in both light and dark
rendering and in greyscale print (fill + hatch/edge differ, not just hue).

    python3 scripts/plot_retained_figure.py \
        --seg reports/retained/K512_p3.json --h2o reports/retained/K512_p3_h2o.json \
        --lines 10 --out EpiKV-overleaf/figures/fig1a_retained.pdf
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Categorical: score-retained (deep blue fill), recency-retained (amber, hatched),
# evicted (no fill, grey text). Chosen so the three separate in greyscale too.
C_SCORE = "#2b5fd9"
C_RECENCY = "#e0a100"
C_EVICT_TXT = "#8a8a8a"
C_KEPT_TXT = "#111111"
C_STRIP = "#2b5fd9"


def load(path):
    d = json.load(open(path))
    toks = d["tokens"]
    for t in toks:
        s = t["text"].replace("\n", "⏎").replace("\t", " ")
        # The model's EOS renders with fullwidth bars that DejaVu lacks.
        if "end▁of▁sentence" in s or "｜" in s:
            s = "<EOS>"
        t["text"] = s
    return d, toks


def layout_rows(toks, width_chars):
    """Pack tokens into rows by character count; returns list of (start, end)."""
    rows, start, used = [], 0, 0
    for i, t in enumerate(toks):
        w = max(1, len(t["text"]))
        if used + w > width_chars and i > start:
            rows.append((start, i))
            start, used = i, 0
        used += w
    rows.append((start, len(toks)))
    return rows


def draw_panel(ax_strip, ax_text, toks, rows, title, show_scores, char_w=1.0):
    ax_text.set_axis_off()
    ax_strip.set_axis_off()
    n_rows = len(rows)
    row_h = 1.0
    strip_h = 0.55
    y_top = n_rows * (row_h + strip_h)
    ax_text.set_xlim(0, max(1, max(sum(max(1, len(t["text"])) for t in toks[s:e]) for s, e in rows)) * char_w)
    ax_text.set_ylim(0, y_top)
    ax_strip.set_xlim(ax_text.get_xlim())
    ax_strip.set_ylim(0, y_top)

    # Score range for the strip (ignore NaN)
    vals = [t["score"] for t in toks if t["score"] is not None and not math.isnan(t["score"])]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    rng = (hi - lo) or 1.0

    for r, (s, e) in enumerate(rows):
        y_row = y_top - (r + 1) * (row_h + strip_h)
        y_strip = y_row + row_h + 0.05
        x = 0.0
        xs, ys = [], []
        for t in toks[s:e]:
            w = max(1, len(t["text"])) * char_w
            if t["kept"] and not t["recency"]:
                ax_text.add_patch(Rectangle((x, y_row + 0.08), w, row_h - 0.16,
                                            facecolor=C_SCORE, alpha=0.22, edgecolor=C_SCORE,
                                            linewidth=0.6))
                col = C_KEPT_TXT
            elif t["recency"]:
                ax_text.add_patch(Rectangle((x, y_row + 0.08), w, row_h - 0.16,
                                            facecolor=C_RECENCY, alpha=0.18, edgecolor=C_RECENCY,
                                            linewidth=0.6, hatch="////"))
                col = C_KEPT_TXT
            else:
                col = C_EVICT_TXT
            ax_text.text(x + w / 2, y_row + row_h / 2, t["text"], ha="center", va="center",
                         fontsize=7.2, family="DejaVu Sans Mono", color=col)
            if show_scores and t["score"] is not None and not math.isnan(t["score"]):
                xs.append(x + w / 2)
                ys.append(y_strip + strip_h * 0.9 * (t["score"] - lo) / rng)
            x += w
        if show_scores and xs:
            ax_strip.plot(xs, ys, color=C_STRIP, linewidth=0.9)
            ax_strip.fill_between(xs, y_strip, ys, color=C_STRIP, alpha=0.12)
    ax_text.set_title(title, fontsize=9, loc="left", pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--h2o", required=False)
    ap.add_argument("--lines", type=int, default=10, help="rows per panel; 0 = full trace")
    ap.add_argument("--width", type=int, default=64, help="characters per row")
    ap.add_argument("--skip_prefix", type=int, default=0,
                    help="skip this many generated tokens before the excerpt")
    ap.add_argument("--skip_rows", type=int, default=0,
                    help="after layout, drop this many leading rows (paginate a full trace)")
    ap.add_argument("--row_in", type=float, default=0.42,
                    help="inches per row (text + score strip)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dseg, tseg = load(args.seg)
    panels = [("EpiKV-Seg (ours)", dseg, tseg, True)]
    if args.h2o:
        dh2o, th2o = load(args.h2o)
        panels.append(("H2O (attention-based)", dh2o, th2o, False))

    fig_rows = []
    for title, d, toks, show in panels:
        toks = toks[args.skip_prefix:]
        rows = layout_rows(toks, args.width)
        rows = rows[args.skip_rows:]
        if args.lines > 0:
            rows = rows[:args.lines]
        # Re-base so the panel starts at the first kept row.
        base = rows[0][0]
        toks = toks[base:]
        rows = [(s - base, e - base) for s, e in rows]
        fig_rows.append((title, d, toks, rows, show))

    n_rows_max = max(len(r[3]) for r in fig_rows)
    fig_h = args.row_in * n_rows_max + 0.9
    # Cell width must equal the glyph advance or tokens run together: at
    # 7.2 pt DejaVu Sans Mono one character is ~0.060 in, so size the panel
    # from the row width in characters rather than from a fixed inch count.
    panel_w = 0.060 * args.width + 0.3
    fig_w = panel_w * len(fig_rows) + 0.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, len(fig_rows), wspace=0.06)
    for k, (title, d, toks, rows, show) in enumerate(fig_rows):
        ax = fig.add_subplot(gs[0, k])
        ax_strip = ax.twinx()  # same box; drawn beneath the text layer
        ax_strip.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        sub = (f"K={d['cache_size']}, {d['generated']} tokens generated, "
               f"{d['retained_generated']} retained")
        # Two-line title: the one-line form ran into the neighbouring panel.
        draw_panel(ax_strip, ax, toks, rows, f"{title}\n{sub}", show)

    # Legend as three labelled swatches, not a matplotlib legend, so the
    # hatch shows in vector output.
    # Legend: swatches spaced in inches converted to figure fraction, so they
    # do not collide whatever the panel count. Strip caption on its own line.
    items = (("retained by score", C_SCORE, None),
             ("retained by recency window", C_RECENCY, "////"),
             ("evicted", "none", None))
    lx_in = 0.15
    for lab, fc, hatch in items:
        lx = lx_in / fig_w
        fig.patches.append(Rectangle((lx, 0.012), 0.14 / fig_w, 0.028, transform=fig.transFigure,
                                     facecolor=fc if fc != "none" else "white",
                                     alpha=0.22 if fc != "none" else 1.0,
                                     edgecolor=fc if fc != "none" else C_EVICT_TXT,
                                     hatch=hatch, linewidth=0.8))
        fig.text(lx + 0.18 / fig_w, 0.02, lab, fontsize=7.5, va="bottom",
                 color=C_KEPT_TXT if fc != "none" else C_EVICT_TXT)
        lx_in += 0.22 + 0.052 * len(lab)
    fig.text(0.99, 0.055, "line above each row: epiphany score s(t), higher = kept",
             fontsize=7, ha="right", va="bottom", color=C_STRIP)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
