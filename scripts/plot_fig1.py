#!/usr/bin/env python3
"""Figure 1 of the EpiKV paper, and the appendix trace figure.

  python scripts/plot_fig1.py --seg reports/retained/K1024_p3.json \
      --out ../EpiKV-overleaf/figures/fig1.pdf
  python scripts/plot_fig1.py --seg reports/retained/K1024_p3.json \
      --appendix --out ../EpiKV-overleaf/figures/appendix_trace.pdf

Panel (a): the whole trace as one strip. Top: the epiphany score s(t) over
every generated token. Below it: a barcode of the eviction decision (blue =
kept by score, amber = recency window, grey = evicted). A bracket picks out
one stretch and shows the words, kept ones highlighted, so the reader sees
both the pattern and what it selects, without reading 1,372 tokens.

Panel (b): peak memory of one forward pass, attention scoring vs. ours,
with the two out-of-memory points and the 16x gap between them.

Appendix mode: the full trace as a highlighted document in a proportional
font, no score strips, one page.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

fs.use()

# Memory panel data. Recovered from the vector paths of the figure shipped in
# the EMNLP version (A100 80 GB, one forward pass, Llama-8B); replaced by
# reports/prefill_memory_80gb.json when that file exists.
MEM_FALLBACK = {
    "gpu_gb": 79.3,
    "eager": {512: 15.6, 1024: 17.3, 2048: 24.1, 4096: 51.2},
    "eager_oom": 8192,
    "flash": {512: 15.2, 1024: 15.5, 2048: 16.0, 4096: 16.9, 8192: 19.0,
              16384: 23.0, 32768: 31.0, 65536: 47.2},
    "flash_oom": 131072,
}


def load_trace(path):
    d = json.load(open(path))
    toks = d["tokens"] if "tokens" in d else next(v for v in d.values() if isinstance(v, list))
    meta = {k: v for k, v in d.items() if k != "tokens" and not (isinstance(v, list) and v is toks)}
    return meta, toks


def load_mem(path):
    if path and Path(path).exists():
        data = json.load(open(path))
        res = data["results"]
        out = {"gpu_gb": data.get("gpu_gb", 80.0)}
        for mode, key in (("eager_attn", "eager"), ("flash", "flash")):
            rows = sorted(res[mode].values(), key=lambda r: r["seq_len"])
            out[key] = {r["seq_len"]: r["peak_gpu_mb"] / 1024 for r in rows if not r["oom"]}
            oom = [r["seq_len"] for r in rows if r["oom"]]
            out[key + "_oom"] = min(oom) if oom else None
        return out
    return MEM_FALLBACK


_MEASURE = {}


def text_width(s, size, fp):
    """Width in points of string s at font size `size`, measured with the
    figure's own renderer (TextPath misbehaves on this matplotlib/numpy)."""
    if not s:
        return 0.0
    key = (s, size)
    if key in _MEASURE:
        return _MEASURE[key]
    fig = _MEASURE.get("_fig")
    if fig is None:
        fig = plt.figure(figsize=(1, 1))
        _MEASURE["_fig"] = fig
        _MEASURE["_renderer"] = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=size, fontproperties=fp)
    w = t.get_window_extent(_MEASURE["_renderer"]).width * 72.0 / fig.dpi
    t.remove()
    w = max(w, 0.25 * size)
    _MEASURE[key] = w
    return w


def mark_after_last_eviction(toks, meta=None):
    """Tokens generated after the last eviction were never scored against a
    full cache: the cache did not exceed the budget again before the trace
    ended. Colouring them "kept by score" would overstate the score's role,
    so they get their own class. Dumps made with eviction_steps recorded use
    the step of the last eviction; older dumps fall back to the last evicted
    position, which bounds it from below (under tau=128 a late eviction can
    drop only old tokens, so the fallback can overstate this class)."""
    steps = (meta or {}).get("eviction_steps")
    if steps:
        # Tokens inside the recency window at the last eviction were protected
        # there, not scored, so the never-scored class starts keep_recent
        # tokens before the last eviction step.
        keep_recent = min(int(meta.get("keep_recent", 128)), int(meta.get("cache_size", 1024)) // 4)
        last = steps[-1] - keep_recent
    else:
        last = max((i for i, t in enumerate(toks) if not t["kept"]), default=-1)
    for i, t in enumerate(toks):
        t["unpressured"] = bool(t["kept"] and not t.get("recency") and i > last)
    return toks


def layout_words(toks, width_pt, size, fp):
    """Same wrapping as draw_words, but returns the lines as lists of
    (token, x, lead, width) so a long trace can be rendered page by page."""
    lines, cur, x = [], [], 0.0
    space = text_width(" ", size, fp)
    for t in toks:
        s = t["text"].replace("\n", " ")
        s_stripped = s.strip()
        if not s_stripped:
            x += space * 0.6
            continue
        w = text_width(s_stripped, size, fp)
        lead = space if s.startswith(" ") and x > 0 else 0.0
        if x + lead + w > width_pt and x > 0:
            lines.append(cur)
            cur, x, lead = [], 0.0, 0.0
        cur.append((t, s_stripped, x + lead, w))
        x += lead + w
    if cur:
        lines.append(cur)
    return lines


def draw_lines(ax, lines, x0, y0, size, fp, line_h, kept_face, rec_face, unp_face="#e4eefb"):
    for li, line in enumerate(lines):
        y = y0 - li * line_h
        for t, s, x, w in line:
            if t["kept"]:
                face = rec_face if t.get("recency") else (unp_face if t.get("unpressured") else kept_face)
                ax.add_patch(Rectangle((x0 + x - 0.6, y - 0.28 * size), w + 1.2, 1.15 * size,
                                       facecolor=face, edgecolor="none", zorder=1))
                col = fs.INK if not t.get("unpressured") else "#4a5a6a"
            else:
                col = "#9a9a9a"
            ax.text(x0 + x, y, s, fontsize=size, color=col, va="baseline", ha="left", zorder=2,
                    fontproperties=fp)


def draw_words(ax, toks, x0, y0, width_pt, size, fp, line_h, kept_face, rec_face,
               max_lines=None, unp_face="#e4eefb"):
    """Lay tokens out as highlighted words in axes points; returns lines used."""
    x, line = 0.0, 0
    space = text_width(" ", size, fp)
    for t in toks:
        s = t["text"].replace("\n", " ")
        s_stripped = s.strip()
        if not s_stripped:
            x += space * 0.6
            continue
        w = text_width(s_stripped, size, fp)
        lead = space if s.startswith(" ") and x > 0 else 0.0
        if x + lead + w > width_pt and x > 0:
            line += 1
            x = 0.0
            lead = 0.0
            if max_lines is not None and line >= max_lines:
                return line
        y = y0 - line * line_h
        if t["kept"]:
            face = rec_face if t.get("recency") else (unp_face if t.get("unpressured") else kept_face)
            ax.add_patch(Rectangle((x0 + x + lead - 0.6, y - 0.28 * size), w + 1.2, 1.15 * size,
                                   facecolor=face, edgecolor="none", zorder=1))
            col = fs.INK if not t.get("unpressured") else "#4a5a6a"
        else:
            col = "#9a9a9a"
        ax.text(x0 + x + lead, y, s_stripped, fontsize=size, color=col, va="baseline",
                ha="left", zorder=2, fontproperties=fp)
        x += lead + w
    return line + 1


def panel_a(fig, rect, meta, toks, excerpt):
    """rect = [left, bottom, width, height] in figure fraction."""
    L, B, W, H = rect
    n = len(toks)
    fp = FontProperties(family=["DejaVu Sans"])
    # --- score strip ------------------------------------------------------
    ax_s = fig.add_axes([L, B + 0.72 * H, W, 0.20 * H])
    sc = [t["score"] if t["score"] is not None and not math.isnan(t["score"]) else 0.0 for t in toks]
    k = 9
    sm = [sum(sc[max(0, i - k // 2):i + k // 2 + 1]) / len(sc[max(0, i - k // 2):i + k // 2 + 1])
          for i in range(n)]
    ax_s.plot(range(n), sc, color=fs.GREY_LIGHT, lw=0.35)
    ax_s.plot(range(n), sm, color=fs.BLUE, lw=0.8)
    ax_s.axhline(0, color=fs.GREY, lw=0.4)
    ax_s.set_xlim(0, n)
    ax_s.set_yticks([])
    ax_s.set_xticks([])
    for s in ("left", "bottom"):
        ax_s.spines[s].set_visible(False)
    ax_s.text(0, 1.02, "epiphany score $s(t)$ per token (grey) and its 9-token mean (blue)",
              transform=ax_s.transAxes, fontsize=6.5, va="bottom", color=fs.INK)
    # --- barcode ----------------------------------------------------------
    ax_b = fig.add_axes([L, B + 0.52 * H, W, 0.09 * H])
    ax_b.set_xlim(0, n)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    colors = [fs.AMBER if t.get("recency") else (fs.BLUE if t["kept"] else fs.GREY_LIGHT) for t in toks]
    ax_b.bar(range(n), [1] * n, width=1.0, color=colors, linewidth=0)
    # excerpt bracket
    e0, e1 = excerpt
    ax_b.add_patch(Rectangle((e0, -0.15), e1 - e0, 1.3, fill=False, edgecolor=fs.INK, lw=0.7,
                             zorder=5, clip_on=False))
    # legend as the barcode's title row, so nothing sits between barcode and excerpt
    ax_b.text(0, 1.35, "eviction decision per token:", color=fs.INK, fontsize=6.5, va="bottom",
              transform=ax_b.transAxes)
    ax_b.text(0.42, 1.35, "kept by score", color=fs.BLUE, fontsize=6.5, va="bottom", transform=ax_b.transAxes)
    ax_b.text(0.63, 1.35, "recency window", color=fs.AMBER, fontsize=6.5, va="bottom", transform=ax_b.transAxes)
    ax_b.text(0.87, 1.35, "evicted", color="#8a8a8a", fontsize=6.5, va="bottom", transform=ax_b.transAxes)
    # --- excerpt words ----------------------------------------------------
    ax_w = fig.add_axes([L, B, W, 0.40 * H])
    ax_w.axis("off")
    fig_w_in = fig.get_size_inches()[0]
    width_pt = W * fig_w_in * 72 - 6
    ax_w.set_xlim(0, width_pt)
    h_pt = 0.44 * H * fig.get_size_inches()[1] * 72
    ax_w.set_ylim(0, h_pt)
    size = 6.6
    line_h = 9.6
    used = draw_words(ax_w, toks[e0:e1], 5, h_pt - 11, width_pt - 4, size, fp, line_h,
                      fs.BLUE_LIGHT, "#f6e3b5", max_lines=int((h_pt - 6) // line_h))
    # frame around the excerpt and two zoom lines from the bracket to it
    ax_w.add_patch(Rectangle((0.5, 0.5), width_pt + 4, h_pt - 1, fill=False, edgecolor=fs.INK,
                             lw=0.6, zorder=0, clip_on=False))
    from matplotlib.patches import ConnectionPatch
    for xb, xw in ((e0, 0.5), (e1, width_pt + 4.5)):
        fig.add_artist(ConnectionPatch(xyA=(xb, -0.15), coordsA=ax_b.transData,
                                       xyB=(xw, h_pt - 0.5), coordsB=ax_w.transData,
                                       color=fs.INK, lw=0.5, ls=(0, (2, 2))))
    return used


def panel_b(fig, rect, mem):
    L, B, W, H = rect
    ax = fig.add_axes([L, B, W, H])
    g = mem["gpu_gb"]
    xe = sorted(mem["eager"])
    xf = sorted(mem["flash"])
    ax.plot(xe, [mem["eager"][x] for x in xe], marker="o", color=fs.RED, ms=3, lw=1.3)
    ax.plot(xf, [mem["flash"][x] for x in xf], marker="s", color=fs.BLUE, ms=3, lw=1.3)
    ax.axhline(g, color=fs.GREY, ls=(0, (3, 2)), lw=0.6)
    ax.text(600, g + 1.5, "80 GB GPU", fontsize=6.5, color=fs.GREY, va="bottom")
    for key, col in (("eager_oom", fs.RED), ("flash_oom", fs.BLUE)):
        if mem.get(key):
            ax.scatter([mem[key]], [g], marker="x", s=32, color=col, lw=1.2, zorder=5, clip_on=False)
    if mem.get("eager_oom") and mem.get("flash_oom"):
        ax.annotate("", xy=(mem["flash_oom"], g - 9), xytext=(mem["eager_oom"], g - 9),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=fs.INK, shrinkA=0, shrinkB=0))
        ratio = mem["flash_oom"] / mem["eager_oom"]
        ax.text(math.sqrt(mem["flash_oom"] * mem["eager_oom"]), g - 12,
                f"{ratio:.0f}$\\times$ longer", ha="center", va="top", fontsize=7,
                color=fs.INK, fontweight="bold")
    ax.text(xe[-1] * 0.85, mem["eager"][xe[-1]] + 1, "attention\nscoring", color=fs.RED,
            fontsize=6.5, ha="right", va="bottom")
    ax.text(3000, 6, "EpiKV (FlashAttention)", color=fs.BLUE, fontsize=6.5, ha="left", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384, 65536])
    ax.set_xticklabels(["1k", "4k", "16k", "64k"])
    ax.set_xlim(400, 2.4e5)
    ax.set_ylim(0, 92)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.text(0, 1.03, "one forward pass, Llama-8B, A100", transform=ax.transAxes, fontsize=6.5,
            color=fs.GREY, va="bottom")
    return ax


def panel_a_only(fig, rect, meta, toks, excerpt):
    """Panel (a) without the score strip: the eviction decision for every
    token as a barcode, with a token-index axis, and a zoom into one stretch
    showing the words. No titles; the caption carries them."""
    L, B, W, H = rect
    n = len(toks)
    fp = FontProperties(family=["DejaVu Sans"])
    ax_b = fig.add_axes([L, B + 0.72 * H, W, 0.13 * H])
    ax_b.set_xlim(0, n)
    ax_b.set_ylim(0, 1)
    mark_after_last_eviction(toks, meta)
    KEPT = "#0b3a8c"   # deep blue: scored and kept
    PALE = "#cfdff5"   # pale blue: never scored against a full cache
    colors = [fs.AMBER if t.get("recency") else (PALE if t.get("unpressured") else
              (KEPT if t["kept"] else fs.GREY_LIGHT)) for t in toks]
    ax_b.bar(range(n), [1] * n, width=1.0, color=colors, linewidth=0)
    ax_b.set_yticks([])
    for s in ("left", "top", "right"):
        ax_b.spines[s].set_visible(False)
    from matplotlib.ticker import MaxNLocator
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 2.5, 5, 10]))
    ax_b.tick_params(axis="x", labelsize=7, length=2, pad=1)
    e0, e1 = excerpt
    ax_b.add_patch(Rectangle((e0, -0.12), e1 - e0, 1.24, fill=False, edgecolor=fs.INK, lw=0.8,
                             zorder=5, clip_on=False))
    ax_b.text(0, 1.25, "kept by score", color=KEPT, fontsize=7.2, va="bottom",
              transform=ax_b.transAxes)
    ax_b.text(0.235, 1.25, "evicted", color="#8a8a8a", fontsize=7.2, va="bottom",
              transform=ax_b.transAxes)
    ax_b.text(0.385, 1.25, "after the last eviction", color="#5b7aa6", fontsize=7.2, va="bottom",
              transform=ax_b.transAxes)
    ax_b.text(0.80, 1.25, "recency", color=fs.AMBER, fontsize=7.2, va="bottom",
              transform=ax_b.transAxes)
    ax_w = fig.add_axes([L, B, W, 0.50 * H])
    ax_w.axis("off")
    fig_w_in, fig_h_in = fig.get_size_inches()
    width_pt = W * fig_w_in * 72 - 6
    h_pt = 0.50 * H * fig_h_in * 72
    ax_w.set_xlim(0, width_pt)
    ax_w.set_ylim(0, h_pt)
    size, line_h = 7.0, 10.2
    used = draw_words(ax_w, toks[e0:e1], 5, h_pt - 11, width_pt - 4, size, fp, line_h,
                      "#9dbdf0", "#f6e3b5", max_lines=int((h_pt - 6) // line_h))
    # Frame hugs the text: its bottom sits one half-line under the last line.
    box_bottom = h_pt - 11 - (used - 1) * line_h - 0.6 * line_h
    ax_w.add_patch(Rectangle((0.5, box_bottom), width_pt + 4, h_pt - 0.5 - box_bottom,
                             fill=False, edgecolor=fs.INK, lw=0.6, zorder=0, clip_on=False))
    from matplotlib.patches import ConnectionPatch
    for xb, xw in ((e0, 0.5), (e1, width_pt + 4.5)):
        fig.add_artist(ConnectionPatch(xyA=(xb, -0.12), coordsA=ax_b.transData,
                                       xyB=(xw, h_pt - 0.5), coordsB=ax_w.transData,
                                       color=fs.INK, lw=0.5, ls=(0, (2, 2))))
    return ax_w, box_bottom


def make_fig1a(args):
    """Panel (a) alone, at 0.60 of the text width; panel (b) is the paper's
    existing prefill_memory.pdf placed beside it in LaTeX."""
    from matplotlib.transforms import Bbox
    meta, toks = load_trace(args.seg)
    fig = plt.figure(figsize=(0.60 * fs.TEXT_W, 1.85))
    ax_w, box_bottom = panel_a_only(fig, [0.02, 0.04, 0.96, 0.90], meta, toks,
                                    (args.excerpt_start, args.excerpt_end))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Crop at the excerpt frame so no empty band is left under it.
    fig.canvas.draw()
    y0_in = ax_w.transData.transform((0, box_bottom))[1] / fig.dpi
    w_in, h_in = fig.get_size_inches()
    fig.savefig(out, bbox_inches=Bbox([[0, max(y0_in - 0.03, 0)], [w_in, h_in]]))
    print("wrote", out)


def make_fig1(args):
    meta, toks = load_trace(args.seg)
    mem = load_mem(args.mem)
    fig = plt.figure(figsize=(fs.TEXT_W, 2.35))
    fig.text(0.005, 0.965, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.665, 0.965, "b", fontsize=9, fontweight="bold", va="top")
    panel_a(fig, [0.03, 0.08, 0.60, 0.80], meta, toks, (args.excerpt_start, args.excerpt_end))
    panel_b(fig, [0.745, 0.17, 0.245, 0.68], mem)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


def make_appendix(args):
    """The whole trace as highlighted text. Traces longer than one page are
    split over several PDFs: <out>, <out stem>_2.pdf, ... so LaTeX can place
    each on its own page."""
    meta, toks = load_trace(args.seg)
    fp = FontProperties(family=["DejaVu Sans"])
    W_in, H_in = fs.TEXT_W, args.page_h
    width_pt = W_in * 72 - 8
    h_pt = H_in * 72
    size, line_h = args.font, args.font * 1.45
    for t in toks:
        if "end" in t["text"] and "sentence" in t["text"]:
            t["text"] = " <EOS>"
    mark_after_last_eviction(toks, meta)
    lines = layout_words(toks, width_pt, size, fp)
    per_page = int((h_pt - 34) // line_h)
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)]
    out = Path(args.out)
    for pi, page in enumerate(pages):
        fig = plt.figure(figsize=(W_in, H_in))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_xlim(0, W_in * 72)
        ax.set_ylim(0, h_pt)
        # One short header line: a long unwrapped line here widened the saved
        # figure to twice the text width and the trace printed at half size.
        head = (f"EpiKV-Seg, $K$=1024, MATH-500 problem {meta['problem_idx']}: "
                f"{meta['generated']:,} tokens generated, {meta['retained_generated']:,} kept.")
        if len(pages) > 1:
            head += f"  (page {pi + 1} of {len(pages)})"
        ax.text(4, h_pt - 10, head, fontsize=7.5, va="top", color=fs.INK)
        draw_lines(ax, page, 4, h_pt - 30, size, fp, line_h, "#9dbdf0", "#f6e3b5")
        p = out if pi == 0 else out.with_name(f"{out.stem}_{pi + 1}{out.suffix}")
        fig.savefig(p)
        plt.close(fig)
        print("wrote", p, f"({len(page)} lines)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--mem", default=str(Path(__file__).resolve().parent.parent / "reports" / "prefill_memory_80gb.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--excerpt_start", type=int, default=243)
    ap.add_argument("--excerpt_end", type=int, default=330)
    ap.add_argument("--appendix", action="store_true")
    ap.add_argument("--panel_a", action="store_true", help="panel (a) alone, 0.6 text width")
    ap.add_argument("--page_h", type=float, default=8.6, help="appendix page height (in)")
    ap.add_argument("--font", type=float, default=7.4, help="appendix font size (pt)")
    a = ap.parse_args()
    (make_appendix if a.appendix else make_fig1a if a.panel_a else make_fig1)(a)
