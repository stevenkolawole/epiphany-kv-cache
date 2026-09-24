"""One figure style for both papers.

Rules, in order of how often they were broken before this file existed:

1. Size the figure at the width it will be printed at, and never scale it in
   LaTeX. ICLR text width is 5.5 in; a half-width panel is 2.65 in. Fonts are
   then real point sizes: 8 pt body, 7 pt ticks, 9 pt panel titles, in the
   paper's own Times (Nimbus Roman), never a sans default.
2. One accent per concept, greys for everything else. Ours is blue; the
   attention-based comparison is red; recency/secondary is amber.
3. Direct labels on the data instead of a legend wherever there is room.
4. The single number the reader must take away is annotated once, in the
   accent colour. Nothing else is annotated.
5. No top/right spines, no gridlines unless the reader needs to read values
   off the plot, and then only light horizontal ones.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_W = 5.5          # ICLR text width, inches
HALF_W = 2.65

# Palette "Oxford" (reports/viz_study/notes_color.md; bible D24-D27). One hue per meaning:
# EpiKV = ink blue (dark for text, mid for lines, light/pale for fills); failure (loops, OOM) =
# vermilion; model internals and baselines = cool greys; reference (no eviction) = near-black dashed.
BLUE = "#095792"      # EpiKV line
BLUE_DARK = "#0A3F6E" # EpiKV text
BLUE_MID = "#81A3CF"  # EpiKV secondary (recency window, variants)
BLUE_LIGHT = "#CCDFF7"  # EpiKV fill
RED = "#C85138"       # failure only: loops, OOM (never a method colour)
RED_LIGHT = "#FDDFD7"
AMBER = BLUE_MID      # retired as a hue; kept as an alias so old scripts stay consistent
GREY_DARK = "#43474B"
GREY = "#6B7076"
GREY_MID = "#92969B"
GREY_LIGHT = "#D5D7DB"
BAND_A = "#F2EFCC"
BAND_B = "#D3C7C0"
INK = "#222222"


def use():
    plt.rcParams.update({
        # Body font of the paper (LaTeX `times` = URW Nimbus Roman), maths in STIX (bible D0c, X21).
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")
