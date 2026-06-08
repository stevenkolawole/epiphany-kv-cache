#!/usr/bin/env python3
"""
plot_layer_rho.py

Figure: per-layer Spearman rho between rolling-64 hidden-state L2 diff and
counterfactual importance labels, across the two competition-math datasets in both
attention back-ends. Band A (7-13, positive) and Band B (18-25, negative) shaded.

Self-contained — values are the verified Phase 0B results (Appendix A of the
paper); no GPU, no data files needed. Runs anywhere with matplotlib.

    python scripts/plot_layer_rho.py --output reports/layer_rho.pdf
"""

import argparse
from pathlib import Path

# Verified rho values (rolling-64 hs_l2_diff), layers 0..31.
LAYERS = list(range(32))
RHO = {
    "MATH-500":        [-.173,-.172,-.121,-.121,-.153,-.079,.011,.079,.065,.082,.112,.093,.038,.071,-.016,.017,-.002,-.003,-.081,-.147,-.191,-.209,-.223,-.254,-.250,-.243,-.211,-.066,-.053,.082,.135,.220],
    "MATH-500 (eager)":[-.199,-.202,-.151,-.124,-.139,-.058,-.007,.047,.078,.107,.141,.144,.077,.089,.006,.016,-.016,-.005,-.045,-.074,-.097,-.109,-.121,-.151,-.146,-.142,-.107,-.003,-.018,.049,.065,.093],
    "AIME-2024":       [-.068,-.089,-.101,-.014,.011,.083,.120,.118,.146,.136,.097,.083,.119,.058,-.020,-.124,-.140,-.147,-.118,-.113,-.105,-.085,-.059,-.021,-.005,-.006,.007,-.041,-.041,-.099,-.121,-.178],
    "AIME-2024 (eager)":[.037,-.052,-.125,-.072,-.052,.093,.150,.156,.155,.130,.120,.120,.114,.065,-.032,-.147,-.165,-.191,-.184,-.208,-.224,-.227,-.217,-.200,-.188,-.187,-.157,-.086,-.059,.030,.062,-.022],
}
BAND_A = (7, 13)
BAND_B = (18, 25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("reports/layer_rho.pdf"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhspan(0, 0, color="black")  # placeholder so legend ordering is stable
    ax.axvspan(BAND_A[0] - 0.5, BAND_A[1] + 0.5, color="tab:green", alpha=0.10)
    ax.axvspan(BAND_B[0] - 0.5, BAND_B[1] + 0.5, color="tab:red", alpha=0.10)
    ax.axhline(0, color="0.5", lw=0.8)

    markers = {"MATH-500": "o", "MATH-500 (eager)": "s",
               "AIME-2024": "^", "AIME-2024 (eager)": "D"}
    for name, ys in RHO.items():
        ax.plot(LAYERS, ys, marker=markers[name], ms=3, lw=1.2, label=name)

    ax.text((BAND_A[0]+BAND_A[1])/2, 0.30, "Band A", color="tab:green",
            ha="center", fontsize=8)
    ax.text((BAND_B[0]+BAND_B[1])/2, 0.30, "Band B", color="tab:red",
            ha="center", fontsize=8)
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"Spearman $\rho$ with importance")
    ax.set_xlim(-0.5, 31.5)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
