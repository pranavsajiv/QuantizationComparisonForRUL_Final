"""
Renders Fig. 3.1 -- the iconic system architecture for the enhanced AeroGuard
study, laid out so that each horizontal band is exactly one of Modules 1-5.

The diagram differs from the original single-model architecture in one
structural way: Module 4 forks into two numeric regimes that are BOTH carried
through to Module 5, because the comparison between those two branches is the
study's actual measurement rather than a deployment afterthought.

Layout rules that keep it readable:
  * every band reserves a fixed title zone at its top; no box or connector
    is ever placed inside that zone
  * inter-module connectors live entirely in the gap between two bands
  * the final band is taller, to hold the paired-delta node
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "figures")

NAVY, BLUE, TEAL, ORANGE, RED, GOLD, GREY = (
    "#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#c1121f", "#e9c46a", "#6c757d")
BAND_FC, BAND_EC, EDGE = "#eef2f7", "#c9d3e0", "#33415c"

plt.rcParams.update({"font.family": "DejaVu Sans"})

TOP = 0.988
BAND_H = 0.165          # standard band height
LAST_EXTRA = 0.062      # extra height for Module 5 (holds the delta node)
GAP = 0.018
BOX_H = 0.068
TITLE_ZONE = 0.078      # reserved at the top of every band


def band_h(k):
    return BAND_H + (LAST_EXTRA if k == 4 else 0.0)


def band_top(k):
    return TOP - k * (BAND_H + GAP)


def band_bot(k):
    return band_top(k) - band_h(k)


def box_y(k):
    return band_top(k) - TITLE_ZONE - BOX_H


def box_mid(k):
    return box_y(k) + BOX_H / 2


def band(ax, k, title, sub):
    ax.add_patch(FancyBboxPatch(
        (0.012, band_bot(k)), 0.976, band_h(k),
        boxstyle="round,pad=0.003,rounding_size=0.010",
        facecolor=BAND_FC, edgecolor=BAND_EC, linewidth=1.0, zorder=1))
    ax.text(0.030, band_top(k) - 0.026, title, fontsize=10.0, weight="bold",
            color=NAVY, va="center", zorder=2)
    ax.text(0.030, band_top(k) - 0.054, sub, fontsize=7.6, color=GREY,
            va="center", style="italic", zorder=2)


def box(ax, x, y, w, h, text, fc, tc="white", fs=8.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=fc, edgecolor=EDGE, linewidth=1.1, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, weight="bold", linespacing=1.45)


def arrow(ax, p1, p2, color=EDGE, lw=1.7):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=13, color=color,
        linewidth=lw, zorder=6, shrinkA=1, shrinkB=1,
        connectionstyle="arc3,rad=0"))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.6, 13.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def flow(k, x=0.5):
        arrow(ax, (x, band_bot(k)), (x, band_top(k + 1)), color=NAVY)

    # ---------------- MODULE 1 ----------------------------------------
    band(ax, 0, "MODULE 1  ·  Data Acquisition & Piecewise RUL Labelling",
         "NASA C-MAPSS FD001 · 100 train / 100 test engines · "
         "21 sensors + 3 operating settings")
    y = box_y(0)
    box(ax, 0.055, y, 0.25, BOX_H, "C-MAPSS FD001\ntrain split", BLUE)
    box(ax, 0.375, y, 0.25, BOX_H,
        "Raw target\nRUL = C$_{max}$ − t\nEq. (1)", NAVY, fs=8.2)
    box(ax, 0.695, y, 0.25, BOX_H,
        "Piecewise cap\nR$_{early}$ = 130\nEq. (2)", ORANGE, fs=8.2)
    arrow(ax, (0.305, box_mid(0)), (0.375, box_mid(0)))
    arrow(ax, (0.625, box_mid(0)), (0.695, box_mid(0)))
    flow(0)

    # ---------------- MODULE 2 ----------------------------------------
    band(ax, 1, "MODULE 2  ·  Feature Masking, Normalisation & "
                "Temporal Windowing",
         "Scaler bounds fit on the training split only, then frozen for "
         "every downstream stage")
    y = box_y(1)
    box(ax, 0.055, y, 0.25, BOX_H,
        "Variance mask\n21 → 14 sensors\nEq. (3)", TEAL, fs=8.2)
    box(ax, 0.375, y, 0.25, BOX_H, "Min–Max scaling\nEq. (4)", TEAL)
    box(ax, 0.695, y, 0.25, BOX_H,
        "Sliding window\nL = 30 cycles\nEq. (5)", TEAL, fs=8.2)
    arrow(ax, (0.305, box_mid(1)), (0.375, box_mid(1)))
    arrow(ax, (0.625, box_mid(1)), (0.695, box_mid(1)))
    flow(1)

    # ---------------- MODULE 3 ----------------------------------------
    band(ax, 2, "MODULE 3  ·  Architecture Bank — the study's "
                "independent variable",
         "Only the network differs — identical (30 × 14) input; Adam(1e-3), "
         "MSE, batch 64, early stopping (patience 5) · Eq. (16)–(17)")
    y = box_y(2)
    for i, (name, col) in enumerate([
            ("1D-CNN\nincumbent\nEq. (6)–(8)", BLUE),
            ("LSTM\nrecurrent\nEq. (9)–(11)", RED),
            ("Transformer\nattention\nEq. (12)–(14)", TEAL),
            ("MLP\ncontrol\nEq. (15)", GOLD)]):
        x = 0.048 + i * 0.234
        box(ax, x, y, 0.202, BOX_H, name, col,
            tc=NAVY if col == GOLD else "white", fs=8.3)
    flow(2)

    # ---------------- MODULE 4 ----------------------------------------
    band(ax, 3, "MODULE 4  ·  Dual-Regime Compression — the "
                "comparison under test",
         "Both branches carry the SAME trained weights; the difference "
         "between them is the measurement")
    y = box_y(3)
    fork = band_top(3) - TITLE_ZONE + 0.008
    box(ax, 0.085, y, 0.31, BOX_H,
        "FP32 TFLite\nreference branch\nno quantisation", BLUE, fs=8.4)
    box(ax, 0.605, y, 0.31, BOX_H,
        "Full static INT8 TFLite\nweights + activations + I/O\n"
        "100-window calibration · Eq. (18)–(20)", ORANGE, fs=7.7)
    arrow(ax, (0.5, fork), (0.240, y + BOX_H), color=BLUE, lw=1.8)
    arrow(ax, (0.5, fork), (0.760, y + BOX_H), color=ORANGE, lw=1.8)
    arrow(ax, (0.240, y), (0.492, band_bot(3) + 0.003), color=BLUE, lw=1.5)
    arrow(ax, (0.760, y), (0.508, band_bot(3) + 0.003), color=ORANGE, lw=1.5)
    flow(3)

    # ---------------- MODULE 5 ----------------------------------------
    band(ax, 4, "MODULE 5  ·  Safety-Asymmetric Evaluation & "
                "Alert Decision",
         "Scored on the DIRECTION of the error, not only its magnitude")
    y = box_y(4)
    metrics = [
        (0.040, "Conventional\nRMSE · MAE\nEq. (21)–(22)", GREY, 8.0),
        (0.360, "Safety-asymmetric\nsigned bias · late fraction\n"
                "asym. score · Eq. (23)–(25)", RED, 7.6),
        (0.680, "Operational\nHealthy / Warning / Critical\nEq. (26)",
         NAVY, 8.0)]
    for x, txt, col, fs in metrics:
        box(ax, x, y, 0.28, BOX_H, txt, col, fs=fs)

    yb = band_bot(4) + 0.016
    box(ax, 0.215, yb, 0.57, 0.042,
        "Δ(INT8 − FP32) per metric, paired within seed  ·  6 seeds",
        ORANGE, fs=9.0)
    for x, _, _, _ in metrics:
        arrow(ax, (x + 0.14, y), (0.5, yb + 0.042), color=GREY, lw=1.2)

    ax.text(0.5, 0.010,
            "Five-module architecture "
            "for architecture × quantisation evaluation",
            ha="center", fontsize=10.4, weight="bold", color=NAVY)

    fig.savefig(os.path.join(OUT, "fig31_architecture.png"),
                facecolor="white", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/fig31_architecture.png")


if __name__ == "__main__":
    main()
