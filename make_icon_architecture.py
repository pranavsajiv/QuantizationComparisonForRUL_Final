"""
Renders the ICONIC system architecture: a left-to-right data-flow picture in
which the artefacts that actually move through the pipeline are drawn from the
real dataset rather than represented by empty boxes.

Every inset panel is plotted from measured data:
  * the raw sensor trace and the masked/normalised trace come from a real
    FD001 engine
  * the window tile is a real 30 x 14 normalised input matrix
  * the residual inset is the real pooled residual distribution of the
    deployed INT8 1D-CNN

This complements the modular architecture figure, which shows the same system
as five stacked functional blocks. This one shows what the data looks like at
each hand-off.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "figures")

NAVY, BLUE, TEAL, ORANGE, RED, GOLD, GREY = (
    "#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#c1121f", "#e9c46a", "#6c757d")
PANEL, EDGE = "#f4f7fa", "#33415c"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def group(ax, x, y, w, h, title, color=GREY):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor="none", edgecolor=color, linewidth=1.2,
        linestyle=(0, (4, 3)), zorder=1))
    ax.text(x + w / 2, y + h + 0.012, title, ha="center", va="bottom",
            fontsize=9.6, weight="bold", color=NAVY, zorder=2)


def chip(ax, x, y, w, h, text, fc, tc="white", fs=7.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.010",
        facecolor=fc, edgecolor=EDGE, linewidth=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", zorder=4, linespacing=1.35)


def arrow(ax, p1, p2, color=EDGE, lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=15, color=color,
        linewidth=lw, zorder=6, shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad=0"))


def caption(ax, x, y, text, fs=7.6, color=NAVY):
    ax.text(x, y, text, ha="center", va="top", fontsize=fs,
            color=color, zorder=5, linespacing=1.3)


def inset(fig, rect, frame_color=EDGE):
    a = fig.add_axes(rect)
    for s in a.spines.values():
        s.set_edgecolor(frame_color)
        s.set_linewidth(1.2)
    a.set_xticks([]); a.set_yticks([])
    a.set_facecolor("white")
    return a


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------- real data for the insets ---------------------------
    df_train, df_test, _ = C.load_raw()
    unit = df_train[df_train["unit"] == 1]
    raw_trace = unit[["s2", "s3", "s4", "s11"]].values
    dead_trace = unit[["s1", "s5", "s10", "s16"]].values

    d = C.add_rul(df_train)
    mn, mx = C.fit_scaler(d)
    ds = C.apply_scale(d, mn, mx)
    win = ds[ds["unit"] == 1][C.FEATURES].values[-30:]
    rul_raw = (unit["cycle"].max() - unit["cycle"]).values
    rul_cap = np.clip(rul_raw, 0, C.RUL_CAP)

    resid = None
    rp = os.path.join(RES, "results.json")
    if os.path.exists(rp):
        with open(rp) as fh:
            R = json.load(fh)
        yt = np.array(R["meta"]["y_true"], float)
        acc = [np.array(r["int8"]["predictions"], float) - yt
               for r in R["runs"]
               if r["arch"] == "cnn" and r.get("int8", {}).get("converted")]
        if acc:
            resid = np.concatenate(acc)

    # ---------------- canvas ---------------------------------------------
    fig = plt.figure(figsize=(15.4, 9.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ============ GROUP 1 : field / data source =========================
    group(ax, 0.018, 0.545, 0.225, 0.375, "Field Layer  ·  Module 1")
    a = inset(fig, [0.042, 0.735, 0.175, 0.135])
    for j in range(raw_trace.shape[1]):
        v = raw_trace[:, j]
        a.plot((v - v.min()) / (np.ptp(v) + 1e-9) + j * 1.15, lw=0.9)
    a.set_title("21 raw sensor channels", fontsize=7.4, color=NAVY, pad=3)
    caption(ax, 0.130, 0.727, "C-MAPSS FD001 run-to-failure trajectories\n"
                              "100 train / 100 test engines")

    a = inset(fig, [0.042, 0.588, 0.175, 0.088])
    a.plot(rul_raw, color=GREY, lw=1.4, ls="--", label="raw")
    a.plot(rul_cap, color=ORANGE, lw=2.0, label="capped")
    a.legend(fontsize=6, frameon=False, loc="upper right")
    a.set_title("Piecewise RUL target, Eq. (2)", fontsize=7.4,
                color=NAVY, pad=3)

    # ============ GROUP 2 : preparation =================================
    group(ax, 0.262, 0.545, 0.205, 0.375, "Preparation  ·  Module 2")
    a = inset(fig, [0.283, 0.735, 0.163, 0.135])
    for j in range(dead_trace.shape[1]):
        a.plot(dead_trace[:, j] + j * 0.02, lw=1.1, color=RED)
    a.set_title("7 near-constant channels dropped", fontsize=7.4,
                color=RED, pad=3)
    caption(ax, 0.364, 0.727, "Variance mask, Eq. (3)  ·  21 → 14 channels\n"
                              "Min–Max scaling, Eq. (4)")

    a = inset(fig, [0.283, 0.588, 0.163, 0.088])
    a.imshow(win.T, aspect="auto", cmap="viridis")
    a.set_title("One input window, Eq. (5):  30 × 14", fontsize=7.4,
                color=NAVY, pad=3)

    # ============ GROUP 3 : architecture bank ===========================
    group(ax, 0.486, 0.545, 0.215, 0.375, "Architecture Bank  ·  Module 3")
    caption(ax, 0.593, 0.905, "one network occupies the pipeline per run;\n"
                              "everything else is held fixed", fs=7.4)
    for i, (nm, eqs, col) in enumerate([
            ("1D-CNN", "Eq. (6)–(8)", BLUE),
            ("LSTM", "Eq. (9)–(11)", RED),
            ("Transformer", "Eq. (12)–(14)", TEAL),
            ("MLP", "Eq. (15)", GOLD)]):
        y = 0.812 - i * 0.062
        chip(ax, 0.506, y, 0.175, 0.048, f"{nm}   ·   {eqs}", col,
             tc=NAVY if col == GOLD else "white", fs=8.0)
    caption(ax, 0.593, 0.612, "Adam(1e-3) · MSE · batch 64\n"
                              "early stopping, Eq. (16)–(17)", fs=7.4)

    # ============ GROUP 4 : dual-regime compression =====================
    group(ax, 0.720, 0.545, 0.262, 0.375,
          "Dual-Regime Compression  ·  Module 4", ORANGE)
    chip(ax, 0.740, 0.800, 0.100, 0.062,
         "FP32\nreference\nbranch", BLUE, fs=8.0)
    chip(ax, 0.862, 0.800, 0.100, 0.062,
         "INT8\nweights +\nactivations + I/O", ORANGE, fs=7.4)
    a = inset(fig, [0.745, 0.600, 0.212, 0.155])
    q = np.clip(np.round(win / (win.std() + 1e-9) * 20), -128, 127)
    a.imshow(q.T, aspect="auto", cmap="cividis")
    a.set_title("same weights, two number systems\n"
                "Eq. (18)–(20): scale, zero-point, clip",
                fontsize=7.4, color=NAVY, pad=3)
    caption(ax, 0.851, 0.594, "100 real windows calibrate the ranges", fs=7.2)

    # ============ arrows across the top row ==============================
    arrow(ax, (0.243, 0.733), (0.262, 0.733))
    arrow(ax, (0.467, 0.733), (0.486, 0.733))
    arrow(ax, (0.701, 0.733), (0.720, 0.733))

    # ============ GROUP 5 : evaluation ==================================
    group(ax, 0.262, 0.075, 0.720, 0.365,
          "Safety-Asymmetric Evaluation & Alert Decision  ·  Module 5", RED)

    if resid is not None:
        a = inset(fig, [0.288, 0.200, 0.200, 0.162])
        a.hist(resid, bins=34, color=BLUE, edgecolor="white", lw=0.3)
        a.axvline(0, color=GREY, lw=1.4, ls="--")
        a.axvline(resid.mean(), color=RED, lw=1.8)
        a.set_title("residuals of the deployed model\n"
                    "(right of 0 = over-estimates life)",
                    fontsize=7.4, color=NAVY, pad=3)
        caption(ax, 0.388, 0.193, "Eq. (23)–(25): signed bias,\n"
                                  "late fraction, asymmetric score", fs=7.4)

    chip(ax, 0.520, 0.330, 0.190, 0.048,
         "Conventional  ·  Eq. (21)–(22)\nRMSE   MAE", GREY, fs=7.8)
    chip(ax, 0.520, 0.262, 0.190, 0.048,
         "Safety-asymmetric  ·  Eq. (23)–(25)\nsigned bias   late fraction",
         RED, fs=7.4)
    chip(ax, 0.520, 0.194, 0.190, 0.048,
         "Operational  ·  Eq. (26)\nHealthy / Warning / Critical", NAVY,
         fs=7.6)

    # alert gauge, drawn from the deployed thresholds
    a = inset(fig, [0.752, 0.200, 0.205, 0.150])
    bands = [(0, 30, RED, "Critical"), (30, 50, GOLD, "Warning"),
             (50, 130, TEAL, "Healthy")]
    for lo, hi, col, lab in bands:
        a.add_patch(Rectangle((lo, 0), hi - lo, 1, color=col, alpha=0.85))
        a.text((lo + hi) / 2, 0.5, lab, ha="center", va="center",
               fontsize=7.6, weight="bold", color="white")
    a.set_xlim(0, 130); a.set_ylim(0, 1)
    a.set_title("alert state emitted on-device, Eq. (26)",
                fontsize=7.4, color=NAVY, pad=3)
    caption(ax, 0.854, 0.193, "decision is made locally and\n"
                              "survives a network outage", fs=7.4)

    chip(ax, 0.372, 0.092, 0.500, 0.050,
         "Δ(INT8 − FP32) per metric, paired within seed, Eq. (27)   ·   "
         "6 seeds   ·   48 evaluated models", ORANGE, fs=8.6)

    arrow(ax, (0.851, 0.545), (0.851, 0.440), color=ORANGE, lw=2.2)
    arrow(ax, (0.500, 0.545), (0.500, 0.440), color=BLUE, lw=2.2)
    arrow(ax, (0.615, 0.194), (0.618, 0.142), color=GREY, lw=1.5)

    fig.savefig(os.path.join(OUT, "fig_icon_architecture.png"),
                facecolor="white", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/fig_icon_architecture.png")


if __name__ == "__main__":
    main()
