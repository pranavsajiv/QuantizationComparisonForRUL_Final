"""
Figure generator for the AeroGuard enhanced study.

Reads results/results.json (written by run_study.py) and emits every figure
and summary table. It computes NOTHING from scratch and invents NOTHING --
each figure is a rendering of measured values, and every panel that shows a
spread shows it across the actual seeds that were run.

Figures
    e1_accuracy_fp32_vs_int8.png   RMSE / MAE per architecture, both regimes
    e2_safety_bias_shift.png       *headline* -- FP32 -> INT8 shift in signed
                                   residual and in the share of engines whose
                                   remaining life is over-estimated
    e3_residual_distributions.png  per-architecture residual histograms,
                                   FP32 vs INT8 overlaid
    e4_asymmetric_score.png        C-MAPSS asymmetric score, both regimes
    e5_deployment_tradeoff.png     INT8 flatbuffer size vs RMSE vs latency
    e6_confusion_grid.png          INT8 health-status confusion per architecture
    e7_delta_heatmap.png           quantization delta across every metric
    e8_loss_curves.png             training/validation convergence per family
    e9_per_seed_deltas.png         every seed plotted individually, so a
                                   divergent training run stays visible

Tables -> results/tables.md
    E1 main results          E2 quantization deltas     E3 deployment profile
    E4 every run per seed    E5 sign test on the paired delta
Machine-readable roll-up -> results/summary.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ORDER, LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Palette matching the existing report figures (Code/create_graphs.py)
NAVY, BLUE, TEAL, ORANGE, RED, GOLD, GREY = (
    "#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#c1121f", "#e9c46a", "#6c757d")
FP32_C, INT8_C = BLUE, ORANGE

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#33415c", "axes.labelcolor": "#1b2430", "text.color": "#1b2430",
    "xtick.color": "#33415c", "ytick.color": "#33415c",
    "axes.grid": True, "grid.color": "#d9dee5", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), facecolor="white", dpi=200)
    plt.close(fig)
    print(f"  saved figures/{name}")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def load():
    with open(os.path.join(RESULTS_DIR, "results.json")) as fh:
        return json.load(fh)


def agg(runs, arch, regime, key):
    """Every seed's value of `key` for one architecture and regime."""
    vals = []
    for r in runs:
        if r["arch"] != arch:
            continue
        blk = r.get(regime, {})
        if regime == "int8" and not blk.get("converted", False):
            continue
        if key in blk:
            vals.append(blk[key])
    return np.array(vals, dtype=np.float64)


def ms(runs, arch, regime, key):
    """(mean, std) over seeds; nan if the regime never converted."""
    v = agg(runs, arch, regime, key)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=0))


def archs_present(runs):
    seen = {r["arch"] for r in runs}
    return [a for a in ORDER if a in seen]


def all_residuals(data, arch, regime):
    """Pooled (pred - true) residuals across seeds for one arch/regime."""
    y_true = np.array(data["meta"]["y_true"], dtype=np.float64)
    out = []
    for r in data["runs"]:
        if r["arch"] != arch:
            continue
        blk = r.get(regime, {})
        if regime == "int8" and not blk.get("converted", False):
            continue
        if "predictions" in blk:
            out.append(np.array(blk["predictions"], dtype=np.float64) - y_true)
    return np.concatenate(out) if out else np.array([])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_accuracy(data, A):
    runs = data["runs"]
    x = np.arange(len(A))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    series = [("rmse", "fp32", NAVY, "RMSE, FP32", -1.5),
              ("rmse", "int8", ORANGE, "RMSE, INT8", -0.5),
              ("mae", "fp32", TEAL, "MAE, FP32", 0.5),
              ("mae", "int8", GOLD, "MAE, INT8", 1.5)]
    for key, reg, col, lab, off in series:
        m = [ms(runs, a, reg, key)[0] for a in A]
        s = [ms(runs, a, reg, key)[1] for a in A]
        b = ax.bar(x + off * w, m, width=w, color=col, label=lab,
                   yerr=s, capsize=3, error_kw=dict(lw=1, ecolor="#33415c"))
        for rect, val in zip(b, m):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6,
                    f"{val:.1f}", ha="center", fontsize=7.4, color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in A], fontsize=9)
    ax.set_ylabel("Error (flight cycles)")
    ax.set_title("Conventional accuracy: FP32 vs full-static INT8\n"
                 f"(mean ± s.d. over {len(data['meta']['seeds'])} seeds, C-MAPSS FD001 test set)",
                 fontsize=11, weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=4, fontsize=8.5)
    save(fig, "e1_accuracy_fp32_vs_int8.png")


def fig_safety_shift(data, A):
    """Headline figure: does quantization move error toward the unsafe side?"""
    runs = data["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))

    # -- left: signed mean residual, FP32 -> INT8 as a slope chart ----------
    ax = axes[0]
    ends = []
    for a in A:
        f, fs = ms(runs, a, "fp32", "mean_error")
        q, qs = ms(runs, a, "int8", "mean_error")
        col = RED if q > f else TEAL
        ax.plot([0, 1], [f, q], color=col, lw=2.4, zorder=2)
        ax.errorbar([0], [f], yerr=[fs], fmt="o", ms=9, color=FP32_C,
                    capsize=4, zorder=3)
        ax.errorbar([1], [q], yerr=[qs], fmt="o", ms=9, color=INT8_C,
                    capsize=4, zorder=3)
        ends.append([q, f"{LABELS[a].splitlines()[0]}  ({q - f:+.2f})"])

    # nudge labels apart so near-identical endpoints stay readable
    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    gap = max(span * 0.13, 0.45)
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for (y_lab, text), a in zip(ends, sorted(A, key=lambda a: ms(runs, a, "int8", "mean_error")[0])):
        y_pt = ms(runs, a, "int8", "mean_error")[0]
        ax.annotate(text, xy=(1.02, y_pt), xytext=(1.14, y_lab),
                    fontsize=8.8, va="center", color=NAVY,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8,
                                    shrinkA=0, shrinkB=2))
    ax.axhline(0, color=GREY, lw=1.2, ls="--")
    ax.set_xlim(-0.25, 2.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FP32", "INT8"])
    ax.set_ylabel("Mean signed residual, $\\hat{RUL}-RUL$ (cycles)")
    ax.set_title("(a) Direction of error\nabove zero = over-estimates life = unsafe",
                 fontsize=10, weight="bold", color=NAVY)

    # -- right: share of engines whose life is over-estimated ---------------
    ax = axes[1]
    x = np.arange(len(A))
    w = 0.36
    for off, reg, col, lab in [(-w / 2, "fp32", FP32_C, "FP32"),
                               (w / 2, "int8", INT8_C, "INT8")]:
        m = [ms(runs, a, reg, "late_fraction")[0] * 100 for a in A]
        s = [ms(runs, a, reg, "late_fraction")[1] * 100 for a in A]
        b = ax.bar(x + off, m, width=w, color=col, label=lab, yerr=s,
                   capsize=3, error_kw=dict(lw=1, ecolor="#33415c"))
        for rect, val in zip(b, m):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.0,
                    f"{val:.0f}%", ha="center", fontsize=8, color=NAVY)
    ax.axhline(50, color=GREY, lw=1.2, ls="--")
    ax.text(len(A) - 0.45, 51.5, "symmetric", fontsize=7.6, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in A], fontsize=8.6)
    ax.set_ylabel("Test engines whose RUL is over-estimated (%)")
    ax.set_title("(b) How often the model is optimistic", fontsize=10,
                 weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.suptitle("Direction of quantization error, by architecture family",
                 fontsize=12, weight="bold", color=NAVY, y=1.0)
    save(fig, "e2_safety_bias_shift.png")


def fig_residuals(data, A):
    n = len(A)
    fig, axes = plt.subplots(1, n, figsize=(3.35 * n, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(-90, 90, 37)
    for ax, a in zip(axes, A):
        rf = all_residuals(data, a, "fp32")
        rq = all_residuals(data, a, "int8")
        ax.hist(rf, bins=bins, color=FP32_C, alpha=0.62, label="FP32",
                edgecolor="white", lw=0.3)
        if rq.size:
            ax.hist(rq, bins=bins, color=INT8_C, alpha=0.62, label="INT8",
                    edgecolor="white", lw=0.3)
        ax.axvline(0, color=GREY, lw=1.2, ls="--")
        if rf.size:
            ax.axvline(rf.mean(), color=FP32_C, lw=1.8)
        if rq.size:
            ax.axvline(rq.mean(), color=INT8_C, lw=1.8)
        ax.set_title(LABELS[a].replace("\n", " "), fontsize=9.5,
                     weight="bold", color=NAVY)
        ax.set_xlabel("$\\hat{RUL}-RUL$ (cycles)")
    axes[0].set_ylabel("Test engines (pooled over seeds)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Residual distributions; vertical lines mark the mean. "
                 "Right of zero is the unsafe tail.",
                 fontsize=11, weight="bold", color=NAVY)
    save(fig, "e3_residual_distributions.png")


def fig_score(data, A):
    runs = data["runs"]
    x = np.arange(len(A))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for off, reg, col, lab in [(-w / 2, "fp32", FP32_C, "FP32"),
                               (w / 2, "int8", INT8_C, "INT8")]:
        m = [ms(runs, a, reg, "cmapss_score")[0] for a in A]
        s = [ms(runs, a, reg, "cmapss_score")[1] for a in A]
        b = ax.bar(x + off, m, width=w, color=col, label=lab, yerr=s,
                   capsize=3, error_kw=dict(lw=1, ecolor="#33415c"))
        for rect, val in zip(b, m):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.06,
                    f"{val:,.0f}", ha="center", fontsize=8, color=NAVY)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in A], fontsize=8.8)
    ax.set_ylabel("Asymmetric C-MAPSS score (log scale, lower is better)")
    ax.set_title("Safety-weighted score, which penalises late\n"
                 "predictions far harder than early ones",
                 fontsize=11, weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2)
    save(fig, "e4_asymmetric_score.png")


def fig_deployment(data, A):
    runs = data["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))

    ax = axes[0]
    pts = []
    for a, col in zip(A, [NAVY, RED, TEAL, ORANGE]):
        size_kb = ms(runs, a, "int8", "size_bytes")[0] / 1024
        rmse = ms(runs, a, "int8", "rmse")[0]
        lat = ms(runs, a, "int8", "latency_ms_median")[0]
        if not np.isfinite(size_kb):
            continue
        ax.scatter(size_kb, rmse, s=120 + 900 * lat, color=col, alpha=0.75,
                   edgecolor=NAVY, lw=1.0, zorder=3)
        pts.append((a, size_kb, rmse, lat))

    # alternate label placement so the tightly-clustered small models stay legible
    for i, (a, size_kb, rmse, lat) in enumerate(sorted(pts, key=lambda p: p[2])):
        offset, ha = ((11, 10), "left") if i % 2 == 0 else ((-11, -22), "right")
        ax.annotate(f"{LABELS[a].splitlines()[0]}\n{lat:.3f} ms",
                    (size_kb, rmse), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=8.4, color=NAVY)
    ax.margins(x=0.22, y=0.16)
    ax.set_xscale("log")
    ax.set_xlabel("INT8 flatbuffer size (KB, log scale)")
    ax.set_ylabel("INT8 test RMSE (cycles)")
    ax.set_title("(a) Deployability vs accuracy\n(bubble area ∝ inference latency)",
                 fontsize=10, weight="bold", color=NAVY)

    ax = axes[1]
    x = np.arange(len(A))
    w = 0.36
    fp = [ms(runs, a, "fp32", "size_bytes")[0] / 1024 for a in A]
    q8 = [ms(runs, a, "int8", "size_bytes")[0] / 1024 for a in A]
    ax.bar(x - w / 2, fp, width=w, color=FP32_C, label="FP32")
    ax.bar(x + w / 2, q8, width=w, color=INT8_C, label="INT8")
    for i, (f, q) in enumerate(zip(fp, q8)):
        if np.isfinite(q) and q > 0:
            ax.text(i, max(f, q) * 1.04, f"{f / q:.2f}×",
                    ha="center", fontsize=8.6, weight="bold", color=NAVY)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in A], fontsize=8.6)
    ax.set_ylabel("TFLite flatbuffer size (KB)")
    ax.set_title("(b) Actual compression achieved\n(label = FP32 ÷ INT8)",
                 fontsize=10, weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2)

    fig.suptitle("Deployment cost of each architecture family",
                 fontsize=12, weight="bold", color=NAVY, y=1.0)
    save(fig, "e5_deployment_tradeoff.png")


def fig_confusion(data, A):
    runs = data["runs"]
    classes = ["Healthy", "Warning", "Critical"]
    n = len(A)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, A):
        cms = [np.array(r["int8"]["confusion_matrix"])
               for r in runs
               if r["arch"] == a and r.get("int8", {}).get("converted")]
        if not cms:
            ax.set_visible(False)
            continue
        cm = np.sum(cms, axis=0)
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                unsafe = (i == 2 and j == 0)
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=11.5, weight="bold",
                        color=RED if unsafe and cm[i, j] > 0
                        else ("white" if cm[i, j] > cm.max() / 2 else NAVY))
        # outline the one cell that matters most: failing engine called healthy
        ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False,
                                   edgecolor=RED, lw=2.2))
        ax.set_xticks(range(3))
        ax.set_xticklabels(classes, fontsize=8, rotation=20)
        ax.set_yticks(range(3))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8.5)
        ax.set_title(LABELS[a].replace("\n", " "), fontsize=9.5,
                     weight="bold", color=NAVY)
        ax.grid(False)
    axes[0].set_ylabel("True", fontsize=8.5)
    fig.suptitle("INT8 health-status confusion, summed over seeds. "
                 "Red box = a failing engine reported as healthy.",
                 fontsize=10.5, weight="bold", color=NAVY)
    save(fig, "e6_confusion_grid.png")


# (json key, display label, +1 if a LARGER delta is worse, -1 if it is better)
DELTA_KEYS = [
    ("rmse", "RMSE", +1),
    ("mae", "MAE", +1),
    ("cmapss_score", "Asymmetric score", +1),
    ("mean_error", "Mean signed residual", +1),
    ("late_fraction", "Over-estimated share", +1),
    ("p90_overestimate", "p90 over-estimate", +1),
    ("status_accuracy", "Status accuracy", -1),
    ("critical_missed", "Critical engines missed", +1),
]


def fig_delta_heatmap(data, A):
    runs = data["runs"]
    M = np.full((len(DELTA_KEYS), len(A)), np.nan)
    for j, a in enumerate(A):
        for i, (k, _, _) in enumerate(DELTA_KEYS):
            f = ms(runs, a, "fp32", k)[0]
            q = ms(runs, a, "int8", k)[0]
            M[i, j] = q - f

    # Normalise each row to its own max magnitude so one large-scale metric
    # (the asymmetric score) does not flatten the rest of the map, and orient
    # every row so that RED always means "INT8 is worse" -- status accuracy is
    # the one metric where a larger delta is an improvement.
    norm = np.zeros_like(M)
    for i, (_, _, sign) in enumerate(DELTA_KEYS):
        mx = np.nanmax(np.abs(M[i]))
        norm[i] = (M[i] / mx * sign) if mx and np.isfinite(mx) else 0.0

    fig, ax = plt.subplots(figsize=(1.9 * len(A) + 4.4, 5.4))
    im = ax.imshow(norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                txt = "n/a"
            elif abs(v) >= 100:
                txt = f"{v:+,.0f}"
            elif abs(v) >= 1:
                txt = f"{v:+.2f}"
            else:
                txt = f"{v:+.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    weight="bold",
                    color="white" if abs(norm[i, j]) > 0.55 else NAVY)
    ax.set_xticks(range(len(A)))
    ax.set_xticklabels([LABELS[a] for a in A], fontsize=9)
    ax.set_yticks(range(len(DELTA_KEYS)))
    ax.set_yticklabels([lbl for _, lbl, _ in DELTA_KEYS], fontsize=9)
    ax.grid(False)
    ax.set_title("Quantization delta (INT8 − FP32), mean over seeds\n"
                 "Red = INT8 is worse on that metric. Colour is scaled per row.",
                 fontsize=11, weight="bold", color=NAVY)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="relative within row")
    save(fig, "e7_delta_heatmap.png")


def paired_deltas(data, arch, key):
    """Per-seed (INT8 - FP32) for one metric. Pairing matters: comparing the
    mean of one regime to the mean of the other would let seed-to-seed
    training variance swamp the quantization effect."""
    out = []
    for r in data["runs"]:
        if r["arch"] != arch:
            continue
        f, q = r.get("fp32", {}), r.get("int8", {})
        if not q.get("converted", False) or key not in f or key not in q:
            continue
        out.append((r["seed"], q[key] - f[key]))
    return out


def fig_per_seed(data, A):
    """Show every seed, so a single divergent training run is visible rather
    than averaged into a misleading mean."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=False)
    for ax, key, lab in [
            (axes[0], "rmse", "$\\Delta$ RMSE (cycles)"),
            (axes[1], "mean_error", "$\\Delta$ mean signed residual (cycles)")]:
        for j, a in enumerate(A):
            d = paired_deltas(data, a, key)
            if not d:
                continue
            vals = [v for _, v in d]
            xs = np.full(len(vals), j) + np.linspace(-0.16, 0.16, len(vals))
            cols = [RED if v > 0 else TEAL for v in vals]
            ax.scatter(xs, vals, c=cols, s=64, zorder=3,
                       edgecolor=NAVY, lw=0.7)
            ax.hlines(np.mean(vals), j - 0.28, j + 0.28,
                      color=NAVY, lw=2.4, zorder=4)
            n_up = sum(1 for v in vals if v > 0)
            ax.text(j, ax.get_ylim()[1], "", fontsize=1)
            ax.annotate(f"{n_up}/{len(vals)}", (j, max(vals)),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.6, weight="bold", color=NAVY)
        ax.axhline(0, color=GREY, lw=1.3, ls="--")
        ax.set_xticks(range(len(A)))
        ax.set_xticklabels([LABELS[a] for a in A], fontsize=8.6)
        ax.set_ylabel(lab)
    axes[0].set_title("(a) Accuracy cost of quantization", fontsize=10,
                      weight="bold", color=NAVY)
    axes[1].set_title("(b) Safety-direction cost\nred = shifted toward "
                      "over-estimating life", fontsize=10,
                      weight="bold", color=NAVY)
    fig.suptitle("Every seed shown individually; bar = mean, "
                 "label = seeds that moved the unsafe way",
                 fontsize=11, weight="bold", color=NAVY, y=1.0)
    save(fig, "e9_per_seed_deltas.png")


def fig_loss(data, A):
    runs = data["runs"]
    seed0 = data["meta"]["seeds"][0]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for a, col in zip(A, [NAVY, RED, TEAL, ORANGE]):
        r = next((r for r in runs if r["arch"] == a and r["seed"] == seed0), None)
        if r is None:
            continue
        h = r["history"]
        e = range(1, len(h["loss"]) + 1)
        ax.plot(e, h["loss"], color=col, lw=1.9,
                label=f"{LABELS[a].splitlines()[0]} — train")
        ax.plot(e, h["val_loss"], color=col, lw=1.5, ls="--", alpha=0.75,
                label=f"{LABELS[a].splitlines()[0]} — val")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (cycles$^2$, log scale)")
    ax.set_title(f"Convergence by architecture family (seed {seed0})",
                 fontsize=11, weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "e8_loss_curves.png")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def write_tables(data, A):
    runs = data["runs"]
    nseeds = len(data["meta"]["seeds"])

    def cell(a, reg, k, fmt="{:.2f}"):
        m, s = ms(runs, a, reg, k)
        if not np.isfinite(m):
            return "n/a"
        return f"{fmt.format(m)} ± {fmt.format(s)}"

    lines = []
    lines.append(f"# Enhanced study results ({nseeds} seeds, C-MAPSS FD001)\n")
    lines.append(f"Generated {data['meta']['generated_utc']} · "
                 f"TensorFlow {data['meta']['tensorflow']} · "
                 f"{data['meta']['quantization']}\n")

    lines.append("\n## Table E1 — Main results (mean ± s.d. over seeds)\n")
    hdr = ("| Architecture | Params | Regime | RMSE | MAE | Asym. score | "
           "Mean signed residual | Over-est. share | Status acc. |")
    lines += [hdr, "|" + "---|" * 9]
    for a in A:
        p = int(np.mean([r["n_params"] for r in runs if r["arch"] == a]))
        for reg in ["fp32", "int8"]:
            lines.append(
                f"| {LABELS[a].replace(chr(10), ' ')} | {p:,} | {reg.upper()} | "
                f"{cell(a, reg, 'rmse')} | {cell(a, reg, 'mae')} | "
                f"{cell(a, reg, 'cmapss_score', '{:,.0f}')} | "
                f"{cell(a, reg, 'mean_error', '{:+.2f}')} | "
                f"{cell(a, reg, 'late_fraction', '{:.3f}')} | "
                f"{cell(a, reg, 'status_accuracy', '{:.3f}')} |")

    lines.append("\n## Table E2 — Quantization delta (INT8 − FP32)\n")
    lines += ["| Architecture | " + " | ".join(l for _, l, _ in DELTA_KEYS) + " |",
              "|" + "---|" * (len(DELTA_KEYS) + 1)]
    for a in A:
        row = [LABELS[a].replace("\n", " ")]
        for k, _, _ in DELTA_KEYS:
            f = ms(runs, a, "fp32", k)[0]
            q = ms(runs, a, "int8", k)[0]
            d = q - f
            row.append("n/a" if not np.isfinite(d)
                       else (f"{d:+,.0f}" if abs(d) >= 100 else f"{d:+.3f}"))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Table E3 — Deployment profile\n")
    lines += ["| Architecture | FP32 KB | INT8 KB | Compression | "
              "FP32 latency (ms) | INT8 latency (ms) | Epochs | Train (s) |",
              "|" + "---|" * 8]
    for a in A:
        f_kb = ms(runs, a, "fp32", "size_bytes")[0] / 1024
        q_kb = ms(runs, a, "int8", "size_bytes")[0] / 1024
        comp = f"{f_kb / q_kb:.2f}×" if np.isfinite(q_kb) and q_kb else "n/a"
        ep = np.mean([r["epochs_run"] for r in runs if r["arch"] == a])
        tr = np.mean([r["train_seconds"] for r in runs if r["arch"] == a])
        lines.append(
            f"| {LABELS[a].replace(chr(10), ' ')} | {f_kb:.1f} | "
            f"{q_kb:.1f} | {comp} | "
            f"{ms(runs, a, 'fp32', 'latency_ms_median')[0]:.3f} | "
            f"{ms(runs, a, 'int8', 'latency_ms_median')[0]:.3f} | "
            f"{ep:.1f} | {tr:.0f} |")

    lines.append("\n## Table E4 — Every run, per seed (no exclusions)\n")
    lines.append("Listed in full so that divergent training runs are visible "
                 "rather than absorbed into a mean.\n")
    lines += ["| Architecture | Seed | Epochs | FP32 RMSE | INT8 RMSE | Δ RMSE | "
              "FP32 bias | INT8 bias | Δ bias | INT8 converted |",
              "|" + "---|" * 10]
    for a in A:
        for r in sorted((r for r in runs if r["arch"] == a),
                        key=lambda r: r["seed"]):
            f, q = r["fp32"], r.get("int8", {})
            ok = q.get("converted", False)
            lines.append(
                f"| {LABELS[a].replace(chr(10), ' ')} | {r['seed']} | "
                f"{r['epochs_run']} | {f['rmse']:.2f} | "
                f"{q['rmse']:.2f} | {q['rmse'] - f['rmse']:+.2f} | "
                f"{f['mean_error']:+.2f} | {q['mean_error']:+.2f} | "
                f"{q['mean_error'] - f['mean_error']:+.2f} | yes |"
                if ok else
                f"| {LABELS[a].replace(chr(10), ' ')} | {r['seed']} | "
                f"{r['epochs_run']} | {f['rmse']:.2f} | n/a | n/a | "
                f"{f['mean_error']:+.2f} | n/a | n/a | NO |")

    lines.append("\n## Table E5 — Sign test on the paired quantization delta\n")
    lines.append("How many seeds moved in the *unsafe* direction "
                 "(INT8 more optimistic than FP32), out of how many ran.\n")
    lines += ["| Architecture | Seeds | Δ bias > 0 | Δ RMSE > 0 | "
              "mean Δ bias | mean Δ RMSE |",
              "|" + "---|" * 6]
    for a in A:
        db = [v for _, v in paired_deltas(data, a, "mean_error")]
        dr = [v for _, v in paired_deltas(data, a, "rmse")]
        if not db:
            continue
        lines.append(
            f"| {LABELS[a].replace(chr(10), ' ')} | {len(db)} | "
            f"{sum(1 for v in db if v > 0)}/{len(db)} | "
            f"{sum(1 for v in dr if v > 0)}/{len(dr)} | "
            f"{np.mean(db):+.2f} | {np.mean(dr):+.2f} |")

    path = os.path.join(RESULTS_DIR, "tables.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  saved results/tables.md")

    # machine-readable summary for anyone reusing the numbers
    summary = {}
    for a in A:
        summary[a] = {
            "n_params": int(np.mean([r["n_params"] for r in runs if r["arch"] == a])),
            **{f"{reg}_{k}": dict(zip(["mean", "std"], ms(runs, a, reg, k)))
               for reg in ["fp32", "int8"]
               for k in ["rmse", "mae", "cmapss_score", "mean_error",
                         "late_fraction", "p90_overestimate", "status_accuracy",
                         "critical_missed", "critical_called_healthy",
                         "size_bytes", "latency_ms_median"]},
        }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("  saved results/summary.json")


def main():
    data = load()
    A = archs_present(data["runs"])
    print(f"Rendering figures for {len(A)} architectures, "
          f"{len(data['meta']['seeds'])} seeds ...")
    fig_accuracy(data, A)
    fig_safety_shift(data, A)
    fig_residuals(data, A)
    fig_score(data, A)
    fig_deployment(data, A)
    fig_confusion(data, A)
    fig_delta_heatmap(data, A)
    fig_per_seed(data, A)
    fig_loss(data, A)
    write_tables(data, A)
    print("Done.")


if __name__ == "__main__":
    main()
