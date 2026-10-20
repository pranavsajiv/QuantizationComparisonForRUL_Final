"""
Figures for the ablation study and the hyperparameter sweeps.

Reads results/ablations.json only, and adapts to however many architectures
that file contains. Variants whose preprocessing failed outright are drawn as
an explicit FAILED marker rather than silently omitted, because that failure is
one of the study's results.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")

NAVY, BLUE, TEAL, ORANGE, RED, GOLD, GREY = (
    "#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#c1121f", "#e9c46a", "#6c757d")

ORDER = ["cnn", "lstm", "transformer", "mlp"]
TITLE = {"cnn": "1D-CNN", "lstm": "LSTM",
         "transformer": "Transformer", "mlp": "MLP"}
ACOLOR = {"cnn": BLUE, "lstm": RED, "transformer": TEAL, "mlp": GOLD}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#33415c", "axes.labelcolor": "#1b2430",
    "text.color": "#1b2430", "xtick.color": "#33415c", "ytick.color": "#33415c",
    "axes.grid": True, "grid.color": "#d9dee5", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

VARIANTS = [
    ("full", "Proposed\n(full system)"),
    ("no_rul_cap", "No piecewise\nRUL cap"),
    ("all_21_naive", "All 21 sensors\nnaive scaler"),
    ("all_21_guarded", "All 21 sensors\nguarded scaler"),
    ("short_window_L15", "Shorter window\nL = 15"),
]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), facecolor="white", dpi=200)
    plt.close(fig)
    print(f"  saved figures/{name}")


def ms(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std(ddof=0))) if v.size else (np.nan, np.nan)


def load():
    with open(os.path.join(RES, "ablations.json")) as fh:
        return json.load(fh)


def archs_in(records):
    seen = {r["arch"] for r in records}
    return [a for a in ORDER if a in seen]


def grid(n):
    """Panel layout for n architectures."""
    if n <= 2:
        return 1, n
    return 2, (n + 1) // 2


def fig_ablation(abl):
    A = archs_in(abl["ablations"])
    nr, nc = grid(len(A))
    fig, axes = plt.subplots(nr, nc, figsize=(6.5 * nc, 4.6 * nr), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, arch in zip(axes, A):
        labels, fm, fs, qm, qs, failed = [], [], [], [], [], []
        for tag, lab in VARIANTS:
            sel = [r for r in abl["ablations"]
                   if r["tag"] == tag and r["arch"] == arch]
            if not sel:
                continue
            labels.append(lab)
            if all(r.get("preprocessing_failed") for r in sel):
                fm.append(0); fs.append(0); qm.append(0); qs.append(0)
                failed.append(True)
                continue
            ok = [r for r in sel if not r.get("preprocessing_failed")]
            a, b = ms([r["fp32"]["rmse"] for r in ok])
            fm.append(a); fs.append(b)
            q = [r["int8_by_calibration"].get("100", {}) for r in ok]
            q = [x for x in q if x.get("converted")]
            c, d = ms([x["rmse"] for x in q])
            qm.append(c); qs.append(d)
            failed.append(False)

        x = np.arange(len(labels)); w = 0.36
        ax.bar(x - w / 2, fm, width=w, color=BLUE, label="FP32", yerr=fs,
               capsize=3, error_kw=dict(lw=1, ecolor="#33415c"))
        ax.bar(x + w / 2, qm, width=w, color=ORANGE, label="INT8", yerr=qs,
               capsize=3, error_kw=dict(lw=1, ecolor="#33415c"))
        top = max([v for v in fm + qm if np.isfinite(v)] + [1])
        for i, (f, q, bad) in enumerate(zip(fm, qm, failed)):
            if bad:
                ax.text(i, top * 0.06, "PIPELINE\nFAILS\n(NaN)", ha="center",
                        va="bottom", fontsize=8.4, weight="bold", color=RED)
            else:
                ax.text(i - w / 2, f + top * 0.02, f"{f:.1f}", ha="center",
                        fontsize=7.6, color=NAVY)
                ax.text(i + w / 2, q + top * 0.02, f"{q:.1f}", ha="center",
                        fontsize=7.6, color=NAVY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.8)
        ax.set_ylabel("Test RMSE (cycles)")
        ax.set_title(TITLE[arch], fontsize=11, weight="bold",
                     color=ACOLOR[arch])
    for ax in axes[len(A):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Ablation study: each variant changes exactly one design "
                 "decision (mean ± s.d. over 3 seeds)",
                 fontsize=11.5, weight="bold", color=NAVY, y=1.0)
    save(fig, "e10_ablation.png")


def fig_window(abl):
    A = archs_in(abl["sweep_window"])
    nr, nc = grid(len(A))
    fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.4 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, arch in zip(axes, A):
        Ls = sorted({r["seq_len"] for r in abl["sweep_window"]
                     if r["arch"] == arch})
        f_m, f_s, q_m, q_s = [], [], [], []
        for L in Ls:
            sel = [r for r in abl["sweep_window"]
                   if r["arch"] == arch and r["seq_len"] == L]
            a, b = ms([r["fp32"]["rmse"] for r in sel])
            f_m.append(a); f_s.append(b)
            q = [r["int8_by_calibration"].get("100", {}) for r in sel]
            q = [x for x in q if x.get("converted")]
            c, d = ms([x["rmse"] for x in q])
            q_m.append(c); q_s.append(d)
        ax.errorbar(Ls, f_m, yerr=f_s, marker="o", color=BLUE, lw=2.0, ms=6,
                    capsize=3, label="FP32")
        ax.errorbar(Ls, q_m, yerr=q_s, marker="s", color=ORANGE, lw=2.0, ms=6,
                    capsize=3, label="INT8")
        if f_m:
            bf = Ls[int(np.nanargmin(f_m))]
            bq = Ls[int(np.nanargmin(q_m))]
            ax.scatter([bf], [np.nanmin(f_m)], s=190, facecolor="none",
                       edgecolor=BLUE, lw=2, zorder=5)
            ax.scatter([bq], [np.nanmin(q_m)], s=190, facecolor="none",
                       edgecolor=ORANGE, lw=2, zorder=5)
            agree = "regimes agree" if bf == bq else "regimes DISAGREE"
            ax.set_title(f"{TITLE[arch]} — best L: FP32 = {bf}, INT8 = {bq} "
                         f"({agree})", fontsize=10.0, weight="bold",
                         color=ACOLOR[arch] if bf == bq else RED)
        ax.axvline(30, color=GREY, ls=":", lw=1.4)
        ax.set_xlabel("Sliding-window length, L (cycles)")
        ax.set_ylabel("Test RMSE (cycles)")
        ax.set_xticks(Ls)
    for ax in axes[len(A):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Window-length sweep, scored in both numeric regimes "
                 "(mean ± s.d. over 3 seeds); the dotted line marks the "
                 "deployed L = 30",
                 fontsize=11.5, weight="bold", color=NAVY, y=1.0)
    save(fig, "e11_window_sweep.png")


def fig_calibration(abl):
    A = archs_in(abl["sweep_calibration"])
    sizes = sorted({int(k) for r in abl["sweep_calibration"]
                    for k in r["int8_by_calibration"]})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, key, lab in [(axes[0], "rmse", "INT8 test RMSE (cycles)"),
                         (axes[1], "mean_error",
                          "INT8 mean signed residual (cycles)")]:
        for arch in A:
            m, s = [], []
            for n in sizes:
                vals = [r["int8_by_calibration"].get(str(n), {})
                        for r in abl["sweep_calibration"] if r["arch"] == arch]
                vals = [v[key] for v in vals if v.get("converted")]
                a, b = ms(vals)
                m.append(a); s.append(b)
            ax.errorbar(sizes, m, yerr=s, marker="o", color=ACOLOR[arch],
                        lw=2.0, ms=6, capsize=3, label=TITLE[arch])
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel("Representative calibration windows, n")
        ax.set_ylabel(lab)
        if key == "mean_error":
            ax.axhline(0, color=GREY, ls="--", lw=1.2)
    axes[0].legend(frameon=False, ncol=2, fontsize=8.5)
    fig.suptitle("Calibration-set size sweep, trained weights held fixed "
                 "(mean ± s.d. over 3 seeds)",
                 fontsize=11.5, weight="bold", color=NAVY, y=1.0)
    save(fig, "e12_calibration_sweep.png")


def main():
    abl = load()
    print("Rendering ablation / sweep figures ...")
    if abl.get("ablations"):
        fig_ablation(abl)
    if abl.get("sweep_window"):
        fig_window(abl)
    if abl.get("sweep_calibration"):
        fig_calibration(abl)
    print("Done.")


if __name__ == "__main__":
    main()
