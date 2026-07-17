"""Consolidated De Carli DD-vs-control figure.

  (A) raw accuracy (mean ± SEM per group, n = 1..7) with per-n significance and the
      winner-model group fit overlaid  — model-agnostic deficit (Tier 1).
  (B) raw median RT (mean ± SEM per group) with the model group fit               (Tier 1).
  (C) per-subject model comparison: resolution-only vs noise-only SSE — every
      subject below the diagonal → the accuracy decline is resolution-shaped (Tier 2).
  (D) per-subject n_width (the continuous width-based index), DD vs control.

Reuses the saved fits/CSVs; non-destructive. Writes decarli_dd_summary.png.

    .venv/bin/python -m scripts.decarli_dd_summary_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.decarli_gaussian_freebeta import DATA, NVALS, GROUPS, OUT_DIR, predict, PNAMES
from figures.plot_style import C_ATTR, C_WARN, apply_style, panel_label

COL = {"DD": C_WARN, "contr": C_ATTR}
LAB = {"DD": "DD", "contr": "Control"}
OUT_FIG = OUT_DIR / "decarli_dd_summary.png"


def main() -> None:
    apply_style(2.0)   # rendered at \columnwidth, 11 in canvas
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)
    fitted = pd.read_csv(OUT_DIR / "gaussian_freebeta_fitted.csv").set_index("group")
    locus = pd.read_csv(OUT_DIR / "gaussian_freebeta_locus_clean.csv")

    # per-group mean ± SEM of accuracy and RT per numerosity
    def summ(metric):
        out = {}
        for grp in GROUPS:
            g = df[df.Group == grp]
            m = np.array([g[g.Numerosity == n][metric].to_numpy() for n in NVALS], dtype=object)
            mean = np.array([v.mean() for v in m]); sem = np.array([v.std(ddof=1) / np.sqrt(v.size) for v in m])
            out[grp] = (mean, sem)
        return out
    acc, rt = summ("Accuracy"), summ("RTs")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))

    # (A) accuracy
    a = ax[0][0]
    for grp in GROUPS:
        mean, sem = acc[grp]
        a.errorbar(NVALS, 100*mean, yerr=100*sem, fmt="o", color=COL[grp], ms=6,
                   capsize=3)
        theta_vec = np.array([fitted.loc[grp, p] for p in PNAMES])
        pe, _ = predict(theta_vec, quad)
        a.plot(NVALS, 100*pe, "-", color=COL[grp], lw=1.4, alpha=0.6)
    a.set_xlabel("Numerosity"); a.set_ylabel("% exact"); a.set_ylim(30, 105)
    a.set_xticks(NVALS)
    # legend shared with (b): color = group, style = data vs model
    handles = [Patch(facecolor=COL["DD"], label="DD"),
               Patch(facecolor=COL["contr"], label="Control"),
               a.errorbar([np.nan], [np.nan], yerr=[1], fmt="o", color="0.3", ms=6,
                          capsize=3, label="Data"),
               Line2D([], [], color="0.3", lw=1.4, label="Model")]
    a.legend(handles=handles, frameon=False, loc="lower left"); panel_label(a, "a")

    # (B) RT
    b = ax[0][1]
    for grp in GROUPS:
        mean, sem = rt[grp]
        b.errorbar(NVALS, mean, yerr=sem, fmt="o", color=COL[grp], ms=6, capsize=3)
        theta_vec = np.array([fitted.loc[grp, p] for p in PNAMES])
        _, prt = predict(theta_vec, quad)
        b.plot(NVALS, prt, "-", color=COL[grp], lw=1.4, alpha=0.6)
    b.set_xlabel("Numerosity"); b.set_ylabel("Median RT (s)")
    b.set_xticks(NVALS)
    panel_label(b, "b")

    # (C) model comparison SSE
    c = ax[1][0]
    for grp in GROUPS:
        g = locus[locus.group == grp]
        c.scatter(g.sse_B, g.sse_A, s=20, color=COL[grp], alpha=0.75, label=LAB[grp])
    lim = [0, max(locus.sse_A.max(), locus.sse_B.max()) * 1.05]
    c.plot(lim, lim, "0.6", lw=1)
    c.set_xlim(lim); c.set_ylim(lim)
    c.set_xlabel("SSE, noise-only"); c.set_ylabel("SSE, resolution-only")
    c.legend(frameon=False, loc="upper left"); panel_label(c, "c")

    # (D) n_width from per-subject free-scale fits with a shared exponent.
    d = ax[1][1]
    width = pd.read_csv(OUT_DIR / "gaussian_freebeta_locus_beta.csv")
    dd = width[width.group == "DD"].n_width_A.to_numpy()
    ct = width[width.group == "contr"].n_width_A.to_numpy()
    u_dd, _ = mannwhitneyu(dd, ct, alternative="two-sided")
    r_rb = 2 * u_dd / (dd.size * ct.size) - 1

    rng = np.random.default_rng(17)
    for position, values, group in ((0, ct, "contr"), (1, dd, "DD")):
        d.scatter(position + rng.normal(0, 0.045, values.size), values, s=20,
                  color=COL[group], alpha=0.75, label=LAB[group])
        d.hlines(np.median(values), position - 0.18, position + 0.18,
                 color="black", linewidth=1.4)
    d.set_xticks([0, 1]); d.set_xticklabels([LAB["contr"], LAB["DD"]])
    d.set_ylabel(r"Per-subject $n_{\rm width}$")
    d.text(0.02, 0.97, rf"$r_{{\rm rb}}$ = {r_rb:.2f}", transform=d.transAxes,
           ha="left", va="top", fontsize=plt.rcParams["axes.labelsize"])
    panel_label(d, "d")

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {OUT_FIG} and .pdf")


if __name__ == "__main__":
    main()
