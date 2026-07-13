"""Clean locus test: is the De Carli DD deficit resolution or sensory noise?

Strengthens the DD argument (SM5) against the reviewer's suggestion to drop it.
Two restricted, single-mechanism models are fit to each subject's ACCURACY curve
(n = 1..7) with an exhaustive **global** grid search over one free parameter and a
**fixed common lapse** (no escape hatch):

  * RESOLUTION-only : sensory noise off (s0 = floor); free well-width scale sigma_c
                      per subject → per-subject n_crit (σ_n = 1). Accuracy decline
                      is carried entirely by the discrete code merging.
  * NOISE-only      : well-width off (σ_c tiny → wells never merge in range); free
                      sensory noise s0 per subject → per-subject σ(n). Accuracy
                      decline is carried entirely by sensory noise.

Shared shape params (beta, lambda0) and the common lapse come from a pooled fit.
We then ask: (a) does the DD-vs-control group difference localise in each model,
and (b) which single mechanism *fits the accuracy curves better* (paired SSE). If
resolution both localises the group effect AND fits better, the "coarser discrete
code in DD" account is supported over "noisier perception".

    .venv/bin/python -m scripts.decarli_locus_clean
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import mannwhitneyu, wilcoxon

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.decarli_gaussian_freebeta import DATA, NVALS, GROUPS, ncrit, evidence_sd

OUT_DIR = _ROOT / "output" / "decarli_fit"
S0_FLOOR = 1e-6            # noise "off"
SIGMAC_OFF = 0.003        # well-width "off" (σ_well << 1 across n=1..7)
NG = 200                  # grid resolution for the exhaustive 1-D search


def acc_curve(sigma_c, s0, lambda0, pow_, lapse, quad) -> np.ndarray:
    theta = {"sigma_c": sigma_c, "s0": s0, "lambda0": lambda0, "pow": pow_}
    grid = T.compute_grid(SPEC := T.MODEL_BY_KEY[T.WINNER_KEY],
                          theta, quad, n_values=NVALS.astype(float))
    p = np.array([grid["p_resp"][i, int(n) - 1] for i, n in enumerate(NVALS)])
    return (1.0 - lapse) * p + lapse / T.N_WELLS


def subj_acc(df, grp, s) -> np.ndarray:
    g = df[(df.Group == grp) & (df.Sub == s)]
    return np.array([float(g[g.Numerosity == n].Accuracy.mean()) for n in NVALS])


def pooled_common(df, quad) -> dict:
    """Pooled accuracy fit → shared pow, lambda0, lapse (+ reference sigma_c, s0)."""
    obs = np.array([float(df[df.Numerosity == n].Accuracy.mean()) for n in NVALS])
    bounds = [(0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0), (1e-4, 0.5)]

    def neg(v):
        try:
            pred = acc_curve(*v, quad)
        except (FloatingPointError, ValueError):
            return 1e18
        return float(np.mean((pred - obs) ** 2)) if np.all(np.isfinite(pred)) else 1e18

    de = differential_evolution(neg, bounds, maxiter=120, popsize=15, tol=1e-9,
                                seed=5, init="latinhypercube", polish=False,
                                updating="deferred", workers=1)
    pol = minimize(neg, de.x, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 600, "xatol": 1e-7, "fatol": 1e-9})
    v = pol.x if pol.fun <= de.fun else de.x
    return {"sigma_c": v[0], "s0": v[1], "lambda0": v[2], "pow": v[3], "lapse": v[4]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = T.make_quadrature(301, kind="grid")
    df = pd.read_csv(DATA)
    common = pooled_common(df, quad)
    print(f"Pooled common: pow={common['pow']:.2f} lambda0={common['lambda0']:.2f} "
          f"lapse={common['lapse']:.3f}", flush=True)

    # Precompute predicted accuracy curves along each 1-D grid (landscape common
    # except the one varying param + common lapse), then match each subject by SSE.
    sc_grid = np.geomspace(0.005, 0.5, NG)                  # resolution model
    a_curves = np.array([acc_curve(sc, S0_FLOOR, common["lambda0"], common["pow"],
                                   common["lapse"], quad) for sc in sc_grid])
    s0_grid = np.geomspace(1e-4, 0.5, NG)                   # noise model
    b_curves = np.array([acc_curve(SIGMAC_OFF, s0, common["lambda0"], common["pow"],
                                   common["lapse"], quad) for s0 in s0_grid])

    rows = []
    for grp in GROUPS:
        for s in sorted(df[df.Group == grp].Sub.unique()):
            obs = subj_acc(df, grp, s)
            sse_a = np.mean((a_curves - obs[None, :]) ** 2, axis=1)
            ia = int(np.argmin(sse_a))
            sse_b = np.mean((b_curves - obs[None, :]) ** 2, axis=1)
            ib = int(np.argmin(sse_b))
            theta_a = {"sigma_c": sc_grid[ia], "s0": S0_FLOOR,
                       "lambda0": common["lambda0"], "pow": common["pow"]}
            rows.append({
                "group": grp, "subject": int(s),
                "sigma_c_A": sc_grid[ia], "n_crit_A": ncrit(theta_a, 1.0),
                "sse_A": float(sse_a[ia]),
                "s0_B": s0_grid[ib],
                "sigma_n5_B": evidence_sd({"s0": s0_grid[ib], "lambda0": common["lambda0"]}, 5),
                "sse_B": float(sse_b[ib]),
            })
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT_DIR / "gaussian_freebeta_locus_clean.csv", index=False)

    def mw(col):
        dd = sub[sub.group == "DD"][col].to_numpy()
        ct = sub[sub.group == "contr"][col].to_numpy()
        return np.median(dd), np.median(ct), mannwhitneyu(dd, ct, alternative="two-sided")[1]

    print("\n=== group localisation (global search, lapse fixed) ===")
    md, mc, p = mw("n_crit_A")
    print(f"  RESOLUTION  n_crit: DD {md:.2f} vs contr {mc:.2f}  MW p={p:.4f}")
    md, mc, p = mw("sigma_n5_B")
    print(f"  NOISE     σ(n=5):  DD {md:.4f} vs contr {mc:.4f}  MW p={p:.4f}")

    print("\n=== which single mechanism fits the accuracy curves better? ===")
    a, b = sub["sse_A"].to_numpy(), sub["sse_B"].to_numpy()
    wp = wilcoxon(a, b).pvalue
    n_res_better = int(np.sum(a < b))
    print(f"  mean SSE: resolution {a.mean():.5f}  vs  noise {b.mean():.5f}")
    print(f"  resolution fits better in {n_res_better}/{len(a)} subjects  "
          f"(Wilcoxon paired p={wp:.4g})")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.2))
    colors = {"DD": "#c0392b", "contr": "#2c7fb8"}
    for grp in GROUPS:
        g = sub[sub.group == grp]
        ax[0].scatter(g["sse_B"], g["sse_A"], s=18, color=colors[grp], alpha=0.7,
                      label=grp)
    lim = [0, max(sub["sse_A"].max(), sub["sse_B"].max()) * 1.05]
    ax[0].plot(lim, lim, "0.6", lw=1)
    ax[0].set_xlabel("SSE — noise-only model"); ax[0].set_ylabel("SSE — resolution-only model")
    ax[0].set_title("below diagonal = resolution fits better"); ax[0].legend(frameon=False, fontsize=8)
    dd = sub[sub.group == "DD"]["n_crit_A"].to_numpy()
    ct = sub[sub.group == "contr"]["n_crit_A"].to_numpy()
    parts = ax[1].violinplot([ct, dd], positions=[0, 1], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], ["#2c7fb8", "#c0392b"]):
        pc.set_facecolor(c); pc.set_alpha(0.4)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["control", "DD"])
    ax[1].set_ylabel("n_crit (resolution-only, global fit)")
    ax[1].set_title(f"n_crit  MW p={mw('n_crit_A')[2]:.3f}")
    fig.suptitle("De Carli DD deficit — clean locus test (accuracy-only)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_freebeta_locus_clean.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved gaussian_freebeta_locus_clean.{{csv,png}} in {OUT_DIR}")


if __name__ == "__main__":
    main()
