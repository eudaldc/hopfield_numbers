"""Per-subject De Carli fit with the winner model → individual n_width distributions.

The group-level fit (`decarli_gaussian_freebeta.py`) hides between-subject spread. We
DO have individual data (32 subjects/group, accuracy + RT per numerosity), so here we
estimate an n_width *per subject* and compare the DD vs control distributions.

Identifiability: a subject's accuracy curve is only ~8 coarse points, too thin for a
full 5-parameter landscape fit. So we hold the shared **mechanism** — the width
exponent β and the (floor-level) sensory-noise profile s0, λ0 — at values fitted on
the *pooled* data, and let each subject have their own well-width scale σ_c (plus
their own RT baseline/scale and lapse). n_width(σ_n = 1) is then driven by the
subject's σ_c and summarizes the fitted width profile. Groups are compared
with a Mann–Whitney U test on the per-subject n_width (and σ_c).

Writes gaussian_freebeta_subjects.csv + gaussian_freebeta_subjects.png under
output/decarli_fit/. Non-destructive.

SUPERSEDED for the reported group test. `fit_subject` below runs a single-start
Nelder-Mead from the pooled x0 with no restarts, and it stalls in a local minimum for
10 of the 64 subjects (DD 1, 7, 8, 25, 28; contr 34, 49, 51, 52, 61). It therefore
reports n_width DD 8.12 vs contr 8.99, Mann-Whitney p = 0.005, r_rb = -0.41. Refitting
the *same model and objective* at the global optimum gives 8.14 vs 8.85, p = 0.018,
r_rb = -0.35, which is the number quoted in SM5. Use `scripts/decarli_locus_beta.py`
(model A: exhaustive grid over sigma_c + exact profiling of lapse, t0, rt_scale), which
fits better in 55/64 subjects and reproduces this script wherever the NM did converge.

    .venv/bin/python -m scripts.decarli_gaussian_freebeta_subjects
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
from scipy.stats import mannwhitneyu

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.decarli_gaussian_freebeta import (
    SPEC, DATA, OUT_DIR, NVALS, GROUPS, n_width, sigma_well, evidence_sd,
)

# pooled mechanism params [sigma_c, s0, lambda0, beta, lapse, t0, rt_scale]
POOL_BOUNDS = [(0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0),
               (1e-4, 0.5), (0.05, 1.0), (0.0, 5.0)]
# per-subject free: sigma_c, lapse, t0, rt_scale  (beta, s0, lambda0 held common)
SUBJ_BOUNDS = [(0.001, 0.5), (1e-4, 0.5), (0.05, 1.5), (0.0, 6.0)]


def subj_targets(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    acc, rt = [], []
    for n in NVALS:
        s = g[g["Numerosity"] == n]
        acc.append(float(s["Accuracy"].mean()))
        rt.append(float(s["RTs"].median()))
    return np.array(acc), np.array(rt)


def predict_full(vec7: np.ndarray, quad: T.Quadrature) -> tuple[np.ndarray, np.ndarray]:
    theta = dict(zip(("sigma_c", "s0", "lambda0", "pow"), vec7[:4]))
    lapse, t0, rt_scale = vec7[4], vec7[5], vec7[6]
    grid = T.compute_grid(SPEC, theta, quad, n_values=NVALS.astype(float))
    p = np.array([grid["p_resp"][i, int(n) - 1] for i, n in enumerate(NVALS)])
    p = (1 - lapse) * p + lapse / T.N_WELLS
    return p, t0 + rt_scale * grid["EH"]


def negll_full(vec: np.ndarray, acc, rt, quad) -> float:
    try:
        p, prt = predict_full(vec, quad)
    except (FloatingPointError, ValueError):
        return 1e18
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(prt))):
        return 1e18
    ma, mr = np.mean((p - acc) ** 2), np.mean((prt - rt) ** 2)
    if ma <= 0 or mr <= 0:
        return 1e18
    return 0.5 * len(acc) * (np.log(ma) + np.log(mr))


def pooled_fit(df: pd.DataFrame, quad: T.Quadrature) -> np.ndarray:
    acc, rt = [], []
    for n in NVALS:
        s = df[df["Numerosity"] == n]
        acc.append(float(s["Accuracy"].mean())); rt.append(float(s["RTs"].median()))
    acc, rt = np.array(acc), np.array(rt)
    de = differential_evolution(negll_full, POOL_BOUNDS, args=(acc, rt, quad),
                                maxiter=100, popsize=15, tol=1e-7, seed=3,
                                init="latinhypercube", polish=False,
                                updating="deferred", workers=1)
    pol = minimize(negll_full, de.x, args=(acc, rt, quad), method="Nelder-Mead",
                   bounds=POOL_BOUNDS, options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6})
    return (pol.x if pol.fun <= de.fun else de.x)


def fit_subject(acc, rt, common: dict, quad, x0) -> dict:
    """Fit sigma_c, lapse, t0, rt_scale with pow, s0, lambda0 held at common."""
    def negll(v):
        full = np.array([v[0], common["s0"], common["lambda0"], common["pow"],
                         v[1], v[2], v[3]])
        return negll_full(full, acc, rt, quad)
    pol = minimize(negll, x0, method="Nelder-Mead", bounds=SUBJ_BOUNDS,
                   options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-7})
    v = pol.x
    theta = {"sigma_c": v[0], "s0": common["s0"], "lambda0": common["lambda0"],
             "pow": common["pow"]}
    return {"sigma_c": v[0], "lapse": v[1], "t0": v[2], "rt_scale": v[3],
            "n_width_sigma1": n_width(theta, 1.0), "neg_ll": float(pol.fun)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)

    pooled = pooled_fit(df, quad)
    common = {"s0": pooled[1], "lambda0": pooled[2], "pow": pooled[3]}
    x0 = np.array([pooled[0], pooled[4], pooled[5], pooled[6]])
    print(f"Pooled mechanism: pow={common['pow']:.2f} s0={common['s0']:.4g} "
          f"lambda0={common['lambda0']:.2f} (sigma_c per subject)", flush=True)

    rows = []
    for grp in GROUPS:
        subs = sorted(df[df["Group"] == grp]["Sub"].unique())
        for s in subs:
            g = df[(df["Group"] == grp) & (df["Sub"] == s)]
            acc, rt = subj_targets(g)
            if not (np.all(np.isfinite(acc)) and np.all(np.isfinite(rt))):
                continue
            f = fit_subject(acc, rt, common, quad, x0)
            rows.append({"group": grp, "subject": int(s), **f})
        print(f"  fitted {len(subs)} {grp} subjects", flush=True)

    sub = pd.DataFrame(rows)
    sub.to_csv(OUT_DIR / "gaussian_freebeta_subjects.csv", index=False)

    dd = sub[sub.group == "DD"]["n_width_sigma1"].to_numpy()
    ct = sub[sub.group == "contr"]["n_width_sigma1"].to_numpy()
    dd, ct = dd[np.isfinite(dd)], ct[np.isfinite(ct)]
    U, p = mannwhitneyu(dd, ct, alternative="two-sided")
    print("\n=== per-subject n_width(σ=1): DD vs control ===")
    print(f"  DD:      median {np.median(dd):.2f}  IQR [{np.percentile(dd,25):.2f}, "
          f"{np.percentile(dd,75):.2f}]  (n={dd.size})")
    print(f"  contr:   median {np.median(ct):.2f}  IQR [{np.percentile(ct,25):.2f}, "
          f"{np.percentile(ct,75):.2f}]  (n={ct.size})")
    print(f"  Mann-Whitney U={U:.0f}, p={p:.3f}  (median diff DD-contr="
          f"{np.median(dd)-np.median(ct):+.2f})")

    # also compare sigma_c
    dds, cts = sub[sub.group=="DD"]["sigma_c"].to_numpy(), sub[sub.group=="contr"]["sigma_c"].to_numpy()
    Us, ps = mannwhitneyu(dds, cts, alternative="two-sided")
    print(f"  sigma_c: DD median {np.median(dds):.4f} vs contr {np.median(cts):.4f}, "
          f"Mann-Whitney p={ps:.3f}")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    data = [ct, dd]
    parts = ax.violinplot(data, positions=[0, 1], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], ["#2c7fb8", "#c0392b"]):
        pc.set_facecolor(c); pc.set_alpha(0.4)
    for i, (arr, c) in enumerate(zip([ct, dd], ["#2c7fb8", "#c0392b"])):
        ax.scatter(np.full(arr.size, i) + np.random.default_rng(0).normal(0, 0.04, arr.size),
                   arr, s=14, color=c, alpha=0.7, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["control", "DD"])
    ax.set_ylabel("Per-subject n_width (σ_n = 1)")
    ax.set_title(f"De Carli individual n_width (winner model)\nMann-Whitney p={p:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_freebeta_subjects.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved gaussian_freebeta_subjects.csv, .png in {OUT_DIR}")


if __name__ == "__main__":
    main()
