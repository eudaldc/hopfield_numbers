"""Is the De Carli DD deficit in the well-width SCALE (sigma_c) or the EXPONENT (beta)?

The per-subject fit (`decarli_gaussian_freebeta_subjects.py`) forces all between-subject
variation into the width scale sigma_c, holding the exponent beta common. `decarli_locus.py`
runs the noise mirror (free s0, landscape common). This script runs the **third mirror**:
hold sigma_c common and let the per-subject width EXPONENT beta vary.

Motivation: for power-law wells sigma_n = sigma_c * n^beta, the resolvability limit is

    n_width = sigma_c ** (-1 / beta)

so sigma_c and beta are two routes to the same n_width and must trade off. In the published
*group* fit (`gaussian_freebeta_fitted.csv`) the two groups have essentially the same
sigma_c (0.0440 DD vs 0.0442 contr) and differ in beta (1.515 vs 1.470): at the group level
the whole n_width gap is carried by the exponent, not the scale. This script asks whether
that also holds per subject, and whether a free-beta account fits the individual curves as
well as the free-sigma_c one.

Estimator. All three restricted models have the same free parameters per subject: one
landscape parameter (sigma_c / beta / s0) plus lapse, t0, rt_scale. Given the landscape
parameters the nuisance parameters enter *linearly* --

    p_exact = p + lapse * (1/N_WELLS - p)        (accuracy channel)
    rt      = t0 + rt_scale * EH                 (RT channel)

-- and the profiled objective 0.5*n*(log mse_acc + log mse_rt) is separable in the two
channels, so lapse and (t0, rt_scale) are solved exactly by box-constrained least squares
rather than searched. That leaves a 1-D problem in the landscape parameter, and since the
model curves (p, EH) do not depend on the subject, they are precomputed once on a dense
grid and every subject is then pure linear algebra. This is the same precompute-and-match
strategy already used in `decarli_locus_clean.py`, and it makes the landscape search
exhaustive/global instead of a local Nelder-Mead descent.

Because n, k and the objective are identical across the three models, their profiled
neg_ll are directly comparable (neg_ll differences ARE AIC differences).

Requires the cached pooled landscape:
    .venv/bin/python -m scripts.decarli_pooled_cache
    .venv/bin/python -m scripts.decarli_locus_beta
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.stats import mannwhitneyu, wilcoxon

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.decarli_gaussian_freebeta import SPEC, DATA, OUT_DIR, NVALS, GROUPS, evidence_sd
from scripts.decarli_gaussian_freebeta_subjects import subj_targets
from scripts.decarli_locus import rank_biserial
from scripts.decarli_pooled_cache import load_pooled

NG = 800                       # landscape-grid resolution for the exhaustive 1-D search
LAPSE_BOUNDS = (1e-4, 0.5)     # == SUBJ_BOUNDS / MIRROR_BOUNDS in the sibling scripts
RT_LO, RT_HI = (0.05, 0.0), (1.5, 6.0)      # (t0, rt_scale)

# free landscape parameter -> grid over its published bounds
GRIDS = {
    "sigma_c": np.geomspace(0.001, 0.5, NG),
    "beta":    np.geomspace(0.1, 6.0, NG),
    "s0":      np.geomspace(1e-6, 0.5, NG),
}


def n_width_closed(sigma_c: float, beta: float) -> float:
    """Continuous width index sigma_n = 1: n_width = sigma_c^(-1/beta).

    Exact inverse of `decarli_gaussian_freebeta.n_width`, which grid-searches and caps at
    n = 60 (returning NaN beyond). With beta free per subject some subjects legitimately
    land above that cap, and censoring them would bias the group medians.
    """
    return float(np.power(float(sigma_c), -1.0 / float(beta)))


def model_curves(theta: dict, quad: T.Quadrature) -> tuple[np.ndarray, np.ndarray]:
    """Clean (pre-lapse) exact-response probability and expected entropy per numerosity."""
    grid = T.compute_grid(SPEC, theta, quad, n_values=NVALS.astype(float))
    p = np.array([grid["p_resp"][i, int(n) - 1] for i, n in enumerate(NVALS)])
    return p, grid["EH"]


SLOT = {"sigma_c": "sigma_c", "beta": "pow", "s0": "s0"}


def theta_for(which: str, v: float, pooled: np.ndarray) -> dict:
    sc0, s00, lam0, beta0 = pooled[:4]
    theta = {"sigma_c": sc0, "s0": s00, "lambda0": lam0, "pow": beta0}
    theta[SLOT[which]] = float(v)
    return theta


def build_curves(which: str, pooled: np.ndarray, quad: T.Quadrature) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (p, EH) along the 1-D grid of the free landscape parameter."""
    ps, ehs = [], []
    for v in GRIDS[which]:
        p, eh = model_curves(theta_for(which, v, pooled), quad)
        ps.append(p); ehs.append(eh)
    return np.array(ps), np.array(ehs)


def profile_subject(p: np.ndarray, eh: np.ndarray, acc: np.ndarray, rt: np.ndarray):
    """Exact box-constrained LSQ for (lapse) and (t0, rt_scale); return neg_ll + params.

    Accuracy: pred = p + lapse * (1/W - p). Quadratic in lapse -> closed form + clip.
    RT:       pred = t0 + rt_scale * EH.     2-D box-constrained LSQ via lsq_linear.
    """
    w = 1.0 / T.N_WELLS
    g = w - p                                   # d pred / d lapse
    r0 = p - acc
    denom = float(g @ g)
    lapse = -float(r0 @ g) / denom if denom > 0 else LAPSE_BOUNDS[0]
    lapse = float(np.clip(lapse, *LAPSE_BOUNDS))
    resid_acc = r0 + lapse * g
    mse_acc = float(np.mean(resid_acc ** 2))

    A = np.column_stack([np.ones_like(eh), eh])
    sol = lsq_linear(A, rt, bounds=(RT_LO, RT_HI))
    t0, rt_scale = float(sol.x[0]), float(sol.x[1])
    mse_rt = float(np.mean((A @ sol.x - rt) ** 2))

    if mse_acc <= 0 or mse_rt <= 0:
        return np.inf, lapse, t0, rt_scale
    n = acc.size
    return 0.5 * n * (np.log(mse_acc) + np.log(mse_rt)), lapse, t0, rt_scale


def fit_subject(which: str, curves, acc: np.ndarray, rt: np.ndarray,
                pooled: np.ndarray, quad: T.Quadrature) -> dict:
    """Exhaustive grid over the landscape parameter, then a bounded local refine.

    The grid gives the global basin (no local-minimum risk); the refine removes the
    discretisation error so the estimate is not limited by grid step.
    """
    ps, ehs = curves
    g = GRIDS[which]
    nlls = np.array([profile_subject(ps[i], ehs[i], acc, rt)[0] for i in range(ps.shape[0])])
    i = int(np.argmin(nlls))
    val, nll = float(g[i]), float(nlls[i])

    lo, hi = g[max(0, i - 1)], g[min(g.size - 1, i + 1)]
    if hi > lo:
        def obj(v: float) -> float:
            p, eh = model_curves(theta_for(which, v, pooled), quad)
            return profile_subject(p, eh, acc, rt)[0]
        r = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                            options={"xatol": (hi - lo) * 1e-4, "maxiter": 15})
        if np.isfinite(r.fun) and r.fun < nll:
            val, nll = float(r.x), float(r.fun)

    p, eh = model_curves(theta_for(which, val, pooled), quad)
    nll, lapse, t0, rts = profile_subject(p, eh, acc, rt)
    return {"value": float(val), "neg_ll": float(nll), "lapse": lapse,
            "t0": t0, "rt_scale": rts}


def mw(sub: pd.DataFrame, col: str) -> tuple[float, float, float, float]:
    dd = sub[sub.group == "DD"][col].to_numpy()
    ct = sub[sub.group == "contr"][col].to_numpy()
    p = mannwhitneyu(dd, ct, alternative="two-sided")[1]
    return float(np.median(dd)), float(np.median(ct)), float(p), rank_biserial(dd, ct)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)
    pooled = load_pooled()
    sc0, s00, lam0, beta0 = pooled[:4]
    print(f"Pooled landscape: sigma_c={sc0:.6f} s0={s00:.4g} lambda0={lam0:.6f} beta={beta0:.6f}")
    print(f"  pooled n_width = {n_width_closed(sc0, beta0):.3f}\n", flush=True)

    print("Precomputing model curves along each 1-D landscape grid ...", flush=True)
    curves = {}
    for which in ("beta", "sigma_c", "s0"):
        curves[which] = build_curves(which, pooled, quad)
        print(f"  {which}: {NG} curves", flush=True)

    rows = []
    for grp in GROUPS:
        for s in sorted(df[df.Group == grp].Sub.unique()):
            g = df[(df.Group == grp) & (df.Sub == s)]
            acc, rt = subj_targets(g)
            fb = fit_subject("beta", curves["beta"], acc, rt, pooled, quad)
            fs = fit_subject("sigma_c", curves["sigma_c"], acc, rt, pooled, quad)
            fn = fit_subject("s0", curves["s0"], acc, rt, pooled, quad)
            rows.append({
                "group": grp, "subject": int(s),
                # C: free exponent (this script's question)
                "beta_C": fb["value"], "n_width_C": n_width_closed(sc0, fb["value"]),
                "well_sigma_n7_C": sc0 * 7.0 ** fb["value"],
                "lapse_C": fb["lapse"], "neg_ll_C": fb["neg_ll"],
                # A: free scale (published per-subject analysis)
                "sigma_c_A": fs["value"], "n_width_A": n_width_closed(fs["value"], beta0),
                "neg_ll_A": fs["neg_ll"],
                # B: free sensory noise (decarli_locus mirror)
                "s0_B": fn["value"],
                "sigma_n5_B": evidence_sd({"s0": fn["value"], "lambda0": lam0}, 5),
                "neg_ll_B": fn["neg_ll"],
            })
        print(f"  fitted {grp}", flush=True)

    sub = pd.DataFrame(rows)
    sub.to_csv(OUT_DIR / "gaussian_freebeta_locus_beta.csv", index=False)

    # ---- boundary diagnostics: a parameter pinned at its bound is not identified ----
    print("\n=== boundary diagnostics ===")
    for col, which in (("beta_C", "beta"), ("sigma_c_A", "sigma_c"), ("s0_B", "s0")):
        lo, hi = GRIDS[which][0], GRIDS[which][-1]
        n_lo = int(np.sum(sub[col] <= lo * (1 + 1e-4)))
        n_hi = int(np.sum(sub[col] >= hi * (1 - 1e-4)))
        flag = "  !" if (n_lo or n_hi) else "   "
        print(f"{flag} {col}: {n_lo} at lower bound, {n_hi} at upper bound")

    print("\n=== C: FREE EXPONENT beta (sigma_c held common) ===")
    for col, lab in (("beta_C", "beta"), ("n_width_C", "n_width"),
                     ("well_sigma_n7_C", "well sigma(n=7)")):
        md, mc, p, r = mw(sub, col)
        print(f"  {lab:>16}: DD {md:.3f}  contr {mc:.3f}   MW p={p:.4f}  r_rb={r:+.2f}")

    print("\n=== A: free scale sigma_c (beta held common) ===")
    for col, lab in (("sigma_c_A", "sigma_c"), ("n_width_A", "n_width")):
        md, mc, p, r = mw(sub, col)
        print(f"  {lab:>16}: DD {md:.4f}  contr {mc:.4f}   MW p={p:.4f}  r_rb={r:+.2f}")

    print("\n=== B: free sensory noise s0 (landscape held common) ===")
    for col, lab in (("s0_B", "s0"), ("sigma_n5_B", "sigma(n=5)")):
        md, mc, p, r = mw(sub, col)
        print(f"  {lab:>16}: DD {md:.4f}  contr {mc:.4f}   MW p={p:.4f}  r_rb={r:+.2f}")

    print("\n=== which single free parameter fits the individual curves better? ===")
    print("    (same n, same k -> neg_ll differences are AIC differences)")
    a, b, c = (sub["neg_ll_A"].to_numpy(), sub["neg_ll_B"].to_numpy(),
               sub["neg_ll_C"].to_numpy())
    print(f"  mean neg_ll: sigma_c {a.mean():.3f} | s0 {b.mean():.3f} | beta {c.mean():.3f}")
    for x, y, nx, ny in ((c, a, "beta", "sigma_c"), (c, b, "beta", "s0"),
                         (a, b, "sigma_c", "s0")):
        wp = wilcoxon(x, y).pvalue
        print(f"  {nx} better than {ny} in {int(np.sum(x < y))}/{len(x)} subjects "
              f"(Wilcoxon p={wp:.4g}, mean dnegll={np.mean(x - y):+.3f})")

    r_nc = np.corrcoef(sub["n_width_A"], sub["n_width_C"])[0, 1]
    print(f"\n  n_width(free sigma_c) vs n_width(free beta): Pearson r={r_nc:.3f}")

    # ---- validation against the published per-subject fit (single-start Nelder-Mead) ----
    # Model A here is the SAME model and objective as `decarli_gaussian_freebeta_subjects.py`,
    # so its neg_ll must be <= the published one up to refinement tolerance. Any subject where
    # the published fit is clearly worse is a subject where that single-start NM stalled in a
    # local minimum.
    pub = OUT_DIR / "gaussian_freebeta_subjects.csv"
    if pub.exists():
        old = pd.read_csv(pub).rename(columns={"n_width_sigma1": "n_width_pub",
                                               "neg_ll": "neg_ll_pub"})
        m = sub.merge(old[["group", "subject", "n_width_pub", "neg_ll_pub"]],
                      on=["group", "subject"])
        d = m["neg_ll_A"] - m["neg_ll_pub"]          # negative = this fit is better
        stalled = m.loc[d < -0.5]
        print("\n=== validation vs published single-start Nelder-Mead (model A) ===")
        print(f"  this fit better in {int((d < -1e-6).sum())}/{len(m)} subjects; "
              f"published better in {int((d > 1e-6).sum())} (max by {max(0.0, d.max()):.4f})")
        print(f"  published NM stalled (dnegll < -0.5) in {len(stalled)} subjects: "
              f"{[f'{r.group}{r.subject}' for r in stalled.itertuples()]}")
        r = np.corrcoef(m["n_width_A"], m["n_width_pub"])[0, 1]
        print(f"  n_width_A vs published: r={r:.4f}, median abs diff="
              f"{np.median(np.abs(m['n_width_A'] - m['n_width_pub'])):.3f}")
        dd, ct = m[m.group == "DD"], m[m.group == "contr"]
        p_pub = mannwhitneyu(dd["n_width_pub"], ct["n_width_pub"], alternative="two-sided")[1]
        p_new = mannwhitneyu(dd["n_width_A"], ct["n_width_A"], alternative="two-sided")[1]
        print(f"  n_width group test: published p={p_pub:.4f}  ->  global-optimum p={p_new:.4f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.2))
    colors = {"DD": "#c0392b", "contr": "#2c7fb8"}
    rng = np.random.default_rng(0)
    for k, (col, lab) in enumerate((("beta_C", "per-subject beta (free exponent)"),
                                    ("n_width_C", "per-subject n_width (free exponent)"))):
        dd = sub[sub.group == "DD"][col].to_numpy()
        ct = sub[sub.group == "contr"][col].to_numpy()
        parts = ax[k].violinplot([ct, dd], positions=[0, 1], showmedians=True, widths=0.8)
        for pc, cc in zip(parts["bodies"], ["#2c7fb8", "#c0392b"]):
            pc.set_facecolor(cc); pc.set_alpha(0.4)
        for i, (arr, cc) in enumerate(zip([ct, dd], ["#2c7fb8", "#c0392b"])):
            ax[k].scatter(np.full(arr.size, i) + rng.normal(0, 0.04, arr.size), arr,
                          s=14, color=cc, alpha=0.7, zorder=3)
        ax[k].set_xticks([0, 1]); ax[k].set_xticklabels(["control", "DD"])
        ax[k].set_ylabel(lab)
        ax[k].set_title(f"MW p={mw(sub, col)[2]:.3f}")

    for grp in GROUPS:
        g = sub[sub.group == grp]
        ax[2].scatter(g["neg_ll_A"], g["neg_ll_C"], s=18, color=colors[grp], alpha=0.7, label=grp)
    lim = [min(a.min(), c.min()) - 0.5, max(a.max(), c.max()) + 0.5]
    ax[2].plot(lim, lim, "0.6", lw=1)
    ax[2].set_xlabel("neg_ll — free sigma_c"); ax[2].set_ylabel("neg_ll — free beta")
    ax[2].set_title("below diagonal = free beta fits better")
    ax[2].legend(frameon=False, fontsize=8)

    fig.suptitle("De Carli — third mirror: free width EXPONENT beta (sigma_c common)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_freebeta_locus_beta.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved gaussian_freebeta_locus_beta.{{csv,png}} in {OUT_DIR}")


if __name__ == "__main__":
    main()
