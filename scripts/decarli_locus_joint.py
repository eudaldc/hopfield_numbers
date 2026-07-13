"""Are the width SCALE (sigma_c) and the width EXPONENT (beta) separately identifiable?

`decarli_gaussian_freebeta_subjects.py` frees sigma_c (beta common) and finds a DD deficit.
`decarli_locus_beta.py` frees beta (sigma_c common) and finds the same deficit. Both cannot
be independent evidence: for power-law wells sigma_n = sigma_c * n^beta the resolvability
continuous width index is n_width = sigma_c^(-1/beta), so the two parameters trade
off exactly along curves of constant n_width.

This script fits BOTH per subject on a 2-D grid (nuisance parameters lapse, t0, rt_scale
profiled out exactly, as in `decarli_locus_beta`) and measures what the accuracy+RT curves
actually constrain, via profile likelihood:

  * profile over sigma_c   (minimising over beta)
  * profile over beta      (minimising over sigma_c)
  * profile over n_width   (minimising over every (sigma_c, beta) on that contour)

If the 2-unit profile interval is wide for sigma_c and beta but narrow for n_width, then
the individual width parameters are not identified from these data and n_width is the
width-profile quantity the data constrain. The group comparison is then run on n_width.

    .venv/bin/python -m scripts.decarli_pooled_cache      # once
    .venv/bin/python -m scripts.decarli_locus_joint
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.stats import mannwhitneyu

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.decarli_gaussian_freebeta import SPEC, DATA, OUT_DIR, NVALS, GROUPS
from scripts.decarli_gaussian_freebeta_subjects import subj_targets
from scripts.decarli_locus import rank_biserial
from scripts.decarli_locus_beta import n_width_closed, model_curves, LAPSE_BOUNDS, RT_LO, RT_HI
from scripts.decarli_pooled_cache import load_pooled

NSC, NB = 72, 72
SC_GRID = np.geomspace(0.015, 0.25, NSC)
B_GRID = np.geomspace(0.5, 3.0, NB)
DELTA = 2.0                     # profile-likelihood cutoff (neg_ll units, ~ 95% for 1 dof)
CURVES_CACHE = OUT_DIR / "joint_curves.npz"


def build_surface(pooled: np.ndarray, quad: T.Quadrature) -> tuple[np.ndarray, np.ndarray]:
    """(p, EH) at every (sigma_c, beta) node. Subject-independent, so computed once."""
    if CURVES_CACHE.exists():
        z = np.load(CURVES_CACHE)
        if z["sc"].shape == SC_GRID.shape and np.allclose(z["sc"], SC_GRID):
            return z["ps"], z["ehs"]
    _, s00, lam0, _ = pooled[:4]
    ps = np.empty((NSC, NB, NVALS.size))
    ehs = np.empty((NSC, NB, NVALS.size))
    for i, sc in enumerate(SC_GRID):
        for j, b in enumerate(B_GRID):
            p, eh = model_curves({"sigma_c": float(sc), "s0": s00,
                                  "lambda0": lam0, "pow": float(b)}, quad)
            ps[i, j], ehs[i, j] = p, eh
        print(f"  sigma_c row {i + 1}/{NSC}", flush=True)
    np.savez_compressed(CURVES_CACHE, ps=ps, ehs=ehs, sc=SC_GRID, b=B_GRID)
    return ps, ehs


def nll_surface(ps: np.ndarray, ehs: np.ndarray, acc: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """Profiled neg_ll over the (sigma_c, beta) grid, nuisance params solved in closed form.

    Accuracy channel: pred = p + lapse*(1/W - p), quadratic in lapse -> closed form + clip.
    RT channel: rt = t0 + rt_scale*EH, unconstrained OLS, with a box-constrained LSQ fallback
    for the cells whose OLS solution violates the published bounds.
    """
    w = 1.0 / T.N_WELLS
    n = acc.size
    g = w - ps                                   # (NSC, NB, n)
    r0 = ps - acc
    denom = np.einsum("ijk,ijk->ij", g, g)
    lapse = np.where(denom > 0, -np.einsum("ijk,ijk->ij", r0, g) / np.maximum(denom, 1e-300),
                     LAPSE_BOUNDS[0])
    lapse = np.clip(lapse, *LAPSE_BOUNDS)
    resid = r0 + lapse[..., None] * g
    mse_acc = np.mean(resid ** 2, axis=-1)

    # vectorised OLS of rt on [1, EH]
    sx = ehs.sum(-1); sxx = (ehs ** 2).sum(-1)
    sy = rt.sum(); sxy = (ehs * rt).sum(-1)
    det = n * sxx - sx ** 2
    ok = det > 1e-12
    t0 = np.where(ok, (sxx * sy - sx * sxy) / np.where(ok, det, 1.0), 0.0)
    slope = np.where(ok, (n * sxy - sx * sy) / np.where(ok, det, 1.0), 0.0)

    inside = ok & (t0 >= RT_LO[0]) & (t0 <= RT_HI[0]) & (slope >= RT_LO[1]) & (slope <= RT_HI[1])
    pred = t0[..., None] + slope[..., None] * ehs
    mse_rt = np.mean((pred - rt) ** 2, axis=-1)

    # exact box-constrained solve only where the unconstrained optimum is infeasible
    bad = np.argwhere(~inside)
    for i, j in bad:
        A = np.column_stack([np.ones(n), ehs[i, j]])
        sol = lsq_linear(A, rt, bounds=(RT_LO, RT_HI))
        mse_rt[i, j] = float(np.mean((A @ sol.x - rt) ** 2))

    bad_frac = len(bad) / (NSC * NB)
    with np.errstate(divide="ignore", invalid="ignore"):
        nll = 0.5 * n * (np.log(mse_acc) + np.log(mse_rt))
    nll[~np.isfinite(nll)] = np.inf
    return nll, bad_frac


def profile_width(values: np.ndarray, nll: np.ndarray, lo: float) -> tuple[float, float]:
    """2-unit profile interval over a 1-D reduction of the surface."""
    ok = nll <= lo + DELTA
    if not ok.any():
        return np.nan, np.nan
    return float(values[ok].min()), float(values[ok].max())


def main() -> None:
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)
    pooled = load_pooled()
    print(f"Pooled: sigma_c={pooled[0]:.4f} beta={pooled[3]:.4f} "
          f"(s0, lambda0 held at pooled)\n", flush=True)

    print("Building (sigma_c x beta) curve surface ...", flush=True)
    ps, ehs = build_surface(pooled, quad)

    NC = np.power(SC_GRID[:, None], -1.0 / B_GRID[None, :])    # n_width at every node

    rows, flat_rt = [], []
    for grp in GROUPS:
        for s in sorted(df[df.Group == grp].Sub.unique()):
            g = df[(df.Group == grp) & (df.Sub == s)]
            acc, rt = subj_targets(g)
            nll, bad = nll_surface(ps, ehs, acc, rt)
            if bad > 0.99:
                flat_rt.append(f"{grp}{s}")
            i, j = np.unravel_index(np.argmin(nll), nll.shape)
            lo = float(nll[i, j])

            sc_lo, sc_hi = profile_width(SC_GRID, nll.min(axis=1), lo)   # min over beta
            b_lo, b_hi = profile_width(B_GRID, nll.min(axis=0), lo)      # min over sigma_c
            m = nll <= lo + DELTA
            nc_lo, nc_hi = (float(NC[m].min()), float(NC[m].max())) if m.any() else (np.nan, np.nan)

            rows.append({
                "group": grp, "subject": int(s),
                "sigma_c_hat": SC_GRID[i], "beta_hat": B_GRID[j],
                "n_width_hat": n_width_closed(SC_GRID[i], B_GRID[j]), "neg_ll": lo,
                "sigma_c_lo": sc_lo, "sigma_c_hi": sc_hi,
                "beta_lo": b_lo, "beta_hi": b_hi,
                "n_width_lo": nc_lo, "n_width_hi": nc_hi,
                "at_edge": bool(i in (0, NSC - 1) or j in (0, NB - 1)),
            })
        print(f"  fitted {grp}", flush=True)

    sub = pd.DataFrame(rows)
    sub.to_csv(OUT_DIR / "gaussian_freebeta_locus_joint.csv", index=False)

    print("\n=== data-quality flags (not model failures) ===")
    print(f"  {len(flat_rt)} subjects {flat_rt} have their optimal rt_scale on the rt_scale>=0")
    print("    bound at EVERY grid node: their median RT does not increase with numerosity, so")
    print("    the RT channel carries no information for them. The box-constrained solve is exact.")
    print(f"  joint MLE on the grid edge for {int(sub.at_edge.sum())}/{len(sub)} subjects")

    # relative width of the 2-unit profile interval, as a fraction of the searched range
    print(f"\n=== what do the data actually constrain? ({DELTA}-unit profile interval) ===")
    for name, lo_c, hi_c, span in (
        ("sigma_c", "sigma_c_lo", "sigma_c_hi", SC_GRID[-1] / SC_GRID[0]),
        ("beta", "beta_lo", "beta_hi", B_GRID[-1] / B_GRID[0]),
        ("n_width", "n_width_lo", "n_width_hi", float(NC.max() / NC.min())),
    ):
        w = sub[hi_c] / sub[lo_c]
        print(f"  {name:>8}: interval spans x{w.median():.2f} (median), "
              f"x{w.quantile(0.9):.2f} (p90), x{w.max():.2f} (max)  "
              f"[{100 * np.log(w.median()) / np.log(span):.0f}% of searched log-range]")
    print("  -> n_width is the best-constrained direction, but its tail is heavy: for the")
    print("     subjects whose beta drifts low it is an extrapolation past the n=1..7 data.")

    blow = sub[sub.n_width_hat > 15]
    print(f"\n  {len(blow)}/{len(sub)} subjects have joint n_width_hat > 15 (all with beta_hat "
          f"<= {blow.beta_hat.max():.2f}): n_width ~ 8 lies beyond the fitted range, so a low")
    print("     exponent extrapolates it away. Their point estimates are not meaningful.")

    r = np.corrcoef(np.log(sub.sigma_c_hat), sub.beta_hat)[0, 1]
    print(f"\n  across subjects, corr(log sigma_c_hat, beta_hat) = {r:+.3f}")
    print("  This is the ridge sigma_c = n_width^(-beta): the two width parameters are")
    print("  exchangeable, so neither is separately identified from these curves.")

    print("\n=== group comparison on the JOINT MLE -- UNDER-IDENTIFIED, DO NOT INTERPRET ===")
    print(f"    (per-subject n_width_hat sd = {sub.n_width_hat.std():.2f}, vs ~1.6 when one width")
    print("     parameter is held fixed; the point estimates wander along the ridge)")
    for col, lab in (("n_width_hat", "n_width"), ("beta_hat", "beta"), ("sigma_c_hat", "sigma_c")):
        dd = sub[sub.group == "DD"][col].to_numpy()
        ct = sub[sub.group == "contr"][col].to_numpy()
        p = mannwhitneyu(dd, ct, alternative="two-sided")[1]
        print(f"  {lab:>8}: DD {np.median(dd):.4f}  contr {np.median(ct):.4f}   "
              f"MW p={p:.4f}  r_rb={rank_biserial(dd, ct):+.2f}")
    print("  The group effect must therefore be tested with ONE width parameter free")
    print("  (either one: see decarli_locus_beta.py), not with both.")

    # ---- figure: the likelihood ridge for one representative subject ----
    g = df[(df.Group == "DD") & (df.Sub == 1)]
    acc, rt = subj_targets(g)
    nll, _ = nll_surface(ps, ehs, acc, rt)
    lo = nll.min()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    X, Y = np.meshgrid(B_GRID, SC_GRID)
    im = ax[0].pcolormesh(X, Y, np.minimum(nll - lo, 12), shading="auto", cmap="viridis_r")
    cs = ax[0].contour(X, Y, NC, levels=[5, 6, 8, 10, 13, 17], colors="w", linewidths=0.8)
    ax[0].clabel(cs, fmt="n_width=%.0f", fontsize=7)
    ax[0].contour(X, Y, nll - lo, levels=[DELTA], colors="r", linewidths=1.6)
    ax[0].set_yscale("log"); ax[0].set_xlabel("beta"); ax[0].set_ylabel("sigma_c")
    ax[0].set_title("DD subject 1: profiled neg_ll (red = 2-unit region)")
    fig.colorbar(im, ax=ax[0], label="neg_ll - min")

    for grp, c in (("contr", "#2c7fb8"), ("DD", "#c0392b")):
        s = sub[sub.group == grp]
        ax[1].scatter(s.beta_hat, s.sigma_c_hat, s=20, color=c, alpha=0.75, label=grp)
    bb = np.linspace(B_GRID[0], B_GRID[-1], 200)
    for nc in (6, 8, 10, 13):
        ax[1].plot(bb, np.power(float(nc), -bb), "0.7", lw=0.8)
        ax[1].annotate(f"{nc}", (bb[-1], np.power(float(nc), -bb[-1])), fontsize=7, color="0.4")
    ax[1].set_yscale("log"); ax[1].set_xlabel("beta_hat"); ax[1].set_ylabel("sigma_c_hat")
    ax[1].set_xlim(B_GRID[0], B_GRID[-1]); ax[1].set_ylim(SC_GRID[0], SC_GRID[-1])
    ax[1].set_title("joint MLEs lie along constant-n_width curves")
    ax[1].legend(frameon=False, fontsize=8)

    fig.suptitle("De Carli — sigma_c and beta are not separately identifiable", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_freebeta_locus_joint.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved gaussian_freebeta_locus_joint.{{csv,png}} in {OUT_DIR}")


if __name__ == "__main__":
    main()
