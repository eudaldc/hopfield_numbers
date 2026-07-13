"""Where does the De Carli DD difference live: well width (n_width) or sensory noise?

The per-subject n_width analysis forced all between-subject variation into the
well-width scale sigma_c (holding the noise common), and found a small but
DD difference (n_width 8.2 vs 8.8). But accuracy alone cannot separate
sensory NOISE from WELL WIDTH — both spread the read-out. This script runs the
**mirror** fit (hold the well width common, let the per-subject sensory noise s0
vary) and asks whether the group difference re-appears there too. If it does, the
deficit is real but its *locus* (noise vs resolution) is not identifiable from these
data. Also reports the model-agnostic raw effect size of the deficit.

    .venv/bin/python -m scripts.decarli_locus
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.decarli_gaussian_freebeta import DATA, NVALS, GROUPS, evidence_sd
from scripts.decarli_gaussian_freebeta_subjects import (
    SPEC, subj_targets, negll_full, pooled_fit,
)

# per-subject free: s0 (noise), lapse, t0, rt_scale  (sigma_c, beta, lambda0 common)
MIRROR_BOUNDS = [(1e-6, 0.5), (1e-4, 0.5), (0.05, 1.5), (0.0, 6.0)]


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation for the Mann-Whitney comparison of a vs b.

    Positive means a tends to rank above b; ties handled via scipy's U (which
    splits them at 0.5). This is the effect size that matches the MW test.
    """
    u = mannwhitneyu(a, b, alternative="two-sided")[0]
    return float(2.0 * u / (a.size * b.size) - 1.0)


def fit_noise_subject(acc, rt, common, quad, x0) -> dict:
    """Free s0 (noise); hold sigma_c, pow, lambda0 at common values."""
    def negll(v):
        full = np.array([common["sigma_c"], v[0], common["lambda0"],
                         common["pow"], v[1], v[2], v[3]])
        return negll_full(full, acc, rt, quad)
    pol = minimize(negll, x0, method="Nelder-Mead", bounds=MIRROR_BOUNDS,
                   options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-7})
    v = pol.x
    theta = {"s0": v[0], "lambda0": common["lambda0"]}
    return {"s0": v[0], "sigma_n4": evidence_sd(theta, 4),
            "sigma_n5": evidence_sd(theta, 5), "lapse": v[1], "t0": v[2]}


def main() -> None:
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)
    pooled = pooled_fit(df, quad)
    common = {"sigma_c": pooled[0], "s0": pooled[1], "lambda0": pooled[2],
              "pow": pooled[3]}
    x0 = np.array([pooled[1], pooled[4], pooled[5], pooled[6]])
    print(f"Common well width: sigma_c={common['sigma_c']:.4f} pow={common['pow']:.2f} "
          f"(s0 per subject; lambda0={common['lambda0']:.2f})", flush=True)

    rows = []
    for grp in GROUPS:
        for s in sorted(df[df.Group == grp].Sub.unique()):
            g = df[(df.Group == grp) & (df.Sub == s)]
            acc, rt = subj_targets(g)
            f = fit_noise_subject(acc, rt, common, quad, x0)
            rows.append({"group": grp, "subject": int(s), **f})
    sub = pd.DataFrame(rows)
    sub.to_csv(_ROOT / "output" / "decarli_fit" / "gaussian_freebeta_locus.csv", index=False)

    # (1) mirror fit: does the group difference show up in the sensory noise too?
    print("\n=== MIRROR fit: hold width common, free per-subject noise ===")
    for q in ("sigma_n4", "sigma_n5", "s0"):
        dd = sub[sub.group == "DD"][q].to_numpy()
        ct = sub[sub.group == "contr"][q].to_numpy()
        U, p = mannwhitneyu(dd, ct, alternative="two-sided")
        print(f"  {q}: DD median {np.median(dd):.4f} vs contr {np.median(ct):.4f}  "
              f"MW p={p:.3f}")

    # (2) model-agnostic effect sizes of the raw deficit
    print("\n=== model-agnostic raw effect sizes (per subject) ===")
    def band_acc(g):
        d = df[df.Group == g]
        return d[d.Numerosity.isin([4, 5, 6])].groupby("Sub").Accuracy.mean().to_numpy()
    def band_rt(g):
        d = df[df.Group == g]
        return d[d.Numerosity.isin([4, 5, 6, 7])].groupby("Sub").RTs.median().to_numpy()
    dd_a, ct_a = band_acc("DD"), band_acc("contr")
    dd_r, ct_r = band_rt("DD"), band_rt("contr")
    print(f"  accuracy n=4-6: DD {dd_a.mean():.3f} vs contr {ct_a.mean():.3f}  "
          f"r_rb={rank_biserial(dd_a, ct_a):+.2f}")
    print(f"  median RT n=4-7: DD {dd_r.mean():.3f}s vs contr {ct_r.mean():.3f}s  "
          f"r_rb={rank_biserial(dd_r, ct_r):+.2f}")
    # n_width effect size for comparison (from the width-free fit)
    subj = pd.read_csv(_ROOT / "output" / "decarli_fit" / "gaussian_freebeta_subjects.csv")
    ncd = subj[subj.group == "DD"].n_width_sigma1.to_numpy()
    ncc = subj[subj.group == "contr"].n_width_sigma1.to_numpy()
    print(f"  n_width (width-free fit): DD {np.median(ncd):.2f} vs contr {np.median(ncc):.2f}  "
          f"r_rb={rank_biserial(ncd, ncc):+.2f}")
    print("\nSaved gaussian_freebeta_locus.csv")


if __name__ == "__main__":
    main()
