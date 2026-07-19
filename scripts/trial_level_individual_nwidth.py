"""Per-subject width index vs numerical acuity (individual differences).

Does the fitted landscape width index track a subject's numerical acuity? We fit
a per-subject ``n_width`` on each participant's exact-accuracy curve and correlate
it with their Weber fraction (estimation-channel acuity). This is the individual-
differences replacement for the (removed) "n_width sharpens with exposure" claim.

Method mirrors the De Carli SM5 locus test (``scripts/decarli_locus_clean.py``):
with the width-law exponent held at the group joint value and sensory noise off
(WIDTH-only), the accuracy fall-off is carried by broader, more overlapping wells,
so a 1-D grid over the well-scale sigma_c matched to each subject's accuracy curve
yields ``n_width = sigma_c^(-1/p)``. A NOISE-only mirror (width fixed, sensory
noise free) provides the within-subject locus check.

We restrict to the long exposures (320 + 640 ms), where the fitted sensory noise
sigma(n=5) < 1 so resolution is landscape-limited rather than noise-limited (at
40 ms sigma(n=5) ~ 1.5, noise-dominated). The correlation is robust to this cut
(see the header of ``figure2_model_comparison`` / SM): r stays ~-0.5 for every
subset from all-five to 320+640.

    .venv/bin/python -m scripts.trial_level_individual_nwidth

Output: ``output/trial_level_model_comparison/individual_nwidth_acuity.csv``
(one row per subject) for Figure 2 panel c.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

SPEC = T.MODEL_BY_KEY[T.WINNER_KEY]
DURS = (320, 640)          # long exposures: sigma(n=5) < 1, landscape-limited
NVALS = np.arange(1, 16)
MIN_TRIALS = 60
WEBER_LO, WEBER_HI = 7, 14  # approximate range for the Weber CoV plateau
S0_FLOOR = 1e-6             # sensory noise "off"
SIGMAC_OFF = 0.003         # well width "off" (wells never merge in range)
NG = 200                   # 1-D grid resolution
OUT_CSV = T.OUT_DIR / "individual_nwidth_acuity.csv"


def acc_curve(sigma_c, s0, lambda0, pow_, lapse, quad):
    theta = {"sigma_c": sigma_c, "s0": s0, "lambda0": lambda0, "pow": pow_}
    grid = T.compute_grid(SPEC, theta, quad, n_values=NVALS.astype(float))
    p = np.array([grid["p_resp"][i, int(n) - 1] for i, n in enumerate(NVALS)])
    return (1.0 - lapse) * p + lapse / T.N_WELLS


def n_width_of(sigma_c, pow_):
    return float(sigma_c ** (-1.0 / pow_))


def subj_acc(g):
    return np.array([float((g[g.n_shown == n].n_guess == n).mean())
                     if (g.n_shown == n).any() else np.nan for n in NVALS])


def weber_fraction(g, min_per_n=4, min_cells=4):
    """Median CoV of estimates over the plateau n in [WEBER_LO, WEBER_HI]."""
    cvs = []
    for n in range(WEBER_LO, WEBER_HI + 1):
        e = g[g.n_shown == n].n_guess.to_numpy(float)
        if e.size >= min_per_n and e.mean() > 0:
            cvs.append(e.std() / e.mean())
    return float(np.median(cvs)) if len(cvs) >= min_cells else np.nan


def main() -> None:
    quad = T.make_quadrature(301, kind="grid")
    pow_ = float(pd.read_csv(T.OUT_DIR / "shared_landscape_ladder.csv")
                 .set_index("model_key").loc[T.WINNER_KEY, "shared_pow"])

    d = pd.read_csv(T.DATA_PATH)
    d = d[d.cond == "DURATION"]
    tr = d.drop_duplicates(["uniqueid", "pid", "trial_id", "datetime"]).copy()
    tr = tr[tr.exp_dur.isin(DURS)]
    tr = tr[(tr.n_shown >= 1) & (tr.n_shown <= 15)
            & (tr.n_guess >= 1) & (tr.n_guess <= 40)]

    # pooled lapse (width-only model, exponent fixed) from the pooled accuracy curve
    pooled = np.array([float((tr[tr.n_shown == n].n_guess == n).mean()) for n in NVALS])

    def pooled_sse(lapse):
        best = np.inf
        for sc in np.geomspace(0.005, 0.5, 60):
            pred = acc_curve(sc, S0_FLOOR, 0.1, pow_, lapse, quad)
            best = min(best, float(np.mean((pred - pooled) ** 2)))
        return best
    lapse = float(minimize_scalar(pooled_sse, bounds=(1e-4, 0.4),
                                  method="bounded").x)
    print(f"exponent p={pow_:.3f} (group joint)   pooled lapse={lapse:.3f}", flush=True)

    sc_grid = np.geomspace(0.005, 0.5, NG)
    a_curves = np.array([acc_curve(sc, S0_FLOOR, 0.1, pow_, lapse, quad) for sc in sc_grid])
    s0_grid = np.geomspace(1e-4, 0.5, NG)
    b_curves = np.array([acc_curve(SIGMAC_OFF, s0, 0.1, pow_, lapse, quad) for s0 in s0_grid])

    rows = []
    for pid, g in tr.groupby("pid"):
        if len(g) < MIN_TRIALS:
            continue
        obs = subj_acc(g)
        m = np.isfinite(obs)
        sse_w = np.mean((a_curves[:, m] - obs[None, m]) ** 2, axis=1)
        ia = int(np.argmin(sse_w))
        sse_n = float(np.min(np.mean((b_curves[:, m] - obs[None, m]) ** 2, axis=1)))
        rows.append({
            "pid": pid, "n_trials": int(len(g)),
            "n_width": n_width_of(sc_grid[ia], pow_),
            "sse_width": float(sse_w[ia]), "sse_noise": sse_n,
            "weber": weber_fraction(g),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    dd = df[["weber", "n_width"]].dropna()
    rs, ps = spearmanr(dd.weber, dd.n_width)
    print(f"N = {len(df)} subjects ({'+'.join(map(str, DURS))} ms, >= {MIN_TRIALS} trials)")
    print(f"width-only beats noise-only in {(df.sse_width < df.sse_noise).sum()}/{len(df)}")
    print(f"n_width vs Weber: Spearman rho = {rs:+.3f}  p = {ps:.2e}  (n = {len(dd)})")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
