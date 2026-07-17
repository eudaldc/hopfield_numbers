"""Recovery + posterior-predictive checks for kalman_batch_static, the leading
round-2 model (task 13). Runs only because it shows a held-out advantage
over M0-shared.

Recovery: 24 synthetic datasets at jittered truths, refit from scratch,
identity R^2 for the 6 mechanism params + the interpreted shrinkage
(prior mean mu0, prior weight at 40 ms). PP: mean guess / P(exact) / CoV
overlay vs data and vs M0-shared.

    .venv/bin/python driven_attractor/recover_pp_kalman.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D  # noqa: E402
from driven_attractor.driven import evidence2 as E  # noqa: E402
from driven_attractor.driven import fitting2 as F2  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
FIG = ROOT / "driven_attractor" / "figures"
RSEED = 71
FAM, VAR = "kalman", "batch_static"
NAMES = E.param_names(FAM, VAR)

_G: dict = {}


def _init(cell_counts, r_max):
    _G.update(cell_counts=cell_counts, r_max=r_max)


def _one(job):
    s, truth_vec = job
    r_max = _G["r_max"]
    truth = F2.vec_to_theta(NAMES, truth_vec)
    rng = np.random.default_rng(RSEED + 100 + s)
    ev = E.evaluate(FAM, VAR, truth, quad_nodes=801)
    counts = {}
    for i, d in enumerate(E.DURATIONS):
        pf = M.p_final_full(ev["p_wells"][i], ev["p_guess"][i], 0.07, r_max)
        c = np.zeros_like(pf)
        for ni in range(15):
            c[ni] = rng.multinomial(int(_G["cell_counts"][d][ni]), pf[ni])
        counts[d] = c
    bnds = F2.opt_bounds(NAMES)
    args = (FAM, VAR, NAMES, counts, r_max, 1, 401)
    from scipy.optimize import differential_evolution
    de = differential_evolution(F2._negll, bounds=bnds, args=args, maxiter=20,
                                popsize=8, tol=1e-6, seed=RSEED + s, init="sobol",
                                polish=False, updating="deferred", workers=1)
    pol = minimize(F2._negll, de.x, args=args, method="Nelder-Mead", bounds=bnds,
                   options={"maxiter": 200, "xatol": 1e-4, "fatol": 0.05})
    x = pol.x if pol.fun <= de.fun else de.x
    est = F2.vec_to_theta(NAMES, x)
    row = {"synth": s}
    for n in NAMES:
        row[f"true_{n}"] = truth[n]
        row[f"est_{n}"] = est[n]
    return row


def recovery(dat, r_max, n_synth=24):
    fit = json.load(open(OUT / f"fits2/{FAM}_{VAR}.json"))
    x_hat = np.asarray(fit["x_opt"], float)
    cell = {d: np.array([int((dat["exp_dur"] == d).sum() / 15)] * 15)
            for d in E.DURATIONS}
    # use the real per-cell counts
    cell = {}
    for d in E.DURATIONS:
        c = np.zeros(15)
        for n in range(1, 16):
            c[n - 1] = int(((dat["exp_dur"] == d) & (dat["n_shown"] == n)).sum())
        cell[d] = c
    rng = np.random.default_rng(RSEED)
    jobs = []
    for s in range(n_synth):
        jit = x_hat + rng.normal(0, 0.25, size=x_hat.size)
        jit = np.clip(jit, [b[0] for b in F2.opt_bounds(NAMES)],
                      [b[1] for b in F2.opt_bounds(NAMES)])
        jobs.append((s, jit))
    with Pool(12, initializer=_init, initargs=(cell, r_max)) as pool:
        rows = pool.map(_one, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recovery_kalman_batch_static.csv", index=False)
    print("=== recovery: kalman_batch_static ===")
    for n in NAMES:
        t, e = df[f"true_{n}"], df[f"est_{n}"]
        r2 = 1 - ((t - e) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        rho = spearmanr(t, e).statistic
        print(f"  {n:>8}: R2 {r2:6.3f}  rho {rho:6.3f}")
    return df


def pp(dat, r_max):
    ct = {}
    for d in E.DURATIONS:
        c = np.zeros((15, r_max))
        sel = dat["exp_dur"] == d
        np.add.at(c, (dat["n_shown"][sel] - 1,
                      np.clip(dat["n_guess"][sel], 1, r_max) - 1), 1.0)
        ct[d] = c
    ev = E.evaluate(FAM, VAR, json.load(open(OUT / f"fits2/{FAM}_{VAR}.json"))["theta"],
                    m_paths=16384, quad_nodes=1601)
    rr = np.arange(1, r_max + 1, dtype=float)
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=True)
    for j, d in enumerate(E.DURATIONS):
        obs = ct[d] / ct[d].sum(axis=1, keepdims=True)
        lapse, _ = M.profile_lapse(ev["p_wells"][j], ev["p_guess"][j], ct[d], r_max)
        pf = M.p_final_full(ev["p_wells"][j], ev["p_guess"][j], lapse, r_max)
        for i, (lab, tab) in enumerate((("mean", None), ("exact", None), ("cov", None))):
            ax = axes[i, j]
            def stat(P):
                m = (P[:, :19] / P[:, :19].sum(1, keepdims=True) * rr[:19]).sum(1)
                ex = np.array([P[k, k] for k in range(15)])
                v = (P[:, :19] / P[:, :19].sum(1, keepdims=True) * rr[:19] ** 2).sum(1) - m ** 2
                return {"mean": m, "exact": ex, "cov": np.sqrt(np.maximum(v, 0)) / m}[lab]
            ax.plot(range(1, 16), stat(obs), "ko", ms=4, label="data")
            ax.plot(range(1, 16), stat(pf), "-", color="tab:green", label="Kalman static")
            if i == 0:
                ax.set_title(f"{d} ms"); ax.plot(range(1, 16), range(1, 16), ":", c="grey", lw=.7)
            if j == 0:
                ax.set_ylabel(["mean guess", "P(exact)", "CoV"][i])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("kalman_batch_static (Bayesian shrinkage sensory channel, no attractor) vs data")
    fig.tight_layout()
    fig.savefig(FIG / "kalman_batch_static_pp.png", dpi=140)
    print(f"saved {FIG / 'kalman_batch_static_pp.png'}")


def main():
    dat = D.load()
    r_max = D.r_max(dat)
    recovery(dat, r_max)
    pp(dat, r_max)


if __name__ == "__main__":
    main()
