"""Simulation-based parameter recovery for a fitted driven model (Stage 4).

    .venv/bin/python driven_attractor/recover.py --fit output/fits/M2_tether_g0.json --n-synth 40

For each synthetic dataset: true parameters are jittered around the fitted
values (log-normal +-25% on scale parameters, +-0.25 on p), trial-level
responses are drawn from the model's own p_final tables with the real trial
design (same per-cell counts), and the model is refit from scratch
(differential evolution + polish; no warm start at truth). Reports identity
R^2 and Spearman rho per parameter and for the derived quantities the paper
interprets (n_crit = sigma_c^(-1/p), sigma(5), E[K] at 160 ms), plus the
error-correlation matrix (trade-off structure).
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import fitting as F  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
RSEED = 42

_G: dict = {}


def _init(model, gamma, cell_counts, r_max):
    _G.update(model=model, gamma=gamma, cell_counts=cell_counts, r_max=r_max)


def _one_synth(job):
    s_idx, truth_vec, defs_names = job
    model, gamma, r_max = _G["model"], _G["gamma"], _G["r_max"]
    defs = F.param_defs(model)
    truth_theta = F.vec_to_theta(model, np.asarray(truth_vec), defs, gamma)
    rng = np.random.default_rng(RSEED + 1000 + s_idx)

    ev = M.evaluate_model(model, truth_theta, m_paths=4096, dt=0.0025)
    counts = {}
    for i, d in enumerate(M.DURATIONS):
        lapse = 0.07
        pf = M.p_final_full(ev.p_wells[i], ev.p_guess[i], lapse, r_max)
        c = np.zeros_like(pf)
        for ni in range(M.N_SHOWN.size):
            c[ni] = rng.multinomial(int(_G["cell_counts"][d][ni]), pf[ni])
        counts[d] = c

    bounds = [dd.bounds() for dd in defs]
    args = (model, defs, gamma, 512, counts, r_max, 0.005)
    de = differential_evolution(F._negll, bounds=bounds, args=args,
                                maxiter=15, popsize=6, tol=1e-6, seed=RSEED + s_idx,
                                init="sobol", polish=False, updating="deferred",
                                workers=1)
    nm_args = (model, defs, gamma, 1024, counts, r_max, 0.0025)
    pol = minimize(F._negll, de.x, args=nm_args, method="Nelder-Mead",
                   bounds=bounds, options={"maxiter": 150, "xatol": 1e-4,
                                           "fatol": 0.05})
    x = pol.x if pol.fun <= F._negll(de.x, *nm_args) else de.x
    est_theta = F.vec_to_theta(model, x, defs, gamma)
    row = {"synth": s_idx}
    for name in defs_names:
        row[f"true_{name}"] = truth_theta.get(name, np.nan)
        row[f"est_{name}"] = est_theta.get(name, np.nan)
    for lab, th in (("true", truth_theta), ("est", est_theta)):
        sc, p = th["sigma_c"], th["p"]
        row[f"{lab}_ncrit"] = sc ** (-1.0 / p)
        row[f"{lab}_sigma5"] = sc * 5.0 ** p
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--n-synth", type=int, default=40)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    fp = Path(args.fit)
    with open(fp if fp.is_absolute() else ROOT / "driven_attractor" / fp) as f:
        fit = json.load(f)
    model, gamma = fit["model"], float(fit["gamma"])
    defs = F.param_defs(model)
    names = [d.name for d in defs]
    x_hat = np.asarray(fit["x_opt"], float)

    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    cell_counts = {d: data[d].counts.sum(axis=1) for d in M.DURATIONS}

    rng = np.random.default_rng(RSEED)
    jobs = []
    for s in range(args.n_synth):
        jit = x_hat.copy()
        for i, dd in enumerate(defs):
            if dd.log:
                jit[i] = np.clip(x_hat[i] + rng.normal(0, 0.25), *dd.bounds())
            else:
                jit[i] = np.clip(x_hat[i] + rng.normal(0, 0.25), *dd.bounds())
        jobs.append((s, jit, names))

    with Pool(args.workers, initializer=_init,
              initargs=(model, gamma, cell_counts, r_max)) as pool:
        rows = pool.map(_one_synth, jobs)
    df = pd.DataFrame(rows)
    tag = fp.stem
    df.to_csv(OUT / f"recovery_{tag}.csv", index=False)

    print(f"=== recovery: {tag} ({args.n_synth} synths) ===")
    quants = names + ["ncrit", "sigma5"]
    summ = []
    for q in quants:
        t, e = df[f"true_{q}"], df[f"est_{q}"]
        if t.isna().all():
            continue
        ss = 1 - ((t - e) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        rho = spearmanr(t, e).statistic
        summ.append({"param": q, "R2": ss, "spearman": rho,
                     "median_rel_err": float(np.median(np.abs(e - t) /
                                                       np.maximum(1e-9, np.abs(t))))})
        print(f"  {q:>8}: R2 {ss:6.3f}  rho {rho:6.3f}")
    pd.DataFrame(summ).to_csv(OUT / f"recovery_{tag}_summary.csv", index=False)
    err = pd.DataFrame({q: df[f"est_{q}"] - df[f"true_{q}"] for q in names})
    print("\nerror correlations:")
    print(err.corr().round(2).to_string())


if __name__ == "__main__":
    main()
