"""Participant-blocked CV for the round-2 evidence families (+ M0 references).

Same folds as round 1 (seed 20260711, 5 folds over the 379 participants), so
the numbers are directly comparable with output/cv_lpd_summary.csv.

    .venv/bin/python driven_attractor/cv_family.py --fold 0 \
        --models ddm_batch_attr,ddm_online_attr,ddm_online_static,ddm_batch_static

Every model is refit on the training participants (Nelder-Mead warm-started
from its full-data optimum) and scored on the held-out participants. The
lapse is estimated on the training fold only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D  # noqa: E402
from driven_attractor.driven import evidence2 as E  # noqa: E402
from driven_attractor.driven import fitting2 as F2  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
FITS = OUT / "fits2"
CV_SEED = 20260711
N_FOLDS = 5
M_CV_FIT = 1024
M_CV_FINAL = 4096   # lpd noise ~10 logL on 10k test trials; effects are 100s


def folds_of(subject: np.ndarray) -> np.ndarray:
    n_subj = int(subject.max()) + 1
    rng = np.random.default_rng(CV_SEED)
    perm = rng.permutation(n_subj)
    return perm[subject] % N_FOLDS


def trial_lpd(dat: dict, sel: np.ndarray, pf: dict[int, np.ndarray]) -> np.ndarray:
    lp = np.empty(int(sel.sum()))
    ed, ns, ng = dat["exp_dur"][sel], dat["n_shown"][sel], dat["n_guess"][sel]
    for i in range(lp.size):
        tab = pf[int(ed[i])]
        lp[i] = np.log(tab[ns[i] - 1, min(int(ng[i]), tab.shape[1]) - 1])
    return lp


def m0_tables(which: str, counts_train: dict, r_max: int) -> dict[int, np.ndarray]:
    """M0 refit on the training counts (per-duration or shared landscape)."""
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(201, kind="grid")
    quad_hi = T.make_quadrature(1601, kind="grid")
    pt = pd.read_csv(ROOT / "legacy/trial_level_per_duration/output/pow_theta.csv")
    warm = {int(r.duration_ms): np.array([r.sigma_c, r.s0, r.lambda0,
                                          r["pow"], r.lapse])
            for _, r in pt.iterrows()}
    out = {}
    if which == "perdur":
        for d in E.DURATIONS:
            def neg(v, d=d):
                return -T.response_loglik(spec, v, quad, counts_train[d], r_max)
            res = minimize(neg, warm[d], method="Nelder-Mead",
                           bounds=list(spec.bounds) + [T.LAPSE_BOUNDS],
                           options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-3})
            v = res.x if res.fun <= neg(warm[d]) else warm[d]
            g = T.compute_grid(spec, dict(zip(spec.param_names, v[:-1])), quad_hi)
            out[d] = T.apply_lapse(g["p_resp"], float(v[-1]), r_max)
        return out
    sl = pd.read_csv(ROOT / "output/trial_level_model_comparison/"
                     "shared_landscape_ladder_theta.csv")
    sl = sl[sl.model_key == "gaussian_pow_attractor"]
    shared = np.array([sl["sigma_c"].iloc[0], sl["pow"].iloc[0]])
    nuis = {int(r.duration_ms): np.array([r.s0, r.lambda0, r.lapse])
            for _, r in sl.iterrows()}
    sb = [spec.bounds[0], spec.bounds[3]]
    nb = [spec.bounds[1], spec.bounds[2], T.LAPSE_BOUNDS]

    def ll_d(sh, nu, d, q):
        v = np.array([sh[0], nu[0], nu[1], sh[1], nu[2]])
        return T.response_loglik(spec, v, q, counts_train[d], r_max)

    # warm-started at the full-data optimum -> one polish round suffices
    for _ in range(1):
        for d in E.DURATIONS:
            r = minimize(lambda v, d=d: -ll_d(shared, v, d, quad), nuis[d],
                         method="Nelder-Mead", bounds=nb,
                         options={"maxiter": 150, "xatol": 1e-5, "fatol": 1e-3})
            if -r.fun >= ll_d(shared, nuis[d], d, quad):
                nuis[d] = r.x
        r = minimize(lambda s: -sum(ll_d(s, nuis[d], d, quad) for d in E.DURATIONS),
                     shared, method="Nelder-Mead", bounds=sb,
                     options={"maxiter": 150, "xatol": 1e-6, "fatol": 1e-3})
        if -r.fun >= sum(ll_d(shared, nuis[d], d, quad) for d in E.DURATIONS):
            shared = r.x
    for d in E.DURATIONS:
        th = {"sigma_c": shared[0], "s0": nuis[d][0], "lambda0": nuis[d][1],
              "pow": shared[1]}
        g = T.compute_grid(spec, th, quad_hi)
        out[d] = T.apply_lapse(g["p_resp"], float(nuis[d][2]), r_max)
    return out


def family_tables(name: str, counts_train: dict, r_max: int,
                  budget: int) -> dict[int, np.ndarray]:
    with open(FITS / f"{name}.json") as f:
        fit = json.load(f)
    family, variant = fit["family"], fit["variant"]
    names = E.param_names(family, variant)
    bnds = F2.opt_bounds(names)
    is_batch = variant.startswith("batch")
    m_fit = 1024 if not is_batch else M_CV_FIT
    args = (family, variant, names, counts_train, r_max, m_fit, F2.QUAD_FIT)
    x0 = np.asarray(fit["x_opt"], float)
    res = minimize(F2._negll, x0, args=args, method="Nelder-Mead", bounds=bnds,
                   options={"maxiter": budget, "xatol": 1e-4, "fatol": 0.05})
    x = res.x if res.fun <= F2._negll(x0, *args) else x0
    theta = F2.vec_to_theta(names, x)
    ev = E.evaluate(family, variant, theta,
                    m_paths=M_CV_FINAL if not is_batch else M_CV_FINAL,
                    quad_nodes=F2.QUAD_FINAL)
    out = {}
    for i, d in enumerate(E.DURATIONS):
        lapse, _ = M.profile_lapse(ev["p_wells"][i], ev["p_guess"][i],
                                   counts_train[d], r_max)
        out[d] = M.p_final_full(ev["p_wells"][i], ev["p_guess"][i], lapse, r_max)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--with-m0", action="store_true")
    ap.add_argument("--budget", type=int, default=150)
    args = ap.parse_args()

    dat = D.load()
    r_max = D.r_max(dat)
    fold = folds_of(dat["subject"])
    tr, te = fold != args.fold, fold == args.fold
    counts_tr = D.counts_by_duration(dat, r_max, mask=tr)
    print(f"fold {args.fold}: {np.unique(dat['subject'][tr]).size} train / "
          f"{np.unique(dat['subject'][te]).size} test participants, "
          f"{int(te.sum())} test trials", flush=True)

    rows, prows = [], []
    jobs = [(m, None) for m in args.models.split(",") if m]
    if args.with_m0:
        jobs = [("M0_shared", "shared"), ("M0_perdur", "perdur")] + jobs

    for name, m0kind in jobs:
        t0 = time.time()
        pf = (m0_tables(m0kind, counts_tr, r_max) if m0kind
              else family_tables(name, counts_tr, r_max, args.budget))
        lp = trial_lpd(dat, te, pf)
        rows.append({"model": name, "fold": args.fold, "lpd": float(lp.sum()),
                     "n_trials": int(te.sum()), "mean_lpd": float(lp.mean())})
        subj = dat["subject"][te]
        for s in np.unique(subj):
            prows.append({"model": name, "fold": args.fold, "subject": int(s),
                          "lpd": float(lp[subj == s].sum())})
        print(f"  {name}: held-out lpd {lp.sum():.1f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"cv2_folds_fold{args.fold}.csv", index=False)
    pd.DataFrame(prows).to_csv(OUT / f"cv2_participant_fold{args.fold}.csv",
                               index=False)


if __name__ == "__main__":
    main()
