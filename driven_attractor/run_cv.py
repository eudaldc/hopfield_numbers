"""Participant-blocked 5-fold cross-validation of M0 vs the driven family.

Requires the full-data fits in driven_attractor/output/fits/ (for warm
starts). Fits every model on each fold's training participants and scores
held-out participants. Run:

    .venv/bin/python driven_attractor/run_cv.py [--models m1,m2,...]

Outputs:
  output/cv_lpd_summary.csv    total + per-fold held-out lpd per model
  output/cv_lpd_participant.csv per-participant held-out lpd per model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import cv as CV  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
FITS = OUT / "fits"

DRIVEN_DEFAULT = ["M2_tether_g0", "M2_tether_g1", "M1_offset_g0",
                  "M1_offset_g1", "M3_static_g0", "M2_pulse_g0"]


def load_fit(name: str) -> dict:
    with open(FITS / f"{name}.json") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DRIVEN_DEFAULT))
    ap.add_argument("--skip-m0", action="store_true")
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--fold", type=int, default=-1,
                    help="run only this fold (parallel across processes)")
    args = ap.parse_args()
    driven_models = [m for m in args.models.split(",") if m]

    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    trial = trial.assign(fold=CV.make_folds(trial))

    # warm starts
    pow_theta = pd.read_csv(ROOT / "legacy/trial_level_per_duration/output/pow_theta.csv")
    warm_perdur = {int(r.duration_ms): np.array([r.sigma_c, r.s0, r.lambda0,
                                                 r["pow"], r.lapse])
                   for _, r in pow_theta.iterrows()}
    sl_theta = pd.read_csv(
        ROOT / "output/trial_level_model_comparison/shared_landscape_ladder_theta.csv")
    sl = sl_theta[sl_theta.model_key == "gaussian_pow_attractor"]
    warm_shared = np.array([sl["sigma_c"].iloc[0], sl["pow"].iloc[0]])
    warm_nuis = {int(r.duration_ms): np.array([r.s0, r.lambda0, r.lapse])
                 for _, r in sl.iterrows()}

    rows, part_rows = [], []
    folds = range(CV.N_FOLDS) if args.fold < 0 else [args.fold]
    for fold in folds:
        tr = trial[trial.fold != fold]
        te = trial[trial.fold == fold]
        c_tr = CV.counts_from(tr, r_max)
        c_te = CV.counts_from(te, r_max)
        n_te = len(te)
        print(f"── fold {fold}: {tr.uniqueid.nunique()} train / "
              f"{te.uniqueid.nunique()} test participants, {n_te} test trials",
              flush=True)

        tables_by_model: dict[str, dict[int, np.ndarray]] = {}

        if not args.skip_m0:
            t0 = time.time()
            f_pd = CV.fit_m0_perdur(c_tr, r_max, warm_perdur, budget=400)
            tables_by_model["M0_perdur"] = CV.m0_tables(
                f_pd["theta"], f_pd["lapse"], r_max)
            f_sh = CV.fit_m0_shared(c_tr, r_max, warm_shared, warm_nuis)
            tables_by_model["M0_shared"] = CV.m0_tables(
                f_sh["theta"], f_sh["lapse"], r_max)
            print(f"   M0 refits done ({(time.time()-t0)/60:.1f} min)", flush=True)

        for name in driven_models:
            t0 = time.time()
            full = load_fit(name)
            model, gamma = full["model"], float(full["gamma"])
            ff = CV.fit_driven_fold(model, gamma, c_tr, r_max,
                                    np.asarray(full["x_opt"], float),
                                    budget=args.budget)
            tables_by_model[name] = CV.driven_tables(model, ff["theta"], c_tr,
                                                     r_max)
            print(f"   {name} refit done ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

        for name, tables in tables_by_model.items():
            lp = CV.trial_lpd(te, tables)
            rows.append({"model": name, "fold": fold, "lpd": float(lp.sum()),
                         "n_trials": n_te, "mean_lpd": float(lp.mean())})
            per = te.assign(lp=lp).groupby("uniqueid")["lp"].sum()
            for uid, v in per.items():
                part_rows.append({"model": name, "fold": fold,
                                  "uniqueid": uid, "lpd": float(v)})
            print(f"   [fold {fold}] {name}: held-out lpd {lp.sum():.1f}",
                  flush=True)

    df = pd.DataFrame(rows)
    suffix = f"_fold{args.fold}" if args.fold >= 0 else ""
    df.to_csv(OUT / f"cv_lpd_folds{suffix}.csv", index=False)
    pd.DataFrame(part_rows).to_csv(OUT / f"cv_lpd_participant{suffix}.csv",
                                   index=False)
    tot = (df.groupby("model")
           .agg(lpd_total=("lpd", "sum"), n=("n_trials", "sum"))
           .assign(mean_lpd=lambda x: x.lpd_total / x.n)
           .sort_values("lpd_total", ascending=False))
    print("\n=== participant-blocked CV: held-out log predictive density ===")
    print(tot.round(2).to_string())


if __name__ == "__main__":
    main()
