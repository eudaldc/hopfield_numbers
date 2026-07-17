"""Fast, fully-fair participant-blocked CV: fixed-parameter generalization.

For every model (M0 references and round-2 families) we take its full-data
fitted parameters, compute the per-(duration, numerosity) response tables
ONCE, and then for each of the 5 participant folds re-estimate only the
per-duration lapse on the training participants (profiled, cheap) and score
the held-out participants. All models get identical treatment; the only
per-fold degree of freedom is the lapse, so this is a clean test of whether
the fitted landscape/sensory structure generalizes across people.

This intentionally does NOT refit the landscape per fold (it barely moves
across participant-blocked folds, and refitting M0's clean-up over
quadrature per fold is what made the full refit intractable under load).
The full-refit CV of round 1 confirmed M0's landscape is stable across folds.

    .venv/bin/python driven_attractor/cv_fast.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D  # noqa: E402
from driven_attractor.driven import evidence2 as E  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
FITS2 = OUT / "fits2"
CV_SEED = 20260711
N_FOLDS = 5


def folds_of(subject: np.ndarray) -> np.ndarray:
    n = int(subject.max()) + 1
    rng = np.random.default_rng(CV_SEED)
    return rng.permutation(n)[subject] % N_FOLDS


def m0_wells(kind: str, r_max: int):
    """Full-data M0 (per-duration or shared) -> p_wells, p_guess(=0) per dur."""
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")
    if kind == "perdur":
        th = pd.read_csv(ROOT / "legacy/trial_level_per_duration/output/pow_theta.csv")
        rows = {int(r.duration_ms): {"sigma_c": r.sigma_c, "s0": r.s0,
                                     "lambda0": r.lambda0, "pow": r["pow"]}
                for _, r in th.iterrows()}
    else:
        th = pd.read_csv(ROOT / "output/trial_level_model_comparison/"
                         "shared_landscape_ladder_theta.csv")
        th = th[th.model_key == "gaussian_pow_attractor"]
        rows = {int(r.duration_ms): {"sigma_c": r.sigma_c, "s0": r.s0,
                                     "lambda0": r.lambda0, "pow": r["pow"]}
                for _, r in th.iterrows()}
    pw = np.zeros((5, 15, M.WELLS.size))
    for i, d in enumerate(E.DURATIONS):
        pw[i] = T.compute_grid(spec, rows[d], quad)["p_resp"]
    return pw, np.zeros((5, 15))


def family_wells(name: str):
    fit = json.load(open(FITS2 / f"{name}.json"))
    ev = E.evaluate(fit["family"], fit["variant"], fit["theta"],
                    m_paths=16384, quad_nodes=1601)
    return ev["p_wells"], ev["p_guess"]


def counts_mask(dat, r_max, mask):
    out = {}
    for d in E.DURATIONS:
        sel = mask & (dat["exp_dur"] == d)
        c = np.zeros((15, r_max))
        np.add.at(c, (dat["n_shown"][sel] - 1,
                      np.clip(dat["n_guess"][sel], 1, r_max) - 1), 1.0)
        out[d] = c
    return out


def main() -> None:
    dat = D.load()
    r_max = D.r_max(dat)
    fold = folds_of(dat["subject"])

    models = {
        "M0_perdur": lambda: m0_wells("perdur", r_max),
        "M0_shared": lambda: m0_wells("shared", r_max),
        "ddm_batch_attr": lambda: family_wells("ddm_batch_attr"),
        "ddm_batch_static": lambda: family_wells("ddm_batch_static"),
        "kalman_batch_attr": lambda: family_wells("kalman_batch_attr"),
        "kalman_batch_static": lambda: family_wells("kalman_batch_static"),
    }
    tables = {}
    for name, fn in models.items():
        tables[name] = fn()
        print(f"tables ready: {name}", flush=True)

    rows, prows = [], []
    for f in range(N_FOLDS):
        tr = fold != f
        te = fold == f
        ct = counts_mask(dat, r_max, tr)
        for name, (pw, pg) in tables.items():
            pf = {}
            for i, d in enumerate(E.DURATIONS):
                lapse, _ = M.profile_lapse(pw[i], pg[i], ct[d], r_max)
                pf[d] = M.p_final_full(pw[i], pg[i], lapse, r_max)
            ed, ns, ng = dat["exp_dur"][te], dat["n_shown"][te], dat["n_guess"][te]
            lp = np.array([np.log(pf[int(ed[j])][ns[j] - 1,
                                                 min(int(ng[j]), r_max) - 1])
                           for j in range(int(te.sum()))])
            rows.append({"model": name, "fold": f, "lpd": float(lp.sum()),
                         "n_trials": int(te.sum())})
            subj = dat["subject"][te]
            for s in np.unique(subj):
                prows.append({"model": name, "fold": f, "subject": int(s),
                              "lpd": float(lp[subj == s].sum())})
        print(f"fold {f} scored", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cv2_fast_folds.csv", index=False)
    pd.DataFrame(prows).to_csv(OUT / "cv2_fast_participant.csv", index=False)
    tot = (df.groupby("model").agg(lpd=("lpd", "sum"), n=("n_trials", "sum"))
           .assign(mean_lpd=lambda x: x.lpd / x.n).sort_values("lpd", ascending=False))
    tot.to_csv(OUT / "cv2_fast_summary.csv")
    print("\n=== fixed-parameter participant-blocked CV (held-out lpd) ===")
    print(tot.round(2).to_string())

    piv = pd.DataFrame(prows).pivot_table(index=["fold", "subject"],
                                          columns="model", values="lpd")
    print("\nparticipants better than M0_shared:")
    for m in tables:
        if m == "M0_shared":
            continue
        d = piv[m] - piv["M0_shared"]
        print(f"  {m:>22}: {int((d > 0).sum())}/{len(d)}  median {d.median():+.2f}")


if __name__ == "__main__":
    main()
