"""High-MC delta estimate: driven sibling vs M0 at optimal beta, low noise.

    .venv/bin/python driven_attractor/highmc_delta.py --m 49152
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D, model as M  # noqa: E402
import driven_attractor.driven_sibling as S  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402


def m0_ref(par, counts, r_max):
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")
    tot = 0.0
    for d in M.DURATIONS:
        th = {"sigma_c": par[d]["sigma_c"], "s0": par[d]["s0"],
              "lambda0": par[d]["lam"], "pow": par[d]["p"]}
        g = T.compute_grid(spec, th, quad)
        _, ll = M.profile_lapse(g["p_resp"], np.zeros(15), counts[d], r_max)
        tot += ll
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=49152)
    args = ap.parse_args()
    dat = D.load(); r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)
    for which in ("shared", "perdur"):
        par = S.m0_params(which)
        m0 = m0_ref(par, counts, r_max)
        for seed in (909, 4242, 7):        # three seeds -> MC spread
            tot, _ = S.total_loglik(par, 5000.0, counts, r_max, args.m, seed=seed)
            print(f"{which:>7} beta=5000 seed={seed}: driven {tot:.1f}  "
                  f"M0 {m0:.1f}  dlogL {tot - m0:+.1f}", flush=True)
    print("HIGHMC_DONE", flush=True)


if __name__ == "__main__":
    main()
