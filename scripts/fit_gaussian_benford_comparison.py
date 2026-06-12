"""Fit the Gaussian-Benford *attractor* (Supplementary Material 1) per duration,
under the same conventions as the main Poisson model, so its AIC is directly
comparable to the Hopfield-Poisson fit in the model comparison.

This must be run with a gaussian ``configs/config.yaml`` (well_type: gaussian,
benford_prior: true, no_anchor: true, fixed_beta/kappa/eta, fixed_t0: 1.3). The
driver swaps the config in and out; here we only assert it is in place.

    PYTHONPATH=. .venv/bin/python scripts/fit_gaussian_benford_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.fit_helpers as fh

DURATIONS = (40, 80, 160, 320, 640)
OUT = Path("output/model_comparison/gaussian_benford_attractor_fit.csv")

assert fh.WELL_TYPE == "gaussian", "config.yaml must set well_type: gaussian"
assert fh.RT_MODE == "entropy"
assert list(fh.PARAM_NAMES) == ["sigma_c", "s0", "lambda0", "kappa"], fh.PARAM_NAMES


def fit_one(emp: pd.DataFrame, maxiter: int = 150, seed: int = 17) -> tuple[dict, float]:
    emp_arr = fh._extract_empirical_arrays(emp)
    sample_state = [300]
    res = differential_evolution(
        fh.loss_fn,
        bounds=fh.BOUNDS,
        args=(emp_arr, sample_state, 73),
        maxiter=maxiter,
        popsize=8,
        seed=seed,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.85,
        polish=True,
        init="latinhypercube",
        vectorized=True,
        updating="deferred",
    )
    theta = fh.complete_piantadosi_theta(dict(zip(fh.PARAM_NAMES, res.x)))
    return theta, float(res.fun)


def main() -> None:
    rows = []
    for d in DURATIONS:
        emp = fh.extract_empirical(d)
        theta, loss = fit_one(emp)
        model = fh.simulate(theta, n_samples_per_n=5000, seed=73)
        g = fh.compute_goodness(emp, model, k=4)
        rows.append(
            {
                "duration_ms": d,
                "loss": loss,
                "sigma_c": theta["sigma_c"],
                "s0": theta["s0"],
                "lambda0": theta["lambda0"],
                "kappa": theta["kappa"],
                "R2_mean_guess": g["R2_mean_guess"],
                "R2_pct_exact": g["R2_pct_exact"],
                "R2_rt": g["R2_rt"],
                "logL": g["logL"],
                "AIC": g["AIC"],
                "BIC": g["BIC"],
            }
        )
        print(
            f"{d}ms loss={loss:.4f} AIC={g['AIC']:.1f} "
            f"R2(guess/exact/rt)={g['R2_mean_guess']:.3f}/{g['R2_pct_exact']:.3f}/{g['R2_rt']:.3f}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGB-attractor mean AIC = {df.AIC.mean():.1f}  mean BIC = {df.BIC.mean():.1f}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
