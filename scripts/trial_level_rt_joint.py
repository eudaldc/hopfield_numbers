"""Joint RT fit: one response-time mapping shared across the five durations.

The per-duration RT comparison (rt_exploration.csv) fits the RT link separately
for every exposure. The parallel to the shared-landscape response fit is a single
RT mapping: the scale coefficients that turn convergence time and readout entropy
into an expected response time are shared across all five durations, on the
winning landscape's per-duration grids, leaving only the residual scatter sigma_rt
free per duration (the nuisance). This is the rightmost "joint" column of the RT
heatmap: even with one mapping for every exposure, the dynamics-based link wins.

    .venv/bin/python -m scripts.trial_level_rt_joint
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

import scripts.trial_level_model_comparison as T

OUT = T.OUT_DIR / "rt_joint.csv"
METRICS = ("hybrid", "steps", "entropy")

# Winning landscape: power-law-width Gaussian attractor (module WINNER_KEY).
WINNER = T.MODEL_BY_KEY[T.WINNER_KEY]


def main() -> None:
    quad = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    theta_df = pd.read_csv(T.OUT_DIR / "pow_theta.csv").set_index("duration_ms")
    durs = list(T.DURATIONS)

    # Per-duration landscape quantities on the winner's fitted theta.
    per = {}
    for d in durs:
        th = {p: float(theta_df.loc[d, p]) for p in WINNER.param_names}
        grid = T.compute_grid(WINNER, th, quad)
        per[d] = {
            "eh": grid["EH"], "et": grid["ET"], "et_flow": grid["ET_flow"],
            "idx": data[d].n_trial - T.N_TRUE_MIN,
            "logrt": data[d].logrt_trial,
        }
    n_tot = sum(per[d]["logrt"].size for d in durs)
    jac = sum(float(np.sum(per[d]["logrt"])) for d in durs)

    rows = []
    for metric in METRICS:
        bounds = T._rt_scale_bounds(metric)

        def summed_negll(scale: np.ndarray) -> float:
            """Shared scale, sigma_rt profiled per duration (the nuisance)."""
            tot = 0.0
            for d in durs:
                link = T._rt_link(metric, scale, per[d]["eh"], per[d]["et"], per[d]["et_flow"])
                mu = np.log(T.T0 + link)
                if not np.all(np.isfinite(mu)):
                    return 1e18
                resid = per[d]["logrt"] - mu[per[d]["idx"]]
                s2 = float(np.mean(resid * resid))
                if not (s2 > 0 and np.isfinite(s2)):
                    return 1e18
                tot += 0.5 * resid.size * (np.log(2.0 * np.pi * s2) + 1.0)
            return tot

        de = differential_evolution(summed_negll, bounds=bounds, maxiter=200,
                                    popsize=15, tol=1e-7, seed=T.SEED, polish=True,
                                    updating="deferred", workers=1)
        scale = de.x
        logl = -float(de.fun) - jac        # density of rt (sum of per-dur log-normals)
        k = len(bounds) + len(durs)         # shared scale(s) + one sigma_rt per duration
        rows.append({
            "metric": metric, "logL": float(logl), "k": int(k),
            "AIC": float(2 * k - 2 * logl), "BIC": float(k * np.log(n_tot) - 2 * logl),
            "scale": ",".join(f"{v:.5f}" for v in scale),
        })
        print(f"  [{metric}] shared scale={np.round(scale, 5)} k={k} "
              f"AIC={2 * k - 2 * logl:.1f}", flush=True)

    out = pd.DataFrame(rows)
    out["dAIC"] = out["AIC"] - out["AIC"].min()
    out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
