"""Follow-up: does the readout *well-width scaling* explain why Weber/ANS beats
the Gaussian-Benford / Poisson static read-outs at the trial level?

The 2x2 fixes the Gaussian-Benford width exponent at beta = 0.5, which makes the
well width grow as ~sqrt(n) -- the same scaling as the Poisson family -- whereas
Weber/ANS uses a linear (constant-CoV) width sigma_n proportional to n. This
script fits two *static* (no clean-up) Gaussian read-outs that free that scaling:

  * gaussian_benford_free_beta : sigma_n = (sigma_c / freq_n)^beta, beta free
  * gaussian_pow_readout       : sigma_n = sigma_c * n^p,           p free (=1 -> Weber)

and compares their trial-level response AIC against the already-fitted baselines
in ``per_duration_response_likelihood.csv``. Prediction: the freed width moves
toward linear (beta high / p ~ 1) and the AIC closes most of the gap to Weber.

    .venv/bin/python -m scripts.trial_level_well_width_followup
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT = T.OUT_DIR / "well_width_followup.csv"

NEW_SPECS = (
    T.ModelSpec(
        key="gaussian_benford_free_beta",
        label="Gaussian-Benford readout, free beta",
        family="gaussian_freebeta",
        cleanup=False,
        param_names=("sigma_c", "s0", "lambda0", "beta"),
        # widened beta upper bound to locate where the width exponent settles
        bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
    ),
    T.ModelSpec(
        key="gaussian_pow_readout",
        label="Gaussian power-law readout (sigma_n = c n^p)",
        family="gaussian_pow",
        cleanup=False,
        param_names=("sigma_c", "s0", "lambda0", "pow"),
        # widened pow upper bound; sigma_c can shrink to keep small-n wells tight
        bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0)),
    ),
)

BASELINES = ("weber_ans", "gaussian_benford_no_cleanup",
             "no_cleanup_poisson", "hopfield_poisson")


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)
    durations = list(T.DURATIONS)

    rows = []
    for spec in NEW_SPECS:
        for d in durations:
            rf = T.fit_response(spec, data_by_dur[d], quad_fit, r_max,
                                maxiter=35, popsize=10, seed=T.SEED + d,
                                quad_eval=quad_eval)
            free = spec.param_names[-1]  # 'beta' or 'pow'
            rows.append({
                "model": spec.label, "model_key": spec.key, "duration_ms": d,
                "free_param": free, "free_value": float(rf["theta"][free]),
                "lapse": rf["lapse"], "logL": rf["logL"],
                "k_params": rf["k_params"], "AIC": rf["AIC"], "BIC": rf["BIC"],
            })
            print(f"  [{spec.key} @ {d}ms] {free}={rf['theta'][free]:.3f} "
                  f"AIC={rf['AIC']:.1f} logL={rf['logL']:.1f}", flush=True)
    new_df = pd.DataFrame(rows)
    new_df.to_csv(OUT, index=False)

    # --- merge with saved baselines and summarise -------------------------
    saved = pd.read_csv(T.OUT_DIR / "per_duration_response_likelihood.csv")
    base = saved[saved.model_key.isin(BASELINES)][
        ["model", "model_key", "duration_ms", "k_params", "logL", "AIC"]
    ]
    allm = pd.concat([base, new_df[["model", "model_key", "duration_ms",
                                    "k_params", "logL", "AIC"]]], ignore_index=True)
    tot = (allm.groupby(["model_key", "model"])
           .agg(k_total=("k_params", "sum"), logL_total=("logL", "sum"),
                AIC_total=("AIC", "sum"))
           .reset_index().sort_values("AIC_total"))
    tot["dAIC"] = tot["AIC_total"] - tot["AIC_total"].min()

    print("\n=== Response-only AIC (summed over durations), with width-freed variants ===")
    print(tot[["model", "k_total", "AIC_total", "dAIC"]].round(1).to_string(index=False))

    print("\n=== Fitted width exponent by duration ===")
    piv = new_df.pivot_table(index="duration_ms", columns="model_key", values="free_value")
    print(piv.round(3).to_string())
    print("\n(reference: fixed Gaussian-Benford beta = 0.5 -> width ~ n^0.45; "
          "Weber linear -> p = 1.0)")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
