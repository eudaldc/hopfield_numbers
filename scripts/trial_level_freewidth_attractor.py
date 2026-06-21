"""Does an attractor with *free-width* wells beat everything?

The follow-up (`trial_level_well_width_followup.py`) showed that *static*
Gaussian read-outs with a free width exponent (β / p ≈ 1–2, not the √n of the
Poisson / fixed-Benford wells) dominate the trial-level response comparison. The
separate finding was that *clean-up* helps within each well family. The natural
combined model is therefore an **attractor (clean-up) with free-width wells**:
clean-up *and* the width the data actually want.

This fits the clean-up versions of the two free-width Gaussian families and
compares them, on the same trial-level response likelihood, against:
  * the static free-width read-outs (does clean-up add anything on top?),
  * Hopfield-Poisson and the no-clean-up baselines,
  * Weber/ANS.

E[T_conv] (mean clean-up steps) is reported to confirm the dynamics actually run
when the large-n wells are very wide.

    .venv/bin/python -m scripts.trial_level_freewidth_attractor
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

OUT = T.OUT_DIR / "freewidth_attractor.csv"

ATTRACTOR_SPECS = (
    T.ModelSpec(
        key="gaussian_pow_attractor",
        label="Power-law wells + clean-up (sigma_n = c n^p)",
        family="gaussian_pow",
        cleanup=True,
        param_names=("sigma_c", "s0", "lambda0", "pow"),
        bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0)),
    ),
    T.ModelSpec(
        key="gaussian_freebeta_attractor",
        label="Benford wells, free beta + clean-up",
        family="gaussian_freebeta",
        cleanup=True,
        param_names=("sigma_c", "s0", "lambda0", "beta"),
        bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
    ),
)


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    rows = []
    for spec in ATTRACTOR_SPECS:
        for d in T.DURATIONS:
            rf = T.fit_response(spec, data_by_dur[d], quad_fit, r_max,
                                maxiter=35, popsize=10, seed=T.SEED + d,
                                quad_eval=quad_eval)
            free = spec.param_names[-1]
            grid = T.compute_grid(spec, rf["theta"], quad_eval)
            et = float(grid["ET"].mean())
            rows.append({
                "model": spec.label, "model_key": spec.key, "duration_ms": d,
                "free_param": free, "free_value": float(rf["theta"][free]),
                "mean_conv_steps": et, "lapse": rf["lapse"], "logL": rf["logL"],
                "k_params": rf["k_params"], "AIC": rf["AIC"], "BIC": rf["BIC"],
            })
            print(f"  [{spec.key} @ {d}ms] {free}={rf['theta'][free]:.3f} "
                  f"E[T]={et:.1f} AIC={rf['AIC']:.1f} logL={rf['logL']:.1f}", flush=True)
    new_df = pd.DataFrame(rows)
    new_df.to_csv(OUT, index=False)

    # ---- merge with everything already fitted -----------------------------
    resp = pd.read_csv(T.OUT_DIR / "per_duration_response_likelihood.csv")
    base_keys = ("weber_ans", "hopfield_poisson", "no_cleanup_poisson",
                 "gaussian_benford_no_cleanup")
    base = resp[resp.model_key.isin(base_keys)][
        ["model", "model_key", "duration_ms", "k_params", "logL", "AIC"]]
    ww = pd.read_csv(T.OUT_DIR / "well_width_followup.csv")[
        ["model", "model_key", "duration_ms", "k_params", "logL", "AIC"]]
    allm = pd.concat(
        [base, ww, new_df[["model", "model_key", "duration_ms", "k_params", "logL", "AIC"]]],
        ignore_index=True)
    tot = (allm.groupby(["model_key", "model"])
           .agg(k_total=("k_params", "sum"), logL_total=("logL", "sum"),
                AIC_total=("AIC", "sum"))
           .reset_index().sort_values("AIC_total"))
    tot["dAIC"] = tot["AIC_total"] - tot["AIC_total"].min()

    print("\n=== Response-only AIC (summed over durations) — full ladder ===")
    print(tot[["model", "k_total", "AIC_total", "dAIC"]].round(1).to_string(index=False))

    print("\n=== Free-width attractor: fitted exponent, mean clean-up steps, "
          "vs its static twin ===")
    twin = {"gaussian_pow_attractor": "gaussian_pow_readout",
            "gaussian_freebeta_attractor": "gaussian_benford_free_beta"}
    for spec in ATTRACTOR_SPECS:
        a = new_df[new_df.model_key == spec.key]
        st = ww[ww.model_key == twin[spec.key]]
        d_aic = float(a["AIC"].sum() - st["AIC"].sum())
        print(f"\n{spec.key}:")
        print(f"  exponent by dur: {a['free_value'].round(2).tolist()}")
        print(f"  E[T] by dur:     {a['mean_conv_steps'].round(1).tolist()}")
        print(f"  clean-up effect vs static twin (AIC_static - AIC_attractor): "
              f"{st['AIC'].sum() - a['AIC'].sum():+.1f}  (>0 => clean-up helps)")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
