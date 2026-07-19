"""Equal-footing test: give the *Poisson* family a free width exponent too.

The free-width Gaussian read-outs carry one extra parameter (the width exponent)
that the Poisson family lacks -- the Poisson well is structurally locked to
Var ∝ n (σ ∝ √n). AIC already penalises that extra parameter, but the cleaner
question is: with the Poisson *shape* but a *free* variance exponent
(Var ∝ n^qvar, qvar = 1 is standard Poisson), does the Poisson family match the
free-width Gaussian? If yes, the well *family* (skewed Poisson PMF vs symmetric
Gaussian) does not matter and it is purely the width scaling; if the Poisson
still trails, the Gaussian shape is genuinely better.

Fits the static and attractor free-variance Poisson per duration and slots them
into the full ladder (all other models already fitted).

    .venv/bin/python -m scripts.trial_level_poisson_freevar
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

OUT = T.LEGACY_OUT_DIR / "poisson_freevar.csv"
_PBOUNDS = ((0.01, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.3, 4.5))  # +qvar

SPECS = (
    T.ModelSpec(
        key="poisson_pow_readout",
        label="Poisson wells, free variance (no clean-up)",
        family="poisson_pow", cleanup=False,
        param_names=("sigma_c", "s0", "lambda0", "qvar"), bounds=_PBOUNDS,
    ),
    T.ModelSpec(
        key="poisson_pow_attractor",
        label="Poisson wells, free variance + clean-up",
        family="poisson_pow", cleanup=True,
        param_names=("sigma_c", "s0", "lambda0", "qvar"), bounds=_PBOUNDS,
    ),
)


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    rows = []
    for spec in SPECS:
        for d in T.DURATIONS:
            rf = T.fit_response(spec, data_by_dur[d], quad_fit, r_max,
                                maxiter=35, popsize=10, seed=T.SEED + d,
                                quad_eval=quad_eval)
            grid = T.compute_grid(spec, rf["theta"], quad_eval)
            # width exponent on sigma: sigma_n ~ n^(qvar/2)
            qvar = float(rf["theta"]["qvar"])
            rows.append({
                "model": spec.label, "model_key": spec.key, "duration_ms": d,
                "qvar": qvar, "sigma_exponent": qvar / 2.0,
                "mean_conv_steps": float(grid["ET"].mean()),
                "lapse": rf["lapse"], "logL": rf["logL"],
                "k_params": rf["k_params"], "AIC": rf["AIC"], "BIC": rf["BIC"],
            })
            print(f"  [{spec.key} @ {d}ms] qvar={qvar:.3f} (sigma~n^{qvar/2:.2f}) "
                  f"E[T]={grid['ET'].mean():.1f} AIC={rf['AIC']:.1f}", flush=True)
    new_df = pd.DataFrame(rows)
    new_df.to_csv(OUT, index=False)

    # ---- full ladder ------------------------------------------------------
    resp = pd.read_csv(T.LEGACY_OUT_DIR / "per_duration_response_likelihood.csv")
    base = resp[resp.model_key.isin(
        ("weber_ans", "hopfield_poisson", "no_cleanup_poisson",
         "gaussian_benford_no_cleanup"))][
        ["model", "model_key", "duration_ms", "k_params", "logL", "AIC"]]
    pieces = [base]
    for f in ("well_width_followup.csv", "freewidth_attractor.csv"):
        p = T.LEGACY_OUT_DIR / f
        if p.exists():
            pieces.append(pd.read_csv(p)[["model", "model_key", "duration_ms",
                                          "k_params", "logL", "AIC"]])
    pieces.append(new_df[["model", "model_key", "duration_ms", "k_params", "logL", "AIC"]])
    allm = pd.concat(pieces, ignore_index=True)
    tot = (allm.groupby(["model_key", "model"])
           .agg(k_total=("k_params", "sum"), AIC_total=("AIC", "sum"))
           .reset_index().sort_values("AIC_total"))
    tot["dAIC"] = tot["AIC_total"] - tot["AIC_total"].min()

    print("\n=== Response-only AIC (summed) — full ladder incl. free-variance Poisson ===")
    print(tot[["model", "k_total", "AIC_total", "dAIC"]].round(1).to_string(index=False))

    pp_static = new_df[new_df.model_key == "poisson_pow_readout"]["AIC"].sum()
    pp_attr = new_df[new_df.model_key == "poisson_pow_attractor"]["AIC"].sum()
    print(f"\nfree-variance Poisson: clean-up effect (static - attractor) = "
          f"{pp_static - pp_attr:+.1f}  (>0 => clean-up helps)")
    print("sigma exponent by duration (qvar/2):")
    print(new_df.pivot_table(index="duration_ms", columns="model_key",
                             values="sigma_exponent").round(2).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
