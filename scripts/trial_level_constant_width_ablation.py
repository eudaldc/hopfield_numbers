"""Fit the p=0 constant-width ablation and its matched static readout.

The full run preserves the headline response-fitting pipeline: a 201-node
optimisation grid, 1601-node likelihood re-evaluation, differential evolution
followed by Nelder--Mead, and a separately fitted lapse for each duration.

    .venv/bin/python -m scripts.trial_level_constant_width_ablation
    .venv/bin/python -m scripts.trial_level_constant_width_ablation --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT = T.OUT_DIR / "constant_width_ablation.csv"
SPECS = (
    T.MODEL_BY_KEY["gaussian_const_attractor"],
    T.MODEL_BY_KEY["gaussian_const_readout"],
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    fit_nodes = 31 if args.smoke else 201
    eval_nodes = 61 if args.smoke else 1601
    maxiter = 2 if args.smoke else 35
    popsize = 4 if args.smoke else 10
    durations = T.DURATIONS[:1] if args.smoke else T.DURATIONS

    quad_fit = T.make_quadrature(fit_nodes, kind="grid")
    quad_eval = T.make_quadrature(eval_nodes, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    rows = []
    tasks = [(spec, d) for spec in SPECS for d in durations]
    for spec, d in tqdm(tasks, desc="Constant-width fits", unit="fit"):
        fit = T.fit_response(
            spec,
            data_by_dur[d],
            quad_fit,
            r_max,
            maxiter=maxiter,
            popsize=popsize,
            seed=T.SEED + d,
            quad_eval=quad_eval,
        )
        grid = T.compute_grid(spec, fit["theta"], quad_eval)
        rows.append(
            {
                "model": spec.label,
                "model_key": spec.key,
                "duration_ms": d,
                "sigma_c": float(fit["theta"]["sigma_c"]),
                "s0": float(fit["theta"]["s0"]),
                "lambda0": float(fit["theta"]["lambda0"]),
                "lapse": fit["lapse"],
                "mean_conv_steps": float(grid["ET"].mean()),
                "logL": fit["logL"],
                "k_params": fit["k_params"],
                "AIC": fit["AIC"],
                "BIC": fit["BIC"],
            }
        )

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))
    if args.smoke:
        return

    result.to_csv(OUT, index=False)
    reference = pd.read_csv(T.OUT_DIR / "freewidth_attractor.csv")
    reference = reference[reference["model_key"] == "gaussian_pow_attractor"]
    ref_aic = float(reference["AIC"].sum())
    ref_bic = float(reference["BIC"].sum())
    totals = (
        result.groupby(["model_key", "model"], as_index=False)
        .agg(k=("k_params", "sum"), logL=("logL", "sum"), AIC=("AIC", "sum"), BIC=("BIC", "sum"))
    )
    totals["dAIC"] = totals["AIC"] - ref_aic
    totals["dBIC"] = totals["BIC"] - ref_bic
    print("\n=== Constant-width ablation relative to power-law attractor ===")
    print(totals.round(1).to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
