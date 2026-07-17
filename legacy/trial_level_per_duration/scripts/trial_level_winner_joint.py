"""Joint (response + RT) for the winning free-width attractor, with the RT
channel driven by the clean-up's convergence dynamics.

RT-only exploration showed that on the super-linear landscape the convergence
step-count beats entropy and steps+entropy beats it further. Here we close the
loop with the coupled joint p(r, rt | n): response (the winner's best-in-class
channel) + conditional RT, comparing an entropy RT against the dynamics
(hybrid = steps + entropy). The winner's response fit is recomputed once and its
theta cached to ``winner_theta.csv`` for reuse.

    .venv/bin/python -m scripts.trial_level_winner_joint
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

OUT = T.LEGACY_OUT_DIR / "winner_joint.csv"
THETA_OUT = T.LEGACY_OUT_DIR / "winner_theta.csv"

WINNER = T.ModelSpec(
    key="gaussian_freebeta_attractor",
    label="Benford wells, free beta + clean-up",
    family="gaussian_freebeta", cleanup=True,
    param_names=("sigma_c", "s0", "lambda0", "beta"),
    bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
)


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(1601, kind="grid")
    quad_joint = T.make_quadrature(101, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    theta_rows, joint_rows = [], []
    for d in T.DURATIONS:
        rf = T.fit_response(WINNER, data_by_dur[d], quad_fit, r_max,
                            maxiter=35, popsize=10, seed=T.SEED + d,
                            quad_eval=quad_eval)
        theta_rows.append({"duration_ms": d, **rf["theta"], "lapse": rf["lapse"],
                           "resp_logL": rf["logL"], "resp_AIC": rf["AIC"]})
        for metric in ("entropy", "hybrid"):
            jt = T.fit_joint(WINNER, data_by_dur[d], rf, quad_joint, r_max,
                             metric=metric, seed=T.SEED + d)
            joint_rows.append({"duration_ms": d, "metric": metric,
                               "resp_logL": jt["resp_logL"],
                               "cond_rt_logL": jt["cond_rt_logL"],
                               "logL": jt["logL"], "k_params": jt["k_params"],
                               "AIC": jt["AIC"]})
            print(f"  [{d}ms {metric}] joint AIC={jt['AIC']:.1f} "
                  f"(resp {jt['resp_logL']:.0f} + rt {jt['cond_rt_logL']:.0f})", flush=True)
    pd.DataFrame(theta_rows).to_csv(THETA_OUT, index=False)
    jdf = pd.DataFrame(joint_rows)
    jdf.to_csv(OUT, index=False)

    win = jdf.groupby("metric").agg(AIC_total=("AIC", "sum"),
                                    k_total=("k_params", "sum")).reset_index()
    print("\n=== Winner joint (summed over durations) ===")
    print(win.round(1).to_string(index=False))

    # slot into the entropy-joint ladder of the original six models
    prev = pd.read_csv(T.LEGACY_OUT_DIR / "per_duration_joint_likelihood.csv")
    prev = prev[prev.metric == "entropy"].groupby(["model_key"]).agg(
        AIC_total=("AIC", "sum"), k_total=("k_params", "sum")).reset_index()
    prev["model_key"] = prev["model_key"]
    rows = prev.to_dict("records")
    for _, r in win.iterrows():
        rows.append({"model_key": f"winner_{r['metric']}",
                     "AIC_total": r["AIC_total"], "k_total": int(r["k_total"])})
    lad = pd.DataFrame(rows).sort_values("AIC_total")
    lad["dAIC"] = lad["AIC_total"] - lad["AIC_total"].min()
    print("\n=== Joint ladder (entropy joints of the 6 originals + winner) ===")
    print(lad.round(1).to_string(index=False))
    print(f"\nSaved {OUT} and {THETA_OUT}")


if __name__ == "__main__":
    main()
