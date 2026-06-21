"""Re-compare models with a FULL joint fit (landscape + lapse + RT co-optimised
on response + RT together), instead of response-first / RT-second.

This answers: when RT is fit jointly with the landscape (not as a downstream
channel), (a) how much does the RT fit recover, and (b) does the winning
super-linear attractor still win the joint comparison?

Models (attractors use the dynamics RT = hybrid steps+entropy; static read-outs
use entropy):
  gaussian_freebeta_attractor (winner) · hopfield_poisson (√n) ·
  gaussian_freebeta_no_cleanup · weber_ans · no_cleanup_poisson

    .venv/bin/python -m scripts.trial_level_joint_refit
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

OUT = T.OUT_DIR / "joint_refit.csv"
THETA_OUT = T.OUT_DIR / "joint_refit_theta.csv"
FIG2_RT_R2 = {40: 0.91, 80: 0.98, 160: 0.98, 320: 0.93, 640: 0.89}

# (model_key, RT metric); attractors -> hybrid (dynamics), statics -> entropy
PLAN = [
    ("gaussian_freebeta_attractor", "hybrid"),
    ("hopfield_poisson", "hybrid"),
    ("gaussian_freebeta_no_cleanup", "entropy"),
    ("weber_ans", "entropy"),
    ("no_cleanup_poisson", "entropy"),
]


def r2(obs, pred):
    return 1.0 - np.sum((obs - pred) ** 2) / (np.sum((obs - obs.mean()) ** 2) + 1e-12)


def cached_resp_fits() -> dict:
    """Cached response-only fits per (model_key, duration): {theta, lapse}.

    Reuses winner_theta.csv (winner) and per_duration_response_likelihood.csv
    (the six original models) so the joint warm-start skips re-fitting the
    expensive attractor landscapes. Models not cached are refit (cheap statics)."""
    out: dict = {}
    # logL/k_params are dummies: fit_joint_full only uses the RT params from the
    # stage-2 conditional fit, not these fields.
    wt = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    for d in T.DURATIONS:
        out[("gaussian_freebeta_attractor", d)] = {
            "theta": {c: float(wt.loc[d, c]) for c in ("sigma_c", "s0", "lambda0", "beta")},
            "lapse": float(wt.loc[d, "lapse"]), "logL": 0.0, "k_params": 0}
    resp = pd.read_csv(T.OUT_DIR / "per_duration_response_likelihood.csv")
    for _, r in resp.iterrows():
        spec = T.MODEL_BY_KEY[r["model_key"]]
        out[(r["model_key"], int(r["duration_ms"]))] = {
            "theta": {p: float(r[f"theta_{p}"]) for p in spec.param_names},
            "lapse": float(r["lapse"]), "logL": 0.0, "k_params": 0}
    return out


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(801, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)
    cache = cached_resp_fits()

    rows, theta_rows = [], []
    for key, metric in PLAN:
        spec = T.MODEL_BY_KEY[key]
        for d in T.DURATIONS:
            fj = T.fit_joint_full(spec, data_by_dur[d], quad_fit, r_max,
                                  metric=metric, maxiter=30, popsize=8,
                                  seed=T.SEED + d, quad_eval=quad_eval,
                                  resp_fit=cache.get((key, d)))
            rows.append({"model_key": key, "metric": metric, "duration_ms": d,
                         "resp_logL": fj["resp_logL"], "cond_rt_logL": fj["cond_rt_logL"],
                         "logL": fj["logL"], "k_params": fj["k_params"],
                         "AIC": fj["AIC"]})
            theta_rows.append({"model_key": key, "duration_ms": d, **fj["theta"],
                               "lapse": fj["lapse"],
                               **{f"rt{i}": v for i, v in enumerate(fj["rt_params"])}})
            print(f"  [{key} {metric} @ {d}ms] joint AIC={fj['AIC']:.1f} "
                  f"(resp {fj['resp_logL']:.0f} + rt {fj['cond_rt_logL']:.0f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.DataFrame(theta_rows).to_csv(THETA_OUT, index=False)

    lad = df.groupby(["model_key", "metric"]).agg(
        AIC_total=("AIC", "sum"), k_total=("k_params", "sum")).reset_index()
    lad = lad.sort_values("AIC_total")
    lad["dAIC"] = lad["AIC_total"] - lad["AIC_total"].min()
    print("\n=== JOINT-REFIT ladder (response + RT co-fit), summed AIC ===")
    print(lad.round(1).to_string(index=False))

    # winner RT recovery: median-RT-curve R^2 from the JOINT-fit params
    print("\n=== Winner (joint-fit) median-RT R^2 vs published Fig.2 ===")
    th = pd.DataFrame(theta_rows)
    th = th[th.model_key == "gaussian_freebeta_attractor"].set_index("duration_ms")
    win = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
    obs_all, pred_all = [], []
    for d in T.DURATIONS:
        theta = {c: float(th.loc[d, c]) for c in ("sigma_c", "s0", "lambda0", "beta")}
        g = T.compute_grid(win, theta, quad_eval)
        alpha, beta_rt = float(th.loc[d, "rt0"]), float(th.loc[d, "rt1"])
        pred = T.T0 + alpha * g["ET"] + beta_rt * g["EH"]
        data = data_by_dur[d]
        obs = np.array([np.median(np.exp(data.logrt_trial[data.n_trial == n]))
                        for n in T.N_SHOWN])
        print(f"  {d}ms: joint R²={r2(obs, pred):.3f}  "
              f"(separate-fit ~0.6-0.77; Fig2 {FIG2_RT_R2[d]:.2f})")
        obs_all.append(obs); pred_all.append(pred)
    obs_all, pred_all = np.concatenate(obs_all), np.concatenate(pred_all)
    print(f"  pooled joint R²={r2(obs_all, pred_all):.3f}")
    print(f"\nSaved {OUT} and {THETA_OUT}")


if __name__ == "__main__":
    main()
