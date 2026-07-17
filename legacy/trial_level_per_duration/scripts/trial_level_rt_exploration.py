"""Can the clean-up *dynamics* drive RT better than posterior entropy?

The headline RT result was negative: raw clean-up step-count fit far worse than
posterior entropy. Hypothesis for *why*: the raw step count is dominated by the
``max_step`` clamp (steep regions forced into many tiny steps), not by genuine
difficulty. The fix is the continuous **gradient-flow convergence time**
``integral dx / (eta|pull|)`` (``ET_flow``), which discounts clamped steps and
diverges near basin boundaries (critical slowing down) — a proper first-passage
RT. And the *well shape* matters: on the data-preferred super-linear landscape
the dynamics may track RT where the √n Poisson did not.

For each attractor landscape this fits the RT-only channel with every metric
(entropy / raw steps / flow-time / steps+entropy / flow+entropy) and reports AIC
(summed over durations), so we can see whether any *dynamical* RT beats entropy.

Landscapes:
  * hopfield_poisson, gaussian_benford_attractor  -- theta reloaded (fast)
  * gaussian_freebeta_attractor (the winner)      -- response refit here

    .venv/bin/python -m scripts.trial_level_rt_exploration
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

OUT = T.LEGACY_OUT_DIR / "rt_exploration.csv"
METRICS = ("entropy", "steps", "flowtime", "hybrid", "flow_entropy")

# winner (refit here to recover full theta)
WINNER = T.ModelSpec(
    key="gaussian_freebeta_attractor",
    label="Benford wells, free beta + clean-up",
    family="gaussian_freebeta", cleanup=True,
    param_names=("sigma_c", "s0", "lambda0", "beta"),
    bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
)


def theta_from_saved(spec: T.ModelSpec, resp: pd.DataFrame, d: int) -> dict:
    row = resp[(resp.model_key == spec.key) & (resp.duration_ms == d)].iloc[0]
    theta = {p: float(row[f"theta_{p}"]) for p in spec.param_names}
    return {"theta": theta, "lapse": float(row["lapse"])}


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)
    resp = pd.read_csv(T.LEGACY_OUT_DIR / "per_duration_response_likelihood.csv")

    saved = [T.MODEL_BY_KEY["hopfield_poisson"],
             T.MODEL_BY_KEY["gaussian_benford_attractor"]]

    rows = []

    def run_landscape(spec, resp_fit_by_dur, tag):
        local = []
        for d in T.DURATIONS:
            rf = resp_fit_by_dur[d]
            for metric in METRICS:
                rt = T.fit_rt(spec, data_by_dur[d], rf, quad_eval,
                              metric=metric, seed=T.SEED + d)
                rec = {"landscape": tag, "model_key": spec.key,
                       "duration_ms": d, "metric": metric,
                       "logL": rt["logL"], "k": rt["k_params"],
                       "AIC": rt["AIC"], **rt["scale"]}
                rows.append(rec)
                local.append(rec)
        sub = pd.DataFrame(local).groupby("metric")["AIC"].sum()
        ent = sub["entropy"]
        print(f"\n[{tag}] RT-only AIC by metric (summed); ΔAIC vs entropy:", flush=True)
        for m in METRICS:
            flag = "   <-- beats entropy" if sub[m] < ent - 2 else ""
            print(f"  {m:13s} AIC={sub[m]:10.1f}  dAIC={sub[m]-ent:+8.1f}{flag}", flush=True)

    for spec in saved:
        rfbd = {d: theta_from_saved(spec, resp, d) for d in T.DURATIONS}
        run_landscape(spec, rfbd, spec.key)

    # winner: refit response to recover theta
    print("  refitting winner (gaussian_freebeta_attractor) ...", flush=True)
    rfbd = {}
    for d in T.DURATIONS:
        rfbd[d] = T.fit_response(WINNER, data_by_dur[d], quad_fit, r_max,
                                 maxiter=35, popsize=10, seed=T.SEED + d,
                                 quad_eval=quad_eval)
    run_landscape(WINNER, rfbd, WINNER.key)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n=== RT-only AIC by metric (summed over durations); ΔAIC vs entropy ===")
    for tag in df.landscape.unique():
        sub = df[df.landscape == tag].groupby("metric")["AIC"].sum()
        ent = sub["entropy"]
        print(f"\n{tag}:")
        for m in METRICS:
            print(f"  {m:13s} AIC={sub[m]:10.1f}  dAIC_vs_entropy={sub[m]-ent:+8.1f}"
                  f"{'   <-- beats entropy' if sub[m] < ent - 2 else ''}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
