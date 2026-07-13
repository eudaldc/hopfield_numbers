"""Power-law winner: RT link fit on the fixed response landscape + n_crit.

The winning model is now the power-law-width Gaussian attractor
(``gaussian_pow_attractor``, sigma_n = sigma_c * n^p, free exponent). Its
response landscape is the cached two-stage fit in ``pow_theta.csv`` (identical to
``response_per_duration_aic.csv``). Here we:

  1. fit the RT link (entropy / steps / hybrid) on that fixed landscape, so RT is
     a genuine cross-prediction -> ``pow_winner_rt.csv`` (alpha=steps coeff,
     beta=entropy coeff for the hybrid);
  2. derive the resolution limit n_crit from sigma_n = sigma_c * n^p = 1, i.e.
     n_crit = sigma_c^(-1/p) -> ``pow_winner_ncrit.csv``.

    .venv/bin/python -m scripts.trial_level_pow_winner
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

WINNER = T.WINNER_KEY  # gaussian_pow_attractor
METRICS = ("entropy", "steps", "hybrid")
THETA_CSV = T.OUT_DIR / "pow_theta.csv"
OUT_RT = T.OUT_DIR / "pow_winner_rt.csv"
OUT_NCRIT = T.OUT_DIR / "pow_winner_ncrit.csv"


def main() -> None:
    spec = T.MODEL_BY_KEY[WINNER]
    quad_eval = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    theta_df = pd.read_csv(THETA_CSV).set_index("duration_ms")

    rt_rows, ncrit_rows = [], []
    for d in T.DURATIONS:
        row = theta_df.loc[d]
        theta = {p: float(row[p]) for p in spec.param_names}
        lapse = float(row["lapse"])
        resp_fit = {"theta": theta, "lapse": lapse}

        for metric in METRICS:
            rt = T.fit_rt(spec, data_by_dur[d], resp_fit, quad_eval,
                          metric=metric, seed=T.SEED + d)
            rec = {"model_key": WINNER, "duration_ms": d, "metric": metric,
                   "logL": rt["logL"], "k": rt["k_params"], "AIC": rt["AIC"],
                   "alpha": np.nan, "beta": np.nan, "rt_scale": np.nan}
            rec.update(rt["scale"])
            rt_rows.append(rec)

        # resolution limit: sigma_n = sigma_c * n^p = 1  ->  n_crit = sigma_c^(-1/p)
        sc, p = theta["sigma_c"], theta["pow"]
        n_crit = float(sc ** (-1.0 / p))
        ncrit_rows.append({"duration_ms": d, "sigma_c": sc, "pow": p,
                           "n_crit": n_crit})

    rt_df = pd.DataFrame(rt_rows)
    rt_df.to_csv(OUT_RT, index=False)
    ncrit_df = pd.DataFrame(ncrit_rows)
    ncrit_df.to_csv(OUT_NCRIT, index=False)

    # summary: RT-only AIC by metric, hybrid vs entropy
    print(f"Saved {OUT_RT}\nSaved {OUT_NCRIT}\n")
    piv = rt_df.pivot(index="duration_ms", columns="metric", values="AIC")
    tot = piv.sum()
    print("RT-only AIC summed over durations (dAIC vs entropy):")
    for m in METRICS:
        print(f"  {m:8s} AIC={tot[m]:10.1f}  dAIC={tot[m]-tot['entropy']:+8.1f}")
    print("\nn_crit (sigma_n = 1) per duration:")
    for _, r in ncrit_df.iterrows():
        print(f"  {int(r.duration_ms):3d} ms: p={r['pow']:.2f}  "
              f"sigma_c={r['sigma_c']:.4f}  n_crit={r['n_crit']:.2f}")


if __name__ == "__main__":
    main()
