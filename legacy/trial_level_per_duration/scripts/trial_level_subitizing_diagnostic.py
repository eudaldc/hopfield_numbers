"""Which super-linear well form captures the subitizing range (esp. n=3) better?

Head-to-head between the two candidate "winner" well shapes, both as attractors:
  * power-law      sigma_n = c * n^p          (gaussian_pow)
  * Benford free-b sigma_n = (c / freq_n)^b   (gaussian_freebeta)

For each, per-numerosity response log-likelihood (summed over durations) and the
predicted vs observed exact accuracy / response distribution, with emphasis on
the subitizing band n = 1..4. Whichever has the higher per-n LL at n=3 (and 1-4)
captures subitizing better. The free-beta theta is reused from winner_theta.csv;
the power-law attractor is refit here and cached to pow_theta.csv.

    .venv/bin/python -m scripts.trial_level_subitizing_diagnostic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT = T.LEGACY_OUT_DIR / "subitizing_diagnostic.csv"
POW_THETA = T.LEGACY_OUT_DIR / "pow_theta.csv"

FREEBETA = T.ModelSpec(
    key="gaussian_freebeta_attractor", label="Benford free-beta + clean-up",
    family="gaussian_freebeta", cleanup=True,
    param_names=("sigma_c", "s0", "lambda0", "beta"),
    bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)))
POW = T.ModelSpec(
    key="gaussian_pow_attractor", label="Power-law n^p + clean-up",
    family="gaussian_pow", cleanup=True,
    param_names=("sigma_c", "s0", "lambda0", "pow"),
    bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0)))


def main() -> None:
    quad_fit = T.make_quadrature(201, kind="grid")
    quad_eval = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    # free-beta theta from cache
    wt = pd.read_csv(T.LEGACY_OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    fb_theta = {d: {"sigma_c": float(wt.loc[d, "sigma_c"]), "s0": float(wt.loc[d, "s0"]),
                    "lambda0": float(wt.loc[d, "lambda0"]), "beta": float(wt.loc[d, "beta"]),
                    "lapse": float(wt.loc[d, "lapse"])} for d in T.DURATIONS}

    # power-law: refit (cache)
    print("  refitting power-law attractor ...", flush=True)
    pow_rows, pw_theta = [], {}
    for d in tqdm(T.DURATIONS, desc="Power-law response fits", unit="duration"):
        rf = T.fit_response(POW, data_by_dur[d], quad_fit, r_max,
                            maxiter=35, popsize=10, seed=T.SEED + d, quad_eval=quad_eval)
        pw_theta[d] = {**rf["theta"], "lapse": rf["lapse"]}
        pow_rows.append({"duration_ms": d, **rf["theta"], "lapse": rf["lapse"],
                         "resp_AIC": rf["AIC"]})
        print(f"    pow @ {d}ms p={rf['theta']['pow']:.3f} AIC={rf['AIC']:.1f}", flush=True)
    pd.DataFrame(pow_rows).to_csv(POW_THETA, index=False)

    # per-n log-likelihood (summed over durations) for each form
    forms = {"power_law": (POW, pw_theta), "benford_freebeta": (FREEBETA, fb_theta)}
    per_n = {name: np.zeros(T.N_SHOWN.size) for name in forms}
    pred_acc = {name: np.zeros(T.N_SHOWN.size) for name in forms}
    obs_acc = np.zeros(T.N_SHOWN.size)
    obs_count = np.zeros(T.N_SHOWN.size)
    rows = []
    for d in T.DURATIONS:
        data = data_by_dur[d]
        counts = data.counts                          # (15, r_max)
        for name, (spec, th_by_d) in forms.items():
            th = th_by_d[d]
            grid = T.compute_grid(spec, th, quad_eval)
            pf = T.apply_lapse(grid["p_resp"], th["lapse"], r_max)
            ll_n = (counts * np.log(pf)).sum(axis=1)   # per-n LL contribution
            per_n[name] += ll_n
            pred_acc[name] += np.array([pf[i, int(n) - 1] * counts[i].sum()
                                        for i, n in enumerate(T.N_SHOWN)])
            for i, n in enumerate(T.N_SHOWN):
                rows.append({"duration_ms": d, "form": name, "n_shown": int(n),
                             "ll": float(ll_n[i]),
                             "pred_pexact": float(pf[i, int(n) - 1])})
        obs_acc += counts[np.arange(T.N_SHOWN.size), (T.N_SHOWN - 1)]
        obs_count += counts.sum(axis=1)
    for name in forms:
        pred_acc[name] /= obs_count
    obs_acc_rate = obs_acc / obs_count
    pd.DataFrame(rows).to_csv(OUT, index=False)

    print("\n=== Per-n response log-likelihood (summed over durations) ===")
    print("  n  |   power_law      benford_fb   |  dLL(pow-fb)  | obs_acc | pred_pow pred_fb")
    for i, n in enumerate(T.N_SHOWN):
        dll = per_n["power_law"][i] - per_n["benford_freebeta"][i]
        star = "  <--subitizing" if n <= 4 else ""
        print(f"  {int(n):2d} | {per_n['power_law'][i]:12.1f}  {per_n['benford_freebeta'][i]:12.1f}"
              f" | {dll:+10.1f}  | {obs_acc_rate[i]:6.3f}  | "
              f"{pred_acc['power_law'][i]:6.3f}  {pred_acc['benford_freebeta'][i]:6.3f}{star}")

    sub = slice(0, 4)  # n=1..4
    print("\nSubitizing band (n=1..4) total response LL:")
    print(f"  power_law       : {per_n['power_law'][sub].sum():.1f}")
    print(f"  benford_freebeta: {per_n['benford_freebeta'][sub].sum():.1f}")
    print(f"  -> dLL(pow - fb) over n=1..4 = {per_n['power_law'][sub].sum() - per_n['benford_freebeta'][sub].sum():+.1f}")
    print(f"n=3 alone dLL(pow - fb) = {per_n['power_law'][2] - per_n['benford_freebeta'][2]:+.1f}")
    print(f"\nSaved {OUT} and {POW_THETA}")


if __name__ == "__main__":
    main()
