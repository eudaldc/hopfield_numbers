"""Stage 0: reproduce the current headline results from cached parameters.

Run from the repo root:
    .venv/bin/python driven_attractor/repro_stage0.py [--refit]

Checks, in order:
  1. Winner (gaussian_pow_attractor) per-duration response logL at the cached
     thetas (pow_theta.csv), evaluated with the registered 1601-node grid
     quadrature -> must match freewidth_attractor.csv logL to ~1e-4.
  2. Shared-landscape logL for the winner and its static twin at the cached
     shared thetas -> must match shared_landscape_ladder.csv.
  3. RT channel: re-fit the RT scale parameters at the cached response thetas
     (the registered two-stage procedure) -> must match pow_winner_rt.csv.
  4. (--refit) one from-scratch refit of the winner at 160 ms with the
     registered optimiser settings and seed -> logL within ~1 of the cache.

Writes driven_attractor/output/stage0_reproduction.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
OUT.mkdir(parents=True, exist_ok=True)
TL = ROOT / "output" / "trial_level_model_comparison"
LEGACY_TL = ROOT / "legacy" / "trial_level_per_duration" / "output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="also refit 160 ms from scratch")
    args = ap.parse_args()

    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    quad_eval = T.make_quadrature(1601, kind="grid")
    rows = []

    # ── 1. winner per-duration logL at cached thetas ─────────────────────
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    pow_theta = pd.read_csv(LEGACY_TL / "pow_theta.csv")
    fw = pd.read_csv(LEGACY_TL / "freewidth_attractor.csv")
    fw = fw[fw.model_key == "gaussian_pow_attractor"].set_index("duration_ms")
    print("── winner per-duration response logL (cached theta, eval grid 1601) ──")
    resp_fits = {}
    for _, r in pow_theta.iterrows():
        d = int(r.duration_ms)
        vec = np.array([r.sigma_c, r.s0, r.lambda0, r["pow"], r.lapse])
        ll = T.response_loglik(spec, vec, quad_eval, data[d].counts, r_max)
        target = float(fw.loc[d, "logL"])
        rows.append({"check": f"winner_resp_{d}ms", "value": ll, "target": target,
                     "abs_diff": abs(ll - target)})
        resp_fits[d] = {"theta": {"sigma_c": r.sigma_c, "s0": r.s0,
                                  "lambda0": r.lambda0, "pow": r["pow"]},
                        "lapse": r.lapse}
        print(f"  {d:>3} ms: logL {ll:.4f}  target {target:.4f}  diff {ll-target:+.2e}")

    # ── 2. shared-landscape logL at cached shared thetas ─────────────────
    print("── shared-landscape logL (cached thetas) ──")
    ladder = pd.read_csv(TL / "shared_landscape_ladder.csv").set_index("model_key")
    th = pd.read_csv(TL / "shared_landscape_ladder_theta.csv")
    for key in ("gaussian_pow_attractor", "gaussian_pow_readout"):
        sp = (T.MODEL_BY_KEY[key] if key in T.MODEL_BY_KEY
              else T.ModelSpec(key=key, label=key, family="gaussian_pow", cleanup=False,
                               param_names=("sigma_c", "s0", "lambda0", "pow"),
                               bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0))))
        tot = 0.0
        for _, r in th[th.model_key == key].iterrows():
            d = int(r.duration_ms)
            vec = np.array([r.sigma_c, r.s0, r.lambda0, r["pow"], r.lapse])
            tot += T.response_loglik(sp, vec, quad_eval, data[d].counts, r_max)
        target = float(ladder.loc[key, "shared_logL"])
        rows.append({"check": f"shared_{key}", "value": tot, "target": target,
                     "abs_diff": abs(tot - target)})
        print(f"  {key}: logL {tot:.3f}  target {target:.3f}  diff {tot-target:+.2e}")

    # ── 3. RT channel at cached response thetas ──────────────────────────
    print("── RT channel (two-stage refit at cached thetas) ──")
    rt_target = pd.read_csv(LEGACY_TL / "pow_winner_rt.csv").set_index(["duration_ms", "metric"])
    for d in T.DURATIONS:
        for metric in ("entropy", "steps", "hybrid"):
            rt = T.fit_rt(spec, data[d], resp_fits[d], quad_eval,
                          metric=metric, seed=T.SEED + d)
            target = float(rt_target.loc[(d, metric), "logL"])
            rows.append({"check": f"rt_{metric}_{d}ms", "value": rt["logL"],
                         "target": target, "abs_diff": abs(rt["logL"] - target)})
            print(f"  {d:>3} ms {metric:>8}: logL {rt['logL']:.3f}  target {target:.3f}"
                  f"  diff {rt['logL']-target:+.2e}")

    # ── 4. optional from-scratch refit (registered optimiser settings) ───
    if args.refit:
        print("── from-scratch refit, winner @160 ms (DE maxiter 35, popsize 10, seed 73+160) ──")
        quad_fit = T.make_quadrature(201, kind="grid")
        rf = T.fit_response(spec, data[160], quad_fit, r_max, maxiter=35,
                            popsize=10, seed=T.SEED + 160, quad_eval=quad_eval)
        target = float(fw.loc[160, "logL"])
        rows.append({"check": "refit_160ms", "value": rf["logL"], "target": target,
                     "abs_diff": abs(rf["logL"] - target)})
        print(f"  refit logL {rf['logL']:.3f}  cached {target:.3f}  "
              f"diff {rf['logL']-target:+.3f}")
        print(f"  refit theta: {rf['theta']}  lapse {rf['lapse']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "stage0_reproduction.csv", index=False)
    worst = out.abs_diff.max()
    print(f"\nWorst |diff| = {worst:.4g}   ({'OK' if worst < 2.0 else 'INVESTIGATE'})")
    print(f"Saved {OUT / 'stage0_reproduction.csv'}")


if __name__ == "__main__":
    main()
