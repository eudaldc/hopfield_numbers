"""Stage 7: RT channel comparison, driven model vs M0, identical convention.

    .venv/bin/python driven_attractor/fit_rt_driven.py --fit output/fits/M2_tether_g0.json

Two-stage logic (as in the paper): landscape/evidence parameters held at the
response-fit values; only the RT link is fitted. Likelihood: log rt ~
Normal(log(t0 + a*E[T] + b*E[H]), sigma), t0 = 1.30 s, sigma profiled.

Rows produced (all with a SHARED mapping across durations, plus per-duration
variants for reference):
  driven settle hybrid / dynamics-only / entropy-only / physical (a = 1)
  driven entrate hybrid (diminishing-information stopping)
  M0 per-duration landscape, steps+entropy hybrid / entropy-only
  M0 shared landscape, steps+entropy hybrid / entropy-only

The driven E[T] is the post-offset settling time in SECONDS -- RT in this
dataset is timed from stimulus offset (stage0_audit), so 'physical' (a = 1)
is the maximally constrained, directly falsifiable link.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import model as M  # noqa: E402
from driven_attractor.driven import rt as RT  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"


def m0_rt_ingredients(which: str) -> tuple[np.ndarray, np.ndarray]:
    """(E[steps], E[H]) per (duration, n) at the registered M0 fits."""
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")
    eT = np.zeros((5, 15))
    eH = np.zeros((5, 15))
    if which == "perdur":
        th = pd.read_csv(ROOT / "output/trial_level_model_comparison/pow_theta.csv")
        rows = {int(r.duration_ms): {"sigma_c": r.sigma_c, "s0": r.s0,
                                     "lambda0": r.lambda0, "pow": r["pow"]}
                for _, r in th.iterrows()}
    else:
        th = pd.read_csv(ROOT / "output/trial_level_model_comparison/shared_landscape_ladder_theta.csv")
        th = th[th.model_key == "gaussian_pow_attractor"]
        rows = {int(r.duration_ms): {"sigma_c": r.sigma_c, "s0": r.s0,
                                     "lambda0": r.lambda0, "pow": r["pow"]}
                for _, r in th.iterrows()}
    for i, d in enumerate(M.DURATIONS):
        grid = T.compute_grid(spec, rows[d], quad)
        eT[i] = grid["ET"]
        eH[i] = grid["EH"]
    return eT, eH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--m-paths", type=int, default=8192)
    args = ap.parse_args()
    fp = Path(args.fit)
    with open(fp if fp.is_absolute() else ROOT / "driven_attractor" / fp) as f:
        fit = json.load(f)
    tag = fp.stem

    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)

    rows = []

    # driven-model ingredients
    rs = RT.rt_summaries(fit["model"], fit["theta"], m_paths=args.m_paths)
    menu = [("settle", "hybrid"), ("settle", "dynamics"), ("settle", "entropy"),
            ("settle", "physical")]
    if "entrate_T" in rs:
        menu += [("entrate", "hybrid"), ("entrate", "physical")]
    for rule, variant in menu:
        eT = rs["settle_T"] if rule == "settle" else rs["entrate_T"]
        for shared in (True, False):
            r = RT.fit_rt_link(eT, rs["H"], data, variant=variant, shared=shared)
            rows.append({"source": f"driven:{tag}", "rule": rule, **r,
                         "params": json.dumps(r["params"])})
            print(f"driven {rule:>7} {variant:>9} shared={shared}: "
                  f"logL {r['logL']:.1f} AIC {r['AIC']:.1f} k={r['k']}", flush=True)

    # M0 references under the identical fitting code
    for which in ("perdur", "shared"):
        eT, eH = m0_rt_ingredients(which)
        for variant in ("hybrid", "dynamics", "entropy"):
            for shared in (True, False):
                r = RT.fit_rt_link(eT, eH, data, variant=variant, shared=shared)
                rows.append({"source": f"M0_{which}", "rule": "steps", **r,
                             "params": json.dumps(r["params"])})
                print(f"M0[{which}] {variant:>9} shared={shared}: "
                      f"logL {r['logL']:.1f} AIC {r['AIC']:.1f} k={r['k']}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"rt_comparison_{tag}.csv", index=False)
    print(f"saved {OUT / f'rt_comparison_{tag}.csv'}")


if __name__ == "__main__":
    main()
