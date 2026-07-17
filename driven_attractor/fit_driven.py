"""Fit one driven-family model variant to the trial-level count tables.

Run from the repo root, e.g.:
    .venv/bin/python driven_attractor/fit_driven.py --model M2_tether --gamma 0
    .venv/bin/python driven_attractor/fit_driven.py --model M2_tether_const --gamma 0
    .venv/bin/python driven_attractor/fit_driven.py --model M1_offset --gamma 1
    .venv/bin/python driven_attractor/fit_driven.py --model M2_tether --gamma 0 --free-delta

Saves JSON to driven_attractor/output/fits/<model>_g<gamma><suffix>.json.
Seeds fixed; likelihood deterministic (fixed Sobol CRN).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import fitting as F  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output" / "fits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(M.MODEL_KEYS))
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--free-gamma", action="store_true")
    ap.add_argument("--free-delta", action="store_true")
    ap.add_argument("--maxiter", type=int, default=45)
    ap.add_argument("--popsize", type=int, default=9)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--m-fit", type=int, default=F.M_FIT)
    ap.add_argument("--m-final", type=int, default=F.M_FINAL)
    ap.add_argument("--seed", type=int, default=F.FIT_SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--warm-start", default="",
                    help="path to a previous fit JSON to warm-start (skips DE)")
    args = ap.parse_args()

    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    counts = {d: data[d].counts for d in M.DURATIONS}

    x0 = None
    if args.warm_start:
        import json
        with open(args.warm_start) as f:
            x0 = np.asarray(json.load(f)["x_opt"], float)

    fit = F.fit_model(args.model, counts, r_max, gamma=args.gamma,
                      free_gamma=args.free_gamma, free_delta=args.free_delta,
                      maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
                      m_fit=args.m_fit, m_final=args.m_final,
                      workers=args.workers, x0=x0)
    g = "free" if args.free_gamma else f"{args.gamma:g}"
    name = (f"{args.model}_g{g}{'_delta' if args.free_delta else ''}"
            f"{('_' + args.tag) if args.tag else ''}.json")
    F.save_fit(fit, OUT / name)
    print(f"[done] {name}: logL {fit['logL']:.2f} (fit-grid {fit['logL_fit']:.2f}) "
          f"AIC {fit['AIC']:.1f} k={fit['k']}  wall {fit['wall_min']:.1f} min")
    print(f"  theta: { {k: round(v, 5) for k, v in fit['theta'].items()} }")
    print(f"  lapses: { {k: round(v, 4) for k, v in fit['lapses'].items()} }")


if __name__ == "__main__":
    main()
