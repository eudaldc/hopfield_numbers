"""Fit one (family, variant) of the round-2 evidence models.

    .venv/bin/python driven_attractor/fit_family.py --family ddm --variant batch_attr
    .venv/bin/python driven_attractor/fit_family.py --family kalman --variant online_attr \
        --warm output/fits2/kalman_batch_attr.json

Warm starts: an online variant can be started from the matched batch fit
(shared landscape + evidence parameters), which is both cheaper and the
correct nesting direction (online_attr nests batch_attr as beta -> inf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D  # noqa: E402
from driven_attractor.driven import evidence2 as E  # noqa: E402
from driven_attractor.driven import fitting2 as F2  # noqa: E402

OUT = ROOT / "driven_attractor" / "output" / "fits2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(E.FAMILIES))
    ap.add_argument("--variant", required=True, choices=list(E.VARIANTS))
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=9)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--m-fit", type=int, default=F2.M_FIT)
    ap.add_argument("--m-final", type=int, default=F2.M_FINAL)
    ap.add_argument("--warm", default="", help="batch fit JSON to warm-start from")
    args = ap.parse_args()

    dat = D.load()
    r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)

    x0 = None
    if args.warm:
        wp = Path(args.warm)
        with open(wp if wp.is_absolute() else ROOT / "driven_attractor" / wp) as f:
            w = json.load(f)
        names = E.param_names(args.family, args.variant)
        seed_theta = dict(w["theta"])
        # Start at the nesting limit: a tether far stiffer than the landscape's
        # restoring force (~1/sigma_n^2, up to ~2e3), so x(d) -> u(d) and the
        # online model reproduces its matched batch fit. The optimiser can only
        # move away from that corner if online driving actually helps.
        seed_theta.setdefault("r_tether", 2e4)
        seed_theta.setdefault("tau_x", 0.05)
        seed_theta.setdefault("beta", 2e4)
        x0 = F2.theta_to_vec(names, seed_theta)

    fit = F2.fit(args.family, args.variant, counts, r_max,
                 maxiter=args.maxiter, popsize=args.popsize,
                 workers=args.workers, m_fit=args.m_fit,
                 m_final=args.m_final, x0=x0)
    name = f"{args.family}_{args.variant}.json"
    F2.save(fit, OUT / name)
    print(f"[done] {name}: logL {fit['logL']:.2f} AIC {fit['AIC']:.1f} "
          f"k={fit['k']} wall {fit['wall_min']:.1f} min")
    print(f"  theta: { {k: round(v, 5) for k, v in fit['theta'].items()} }")
    print(f"  lapses: { {k: round(v, 4) for k, v in fit['lapses'].items()} }")


if __name__ == "__main__":
    main()
