"""Direct test of "does online delivery beat batch?" for the Gaussian
families, without an expensive full online fit.

Rationale: the online_attr model nests its matched batch_attr fit as the
tether stiffness beta -> infinity (x(t) -> u(t), so x(d) = u(d)). At finite
beta the landscape's restoring force (~1/sigma_n^2) partially captures the
state during exposure. So online can, at best, tie batch; a beta-sweep at
the batch-fit landscape shows whether any coupling improves on batch.

For online_static the analogue is the tether rate r = beta/tau; r -> inf
reproduces batch_static.

    .venv/bin/python driven_attractor/online_vs_batch.py --family ddm
    .venv/bin/python driven_attractor/online_vs_batch.py --family kalman
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


def score(family, variant, theta, counts, r_max, m):
    ll, _ = E.total_loglik(family, variant, theta, counts, r_max,
                           m_paths=m, quad_nodes=1601)
    return ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="ddm", choices=["ddm", "kalman"])
    ap.add_argument("--m", type=int, default=8192)
    args = ap.parse_args()
    fam = args.family

    dat = D.load()
    r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)

    fa = json.load(open(ROOT / f"driven_attractor/output/fits2/{fam}_batch_attr.json"))
    fs = json.load(open(ROOT / f"driven_attractor/output/fits2/{fam}_batch_static.json"))
    ll_ba = fa["logL"]
    ll_bs = fs["logL"]
    print(f"=== {fam}: online vs batch ===")
    print(f"batch_attr   (reference) logL {ll_ba:.1f}")
    print(f"batch_static (reference) logL {ll_bs:.1f}")

    print("\nonline_attr at the batch_attr landscape, sweeping tether beta "
          "(tau_x=0.05):")
    best_oa = -np.inf
    for beta in (5.0, 50.0, 500.0, 5e3, 5e4):
        th = dict(fa["theta"], tau_x=0.05, beta=beta)
        ll = score(fam, "online_attr", th, counts, r_max, args.m)
        best_oa = max(best_oa, ll)
        print(f"  beta={beta:8.0f}: {ll:11.1f}  (vs batch_attr {ll - ll_ba:+8.1f})",
              flush=True)
    print(f"  best online_attr found: {best_oa - ll_ba:+.1f} vs batch_attr")

    print("\nonline_static at the batch_static landscape, sweeping tether rate r:")
    best_os = -np.inf
    for r in (5.0, 50.0, 500.0, 5e3, 5e4):
        th = dict(fs["theta"], r_tether=r)
        ll = score(fam, "online_static", th, counts, r_max, args.m)
        best_os = max(best_os, ll)
        print(f"  r={r:11.0f}: {ll:11.1f}  (vs batch_static {ll - ll_bs:+8.1f})",
              flush=True)
    print(f"  best online_static found: {best_os - ll_bs:+.1f} vs batch_static")

    print(f"\nCONCLUSION [{fam}]: online delivery "
          f"{'IMPROVES on' if max(best_oa - ll_ba, best_os - ll_bs) > 20 else 'does NOT beat'}"
          " batch (threshold 20 logL over numerical noise).")


if __name__ == "__main__":
    main()
