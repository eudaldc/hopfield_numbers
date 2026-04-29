"""Regenerate log-Hessian + profile-likelihood CIs from an existing fit summary.

Reads the fitted theta for each duration from
``output/piantadosi_fit/summary_all_durations.csv``, recomputes log-space
Hessian CIs, refines problematic params via profile likelihood, and
rewrites both the per-duration ``fit_{dur}ms_ci_hessian.csv`` files and
the aggregate summary CSV.

Usage:
    python scripts/regenerate_cis.py [--csv path]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    BOUNDS,
    PARAM_NAMES,
    compute_goodness,
    extract_empirical,
    hessian_ci,
    refine_cis_with_profile,
    simulate,
)
from scripts.fit_piantadosi import (
    _make_fit_with_fixed,
    _make_nll_fn,
    _make_nll_of_theta,
)
from scripts.plots import _load_ci_flat


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="output/piantadosi_fit/summary_all_durations.csv",
    )
    ap.add_argument("--skip-profile", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.csv)
    out_dir = summary_path.parent
    df = pd.read_csv(summary_path)

    new_rows = []
    for _, row in df.iterrows():
        dur = int(row["duration_ms"])
        theta = {p: float(row[p]) for p in PARAM_NAMES}
        print(f"\n=== {dur} ms  θ = {theta} ===", flush=True)

        empirical = extract_empirical(dur)
        nll_fn = _make_nll_fn(empirical)
        theta_vec = np.array([theta[p] for p in PARAM_NAMES])

        ci = hessian_ci(nll_fn, theta_vec, PARAM_NAMES, BOUNDS)
        print("Hessian CIs:")
        print(ci.to_string(index=False))

        needs = ci.apply(
            lambda r: (not np.isfinite(r["se"]))
            or r["status"] in ("non_pd", "singular", "at_bound"),
            axis=1,
        )
        if needs.any() and not args.skip_profile:
            bad = ", ".join(ci.loc[needs, "param"])
            print(f"Refining {int(needs.sum())} params via profile likelihood: {bad}")
            ci = refine_cis_with_profile(
                ci,
                nll_of_theta=_make_nll_of_theta(empirical),
                fit_with_fixed=_make_fit_with_fixed(empirical),
                theta_hat=theta,
                nll_hat=float(nll_fn(theta_vec)),
                bounds=BOUNDS,
                verbose=True,
            )
            print("Refined CIs:")
            print(ci.to_string(index=False))

        ci_path = out_dir / f"fit_{dur}ms_ci_hessian.csv"
        ci.to_csv(ci_path, index=False)
        print(f"Saved {ci_path}")

        model = simulate(theta, n_samples_per_n=5000, seed=73)
        goodness = compute_goodness(empirical, model, k=len(PARAM_NAMES))

        base = {"duration_ms": dur, **theta, **goodness}
        base.update(_load_ci_flat(ci_path))
        new_rows.append(base)

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(summary_path, index=False)
    print(f"\nSaved updated {summary_path}")


if __name__ == "__main__":
    main()
