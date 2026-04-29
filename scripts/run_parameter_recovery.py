"""CLI orchestrator for the three identifiability analyses.

Sub-commands
------------
  recovery   parameter recovery across all 5 Piantadosi durations
  profile    profile-likelihood for sigma_c (c) at one condition
  dd         parametric bootstrap of the DD vs Controls fit
  all        runs the three in sequence

Example
-------
  # Quick smoke test (2 runs, 1 duration, coarse budget)
  python scripts/run_parameter_recovery.py recovery --smoke-test

  # Full sweep (parallel fits; default --n-jobs caps at 8)
  python scripts/run_parameter_recovery.py all --n-runs 50 --dd-runs 30

  # Serial fits
  python scripts/run_parameter_recovery.py recovery --n-jobs 1

Outputs are written under ``output/parameter_recovery/`` as CSV files; plotting
lives in ``scripts.plot_parameter_recovery`` and is invoked separately.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    PARAM_NAMES,
    extract_empirical,
    _extract_empirical_arrays,
)
from scripts.parameter_recovery import (
    RECOVERY_BUDGET,
    RecoverySpec,
    run_dd_bootstrap,
    run_profile_likelihood,
    run_recovery_sweep,
)


OUTPUT_DIR = Path("output/parameter_recovery")
PIANT_PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations.csv")
DECARLI_PARAMS_CSV = Path("output/decarli_fit/fitted_params.csv")
DURATIONS = [40, 80, 160, 320, 640]
PROFILE_REF_DURATION = 160


# ── Parameter name translation (fit csv ↔ PARAM_NAMES) ───────────────────

CSV_TO_PARAM = {
    "sigma_c": "sigma_c",
    "s0": "s0",
    "lambda0": "lambda0",
    "kappa": "kappa",
}


def _center_from_piant_row(row: pd.Series) -> dict[str, float]:
    """Extract the 6 fitted params from a summary CSV row into a theta dict."""
    return {p: float(row[p]) for p in PARAM_NAMES}


def _center_from_decarli(group: str) -> dict[str, float]:
    df = pd.read_csv(DECARLI_PARAMS_CSV)
    row = df[df["group"] == group].iloc[0]
    return {p: float(row[p]) for p in PARAM_NAMES}


# ── Budgets ───────────────────────────────────────────────────────────────


def smoke_budget() -> dict:
    return {
        **RECOVERY_BUDGET,
        "popsize": 4,
        "maxiter": 15,
        "patience": 2,
        "es_start": 5,
        "n_samples": 60,
        "mc_final": 120,
    }


# ── Commands ──────────────────────────────────────────────────────────────


def cmd_recovery(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = smoke_budget() if args.smoke_test else None
    n_runs = 2 if args.smoke_test else args.n_runs
    durations = (
        [PROFILE_REF_DURATION] if args.smoke_test else list(args.durations)
    )

    piant = pd.read_csv(PIANT_PARAMS_CSV)
    specs: list[RecoverySpec] = []
    for dur in durations:
        row = piant[piant["duration_ms"] == dur].iloc[0]
        specs.append(
            RecoverySpec(
                label=f"piant_{dur}ms",
                center=_center_from_piant_row(row),
                include_mean_guess=True,
            )
        )

    print(f"Running recovery: {len(specs)} centers × {n_runs} runs")
    df = run_recovery_sweep(
        specs,
        n_runs=n_runs,
        n_trials_per_n=args.n_trials_per_n,
        jitter_frac=args.jitter_frac,
        budget=budget,
        verbose=True,
        n_jobs=args.n_jobs,
    )
    tag = "_smoke" if args.smoke_test else ""
    path = OUTPUT_DIR / f"recovery_all_durations{tag}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def cmd_profile(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = smoke_budget() if args.smoke_test else None

    emp = extract_empirical(PROFILE_REF_DURATION)
    emp_arr = _extract_empirical_arrays(emp)

    grid_sigma = (
        np.array([0.04, 0.06])
        if args.smoke_test
        else np.linspace(0.015, 0.12, args.profile_n)
    )

    name = "sigma_c"
    grid = grid_sigma
    print(f"\n=== Profile likelihood: {name} ({len(grid)} points) ===")
    center = {
        p: float(
            pd.read_csv(PIANT_PARAMS_CSV)
            .query("duration_ms == @PROFILE_REF_DURATION")
            .iloc[0][p]
        )
        for p in PARAM_NAMES
    }
    df = run_profile_likelihood(
        center,
        name,
        grid,
        emp_arr,
        budget=budget,
        verbose=True,
        n_jobs=args.n_jobs,
    )
    tag = "_smoke" if args.smoke_test else ""
    path = OUTPUT_DIR / f"profile_likelihood{tag}.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved {path}")


def cmd_dd(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = smoke_budget() if args.smoke_test else None
    n_runs = 2 if args.smoke_test else args.dd_runs

    theta_dd = _center_from_decarli("DD")
    theta_ctrl = _center_from_decarli("Controls")

    df = run_dd_bootstrap(
        theta_dd,
        theta_ctrl,
        n_runs=n_runs,
        n_trials_per_n=args.dd_trials_per_n,
        n_shown_keep=range(1, 8),
        budget=budget,
        verbose=True,
        n_jobs=args.n_jobs,
    )
    tag = "_smoke" if args.smoke_test else ""
    path = OUTPUT_DIR / f"dd_bootstrap{tag}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_recovery(args)
    cmd_profile(args)
    cmd_dd(args)


# ── Arg parsing ───────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny budget that finishes in ~30s end-to-end (for CI / sanity).",
    )
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--dd-runs", type=int, default=30)
    p.add_argument("--profile-n", type=int, default=10)
    p.add_argument("--n-trials-per-n", type=int, default=100)
    p.add_argument("--dd-trials-per-n", type=int, default=32)
    p.add_argument("--jitter-frac", type=float, default=0.35)
    p.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=DURATIONS,
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Parallel worker processes for recovery, profile grid, and DD bootstrap.",
    )


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("recovery", "profile", "dd", "all"):
        argv = ["recovery", *argv]

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("recovery", cmd_recovery),
        ("profile", cmd_profile),
        ("dd", cmd_dd),
        ("all", cmd_all),
    ]:
        sp = sub.add_parser(name)
        _add_common(sp)
        sp.set_defaults(fn=fn)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
