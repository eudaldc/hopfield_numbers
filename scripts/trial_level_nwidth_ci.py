"""Profile-likelihood interval for the shared width index.

The shared landscape is parameterized directly by ``(n_width, p)``, with
``sigma_c = n_width**(-p)``. For every fixed value of ``n_width``, the exponent
and all duration-specific sensory and lapse parameters are re-optimized. The
result is a model-based interval conditional on the selected clean-up budget
``G=120`` and the trial-level likelihood used in the manuscript.

This is a long calculation. Run it from the project root:

    .venv/bin/python -m scripts.trial_level_nwidth_ci \
        --nodes 1601 --workers 5 --start-n-width 8.62 --start-p 1.86

Interrupted runs can continue from the saved profile points:

    .venv/bin/python -m scripts.trial_level_nwidth_ci \
        --nodes 1601 --workers 5 --resume
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.trial_level_shared_landscape_ladder import SPECS

MODEL_KEY = "gaussian_pow_attractor"
PROFILE_CSV = T.OUT_DIR / "shared_nwidth_profile.csv"
PROFILE_THETA_CSV = T.OUT_DIR / "shared_nwidth_profile_theta.csv"
MLE_CSV = T.OUT_DIR / "shared_nwidth_mle.csv"
MLE_THETA_CSV = T.OUT_DIR / "shared_nwidth_mle_theta.csv"
CI_CSV = T.OUT_DIR / "shared_nwidth_ci.csv"

NUISANCE_BOUNDS = (
    (np.log(1e-6), np.log(0.1)),
    (0.1, 4.0),
    (logit(T.LAPSE_BOUNDS[0]), logit(T.LAPSE_BOUNDS[1])),
)
SHARED_BOUNDS = ((np.log(1.0), np.log(60.0)), (0.1, 4.0))
_W: dict = {}


@dataclass
class FitState:
    n_width: float
    p: float
    nuisance: dict[int, np.ndarray]
    nll: float


class MleUpdated(RuntimeError):
    def __init__(self, state: FitState) -> None:
        self.state = state
        super().__init__(
            f"joint MLE updated to n_width={state.n_width:.6f}, logL={-state.nll:.6f}"
        )


def _init_worker(nodes: int) -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    _W["r_max"] = r_max
    _W["data"] = T.build_duration_data(trial, r_max)
    _W["quad"] = T.make_quadrature(nodes, kind="grid")


def _response_vector(n_width: float, p: float, nuisance: np.ndarray) -> np.ndarray:
    return np.array(
        [
            n_width ** (-p),
            np.exp(nuisance[0]),
            nuisance[1],
            p,
            expit(nuisance[2]),
        ]
    )


def _duration_nll(duration: int, n_width: float, p: float, nuisance: np.ndarray) -> float:
    return -T.response_loglik(
        SPECS[MODEL_KEY],
        _response_vector(n_width, p, nuisance),
        _W["quad"],
        _W["data"][duration].counts,
        _W["r_max"],
    )


def _fit_nuisance(
    task: tuple[int, float, float, np.ndarray, int],
) -> tuple[int, np.ndarray, float, int, bool]:
    duration, n_width, p, start, maxiter = task

    def objective(nuisance: np.ndarray) -> float:
        return _duration_nll(duration, n_width, p, nuisance)

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=NUISANCE_BOUNDS,
        options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-3},
    )
    start_nll = objective(start)
    if result.fun <= start_nll:
        return duration, np.asarray(result.x), float(result.fun), int(result.nfev), bool(result.success)
    return duration, np.asarray(start), float(start_nll), int(result.nfev), False


def _total_nll(n_width: float, p: float, nuisance: dict[int, np.ndarray]) -> float:
    return float(
        sum(
            _duration_nll(int(duration), n_width, p, nuisance[int(duration)])
            for duration in T.DURATIONS
        )
    )


def _fit_nuisance_block(
    pool: Pool,
    state: FitState,
    maxiter: int,
    label: str,
) -> dict[int, np.ndarray]:
    tasks = [
        (int(duration), state.n_width, state.p, state.nuisance[int(duration)], maxiter)
        for duration in T.DURATIONS
    ]
    nuisance = dict(state.nuisance)
    with tqdm(total=len(tasks), desc=f"{label}: nuisance", unit="duration", leave=False) as progress:
        for duration, fitted, _, _, _ in pool.imap_unordered(_fit_nuisance, tasks):
            nuisance[duration] = fitted
            progress.update()
    return nuisance


def _fit_shared_block(
    state: FitState,
    free_n_width: bool,
    maxiter: int,
) -> tuple[float, float]:
    if free_n_width:
        start = np.array([np.log(state.n_width), state.p])

        def objective(shared: np.ndarray) -> float:
            return _total_nll(np.exp(shared[0]), shared[1], state.nuisance)

        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=SHARED_BOUNDS,
            options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-3},
        )
        if result.fun <= objective(start):
            return float(np.exp(result.x[0])), float(result.x[1])
        return state.n_width, state.p

    start = np.array([state.p])

    def objective(exponent: np.ndarray) -> float:
        return _total_nll(state.n_width, exponent[0], state.nuisance)

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=[SHARED_BOUNDS[1]],
        options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-3},
    )
    if result.fun <= objective(start):
        return state.n_width, float(result.x[0])
    return state.n_width, state.p


def _coordinate_fit(
    pool: Pool,
    start: FitState,
    *,
    free_n_width: bool,
    nuisance_maxiter: int,
    shared_maxiter: int,
    max_rounds: int,
    convergence_tol: float,
    label: str,
) -> FitState:
    state = FitState(
        n_width=start.n_width,
        p=start.p,
        nuisance={duration: values.copy() for duration, values in start.nuisance.items()},
        nll=_total_nll(start.n_width, start.p, start.nuisance),
    )
    previous = state.nll
    for round_index in range(max_rounds):
        state.nuisance = _fit_nuisance_block(pool, state, nuisance_maxiter, label)
        state.n_width, state.p = _fit_shared_block(state, free_n_width, shared_maxiter)
        state.nll = _total_nll(state.n_width, state.p, state.nuisance)
        improvement = previous - state.nll
        print(
            f"{label}: round {round_index + 1}/{max_rounds}, "
            f"n_width={state.n_width:.6f}, p={state.p:.6f}, "
            f"logL={-state.nll:.6f}, improvement={improvement:.4f}",
            flush=True,
        )
        if improvement >= 0.0 and improvement < convergence_tol:
            break
        previous = state.nll
    return state


def _state_from_shared_theta(start_n_width: float | None, start_p: float | None) -> FitState:
    theta = pd.read_csv(T.OUT_DIR / "shared_landscape_ladder_theta.csv")
    theta = theta[theta["model_key"] == MODEL_KEY].set_index("duration_ms")
    p = float(theta.iloc[0]["pow"]) if start_p is None else start_p
    fitted_n_width = float(theta.iloc[0]["sigma_c"]) ** (-1.0 / float(theta.iloc[0]["pow"]))
    n_width = fitted_n_width if start_n_width is None else start_n_width
    nuisance = {
        int(duration): np.array(
            [
                np.log(float(theta.loc[duration, "s0"])),
                float(theta.loc[duration, "lambda0"]),
                logit(float(theta.loc[duration, "lapse"])),
            ]
        )
        for duration in T.DURATIONS
    }
    return FitState(n_width=n_width, p=p, nuisance=nuisance, nll=np.nan)


def _theta_rows(state: FitState, kind: str) -> list[dict]:
    rows = []
    for duration in T.DURATIONS:
        nuisance = state.nuisance[int(duration)]
        rows.append(
            {
                "kind": kind,
                "n_width": state.n_width,
                "duration_ms": int(duration),
                "p": state.p,
                "sigma_c": state.n_width ** (-state.p),
                "s0": np.exp(nuisance[0]),
                "lambda0": nuisance[1],
                "lapse": expit(nuisance[2]),
                "logL": -state.nll,
            }
        )
    return rows


def _save_mle(state: FitState, nodes: int) -> None:
    pd.DataFrame(
        [
            {
                "n_width": state.n_width,
                "p": state.p,
                "sigma_c": state.n_width ** (-state.p),
                "logL": -state.nll,
                "nodes": nodes,
                "max_steps": T.MAX_STEPS,
            }
        ]
    ).to_csv(MLE_CSV, index=False)
    pd.DataFrame(_theta_rows(state, "mle")).to_csv(MLE_THETA_CSV, index=False)


def _load_mle() -> FitState:
    theta = pd.read_csv(MLE_THETA_CSV)
    first = theta.iloc[0]
    nuisance = {
        int(row["duration_ms"]): np.array(
            [np.log(row["s0"]), row["lambda0"], logit(row["lapse"])]
        )
        for _, row in theta.iterrows()
    }
    return FitState(
        n_width=float(first["n_width"]),
        p=float(first["p"]),
        nuisance=nuisance,
        nll=-float(first["logL"]),
    )


def _save_profile(states: dict[float, FitState], mle: FitState, nodes: int) -> None:
    summary_rows = []
    theta_rows = []
    for n_width in sorted(states):
        state = states[n_width]
        summary_rows.append(
            {
                "n_width": state.n_width,
                "p": state.p,
                "sigma_c": state.n_width ** (-state.p),
                "logL": -state.nll,
                "delta_logL": state.nll - mle.nll,
                "nodes": nodes,
            }
        )
        theta_rows.extend(_theta_rows(state, "profile"))
    pd.DataFrame(summary_rows).to_csv(PROFILE_CSV, index=False)
    pd.DataFrame(theta_rows).to_csv(PROFILE_THETA_CSV, index=False)


def _load_profile() -> dict[float, FitState]:
    if not PROFILE_THETA_CSV.exists():
        return {}
    theta = pd.read_csv(PROFILE_THETA_CSV)
    states = {}
    for n_width, group in theta.groupby("n_width", sort=False):
        first = group.iloc[0]
        nuisance = {
            int(row["duration_ms"]): np.array(
                [np.log(row["s0"]), row["lambda0"], logit(row["lapse"])]
            )
            for _, row in group.iterrows()
        }
        states[float(n_width)] = FitState(
            n_width=float(n_width),
            p=float(first["p"]),
            nuisance=nuisance,
            nll=-float(first["logL"]),
        )
    return states


def _cached_state(states: dict[float, FitState], n_width: float) -> FitState | None:
    for cached_n_width, state in states.items():
        if np.isclose(cached_n_width, n_width, rtol=0.0, atol=1e-10):
            return state
    return None


def _profile_point(
    pool: Pool,
    n_width: float,
    states: dict[float, FitState],
    mle: FitState,
    args: argparse.Namespace,
    progress: tqdm,
) -> FitState:
    cached = _cached_state(states, n_width)
    if cached is not None:
        return cached
    nearest = min(states.values(), key=lambda state: abs(np.log(state.n_width / n_width)))
    start = FitState(
        n_width=n_width,
        p=nearest.p,
        nuisance={duration: values.copy() for duration, values in nearest.nuisance.items()},
        nll=np.nan,
    )
    state = _coordinate_fit(
        pool,
        start,
        free_n_width=False,
        nuisance_maxiter=args.nuisance_maxiter,
        shared_maxiter=args.shared_maxiter,
        max_rounds=args.profile_rounds,
        convergence_tol=args.convergence_tol,
        label=f"profile n={n_width:.5f}",
    )
    states[state.n_width] = state
    _save_profile(states, mle, args.nodes)
    progress.update()
    if state.nll < mle.nll - args.optimizer_tolerance:
        print(
            f"profile n={n_width:.5f} improved the joint MLE by "
            f"{mle.nll - state.nll:.3f}; refining the MLE from this point",
            flush=True,
        )
        promoted = _coordinate_fit(
            pool,
            state,
            free_n_width=True,
            nuisance_maxiter=args.nuisance_maxiter,
            shared_maxiter=args.shared_maxiter,
            max_rounds=args.mle_rounds,
            convergence_tol=args.convergence_tol,
            label="promoted joint MLE",
        )
        states[promoted.n_width] = promoted
        _save_mle(promoted, args.nodes)
        _save_profile(states, promoted, args.nodes)
        raise MleUpdated(promoted)
    return state


def _bracket(
    pool: Pool,
    side: int,
    states: dict[float, FitState],
    mle: FitState,
    cutoff: float,
    args: argparse.Namespace,
    progress: tqdm,
) -> tuple[FitState, FitState]:
    inside = mle
    step = args.initial_step
    for _ in range(args.max_bracket_steps):
        candidate = mle.n_width + side * step
        if not 1.0 < candidate < 60.0:
            raise RuntimeError("profile interval reached the n_width search boundary")
        state = _profile_point(pool, candidate, states, mle, args, progress)
        if state.nll - mle.nll >= cutoff:
            return inside, state
        inside = state
        step *= args.bracket_growth
    raise RuntimeError("failed to bracket the profile-likelihood cutoff")


def _bisect_endpoint(
    pool: Pool,
    inside: FitState,
    outside: FitState,
    states: dict[float, FitState],
    mle: FitState,
    cutoff: float,
    args: argparse.Namespace,
    progress: tqdm,
) -> float:
    for _ in range(args.max_bisections):
        midpoint = 0.5 * (inside.n_width + outside.n_width)
        state = _profile_point(pool, midpoint, states, mle, args, progress)
        if state.nll - mle.nll < cutoff:
            inside = state
        else:
            outside = state
        if abs(outside.n_width - inside.n_width) <= args.ci_tolerance:
            break
    delta_inside = inside.nll - mle.nll
    delta_outside = outside.nll - mle.nll
    if delta_outside <= delta_inside:
        raise RuntimeError("profile likelihood is not monotonic at the confidence limit")
    weight = (cutoff - delta_inside) / (delta_outside - delta_inside)
    return float(inside.n_width + weight * (outside.n_width - inside.n_width))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1601)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--start-n-width", type=float)
    parser.add_argument("--start-p", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nuisance-maxiter", type=int, default=120)
    parser.add_argument("--shared-maxiter", type=int, default=80)
    parser.add_argument("--mle-rounds", type=int, default=8)
    parser.add_argument("--profile-rounds", type=int, default=6)
    parser.add_argument("--convergence-tol", type=float, default=0.05)
    parser.add_argument("--optimizer-tolerance", type=float, default=0.1)
    parser.add_argument("--initial-step", type=float, default=0.04)
    parser.add_argument("--bracket-growth", type=float, default=1.6)
    parser.add_argument("--max-bracket-steps", type=int, default=8)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--max-mle-updates", type=int, default=3)
    parser.add_argument("--ci-tolerance", type=float, default=0.002)
    args = parser.parse_args(argv)

    _init_worker(args.nodes)
    with Pool(args.workers, initializer=_init_worker, initargs=(args.nodes,)) as pool:
        if args.resume and MLE_THETA_CSV.exists():
            mle = _load_mle()
            print(
                f"Resuming from MLE n_width={mle.n_width:.6f}, "
                f"p={mle.p:.6f}, logL={-mle.nll:.6f}",
                flush=True,
            )
        else:
            start = _state_from_shared_theta(args.start_n_width, args.start_p)
            mle = _coordinate_fit(
                pool,
                start,
                free_n_width=True,
                nuisance_maxiter=args.nuisance_maxiter,
                shared_maxiter=args.shared_maxiter,
                max_rounds=args.mle_rounds,
                convergence_tol=args.convergence_tol,
                label="joint MLE",
            )
            _save_mle(mle, args.nodes)

        states = _load_profile() if args.resume else {}
        states[mle.n_width] = mle
        _save_profile(states, mle, args.nodes)
        cutoff = 0.5 * chi2.ppf(0.95, df=1)

        with tqdm(desc="Profile-likelihood points", unit="point") as progress:
            update_count = 0
            while True:
                try:
                    lower_inside, lower_outside = _bracket(
                        pool, -1, states, mle, cutoff, args, progress
                    )
                    upper_inside, upper_outside = _bracket(
                        pool, 1, states, mle, cutoff, args, progress
                    )
                    lower = _bisect_endpoint(
                        pool,
                        lower_inside,
                        lower_outside,
                        states,
                        mle,
                        cutoff,
                        args,
                        progress,
                    )
                    upper = _bisect_endpoint(
                        pool,
                        upper_inside,
                        upper_outside,
                        states,
                        mle,
                        cutoff,
                        args,
                        progress,
                    )
                    break
                except MleUpdated as update:
                    update_count += 1
                    if update_count > args.max_mle_updates:
                        raise RuntimeError(
                            f"profile required more than {args.max_mle_updates} MLE updates"
                        ) from update
                    mle = update.state
                    print(
                        f"Restarting profile after MLE update "
                        f"{update_count}/{args.max_mle_updates}",
                        flush=True,
                    )

    result = pd.DataFrame(
        [
            {
                "n_width": mle.n_width,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "p": mle.p,
                "sigma_c": mle.n_width ** (-mle.p),
                "logL": -mle.nll,
                "delta_logL_cutoff": cutoff,
                "nodes": args.nodes,
                "max_steps": T.MAX_STEPS,
                "method": "joint profile likelihood conditional on G=120",
            }
        ]
    )
    result.to_csv(CI_CSV, index=False)
    print(result.to_string(index=False))
    print(f"Saved {CI_CSV}, {PROFILE_CSV}, and {PROFILE_THETA_CSV}")


if __name__ == "__main__":
    main()
