"""Joint parameter recovery for the shared-landscape winning model.

Each synthetic dataset contains all five exposure durations.  The generating
and recovered models share one power-law Gaussian landscape (``sigma_c``,
``pow``) across durations, while sensory noise (``s0``, ``lambda0``) and lapse
remain duration-specific.  This is the parameterization and categorical
trial-level likelihood used by the headline fit.

Synthetic response counts preserve the empirical number of trials in every
duration-by-numerosity cell.  They are generated on a 1601-node grid and fitted
on the same 201-node grid used for model selection, with a high-resolution
evaluation of the recovered solution.  The optimizer alternates global/local
nuisance fits with a joint shared-landscape block.

    .venv/bin/python -m scripts.trial_level_recovery_winner
    .venv/bin/python -m scripts.trial_level_recovery_winner --smoke
    .venv/bin/python -m scripts.trial_level_recovery_winner --resume
    .venv/bin/python -m scripts.trial_level_recovery_winner --refit-failed
    .venv/bin/python -m scripts.trial_level_recovery_winner --replot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit, logit

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hopfield-numbers-matplotlib")

OUT_CSV = T.OUT_DIR / "recovery_winner.csv"
OUT_SUMMARY = T.OUT_DIR / "recovery_winner_summary.csv"
OUT_PNG = _ROOT / "figures" / "supfigure2_parameter_recovery.png"

N_REPS = 20
N_WORKERS = 20
FIT_NODES = 201
GEN_NODES = 1601

SHARED_PARAMS = ("sigma_c", "pow")
NUISANCE_PARAMS = ("s0", "lambda0", "lapse")
PARAMS = SHARED_PARAMS + NUISANCE_PARAMS
DERIVED = ("n_width", "sigma3", "sensory_sd5")
SHARED_QUANTITIES = frozenset(("sigma_c", "pow", "n_width", "sigma3"))
LOG_SCALE = frozenset(("sigma_c", "s0", "lapse", "sigma3"))

# Multiplicative jitter half-width around the empirical shared-landscape fit.
JITTER = {
    "sigma_c": 0.35,
    "pow": 0.25,
    "s0": 0.50,
    "lambda0": 0.20,
    "lapse": 0.40,
}

_W: dict = {}


def _sigma_n(sigma_c: float, p: float, n: np.ndarray) -> np.ndarray:
    return sigma_c * np.power(np.asarray(n, float), p)


def n_width_sigma1(sigma_c: float, p: float) -> float:
    """Continuous width index where ``sigma_c * n**p = 1``."""
    return float(sigma_c ** (-1.0 / p)) if p > 0 else float("nan")


def _reference_theta() -> pd.DataFrame:
    theta = T.load_shared_theta(T.WINNER_KEY).sort_index()
    missing = set(T.DURATIONS) - set(theta.index.astype(int))
    if missing:
        raise ValueError(f"shared-landscape theta is missing durations: {sorted(missing)}")
    for parameter in SHARED_PARAMS:
        values = theta[parameter].to_numpy(float)
        if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{parameter} is not shared across durations")
    return theta


def _init_worker(
    fit_nodes: int,
    gen_nodes: int,
    nuisance_de_maxiter: int,
    shared_de_maxiter: int,
    popsize: int,
    max_rounds: int,
    nuisance_local_maxiter: int,
    shared_local_maxiter: int,
) -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    _W["r_max"] = r_max
    _W["data"] = T.build_duration_data(trial, r_max)
    _W["quad_fit"] = T.make_quadrature(fit_nodes, kind="grid")
    _W["quad_gen"] = T.make_quadrature(gen_nodes, kind="grid")
    _W["theta"] = _reference_theta()
    _W["nuisance_de_maxiter"] = nuisance_de_maxiter
    _W["shared_de_maxiter"] = shared_de_maxiter
    _W["popsize"] = popsize
    _W["max_rounds"] = max_rounds
    _W["nuisance_local_maxiter"] = nuisance_local_maxiter
    _W["shared_local_maxiter"] = shared_local_maxiter


def _raw_bounds() -> dict[str, tuple[float, float]]:
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    return dict(zip(spec.param_names, spec.bounds)) | {"lapse": T.LAPSE_BOUNDS}


def _shared_bounds() -> tuple[tuple[float, float], ...]:
    bounds = _raw_bounds()
    return ((np.log(bounds["sigma_c"][0]), np.log(bounds["sigma_c"][1])), bounds["pow"])


def _nuisance_bounds() -> tuple[tuple[float, float], ...]:
    bounds = _raw_bounds()
    return (
        (np.log(bounds["s0"][0]), np.log(bounds["s0"][1])),
        bounds["lambda0"],
        (logit(bounds["lapse"][0]), logit(bounds["lapse"][1])),
    )


def _shared_vector(theta: pd.Series | dict) -> np.ndarray:
    return np.array([np.log(float(theta["sigma_c"])), float(theta["pow"])])


def _nuisance_vector(theta: pd.Series | dict) -> np.ndarray:
    return np.array(
        [np.log(float(theta["s0"])), float(theta["lambda0"]), logit(float(theta["lapse"]))]
    )


def _decode_shared(vector: np.ndarray) -> dict[str, float]:
    return {"sigma_c": float(np.exp(vector[0])), "pow": float(vector[1])}


def _decode_nuisance(vector: np.ndarray) -> dict[str, float]:
    return {
        "s0": float(np.exp(vector[0])),
        "lambda0": float(vector[1]),
        "lapse": float(expit(vector[2])),
    }


def _theta_vector(shared: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    values = _decode_shared(shared) | _decode_nuisance(nuisance)
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    return np.array([values[name] for name in spec.param_names] + [values["lapse"]])


def _joint_nll(
    shared: np.ndarray,
    nuisance: dict[int, np.ndarray],
    counts: dict[int, np.ndarray],
    quad: T.Quadrature,
) -> float:
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    r_max = _W["r_max"]
    return -float(
        sum(
            T.response_loglik(
                spec,
                _theta_vector(shared, nuisance[int(duration)]),
                quad,
                counts[int(duration)],
                r_max,
            )
            for duration in T.DURATIONS
        )
    )


def _bounded_population(
    bounds: tuple[tuple[float, float], ...],
    start: np.ndarray,
    popsize: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size = max(5, popsize * len(bounds))
    population = np.column_stack([rng.uniform(lo, hi, size=size) for lo, hi in bounds])
    population[0] = np.array(
        [np.clip(value, lo, hi) for value, (lo, hi) in zip(start, bounds)], dtype=float
    )
    return population


def _fit_nuisance(
    duration: int,
    shared: np.ndarray,
    start: np.ndarray,
    counts: dict[int, np.ndarray],
    rng: np.random.Generator,
    *,
    global_search: bool,
) -> np.ndarray:
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    bounds = _nuisance_bounds()

    def objective(vector: np.ndarray) -> float:
        return -T.response_loglik(
            spec,
            _theta_vector(shared, vector),
            _W["quad_fit"],
            counts[duration],
            _W["r_max"],
        )

    candidates: list[tuple[float, np.ndarray]] = [(float(objective(start)), start.copy())]
    if global_search:
        population = _bounded_population(bounds, start, _W["popsize"], rng)
        result_de = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=_W["nuisance_de_maxiter"],
            popsize=_W["popsize"],
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.85,
            seed=rng,
            init=population,
            polish=False,
            updating="deferred",
            workers=1,
        )
        candidates.append((float(result_de.fun), np.asarray(result_de.x, float)))

    _, best = min(candidates, key=lambda pair: pair[0])
    result_local = minimize(
        objective,
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxiter": _W["nuisance_local_maxiter"],
            "xatol": 1e-5,
            "fatol": 1e-3,
        },
    )
    if np.isfinite(result_local.fun) and result_local.fun <= objective(best):
        return np.asarray(result_local.x, float)
    return best


def _fit_shared(
    start: np.ndarray,
    nuisance: dict[int, np.ndarray],
    counts: dict[int, np.ndarray],
    rng: np.random.Generator,
    *,
    global_search: bool,
) -> np.ndarray:
    bounds = _shared_bounds()

    def objective(vector: np.ndarray) -> float:
        return _joint_nll(vector, nuisance, counts, _W["quad_fit"])

    candidates: list[tuple[float, np.ndarray]] = [(float(objective(start)), start.copy())]
    if global_search:
        population = _bounded_population(bounds, start, _W["popsize"], rng)
        result_de = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=_W["shared_de_maxiter"],
            popsize=_W["popsize"],
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.85,
            seed=rng,
            init=population,
            polish=False,
            updating="deferred",
            workers=1,
        )
        candidates.append((float(result_de.fun), np.asarray(result_de.x, float)))

    _, best = min(candidates, key=lambda pair: pair[0])
    result_local = minimize(
        objective,
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxiter": _W["shared_local_maxiter"],
            "xatol": 1e-6,
            "fatol": 1e-3,
        },
    )
    if np.isfinite(result_local.fun) and result_local.fun <= objective(best):
        return np.asarray(result_local.x, float)
    return best


def _jitter_parameter(
    name: str,
    center: float,
    rng: np.random.Generator,
) -> float:
    lo, hi = _raw_bounds()[name]
    value = center * float(np.exp(rng.uniform(-JITTER[name], JITTER[name])))
    return float(np.clip(value, lo * 1.02, hi * 0.98))


def _generate_counts(
    rep: int,
) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[int, np.ndarray]]:
    theta = _W["theta"]
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    r_max = _W["r_max"]
    rng = np.random.default_rng(100_003 + rep)
    first = theta.iloc[0]
    shared = {
        name: _jitter_parameter(name, float(first[name]), rng) for name in SHARED_PARAMS
    }
    nuisance: dict[int, dict[str, float]] = {}
    counts: dict[int, np.ndarray] = {}

    for duration in T.DURATIONS:
        row = theta.loc[int(duration)]
        nuisance[int(duration)] = {
            name: _jitter_parameter(name, float(row[name]), rng) for name in NUISANCE_PARAMS
        }
        parameters = shared | nuisance[int(duration)]
        grid = T.compute_grid(
            spec,
            {name: parameters[name] for name in spec.param_names},
            _W["quad_gen"],
        )
        probabilities = T.apply_lapse(grid["p_resp"], parameters["lapse"], r_max)
        real = _W["data"][int(duration)]
        synthetic = np.zeros_like(real.counts)
        for n_index in range(real.counts.shape[0]):
            p = probabilities[n_index] / probabilities[n_index].sum()
            synthetic[n_index] = rng.multinomial(int(real.counts[n_index].sum()), p)
        counts[int(duration)] = synthetic
    return shared, nuisance, counts


def _fit_joint_counts(
    rep: int,
    counts: dict[int, np.ndarray],
    start_shared: np.ndarray | None = None,
    start_nuisance: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], float, int]:
    theta = _W["theta"]
    shared = (
        _shared_vector(theta.iloc[0])
        if start_shared is None
        else np.asarray(start_shared, float).copy()
    )
    nuisance = (
        {
            int(duration): _nuisance_vector(theta.loc[int(duration)])
            for duration in T.DURATIONS
        }
        if start_nuisance is None
        else {
            int(duration): np.asarray(start_nuisance[int(duration)], float).copy()
            for duration in T.DURATIONS
        }
    )
    rng = np.random.default_rng(900_001 + rep)

    for duration in T.DURATIONS:
        nuisance[int(duration)] = _fit_nuisance(
            int(duration),
            shared,
            nuisance[int(duration)],
            counts,
            rng,
            global_search=True,
        )
    shared = _fit_shared(shared, nuisance, counts, rng, global_search=True)
    previous = _joint_nll(shared, nuisance, counts, _W["quad_fit"])

    rounds_used = 0
    for round_index in range(_W["max_rounds"]):
        rounds_used = round_index + 1
        for duration in T.DURATIONS:
            nuisance[int(duration)] = _fit_nuisance(
                int(duration),
                shared,
                nuisance[int(duration)],
                counts,
                rng,
                global_search=False,
            )
        shared = _fit_shared(shared, nuisance, counts, rng, global_search=False)
        current = _joint_nll(shared, nuisance, counts, _W["quad_fit"])
        improvement = previous - current
        previous = current
        if 0.0 <= improvement < 0.1:
            break
    return shared, nuisance, -previous, rounds_used


def _refine_until_generator_beaten(
    rep: int,
    shared: np.ndarray,
    nuisance: dict[int, np.ndarray],
    counts: dict[int, np.ndarray],
    fit_logl: float,
    generator_logl: float,
    max_retries: int = 3,
) -> tuple[np.ndarray, dict[int, np.ndarray], float, int]:
    """Add coordinate rounds when the candidate is not yet a valid MLE.

    The generating parameters are a feasible point on the fitting grid.  A
    recovered solution below their likelihood is therefore an optimizer
    failure, not poor statistical recovery.  The values themselves are never
    supplied as a start; they only trigger additional optimization.
    """
    rng = np.random.default_rng(1_500_007 + rep)
    retries = 0
    while fit_logl < generator_logl - 0.5 and retries < max_retries:
        retries += 1
        for duration in T.DURATIONS:
            nuisance[int(duration)] = _fit_nuisance(
                int(duration),
                shared,
                nuisance[int(duration)],
                counts,
                rng,
                global_search=False,
            )
        shared = _fit_shared(shared, nuisance, counts, rng, global_search=False)
        fit_logl = -_joint_nll(shared, nuisance, counts, _W["quad_fit"])
    return shared, nuisance, fit_logl, retries


def _result_rows(
    rep: int,
    generated_shared: dict[str, float],
    generated_nuisance: dict[int, dict[str, float]],
    counts: dict[int, np.ndarray],
    recovered_shared_vector: np.ndarray,
    recovered_nuisance_vectors: dict[int, np.ndarray],
    fit_logl: float,
    rounds: int,
    retries: int,
) -> list[dict]:
    generated_nuisance_vectors = {
        duration: _nuisance_vector(values) for duration, values in generated_nuisance.items()
    }
    gen_shared_vector = _shared_vector(generated_shared)
    gen_fit_logl = -_joint_nll(
        gen_shared_vector,
        generated_nuisance_vectors,
        counts,
        _W["quad_fit"],
    )
    recovered_shared = _decode_shared(recovered_shared_vector)

    gen_logl = -_joint_nll(gen_shared_vector, generated_nuisance_vectors, counts, _W["quad_gen"])
    rec_logl = -_joint_nll(
        recovered_shared_vector,
        recovered_nuisance_vectors,
        counts,
        _W["quad_gen"],
    )

    generated_n_width = n_width_sigma1(generated_shared["sigma_c"], generated_shared["pow"])
    recovered_n_width = n_width_sigma1(recovered_shared["sigma_c"], recovered_shared["pow"])
    rows = []
    for duration in T.DURATIONS:
        generated = generated_shared | generated_nuisance[int(duration)]
        recovered = recovered_shared | _decode_nuisance(recovered_nuisance_vectors[int(duration)])
        row: dict = {
            "recovery_scope": "shared_joint",
            "rep": int(rep),
            "duration_ms": int(duration),
        }
        row |= {f"gen_{name}": generated[name] for name in PARAMS}
        row |= {f"rec_{name}": recovered[name] for name in PARAMS}
        row["gen_n_width"] = generated_n_width
        row["rec_n_width"] = recovered_n_width
        row["gen_sigma3"] = float(
            _sigma_n(generated_shared["sigma_c"], generated_shared["pow"], np.array([3.0]))[0]
        )
        row["rec_sigma3"] = float(
            _sigma_n(recovered_shared["sigma_c"], recovered_shared["pow"], np.array([3.0]))[0]
        )
        row["gen_sensory_sd5"] = float(
            T.evidence_sd(np.array([5.0]), generated["s0"], generated["lambda0"])[0]
        )
        row["rec_sensory_sd5"] = float(
            T.evidence_sd(np.array([5.0]), recovered["s0"], recovered["lambda0"])[0]
        )
        row["gen_logL_joint"] = float(gen_logl)
        row["rec_logL_joint"] = float(rec_logl)
        row["gen_logL_fit_grid"] = float(gen_fit_logl)
        row["rec_logL_fit_grid"] = float(fit_logl)
        row["optimizer_rounds"] = int(rounds)
        row["optimizer_retries"] = int(retries)
        rows.append(row)
    return rows


def _one(rep: int) -> list[dict]:
    generated_shared, generated_nuisance, counts = _generate_counts(rep)
    generated_nuisance_vectors = {
        duration: _nuisance_vector(values) for duration, values in generated_nuisance.items()
    }
    gen_fit_logl = -_joint_nll(
        _shared_vector(generated_shared),
        generated_nuisance_vectors,
        counts,
        _W["quad_fit"],
    )
    recovered_shared_vector, recovered_nuisance_vectors, fit_logl, rounds = _fit_joint_counts(
        rep, counts
    )
    (
        recovered_shared_vector,
        recovered_nuisance_vectors,
        fit_logl,
        retries,
    ) = _refine_until_generator_beaten(
        rep,
        recovered_shared_vector,
        recovered_nuisance_vectors,
        counts,
        fit_logl,
        gen_fit_logl,
    )
    return _result_rows(
        rep,
        generated_shared,
        generated_nuisance,
        counts,
        recovered_shared_vector,
        recovered_nuisance_vectors,
        fit_logl,
        rounds,
        retries,
    )


def _refit_failed_one(rep: int) -> list[dict]:
    existing = pd.read_csv(OUT_CSV)
    group = existing[existing["rep"] == rep].set_index("duration_ms")
    if group.empty:
        raise ValueError(f"rep {rep} is absent from {OUT_CSV}")

    generated_shared, generated_nuisance, counts = _generate_counts(rep)
    first = group.iloc[0]
    start_shared = _shared_vector(
        {"sigma_c": first["rec_sigma_c"], "pow": first["rec_pow"]}
    )
    start_nuisance = {
        int(duration): _nuisance_vector(
            {
                "s0": group.loc[int(duration), "rec_s0"],
                "lambda0": group.loc[int(duration), "rec_lambda0"],
                "lapse": group.loc[int(duration), "rec_lapse"],
            }
        )
        for duration in T.DURATIONS
    }
    generated_nuisance_vectors = {
        duration: _nuisance_vector(values) for duration, values in generated_nuisance.items()
    }
    gen_fit_logl = -_joint_nll(
        _shared_vector(generated_shared),
        generated_nuisance_vectors,
        counts,
        _W["quad_fit"],
    )
    recovered_shared = start_shared
    recovered_nuisance = start_nuisance
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    duration_deltas = {}
    generated_shared_vector = _shared_vector(generated_shared)
    for duration in T.DURATIONS:
        generated_logl = T.response_loglik(
            spec,
            _theta_vector(generated_shared_vector, generated_nuisance_vectors[int(duration)]),
            _W["quad_fit"],
            counts[int(duration)],
            _W["r_max"],
        )
        recovered_logl = T.response_loglik(
            spec,
            _theta_vector(recovered_shared, recovered_nuisance[int(duration)]),
            _W["quad_fit"],
            counts[int(duration)],
            _W["r_max"],
        )
        duration_deltas[int(duration)] = recovered_logl - generated_logl

    hard_durations = [
        duration
        for duration, _ in sorted(duration_deltas.items(), key=lambda item: item[1])
        if duration_deltas[duration] < -0.5
    ][:2]
    saved_budget = (
        _W["nuisance_de_maxiter"],
        _W["popsize"],
        _W["nuisance_local_maxiter"],
    )
    _W["nuisance_de_maxiter"] = 35
    _W["popsize"] = 10
    _W["nuisance_local_maxiter"] = 300
    targeted_rng = np.random.default_rng(318_000 + rep)
    for duration in hard_durations:
        recovered_nuisance[duration] = _fit_nuisance(
            duration,
            recovered_shared,
            recovered_nuisance[duration],
            counts,
            targeted_rng,
            global_search=True,
        )
    (
        _W["nuisance_de_maxiter"],
        _W["popsize"],
        _W["nuisance_local_maxiter"],
    ) = saved_budget

    coordinate_rng = np.random.default_rng(2_700_001 + rep)
    recovered_shared = _fit_shared(
        recovered_shared,
        recovered_nuisance,
        counts,
        coordinate_rng,
        global_search=True,
    )
    previous = _joint_nll(recovered_shared, recovered_nuisance, counts, _W["quad_fit"])
    rounds = 0
    for round_index in range(_W["max_rounds"]):
        rounds = round_index + 1
        for duration in T.DURATIONS:
            recovered_nuisance[int(duration)] = _fit_nuisance(
                int(duration),
                recovered_shared,
                recovered_nuisance[int(duration)],
                counts,
                coordinate_rng,
                global_search=False,
            )
        recovered_shared = _fit_shared(
            recovered_shared,
            recovered_nuisance,
            counts,
            coordinate_rng,
            global_search=False,
        )
        current = _joint_nll(recovered_shared, recovered_nuisance, counts, _W["quad_fit"])
        improvement = previous - current
        previous = current
        if 0.0 <= improvement < 0.1:
            break
    fit_logl = -previous
    recovered_shared, recovered_nuisance, fit_logl, extra_retries = (
        _refine_until_generator_beaten(
            rep,
            recovered_shared,
            recovered_nuisance,
            counts,
            fit_logl,
            gen_fit_logl,
            max_retries=2,
        )
    )
    retries = len(hard_durations) + extra_retries
    return _result_rows(
        rep,
        generated_shared,
        generated_nuisance,
        counts,
        recovered_shared,
        recovered_nuisance,
        fit_logl,
        rounds,
        retries,
    )


def _validate_joint_output(df: pd.DataFrame) -> None:
    if "recovery_scope" not in df or set(df["recovery_scope"]) != {"shared_joint"}:
        raise ValueError("recovery CSV is not from the shared-landscape joint pipeline")
    expected_durations = set(T.DURATIONS)
    for rep, group in df.groupby("rep"):
        if set(group["duration_ms"].astype(int)) != expected_durations:
            raise ValueError(f"rep {rep} does not contain all five durations")
        for name in SHARED_PARAMS:
            if group[f"gen_{name}"].nunique() != 1 or group[f"rec_{name}"].nunique() != 1:
                raise ValueError(f"rep {rep} does not share {name} across durations")


def _failed_recovery_reps(df: pd.DataFrame, tolerance: float = 0.5) -> list[int]:
    diagnostic = df.sort_values(["rep", "duration_ms"]).drop_duplicates("rep")
    failed = diagnostic.loc[
        diagnostic["rec_logL_fit_grid"]
        < diagnostic["gen_logL_fit_grid"] - tolerance,
        "rep",
    ]
    return failed.astype(int).tolist()


def _quantity_frame(df: pd.DataFrame, quantity: str) -> pd.DataFrame:
    if quantity in SHARED_QUANTITIES:
        return df.sort_values(["rep", "duration_ms"]).drop_duplicates("rep")
    return df


def _pair(df: pd.DataFrame, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    frame = _quantity_frame(df, quantity)
    generated = frame[f"gen_{quantity}"].to_numpy(float)
    recovered = frame[f"rec_{quantity}"].to_numpy(float)
    if quantity in LOG_SCALE:
        return np.log10(generated), np.log10(recovered)
    return generated, recovered


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for quantity in PARAMS + DERIVED:
        frame = _quantity_frame(df, quantity)
        generated, recovered = _pair(df, quantity)
        ok = np.isfinite(generated) & np.isfinite(recovered)
        generated, recovered = generated[ok], recovered[ok]
        ss_res = float(np.sum((recovered - generated) ** 2))
        ss_tot = float(np.sum((generated - generated.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rho = float(spearmanr(generated, recovered).statistic) if generated.size > 1 else np.nan
        within_r2 = np.nan
        within_rho = np.nan
        if quantity not in SHARED_QUANTITIES:
            stratified = pd.DataFrame(
                {
                    "duration_ms": frame.loc[ok, "duration_ms"].to_numpy(int),
                    "generated": generated,
                    "recovered": recovered,
                }
            )
            within_ss_tot = float(
                sum(
                    np.sum((group["generated"] - group["generated"].mean()) ** 2)
                    for _, group in stratified.groupby("duration_ms")
                )
            )
            within_r2 = 1.0 - ss_res / within_ss_tot if within_ss_tot > 0 else np.nan
            generated_rank = stratified.groupby("duration_ms")["generated"].rank()
            recovered_rank = stratified.groupby("duration_ms")["recovered"].rank()
            within_rho = float(spearmanr(generated_rank, recovered_rank).statistic)
        rows.append(
            {
                "quantity": quantity,
                "level": "shared_dataset" if quantity in SHARED_QUANTITIES else "duration_block",
                "scale": "log10" if quantity in LOG_SCALE else "linear",
                "n": int(generated.size),
                "R2_identity": r2,
                "spearman_rho": rho,
                "R2_within_duration": within_r2,
                "spearman_within_duration": within_rho,
                "reported_R2": within_r2 if quantity not in SHARED_QUANTITIES else r2,
                "reported_spearman": within_rho if quantity not in SHARED_QUANTITIES else rho,
                "median_abs_error": float(np.median(np.abs(recovered - generated))),
            }
        )
    return pd.DataFrame(rows)


def error_corr(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    errors = {}
    for quantity in PARAMS:
        generated, recovered = _pair(df, quantity)
        errors[quantity] = recovered - generated

    matrix = np.eye(len(PARAMS))
    for i, first in enumerate(PARAMS):
        for j, second in enumerate(PARAMS):
            if first in SHARED_QUANTITIES and second in SHARED_QUANTITIES:
                left, right = errors[first], errors[second]
            elif first not in SHARED_QUANTITIES and second not in SHARED_QUANTITIES:
                left, right = errors[first], errors[second]
            else:
                shared_name = first if first in SHARED_QUANTITIES else second
                nuisance_name = second if shared_name == first else first
                shared_errors = pd.Series(errors[shared_name], index=sorted(df["rep"].unique()))
                nuisance_frame = df[["rep"]].copy()
                nuisance_frame["error"] = errors[nuisance_name]
                nuisance_mean = nuisance_frame.groupby("rep")["error"].mean()
                common = shared_errors.index.intersection(nuisance_mean.index)
                left = shared_errors.loc[common].to_numpy()
                right = nuisance_mean.loc[common].to_numpy()
                if first != shared_name:
                    left, right = right, left
            matrix[i, j] = float(spearmanr(left, right).statistic)
    return pd.DataFrame(matrix, index=PARAMS, columns=PARAMS)


def make_figure(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.plot_style import apply_style, panel_label

    apply_style(1.5)
    quantities = list(PARAMS) + ["n_width", "sigma3"]
    labels = {
        "sigma_c": r"$\sigma_c$ (log$_{10}$)",
        "pow": r"$p$",
        "s0": r"$s_0$ (log$_{10}$)",
        "lambda0": r"$\lambda_0$",
        "lapse": "lapse (log$_{10}$)",
        "n_width": r"$n_{\rm width}$ ($\sigma_n=1$)",
        "sigma3": r"$\sigma_3$ (log$_{10}$)",
    }
    letters = "abcdefgh"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = plt.get_cmap("magma")
    durations = sorted(df["duration_ms"].unique())
    colors = {
        duration: cmap(index / max(1, len(durations) - 1))
        for index, duration in enumerate(durations)
    }

    for index, (axis, quantity) in enumerate(zip(axes.flat, quantities)):
        frame = _quantity_frame(df, quantity)
        if quantity in SHARED_QUANTITIES:
            generated, recovered = _pair(frame, quantity)
            axis.scatter(generated, recovered, s=22, alpha=0.8, color="0.2")
        else:
            for duration in durations:
                duration_frame = frame[frame["duration_ms"] == duration]
                generated, recovered = _pair(duration_frame, quantity)
                axis.scatter(
                    generated,
                    recovered,
                    s=16,
                    alpha=0.75,
                    color=colors[duration],
                    label=f"{duration} ms",
                )
        generated, recovered = _pair(frame, quantity)
        if quantity == "n_width":
            low, high = 1.0, 20.0
        else:
            low = float(np.nanmin([generated.min(), recovered.min()]))
            high = float(np.nanmax([generated.max(), recovered.max()]))
        axis.plot([low, high], [low, high], "--", color="0.4", lw=1)
        if quantity == "n_width":
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
        axis.set_xlabel(f"generating {labels[quantity]}")
        axis.set_ylabel(f"recovered {labels[quantity]}")
        panel_label(axis, letters[index])

    axis = axes.flat[-1]
    image = axis.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ticks = [r"$\sigma_c$", r"$p$", r"$s_0$", r"$\lambda_0$", "lapse"]
    axis.set_xticks(range(len(PARAMS)), ticks)
    axis.set_yticks(range(len(PARAMS)), ticks)
    panel_label(axis, letters[len(quantities)])
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes.flat[2].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PNG.with_suffix(".pdf"))
    print(f"Saved {OUT_PNG} and .pdf")


def report(df: pd.DataFrame) -> None:
    _validate_joint_output(df)
    summary = summarize(df)
    corr = error_corr(df)
    summary.to_csv(OUT_SUMMARY, index=False)
    pd.set_option("display.width", 220)
    print("\n=== Shared-landscape recovery summary ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== Recovery-error correlation matrix (Spearman) ===")
    print(corr.round(2).to_string())
    print(f"\nSaved {OUT_SUMMARY}")
    make_figure(df, corr)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one joint dataset, small grids")
    parser.add_argument("--replot", action="store_true", help="rebuild summary and figure")
    parser.add_argument("--resume", action="store_true", help="resume a saved partial run")
    parser.add_argument(
        "--refit-failed",
        action="store_true",
        help="reoptimize saved reps whose fit-grid likelihood is below the generator",
    )
    parser.add_argument("--reps", type=int, default=N_REPS, help="number of joint datasets")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args(argv)

    if args.replot:
        report(pd.read_csv(OUT_CSV))
        return
    if args.reps < 1:
        parser.error("--reps must be positive")

    if args.refit_failed:
        existing = pd.read_csv(OUT_CSV)
        _validate_joint_output(existing)
        failed = _failed_recovery_reps(existing)
        if not failed:
            print("No failed recovery fits to reoptimize")
            report(existing)
            return

        print(f"Reoptimizing failed reps: {failed}", flush=True)
        refitted_rows: list[dict] = []
        started = time.time()
        with Pool(
            min(args.workers, len(failed)),
            initializer=_init_worker,
            initargs=(FIT_NODES, GEN_NODES, 10, 6, 7, 3, 160, 120),
        ) as pool:
            for finished, result_rows in enumerate(pool.imap_unordered(_refit_failed_one, failed), 1):
                refitted_rows.extend(result_rows)
                result = result_rows[0]
                print(
                    f"  [{finished}/{len(failed)}] rep={result['rep']} "
                    f"fit_delta={result['rec_logL_fit_grid'] - result['gen_logL_fit_grid']:.2f} "
                    f"rec_n_width={result['rec_n_width']:.2f} "
                    f"({(time.time() - started) / 60:.1f} min)",
                    flush=True,
                )

        replacement = pd.DataFrame(refitted_rows)
        output = pd.concat(
            [existing[~existing["rep"].isin(failed)], replacement],
            ignore_index=True,
        ).sort_values(["rep", "duration_ms"]).reset_index(drop=True)
        _validate_joint_output(output)
        output.to_csv(OUT_CSV, index=False)
        print(f"Saved {OUT_CSV}")
        report(output)
        return

    if args.smoke:
        n_reps = 1
        fit_nodes, gen_nodes = 51, 101
        nuisance_de_maxiter, shared_de_maxiter, popsize, max_rounds = 2, 1, 4, 1
        nuisance_local_maxiter, shared_local_maxiter = 30, 25
        out_csv = OUT_CSV.with_name("recovery_winner_smoke.csv")
        workers = 1
    else:
        n_reps = args.reps
        fit_nodes, gen_nodes = FIT_NODES, GEN_NODES
        nuisance_de_maxiter, shared_de_maxiter, popsize, max_rounds = 6, 3, 5, 2
        nuisance_local_maxiter, shared_local_maxiter = 100, 80
        out_csv = OUT_CSV
        workers = min(args.workers, n_reps)

    partial_csv = out_csv.with_name(f"{out_csv.stem}.partial.csv")
    rows: list[dict] = []
    completed: set[int] = set()
    if args.resume and partial_csv.exists():
        partial = pd.read_csv(partial_csv)
        _validate_joint_output(partial)
        rows = partial.to_dict("records")
        completed = set(partial["rep"].astype(int))
        print(f"Resuming after {len(completed)} completed joint datasets", flush=True)
    elif partial_csv.exists():
        partial_csv.unlink()

    reps = [rep for rep in range(n_reps) if rep not in completed]
    started = time.time()
    if reps:
        with Pool(
            workers,
            initializer=_init_worker,
            initargs=(
                fit_nodes,
                gen_nodes,
                nuisance_de_maxiter,
                shared_de_maxiter,
                popsize,
                max_rounds,
                nuisance_local_maxiter,
                shared_local_maxiter,
            ),
        ) as pool:
            for finished, result_rows in enumerate(pool.imap_unordered(_one, reps), 1):
                rows.extend(result_rows)
                result = result_rows[0]
                frame = pd.DataFrame(rows).sort_values(["rep", "duration_ms"])
                frame.to_csv(partial_csv, index=False)
                done = len(completed) + finished
                print(
                    f"  [{done}/{n_reps}] rep={result['rep']} "
                    f"gen_n_width={result['gen_n_width']:.2f} "
                    f"rec_n_width={result['rec_n_width']:.2f} "
                    f"fit_delta={result['rec_logL_fit_grid'] - result['gen_logL_fit_grid']:.2f} "
                    f"eval_delta={result['rec_logL_joint'] - result['gen_logL_joint']:.2f} "
                    f"({(time.time() - started) / 60:.1f} min)",
                    flush=True,
                )

    output = pd.DataFrame(rows).sort_values(["rep", "duration_ms"]).reset_index(drop=True)
    _validate_joint_output(output)
    output.to_csv(out_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()
    print(f"Saved {out_csv}")
    if args.smoke:
        print(output.round(4).to_string(index=False))
    else:
        failed = _failed_recovery_reps(output)
        if failed:
            print(
                f"Re-running {len(failed)} optimizer-limited reps before reporting",
                flush=True,
            )
            main(["--refit-failed", "--workers", str(args.workers)])
        else:
            report(output)


if __name__ == "__main__":
    main()
