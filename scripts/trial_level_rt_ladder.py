"""RT model ladder on the fixed G=120 shared landscape.

The response-fitted landscape is never refitted here. The five requested RT
links differ only in how convergence time and readout entropy enter the
log-normal location. Residual RT scatter is profiled separately by duration.

Full analysis (bootstrap and recovery can take more than two minutes):

    .venv/bin/python -m scripts.trial_level_rt_ladder

Fast implementation check:

    .venv/bin/python -m scripts.trial_level_rt_ladder --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.trial_level_model_comparison as T
from scripts.plot_style import C_ATTR, C_BENCH, C_STATIC, apply_style, panel_label


DURS = np.asarray(T.DURATIONS, dtype=int)
NVALS = np.asarray(T.N_SHOWN, dtype=int)
LOG_DUR = np.log(DURS / 160.0)
CV_SEED = 20260717
MODEL_IDS = ("M1", "M2", "M3", "M4", "M5")
DIAGNOSTIC_ID = "M4_resid"
MODEL_LABELS = {
    "M1": "Shared convergence",
    "M2": "Shared convergence + entropy",
    "M3": "Duration-dependent convergence",
    "M4": "Duration-dependent convergence + entropy",
    "M5": "Saturated ceiling",
    DIAGNOSTIC_ID: "Duration convergence + residual entropy",
}


@dataclass(frozen=True)
class CellStats:
    count: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray

    @property
    def n_total(self) -> int:
        return int(np.sum(self.count))


@dataclass(frozen=True)
class Prepared:
    participant_ids: np.ndarray
    count_by_participant: np.ndarray
    sum_by_participant: np.ndarray
    sum2_by_participant: np.ndarray
    et: np.ndarray
    eh: np.ndarray
    eh_resid: np.ndarray
    entropy_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class FitResult:
    model_id: str
    vector: np.ndarray
    tau: np.ndarray
    kappa: np.ndarray
    sigma: np.ndarray
    logl: float
    k: int
    converged: bool

    @property
    def aic(self) -> float:
        return float(2 * self.k - 2 * self.logl)

    def bic(self, n: int) -> float:
        return float(self.k * np.log(n) - 2 * self.logl)


def _weighted_entropy_residuals(
    et: np.ndarray, eh: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    resid = np.zeros_like(eh)
    rows = []
    for di, d in enumerate(DURS):
        w = counts[di].astype(float)
        x = et[di]
        y = eh[di]
        design = np.column_stack([np.ones_like(x), x])
        root_w = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
        fitted = design @ coef
        resid[di] = y - fitted
        y_bar = np.average(y, weights=w)
        ss_tot = np.sum(w * (y - y_bar) ** 2)
        ss_res = np.sum(w * resid[di] ** 2)
        corr = np.corrcoef(
            np.repeat(x, w.astype(int)), np.repeat(y, w.astype(int))
        )[0, 1]
        rows.append(
            {
                "duration_ms": int(d),
                "corr_ET_EH": float(corr),
                "entropy_on_steps_intercept": float(coef[0]),
                "entropy_on_steps_slope": float(coef[1]),
                "residual_entropy_fraction": float(ss_res / ss_tot),
                "weighted_residual_mean": float(np.average(resid[di], weights=w)),
                "weighted_residual_steps_cov": float(
                    np.average(resid[di] * (x - np.average(x, weights=w)), weights=w)
                ),
            }
        )
    return resid, pd.DataFrame(rows)


def prepare_data(nodes: int = 1601) -> Prepared:
    T.MAX_STEPS = 120
    trial = T.load_trials()
    participant_ids, participant_index = np.unique(
        trial["uniqueid"].astype(str).to_numpy(), return_inverse=True
    )
    dur_index = pd.Categorical(trial["exp_dur"], categories=DURS).codes
    n_index = trial["n_shown"].to_numpy(int) - T.N_TRUE_MIN
    y = np.log(trial["rt_s"].to_numpy(float))

    shape = (participant_ids.size, DURS.size, NVALS.size)
    count = np.zeros(shape, dtype=float)
    sum_y = np.zeros(shape, dtype=float)
    sum_y2 = np.zeros(shape, dtype=float)
    loc = (participant_index, dur_index, n_index)
    np.add.at(count, loc, 1.0)
    np.add.at(sum_y, loc, y)
    np.add.at(sum_y2, loc, y * y)

    theta = T.load_shared_theta()
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    quad = T.make_quadrature(nodes, kind="grid")
    et = np.zeros((DURS.size, NVALS.size))
    eh = np.zeros_like(et)
    for di, d in enumerate(DURS):
        params = {name: float(theta.loc[d, name]) for name in spec.param_names}
        grid = T.compute_grid(spec, params, quad)
        et[di] = grid["ET"]
        eh[di] = grid["EH"]

    total_counts = np.sum(count, axis=0)
    eh_resid, diagnostics = _weighted_entropy_residuals(et, eh, total_counts)
    return Prepared(
        participant_ids=participant_ids,
        count_by_participant=count,
        sum_by_participant=sum_y,
        sum2_by_participant=sum_y2,
        et=et,
        eh=eh,
        eh_resid=eh_resid,
        entropy_diagnostics=diagnostics,
    )


def aggregate(prepared: Prepared, participant_indices: np.ndarray | None = None) -> CellStats:
    if participant_indices is None:
        participant_indices = np.arange(prepared.participant_ids.size)
    return CellStats(
        count=np.sum(prepared.count_by_participant[participant_indices], axis=0),
        sum_y=np.sum(prepared.sum_by_participant[participant_indices], axis=0),
        sum_y2=np.sum(prepared.sum2_by_participant[participant_indices], axis=0),
    )


def parameter_names(model_id: str) -> tuple[str, ...]:
    if model_id == "M1":
        return ("a_tau",)
    if model_id == "M2":
        return ("a_tau", "a_kappa")
    if model_id == "M3":
        return ("a_tau", "b_tau")
    if model_id in ("M4", DIAGNOSTIC_ID):
        return ("a_tau", "b_tau", "a_kappa", "b_kappa")
    if model_id == "M5":
        return tuple([f"log_tau_{d}" for d in DURS] + [f"log_kappa_{d}" for d in DURS])
    raise ValueError(f"unknown model {model_id!r}")


def decode_link(model_id: str, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    if model_id == "M1":
        tau = np.repeat(np.exp(vector[0]), DURS.size)
        kappa = np.zeros(DURS.size)
    elif model_id == "M2":
        tau = np.repeat(np.exp(vector[0]), DURS.size)
        kappa = np.repeat(np.exp(vector[1]), DURS.size)
    elif model_id == "M3":
        tau = np.exp(vector[0] + vector[1] * LOG_DUR)
        kappa = np.zeros(DURS.size)
    elif model_id in ("M4", DIAGNOSTIC_ID):
        tau = np.exp(vector[0] + vector[1] * LOG_DUR)
        kappa = np.exp(vector[2] + vector[3] * LOG_DUR)
    elif model_id == "M5":
        tau = np.exp(vector[: DURS.size])
        kappa = np.exp(vector[DURS.size :])
    else:
        raise ValueError(f"unknown model {model_id!r}")
    return tau, kappa


def model_location(
    prepared: Prepared, model_id: str, vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau, kappa = decode_link(model_id, vector)
    entropy = prepared.eh_resid if model_id == DIAGNOSTIC_ID else prepared.eh
    location = T.T0 + tau[:, None] * prepared.et + kappa[:, None] * entropy
    return location, tau, kappa


def _profiled_objective(
    prepared: Prepared, stats: CellStats, model_id: str, vector: np.ndarray
) -> tuple[float, np.ndarray]:
    location, _, _ = model_location(prepared, model_id, vector)
    if np.any(location <= 0) or not np.all(np.isfinite(location)):
        return 1e100, np.full(DURS.size, np.nan)
    mu = np.log(location)
    sse_cell = stats.sum_y2 - 2.0 * mu * stats.sum_y + stats.count * mu * mu
    n_d = np.sum(stats.count, axis=1)
    sse_d = np.sum(sse_cell, axis=1)
    sigma2 = sse_d / n_d
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e100, np.full(DURS.size, np.nan)
    neg_log_normal = 0.5 * np.sum(n_d * (np.log(2.0 * np.pi * sigma2) + 1.0))
    return float(neg_log_normal), np.sqrt(sigma2)


def _initial_vector(model_id: str) -> np.ndarray:
    log_tau = np.log(0.011)
    log_kappa = np.log(0.05)
    if model_id == "M1":
        return np.array([log_tau])
    if model_id == "M2":
        return np.array([log_tau, log_kappa])
    if model_id == "M3":
        return np.array([log_tau, 0.0])
    if model_id in ("M4", DIAGNOSTIC_ID):
        return np.array([log_tau, 0.0, log_kappa, 0.0])
    if model_id == "M5":
        return np.r_[np.repeat(log_tau, DURS.size), np.repeat(log_kappa, DURS.size)]
    raise ValueError(f"unknown model {model_id!r}")


def _bounds(model_id: str) -> list[tuple[float, float]]:
    amplitude = (np.log(1e-8), np.log(5.0))
    slope = (-3.0, 3.0)
    if model_id == "M1":
        return [amplitude]
    if model_id == "M2":
        return [amplitude, amplitude]
    if model_id == "M3":
        return [amplitude, slope]
    if model_id in ("M4", DIAGNOSTIC_ID):
        return [amplitude, slope, amplitude, slope]
    if model_id == "M5":
        return [amplitude] * (2 * DURS.size)
    raise ValueError(f"unknown model {model_id!r}")


def fit_model(
    prepared: Prepared,
    stats: CellStats,
    model_id: str,
    *,
    seed: int = CV_SEED,
    starts: int = 4,
) -> FitResult:
    rng = np.random.default_rng(seed)
    x0 = _initial_vector(model_id)
    candidates = [x0]
    if "a_kappa" in parameter_names(model_id) or model_id == "M5":
        low_entropy = x0.copy()
        if model_id == "M5":
            low_entropy[DURS.size :] = np.log(1e-5)
        else:
            low_entropy[parameter_names(model_id).index("a_kappa")] = np.log(1e-5)
        candidates.append(low_entropy)
    while len(candidates) < starts:
        candidates.append(x0 + rng.normal(0.0, 0.35, size=x0.size))

    best = None
    bounds = _bounds(model_id)
    for start in candidates:
        result = minimize(
            lambda value: _profiled_objective(prepared, stats, model_id, value)[0],
            np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    if model_id in ("M2", "M4", "M5", DIAGNOSTIC_ID):
        powell_starts = [best.x]
        if model_id != "M5":
            powell_starts.append(x0)
        for start in powell_starts:
            result = minimize(
                lambda value: _profiled_objective(prepared, stats, model_id, value)[0],
                start,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 2000, "xtol": 1e-7, "ftol": 1e-10},
            )
            if result.fun < best.fun:
                best = result

    neg_log_normal, sigma = _profiled_objective(prepared, stats, model_id, best.x)
    logl = -neg_log_normal - float(np.sum(stats.sum_y))
    tau, kappa = decode_link(model_id, best.x)
    return FitResult(
        model_id=model_id,
        vector=np.asarray(best.x, float),
        tau=tau,
        kappa=kappa,
        sigma=sigma,
        logl=float(logl),
        k=len(parameter_names(model_id)) + DURS.size,
        converged=bool(best.success),
    )


def score_stats(
    prepared: Prepared, stats: CellStats, fit: FitResult
) -> tuple[float, np.ndarray]:
    location, _, _ = model_location(prepared, fit.model_id, fit.vector)
    mu = np.log(location)
    sse_cell = stats.sum_y2 - 2.0 * mu * stats.sum_y + stats.count * mu * mu
    logl_cell = (
        -0.5 * stats.count * np.log(2.0 * np.pi * fit.sigma[:, None] ** 2)
        - 0.5 * sse_cell / fit.sigma[:, None] ** 2
        - stats.sum_y
    )
    return float(np.sum(logl_cell)), logl_cell


def fit_row(fit: FitResult, n: int) -> dict:
    row = {
        "model_id": fit.model_id,
        "label": MODEL_LABELS[fit.model_id],
        "logL": fit.logl,
        "k": fit.k,
        "AIC": fit.aic,
        "BIC": fit.bic(n),
        "converged": fit.converged,
    }
    row.update(dict(zip(parameter_names(fit.model_id), fit.vector)))
    for di, d in enumerate(DURS):
        row[f"tau_{d}"] = fit.tau[di]
        row[f"kappa_{d}"] = fit.kappa[di]
        row[f"sigma_rt_{d}"] = fit.sigma[di]
    return row


def _cell_r2(stats: CellStats, location: np.ndarray) -> tuple[float, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.exp(stats.sum_y / stats.count)
    valid = stats.count > 0
    obs = observed[valid]
    pred = location[valid]
    total = np.sum((obs - np.mean(obs)) ** 2)
    pooled = float(1.0 - np.sum((obs - pred) ** 2) / total)
    by_duration = np.full(DURS.size, np.nan)
    for di in range(DURS.size):
        keep = valid[di]
        obs_d = observed[di, keep]
        pred_d = location[di, keep]
        total_d = np.sum((obs_d - np.mean(obs_d)) ** 2)
        by_duration[di] = 1.0 - np.sum((obs_d - pred_d) ** 2) / total_d
    return pooled, by_duration


def prediction_rows(
    prepared: Prepared, stats: CellStats, fits: dict[str, FitResult]
) -> tuple[list[dict], dict[str, tuple[float, np.ndarray]]]:
    rows = []
    r2 = {}
    observed = np.exp(
        np.divide(stats.sum_y, stats.count, out=np.full_like(stats.sum_y, np.nan), where=stats.count > 0)
    )
    for model_id, fit in fits.items():
        location, _, _ = model_location(prepared, model_id, fit.vector)
        r2[model_id] = _cell_r2(stats, location)
        for di, d in enumerate(DURS):
            for ni, n in enumerate(NVALS):
                rows.append(
                    {
                        "model_id": model_id,
                        "duration_ms": int(d),
                        "n": int(n),
                        "n_trials": int(stats.count[di, ni]),
                        "observed_geomean_rt": observed[di, ni],
                        "predicted_geomean_rt": location[di, ni],
                    }
                )
    return rows, r2


def participant_folds(prepared: Prepared, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(prepared.participant_ids.size)
    fold = np.empty(prepared.participant_ids.size, dtype=int)
    fold[order] = np.arange(order.size) % n_folds
    return fold


def run_cv(
    prepared: Prepared,
    model_ids: tuple[str, ...],
    *,
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = participant_folds(prepared, n_folds, seed)
    fold_rows = []
    participant_rows = []
    prediction_rows_cv = []
    for fold in tqdm(range(n_folds), desc="Participant CV", unit="fold"):
        train_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)
        train_stats = aggregate(prepared, train_idx)
        test_stats = aggregate(prepared, test_idx)
        for model_id in model_ids:
            fit = fit_model(prepared, train_stats, model_id, seed=seed + 101 * fold)
            test_logl, _ = score_stats(prepared, test_stats, fit)
            location, _, _ = model_location(prepared, model_id, fit.vector)
            pooled_r2, duration_r2 = _cell_r2(test_stats, location)
            row = {
                "fold": fold,
                "model_id": model_id,
                "heldout_logL": test_logl,
                "n_trials": test_stats.n_total,
                "mean_logL": test_logl / test_stats.n_total,
                "R2_pooled": pooled_r2,
            }
            row.update(dict(zip(parameter_names(model_id), fit.vector)))
            for di, d in enumerate(DURS):
                row[f"tau_{d}"] = fit.tau[di]
                row[f"kappa_{d}"] = fit.kappa[di]
                row[f"sigma_rt_{d}"] = fit.sigma[di]
            row.update({f"R2_{d}": duration_r2[di] for di, d in enumerate(DURS)})
            fold_rows.append(row)

            observed = np.exp(
                np.divide(
                    test_stats.sum_y,
                    test_stats.count,
                    out=np.full_like(test_stats.sum_y, np.nan),
                    where=test_stats.count > 0,
                )
            )
            for di, d in enumerate(DURS):
                for ni, n in enumerate(NVALS):
                    prediction_rows_cv.append(
                        {
                            "fold": fold,
                            "model_id": model_id,
                            "duration_ms": int(d),
                            "n": int(n),
                            "n_trials": int(test_stats.count[di, ni]),
                            "observed_geomean_rt": observed[di, ni],
                            "predicted_geomean_rt": location[di, ni],
                        }
                    )

            for pi in test_idx:
                participant_stats = aggregate(prepared, np.array([pi]))
                participant_logl, _ = score_stats(prepared, participant_stats, fit)
                participant_rows.append(
                    {
                        "fold": fold,
                        "model_id": model_id,
                        "uniqueid": prepared.participant_ids[pi],
                        "heldout_logL": participant_logl,
                        "n_trials": participant_stats.n_total,
                    }
                )
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(participant_rows),
        pd.DataFrame(prediction_rows_cv),
    )


def cv_r2_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, group in predictions.groupby("model_id"):
        valid = group["n_trials"].to_numpy() > 0
        observed = group.loc[valid, "observed_geomean_rt"].to_numpy(float)
        predicted = group.loc[valid, "predicted_geomean_rt"].to_numpy(float)
        total = np.sum((observed - np.mean(observed)) ** 2)
        row = {
            "model_id": model_id,
            "heldout_R2_pooled": 1.0 - np.sum((observed - predicted) ** 2) / total,
        }
        for d, duration_group in group[valid].groupby("duration_ms"):
            observed_d = duration_group["observed_geomean_rt"].to_numpy(float)
            predicted_d = duration_group["predicted_geomean_rt"].to_numpy(float)
            total_d = np.sum((observed_d - np.mean(observed_d)) ** 2)
            row[f"heldout_R2_{d}"] = (
                1.0 - np.sum((observed_d - predicted_d) ** 2) / total_d
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_bootstrap(
    prepared: Prepared,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    model_ids = ("M3", "M4", DIAGNOSTIC_ID)
    for replicate in tqdm(range(n_boot), desc="Participant bootstrap", unit="rep"):
        sampled = rng.integers(0, prepared.participant_ids.size, prepared.participant_ids.size)
        stats = aggregate(prepared, sampled)
        for mi, model_id in enumerate(model_ids):
            fit = fit_model(prepared, stats, model_id, seed=seed + 1009 * replicate + mi)
            row = {"replicate": replicate, **fit_row(fit, stats.n_total)}
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(draws: pd.DataFrame) -> pd.DataFrame:
    identity = {"replicate", "model_id", "label", "converged"}
    numeric = [
        column
        for column in draws.columns
        if column not in identity and pd.api.types.is_numeric_dtype(draws[column])
    ]
    rows = []
    for model_id, group in draws.groupby("model_id"):
        for parameter in numeric:
            values = group[parameter].dropna().to_numpy(float)
            if values.size == 0:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "parameter": parameter,
                    "median": np.median(values),
                    "ci_lo": np.quantile(values, 0.025),
                    "ci_hi": np.quantile(values, 0.975),
                    "boundary_fraction": np.mean(values <= 1e-4)
                    if parameter.startswith("kappa_")
                    else np.nan,
                    "n_boot": values.size,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_correlations(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, group in draws.groupby("model_id"):
        parameters = [
            name
            for name in parameter_names(model_id)
            if name in group and group[name].notna().sum() > 1
        ]
        for i, first in enumerate(parameters):
            for second in parameters[i + 1 :]:
                rows.append(
                    {
                        "model_id": model_id,
                        "parameter_1": first,
                        "parameter_2": second,
                        "correlation": float(group[[first, second]].corr().iloc[0, 1]),
                    }
                )
    return pd.DataFrame(rows)


def _truth_from_row(row: pd.Series, model_id: str, fallback: FitResult) -> FitResult:
    names = parameter_names(model_id)
    if all(name in row and pd.notna(row[name]) for name in names):
        vector = np.array([float(row[name]) for name in names])
    else:
        vector = fallback.vector
    tau, kappa = decode_link(model_id, vector)
    sigma = np.array(
        [float(row.get(f"sigma_rt_{d}", fallback.sigma[di])) for di, d in enumerate(DURS)]
    )
    return FitResult(model_id, vector, tau, kappa, sigma, np.nan, len(names) + DURS.size, True)


def simulate_stats(
    prepared: Prepared, counts: np.ndarray, truth: FitResult, rng: np.random.Generator
) -> CellStats:
    location, _, _ = model_location(prepared, truth.model_id, truth.vector)
    mu = np.log(location)
    sum_y = np.zeros_like(mu)
    sum_y2 = np.zeros_like(mu)
    for di in range(DURS.size):
        for ni in range(NVALS.size):
            n = int(counts[di, ni])
            if n == 0:
                continue
            mean = rng.normal(mu[di, ni], truth.sigma[di] / np.sqrt(n))
            within = (
                truth.sigma[di] ** 2 * rng.chisquare(n - 1) if n > 1 else 0.0
            )
            sum_y[di, ni] = n * mean
            sum_y2[di, ni] = within + n * mean * mean
    return CellStats(counts.copy(), sum_y, sum_y2)


def run_recovery(
    prepared: Prepared,
    full_fits: dict[str, FitResult],
    bootstrap: pd.DataFrame,
    *,
    n_recovery: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = aggregate(prepared).count
    rows = []
    half = n_recovery // 2
    generators = ["M3"] * half + ["M4"] * (n_recovery - half)
    bootstrap_by_model = {
        model_id: group.reset_index(drop=True)
        for model_id, group in bootstrap.groupby("model_id")
        if model_id in ("M3", "M4")
    }
    for replicate, generating_model in enumerate(
        tqdm(generators, desc="Parametric recovery", unit="rep")
    ):
        candidates = bootstrap_by_model.get(generating_model)
        if candidates is not None and not candidates.empty:
            source = candidates.iloc[rng.integers(0, len(candidates))]
            truth = _truth_from_row(source, generating_model, full_fits[generating_model])
        else:
            truth = full_fits[generating_model]
        simulated = simulate_stats(prepared, counts, truth, rng)
        fitted = {
            model_id: fit_model(
                prepared, simulated, model_id, seed=seed + 2029 * replicate + mi
            )
            for mi, model_id in enumerate(("M3", "M4"))
        }
        truth_location, _, _ = model_location(
            prepared, generating_model, truth.vector
        )
        for fitted_model, fit in fitted.items():
            fitted_location, _, _ = model_location(prepared, fitted_model, fit.vector)
            row = {
                "replicate": replicate,
                "generating_model": generating_model,
                "fitted_model": fitted_model,
                "rt_curve_rmse": float(
                    np.sqrt(np.mean((fitted_location - truth_location) ** 2))
                ),
                **fit_row(fit, simulated.n_total),
            }
            for name, value in zip(parameter_names(generating_model), truth.vector):
                row[f"true_{name}"] = value
            for di, d in enumerate(DURS):
                row[f"true_tau_{d}"] = truth.tau[di]
                row[f"true_kappa_{d}"] = truth.kappa[di]
            rows.append(row)
    return pd.DataFrame(rows)


def recovery_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for criterion in ("AIC", "BIC"):
        pivot = draws.pivot_table(
            index=["replicate", "generating_model"],
            columns="fitted_model",
            values=criterion,
        ).reset_index()
        for generating_model, group in pivot.groupby("generating_model"):
            rows.append(
                {
                    "generating_model": generating_model,
                    "fitted_model": "selection",
                    "parameter": f"M4_selected_by_{criterion}",
                    "value": float(np.mean(group["M4"] < group["M3"])),
                    "bias": np.nan,
                    "rmse": np.nan,
                    "correlation": np.nan,
                    "n": len(group),
                }
            )

    for (generating_model, fitted_model), group in draws.groupby(
        ["generating_model", "fitted_model"]
    ):
        rows.append(
            {
                "generating_model": generating_model,
                "fitted_model": fitted_model,
                "parameter": "rt_curve_rmse",
                "value": float(np.mean(group["rt_curve_rmse"])),
                "bias": np.nan,
                "rmse": float(
                    np.sqrt(np.mean(group["rt_curve_rmse"].to_numpy(float) ** 2))
                ),
                "correlation": np.nan,
                "n": len(group),
            }
        )
        for parameter in [f"tau_{d}" for d in DURS] + [f"kappa_{d}" for d in DURS]:
            true_name = f"true_{parameter}"
            if true_name not in group or parameter not in group:
                continue
            true = group[true_name].to_numpy(float)
            estimate = group[parameter].to_numpy(float)
            correlation = (
                np.corrcoef(true, estimate)[0, 1]
                if np.std(true) > 0 and np.std(estimate) > 0
                else np.nan
            )
            rows.append(
                {
                    "generating_model": generating_model,
                    "fitted_model": fitted_model,
                    "parameter": parameter,
                    "value": np.nan,
                    "bias": float(np.mean(estimate - true)),
                    "rmse": float(np.sqrt(np.mean((estimate - true) ** 2))),
                    "correlation": float(correlation),
                    "n": len(group),
                }
            )
        if generating_model == "M4" and fitted_model == "M4":
            tau_errors = []
            kappa_errors = []
            for d in DURS:
                tau_errors.append(
                    group[f"tau_{d}"].to_numpy(float)
                    - group[f"true_tau_{d}"].to_numpy(float)
                )
                kappa_errors.append(
                    group[f"kappa_{d}"].to_numpy(float)
                    - group[f"true_kappa_{d}"].to_numpy(float)
                )
            tau_error = np.concatenate(tau_errors)
            kappa_error = np.concatenate(kappa_errors)
            rows.append(
                {
                    "generating_model": generating_model,
                    "fitted_model": fitted_model,
                    "parameter": "tau_kappa_error_correlation",
                    "value": float(np.corrcoef(tau_error, kappa_error)[0, 1]),
                    "bias": np.nan,
                    "rmse": np.nan,
                    "correlation": np.nan,
                    "n": tau_error.size,
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    ladder: pd.DataFrame,
    cv: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    recovery: pd.DataFrame,
    output_stem: Path,
) -> None:
    apply_style(1.0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    main = ladder[ladder["model_id"].isin(MODEL_IDS)].copy()
    order = list(MODEL_IDS)
    main = main.set_index("model_id").loc[order]
    x = np.arange(len(order))
    ax_a.bar(x - 0.18, main["dAIC"], width=0.36, color=C_ATTR, label="AIC")
    ax_a.bar(x + 0.18, main["dBIC"], width=0.36, color=C_STATIC, label="BIC")
    ax_a.set_xticks(x, order)
    ax_a.set_ylabel("Difference from best")
    ax_a.legend(frameon=False)
    panel_label(ax_a, "a")

    cv_sum = cv.groupby("model_id", as_index=False).agg(
        heldout_logL=("heldout_logL", "sum"), n_trials=("n_trials", "sum")
    )
    cv_sum["mean_logL"] = cv_sum["heldout_logL"] / cv_sum["n_trials"]
    cv_sum = cv_sum.set_index("model_id").loc[order]
    reference = cv_sum.loc["M3", "mean_logL"]
    ax_b.bar(x, cv_sum["mean_logL"] - reference, color=C_BENCH)
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_xticks(x, order)
    ax_b.set_ylabel("Held-out mean logL minus M3")
    panel_label(ax_b, "b")

    for model_id, color, marker in (("M3", C_STATIC, "o"), ("M4", C_ATTR, "s"), ("M5", C_BENCH, "^")):
        row = ladder.set_index("model_id").loc[model_id]
        ax_c.plot(DURS, [row[f"tau_{d}"] for d in DURS], marker=marker, color=color, label=f"{model_id} tau")
        if model_id != "M3":
            ax_c.plot(
                DURS,
                [row[f"kappa_{d}"] for d in DURS],
                marker=marker,
                color=color,
                ls="--",
                label=f"{model_id} kappa",
            )
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(DURS, DURS)
    ax_c.set_ylabel("RT-link coefficient")
    ax_c.set_xlabel("Exposure duration (ms)")
    ax_c.legend(frameon=False, ncols=2)
    panel_label(ax_c, "c")

    selection = recovery[recovery["parameter"] == "M4_selected_by_AIC"]
    text = ["M4 selected in recovery:"]
    for _, row in selection.iterrows():
        text.append(f"{row['generating_model']} truth: {row['value']:.1%}")
    if not bootstrap_ci.empty:
        kappa = bootstrap_ci[
            (bootstrap_ci["model_id"] == "M4")
            & (bootstrap_ci["parameter"].str.startswith("kappa_"))
        ]
        if not kappa.empty:
            text.append("")
            text.append(
                f"M4 kappa boundary mass: {kappa['boundary_fraction'].mean():.1%}"
            )
    ax_d.axis("off")
    ax_d.text(0.02, 0.95, "\n".join(text), va="top")
    panel_label(ax_d, "d")

    fig.tight_layout()
    fig.savefig(f"{output_stem}.png")
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--recovery", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=CV_SEED)
    parser.add_argument("--nodes", type=int, default=1601)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        args.bootstrap = min(args.bootstrap, 4)
        args.recovery = min(args.recovery, 6)
        args.folds = min(args.folds, 2)
        args.nodes = min(args.nodes, 201)

    prepared = prepare_data(args.nodes)
    stats = aggregate(prepared)
    all_models = MODEL_IDS + (DIAGNOSTIC_ID,)
    fits = {
        model_id: fit_model(prepared, stats, model_id, seed=args.seed + mi)
        for mi, model_id in enumerate(tqdm(all_models, desc="Full-data ladder", unit="model"))
    }
    rows = [fit_row(fits[model_id], stats.n_total) for model_id in all_models]
    predictions, r2_values = prediction_rows(prepared, stats, fits)
    ladder = pd.DataFrame(rows)
    for model_id, (pooled, by_duration) in r2_values.items():
        ladder.loc[ladder.model_id == model_id, "R2_pooled"] = pooled
        for di, d in enumerate(DURS):
            ladder.loc[ladder.model_id == model_id, f"R2_{d}"] = by_duration[di]
    ladder["dAIC"] = ladder["AIC"] - ladder.loc[ladder.model_id.isin(MODEL_IDS), "AIC"].min()
    ladder["dBIC"] = ladder["BIC"] - ladder.loc[ladder.model_id.isin(MODEL_IDS), "BIC"].min()

    cv, cv_participants, cv_predictions = run_cv(
        prepared, all_models, n_folds=args.folds, seed=args.seed
    )
    cv_totals = cv.groupby("model_id", as_index=False).agg(
        heldout_logL=("heldout_logL", "sum"), heldout_n=("n_trials", "sum")
    )
    ladder = ladder.merge(cv_totals, on="model_id", how="left")
    ladder = ladder.merge(cv_r2_summary(cv_predictions), on="model_id", how="left")
    ladder["heldout_mean_logL"] = ladder["heldout_logL"] / ladder["heldout_n"]

    bootstrap = run_bootstrap(prepared, n_boot=args.bootstrap, seed=args.seed + 1)
    bootstrap_ci = bootstrap_summary(bootstrap)
    bootstrap_corr = bootstrap_correlations(bootstrap)
    recovery = run_recovery(
        prepared,
        fits,
        bootstrap,
        n_recovery=args.recovery,
        seed=args.seed + 2,
    )
    recovery_ci = recovery_summary(recovery)

    out = T.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"rt_ladder{suffix}.csv": ladder,
        f"rt_ladder_predictions{suffix}.csv": pd.DataFrame(predictions),
        f"rt_ladder_entropy_residuals{suffix}.csv": prepared.entropy_diagnostics,
        f"rt_ladder_cv_folds{suffix}.csv": cv,
        f"rt_ladder_cv_participants{suffix}.csv": cv_participants,
        f"rt_ladder_cv_predictions{suffix}.csv": cv_predictions,
        f"rt_ladder_bootstrap{suffix}.csv": bootstrap,
        f"rt_ladder_bootstrap_summary{suffix}.csv": bootstrap_ci,
        f"rt_ladder_bootstrap_correlations{suffix}.csv": bootstrap_corr,
        f"rt_ladder_recovery{suffix}.csv": recovery,
        f"rt_ladder_recovery_summary{suffix}.csv": recovery_ci,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)

    make_figure(
        ladder,
        cv,
        bootstrap_ci,
        recovery_ci,
        out / f"rt_ladder_diagnostic{suffix}",
    )
    print(ladder.round(4).to_string(index=False))
    print(f"\nSaved RT ladder outputs in {out}")


if __name__ == "__main__":
    main()
