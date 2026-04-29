"""Parameter recovery and identifiability analyses for the Piantadosi attractor fit.

Three analyses are exposed:

1. ``run_recovery_sweep``      — generate synthetic datasets from known ground
   truth parameters, refit, and compare recovered vs true to assess
   identifiability and per-parameter trade-offs.
2. ``run_profile_likelihood``  — fix one parameter at a grid of values, refit
   the rest, and report the loss profile (sharpness of the minimum).
3. ``run_dd_bootstrap``        — parametric bootstrap of the DD / control fit
   to verify that the tau_DD / tau_control ratio is robust.

All three reuse the existing fit machinery from ``scripts.fit_helpers`` but run
with a reduced DE budget (``popsize=6``, ``maxiter=80``, early stopping
``patience=3``) to keep the total runtime tractable across hundreds of fits.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm

# ``fork`` after the parent has initialized CUDA breaks child workers; ``spawn``
# starts a clean interpreter per worker (same rationale as typical torch MP).
_MP_SPAWN = mp.get_context("spawn")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from scripts.fit_helpers import (
    BOUNDS,
    FIXED_S0,
    FIXED_T0,
    N_TRUE_MIN,
    PARAM_NAMES,
    _extract_empirical_arrays,
    _simulate_core,
    complete_piantadosi_theta,
    hessian_ci,
    refine_cis_with_profile,
)


# ── Budget defaults ──────────────────────────────────────────────────────

RECOVERY_BUDGET = dict(
    popsize=6,
    maxiter=80,
    patience=3,
    es_start=15,
    min_delta=1e-4,
    tol=1e-4,
    n_samples=200,
    mc_final=1000,
)

LOG_PARAMS = {"sigma_c", "s0", "lambda0", "kappa", "c_steps"}


# ── Silent fit (no wandb, reduced budget) ────────────────────────────────


def _loss_for_emp_arrays(emp_arrays: dict, n_samples: int, seed: int) -> Callable:
    """Build a loss function closure over a fixed sample budget.

    The Piantadosi fit uses a 3-term loss (mean_guess + pct_exact + RT). When
    the synthetic targets do not include mean_guess (DeCarli-style), the
    corresponding term is omitted by passing ``nan`` as the empirical mean_guess.
    """
    include_mg = not np.all(np.isnan(emp_arrays["mean_guess"]))
    mask = emp_arrays["model_mask"]
    n_shown = emp_arrays["n_shown"]

    def _single(theta_vec: np.ndarray) -> float:
        theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, theta_vec)))
        try:
            model = _simulate_core(theta, n_samples_per_n=n_samples, seed=seed)
        except Exception:
            return 1e6

        mg_m = model["mean_guess"][mask]
        ac_m = model["pct_exact"][mask]
        rt_m = model["median_rt"][mask]

        total = 0.0
        n_terms = 0
        if include_mg:
            mg_e = emp_arrays["mean_guess"]
            total += np.mean((mg_e - mg_m) ** 2) / (np.ptp(mg_e) ** 2 + 1e-6)
            n_terms += 1
        ac_e = emp_arrays["pct_exact"]
        total += np.mean((ac_e - ac_m) ** 2) / (np.ptp(ac_e) ** 2 + 1e-6)
        n_terms += 1
        rt_e = emp_arrays["median_rt_s"]
        total += np.mean((rt_e - rt_m) ** 2) / (np.ptp(rt_e) ** 2 + 1e-6)
        n_terms += 1

        total = float(total)
        return 1e6 if (np.isnan(total) or np.isinf(total)) else total

    def loss_fn(theta_input: np.ndarray, *_unused) -> float | np.ndarray:
        if theta_input.ndim == 1:
            return _single(theta_input)
        pop_size = theta_input.shape[1]
        out = np.empty(pop_size)
        for i in range(pop_size):
            out[i] = _single(theta_input[:, i])
        return out

    return loss_fn


def fit_silent(
    emp_arrays: dict,
    fixed_params: dict[str, float] | None = None,
    budget: dict | None = None,
    seed: int = 17,
) -> tuple[dict, float]:
    """DE fit without wandb logging.

    When ``fixed_params`` is given, those entries of the PARAM_NAMES vector are
    clamped to the provided values (used by the profile-likelihood analysis).
    Returns the recovered theta and the final loss.
    """
    b = {**RECOVERY_BUDGET, **(budget or {})}
    fixed_params = fixed_params or {}

    free_idx = [i for i, p in enumerate(PARAM_NAMES) if p not in fixed_params]
    bounds_free = [BOUNDS[i] for i in free_idx]

    def expand(xk_free: np.ndarray) -> np.ndarray:
        full = np.empty(len(PARAM_NAMES))
        for i, p in enumerate(PARAM_NAMES):
            if p in fixed_params:
                full[i] = fixed_params[p]
            else:
                full[i] = xk_free[free_idx.index(i)]
        return full

    base_loss = _loss_for_emp_arrays(emp_arrays, b["n_samples"], seed=73)

    def wrapped(theta_input: np.ndarray, *_a) -> float | np.ndarray:
        if theta_input.ndim == 1:
            return base_loss(expand(theta_input))
        pop = theta_input.shape[1]
        full = np.empty((len(PARAM_NAMES), pop))
        for j in range(pop):
            full[:, j] = expand(theta_input[:, j])
        return base_loss(full)

    best_loss = [float("inf")]
    es_counter = [0]
    es_best = [float("inf")]
    gen_counter = [0]

    def callback(xk, convergence=0.0):
        gen_counter[0] += 1
        gen = gen_counter[0]

        cur = wrapped(np.asarray(xk))
        cur = float(cur) if np.ndim(cur) == 0 else float(np.min(cur))
        if cur < best_loss[0]:
            best_loss[0] = cur

        if gen >= b["es_start"]:
            if es_best[0] - best_loss[0] > b["min_delta"]:
                es_best[0] = best_loss[0]
                es_counter[0] = 0
            else:
                es_counter[0] += 1
            if es_counter[0] >= b["patience"]:
                return True
        return False

    result = differential_evolution(
        wrapped,
        bounds=bounds_free,
        maxiter=b["maxiter"],
        popsize=b["popsize"],
        seed=seed,
        tol=b["tol"],
        mutation=(0.5, 1.0),
        recombination=0.85,
        callback=callback,
        disp=False,
        polish=True,
        init="latinhypercube",
        vectorized=True,
        updating="deferred",
    )

    theta_full = complete_piantadosi_theta(
        dict(zip(PARAM_NAMES, expand(result.x)))
    )
    # Recompute final loss with a larger MC budget for reporting.
    final_loss_fn = _loss_for_emp_arrays(emp_arrays, b["mc_final"], seed=73)
    final_loss = float(final_loss_fn(expand(result.x)))
    return theta_full, final_loss


# ── Synthesis ────────────────────────────────────────────────────────────


def synthesize_targets(
    theta: dict,
    n_trials_per_n: int = 100,
    seed: int = 0,
    include_mean_guess: bool = True,
    n_shown_keep: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Generate a Piantadosi-like empirical DataFrame from the forward model.

    The ``n_trials_per_n`` should mimic the per-numerosity participant count
    so the synthetic aggregate statistics carry realistic sampling noise.
    If ``include_mean_guess`` is False, mean_guess is filled with NaN so that
    the DeCarli-style loss (accuracy + RT only) can be used downstream.
    """
    theta_full = complete_piantadosi_theta(dict(theta))
    sim = _simulate_core(theta_full, n_samples_per_n=n_trials_per_n, seed=seed)
    df = pd.DataFrame(
        {
            "n_shown": sim["n_shown"],
            "mean_guess": sim["mean_guess"] if include_mean_guess else np.nan,
            "pct_exact": sim["pct_exact"],
            "median_rt_s": sim["median_rt"],
            "count": n_trials_per_n,
        }
    )
    if n_shown_keep is not None:
        df = df[df["n_shown"].isin(list(n_shown_keep))].reset_index(drop=True)
    return df


def synth_emp_arrays(
    theta: dict,
    n_trials_per_n: int = 100,
    seed: int = 0,
    include_mean_guess: bool = True,
    n_shown_keep: Iterable[int] | None = None,
) -> dict:
    """Shortcut: synthesize targets and convert to the array dict expected by the loss."""
    df = synthesize_targets(
        theta, n_trials_per_n, seed, include_mean_guess, n_shown_keep
    )
    arr = _extract_empirical_arrays(df)
    if not include_mean_guess:
        arr["mean_guess"] = np.full_like(arr["pct_exact"], np.nan, dtype=float)
    return arr


# ── Ground-truth sampling ────────────────────────────────────────────────


def sample_theta_jittered(
    center: dict[str, float],
    bounds: list[tuple[float, float]] | None = None,
    jitter_frac: float = 0.35,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Draw a theta near ``center`` for use as ground truth.

    Log-scale params (``LOG_PARAMS``) are jittered multiplicatively; linear
    params additively. The result is clipped to the optimizer bounds.
    """
    rng = rng or np.random.default_rng()
    bounds = bounds or BOUNDS
    out: dict[str, float] = {}
    for p, (lo, hi) in zip(PARAM_NAMES, bounds):
        c = float(center[p])
        if p in LOG_PARAMS and c > 0:
            log_c = np.log(c)
            log_width = np.log1p(jitter_frac)
            val = np.exp(log_c + rng.uniform(-log_width, log_width))
        else:
            val = c + rng.uniform(-jitter_frac, jitter_frac) * max(abs(c), 0.1)
        out[p] = float(np.clip(val, lo, hi))
    for k, v in center.items():
        out.setdefault(k, v)
    return out


def _recovery_task_worker(task: dict) -> tuple[int, list[dict]]:
    ix = task["_task_ix"]
    emp_arr = synth_emp_arrays(
        task["theta_true"],
        n_trials_per_n=task["n_trials_per_n"],
        seed=task["seed"],
        include_mean_guess=task["include_mean_guess"],
        n_shown_keep=task["n_shown_keep"],
    )
    theta_hat, loss = fit_silent(
        emp_arr, budget=task["budget"], seed=task["seed"] % 10_000
    )
    rows: list[dict] = []
    for p in PARAM_NAMES:
        rows.append(
            {
                "label": task["label"],
                "run_id": task["run_id"],
                "param": p,
                "true": task["theta_true"][p],
                "recovered": theta_hat[p],
                "loss": loss,
                "seed": task["seed"],
            }
        )
    return ix, rows


def _profile_task_worker(task: dict) -> tuple[int, dict]:
    ix = task["_task_ix"]
    pn = task["param_name"]
    v = float(task["value"])
    theta_hat, loss = fit_silent(
        task["emp_arrays"],
        fixed_params={pn: v},
        budget=task["budget"],
    )
    row: dict = {"param_fixed": pn, "value": v, "loss": loss}
    for p in PARAM_NAMES:
        row[f"hat_{p}"] = theta_hat[p]
    return ix, row


def _dd_bootstrap_task_worker(task: dict) -> tuple[int, list[dict]]:
    ix = task["_task_ix"]
    run_id = task["run_id"]
    seed = task["seed"]
    rows: list[dict] = []
    for group_name, theta_true in [
        ("DD", task["theta_dd"]),
        ("Controls", task["theta_ctrl"]),
    ]:
        emp_arr = synth_emp_arrays(
            theta_true,
            n_trials_per_n=task["n_trials_per_n"],
            seed=seed + (0 if group_name == "DD" else 1),
            include_mean_guess=False,
            n_shown_keep=task["n_shown_keep"],
        )
        theta_hat, loss = fit_silent(
            emp_arr, budget=task["budget"], seed=seed % 10_000
        )
        for p in PARAM_NAMES:
            rows.append(
                {
                    "run_id": run_id,
                    "group": group_name,
                    "param": p,
                    "true": theta_true[p],
                    "recovered": theta_hat[p],
                    "loss": loss,
                    "seed": seed,
                }
            )
    return ix, rows


def _run_tasks_pool(
    worker: Callable[[dict], tuple[int, Any]],
    tasks: list[dict],
    n_jobs: int,
    *,
    verbose: bool,
    desc: str,
    unit: str,
) -> dict[int, Any]:
    if not tasks:
        return {}
    for i, t in enumerate(tasks):
        t["_task_ix"] = i
    n_workers = max(1, min(max(1, n_jobs), len(tasks)))
    out: dict[int, Any] = {}
    if n_workers == 1:
        it = tqdm(
            tasks,
            total=len(tasks),
            desc=desc,
            unit=unit,
            leave=True,
            disable=not verbose,
        )
        for t in it:
            ix, payload = worker(t)
            out[ix] = payload
            if verbose and hasattr(it, "set_postfix"):
                if isinstance(payload, list) and payload:
                    loss = payload[-1].get("loss")
                    if loss is not None:
                        it.set_postfix(loss=f"{float(loss):.4f}")
        return out
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=_MP_SPAWN,
    ) as ex:
        futures = {ex.submit(worker, t): t["_task_ix"] for t in tasks}
        pbar = tqdm(
            total=len(tasks),
            desc=desc,
            unit=unit,
            disable=not verbose,
        )
        try:
            for fut in as_completed(futures):
                ix, payload = fut.result()
                out[ix] = payload
                pbar.update(1)
        finally:
            pbar.close()
    return out


# ── Recovery sweep ────────────────────────────────────────────────────────


@dataclass
class RecoverySpec:
    label: str
    center: dict[str, float]
    n_shown_keep: Iterable[int] | None = None
    include_mean_guess: bool = True


def run_recovery_sweep(
    specs: list[RecoverySpec],
    n_runs: int,
    n_trials_per_n: int = 100,
    jitter_frac: float = 0.35,
    seed0: int = 13,
    budget: dict | None = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """For each spec, draw ``n_runs`` ground-truths, synthesize, refit, record.

    Ground-truth draws use the same RNG order as a serial loop so results match
    ``n_jobs=1``. With ``n_jobs>1``, fits run in worker processes and a single
    tqdm tracks completed tasks.

    Returns a long-format DataFrame with columns:
      label, run_id, param, true, recovered, loss, seed
    """
    tasks: list[dict] = []
    for spec in specs:
        rng = np.random.default_rng(seed0 + hash(spec.label) % 10_000)
        nsk = list(spec.n_shown_keep) if spec.n_shown_keep is not None else None
        for run_id in range(n_runs):
            seed = int(rng.integers(10_000, 1_000_000))
            theta_true = sample_theta_jittered(
                spec.center, jitter_frac=jitter_frac, rng=rng
            )
            tasks.append(
                {
                    "label": spec.label,
                    "run_id": run_id,
                    "seed": seed,
                    "theta_true": theta_true,
                    "n_trials_per_n": n_trials_per_n,
                    "include_mean_guess": spec.include_mean_guess,
                    "n_shown_keep": nsk,
                    "budget": budget,
                }
            )
    desc = "recovery" if len(specs) != 1 else f"recovery:{specs[0].label}"
    combined = _run_tasks_pool(
        _recovery_task_worker,
        tasks,
        n_jobs,
        verbose=verbose,
        desc=desc,
        unit="run",
    )
    rows: list[dict] = []
    for k in range(len(tasks)):
        rows.extend(combined[k])
    return pd.DataFrame(rows)


# ── Profile likelihood ────────────────────────────────────────────────────


def run_profile_likelihood(
    center: dict[str, float],
    param_name: str,
    grid: np.ndarray,
    emp_arrays: dict,
    budget: dict | None = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fix ``param_name`` at each value in ``grid``, refit the rest, record loss.

    Unlike the recovery sweep, this uses the REAL empirical ``emp_arrays``
    (typically for the 160 ms Piantadosi condition) and asks: how much does
    the data constrain each parameter individually?
    """
    _ = center
    tasks = [
        {
            "param_name": param_name,
            "value": float(v),
            "emp_arrays": emp_arrays,
            "budget": budget,
        }
        for v in grid
    ]
    combined = _run_tasks_pool(
        _profile_task_worker,
        tasks,
        n_jobs,
        verbose=verbose,
        desc=f"profile:{param_name}",
        unit="grid",
    )
    rows = [combined[i] for i in range(len(tasks))]
    return pd.DataFrame(rows)


# ── DD bootstrap ──────────────────────────────────────────────────────────


def run_dd_bootstrap(
    theta_dd: dict[str, float],
    theta_ctrl: dict[str, float],
    n_runs: int,
    n_trials_per_n: int = 32,
    n_shown_keep: Iterable[int] = range(1, 8),
    seed0: int = 91,
    budget: dict | None = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Parametric bootstrap of the DD / control fit.

    For each run, simulate a synthetic DD dataset from ``theta_dd`` and a
    synthetic control dataset from ``theta_ctrl`` (accuracy + RT only, matching
    the DeCarli protocol). Refit each and record the recovered parameters.
    Returns long-format columns: run_id, group, param, true, recovered, loss.
    """
    rng = np.random.default_rng(seed0)
    nsk = list(n_shown_keep) if n_shown_keep is not None else None
    tasks: list[dict] = []
    for run_id in range(n_runs):
        seed = int(rng.integers(10_000, 1_000_000))
        tasks.append(
            {
                "run_id": run_id,
                "seed": seed,
                "theta_dd": theta_dd,
                "theta_ctrl": theta_ctrl,
                "n_trials_per_n": n_trials_per_n,
                "n_shown_keep": nsk,
                "budget": budget,
            }
        )
    combined = _run_tasks_pool(
        _dd_bootstrap_task_worker,
        tasks,
        n_jobs,
        verbose=verbose,
        desc="dd bootstrap",
        unit="run",
    )
    rows: list[dict] = []
    for k in range(len(tasks)):
        rows.extend(combined[k])
    return pd.DataFrame(rows)


# ── DeCarli parameter recovery (Hopfield–Poisson, no anchor) ─────────────
#
# Ground-truth dicts: fitted parameters for the scenario plus any fields required
# by ``_simulate_core`` that are not supplied by ``complete_piantadosi_theta``
# (e.g. when ``s_0`` / ``fixed_t0`` are null in ``configs/config.yaml``).

DECARLI_RECOVERY_OUTPUT_DIR = _ROOT / "output" / "decarli_parameter_recovery"

THETA_CONTROLS_DECARLI = {
    "sigma_c": 0.0338,
    "lambda0": 2.3412,
    "kappa": 1.7778,
    "s0": 0.01,
    "t0": 0.3,
}

THETA_DD_DECARLI = {
    "sigma_c": 0.0407,
    "lambda0": 2.2393,
    "kappa": 1.8460,
    "s0": 0.01,
    "t0": 0.3,
}

THETA_SCENARIO3 = {
    "sigma_c": 0.0407,
    "lambda0": 2.3412,
    "kappa": 1.7778,
    "s0": 0.01,
    "t0": 0.3,
}

DECARLI_RECOVERY_SCENARIOS: list[tuple[str, dict[str, float]]] = [
    ("scenario1_controls", THETA_CONTROLS_DECARLI),
    ("scenario2_dd", THETA_DD_DECARLI),
    ("scenario3_mixed", THETA_SCENARIO3),
]


def synthesize_decarli_recovery_targets(
    theta: dict[str, float],
    *,
    rng: np.random.Generator,
    m_sim: int,
    sim_seed: int,
    n_shown_list: Iterable[int] | None = None,
    n_subjects: int = 32,
    trials_per_subject: int = 100,
    pct_jitter_frac: float = 0.02,
    rt_jitter_frac: float = 0.05,
) -> pd.DataFrame:
    """Group-level pct_exact and median_rt curves matching DeCarli aggregation.

    ``theta`` should contain the fitted-parameter ground truth only; fixed fields
    from config (``s_0``, ``fixed_t0``, ``lambda_0``, …) are merged in via
    ``complete_piantadosi_theta`` before the forward model runs.
    """
    if n_shown_list is None:
        n_shown_list = range(1, 8)
    n_shown_list = list(n_shown_list)
    theta_full = complete_piantadosi_theta(dict(theta))
    sim = _simulate_core(
        theta_full,
        n_samples_per_n=m_sim,
        seed=sim_seed,
        include_trials=True,
    )
    rows: list[dict] = []
    for n in n_shown_list:
        idx = int(n) - N_TRUE_MIN
        ex_row = sim["trial_exact"][idx]
        rt_row = sim["trial_rt"][idx]
        subj_pct: list[float] = []
        subj_mrt: list[float] = []
        for _ in range(n_subjects):
            pick = rng.choice(m_sim, size=trials_per_subject, replace=False)
            pct_i = 100.0 * float(ex_row[pick].mean())
            mrt_i = float(np.median(rt_row[pick]))
            subj_pct.append(pct_i)
            subj_mrt.append(mrt_i)
        pct_mean = float(np.mean(subj_pct))
        mrt_mean = float(np.mean(subj_mrt))
        pct_mean *= 1.0 + float(rng.uniform(-pct_jitter_frac, pct_jitter_frac))
        mrt_mean *= 1.0 + float(rng.uniform(-rt_jitter_frac, rt_jitter_frac))
        pct_mean = float(np.clip(pct_mean, 0.0, 100.0))
        rows.append(
            {
                "n_shown": int(n),
                "mean_guess": np.nan,
                "pct_exact": pct_mean,
                "median_rt_s": mrt_mean,
                "count": n_subjects,
            }
        )
    return pd.DataFrame(rows)


def decarli_recovery_ci_table(
    targets: pd.DataFrame,
    theta: dict,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Hessian 95% CIs with profile refinement, matching ``fit_decarli.py``."""
    from scripts import fit_decarli as fd

    nll_fn = fd._make_nll_fn(targets)
    theta_vec = np.array([theta[p] for p in PARAM_NAMES], dtype=float)
    ci = hessian_ci(nll_fn, theta_vec, PARAM_NAMES, BOUNDS)
    needs = ci.apply(
        lambda r: (not np.isfinite(r["se"]))
        or r["status"] in ("non_pd", "singular", "at_bound"),
        axis=1,
    )
    if needs.any():
        nll_of_theta = fd._make_nll_of_theta(targets)
        fit_with_fixed = fd._make_fit_with_fixed_decarli(targets)
        nll_hat = float(nll_fn(theta_vec))
        ci = refine_cis_with_profile(
            ci,
            nll_of_theta=nll_of_theta,
            fit_with_fixed=fit_with_fixed,
            theta_hat=theta,
            nll_hat=nll_hat,
            bounds=BOUNDS,
            verbose=verbose,
        )
    return ci


def _decarli_recovery_worker(payload: dict) -> dict:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    from scripts.fit_decarli import fit_decarli_from_arrays

    scenario = payload["scenario"]
    run_id = payload["run_id"]
    theta_true = payload["theta_true"]
    rng = np.random.default_rng(int(payload["rng_seed"]))
    targets = synthesize_decarli_recovery_targets(
        theta_true,
        rng=rng,
        m_sim=int(payload["m_sim"]),
        sim_seed=int(payload["sim_seed"]),
    )
    emp = _extract_empirical_arrays(targets)
    emp["mean_guess"] = np.full_like(emp["pct_exact"], np.nan, dtype=float)

    theta_hat, result = fit_decarli_from_arrays(
        emp,
        label="",
        de_seed=int(payload["fit_seed"]),
        verbose=False,
    )
    ci = decarli_recovery_ci_table(targets, theta_hat, verbose=False)

    detail_rows: list[dict] = []
    ci_by = {r["param"]: r for _, r in ci.iterrows()}
    for p in PARAM_NAMES:
        if p not in theta_true:
            raise KeyError(
                f"Scenario ground truth missing fitted param {p!r}; extend the "
                f"scenario dict or adjust config so {p!r} is fitted."
            )
        true_v = float(theta_true[p])
        row = ci_by[p]
        lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
        fit_v = float(theta_hat[p])
        detail_rows.append(
            {
                "scenario": scenario,
                "run_id": run_id,
                "param_name": p,
                "true_value": true_v,
                "fitted_value": fit_v,
                "fitted_ci_lo": lo,
                "fitted_ci_hi": hi,
                "ci_contains_true": bool(lo <= true_v <= hi),
                "loss": float(result.fun),
            }
        )
    return {"detail_rows": detail_rows, "loss": float(result.fun)}


def summarize_decarli_recovery(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario in sorted(detail_df["scenario"].unique()):
        sub_s = detail_df[detail_df["scenario"] == scenario]
        for p in PARAM_NAMES:
            sub = sub_s[sub_s["param_name"] == p]
            if sub.empty:
                continue
            true_v = float(sub["true_value"].iloc[0])
            fit = sub["fitted_value"].astype(float)
            bias = float(fit.mean() - true_v)
            var = float(fit.std(ddof=1)) if len(fit) > 1 else 0.0
            rmse = float(np.sqrt(np.mean((fit - true_v) ** 2)))
            coverage = float(sub["ci_contains_true"].mean())
            rows.append(
                {
                    "scenario": scenario,
                    "param_name": p,
                    "bias": bias,
                    "variance": var,
                    "rmse": rmse,
                    "coverage": coverage,
                    "mean_fitted": float(fit.mean()),
                    "n_runs": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def plot_decarli_parameter_recovery(
    detail_df: pd.DataFrame,
    scenarios: list[tuple[str, dict[str, float]]],
    out_path: Path,
    dd_reference: dict[str, float] | None = None,
) -> None:
    """Rows = scenarios, columns = parameters; scenario 3 shows DD dashed refs."""
    dd_axis_refs = [p for p in ("s0", "lambda0") if p in PARAM_NAMES]
    scenario_keys = [s[0] for s in scenarios]
    n_row, n_col = len(scenario_keys), len(PARAM_NAMES)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(2.6 * n_col, 2.8 * n_row),
        squeeze=False,
    )
    for i, sk in enumerate(scenario_keys):
        theta_true = dict(scenarios[i][1])
        for j, p in enumerate(PARAM_NAMES):
            ax = axes[i][j]
            sub = detail_df[
                (detail_df["scenario"] == sk) & (detail_df["param_name"] == p)
            ]
            vals = sub["fitted_value"].astype(float).values
            ax.boxplot(vals, widths=0.45, showfliers=True)
            ax.axhline(theta_true[p], color="C0", linewidth=1.2)
            if (
                sk == "scenario3_mixed"
                and dd_reference is not None
                and p in dd_axis_refs
            ):
                ax.axhline(
                    dd_reference[p],
                    color="C3",
                    linestyle="--",
                    linewidth=1.0,
                )
            ax.set_xticks([1])
            ax.set_xticklabels([p], rotation=30, ha="right")
            if j == 0:
                ax.set_ylabel(sk.replace("_", " "))
            if i == 0:
                ax.set_title(p)
    if dd_axis_refs == ["s0", "lambda0"]:
        cap = "Solid blue: ground truth; dashed red (scenario 3, s0 & λ0): DD point estimates"
    elif dd_axis_refs == ["lambda0"]:
        fixed = []
        if "s0" not in PARAM_NAMES and FIXED_S0 is not None:
            fixed.append(f"s0={FIXED_S0:g}")
        if "t0" not in PARAM_NAMES and FIXED_T0 is not None:
            fixed.append(f"t0={FIXED_T0:g}")
        note = f" ({', '.join(fixed)} from config)" if fixed else ""
        cap = (
            "Solid blue: ground truth; dashed red (scenario 3, λ0): DD point estimate"
            + note
        )
    else:
        cap = "Solid blue: ground truth per parameter"
    fig.suptitle(cap, fontsize=9, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_decarli_recovery_readme(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Recuperació de paràmetres (DeCarli, Hopfield–Poisson)\n\n")
    lines.append(
        "Dades sintètiques: model endavant amb pous de Poisson sense ancoratge, "
        "resmostreig per subjecte com al protocol DeCarli (32 subjectes, 100 "
        "assajos per n i numerositat), més jitter observacional (±2% a "
        "`pct_exact`, ±5% a `median_rt_s`). "
        "L'ajust replica `fit_decarli.py` (evolució diferencial + polish L-BFGS). "
        "Els IC 95% combinen Hessiana en espai log i perfil de versemblança quan cal.\n\n"
    )
    fixed_bits: list[str] = []
    if "s0" not in PARAM_NAMES and FIXED_S0 is not None:
        fixed_bits.append(f"`s0` = {FIXED_S0:g} (`s_0` al config)")
    if "t0" not in PARAM_NAMES and FIXED_T0 is not None:
        fixed_bits.append(f"`t0` = {FIXED_T0:g} (`fixed_t0` al config)")
    if fixed_bits:
        lines.append(
            "Paràmetres fixats (no es recuperen en aquesta corrida): "
            + "; ".join(fixed_bits)
            + ". Els escenaris només fixen valors vertaders per a `PARAM_NAMES`.\n\n"
        )

    s3 = summary_df[summary_df["scenario"] == "scenario3_mixed"]
    s0_rows = s3[s3["param_name"] == "s0"]
    lam_rows = s3[s3["param_name"] == "lambda0"]
    if not s3.empty and "s0" in PARAM_NAMES and len(s0_rows) and len(lam_rows):
        m_s0 = float(s0_rows.iloc[0]["mean_fitted"])
        m_lam = float(lam_rows.iloc[0]["mean_fitted"])
        lines.append("## Escenari 3 (σ_c de DD, paràmetres sensorials de controls)\n\n")
        lines.append(
            f"- Mitjana `s0` recuperada = {m_s0:.4f} "
            f"(vertader 0.0104; ajust DD observat 0.0145)\n"
        )
        lines.append(
            f"- Mitjana `lambda0` recuperada = {m_lam:.4f} "
            f"(vertader 2.3412; ajust DD observat 2.2393)\n\n"
        )
        lines.append("**Interpretació (criteris pre-registrats):**\n\n")
        if m_s0 > 0.012 or m_lam < 2.30:
            label = "Contaminació (compensació amb σ_c): les diferències sensorials aparents podrien ser artefacte."
        elif abs(m_s0 - 0.0104) < 0.002 and abs(m_lam - 2.3412) < 0.05:
            label = (
                "Recuperació neta: les diferències identificades al paper són plausiblament reals i independents."
            )
        else:
            label = "Intermedi: identificabilitat parcial; cal cautela en el text del article."
        lines.append(f"- {label}\n\n")
    elif not s3.empty and "lambda0" in PARAM_NAMES and len(lam_rows):
        m_lam = float(lam_rows.iloc[0]["mean_fitted"])
        lines.append(
            "## Escenari 3 (σ_c de DD, λ0 de controls; `s0`/`t0` fixats al config si aplica)\n\n"
        )
        lines.append(
            f"- Mitjana `lambda0` recuperada = {m_lam:.4f} "
            f"(vertader 2.3412; ajust DD observat 2.2393)\n\n"
        )
        lines.append(
            "**Interpretació (només `lambda0` lliure entre paràmetres sensorials típics):**\n\n"
        )
        if m_lam < 2.30:
            label = "Contaminació (compensació amb σ_c): `lambda0` es tira cap al règim DD."
        elif abs(m_lam - 2.3412) < 0.05:
            label = "Recuperació neta de `lambda0`: el canal d'exponent és identificable malgrat σ_c de DD."
        else:
            label = "Intermedi: identificabilitat parcial de `lambda0`; cal cautela al text."
        lines.append(f"- {label}\n\n")
    elif not s3.empty:
        lines.append(
            "## Escenari 3\n\n"
            "Falten files al resum per interpretar l'escenari (revisa `PARAM_NAMES`).\n\n"
        )

    lines.append("## Valors de pèrdua (convergència)\n\n")
    for sk in sorted(detail_df["scenario"].unique()):
        sub = detail_df[detail_df["scenario"] == sk]
        losses = sub.groupby("run_id")["loss"].first()
        med = float(losses.median())
        bad = losses[losses > 5.0 * med]
        lines.append(
            f"- `{sk}`: pèrdua mediana = {med:.6f}; "
            f"runs amb pèrdua > 5× la mediana: {len(bad)} / {len(losses)}\n"
        )

    lines.append("\n## Resum numèric (bias, variància, RMSE, cobertura)\n\n")
    lines.append("```\n")
    lines.append(summary_df.to_string(index=False))
    lines.append("\n```\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def run_decarli_parameter_recovery(
    *,
    output_dir: Path = DECARLI_RECOVERY_OUTPUT_DIR,
    m_runs: int = 20,
    m_sim: int = 5000,
    n_jobs: int = 4,
    global_seed: int = 1234,
    write_readme: bool = True,
    scenarios_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_list = DECARLI_RECOVERY_SCENARIOS
    if scenarios_filter is not None:
        scenario_list = [s for s in scenario_list if s[0] in scenarios_filter]
        if not scenario_list:
            raise ValueError("No scenarios left after --only filter.")
    payloads: list[dict] = []
    for scen_idx, (scenario, theta_true) in enumerate(scenario_list):
        for run_id in range(m_runs):
            base = global_seed + scen_idx * 100_000 + run_id * 9_779
            payloads.append(
                {
                    "scenario": scenario,
                    "run_id": run_id,
                    "theta_true": theta_true,
                    "m_sim": m_sim,
                    "rng_seed": base + 11,
                    "sim_seed": base + 3,
                    "fit_seed": base + 29,
                }
            )

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_MP_SPAWN) as pool:
            outs = list(
                tqdm(
                    pool.map(_decarli_recovery_worker, payloads, chunksize=1),
                    total=len(payloads),
                    desc="DeCarli recovery",
                    unit="run",
                )
            )
    else:
        outs = [
            _decarli_recovery_worker(p)
            for p in tqdm(
                payloads,
                total=len(payloads),
                desc="DeCarli recovery",
                unit="run",
            )
        ]

    detail_rows: list[dict] = []
    for o in outs:
        detail_rows.extend(o["detail_rows"])
    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_decarli_recovery(detail_df)

    detail_cols = [
        "run_id",
        "param_name",
        "true_value",
        "fitted_value",
        "fitted_ci_lo",
        "fitted_ci_hi",
        "ci_contains_true",
    ]
    for scenario, _ in scenario_list:
        sub = detail_df[detail_df["scenario"] == scenario]
        sub[detail_cols].to_csv(output_dir / f"{scenario}_detail.csv", index=False)
    summary_df.to_csv(output_dir / "recovery_summary.csv", index=False)

    plot_decarli_parameter_recovery(
        detail_df,
        scenario_list,
        output_dir / "parameter_recovery_boxplots.png",
        dd_reference=THETA_DD_DECARLI,
    )
    if write_readme:
        write_decarli_recovery_readme(summary_df, detail_df, output_dir / "README.md")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter recovery for the DeCarli Hopfield–Poisson fit."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DECARLI_RECOVERY_OUTPUT_DIR,
        help="Directory for CSVs, figure, and README.",
    )
    parser.add_argument(
        "--m-runs",
        type=int,
        default=20,
        help="Synthetic datasets per scenario (default 20).",
    )
    parser.add_argument(
        "--m-sim",
        type=int,
        default=5000,
        help="Monte Carlo trials per numerosity in synthesis (default 5000).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel workers (default 4; use 1 for serial).",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=1234,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip writing README.md (tables are still saved as CSV).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[s[0] for s in DECARLI_RECOVERY_SCENARIOS],
        metavar="SCENARIO",
        help="Run only these scenario ids (e.g. scenario3_mixed). Default: all.",
    )
    args = parser.parse_args()
    scen_f = set(args.only) if args.only else None
    run_decarli_parameter_recovery(
        output_dir=args.output_dir,
        m_runs=args.m_runs,
        m_sim=args.m_sim,
        n_jobs=args.n_jobs,
        global_seed=args.global_seed,
        write_readme=not args.no_readme,
        scenarios_filter=scen_f,
    )


if __name__ == "__main__":
    main()
