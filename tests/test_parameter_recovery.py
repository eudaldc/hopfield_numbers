"""Smoke tests for the parameter-recovery / identifiability pipeline.

These run with a minimal DE budget (a few iterations) to verify that the
end-to-end flow works; they do NOT assert identifiability (which requires
many more runs and much more compute).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.fit_helpers import BOUNDS, PARAM_NAMES
from scripts.parameter_recovery import (
    RECOVERY_BUDGET,
    RecoverySpec,
    THETA_CONTROLS_DECARLI,
    fit_silent,
    run_dd_bootstrap,
    run_profile_likelihood,
    run_recovery_sweep,
    sample_theta_jittered,
    synth_emp_arrays,
    synthesize_decarli_recovery_targets,
    synthesize_targets,
)


TINY_BUDGET = {
    **RECOVERY_BUDGET,
    "popsize": 4,
    "maxiter": 6,
    "patience": 2,
    "es_start": 3,
    "n_samples": 40,
    "mc_final": 80,
}


CENTER = {
    "sigma_c": 0.042,
    "tau_in": 0.98,
    "s0": 1.0,
    "lambda0": 1.0,
    "t0": 1.30,
    "kappa": 0.99,
}


# ── Synthesis ────────────────────────────────────────────────────────────


def test_synthesize_targets_columns():
    df = synthesize_targets(CENTER, n_trials_per_n=30, seed=1)
    assert set(df.columns) >= {"n_shown", "mean_guess", "pct_exact", "median_rt_s", "count"}
    assert len(df) >= 10
    assert df["pct_exact"].between(0, 100).all()
    assert (df["median_rt_s"] > 0).all()


def test_synthesize_targets_no_mean_guess():
    df = synthesize_targets(CENTER, n_trials_per_n=30, seed=1, include_mean_guess=False)
    assert df["mean_guess"].isna().all()


def test_synthesize_targets_filter_n_shown():
    df = synthesize_targets(
        CENTER, n_trials_per_n=30, seed=1, n_shown_keep=[1, 2, 3]
    )
    assert set(df["n_shown"].unique()) == {1, 2, 3}


def test_synthesize_decarli_recovery_targets_shape():
    rng = np.random.default_rng(17)
    df = synthesize_decarli_recovery_targets(
        THETA_CONTROLS_DECARLI,
        rng=rng,
        m_sim=400,
        sim_seed=101,
        n_subjects=8,
        trials_per_subject=50,
    )
    assert len(df) == 7
    assert list(df["n_shown"]) == list(range(1, 8))
    assert df["pct_exact"].between(0, 100).all()
    assert (df["median_rt_s"] > 0).all()


# ── Jittered sampling ────────────────────────────────────────────────────


def test_sample_theta_within_bounds():
    rng = np.random.default_rng(7)
    for _ in range(20):
        th = sample_theta_jittered(CENTER, rng=rng)
        for (p, (lo, hi)) in zip(PARAM_NAMES, BOUNDS):
            assert lo <= th[p] <= hi, f"{p}={th[p]} outside [{lo}, {hi}]"


# ── Fit silent ───────────────────────────────────────────────────────────


def test_fit_silent_runs_and_returns_theta():
    emp = synth_emp_arrays(CENTER, n_trials_per_n=30, seed=11)
    theta, loss = fit_silent(emp, budget=TINY_BUDGET, seed=11)
    assert set(theta.keys()) >= set(PARAM_NAMES)
    assert np.isfinite(loss)
    assert loss < 10.0, f"smoke fit unexpectedly bad: loss={loss}"


def test_fit_silent_with_fixed_param():
    emp = synth_emp_arrays(CENTER, n_trials_per_n=30, seed=11)
    theta, _ = fit_silent(
        emp, fixed_params={"sigma_c": 0.04}, budget=TINY_BUDGET, seed=11
    )
    assert theta["sigma_c"] == pytest.approx(0.04)


# ── End-to-end sweep ─────────────────────────────────────────────────────


def test_recovery_sweep_end_to_end():
    specs = [RecoverySpec(label="smoke", center=CENTER, include_mean_guess=True)]
    df = run_recovery_sweep(
        specs,
        n_runs=2,
        n_trials_per_n=30,
        jitter_frac=0.2,
        budget=TINY_BUDGET,
        verbose=False,
        n_jobs=2,
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"label", "run_id", "param", "true", "recovered", "loss"}
    assert set(df["param"].unique()) == set(PARAM_NAMES)
    assert len(df["run_id"].unique()) == 2


# ── Profile likelihood ───────────────────────────────────────────────────


def test_profile_likelihood_shape():
    emp = synth_emp_arrays(CENTER, n_trials_per_n=30, seed=11)
    grid = np.array([0.03, 0.06])
    df = run_profile_likelihood(
        CENTER,
        "sigma_c",
        grid,
        emp,
        budget=TINY_BUDGET,
        verbose=False,
        n_jobs=2,
    )
    assert len(df) == len(grid)
    assert list(df["value"]) == list(grid)
    assert np.isfinite(df["loss"]).all()


# ── DD bootstrap ─────────────────────────────────────────────────────────


def test_dd_bootstrap_returns_both_groups():
    dd = {**CENTER, "tau_in": 10.0}
    ct = {**CENTER, "tau_in": 0.9}
    df = run_dd_bootstrap(
        dd,
        ct,
        n_runs=2,
        n_trials_per_n=30,
        n_shown_keep=range(1, 6),
        budget=TINY_BUDGET,
        verbose=False,
        n_jobs=2,
    )
    assert set(df["group"].unique()) == {"DD", "Controls"}
    assert len(df["run_id"].unique()) == 2
