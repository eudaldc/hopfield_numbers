"""Tests for the summary-level model comparison script."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import model_comparison as mc


def test_model_specs_have_four_free_parameters() -> None:
    assert len(mc.MODEL_SPECS) == 4
    for spec in mc.MODEL_SPECS:
        assert spec.k_params == 4
        assert len(spec.bounds) == spec.k_params


def test_baseline_predictions_have_expected_columns() -> None:
    n_values = np.arange(1, 16, dtype=float)
    eps = mc.make_eps(n_samples=20, seed=19)
    for spec in mc.MODEL_SPECS:
        theta = np.array([(lo + hi) / 2.0 for lo, hi in spec.bounds])
        pred = spec.predict(theta, n_values, eps)
        assert list(pred.columns) == ["n_shown", "mean_guess", "pct_exact", "median_rt"]
        assert pred.shape[0] == n_values.size
        assert np.isfinite(pred[["mean_guess", "pct_exact", "median_rt"]].to_numpy()).all()
        assert ((pred["pct_exact"] >= 0.0) & (pred["pct_exact"] <= 100.0)).all()


def test_profile_metrics_include_information_criteria() -> None:
    emp = pd.DataFrame(
        {
            "n_shown": [1, 2, 3],
            "mean_guess": [1.0, 2.0, 3.0],
            "pct_exact": [95.0, 90.0, 85.0],
            "median_rt_s": [1.1, 1.2, 1.3],
        }
    )
    pred = pd.DataFrame(
        {
            "n_shown": [1, 2, 3],
            "mean_guess": [1.1, 1.9, 3.1],
            "pct_exact": [94.0, 88.0, 84.0],
            "median_rt": [1.0, 1.25, 1.35],
        }
    )
    metrics = mc.profile_metrics(emp, pred, k_params=4)
    assert metrics["k_params"] == 4
    assert metrics["n_data"] == 9
    assert np.isfinite(metrics["AIC"])
    assert np.isfinite(metrics["BIC"])


def test_latex_rows_contains_each_model() -> None:
    summary = pd.DataFrame(
        {
            "model": ["Hopfield-Poisson", "Weber/ANS baseline"],
            "k_params_per_duration": [4, 4],
            "AIC_mean": [10.0, 12.0],
            "BIC_mean": [11.0, 13.0],
            "lono_loss": [0.1, 0.2],
            "lodo_loss": [0.3, 0.4],
        }
    )
    rows = mc.latex_table_rows(summary)
    assert "Hopfield-Poisson" in rows
    assert "Weber/ANS baseline" in rows
