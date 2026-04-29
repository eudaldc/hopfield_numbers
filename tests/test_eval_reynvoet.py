"""Landscape sanity checks for scripts.eval_reynvoet.

The model is fit on Piantadosi data (numerosities 1-15) but evaluated on
Reynvoet Panamath pairs that go up to n=40. These tests lock in the
invariants that keep the Reynvoet evaluation from silently truncating
the posterior at the fitting range:

  1. The integer landscape covers every Panamath stimulus with margin.
  2. An n=40 input is not clipped to the Piantadosi n_max (=15): the
     posterior for n=40 places its mode above 15.
  3. predict_reynvoet runs end-to-end without NaN/Inf for every ratio,
     including the ones that involve n_l = 40.
  4. The setup_wells helper refuses a support that is too small (regression
     guard against future edits that would silently reintroduce a 15-well
     landscape).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import erf

from scripts.fit_helpers import psychometric_weber
from scripts.eval_reynvoet import (
    LANDSCAPE_MARGIN,
    N_MIN,
    N_SAMPLES,
    N_SUPPORT_MAX,
    PANAMATH_N_MAX,
    PANAMATH_N_MIN,
    TARGET_RATIOS,
    _load_fitted_params,
    _process,
    _setup_wells,
    enumerate_pairs,
    infer_n_support_max,
    predict_reynvoet,
    simulate_reynvoet_2afc,
    trialwise_accuracy_one_pair,
)


SEED = 101


@pytest.fixture(scope="module")
def theta_ref() -> dict:
    """Fitted parameters for the 640 ms duration (closest to Reynvoet's 500 ms)."""
    return _load_fitted_params(640)


def test_support_max_covers_panamath() -> None:
    """The landscape must include every presented Panamath numerosity."""
    assert N_SUPPORT_MAX >= PANAMATH_N_MAX
    assert N_SUPPORT_MAX - PANAMATH_N_MAX >= LANDSCAPE_MARGIN
    assert N_MIN <= PANAMATH_N_MIN


def test_psychometric_weber_uses_two_array_comparison_noise() -> None:
    """The fitted Weber fraction is for 2AFC comparisons, not single-array noise."""
    ratio = np.array([1.0, 1.5, 2.0])
    w = 0.2
    expected = 50.0 * (
        1.0 + erf((ratio - 1.0) / (w * np.sqrt(2.0 * (ratio * ratio + 1.0))))
    )
    np.testing.assert_allclose(psychometric_weber(ratio, w), expected)


def test_infer_n_support_max_applies_margin() -> None:
    """``infer_n_support_max`` respects the margin and rejects bad inputs."""
    assert infer_n_support_max(PANAMATH_N_MAX, LANDSCAPE_MARGIN) == N_SUPPORT_MAX
    assert infer_n_support_max(40, 10) == 50
    with pytest.raises(ValueError):
        infer_n_support_max(PANAMATH_N_MIN - 1)
    with pytest.raises(ValueError):
        infer_n_support_max(PANAMATH_N_MAX, margin=-1)


def test_setup_wells_contains_integers_beyond_15(theta_ref: dict) -> None:
    """Wells must extend well past the Piantadosi fitting range (n<=15)."""
    wells = _setup_wells(theta_ref)
    integers = wells["integers"]
    assert int(integers.max().item()) == N_SUPPORT_MAX
    assert int(integers.max().item()) >= PANAMATH_N_MAX
    # Every integer from 16..40 must be represented as a well center.
    beyond_fit = torch.arange(16, PANAMATH_N_MAX + 1, dtype=torch.float32)
    present = torch.isin(beyond_fit, integers)
    assert bool(present.all()), "wells do not cover integers 16..40"


def test_setup_wells_rejects_truncated_support(theta_ref: dict) -> None:
    """A request for a 1..15 landscape must be refused (regression guard)."""
    with pytest.raises(ValueError, match="truncated"):
        _setup_wells(theta_ref, n_min=1, n_max=15)


def test_predict_n40_not_clipped(theta_ref: dict) -> None:
    """Posterior for true n=40 must concentrate mass above the Piantadosi range.

    If the model were silently clipping inputs or wells at n=15 the posterior
    would pile up against the upper edge of [1, 15]. Instead the expected
    integer should sit well above 15.
    """
    wells = _setup_wells(theta_ref)
    g = torch.Generator().manual_seed(SEED)
    out = _process(PANAMATH_N_MAX, theta_ref, wells, n_samples=512, generator=g)
    probs = out["probs"]

    assert probs.shape[-1] == N_SUPPORT_MAX
    assert torch.isfinite(probs).all()
    row_sums = probs.sum(dim=-1)
    np.testing.assert_allclose(row_sums.cpu().numpy(), 1.0, atol=1e-4)

    integers = wells["integers"].to(probs.device)
    expected = (probs * integers).sum(dim=-1).mean().item()
    assert expected > 15.0, (
        f"E[k|x] for true n=40 is {expected:.2f} (<=15): posterior looks "
        "clipped to the Piantadosi fitting range"
    )
    # Also: a non-trivial share of the mass must fall strictly above 15.
    mass_above_15 = probs[:, (integers > 15.0)].sum(dim=-1).mean().item()
    assert mass_above_15 > 0.5, (
        f"only {mass_above_15:.2f} of the posterior mass sits above n=15; "
        "the landscape appears truncated"
    )


def test_predict_reynvoet_covers_all_ratios(theta_ref: dict) -> None:
    """End-to-end prediction must return a finite accuracy for every ratio."""
    # Few samples keeps the smoke test < 10 s; numerical correctness lives in
    # the dedicated posterior tests above.
    pred = predict_reynvoet(theta_ref, n_samples=128, seed=SEED)
    assert set(pred["ratio"].tolist()) == set(TARGET_RATIOS)
    acc = pred["accuracy"].to_numpy()
    assert np.isfinite(acc).all(), f"non-finite accuracies: {acc}"
    assert (acc >= 0.0).all() and (acc <= 100.0).all()
    # Largest ratio (2.0) uses the largest pairs (up to n_l=40); it must not
    # be dropped for lack of valid pairs.
    pairs_r2 = enumerate_pairs(2.0)
    assert any(n_l == PANAMATH_N_MAX for _, n_l in pairs_r2)


def test_process_n40_no_nans(theta_ref: dict) -> None:
    """A single forward pass at n=40 must be finite (no NaN/Inf anywhere)."""
    wells = _setup_wells(theta_ref)
    g = torch.Generator().manual_seed(SEED + 1)
    out = _process(PANAMATH_N_MAX, theta_ref, wells, n_samples=N_SAMPLES, generator=g)
    probs = out["probs"]
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()


def test_simulate_equal_numerosity_near_chance(theta_ref: dict) -> None:
    """Identical arrays: point-estimate 2AFC scores must average ~50% correct."""
    wells = _setup_wells(theta_ref)
    g = torch.Generator().manual_seed(SEED + 11)
    acc = trialwise_accuracy_one_pair(
        theta_ref, 24, 24, wells=wells, n_trials=6000, generator=g
    )
    assert np.isfinite(acc)
    assert 43.0 <= acc <= 57.0


def test_simulate_large_gap_more_accurate_than_small_gap(theta_ref: dict) -> None:
    """Easier pairs (larger count separation) yield higher accuracy."""
    wells = _setup_wells(theta_ref)
    g1 = torch.Generator().manual_seed(SEED + 3)
    g2 = torch.Generator().manual_seed(SEED + 4)
    easy = trialwise_accuracy_one_pair(
        theta_ref, 10, 40, wells=wells, n_trials=2500, generator=g1
    )
    hard = trialwise_accuracy_one_pair(
        theta_ref, 30, 32, wells=wells, n_trials=2500, generator=g2
    )
    assert np.isfinite(easy) and np.isfinite(hard)
    assert easy > hard


def test_simulate_reproducible_with_seed(theta_ref: dict) -> None:
    """Same seed ⇒ identical summarized curve."""
    a = simulate_reynvoet_2afc(
        theta_ref, n_trials=256, seed=SEED + 19, target_ratios=[1.5, 2.0]
    )
    b = simulate_reynvoet_2afc(
        theta_ref, n_trials=256, seed=SEED + 19, target_ratios=[1.5, 2.0]
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_simulate_reynvoet_curve_finite_bounds(theta_ref: dict) -> None:
    """Full ratio sweep: accuracies in [0, 100], no NaNs; includes n>15 pairs."""
    pred = simulate_reynvoet_2afc(theta_ref, n_trials=128, seed=SEED)
    assert set(pred["ratio"].tolist()) == set(TARGET_RATIOS)
    acc = pred["accuracy"].to_numpy()
    assert np.isfinite(acc).all()
    assert (acc >= 0.0).all() and (acc <= 100.0).all()
    pairs_2 = enumerate_pairs(2.0)
    assert any(max(ns, nl) > 15 for ns, nl in pairs_2)
