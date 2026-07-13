"""Tests for the vendored Cheyette & Piantadosi (2020) model and its fit.

These are self-contained (no external clone needed): they check the mathematical
contract of the info-bottleneck posterior and that the exact-C&P fitting layer
runs end to end and produces a valid likelihood.
"""

from __future__ import annotations

import numpy as np

from scripts import cheyette_model as cp
from scripts import trial_level_cp_exact as CPX


def test_prior_normalised_and_zipfian() -> None:
    x = np.arange(1, 20, dtype=float)
    p = cp.P(x, a=2.0)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    # Zipf(2): ratio P(1)/P(2) == 2^2 == 4.
    assert abs(p[0] / p[1] - 4.0) < 1e-12


def test_posterior_rows_normalised() -> None:
    cache = cp.build_q_cache(max_n=8, max_k=12, max_info=4.0, n_steps=2000)
    for mat in cache["q_cache"]:
        assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-8)


def test_kl_matches_info_bound() -> None:
    # find_q_nk should drive KL(Q||P) to the requested budget -- except where the
    # budget exceeds the model ceiling. The maximum achievable KL for numerosity
    # n is -log2(P(n)) (a point mass at k = n), so small, high-prior n saturate
    # below the budget. This is a faithful property of the C&P model, not a bug.
    max_n, max_k = 10, 15
    cache = cp.build_q_cache(max_n=max_n, max_k=max_k, max_info=4.0, n_steps=10000)
    p_k = cache["p_ks"]
    ceiling = -np.log2(p_k[:max_n])  # per-numerosity max KL (k index n-1)
    for target in (1.0, 2.0, 3.0):
        idx = int(round(target / cache["info_step"]))
        q = cache["q_cache"][idx]
        kl = cp.KL(np.tile(p_k, (q.shape[0], 1)), q)
        assert np.all(kl <= target + 0.05), (target, kl)  # never exceeds budget
        assert np.allclose(kl, np.minimum(target, ceiling), atol=0.06), (target, kl)


def test_more_bits_sharpen_the_posterior() -> None:
    # Higher information budget -> lower posterior entropy (sharper Q).
    cache = cp.build_q_cache(max_n=10, max_k=15, max_info=5.0, n_steps=8000)
    def entropy_at(bits: float) -> float:
        q = cp.lookup_q(cache, bits)
        return float(-(q * np.log2(q + 1e-12)).sum(axis=1).mean())
    assert entropy_at(4.0) < entropy_at(2.0) < entropy_at(1.0)


def test_lookup_interpolation_between_grid_points() -> None:
    cache = cp.build_q_cache(max_n=6, max_k=10, max_info=3.0, n_steps=2000)
    lo = cp.lookup_q(cache, 1.0)
    hi = cp.lookup_q(cache, 1.1)
    mid = cp.lookup_q_interp(cache, 1.05)
    assert np.all((mid <= np.maximum(lo, hi) + 1e-9))
    assert np.all((mid >= np.minimum(lo, hi) - 1e-9))


def test_cp_response_loglik_is_finite_and_negative() -> None:
    # Small synthetic count table over the well support: the pooled response
    # log-likelihood must be finite for reasonable (B, R, eps).
    rng = np.random.default_rng(0)
    r_max = CPX.N_SUPPORT_MAX + 5
    cache = cp.build_q_cache(min_n=CPX.N_TRUE_MIN, max_n=int(CPX.N_SHOWN[-1]),
                             max_k=CPX.N_SUPPORT_MAX, max_info=8.0, n_steps=3000)

    class _D:
        def __init__(self, counts):
            self.counts = counts
            self.n_total = int(counts.sum())

    data_by_dur = {}
    for d in CPX.DURATIONS:
        counts = rng.integers(0, 6, size=(CPX.N_SHOWN.size, r_max)).astype(float)
        data_by_dur[d] = _D(counts)

    params = np.array([np.log(4.0), np.log(45.0), -2.0])  # B~4, R~45, eps~0.12
    ll = CPX.cp_response_loglik(params, cache, data_by_dur, r_max, guess="uniform")
    assert np.isfinite(ll) and ll < 0
