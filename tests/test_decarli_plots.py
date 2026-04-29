"""Unit tests for DeCarli plot helpers (hypothesis annotations)."""

import numpy as np
from scipy.stats import false_discovery_control, norm

from scripts.plots import (
    _decarli_mannwhitney_pvalue,
    _decarli_significance_stars,
    _decarli_wald_z_pvalue,
    benjamini_hochberg_adjusted_p,
    _format_decarli_hypothesis_line,
)


def test_wald_z_pvalue_known_z():
    # (m_dd - m_ct) / sqrt(se_dd^2 + se_ct^2) = 2 with pooled variance 1
    s = 1.0 / np.sqrt(2.0)
    p = _decarli_wald_z_pvalue(2.0, 0.0, s, s)
    expected = 2 * (1 - norm.cdf(2.0))
    assert abs(p - expected) < 1e-9
    assert 0.044 < p < 0.046


def test_wald_z_pvalue_identical_means():
    p = _decarli_wald_z_pvalue(1.5, 1.5, 0.2, 0.3)
    assert abs(p - 1.0) < 1e-9


def test_mannwhitney_separated_samples():
    rng = np.random.default_rng(137)
    a = rng.normal(0, 1, size=31)
    b = rng.normal(3, 1, size=29)
    p = _decarli_mannwhitney_pvalue(a, b)
    assert p < 0.001


def test_format_p_stars_and_ns():
    assert "(n.s.)" in _format_decarli_hypothesis_line(0.2)
    assert "***" in _format_decarli_hypothesis_line(0.0001)
    assert "**" in _format_decarli_hypothesis_line(0.005)
    assert "*" in _format_decarli_hypothesis_line(0.04) and "(n.s.)" not in _format_decarli_hypothesis_line(
        0.04
    )


def test_significance_stars_table():
    assert _decarli_significance_stars(0.2) == "n.s."
    assert _decarli_significance_stars(0.0001) == "***"
    assert _decarli_significance_stars(0.005) == "**"
    assert _decarli_significance_stars(0.04) == "*"
    assert _decarli_significance_stars(float("nan")) == "—"


def test_bh_matches_scipy():
    p = np.array([0.01, 0.04, 0.05])
    np.testing.assert_allclose(
        benjamini_hochberg_adjusted_p(p), false_discovery_control(p, method="bh")
    )
    q = benjamini_hochberg_adjusted_p(np.array([0.1, float("nan"), 0.02]))
    assert np.isfinite(q[0]) and np.isnan(q[1]) and np.isfinite(q[2])
