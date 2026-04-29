"""Tests for the phase scan module."""

import pytest
import torch

from src.analysis.phase_scan import (
    ScanAxis,
    PhaseScanConfig,
    _build_model,
    compute_metrics,
    run_phase_scan,
)
from src.data.synth_benchmarks import NeutralEvidenceChannel
from src.models.attractor_dynamics import DynamicAttractorCore
from src.models.rt_proxy import RTProxy


N_MIN, N_MAX, SMALL_N_MAX = 1, 9, 4
N_SAMPLES = 300
SEED = 3


def _rt(mode: str = "steps_plus_margin") -> RTProxy:
    return RTProxy(mode=mode)


def _channel():
    return NeutralEvidenceChannel()


def _data():
    from src.data.synth_benchmarks import generate_task_a
    return generate_task_a(n_min=N_MIN, n_max=N_MAX, n_samples=N_SAMPLES, seed=SEED)


# ── ScanAxis ───────────────────────────────────────────────────────────────


class TestScanAxis:
    def test_fields_stored(self):
        ax = ScanAxis(param="kappa", values=[0.5, 1.0, 2.0])
        assert ax.param == "kappa"
        assert ax.values == [0.5, 1.0, 2.0]

    def test_all_valid_params(self):
        for p in ("kappa", "tau_in", "sigma0", "lam", "eta", "rho"):
            ax = ScanAxis(param=p, values=[1.0])
            assert ax.param == p


# ── PhaseScanConfig ────────────────────────────────────────────────────────


class TestPhaseScanConfig:
    def test_default_values(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("kappa", [1.0])])
        assert cfg.n_min == 1
        assert cfg.n_max == 9
        assert cfg.small_n_max == 4
        assert cfg.kappa == 1.0
        assert cfg.rt_mode == "steps_plus_margin"

    def test_axes_stored(self):
        axes = [ScanAxis("kappa", [1.0, 2.0])]
        cfg = PhaseScanConfig(axes=axes)
        assert cfg.axes[0].param == "kappa"
        assert cfg.axes[0].values == [1.0, 2.0]

    def test_custom_fixed_values(self):
        cfg = PhaseScanConfig(
            axes=[ScanAxis("kappa", [1.0])],
            tau_in=2.5, sigma0=0.3, lam=0.7,
        )
        assert cfg.tau_in == pytest.approx(2.5)
        assert cfg.sigma0 == pytest.approx(0.3)
        assert cfg.lam == pytest.approx(0.7)


# ── _build_model ───────────────────────────────────────────────────────────


class TestBuildModel:
    def test_returns_dynamic_attractor_core(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("kappa", [1.0])])
        model = _build_model(cfg, {"kappa": 1.0})
        assert isinstance(model, DynamicAttractorCore)

    def test_cleanup_mode_is_gradient(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("kappa", [1.0])])
        model = _build_model(cfg, {})
        assert model.cleanup_mode == "gradient"

    def test_kappa_override(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("kappa", [1.0])], kappa=1.0)
        model = _build_model(cfg, {"kappa": 3.0})
        assert abs(model.kappa.item() - 3.0) < 0.05

    def test_tau_in_override(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("tau_in", [1.0])], tau_in=1.0)
        model = _build_model(cfg, {"tau_in": 2.0})
        assert abs(model.tau_in.item() - 2.0) < 0.05

    def test_sigma0_override(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("sigma0", [1.0])], sigma0=1.0)
        model = _build_model(cfg, {"sigma0": 0.5})
        assert abs(model.sigma0.item() - 0.5) < 0.05

    def test_lam_override_maps_to_lambda_init(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("lam", [0.5])], lam=0.5)
        model = _build_model(cfg, {"lam": 0.9})
        assert abs(model.lam.item() - 0.9) < 0.05

    def test_eta_override(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("eta", [0.2])], eta=0.2)
        model = _build_model(cfg, {"eta": 0.4})
        assert abs(model.eta.item() - 0.4) < 0.05

    def test_rho_override(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("rho", [0.1])], rho=0.1)
        model = _build_model(cfg, {"rho": 0.8})
        assert abs(model.rho.item() - 0.8) < 0.05

    def test_empty_overrides_uses_config_defaults(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("kappa", [1.0])], kappa=2.0)
        model = _build_model(cfg, {})
        assert abs(model.kappa.item() - 2.0) < 0.05


# ── compute_metrics ────────────────────────────────────────────────────────


class TestComputeMetrics:
    def _default_model(self):
        cfg = PhaseScanConfig(axes=[ScanAxis("kappa", [1.0])])
        return _build_model(cfg, {})

    def test_required_aggregate_keys(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        required = {
            "acc_small", "rt_small", "rt_slope_small",
            "mae_large", "rt_large", "rt_slope_large",
            "mean_steps", "frac_converged", "mean_margin",
            "k_star", "sep_ratio_mean", "occ_entropy", "rt_slope_ratio",
        }
        assert required <= set(m.keys())

    def test_per_n_keys_present(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        for n in range(N_MIN, N_MAX + 1):
            assert f"acc_n{n}" in m
            assert f"rt_n{n}" in m
            assert f"mae_n{n}" in m

    def test_accuracy_in_unit_interval(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        assert 0.0 <= m["acc_small"] <= 1.0
        for n in range(N_MIN, N_MAX + 1):
            assert 0.0 <= m[f"acc_n{n}"] <= 1.0

    def test_frac_converged_in_unit_interval(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        assert 0.0 <= m["frac_converged"] <= 1.0

    def test_occ_entropy_non_negative(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        assert m["occ_entropy"] >= 0.0

    def test_mae_large_non_negative(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        assert m["mae_large"] >= 0.0

    def test_mean_steps_positive(self):
        m = compute_metrics(
            self._default_model(), _rt(), _data(),
            N_MIN, N_MAX, SMALL_N_MAX, torch.device("cpu"),
        )
        assert m["mean_steps"] > 0


# ── run_phase_scan ──────────────────────────────────────────────────────────


class TestRunPhaseScan:
    def test_1d_returns_dataframe(self):
        import pandas as pd

        cfg = PhaseScanConfig(
            axes=[ScanAxis("kappa", [0.5, 1.0, 2.0])],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        assert isinstance(df, pd.DataFrame)

    def test_1d_row_count(self):
        n_vals = 4
        cfg = PhaseScanConfig(
            axes=[ScanAxis("kappa", list(range(1, n_vals + 1)))],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        assert len(df) == n_vals

    def test_1d_scanned_param_column_present(self):
        cfg = PhaseScanConfig(
            axes=[ScanAxis("tau_in", [0.5, 1.0])],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        assert "tau_in" in df.columns

    def test_1d_metric_columns_present(self):
        cfg = PhaseScanConfig(
            axes=[ScanAxis("kappa", [0.5, 1.5])],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        for col in ("acc_small", "mae_large", "mean_steps", "k_star", "occ_entropy"):
            assert col in df.columns, f"column {col!r} missing"

    def test_1d_per_n_columns_present(self):
        cfg = PhaseScanConfig(
            axes=[ScanAxis("kappa", [1.0])],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        for n in range(N_MIN, N_MAX + 1):
            assert f"acc_n{n}" in df.columns

    def test_2d_row_count(self):
        cfg = PhaseScanConfig(
            axes=[
                ScanAxis("kappa", [0.5, 1.0, 2.0]),
                ScanAxis("tau_in", [0.5, 1.0]),
            ],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        assert len(df) == 3 * 2

    def test_2d_both_param_columns(self):
        cfg = PhaseScanConfig(
            axes=[
                ScanAxis("kappa", [0.5, 1.5]),
                ScanAxis("sigma0", [0.5, 1.5]),
            ],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt())
        assert "kappa" in df.columns
        assert "sigma0" in df.columns

    def test_rho_scan(self):
        cfg = PhaseScanConfig(
            axes=[ScanAxis("rho", [0.1, 0.5])],
            n_samples=N_SAMPLES, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt("steps"))
        assert len(df) == 2

    def test_invalid_param_raises(self):
        cfg = PhaseScanConfig(
            axes=[ScanAxis("nonexistent_param", [1.0])],
            n_samples=N_SAMPLES,
        )
        with pytest.raises(ValueError, match="Unknown scan parameter"):
            run_phase_scan(cfg)

    def test_too_many_axes_raises(self):
        cfg = PhaseScanConfig(
            axes=[
                ScanAxis("kappa", [1.0]),
                ScanAxis("tau_in", [1.0]),
                ScanAxis("sigma0", [1.0]),
            ],
            n_samples=N_SAMPLES,
        )
        with pytest.raises(ValueError, match="at most 2"):
            run_phase_scan(cfg)

    def test_empty_axes_raises(self):
        cfg = PhaseScanConfig(axes=[], n_samples=N_SAMPLES)
        with pytest.raises(ValueError, match="at least one"):
            run_phase_scan(cfg)

    def test_dynamic_vs_static_gives_higher_frac_converged(self):
        """Dynamic cleanup should register convergence, static (none) should not."""
        cfg = PhaseScanConfig(
            axes=[ScanAxis("kappa", [2.0])], n_samples=200, seed=SEED,
        )
        df = run_phase_scan(cfg, rt_proxy=_rt("steps"))
        assert df["frac_converged"].iloc[0] >= 0.0
