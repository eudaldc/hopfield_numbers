"""Smoke tests for WP-H visualization suite.

Each test instantiates minimal data or model objects and checks that
the plot function returns a valid matplotlib Figure without raising.
Figures are closed after each test to avoid memory leaks.
"""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.common.config import (
    AttractorConfig,
    SubjectConfig,
    SupportConfig,
)
from src.models.attractor_core import AttractorCore
from src.models.subject_params import SubjectParams


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def support():
    return SupportConfig(n_min=1, n_max=9)


@pytest.fixture
def core(support):
    sp = SubjectParams(SubjectConfig(n_subjects=20))
    cfg = AttractorConfig(support=support)
    return AttractorCore(cfg, sp)


@pytest.fixture
def subjects_df():
    rng = np.random.default_rng(31)
    n = 60
    return pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "cohort": rng.choice(["A", "B"], n),
        "grade": rng.choice([1, 2], n),
        "age_months": rng.normal(84, 6, n),
        "sex": rng.choice(["M", "F"], n),
        "raven_score": rng.normal(50, 10, n),
        "wm_score": rng.normal(5, 2, n),
        "arithmetic_score": rng.normal(30, 8, n),
        "number_line_score": rng.normal(15, 5, n),
    })


@pytest.fixture
def symbolic_df():
    rng = np.random.default_rng(31)
    n = 200
    left = rng.integers(1, 9, n)
    right = rng.integers(1, 9, n)
    correct = (right > left).astype(int)
    return pd.DataFrame({
        "subject_id": rng.choice([f"S{i:03d}" for i in range(20)], n),
        "left_digit": left,
        "right_digit": right,
        "correct": correct,
        "response_side": ["right" if c else "left" for c in correct],
        "rt_ms": rng.exponential(800, n),
        "grade": rng.choice([1, 2], n),
    })


@pytest.fixture
def dot_df():
    rng = np.random.default_rng(31)
    n = 300
    true_n = rng.integers(4, 9, n)
    shown = rng.integers(4, 9, n)
    same = (true_n == shown).astype(int)
    return pd.DataFrame({
        "subject_id": rng.choice([f"S{i:03d}" for i in range(20)], n),
        "true_n": true_n,
        "shown_digit": shown,
        "response_same": same,
        "correct": same,
        "rt_ms": rng.exponential(1200, n),
        "grade": rng.choice([1, 2], n),
    })


@pytest.fixture
def mts_df():
    rng = np.random.default_rng(31)
    n = 200
    sample_n = rng.integers(2, 9, n)
    target_n = rng.integers(2, 9, n)
    return pd.DataFrame({
        "subject_id": rng.choice([f"S{i:03d}" for i in range(20)], n),
        "sample_true_n": sample_n,
        "target_true_n": target_n,
        "response_same": (sample_n == target_n).astype(int),
        "correct": (sample_n == target_n).astype(int),
        "rt_ms": rng.exponential(900, n),
        "grade": rng.choice([1, 2], n),
    })


@pytest.fixture
def panamath_df():
    rng = np.random.default_rng(31)
    n = 200
    left_n = rng.integers(5, 20, n)
    right_n = rng.integers(5, 20, n)
    correct = (right_n > left_n).astype(int)
    return pd.DataFrame({
        "subject_id": rng.choice([f"S{i:03d}" for i in range(20)], n),
        "left_true_n": left_n,
        "right_true_n": right_n,
        "correct": correct,
        "response_side": ["right" if c else "left" for c in correct],
        "congruency": rng.choice(["congruent", "incongruent"], n),
        "rt_ms": rng.exponential(1000, n),
        "grade": rng.choice([1, 2], n),
    })


@pytest.fixture
def feature_df():
    rng = np.random.default_rng(31)
    n = 100
    return pd.DataFrame({
        "true_n": rng.integers(1, 15, n),
        "total_area": rng.uniform(50, 500, n),
        "mean_dot_size": rng.uniform(5, 30, n),
        "convex_hull": rng.uniform(100, 1000, n),
        "density": rng.uniform(0.01, 0.5, n),
        "spacing": rng.uniform(5, 50, n),
        "fourier_power": rng.uniform(10, 200, n),
    })


@pytest.fixture
def subject_param_df():
    rng = np.random.default_rng(31)
    n = 60
    return pd.DataFrame({
        "subject_idx": list(range(n)),
        "temperature": rng.lognormal(0, 0.3, n),
        "width_scale": rng.lognormal(0, 0.2, n),
        "tau_sym": rng.lognormal(0, 0.3, n),
        "count_bias": rng.normal(0, 0.5, n),
        "kstar": rng.choice([3, 4, 5, 6, 7], n),
        "grade": rng.choice([1, 2], n),
        "arithmetic_score": rng.normal(30, 8, n),
        "number_line_score": rng.normal(15, 5, n),
        "raven_score": rng.normal(50, 10, n),
        "wm_score": rng.normal(5, 2, n),
    })


# ── helper ────────────────────────────────────────────────────────────

def _assert_figure(fig):
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── H1: data QC plots ────────────────────────────────────────────────

class TestPlotsData:
    def test_subject_counts(self, subjects_df, symbolic_df, dot_df):
        from src.viz.plots_data import plot_subject_counts
        task_dfs = {"symbolic_comparison": symbolic_df, "dot_verification": dot_df}
        _assert_figure(plot_subject_counts(subjects_df, task_dfs))

    def test_missingness_heatmap(self, subjects_df):
        from src.viz.plots_data import plot_missingness_heatmap
        _assert_figure(plot_missingness_heatmap(subjects_df))

    def test_trials_per_child(self, symbolic_df, dot_df):
        from src.viz.plots_data import plot_trials_per_child
        _assert_figure(plot_trials_per_child(
            {"symbolic_comparison": symbolic_df, "dot_verification": dot_df}
        ))

    def test_rt_distributions(self, symbolic_df, dot_df):
        from src.viz.plots_data import plot_rt_distributions
        _assert_figure(plot_rt_distributions(
            {"symbolic": symbolic_df, "dotver": dot_df}
        ))

    def test_accuracy_by_task_grade(self, symbolic_df, dot_df):
        from src.viz.plots_data import plot_accuracy_by_task_grade
        _assert_figure(plot_accuracy_by_task_grade(
            {"symbolic": symbolic_df, "dotver": dot_df}
        ))

    def test_filtering_summary(self):
        from src.viz.plots_data import plot_filtering_summary
        _assert_figure(plot_filtering_summary(
            {"sym": 1000, "dot": 2000},
            {"sym": 900, "dot": 1800},
        ))

    def test_saves_to_disk(self, subjects_df, symbolic_df):
        from src.viz.plots_data import plot_missingness_heatmap
        with tempfile.TemporaryDirectory() as tmp:
            fig = plot_missingness_heatmap(subjects_df, save_dir=tmp)
            assert (Path(tmp) / "data_missingness.png").exists()
            plt.close(fig)


# ── H2/H3: model geometry and stimulus diagnostics ───────────────────

class TestPlotsModel:
    def test_centers(self, core):
        from src.viz.plots_model import plot_centers
        _assert_figure(plot_centers(core))

    def test_widths(self, core):
        from src.viz.plots_model import plot_widths
        _assert_figure(plot_widths(core))

    def test_separation(self, core):
        from src.viz.plots_model import plot_separation
        _assert_figure(plot_separation(core))

    def test_precision(self, core):
        from src.viz.plots_model import plot_precision
        _assert_figure(plot_precision(core))

    def test_wells_gaussian(self, core):
        from src.viz.plots_model import plot_wells_gaussian
        _assert_figure(plot_wells_gaussian(core))

    def test_posterior_heatmap(self, core):
        from src.viz.plots_model import plot_posterior_heatmap
        _assert_figure(plot_posterior_heatmap(core))

    def test_energy_landscape(self, core):
        from src.viz.plots_model import plot_energy_landscape
        _assert_figure(plot_energy_landscape(core))

    def test_kstar_distribution(self, core):
        from src.viz.plots_model import plot_kstar_distribution
        _assert_figure(plot_kstar_distribution(core, n_subjects=20))

    def test_feature_correlations(self, feature_df):
        from src.viz.plots_model import plot_feature_correlations
        _assert_figure(plot_feature_correlations(feature_df))

    def test_feature_distributions(self, feature_df):
        from src.viz.plots_model import plot_feature_distributions
        _assert_figure(plot_feature_distributions(feature_df))


# ── H4: task-level fit ───────────────────────────────────────────────

class TestPlotsTasks:
    def test_symbolic_accuracy_by_distance(self, symbolic_df):
        from src.viz.plots_tasks import plot_symbolic_accuracy_by_distance
        _assert_figure(plot_symbolic_accuracy_by_distance(symbolic_df))

    def test_symbolic_accuracy_with_preds(self, symbolic_df):
        from src.viz.plots_tasks import plot_symbolic_accuracy_by_distance
        preds = np.random.default_rng(11).random(len(symbolic_df))
        _assert_figure(plot_symbolic_accuracy_by_distance(symbolic_df, preds))

    def test_symbolic_rt_by_distance(self, symbolic_df):
        from src.viz.plots_tasks import plot_symbolic_rt_by_distance
        _assert_figure(plot_symbolic_rt_by_distance(symbolic_df))

    def test_mts_psychometric(self, mts_df):
        from src.viz.plots_tasks import plot_mts_psychometric
        _assert_figure(plot_mts_psychometric(mts_df))

    def test_dot_verification_heatmap(self, dot_df):
        from src.viz.plots_tasks import plot_dot_verification_heatmap
        _assert_figure(plot_dot_verification_heatmap(dot_df))

    def test_dot_verification_heatmap_with_preds(self, dot_df):
        from src.viz.plots_tasks import plot_dot_verification_heatmap
        preds = np.random.default_rng(11).random(len(dot_df))
        _assert_figure(plot_dot_verification_heatmap(dot_df, preds))

    def test_dot_verification_rt(self, dot_df):
        from src.viz.plots_tasks import plot_dot_verification_rt
        _assert_figure(plot_dot_verification_rt(dot_df))

    def test_counting_route_probability(self, dot_df):
        from src.viz.plots_tasks import plot_counting_route_probability
        pi = np.random.default_rng(11).random(len(dot_df))
        _assert_figure(plot_counting_route_probability(dot_df, pi))

    def test_panamath_psychometric_ratio(self, panamath_df):
        from src.viz.plots_tasks import plot_panamath_psychometric_ratio
        _assert_figure(plot_panamath_psychometric_ratio(panamath_df))

    def test_panamath_congruency(self, panamath_df):
        from src.viz.plots_tasks import plot_panamath_congruency
        _assert_figure(plot_panamath_congruency(panamath_df))


# ── H5/H6: PPC and model comparison ─────────────────────────────────

class TestPlotsPPC:
    def test_obs_vs_pred_accuracy(self, dot_df):
        from src.viz.plots_ppc import plot_obs_vs_pred_accuracy
        preds = np.random.default_rng(11).random(len(dot_df))
        _assert_figure(plot_obs_vs_pred_accuracy(
            dot_df, preds, "true_n", "dot_verification"
        ))

    def test_obs_vs_pred_rt(self, dot_df):
        from src.viz.plots_ppc import plot_obs_vs_pred_rt
        _assert_figure(plot_obs_vs_pred_rt(dot_df, None, "dot_verification"))

    def test_obs_vs_pred_same_heatmap(self, dot_df):
        from src.viz.plots_ppc import plot_obs_vs_pred_same_heatmap
        preds = np.random.default_rng(11).random(len(dot_df))
        _assert_figure(plot_obs_vs_pred_same_heatmap(
            dot_df, preds, "true_n", "shown_digit", "dot_verification"
        ))

    def test_confusion_matrix(self):
        from src.viz.plots_ppc import plot_confusion_matrix
        labels = np.array([0, 0, 1, 1, 1, 0])
        preds = np.array([0, 1, 1, 1, 0, 0])
        _assert_figure(plot_confusion_matrix(labels, preds, "test"))

    def test_calibration_curve(self):
        from src.viz.plots_ppc import plot_calibration_curve
        rng = np.random.default_rng(11)
        preds = rng.random(500)
        labels = (rng.random(500) > 0.5).astype(float)
        _assert_figure(plot_calibration_curve(preds, labels, "test"))

    def test_model_comparison_nll(self):
        from src.viz.plots_ppc import plot_model_comparison_nll
        _assert_figure(plot_model_comparison_nll(
            {"default/dynamic": 0.45, "default/static": 0.52, "weber/dynamic": 0.48}
        ))

    def test_model_comparison_accuracy(self):
        from src.viz.plots_ppc import plot_model_comparison_accuracy
        _assert_figure(plot_model_comparison_accuracy({
            "default/dynamic": {"symbolic": 0.85, "mts": 0.78},
            "weber/dynamic": {"symbolic": 0.82, "mts": 0.75},
        }))


# ── H7: subject-level analysis ───────────────────────────────────────

class TestPlotsSubjects:
    def test_param_histograms(self, subject_param_df):
        from src.viz.plots_subjects import plot_subject_param_histograms
        _assert_figure(plot_subject_param_histograms(subject_param_df))

    def test_param_scattermatrix(self, subject_param_df):
        from src.viz.plots_subjects import plot_subject_param_scattermatrix
        _assert_figure(plot_subject_param_scattermatrix(subject_param_df))

    def test_param_vs_downstream(self, subject_param_df):
        from src.viz.plots_subjects import plot_param_vs_downstream
        _assert_figure(plot_param_vs_downstream(
            subject_param_df, "temperature", "arithmetic_score"
        ))

    def test_all_param_vs_downstream(self, subject_param_df):
        from src.viz.plots_subjects import plot_all_param_vs_downstream
        _assert_figure(plot_all_param_vs_downstream(subject_param_df))

    def test_kstar_vs_grade(self, subject_param_df):
        from src.viz.plots_subjects import plot_kstar_vs_grade
        _assert_figure(plot_kstar_vs_grade(subject_param_df))


# ── style helpers ─────────────────────────────────────────────────────

class TestVizInit:
    def test_apply_style(self):
        from src.viz import apply_style
        apply_style()

    def test_save_figure_to_disk(self):
        from src.viz import save_figure
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            save_figure(fig, "test_save", save_dir=tmp, close=False)
            assert (Path(tmp) / "test_save.png").exists()
        plt.close(fig)

    def test_palette_has_colors(self):
        from src.viz import PALETTE
        assert len(PALETTE) >= 8

    def test_compose_panel(self):
        from src.viz import compose_panel
        figs = []
        for i in range(4):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [i, i + 1])
            ax.set_title(f"Plot {i}")
            figs.append(fig)
        panel = compose_panel(figs, suptitle="Test panel", max_cols=2)
        _assert_figure(panel)

    def test_compose_panel_saves_to_disk(self):
        from src.viz import compose_panel, save_figure
        fig1, ax1 = plt.subplots()
        ax1.plot([0, 1], [0, 1])
        fig2, ax2 = plt.subplots()
        ax2.plot([0, 1], [1, 0])
        panel = compose_panel([fig1, fig2], suptitle="Saved panel")
        with tempfile.TemporaryDirectory() as tmp:
            save_figure(panel, "test_panel", save_dir=tmp, close=False)
            assert (Path(tmp) / "test_panel.png").exists()
        plt.close(panel)

    def test_compose_panel_with_suffix(self):
        from src.viz import compose_panel, save_figure
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        panel = compose_panel([fig], suptitle="Model — preset: weber")
        with tempfile.TemporaryDirectory() as tmp:
            save_figure(panel, "panel_model", save_dir=tmp,
                        close=False, file_suffix="weber")
            assert (Path(tmp) / "panel_model_weber.png").exists()
        plt.close(panel)

    def test_compose_panel_empty(self):
        from src.viz import compose_panel
        panel = compose_panel([])
        _assert_figure(panel)
