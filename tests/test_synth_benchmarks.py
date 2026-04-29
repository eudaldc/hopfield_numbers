"""Tests for the attractor-first synthetic benchmark pipeline."""

import pytest
import torch

from src.data.synth_benchmarks import (
    NeutralEvidenceChannel,
    generate_task_a,
    generate_task_b,
    generate_task_c,
    build_benchmark_datasets,
)


N_MIN, N_MAX = 1, 9
B = 200


def _n(b: int = B) -> torch.Tensor:
    return torch.randint(N_MIN, N_MAX + 1, (b,))


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── NeutralEvidenceChannel ─────────────────────────────────────────────────


class TestNeutralEvidenceChannel:
    def test_sample_shape(self):
        ch = NeutralEvidenceChannel()
        y = ch.sample(_n())
        assert y.shape == (B,)

    def test_sample_dtype(self):
        ch = NeutralEvidenceChannel()
        y = ch.sample(_n())
        assert y.dtype == torch.float32

    def test_reproducible_with_same_seed(self):
        ch = NeutralEvidenceChannel()
        n = _n()
        y1 = ch.sample(n, generator=_gen(11))
        y2 = ch.sample(n, generator=_gen(11))
        assert torch.allclose(y1, y2)

    def test_different_seeds_give_different_y(self):
        ch = NeutralEvidenceChannel()
        n = _n()
        y1 = ch.sample(n, generator=_gen(1))
        y2 = ch.sample(n, generator=_gen(2))
        assert not torch.allclose(y1, y2)

    def test_mean_close_to_boxcox(self):
        ch = NeutralEvidenceChannel(s00=0.001, s0=0.001)
        n = torch.arange(N_MIN, N_MAX + 1)
        y = torch.stack([ch.sample(n) for _ in range(2000)]).mean(0)
        expected = ch.mean(n)
        assert torch.allclose(y, expected, atol=0.05)

    def test_mean_monotone(self):
        ch = NeutralEvidenceChannel()
        mu = ch.mean(torch.arange(N_MIN, N_MAX + 1))
        assert (mu[1:] > mu[:-1]).all()

    def test_std_positive(self):
        ch = NeutralEvidenceChannel()
        s = ch.std(torch.arange(N_MIN, N_MAX + 1))
        assert (s > 0).all()


# ── Task generators ────────────────────────────────────────────────────────


class TestGenerateTaskA:
    def test_shapes(self):
        d = generate_task_a(n_samples=B, seed=5)
        assert d["n_true"].shape == (B,)
        assert d["y_noisy"].shape == (B,)

    def test_no_frontend_family_key(self):
        d = generate_task_a(n_samples=B, seed=5)
        assert "frontend_family" not in d

    def test_n_in_range(self):
        d = generate_task_a(n_min=N_MIN, n_max=N_MAX, n_samples=B, seed=5)
        assert (d["n_true"] >= N_MIN).all()
        assert (d["n_true"] <= N_MAX).all()

    def test_reproducible(self):
        d1 = generate_task_a(n_samples=B, seed=5)
        d2 = generate_task_a(n_samples=B, seed=5)
        assert torch.equal(d1["n_true"], d2["n_true"])
        assert torch.allclose(d1["y_noisy"], d2["y_noisy"])

    def test_custom_channel(self):
        ch = NeutralEvidenceChannel(rho0=1.0, nu0=0.5)
        d = generate_task_a(n_samples=B, seed=5, channel=ch)
        assert d["y_noisy"].shape == (B,)


class TestGenerateTaskB:
    def test_shapes(self):
        d = generate_task_b(n_samples=B, seed=5)
        for key in ("n_left", "n_right", "y_left", "y_right", "label"):
            assert d[key].shape == (B,), key

    def test_label_binary(self):
        d = generate_task_b(n_samples=B * 5, seed=5)
        assert set(d["label"].unique().tolist()).issubset({0, 1})

    def test_label_consistent(self):
        d = generate_task_b(n_samples=B * 5, seed=5)
        expected = (d["n_right"] > d["n_left"]).long()
        assert torch.equal(d["label"], expected)

    def test_reproducible(self):
        d1 = generate_task_b(n_samples=B, seed=5)
        d2 = generate_task_b(n_samples=B, seed=5)
        assert torch.equal(d1["n_left"], d2["n_left"])


class TestGenerateTaskC:
    def test_shapes(self):
        d = generate_task_c(n_samples=B, seed=5)
        for key in ("n_true", "target", "y_noisy", "label"):
            assert d[key].shape == (B,), key

    def test_no_frontend_family_key(self):
        d = generate_task_c(n_samples=B, seed=5)
        assert "frontend_family" not in d

    def test_label_consistent(self):
        d = generate_task_c(n_samples=B * 5, seed=5)
        expected = (d["n_true"] == d["target"]).long()
        assert torch.equal(d["label"], expected)

    def test_both_classes_present(self):
        d = generate_task_c(n_samples=1000, seed=5)
        assert d["label"].sum() > 0
        assert (d["label"] == 0).sum() > 0


class TestBuildBenchmarkDatasets:
    def test_splits_present(self):
        ds = build_benchmark_datasets(
            n_samples_train=50, n_samples_val=20, n_samples_test=20
        )
        assert set(ds.keys()) == {"train", "val", "test"}

    def test_tasks_present(self):
        ds = build_benchmark_datasets(
            n_samples_train=50, n_samples_val=20, n_samples_test=20
        )
        assert set(ds["train"].keys()) == {"task_a", "task_b", "task_c"}

    def test_train_val_test_sizes(self):
        ds = build_benchmark_datasets(
            n_samples_train=100, n_samples_val=40, n_samples_test=30
        )
        assert ds["train"]["task_a"]["n_true"].shape[0] == 100
        assert ds["val"]["task_a"]["n_true"].shape[0] == 40
        assert ds["test"]["task_a"]["n_true"].shape[0] == 30

    def test_train_val_different(self):
        ds = build_benchmark_datasets(
            n_samples_train=100, n_samples_val=100, n_samples_test=100
        )
        assert not torch.equal(
            ds["train"]["task_a"]["n_true"],
            ds["val"]["task_a"]["n_true"],
        )
