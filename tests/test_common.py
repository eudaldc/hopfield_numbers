"""Smoke tests for WP-A shared utilities."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.common import (
    AttractorConfig,
    ExperimentConfig,
    SupportConfig,
    TrainingConfig,
    best_checkpoint_filename,
    get_device,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    ensure_dir,
    accuracy,
    binary_cross_entropy,
    multiclass_accuracy,
    multiclass_nll,
    rmse,
    mae,
    brier_score,
    calibration_bins,
    METRIC_NAMES,
)


# -------------------------------------------------------------------
# Config tests
# -------------------------------------------------------------------

def test_support_config_defaults():
    s = SupportConfig()
    assert s.n_min == 0
    assert s.n_max == 20
    assert s.size == 21
    assert s.integers == list(range(21))


def test_support_config_custom():
    s = SupportConfig(n_min=1, n_max=9)
    assert s.size == 9
    assert s.integers == list(range(1, 10))


def test_experiment_config_composes():
    cfg = ExperimentConfig()
    assert cfg.support.size == 21
    assert cfg.attractor.rho_init == 0.5
    assert cfg.training.lr == 1e-3
    assert cfg.preset == "default"
    assert cfg.family == "attractor"


def test_attractor_config_fields():
    cfg = AttractorConfig(rho_init=0.8, lam_init=0.3)
    assert cfg.rho_init == 0.8
    assert cfg.lam_init == 0.3


def test_training_config_task_weights():
    cfg = TrainingConfig()
    assert "symbolic_comparison" in cfg.task_weights
    assert cfg.task_weights["symbolic_comparison"] == 1.0


# -------------------------------------------------------------------
# Seed tests
# -------------------------------------------------------------------

def test_set_seed_reproducibility():
    set_seed(99)
    a = torch.randn(5)
    set_seed(99)
    b = torch.randn(5)
    assert torch.allclose(a, b)


# -------------------------------------------------------------------
# Device tests
# -------------------------------------------------------------------

def test_get_device_returns_device():
    dev = get_device(prefer_cuda=False)
    assert isinstance(dev, torch.device)


# -------------------------------------------------------------------
# IO tests
# -------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


def test_save_load_checkpoint():
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, opt, epoch=5, metrics={"loss": 0.42})

        model2 = _TinyModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        ckpt = load_checkpoint(path, model2, opt2)

        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["loss"] == 0.42

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_ensure_dir(tmp_path):
    d = tmp_path / "a" / "b" / "c"
    result = ensure_dir(d)
    assert result.exists()
    assert result.is_dir()


def test_best_checkpoint_filename():
    assert best_checkpoint_filename("default") == "best_default_dynamic.pt"
    assert best_checkpoint_filename("default", "static") == "best_default_static.pt"
    assert best_checkpoint_filename("weber") == "best_weber_dynamic.pt"
    assert best_checkpoint_filename("weber", "static") == "best_weber_static.pt"


# -------------------------------------------------------------------
# Metric tests
# -------------------------------------------------------------------

def test_accuracy_perfect():
    probs = torch.tensor([0.9, 0.8, 0.1, 0.2])
    labels = torch.tensor([1, 1, 0, 0])
    assert accuracy(probs, labels).item() == 1.0


def test_accuracy_half():
    probs = torch.tensor([0.9, 0.1, 0.9, 0.1])
    labels = torch.tensor([1, 1, 0, 0])
    assert accuracy(probs, labels).item() == 0.5


def test_binary_cross_entropy_range():
    probs = torch.tensor([0.7, 0.3, 0.9])
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = binary_cross_entropy(probs, labels)
    assert loss.item() > 0.0


def test_multiclass_accuracy_perfect():
    logits = torch.tensor([[10.0, 1.0, 0.0], [0.0, 10.0, 1.0]])
    targets = torch.tensor([0, 1])
    assert multiclass_accuracy(logits, targets).item() == 1.0


def test_multiclass_nll_positive():
    log_probs = torch.log_softmax(torch.randn(8, 5), dim=-1)
    targets = torch.randint(0, 5, (8,))
    loss = multiclass_nll(log_probs, targets)
    assert loss.item() > 0.0


def test_rmse():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    assert rmse(pred, target).item() == 0.0

    pred2 = torch.tensor([2.0, 3.0, 4.0])
    assert rmse(pred2, target).item() == 1.0


def test_mae():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 3.0, 4.0])
    assert mae(pred, target).item() == 1.0


def test_brier_score_perfect():
    probs = torch.tensor([1.0, 0.0, 1.0])
    labels = torch.tensor([1, 0, 1])
    assert brier_score(probs, labels).item() == 0.0


def test_calibration_bins_structure():
    probs = torch.rand(100)
    labels = (torch.rand(100) > 0.5).long()
    result = calibration_bins(probs, labels, n_bins=5)
    assert "ece" in result
    assert result["bin_accs"].shape == (5,)
    assert result["bin_counts"].sum().item() == 100
    assert result["ece"].item() >= 0.0


def test_metric_names_exist():
    assert "train_loss" in METRIC_NAMES
    assert "val_symbolic_acc" in METRIC_NAMES
    assert METRIC_NAMES["rho"] == "analysis/rho"
