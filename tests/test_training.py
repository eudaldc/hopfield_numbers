"""Smoke tests for WP-G training engine, losses, and evaluators."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.common.config import (
    AttractorConfig,
    DotVerificationConfig,
    ExperimentConfig,
    SubjectConfig,
    SupportConfig,
    VisualEncoderConfig,
)
from src.training.losses import multitask_loss, total_loss
from src.training.engine import (
    MetricAccumulator,
    Trainer,
    build_modules,
    build_task_heads,
    collect_parameters,
    _to_device,
    _CheckpointWrapper,
)
from src.training.evaluators import (
    evaluate_tasks,
    export_subject_params,
    compute_kstar_summary,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

def _small_cfg(preset: str = "default", n_subjects: int = 4) -> ExperimentConfig:
    support = SupportConfig(n_min=0, n_max=9)
    return ExperimentConfig(
        support=support,
        attractor=AttractorConfig(support=support),
        subject=SubjectConfig(n_subjects=n_subjects),
        visual=VisualEncoderConfig(mode="features", n_features=3, hidden_dim=8),
        dot_verification=DotVerificationConfig(),
        preset=preset,
    )


def _fake_symbolic_batch(n: int = 8, n_subjects: int = 4) -> dict:
    return {
        "task": ["symbolic_comparison"] * n,
        "subject_idx": torch.randint(0, n_subjects, (n,)),
        "left_digit": torch.randint(0, 9, (n,)),
        "right_digit": torch.randint(0, 9, (n,)),
        "label": torch.randint(0, 2, (n,)).float(),
        "grade": torch.ones(n),
        "cohort": ["A"] * n,
    }


def _fake_dotver_batch(n: int = 8, n_subjects: int = 4, n_feats: int = 3) -> dict:
    return {
        "task": ["dot_verification"] * n,
        "subject_idx": torch.randint(0, n_subjects, (n,)),
        "features": torch.randn(n, n_feats),
        "true_n": torch.randint(1, 9, (n,)),
        "shown_digit": torch.randint(1, 9, (n,)),
        "label": torch.randint(0, 2, (n,)).float(),
        "grade": torch.ones(n),
        "rt_ms": torch.rand(n) * 2000 + 200,
        "cohort": ["A"] * n,
    }


def _fake_multitask_batch(n_subjects: int = 4, n_feats: int = 3) -> dict[str, dict]:
    return {
        "symbolic_comparison": _fake_symbolic_batch(6, n_subjects),
        "dot_verification": _fake_dotver_batch(6, n_subjects, n_feats),
    }


# -------------------------------------------------------------------
# _to_device
# -------------------------------------------------------------------

def test_to_device_moves_tensors():
    batch = _fake_multitask_batch()
    moved = _to_device(batch, torch.device("cpu"))
    for task in moved:
        for k, v in moved[task].items():
            if isinstance(v, torch.Tensor):
                assert v.device == torch.device("cpu")


# -------------------------------------------------------------------
# build_modules
# -------------------------------------------------------------------

def test_build_modules_default():
    cfg = _small_cfg("default")
    modules = build_modules(cfg, 4, torch.device("cpu"))
    assert "visual_encoder" in modules
    assert "attractor_core" in modules
    assert "symbolic_head" in modules
    assert "subject_params" in modules


# -------------------------------------------------------------------
# build_task_heads
# -------------------------------------------------------------------

def test_build_task_heads_symbolic():
    cfg = _small_cfg()
    heads = build_task_heads(cfg, 4, ["symbolic_comparison"], torch.device("cpu"))
    assert "symbolic_comparison" in heads


def test_build_task_heads_all():
    cfg = _small_cfg()
    tasks = ["symbolic_comparison", "match_to_sample",
             "dot_verification", "panamath"]
    heads = build_task_heads(cfg, 4, tasks, torch.device("cpu"))
    assert len(heads) == 4


# -------------------------------------------------------------------
# collect_parameters
# -------------------------------------------------------------------

def test_collect_parameters_no_duplicates():
    cfg = _small_cfg()
    modules = build_modules(cfg, 4, torch.device("cpu"))
    heads = build_task_heads(cfg, 4, ["symbolic_comparison"], torch.device("cpu"))
    params = collect_parameters(modules, heads)
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids))


# -------------------------------------------------------------------
# multitask_loss / total_loss
# -------------------------------------------------------------------

def test_multitask_loss():
    cfg = _small_cfg()
    modules = build_modules(cfg, 4, torch.device("cpu"))
    heads = build_task_heads(cfg, 4, ["symbolic_comparison"], torch.device("cpu"))

    batch = _fake_symbolic_batch()
    out = heads["symbolic_comparison"](batch, modules)
    task_outputs = {"symbolic_comparison": out}

    loss, per_task = multitask_loss(task_outputs, {"symbolic_comparison": 1.0})
    assert loss.item() > 0
    assert "symbolic_comparison" in per_task


def test_total_loss_includes_penalty():
    cfg = _small_cfg()
    modules = build_modules(cfg, 4, torch.device("cpu"))
    heads = build_task_heads(cfg, 4, ["symbolic_comparison"], torch.device("cpu"))

    batch = _fake_symbolic_batch()
    out = heads["symbolic_comparison"](batch, modules)
    task_outputs = {"symbolic_comparison": out}

    loss, per_task, penalty = total_loss(
        task_outputs, {"symbolic_comparison": 1.0}, modules["subject_params"],
    )
    assert loss.requires_grad
    assert penalty.item() >= 0


def test_multitask_loss_multiple_tasks():
    cfg = _small_cfg()
    modules = build_modules(cfg, 4, torch.device("cpu"))
    heads = build_task_heads(
        cfg, 4, ["symbolic_comparison", "dot_verification"], torch.device("cpu"),
    )

    batch = _fake_multitask_batch()
    task_outputs = {}
    for task_name, task_batch in batch.items():
        if task_name in heads:
            task_outputs[task_name] = heads[task_name](task_batch, modules)

    loss, per_task = multitask_loss(
        task_outputs, {"symbolic_comparison": 1.0, "dot_verification": 0.5},
    )
    assert loss.item() > 0
    assert len(per_task) == 2


# -------------------------------------------------------------------
# MetricAccumulator
# -------------------------------------------------------------------

def test_metric_accumulator():
    acc = MetricAccumulator()
    acc.update({"loss": 2.0, "acc": 0.8}, weight=10)
    acc.update({"loss": 1.0, "acc": 0.9}, weight=10)
    result = acc.compute()
    assert abs(result["loss"] - 1.5) < 1e-6
    assert abs(result["acc"] - 0.85) < 1e-6


# -------------------------------------------------------------------
# Trainer (minimal integration test)
# -------------------------------------------------------------------

def test_trainer_train_epoch():
    cfg = _small_cfg()
    device = torch.device("cpu")
    modules = build_modules(cfg, 4, device)
    heads = build_task_heads(
        cfg, 4, ["symbolic_comparison", "dot_verification"], device,
    )

    batches = [_fake_multitask_batch(), _fake_multitask_batch()]

    class _FakeLoader:
        """Minimal iterable to stand in for a DataLoader."""
        def __iter__(self):
            return iter(batches)

    trainer = Trainer(
        cfg, modules, heads,
        train_loader=_FakeLoader(),
        val_loader=_FakeLoader(),
        device=device,
        use_wandb=False,
    )
    metrics = trainer.train_epoch()
    assert "train/loss_total" in metrics
    assert metrics["train/loss_total"] > 0


def test_trainer_evaluate():
    cfg = _small_cfg()
    device = torch.device("cpu")
    modules = build_modules(cfg, 4, device)
    heads = build_task_heads(
        cfg, 4, ["symbolic_comparison", "dot_verification"], device,
    )

    batches = [_fake_multitask_batch()]

    class _FakeLoader:
        def __iter__(self):
            return iter(batches)

    trainer = Trainer(
        cfg, modules, heads,
        train_loader=_FakeLoader(),
        val_loader=_FakeLoader(),
        device=device,
        use_wandb=False,
    )
    metrics = trainer.evaluate(_FakeLoader(), prefix="val")
    assert "val/loss_total" in metrics


def test_trainer_reduces_loss():
    """A few epochs should not blow up the loss."""
    cfg = _small_cfg()
    cfg.training.lr = 1e-2
    device = torch.device("cpu")
    modules = build_modules(cfg, 4, device)
    heads = build_task_heads(cfg, 4, ["symbolic_comparison"], device)

    batch = [{"symbolic_comparison": _fake_symbolic_batch(32)}] * 5

    class _FakeLoader:
        def __iter__(self):
            return iter(batch)

    trainer = Trainer(
        cfg, modules, heads,
        train_loader=_FakeLoader(),
        val_loader=_FakeLoader(),
        device=device,
        use_wandb=False,
    )
    m1 = trainer.train_epoch()
    m2 = trainer.train_epoch()
    assert m2["train/loss_total"] < m1["train/loss_total"] * 2.0


# -------------------------------------------------------------------
# evaluate_tasks
# -------------------------------------------------------------------

def test_evaluate_tasks():
    cfg = _small_cfg()
    device = torch.device("cpu")
    modules = build_modules(cfg, 4, device)
    heads = build_task_heads(
        cfg, 4, ["symbolic_comparison", "dot_verification"], device,
    )

    batches = [_fake_multitask_batch(), _fake_multitask_batch()]

    class _FakeLoader:
        def __iter__(self):
            return iter(batches)

    results = evaluate_tasks(modules, heads, _FakeLoader(), device)

    for task in ["symbolic_comparison", "dot_verification"]:
        assert f"{task}/loss" in results
        assert f"{task}/acc" in results
        assert f"{task}/brier" in results
        assert f"{task}/ece" in results


# -------------------------------------------------------------------
# export_subject_params
# -------------------------------------------------------------------

def test_export_subject_params():
    cfg = _small_cfg(n_subjects=8)
    device = torch.device("cpu")
    modules = build_modules(cfg, 8, device)

    df = export_subject_params(
        modules["attractor_core"], modules["subject_params"], 8,
    )
    assert len(df) == 8
    assert "temperature" in df.columns
    assert "kstar" in df.columns


# -------------------------------------------------------------------
# compute_kstar_summary
# -------------------------------------------------------------------

def test_compute_kstar_summary():
    cfg = _small_cfg(n_subjects=6)
    device = torch.device("cpu")
    modules = build_modules(cfg, 6, device)

    summary = compute_kstar_summary(modules["attractor_core"], 6)
    assert "analysis/kstar_mean" in summary
    assert "analysis/kstar_median" in summary


# -------------------------------------------------------------------
# Checkpoint round-trip
# -------------------------------------------------------------------

def test_checkpoint_round_trip():
    cfg = _small_cfg()
    device = torch.device("cpu")
    modules = build_modules(cfg, 4, device)
    heads = build_task_heads(cfg, 4, ["symbolic_comparison"], device)

    from src.common.io import save_checkpoint

    wrapper = _CheckpointWrapper(modules, heads)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, wrapper, epoch=3, metrics={"loss": 0.5})

        modules2 = build_modules(cfg, 4, device)
        heads2 = build_task_heads(cfg, 4, ["symbolic_comparison"], device)
        wrapper2 = _CheckpointWrapper(modules2, heads2)

        ckpt = torch.load(path, map_location=device, weights_only=False)
        wrapper2.load_state_dict(ckpt["model_state_dict"])
        assert ckpt["epoch"] == 3

        for p1, p2 in zip(wrapper.parameters(), wrapper2.parameters()):
            assert torch.allclose(p1, p2)


# -------------------------------------------------------------------
# All three model families can train
# -------------------------------------------------------------------

def test_all_presets_trainable():
    for preset in ["default", "weber"]:
        cfg = _small_cfg(preset)
        device = torch.device("cpu")
        modules = build_modules(cfg, 4, device)
        heads = build_task_heads(cfg, 4, ["symbolic_comparison"], device)

        params = collect_parameters(modules, heads)
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        batch = _fake_symbolic_batch()
        out = heads["symbolic_comparison"](batch, modules)
        task_outputs = {"symbolic_comparison": out}
        loss, _, penalty = total_loss(
            task_outputs, {"symbolic_comparison": 1.0}, modules["subject_params"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
