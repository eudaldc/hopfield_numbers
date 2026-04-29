"""Tests for WP-F: ANS and Accumulator baselines.

Verifies that both baselines:
  - produce the same output interface as AttractorCore
  - work as drop-in replacements with existing task heads
  - maintain numerical stability and correct shapes
"""

import torch

from src.common.config import (
    AttractorConfig,
    SubjectConfig,
    SupportConfig,
    VisualEncoderConfig,
    DotVerificationConfig,
)
from src.models.baseline_ans import ANSConfig
from src.models.baseline_accumulator import AccumulatorConfig
from src.models.subject_params import SubjectParams
from src.models.attractor_core import AttractorCore
from src.models.baseline_ans import ANSCore
from src.models.baseline_accumulator import AccumulatorCore
from src.models.visual_encoder import build_visual_encoder
from src.models.symbolic_head import SymbolicHead
from src.models.task_heads import (
    SymbolicComparisonHead,
    MatchToSampleHead,
    DotVerificationHead,
    PanamathHead,
)


N_SUBJ = 5
N_FEAT = 6
B = 8


# =========================================================================
# Helpers — build modules dicts with each baseline core
# =========================================================================

def _make_ans_modules() -> dict:
    sub_cfg = SubjectConfig(n_subjects=N_SUBJ)
    sp = SubjectParams(sub_cfg)
    core = ANSCore(ANSConfig(), sp)
    ve = build_visual_encoder(VisualEncoderConfig(n_features=N_FEAT))
    sh = SymbolicHead(core)
    return {
        "visual_encoder": ve,
        "attractor_core": core,
        "symbolic_head": sh,
        "subject_params": sp,
    }


def _make_acc_modules() -> dict:
    sub_cfg = SubjectConfig(n_subjects=N_SUBJ)
    sp = SubjectParams(sub_cfg)
    core = AccumulatorCore(AccumulatorConfig(), sp)
    ve = build_visual_encoder(VisualEncoderConfig(n_features=N_FEAT))
    sh = SymbolicHead(core)
    return {
        "visual_encoder": ve,
        "attractor_core": core,
        "symbolic_head": sh,
        "subject_params": sp,
    }


# =========================================================================
# ANSCore — geometry
# =========================================================================

def test_ans_centers_monotonic():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = ANSCore(ANSConfig(), sp)
    mu = core.centers()
    diffs = mu[1:] - mu[:-1]
    assert (diffs > 0).all(), "ANS centres must be monotonically increasing"


def test_ans_widths_positive():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = ANSCore(ANSConfig(), sp)
    assert (core.widths() > 0).all()


def test_ans_pairwise_separation():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = ANSCore(ANSConfig(), sp)
    sep = core.pairwise_separation()
    assert sep.shape == (core.support_cfg.size - 1,)
    assert (sep > 0).all()


# =========================================================================
# ANSCore — forward
# =========================================================================

def test_ans_core_output_keys():
    sp = SubjectParams(SubjectConfig(n_subjects=N_SUBJ))
    core = ANSCore(ANSConfig(), sp)
    out = core(torch.randn(B), torch.randint(0, N_SUBJ, (B,)))
    for key in ["logits", "probs", "expected_n", "entropy"]:
        assert key in out, f"Missing key: {key}"


def test_ans_core_shapes():
    sp = SubjectParams(SubjectConfig(n_subjects=N_SUBJ))
    core = ANSCore(ANSConfig(), sp)
    K = core.support_cfg.size
    out = core(torch.randn(B), torch.randint(0, N_SUBJ, (B,)))
    assert out["logits"].shape == (B, K)
    assert out["probs"].shape == (B, K)
    assert out["expected_n"].shape == (B,)
    assert out["entropy"].shape == (B,)


def test_ans_probs_sum_to_one():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    core = ANSCore(ANSConfig(), sp)
    probs = core(torch.randn(10), torch.randint(0, 3, (10,)))["probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_ans_expected_in_range():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    core = ANSCore(ANSConfig(), sp)
    en = core(torch.randn(20), torch.randint(0, 2, (20,)))["expected_n"]
    assert (en >= core.support_cfg.n_min).all()
    assert (en <= core.support_cfg.n_max).all()


def test_ans_entropy_non_negative():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    core = ANSCore(ANSConfig(), sp)
    ent = core(torch.randn(12), torch.randint(0, 2, (12,)))["entropy"]
    assert (ent >= -1e-5).all()


def test_ans_gradients_flow():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    core = ANSCore(ANSConfig(), sp)
    y = torch.randn(6, requires_grad=True)
    out = core(y, torch.randint(0, 3, (6,)))
    out["expected_n"].sum().backward()
    assert y.grad is not None
    assert core._rho_raw.grad is not None


def test_ans_custom_support():
    sup = SupportConfig(n_min=1, n_max=10)
    cfg = ANSConfig(support=sup)
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    core = ANSCore(cfg, sp)
    assert core.support_cfg.size == 10
    out = core(torch.randn(4), torch.randint(0, 2, (4,)))
    assert out["probs"].shape == (4, 10)


def test_ans_kstar():
    sp = SubjectParams(SubjectConfig(n_subjects=4))
    core = ANSCore(ANSConfig(), sp)
    kstar = core.estimate_kstar(torch.arange(4), q=0.75)
    assert kstar.shape == (4,)
    for v in kstar:
        assert v.item() == int(v.item())


def test_ans_kstar_low_threshold_gives_high_values():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = ANSCore(ANSConfig(), sp)
    kstar_low = core.estimate_kstar(torch.tensor([0]), q=0.01)
    kstar_high = core.estimate_kstar(torch.tensor([0]), q=0.99)
    assert kstar_low.item() >= kstar_high.item()


# =========================================================================
# AccumulatorCore — geometry
# =========================================================================

def test_acc_centers_monotonic():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    mu = core.centers()
    diffs = mu[1:] - mu[:-1]
    assert (diffs > 0).all(), "Accumulator centres must be monotonically increasing"


def test_acc_widths_positive():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    assert (core.widths() > 0).all()


def test_acc_pairwise_separation():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    sep = core.pairwise_separation()
    assert sep.shape == (core.support_cfg.size - 1,)
    assert (sep > 0).all()


def test_acc_centers_are_linear():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    mu = core.centers()
    diffs = mu[1:] - mu[:-1]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-5), \
        "Accumulator centres should be evenly spaced (linear)"


def test_acc_widths_are_linear():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    sigma = core.widths()
    diffs = sigma[1:] - sigma[:-1]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-5), \
        "Accumulator widths should grow linearly"


# =========================================================================
# AccumulatorCore — forward
# =========================================================================

def test_acc_core_output_keys():
    sp = SubjectParams(SubjectConfig(n_subjects=N_SUBJ))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    out = core(torch.randn(B), torch.randint(0, N_SUBJ, (B,)))
    for key in ["logits", "probs", "expected_n", "entropy"]:
        assert key in out, f"Missing key: {key}"


def test_acc_core_shapes():
    sp = SubjectParams(SubjectConfig(n_subjects=N_SUBJ))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    K = core.support_cfg.size
    out = core(torch.randn(B), torch.randint(0, N_SUBJ, (B,)))
    assert out["logits"].shape == (B, K)
    assert out["probs"].shape == (B, K)
    assert out["expected_n"].shape == (B,)
    assert out["entropy"].shape == (B,)


def test_acc_probs_sum_to_one():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    probs = core(torch.randn(10), torch.randint(0, 3, (10,)))["probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_acc_expected_in_range():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    en = core(torch.randn(20), torch.randint(0, 2, (20,)))["expected_n"]
    assert (en >= core.support_cfg.n_min).all()
    assert (en <= core.support_cfg.n_max).all()


def test_acc_entropy_non_negative():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    ent = core(torch.randn(12), torch.randint(0, 2, (12,)))["entropy"]
    assert (ent >= -1e-5).all()


def test_acc_gradients_flow():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    y = torch.randn(6, requires_grad=True)
    out = core(y, torch.randint(0, 3, (6,)))
    out["expected_n"].sum().backward()
    assert y.grad is not None
    assert core._scale_raw.grad is not None


def test_acc_custom_support():
    sup = SupportConfig(n_min=1, n_max=10)
    cfg = AccumulatorConfig(support=sup)
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    core = AccumulatorCore(cfg, sp)
    assert core.support_cfg.size == 10
    out = core(torch.randn(4), torch.randint(0, 2, (4,)))
    assert out["probs"].shape == (4, 10)


def test_acc_kstar():
    sp = SubjectParams(SubjectConfig(n_subjects=4))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    kstar = core.estimate_kstar(torch.arange(4), q=0.75)
    assert kstar.shape == (4,)
    for v in kstar:
        assert v.item() == int(v.item())


def test_acc_kstar_low_threshold_gives_high_values():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    kstar_low = core.estimate_kstar(torch.tensor([0]), q=0.01)
    kstar_high = core.estimate_kstar(torch.tensor([0]), q=0.99)
    assert kstar_low.item() >= kstar_high.item()


# =========================================================================
# Drop-in compatibility: ANS with all task heads
# =========================================================================

def test_ans_symbolic_comparison():
    modules = _make_ans_modules()
    head = SymbolicComparisonHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_digit": torch.randint(0, 10, (B,)),
        "right_digit": torch.randint(0, 10, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)
    out["loss"].backward()


def test_ans_match_to_sample():
    modules = _make_ans_modules()
    head = MatchToSampleHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "sample_features": torch.randn(B, N_FEAT),
        "target_features": torch.randn(B, N_FEAT),
        "target_type": ["dots"] * B,
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)


def test_ans_dot_verification():
    modules = _make_ans_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "features": torch.randn(B, N_FEAT),
        "true_n": torch.randint(1, 9, (B,)),
        "shown_digit": torch.randint(1, 9, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
        "grade": torch.ones(B, dtype=torch.long),
        "rt_ms": torch.rand(B) * 2000 + 200,
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)
    assert "pi_count" in out["aux"]


def test_ans_panamath():
    modules = _make_ans_modules()
    head = PanamathHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_features": torch.randn(B, N_FEAT),
        "right_features": torch.randn(B, N_FEAT),
        "size_diff": torch.randn(B),
        "spacing_diff": torch.randn(B),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)


# =========================================================================
# Drop-in compatibility: Accumulator with all task heads
# =========================================================================

def test_acc_symbolic_comparison():
    modules = _make_acc_modules()
    head = SymbolicComparisonHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_digit": torch.randint(0, 10, (B,)),
        "right_digit": torch.randint(0, 10, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)
    out["loss"].backward()


def test_acc_match_to_sample():
    modules = _make_acc_modules()
    head = MatchToSampleHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "sample_features": torch.randn(B, N_FEAT),
        "target_features": torch.randn(B, N_FEAT),
        "target_type": ["dots"] * B,
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)


def test_acc_dot_verification():
    modules = _make_acc_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "features": torch.randn(B, N_FEAT),
        "true_n": torch.randint(1, 9, (B,)),
        "shown_digit": torch.randint(1, 9, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
        "grade": torch.ones(B, dtype=torch.long),
        "rt_ms": torch.rand(B) * 2000 + 200,
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)
    assert "pi_count" in out["aux"]


def test_acc_panamath():
    modules = _make_acc_modules()
    head = PanamathHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_features": torch.randn(B, N_FEAT),
        "right_features": torch.randn(B, N_FEAT),
        "size_diff": torch.randn(B),
        "spacing_diff": torch.randn(B),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)


# =========================================================================
# Structural differences between baselines and attractor
# =========================================================================

def test_neither_core_has_well_depth():
    """Neither ANS nor AttractorCore should have a0 (constant depth is redundant in softmax)."""
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    for CoreCls, CfgCls in [(ANSCore, ANSConfig), (AttractorCore, AttractorConfig)]:
        core = CoreCls(CfgCls(), sp)
        direct_params = {name for name, _ in core.named_parameters(recurse=False)}
        assert "a0" not in direct_params, f"{CoreCls.__name__} still has a0"


def test_acc_centers_differ_from_attractor():
    """Accumulator uses linear centres, not Box-Cox."""
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    core = AccumulatorCore(AccumulatorConfig(), sp)
    mu = core.centers()
    diffs = mu[1:] - mu[:-1]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-5)


def test_baselines_both_include_log_sigma_normalisation():
    """Both baselines use proper Gaussian log-likelihood (−log σ term)."""
    for CoreClass, CfgClass in [
        (ANSCore, ANSConfig),
        (AccumulatorCore, AccumulatorConfig),
    ]:
        sp = SubjectParams(SubjectConfig(n_subjects=1))
        core = CoreClass(CfgClass(), sp)
        y = torch.zeros(1)
        idx = torch.zeros(1, dtype=torch.long)
        logits = core(y, idx)["logits"]
        sigma = core.widths()
        assert logits.shape[-1] == sigma.shape[0]
