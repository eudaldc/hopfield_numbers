"""Tests for WP-E: task heads (symbolic comparison, MTS, dot verification, Panamath)."""

import torch

from src.common.config import (
    AttractorConfig,
    SubjectConfig,
    VisualEncoderConfig,
    DotVerificationConfig,
)
from src.models.subject_params import SubjectParams
from src.models.attractor_core import AttractorCore
from src.models.visual_encoder import build_visual_encoder
from src.models.symbolic_head import SymbolicHead
from src.models.task_heads import (
    SymbolicComparisonHead,
    MatchToSampleHead,
    DotVerificationHead,
    PanamathHead,
    TASK_HEAD_REGISTRY,
)


N_SUBJ = 5
N_FEAT = 6
B = 8


def _make_modules() -> dict:
    sub_cfg = SubjectConfig(n_subjects=N_SUBJ)
    sp = SubjectParams(sub_cfg)
    ac = AttractorCore(AttractorConfig(), sp)
    ve = build_visual_encoder(VisualEncoderConfig(n_features=N_FEAT))
    sh = SymbolicHead(ac)
    return {
        "visual_encoder": ve,
        "attractor_core": ac,
        "symbolic_head": sh,
        "subject_params": sp,
    }


# =========================================================================
# SymbolicComparisonHead
# =========================================================================

def test_symbolic_comparison_output_keys():
    modules = _make_modules()
    head = SymbolicComparisonHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_digit": torch.randint(0, 10, (B,)),
        "right_digit": torch.randint(0, 10, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    for key in ["loss", "pred_probs", "pred_labels", "metrics", "aux"]:
        assert key in out, f"Missing key: {key}"


def test_symbolic_comparison_shapes():
    modules = _make_modules()
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
    assert out["pred_labels"].shape == (B,)


def test_symbolic_comparison_probs_range():
    modules = _make_modules()
    head = SymbolicComparisonHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_digit": torch.randint(0, 10, (B,)),
        "right_digit": torch.randint(0, 10, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    p = head(batch, modules)["pred_probs"]
    assert (p >= 0).all() and (p <= 1).all()


def test_symbolic_comparison_gradients():
    modules = _make_modules()
    head = SymbolicComparisonHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_digit": torch.randint(0, 10, (B,)),
        "right_digit": torch.randint(0, 10, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    out["loss"].backward()
    assert head.beta.grad is not None


# =========================================================================
# MatchToSampleHead
# =========================================================================

def test_mts_all_dots():
    modules = _make_modules()
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


def test_mts_all_digit():
    modules = _make_modules()
    head = MatchToSampleHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "sample_features": torch.randn(B, N_FEAT),
        "target_type": ["digit"] * B,
        "target_digit": torch.randint(0, 10, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)


def test_mts_mixed():
    modules = _make_modules()
    head = MatchToSampleHead()
    Bd, Bn = 5, 3
    total = Bd + Bn
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (total,)),
        "sample_features": torch.randn(total, N_FEAT),
        "target_features": torch.randn(Bd, N_FEAT),
        "target_type": ["dots"] * Bd + ["digit"] * Bn,
        "target_digit": torch.randint(0, 10, (Bn,)),
        "label": torch.randint(0, 2, (total,)).float(),
    }
    out = head(batch, modules)
    assert out["pred_probs"].shape == (total,)
    assert out["loss"].shape == ()


def test_mts_output_keys():
    modules = _make_modules()
    head = MatchToSampleHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "sample_features": torch.randn(B, N_FEAT),
        "target_features": torch.randn(B, N_FEAT),
        "target_type": ["dots"] * B,
        "label": torch.randint(0, 2, (B,)).float(),
    }
    out = head(batch, modules)
    for key in ["loss", "pred_probs", "pred_labels", "metrics", "aux"]:
        assert key in out


def test_mts_probs_range():
    modules = _make_modules()
    head = MatchToSampleHead()
    batch = {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "sample_features": torch.randn(B, N_FEAT),
        "target_features": torch.randn(B, N_FEAT),
        "target_type": ["dots"] * B,
        "label": torch.randint(0, 2, (B,)).float(),
    }
    p = head(batch, modules)["pred_probs"]
    assert (p >= 0).all() and (p <= 1).all()


# =========================================================================
# DotVerificationHead
# =========================================================================

def _dotver_batch():
    return {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "features": torch.randn(B, N_FEAT),
        "true_n": torch.randint(1, 9, (B,)),
        "shown_digit": torch.randint(1, 9, (B,)),
        "label": torch.randint(0, 2, (B,)).float(),
        "grade": torch.ones(B, dtype=torch.long),
        "rt_ms": torch.rand(B) * 2000 + 200,
    }


def test_dotver_output_keys():
    modules = _make_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    out = head(_dotver_batch(), modules)
    for key in ["loss", "pred_probs", "pred_labels", "metrics", "aux"]:
        assert key in out


def test_dotver_shapes():
    modules = _make_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    out = head(_dotver_batch(), modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)
    assert out["pred_labels"].shape == (B,)


def test_dotver_probs_range():
    modules = _make_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    p = head(_dotver_batch(), modules)["pred_probs"]
    assert (p >= 0).all() and (p <= 1).all()


def test_dotver_with_rt():
    modules = _make_modules()
    cfg = DotVerificationConfig(model_rt=True)
    head = DotVerificationHead(cfg, n_subjects=N_SUBJ)
    out = head(_dotver_batch(), modules)
    assert out["loss"].shape == ()
    assert "rt_nll" in out["aux"]


def test_dotver_missing_rt():
    modules = _make_modules()
    cfg = DotVerificationConfig(model_rt=True)
    head = DotVerificationHead(cfg, n_subjects=N_SUBJ)
    batch = _dotver_batch()
    batch["rt_ms"] = torch.full((B,), -1.0)
    out = head(batch, modules)
    assert out["loss"].shape == ()
    assert "rt_nll" not in out["aux"]


def test_dotver_aux_has_pi_count():
    modules = _make_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    aux = head(_dotver_batch(), modules)["aux"]
    assert "pi_count" in aux
    assert "p_approx" in aux
    pi = aux["pi_count"]
    assert (pi >= 0).all() and (pi <= 1).all()


def test_dotver_gradients():
    modules = _make_modules()
    head = DotVerificationHead(DotVerificationConfig(), n_subjects=N_SUBJ)
    out = head(_dotver_batch(), modules)
    out["loss"].backward()
    assert head.gamma1.grad is not None


# =========================================================================
# PanamathHead
# =========================================================================

def _panamath_batch():
    return {
        "subject_idx": torch.randint(0, N_SUBJ, (B,)),
        "left_features": torch.randn(B, N_FEAT),
        "right_features": torch.randn(B, N_FEAT),
        "size_diff": torch.randn(B),
        "spacing_diff": torch.randn(B),
        "label": torch.randint(0, 2, (B,)).float(),
    }


def test_panamath_output_keys():
    modules = _make_modules()
    head = PanamathHead()
    out = head(_panamath_batch(), modules)
    for key in ["loss", "pred_probs", "pred_labels", "metrics", "aux"]:
        assert key in out


def test_panamath_shapes():
    modules = _make_modules()
    head = PanamathHead()
    out = head(_panamath_batch(), modules)
    assert out["loss"].shape == ()
    assert out["pred_probs"].shape == (B,)
    assert out["pred_labels"].shape == (B,)


def test_panamath_probs_range():
    modules = _make_modules()
    head = PanamathHead()
    p = head(_panamath_batch(), modules)["pred_probs"]
    assert (p >= 0).all() and (p <= 1).all()


def test_panamath_gradients():
    modules = _make_modules()
    head = PanamathHead()
    out = head(_panamath_batch(), modules)
    out["loss"].backward()
    assert head.delta_num.grad is not None
    assert head.side_bias.grad is not None


def test_panamath_aux_has_expected_n():
    modules = _make_modules()
    head = PanamathHead()
    aux = head(_panamath_batch(), modules)["aux"]
    assert "nhat_left" in aux
    assert "nhat_right" in aux


# =========================================================================
# Registry
# =========================================================================

def test_task_head_registry():
    assert set(TASK_HEAD_REGISTRY.keys()) == {
        "symbolic_comparison",
        "match_to_sample",
        "dot_verification",
        "panamath",
    }
