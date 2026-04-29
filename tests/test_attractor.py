"""Tests for WP-D: subject parameters, attractor core, and symbolic head."""

import torch
import torch.nn.functional as F

from src.common.config import AttractorConfig, SubjectConfig, SupportConfig
from src.models.subject_params import SubjectParams, inverse_softplus
from src.models.attractor_core import AttractorCore
from src.models.symbolic_head import SymbolicHead


# =========================================================================
# inverse_softplus
# =========================================================================

def test_inverse_softplus_roundtrip():
    for x in [0.1, 0.5, 1.0, 5.0, 25.0]:
        raw = inverse_softplus(x)
        recovered = F.softplus(torch.tensor(raw)).item()
        assert abs(recovered - x) < 1e-4, f"Failed for x={x}"


# =========================================================================
# SubjectParams
# =========================================================================

def test_subject_params_shapes():
    sp = SubjectParams(SubjectConfig(n_subjects=10))
    idx = torch.arange(5)
    assert sp.temperature(idx).shape == (5,)
    assert sp.width_scale(idx).shape == (5,)
    assert sp.tau_sym(idx).shape == (5,)
    assert sp.count_bias(idx).shape == (5,)


def test_subject_params_positive():
    sp = SubjectParams(SubjectConfig(n_subjects=8))
    idx = torch.arange(8)
    assert (sp.temperature(idx) > 0).all()
    assert (sp.width_scale(idx) > 0).all()
    assert (sp.tau_sym(idx) > 0).all()


def test_subject_params_temperature_init():
    sp = SubjectParams(SubjectConfig(n_subjects=4, temp_init_mean=2.0))
    temps = sp.temperature(torch.arange(4))
    assert (temps - 2.0).abs().max() < 0.01


def test_subject_params_width_scale_near_one():
    sp = SubjectParams(SubjectConfig(n_subjects=4, width_init_std=0.001))
    ws = sp.width_scale(torch.arange(4))
    assert (ws - 1.0).abs().max() < 0.01


def test_subject_params_tau_sym_init():
    sp = SubjectParams(SubjectConfig(n_subjects=3, tau_sym_init=2.5))
    taus = sp.tau_sym(torch.arange(3))
    assert (taus - 2.5).abs().max() < 0.01


def test_subject_params_l2_penalty():
    sp = SubjectParams(SubjectConfig(n_subjects=5, l2_penalty=0.01))
    p = sp.l2_penalty()
    assert p.item() >= 0.0
    assert p.requires_grad


def test_subject_params_different_subjects():
    sp = SubjectParams(SubjectConfig(n_subjects=3, width_init_std=0.5))
    ws = sp.width_scale(torch.arange(3))
    assert not torch.allclose(ws, ws[0].expand_as(ws), atol=1e-6)


# =========================================================================
# AttractorCore — geometry
# =========================================================================

def test_attractor_centers_monotonic():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    ac = AttractorCore(AttractorConfig(), sp)
    mu = ac.centers()
    diffs = mu[1:] - mu[:-1]
    assert (diffs > 0).all(), "Centres must be monotonically increasing"


def test_attractor_widths_positive():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    ac = AttractorCore(AttractorConfig(), sp)
    assert (ac.widths() > 0).all()


def test_attractor_pairwise_separation():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    ac = AttractorCore(AttractorConfig(), sp)
    sep = ac.pairwise_separation()
    assert sep.shape == (ac.support_cfg.size - 1,)
    assert (sep > 0).all()


# =========================================================================
# AttractorCore — forward
# =========================================================================

def test_attractor_core_shapes():
    sp = SubjectParams(SubjectConfig(n_subjects=5))
    ac = AttractorCore(AttractorConfig(), sp)
    B, K = 8, ac.support_cfg.size
    out = ac(torch.randn(B), torch.randint(0, 5, (B,)))
    assert out["logits"].shape == (B, K)
    assert out["probs"].shape == (B, K)
    assert out["expected_n"].shape == (B,)
    assert out["entropy"].shape == (B,)


def test_attractor_probs_sum_to_one():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    ac = AttractorCore(AttractorConfig(), sp)
    probs = ac(torch.randn(10), torch.randint(0, 3, (10,)))["probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_attractor_expected_in_range():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    ac = AttractorCore(AttractorConfig(), sp)
    en = ac(torch.randn(20), torch.randint(0, 2, (20,)))["expected_n"]
    assert (en >= ac.support_cfg.n_min).all()
    assert (en <= ac.support_cfg.n_max).all()


def test_attractor_entropy_non_negative():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    ac = AttractorCore(AttractorConfig(), sp)
    ent = ac(torch.randn(12), torch.randint(0, 2, (12,)))["entropy"]
    assert (ent >= -1e-5).all()


def test_attractor_gradients_flow():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    ac = AttractorCore(AttractorConfig(), sp)
    y = torch.randn(6, requires_grad=True)
    out = ac(y, torch.randint(0, 3, (6,)))
    out["expected_n"].sum().backward()
    assert y.grad is not None
    assert ac.rho.grad is not None


def test_attractor_custom_support():
    sup = SupportConfig(n_min=1, n_max=10)
    cfg = AttractorConfig(support=sup)
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    ac = AttractorCore(cfg, sp)
    assert ac.support_cfg.size == 10
    out = ac(torch.randn(4), torch.randint(0, 2, (4,)))
    assert out["probs"].shape == (4, 10)


# =========================================================================
# AttractorCore — kstar
# =========================================================================

def test_attractor_kstar():
    sp = SubjectParams(SubjectConfig(n_subjects=4))
    ac = AttractorCore(AttractorConfig(), sp)
    kstar = ac.estimate_kstar(torch.arange(4), q=0.75)
    assert kstar.shape == (4,)
    for v in kstar:
        assert v.item() == int(v.item())


def test_attractor_kstar_low_threshold_gives_high_values():
    sp = SubjectParams(SubjectConfig(n_subjects=1))
    ac = AttractorCore(AttractorConfig(), sp)
    kstar_low = ac.estimate_kstar(torch.tensor([0]), q=0.01)
    kstar_high = ac.estimate_kstar(torch.tensor([0]), q=0.99)
    assert kstar_low.item() >= kstar_high.item()


# =========================================================================
# Box-Cox special cases
# =========================================================================

def test_boxcox_near_zero_rho():
    z = torch.tensor([1.0, 2.0, 5.0, 10.0])
    rho = torch.tensor(1e-8)
    result = AttractorCore._boxcox(z, rho)
    expected = torch.log(z)
    assert torch.allclose(result, expected, atol=1e-4)


def test_boxcox_rho_one():
    z = torch.tensor([1.0, 2.0, 5.0])
    rho = torch.tensor(1.0)
    result = AttractorCore._boxcox(z, rho)
    expected = z - 1.0
    assert torch.allclose(result, expected, atol=1e-4)


# =========================================================================
# SymbolicHead
# =========================================================================

def test_symbolic_head_shapes():
    sp = SubjectParams(SubjectConfig(n_subjects=5))
    ac = AttractorCore(AttractorConfig(), sp)
    sh = SymbolicHead(ac)
    K = ac.support_cfg.size
    out = sh(torch.tensor([3, 7, 5]), torch.randint(0, 5, (3,)))
    assert out["digit_probs"].shape == (3, K)
    assert out["expected_n"].shape == (3,)


def test_symbolic_head_probs_sum_to_one():
    sp = SubjectParams(SubjectConfig(n_subjects=3))
    ac = AttractorCore(AttractorConfig(), sp)
    sh = SymbolicHead(ac)
    probs = sh(torch.randint(0, 10, (8,)), torch.randint(0, 3, (8,)))["digit_probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_symbolic_head_expected_near_digit():
    sp = SubjectParams(SubjectConfig(n_subjects=1, tau_sym_init=0.01))
    ac = AttractorCore(AttractorConfig(), sp)
    sh = SymbolicHead(ac)
    digits = torch.tensor([3, 7, 10])
    en = sh(digits, torch.zeros(3, dtype=torch.long))["expected_n"]
    assert (en - digits.float()).abs().max() < 1.0


def test_symbolic_head_gradients_flow():
    sp = SubjectParams(SubjectConfig(n_subjects=2))
    ac = AttractorCore(AttractorConfig(), sp)
    sh = SymbolicHead(ac)
    out = sh(torch.tensor([5, 8]), torch.tensor([0, 1]))
    out["expected_n"].sum().backward()
    assert ac.rho.grad is not None
