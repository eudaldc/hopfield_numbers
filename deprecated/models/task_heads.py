"""Task heads: per-task likelihoods on top of the attractor + visual encoder.

Each head implements::

    forward(batch, modules) -> {loss, pred_probs, pred_labels, metrics, aux}

``modules`` is a dict with keys:
    visual_encoder, attractor_core, symbolic_head, subject_params
    [optional] dynamic_core  — present when attractor_mode="dynamic"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import DotVerificationConfig
from src.common.metrics import accuracy


# --------------------------------------------------------------------------
# Unified attractor dispatch
# --------------------------------------------------------------------------

def _get_attractor_out(
    y: torch.Tensor,
    subject_idx: torch.Tensor,
    modules: dict,
) -> dict:
    """Return posteriors from whichever attractor core is active.

    In dynamic mode (``modules`` contains ``"dynamic_core"``), runs the
    iterative cleanup and returns cleaned posteriors.  In static mode,
    delegates to ``attractor_core`` with per-subject parameters.

    Returns a dict with:
        "probs"      : [B, K]  posterior over integer support
        "expected_n" : [B]     expected numerosity
        "entropy"    : [B]     Shannon entropy of the posterior
        "n_min"      : int     smallest integer in the support
    """
    if "dynamic_core" in modules:
        dyn = modules["dynamic_core"]
        out = dyn(y)
        probs = out["probs_clean"]
        log_p = torch.log(probs.clamp(min=1e-8))
        entropy = -(probs * log_p).sum(dim=-1)
        return {
            "probs": probs,
            "expected_n": out["expected_n_clean"],
            "entropy": entropy,
            "n_min": dyn.support_min,
        }
    ac = modules["attractor_core"]
    out = ac(y, subject_idx)
    return {
        "probs": out["probs"],
        "expected_n": out["expected_n"],
        "entropy": out["entropy"],
        "n_min": ac.support_cfg.n_min,
    }


# --------------------------------------------------------------------------
# E1.  Symbolic number comparison
# --------------------------------------------------------------------------

class SymbolicComparisonHead(nn.Module):
    """Compare two digits; predict P(child chooses 'right').

    score = E[n | right] − E[n | left]
    P(right) = sigmoid(beta · score + bias)
    """

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: dict, modules: dict) -> dict:
        sh = modules["symbolic_head"]
        subj = batch["subject_idx"]

        left_out = sh(batch["left_digit"], subj)
        right_out = sh(batch["right_digit"], subj)

        score = right_out["expected_n"] - left_out["expected_n"]   # [B]
        logit = self.beta * score + self.bias
        p_right = torch.sigmoid(logit)

        label = batch["label"]
        loss = F.binary_cross_entropy_with_logits(
            logit, label, reduction="mean"
        )

        return {
            "loss": loss,
            "pred_probs": p_right,
            "pred_labels": (p_right >= 0.5).long(),
            "metrics": {"acc": accuracy(p_right, label)},
            "aux": {"score": score},
        }


# --------------------------------------------------------------------------
# E2.  Match-to-sample (dots→dots or dots→digit)
# --------------------------------------------------------------------------

class MatchToSampleHead(nn.Module):
    """Same / different judgement between a sample and a target.

    P(same) = sum_n  p(n | sample) · r(n | target)

    where r = p (attractor posterior) for dots targets and
          r = q (symbolic distribution) for digit targets.

    When dots targets are present, sample and target evidence are
    concatenated into a single attractor call to amortise cleanup cost.
    """

    def forward(self, batch: dict, modules: dict) -> dict:
        ve = modules["visual_encoder"]
        sh = modules["symbolic_head"]
        subj = batch["subject_idx"]

        sample_key = f"sample_{ve.BATCH_KEY}"
        y_sample = ve.encode(batch[sample_key])                    # [B]

        target_type = batch["target_type"]                         # list[str]
        dots_flags = [t == "dots" for t in target_type]
        digit_flags = [t == "digit" for t in target_type]
        has_dots = any(dots_flags)

        if has_dots:
            dots_idx = torch.tensor(
                [i for i, f in enumerate(dots_flags) if f],
                dtype=torch.long, device=y_sample.device,
            )
            target_key = f"target_{ve.BATCH_KEY}"
            y_tgt = ve.encode(batch[target_key])                   # [n_dots]

            y_combined = torch.cat([y_sample, y_tgt])
            subj_combined = torch.cat([subj, subj[dots_idx]])
            combined_att = _get_attractor_out(
                y_combined, subj_combined, modules,
            )
            B = y_sample.shape[0]
            sample_probs = combined_att["probs"][:B]               # [B, K]
            K = sample_probs.shape[1]
            device = sample_probs.device
            target_probs = torch.zeros(B, K, device=device)
            target_probs[dots_idx] = combined_att["probs"][B:]
        else:
            sample_att = _get_attractor_out(y_sample, subj, modules)
            sample_probs = sample_att["probs"]                     # [B, K]
            B, K = sample_probs.shape
            device = sample_probs.device
            target_probs = torch.zeros(B, K, device=device)

        if any(digit_flags):
            digit_idx = torch.tensor(
                [i for i, f in enumerate(digit_flags) if f],
                dtype=torch.long, device=device,
            )
            sym_out = sh(batch["target_digit"], subj[digit_idx])
            target_probs[digit_idx] = sym_out["digit_probs"]

        p_same = (sample_probs * target_probs).sum(dim=-1)
        p_same = p_same.clamp(1e-7, 1.0 - 1e-7)

        label = batch["label"]
        loss = F.binary_cross_entropy(p_same, label, reduction="mean")

        return {
            "loss": loss,
            "pred_probs": p_same,
            "pred_labels": (p_same >= 0.5).long(),
            "metrics": {"acc": accuracy(p_same, label)},
            "aux": {},
        }


# --------------------------------------------------------------------------
# E3.  Dot verification (approximate + counting mixture)
# --------------------------------------------------------------------------

def _gaussian_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """Element-wise log N(x; mu, exp(log_sigma)²)."""
    return (
        -0.5 * math.log(2.0 * math.pi)
        - log_sigma
        - 0.5 * ((x - mu) / torch.exp(log_sigma)).pow(2)
    )


class DotVerificationHead(nn.Module):
    """Same / different with counting-route mixture and optional RT model.

    P(same) = (1 − pi_count) · p(n = m | image)
              + pi_count · P_count(same | N, m)

    logit(pi_count) = gamma0_i + gamma1 · N + gamma2 · grade
    """

    def __init__(
        self,
        cfg: DotVerificationConfig | None = None,
        n_subjects: int = 1,
    ):
        super().__init__()
        if cfg is None:
            cfg = DotVerificationConfig()
        self.eps_count = cfg.eps_count
        self.model_rt = cfg.model_rt
        self.rt_loss_weight = cfg.rt_loss_weight

        self.gamma1 = nn.Parameter(torch.tensor(0.0))
        self.gamma2 = nn.Parameter(torch.tensor(0.0))

        if self.model_rt:
            self.alpha_approx = nn.Embedding(n_subjects, 1)
            self.alpha_count = nn.Embedding(n_subjects, 1)
            nn.init.constant_(self.alpha_approx.weight, 6.0)
            nn.init.constant_(self.alpha_count.weight, 7.0)
            self.b_entropy = nn.Parameter(torch.tensor(0.1))
            self.b_N = nn.Parameter(torch.tensor(0.05))
            self.log_sigma_approx = nn.Parameter(torch.tensor(-0.5))
            self.log_sigma_count = nn.Parameter(torch.tensor(-0.5))

    def forward(self, batch: dict, modules: dict) -> dict:
        ve = modules["visual_encoder"]
        sp = modules["subject_params"]
        subj = batch["subject_idx"]

        y = ve(batch)
        att_out = _get_attractor_out(y, subj, modules)
        probs = att_out["probs"]                                  # [B, K]
        B = probs.shape[0]
        device = probs.device

        n_min = att_out["n_min"]
        digit_idx = (batch["shown_digit"] - n_min).long()
        p_approx = probs[torch.arange(B, device=device), digit_idx]

        same = (batch["true_n"] == batch["shown_digit"]).float()
        p_count = same * (1.0 - self.eps_count) + (1.0 - same) * self.eps_count

        logit_count = (
            sp.count_bias(subj)
            + self.gamma1 * batch["true_n"].float()
            + self.gamma2 * batch["grade"].float()
        )
        pi_count = torch.sigmoid(logit_count)

        p_same = (1.0 - pi_count) * p_approx + pi_count * p_count
        p_same = p_same.clamp(1e-7, 1.0 - 1e-7)

        label = batch["label"]
        loss = F.binary_cross_entropy(p_same, label, reduction="mean")

        aux: dict = {"pi_count": pi_count, "p_approx": p_approx}

        if self.model_rt and "rt_ms" in batch:
            valid = batch["rt_ms"] > 0
            if valid.any():
                log_rt = torch.log(batch["rt_ms"][valid])
                s_v = subj[valid]
                ent_v = att_out["entropy"][valid]
                n_v = batch["true_n"][valid].float()
                lc_v = logit_count[valid]

                mu_a = (
                    self.alpha_approx(s_v).squeeze(-1)
                    + self.b_entropy * ent_v
                )
                mu_c = (
                    self.alpha_count(s_v).squeeze(-1) + self.b_N * n_v
                )

                log_1_pi = F.logsigmoid(-lc_v)
                log_pi = F.logsigmoid(lc_v)

                ll_a = log_1_pi + _gaussian_log_prob(
                    log_rt, mu_a, self.log_sigma_approx
                )
                ll_c = log_pi + _gaussian_log_prob(
                    log_rt, mu_c, self.log_sigma_count
                )

                rt_nll = -torch.logsumexp(
                    torch.stack([ll_a, ll_c], dim=-1), dim=-1
                ).mean()
                loss = loss + self.rt_loss_weight * rt_nll
                aux["rt_nll"] = rt_nll

        return {
            "loss": loss,
            "pred_probs": p_same,
            "pred_labels": (p_same >= 0.5).long(),
            "metrics": {"acc": accuracy(p_same, label)},
            "aux": aux,
        }


# --------------------------------------------------------------------------
# E4.  Panamath (non-symbolic comparison with visual cue regressors)
# --------------------------------------------------------------------------

class PanamathHead(nn.Module):
    """Which side has more dots?  Includes visual-cue regressors.

    score = delta_num · (nhat_right − nhat_left)
          + delta_size · size_diff
          + delta_spacing · spacing_diff
          + side_bias
    P(right larger) = sigmoid(score)
    """

    def __init__(self):
        super().__init__()
        self.delta_num = nn.Parameter(torch.tensor(1.0))
        self.delta_size = nn.Parameter(torch.tensor(0.0))
        self.delta_spacing = nn.Parameter(torch.tensor(0.0))
        self.side_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: dict, modules: dict) -> dict:
        ve = modules["visual_encoder"]
        subj = batch["subject_idx"]
        B = subj.shape[0]

        left_key = f"left_{ve.BATCH_KEY}"
        right_key = f"right_{ve.BATCH_KEY}"
        y_left = ve.encode(batch[left_key])
        y_right = ve.encode(batch[right_key])

        y_combined = torch.cat([y_left, y_right])
        subj_combined = subj.repeat(2)
        combined_att = _get_attractor_out(y_combined, subj_combined, modules)
        nhat_left = combined_att["expected_n"][:B]
        nhat_right = combined_att["expected_n"][B:]

        logit = (
            self.delta_num * (nhat_right - nhat_left)
            + self.delta_size * batch["size_diff"]
            + self.delta_spacing * batch["spacing_diff"]
            + self.side_bias
        )
        p_right = torch.sigmoid(logit)
        label = batch["label"]
        loss = F.binary_cross_entropy_with_logits(
            logit, label, reduction="mean"
        )

        return {
            "loss": loss,
            "pred_probs": p_right,
            "pred_labels": (p_right >= 0.5).long(),
            "metrics": {"acc": accuracy(p_right, label)},
            "aux": {"nhat_left": nhat_left, "nhat_right": nhat_right},
        }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

TASK_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "symbolic_comparison": SymbolicComparisonHead,
    "match_to_sample": MatchToSampleHead,
    "dot_verification": DotVerificationHead,
    "panamath": PanamathHead,
}
