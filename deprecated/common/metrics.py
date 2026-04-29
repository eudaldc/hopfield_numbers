"""Metric helpers and W&B metric-name constants.

All metric functions operate on tensors and return scalars.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# W&B metric name constants
# ---------------------------------------------------------------------------

METRIC_NAMES = {
    # Training / validation aggregates
    "train_loss": "train/loss_total",
    "val_loss": "val/loss_total",
    # Per-task losses
    "val_symbolic_nll": "val/symbolic/nll",
    "val_mts_nll": "val/mts/nll",
    "val_dotmix_nll": "val/dotmix/nll",
    "val_panamath_nll": "val/panamath/nll",
    # Per-task accuracies
    "val_symbolic_acc": "val/symbolic/acc",
    "val_mts_acc": "val/mts/acc",
    "val_dotmix_acc": "val/dotmix/acc",
    "val_panamath_acc": "val/panamath/acc",
    # RT metrics
    "val_dotmix_rt_rmse": "val/dotmix/rt_rmse",
    "val_dotmix_rt_mae": "val/dotmix/rt_mae",
    # Analysis / scientific outputs
    "rho": "analysis/rho",
    "lam": "analysis/lambda",
    "kstar_mean": "analysis/kstar_mean",
    "kstar_median": "analysis/kstar_median",
}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def accuracy(pred_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary accuracy from predicted probabilities and 0/1 labels.

    Args:
        pred_probs: (B,) predicted probability of positive class.
        labels: (B,) binary ground-truth labels.
    """
    preds = (pred_probs >= 0.5).long()
    return (preds == labels.long()).float().mean()


def multiclass_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Top-1 accuracy for multiclass logits.

    Args:
        logits: (B, C) raw logits.
        targets: (B,) integer class labels.
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


def binary_cross_entropy(
    pred_probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Numerically stable BCE, averaged over batch.

    Args:
        pred_probs: (B,) predicted probabilities in (0, 1).
        labels: (B,) binary labels.
        eps: clamp value for numerical safety.
    """
    p = pred_probs.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(p, labels.float(), reduction="mean")


def multiclass_nll(
    log_probs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood for multiclass predictions.

    Args:
        log_probs: (B, C) log-probabilities.
        targets: (B,) integer class labels.
    """
    return F.nll_loss(log_probs, targets, reduction="mean")


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""
    return ((predictions - targets) ** 2).mean().sqrt()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return (predictions - targets).abs().mean()


def brier_score(
    pred_probs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Brier score (mean squared error of probability estimates).

    Lower is better. Range [0, 1].
    """
    return ((pred_probs - labels.float()) ** 2).mean()


def calibration_bins(
    pred_probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> dict[str, torch.Tensor]:
    """Compute calibration statistics binned by predicted probability.

    Returns a dict with bin_edges, bin_accs, bin_confs, bin_counts, and
    expected_calibration_error (ECE).
    """
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=pred_probs.device)
    bin_accs = torch.zeros(n_bins, device=pred_probs.device)
    bin_confs = torch.zeros(n_bins, device=pred_probs.device)
    bin_counts = torch.zeros(n_bins, device=pred_probs.device)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (pred_probs >= lo) & (pred_probs <= hi)
        else:
            mask = (pred_probs >= lo) & (pred_probs < hi)
        count = mask.sum().float()
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = labels[mask].float().mean()
            bin_confs[i] = pred_probs[mask].mean()

    total = bin_counts.sum()
    ece = torch.tensor(0.0, device=pred_probs.device)
    if total > 0:
        ece = ((bin_counts / total) * (bin_accs - bin_confs).abs()).sum()

    return {
        "bin_edges": bin_boundaries,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "ece": ece,
    }
