"""Loss computation: multitask aggregation and subject-level L2 penalty."""

import torch
import torch.nn.functional as F


def multitask_loss(
    task_outputs: dict[str, dict],
    task_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted sum of per-task losses.

    Parameters
    ----------
    task_outputs : {task_name: head_output_dict}
    task_weights : {task_name: float weight}

    Returns
    -------
    (total_loss, {task_name: unweighted_loss})
    """
    device = next(iter(task_outputs.values()))["loss"].device
    total = torch.tensor(0.0, device=device)
    per_task: dict[str, torch.Tensor] = {}
    for task_name, out in task_outputs.items():
        w = task_weights.get(task_name, 1.0)
        per_task[task_name] = out["loss"]
        total = total + w * out["loss"]
    return total, per_task


def total_loss(
    task_outputs: dict[str, dict],
    task_weights: dict[str, float],
    subject_params,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Total loss = weighted task losses + subject L2 penalty.

    Returns
    -------
    (total_with_penalty, per_task_losses, l2_penalty)
    """
    loss, per_task = multitask_loss(task_outputs, task_weights)
    penalty = subject_params.l2_penalty()
    return loss + penalty, per_task, penalty


def rt_loss(
    rt_pred: torch.Tensor,
    rt_obs: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """MSE loss between predicted and observed reaction times.

    Not used in v0.2 synthetic benchmarks (RT proxy is for analysis only).
    Provided for future empirical RT fitting.

    Parameters
    ----------
    rt_pred : [B] predicted RT (from RTProxy)
    rt_obs  : [B] observed RT (in matching units)
    weight  : scalar multiplier

    Returns
    -------
    Scalar loss tensor.
    """
    return weight * F.mse_loss(rt_pred, rt_obs)
