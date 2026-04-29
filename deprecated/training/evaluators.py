"""Evaluation utilities and subject-parameter export.

Provides richer evaluation than ``Trainer.evaluate`` — including
calibration metrics, Brier scores, and subject-level parameter
extraction for downstream analyses.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.metrics import brier_score, calibration_bins
from src.training.engine import _to_device, MetricAccumulator
from src.training.losses import multitask_loss


# ======================================================================
# Full evaluation
# ======================================================================

@torch.no_grad()
def evaluate_tasks(
    modules: dict,
    heads: dict,
    loader: DataLoader,
    device: torch.device,
    task_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate with per-task breakdown, calibration, and Brier scores.

    Parameters
    ----------
    modules : modules dict (visual_encoder, attractor_core, …)
    heads   : {task_name: head_module}
    loader  : DataLoader (should be deterministic / no shuffle)
    device  : torch device

    Returns
    -------
    Flat dict of metric_name → float.
    """
    for m in modules.values():
        m.eval()
    for h in heads.values():
        h.eval()

    if task_weights is None:
        task_weights = {t: 1.0 for t in heads}

    acc = MetricAccumulator()
    task_preds: dict[str, list[torch.Tensor]] = {t: [] for t in heads}
    task_labels: dict[str, list[torch.Tensor]] = {t: [] for t in heads}

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _to_device(batch, device)
        task_outputs: dict[str, dict] = {}
        n_samples = 0

        for task_name, task_batch in batch.items():
            if task_name not in heads:
                continue
            out = heads[task_name](task_batch, modules)
            task_outputs[task_name] = out
            bs = task_batch["label"].shape[0] if "label" in task_batch else 1
            n_samples += bs

            acc.update({f"{task_name}/loss": out["loss"]}, weight=bs)
            for mk, mv in out["metrics"].items():
                acc.update({f"{task_name}/{mk}": mv}, weight=bs)

            task_preds[task_name].append(out["pred_probs"].cpu())
            task_labels[task_name].append(task_batch["label"].cpu())

        if task_outputs:
            agg_loss, _ = multitask_loss(task_outputs, task_weights)
            acc.update({"loss_total": agg_loss}, weight=n_samples)

    results = acc.compute()

    for task_name in heads:
        if not task_preds[task_name]:
            continue
        preds = torch.cat(task_preds[task_name])
        labels = torch.cat(task_labels[task_name])
        results[f"{task_name}/brier"] = brier_score(preds, labels).item()
        cal = calibration_bins(preds, labels)
        results[f"{task_name}/ece"] = cal["ece"].item()

    return results


# ======================================================================
# Subject-parameter export
# ======================================================================

@torch.no_grad()
def export_subject_params(
    core,
    subject_params,
    n_subjects: int,
    q: float = 0.75,
) -> pd.DataFrame:
    """Export per-subject learned parameters to a DataFrame.

    Columns: subject_idx, temperature, width_scale, tau_sym,
    count_bias, kstar.
    """
    device = next(subject_params.parameters()).device
    idx = torch.arange(n_subjects, device=device)

    records: dict[str, np.ndarray | list] = {
        "subject_idx": list(range(n_subjects)),
        "temperature": subject_params.temperature(idx).cpu().numpy(),
        "width_scale": subject_params.width_scale(idx).cpu().numpy(),
        "tau_sym": subject_params.tau_sym(idx).cpu().numpy(),
        "count_bias": subject_params.count_bias(idx).cpu().numpy(),
    }

    kstar = core.estimate_kstar(idx, q=q)
    records["kstar"] = kstar.cpu().numpy()

    return pd.DataFrame(records)


@torch.no_grad()
def compute_synth_rt_metrics(
    cleanup_outputs: list[dict],
    n_true_list: list[torch.Tensor],
    rt_proxy,
    n_min: int = 1,
    n_max: int = 9,
) -> dict[str, float]:
    """Compute RT-proxy statistics over synthetic benchmark data.

    Parameters
    ----------
    cleanup_outputs : list of output dicts from DynamicAttractorCore.forward
    n_true_list     : list of true-numerosity tensors (matching cleanup_outputs)
    rt_proxy        : RTProxy instance
    n_min, n_max    : integer range for per-N breakdown

    Returns
    -------
    Flat dict with keys like ``rt/mean_n{N}``, ``rt/mean_steps``,
    ``rt/frac_converged``, ``rt/mean_basin_margin``.
    """
    all_n_steps: list[torch.Tensor] = []
    all_margin: list[torch.Tensor] = []
    all_converged: list[torch.Tensor] = []
    all_n_true: list[torch.Tensor] = []
    all_rt: list[torch.Tensor] = []

    for out, n_true in zip(cleanup_outputs, n_true_list):
        n_steps = out["n_steps"]
        margin = out["basin_margin"]
        rt = rt_proxy(n_steps, basin_margin=margin)
        all_n_steps.append(n_steps.float())
        all_margin.append(margin)
        all_converged.append(out["converged"].float())
        all_n_true.append(n_true)
        all_rt.append(rt)

    n_steps_cat = torch.cat(all_n_steps)
    margin_cat = torch.cat(all_margin)
    converged_cat = torch.cat(all_converged)
    n_true_cat = torch.cat(all_n_true)
    rt_cat = torch.cat(all_rt)

    results: dict[str, float] = {
        "rt/mean_steps": n_steps_cat.mean().item(),
        "rt/frac_converged": converged_cat.mean().item(),
        "rt/mean_basin_margin": margin_cat.mean().item(),
        "rt/mean_rt": rt_cat.mean().item(),
    }
    for n in range(n_min, n_max + 1):
        mask = n_true_cat == n
        if mask.sum() == 0:
            continue
        results[f"rt/mean_rt_n{n}"] = rt_cat[mask].mean().item()
        results[f"rt/mean_steps_n{n}"] = n_steps_cat[mask].mean().item()

    return results


@torch.no_grad()
def compute_kstar_summary(
    core,
    n_subjects: int,
    q: float = 0.75,
) -> dict[str, float]:
    """Mean and median K* across all subjects."""
    device = next(core.parameters()).device
    idx = torch.arange(n_subjects, device=device)
    kstar = core.estimate_kstar(idx, q=q).float()
    return {
        "analysis/kstar_mean": kstar.mean().item(),
        "analysis/kstar_median": kstar.median().item(),
    }
