"""Phase-transition parameter scan for the dynamic attractor model.

Sweeps one or two parameters of DynamicAttractorCore over a grid and
returns a DataFrame with metrics at each grid point.

Computed metrics
----------------
Small-number regime (N in [n_min, small_n_max]):
    acc_small      : mean accuracy
    rt_small       : mean RT proxy
    rt_slope_small : slope of mean RT vs N (linear fit)

Larger-number regime (N in (small_n_max, n_max]):
    mae_large      : mean absolute error
    rt_large       : mean RT proxy
    rt_slope_large : slope of mean RT vs N

Cleanup geometry:
    mean_steps     : mean cleanup iterations until convergence
    frac_converged : fraction of samples converged within tol
    mean_margin    : mean basin margin (top1 − top2 posterior)
    k_star         : K* geometry threshold (q = 0.75)
    sep_ratio_mean : mean consecutive-well separation ratio
    occ_entropy    : entropy of the basin-occupancy distribution

Per-N breakdown (also stored as DataFrame columns):
    acc_n{N}, rt_n{N}, mae_n{N}  for N = n_min .. n_max

Derived:
    rt_slope_ratio : rt_slope_large / (|rt_slope_small| + eps)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.data.synth_benchmarks import NeutralEvidenceChannel, generate_task_a
from src.models.attractor_dynamics import DynamicAttractorCore
from src.models.rt_proxy import RTProxy


# ── data classes ───────────────────────────────────────────────────────────


@dataclass
class ScanAxis:
    """One parameter axis for the phase scan.

    Parameters
    ----------
    param  : name of the parameter — "kappa" | "tau_in" | "sigma0" | "lam" | "eta" | "rho"
    values : explicit list of values to sweep along this axis
    """

    param: str
    values: list[float]


_VALID_SCAN_PARAMS = frozenset({"kappa", "tau_in", "sigma0", "lam", "eta", "rho"})


@dataclass
class PhaseScanConfig:
    """Configuration for a full phase scan run.

    Parameters
    ----------
    axes        : 1 or 2 ScanAxis objects (2 → 2-D grid scan)
    n_samples   : Task-A samples generated once and reused at every grid point
    n_min       : minimum true numerosity
    n_max       : maximum true numerosity
    small_n_max : threshold separating small (≤) from large (>) regimes
    seed        : random seed for data generation
    rt_mode     : RTProxy mode — "steps" | "steps_plus_margin"
    kappa / tau_in / sigma0 / lam / eta / rho / nu :
        fixed values for parameters not being scanned
    support_min / support_max / max_steps / tol :
        forwarded to DynamicAttractorCore
    """

    axes: list[ScanAxis]
    n_samples: int = 5_000
    n_min: int = 1
    n_max: int = 9
    small_n_max: int = 4
    seed: int = 7
    rt_mode: str = "steps_plus_margin"
    # fixed baseline values for non-scanned parameters
    kappa: float = 1.0
    tau_in: float = 1.0
    sigma0: float = 1.0
    lam: float = 0.5
    eta: float = 0.2
    rho: float = 0.1
    nu: float = 1.0
    support_min: int = 1
    support_max: int = 9
    max_steps: int = 16
    tol: float = 1e-3


# ── model builder ──────────────────────────────────────────────────────────


def _build_model(cfg: PhaseScanConfig, overrides: dict[str, float]) -> DynamicAttractorCore:
    """Build DynamicAttractorCore from config defaults with per-point overrides."""
    init_kwargs: dict[str, float] = {
        "kappa_init": cfg.kappa,
        "tau_in_init": cfg.tau_in,
        "sigma0_init": cfg.sigma0,
        "lambda_init": cfg.lam,
        "eta_init": cfg.eta,
    }
    rho_val = cfg.rho
    for k, v in overrides.items():
        if k == "lam":
            init_kwargs["lambda_init"] = v
        elif k == "rho":
            rho_val = v
        else:
            init_kwargs[f"{k}_init"] = v

    return DynamicAttractorCore(
        support_min=cfg.support_min,
        support_max=cfg.support_max,
        rho_init=rho_val,
        nu_init=cfg.nu,
        cleanup_mode="gradient",
        max_steps=cfg.max_steps,
        tol=cfg.tol,
        **init_kwargs,
    )


# ── metrics ────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_metrics(
    model: DynamicAttractorCore,
    rt_proxy: RTProxy,
    data: dict[str, torch.Tensor],
    n_min: int,
    n_max: int,
    small_n_max: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on pre-generated Task-A data and return a metrics dict.

    Parameters
    ----------
    model      : DynamicAttractorCore — moved to device internally
    rt_proxy   : RTProxy — moved to device internally
    data       : dict from generate_task_a
    n_min/n_max: inclusive numerosity range
    small_n_max: upper bound of small-number regime
    device     : computation device

    Returns
    -------
    dict of scalar float metrics (see module docstring for full list)
    """
    n_true = data["n_true"].to(device)
    y = data["y_noisy"].to(device)

    model = model.to(device).eval()
    rt_proxy = rt_proxy.to(device).eval()

    out = model(y)
    n_hat = out["n_hat"].long()
    n_steps = out["n_steps"]
    basin_margin = out["basin_margin"]
    rt = rt_proxy(n_steps, basin_margin=basin_margin)

    per_n_acc: dict[int, float] = {}
    per_n_rt: dict[int, float] = {}
    per_n_mae: dict[int, float] = {}
    for n in range(n_min, n_max + 1):
        mask = n_true == n
        if mask.sum() == 0:
            continue
        per_n_acc[n] = (n_hat[mask] == n).float().mean().item()
        per_n_rt[n] = rt[mask].mean().item()
        per_n_mae[n] = (n_hat[mask].float() - float(n)).abs().mean().item()

    small_ns = [n for n in range(n_min, small_n_max + 1) if n in per_n_acc]
    large_ns = [n for n in range(small_n_max + 1, n_max + 1) if n in per_n_acc]

    def _slope(xs: list[int], ys_: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return float(np.polyfit(xs, ys_, 1)[0])

    rt_slope_small = _slope(
        [n for n in small_ns if n in per_n_rt],
        [per_n_rt[n] for n in small_ns if n in per_n_rt],
    )
    rt_slope_large = _slope(
        [n for n in large_ns if n in per_n_rt],
        [per_n_rt[n] for n in large_ns if n in per_n_rt],
    )

    mean_steps = n_steps.float().mean().item()
    frac_converged = out["converged"].float().mean().item()
    mean_margin = basin_margin.mean().item()
    k_star = model.k_star_geometry(q=0.75).item()
    sep_ratio_mean = model.separation_ratio().mean().item()

    occ = out["probs_clean"].mean(dim=0)
    occ = occ / (occ.sum() + 1e-12)
    occ_entropy = -(occ * (occ + 1e-12).log()).sum().item()

    eps = 1e-6
    result: dict[str, float] = {
        "acc_small": float(np.mean([per_n_acc[n] for n in small_ns])) if small_ns else float("nan"),
        "rt_small": float(np.mean([per_n_rt[n] for n in small_ns])) if small_ns else float("nan"),
        "rt_slope_small": rt_slope_small,
        "mae_large": float(np.mean([per_n_mae[n] for n in large_ns])) if large_ns else float("nan"),
        "rt_large": float(np.mean([per_n_rt[n] for n in large_ns])) if large_ns else float("nan"),
        "rt_slope_large": rt_slope_large,
        "mean_steps": mean_steps,
        "frac_converged": frac_converged,
        "mean_margin": mean_margin,
        "k_star": k_star,
        "sep_ratio_mean": sep_ratio_mean,
        "occ_entropy": occ_entropy,
        "rt_slope_ratio": rt_slope_large / (abs(rt_slope_small) + eps),
    }
    for n in range(n_min, n_max + 1):
        if n in per_n_acc:
            result[f"acc_n{n}"] = per_n_acc[n]
        if n in per_n_rt:
            result[f"rt_n{n}"] = per_n_rt[n]
        if n in per_n_mae:
            result[f"mae_n{n}"] = per_n_mae[n]

    return result


# ── main scan function ─────────────────────────────────────────────────────


def run_phase_scan(
    cfg: PhaseScanConfig,
    rt_proxy: Optional[RTProxy] = None,
    device: Optional[torch.device] = None,
    channel: Optional[NeutralEvidenceChannel] = None,
) -> "pd.DataFrame":
    """Sweep parameter grid and return a DataFrame of metrics.

    Parameters
    ----------
    cfg      : PhaseScanConfig
    rt_proxy : RTProxy; instantiated with cfg.rt_mode if None
    device   : computation device; defaults to CPU
    channel  : NeutralEvidenceChannel; uses default if None

    Returns
    -------
    pandas DataFrame with one row per grid point.
    Columns: scanned parameter(s) followed by all metrics from compute_metrics.
    """
    import pandas as pd

    if device is None:
        device = torch.device("cpu")
    if rt_proxy is None:
        rt_proxy = RTProxy(mode=cfg.rt_mode)
    if channel is None:
        channel = NeutralEvidenceChannel()

    if not cfg.axes:
        raise ValueError("PhaseScanConfig requires at least one ScanAxis.")
    if len(cfg.axes) > 2:
        raise ValueError("Phase scan supports at most 2 axes.")
    for ax in cfg.axes:
        if ax.param not in _VALID_SCAN_PARAMS:
            raise ValueError(
                f"Unknown scan parameter {ax.param!r}. "
                f"Valid choices: {sorted(_VALID_SCAN_PARAMS)}"
            )

    data = generate_task_a(
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        channel=channel,
    )

    if len(cfg.axes) == 1:
        combos: list[tuple] = [(v,) for v in cfg.axes[0].values]
        param_names = [cfg.axes[0].param]
    else:
        combos = list(itertools.product(cfg.axes[0].values, cfg.axes[1].values))
        param_names = [cfg.axes[0].param, cfg.axes[1].param]

    records: list[dict] = []
    for combo in combos:
        overrides = dict(zip(param_names, combo))
        model = _build_model(cfg, overrides)
        metrics = compute_metrics(
            model, rt_proxy, data,
            cfg.n_min, cfg.n_max, cfg.small_n_max, device,
        )
        records.append({**overrides, **metrics})

    return pd.DataFrame(records)
