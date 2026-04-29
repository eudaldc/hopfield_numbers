#!/usr/bin/env python
"""Unified training entry point for the Numerical Attractors PoC.

Covers empirical training (real data) and synthetic benchmark evaluation.
The two key axes are **preset** (``default`` | ``weber``) and
**attractor mode** (``dynamic`` | ``static``).

Checkpoints are saved as ``best_{preset}_{mode}.pt``, e.g.
``best_default_dynamic.pt``, ``best_weber_static.pt``.

Usage examples
--------------

Empirical training with the *default* preset (dynamic mode is the default)::

    python scripts/train_unified.py --preset default

Empirical training with *weber* preset in static (ablation) mode::

    python scripts/train_unified.py --preset weber --attractor-mode static

Synthetic benchmarks with the default preset (dynamic)::

    python scripts/train_unified.py --preset default --synthetic

Synthetic benchmarks with weber preset in static mode::

    python scripts/train_unified.py --preset weber --attractor-mode static --synthetic

Related commands
----------------
::

    # Phase-transition parameter scan
    python scripts/run_phase_scan.py --param1 kappa

    # Visualization suite for dynamic attractor diagnostics
    python scripts/run_viz_suite.py --groups dynamic --preset default

    # Multi-preset comparison on held-out data
    python scripts/eval_core.py --presets default weber --mode dynamic static
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.common.config import load_config
from src.common.device import get_device
from src.common.logging import get_logger
from src.common.seed import set_seed
from src.training.engine import (
    Trainer,
    _make_eval_loader,
    build_modules,
    build_task_heads,
    load_task_data,
    make_parser,
    resolve_config,
)

log = get_logger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================


def make_unified_parser() -> argparse.ArgumentParser:
    parser = make_parser()
    parser.description = (
        "Unified training: empirical (real data) or synthetic benchmarks."
    )
    parser.add_argument(
        "--attractor-mode",
        default=None,
        choices=["static", "dynamic"],
        dest="attractor_mode",
        help="Override attractor_mode from config (static | dynamic).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Run synthetic benchmark suite (Tasks A/B/C) instead of empirical training.",
    )
    parser.add_argument(
        "--rt-mode",
        default=None,
        dest="rt_mode",
        choices=["steps", "steps_plus_margin"],
        help="RT proxy mode for synthetic mode (default: config rt_proxy.mode).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        dest="n_samples",
        help="Samples per task in synthetic mode (default: config synthetic.n_samples_test).",
    )
    return parser


# ======================================================================
# Synthetic benchmark mode
# ======================================================================


@torch.no_grad()
def _eval_task_a(
    data: dict,
    model,
    rt_proxy,
    attractor_mode: str,
    n_min: int,
    n_max: int,
    device: torch.device,
) -> dict[str, float]:
    n_true = data["n_true"].to(device)
    y = data["y_noisy"].to(device)

    out = model(y)
    n_hat = out["n_hat"].long()
    n_steps = out["n_steps"]
    basin_margin = out["basin_margin"]

    rt = rt_proxy(n_steps, basin_margin=basin_margin)
    metrics: dict[str, float] = {
        "accuracy": (n_hat == n_true).float().mean().item(),
        "mae": (n_hat.float() - n_true.float()).abs().mean().item(),
        "mean_rt": rt.mean().item(),
        "mean_steps": n_steps.float().mean().item(),
    }
    for n in range(n_min, n_max + 1):
        mask = n_true == n
        if mask.sum() == 0:
            continue
        metrics[f"acc_n{n}"] = (n_hat[mask] == n).float().mean().item()
        metrics[f"rt_n{n}"] = rt[mask].mean().item()
    return metrics


@torch.no_grad()
def _eval_task_b(
    data: dict,
    model,
    attractor_mode: str,
    device: torch.device,
) -> dict[str, float]:
    n_left = data["n_left"].to(device)
    n_right = data["n_right"].to(device)
    y_left = data["y_left"].to(device)
    y_right = data["y_right"].to(device)
    label = data["label"].to(device)

    unequal = n_left != n_right
    if unequal.sum() == 0:
        return {"accuracy": float("nan"), "n_pairs": 0}

    n_left = n_left[unequal]
    n_right = n_right[unequal]
    y_left = y_left[unequal]
    y_right = y_right[unequal]
    label = label[unequal]

    out_l = model(y_left)
    out_r = model(y_right)
    pred = (out_r["expected_n_clean"] > out_l["expected_n_clean"]).long()

    dist = (n_right - n_left).abs()
    metrics: dict[str, float] = {
        "accuracy": (pred == label).float().mean().item(),
        "n_pairs": int(unequal.sum()),
    }
    max_dist = int((n_right.max() - n_left.min()).abs().item())
    for d in range(1, max_dist + 1):
        mask = dist == d
        if mask.sum() == 0:
            continue
        metrics[f"acc_dist{d}"] = (pred[mask] == label[mask]).float().mean().item()
    return metrics


@torch.no_grad()
def _eval_task_c(
    data: dict,
    model,
    attractor_mode: str,
    n_min: int,
    n_max: int,
    device: torch.device,
) -> dict[str, float]:
    n_true = data["n_true"].to(device)
    target = data["target"].to(device)
    y = data["y_noisy"].to(device)
    label = data["label"].to(device)

    out = model(y)
    target_idx = (target - n_min).clamp(0, model.integers.shape[0] - 1)
    p_same = out["probs_clean"][
        torch.arange(len(target), device=device), target_idx
    ]
    pred = (p_same > 0.5).long()

    tp = ((pred == 1) & (label == 1)).sum().item()
    fp = ((pred == 1) & (label == 0)).sum().item()
    fn = ((pred == 0) & (label == 1)).sum().item()
    return {
        "accuracy": (pred == label).float().mean().item(),
        "precision": tp / (tp + fp + 1e-9),
        "recall": tp / (tp + fn + 1e-9),
    }


def run_synthetic_benchmarks(args: argparse.Namespace) -> None:
    """Run Tasks A / B / C on synthetic data, driven by config + CLI overrides."""
    from src.data.synth_benchmarks import (
        NeutralEvidenceChannel,
        generate_task_a,
        generate_task_b,
        generate_task_c,
    )
    from src.models.attractor_dynamics import DynamicAttractorCore
    from src.models.rt_proxy import RTProxy

    cfg = load_config(args.config)

    attractor_mode = args.attractor_mode or cfg.attractor_mode
    n_samples = args.n_samples or cfg.synthetic.n_samples_test
    rt_mode = args.rt_mode or cfg.rt_proxy.mode
    seed = args.seed if args.seed is not None else cfg.synthetic.seed
    device = torch.device("cpu")

    set_seed(seed)

    cleanup_mode = "gradient" if attractor_mode == "dynamic" else "none"

    ev = cfg.synthetic.evidence
    channel = NeutralEvidenceChannel(
        rho0=ev.rho0, nu0=ev.nu0, s00=ev.s00, s0=ev.s0, lambda0=ev.lambda0,
    )

    dyn_cfg = cfg.dynamic_attractor
    model = DynamicAttractorCore(
        support_min=dyn_cfg.support_min,
        support_max=dyn_cfg.support_max,
        rho_init=dyn_cfg.rho_init,
        lambda_init=dyn_cfg.lambda_init,
        sigma0_init=dyn_cfg.sigma0_init,
        nu_init=dyn_cfg.nu_init,
        tau_in_init=dyn_cfg.tau_in_init,
        kappa_init=dyn_cfg.kappa_init,
        eta_init=dyn_cfg.eta_init,
        cleanup_mode=cleanup_mode,
        max_steps=dyn_cfg.max_steps,
        tol=dyn_cfg.tol,
    ).to(device).eval()

    rt_proxy = RTProxy(
        mode=rt_mode,
        t0_init=cfg.rt_proxy.t0_init,
        c_steps_init=cfg.rt_proxy.c_steps_init,
        c_margin_init=cfg.rt_proxy.c_margin_init,
    ).to(device)

    n_min = dyn_cfg.support_min
    n_max = dyn_cfg.support_max

    sep = "─" * 60
    print(sep)
    print(
        f"  attractor_mode={attractor_mode!r}  "
        f"N=[{n_min},{n_max}]  n_samples={n_samples}"
    )
    print(sep)

    data_a = generate_task_a(n_min, n_max, n_samples, seed=seed, channel=channel)
    m_a = _eval_task_a(data_a, model, rt_proxy, attractor_mode, n_min, n_max, device)
    print("\nTask A — Absolute estimation")
    print(f"  Accuracy  : {m_a['accuracy']:.4f}")
    print(f"  MAE       : {m_a['mae']:.4f}")
    print(f"  Mean RT   : {m_a['mean_rt']:.4f}")
    print(f"  Mean steps: {m_a['mean_steps']:.2f}")
    print()
    print(f"  {'N':>4}  {'Accuracy':>10}  {'RT':>10}")
    for n in range(n_min, n_max + 1):
        print(
            f"  {n:>4}"
            f"  {m_a.get(f'acc_n{n}', float('nan')):>10.4f}"
            f"  {m_a.get(f'rt_n{n}', float('nan')):>10.4f}"
        )

    data_b = generate_task_b(n_min, n_max, n_samples, seed=seed + 1, channel=channel)
    m_b = _eval_task_b(data_b, model, attractor_mode, device)
    print(f"\nTask B — Pairwise comparison  (unequal pairs: {m_b.get('n_pairs', '?')})")
    print(f"  Accuracy  : {m_b['accuracy']:.4f}")
    dist_keys = sorted(
        (k for k in m_b if k.startswith("acc_dist")),
        key=lambda k: int(k.replace("acc_dist", "")),
    )
    if dist_keys:
        print()
        print(f"  {'dist':>6}  {'Accuracy':>10}")
        for k in dist_keys:
            print(f"  {k.replace('acc_dist', ''):>6}  {m_b[k]:>10.4f}")

    data_c = generate_task_c(
        n_min, n_max, n_samples, seed=seed + 2, channel=channel,
    )
    m_c = _eval_task_c(data_c, model, attractor_mode, n_min, n_max, device)
    print("\nTask C — Same/different verification")
    print(f"  Accuracy  : {m_c['accuracy']:.4f}")
    print(f"  Precision : {m_c['precision']:.4f}")
    print(f"  Recall    : {m_c['recall']:.4f}")
    print()
    print(sep)


# ======================================================================
# Empirical training mode
# ======================================================================


def run_empirical(args: argparse.Namespace) -> dict[str, float]:
    """Empirical multitask training with optional attractor-mode override."""
    from src.data.dataset_builders import build_dataloader

    cfg = resolve_config(args)

    if args.attractor_mode is not None and args.attractor_mode != cfg.attractor_mode:
        log.warning(
            "CLI --attractor-mode=%r overrides config attractor_mode=%r",
            args.attractor_mode,
            cfg.attractor_mode,
        )
        cfg.attractor_mode = args.attractor_mode

    tcfg = cfg.training
    set_seed(tcfg.seed)
    device = get_device()

    log.info("Training %s (%s) — attractor_mode=%s", cfg.preset, cfg.family, cfg.attractor_mode)
    log.info("Device: %s", device)
    log.info("Tasks: %s", tcfg.tasks)
    log.info("Data: %s", tcfg.data_dir)

    train_ds, val_ds, test_ds, subject_map = load_task_data(
        tcfg.data_dir, tcfg.tasks, seed=tcfg.seed,
    )
    cfg.subject.n_subjects = subject_map.n_subjects
    available_tasks = list(train_ds.keys())

    log.info("Subjects: %d", subject_map.n_subjects)
    log.info("Available tasks: %s", available_tasks)
    for t, ds in train_ds.items():
        log.info("  %s: %d train trials", t, len(ds))

    use_cuda = device.type == "cuda"
    train_loader = build_dataloader(
        train_ds,
        batch_size=tcfg.batch_size,
        seed=tcfg.seed,
        task_weights=tcfg.task_weights,
        num_workers=tcfg.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = _make_eval_loader(
        val_ds, tcfg.batch_size,
        num_workers=tcfg.num_workers, pin_memory=use_cuda,
    )

    modules = build_modules(cfg, subject_map.n_subjects, device)
    heads = build_task_heads(cfg, subject_map.n_subjects, available_tasks, device)

    trainer = Trainer(
        cfg, modules, heads, train_loader, val_loader, device,
        use_wandb=tcfg.enable_wandb,
    )

    if tcfg.resume:
        log.info("Resuming from %s", tcfg.resume)
        trainer.load_checkpoint(tcfg.resume)

    best_metrics = trainer.fit()

    if test_ds:
        test_loader = _make_eval_loader(
            test_ds, tcfg.batch_size,
            num_workers=tcfg.num_workers, pin_memory=use_cuda,
        )
        test_metrics = trainer.evaluate(test_loader, prefix="test")
        log.info("Test set metrics:")
        for k, v in sorted(test_metrics.items()):
            log.info("  %s: %.4f", k, v)

    return best_metrics


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    parser = make_unified_parser()
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_benchmarks(args)
    else:
        run_empirical(args)


if __name__ == "__main__":
    main()
