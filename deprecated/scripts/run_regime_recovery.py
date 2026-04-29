#!/usr/bin/env python
"""Effective-model regime recovery analysis.

Tests whether the attractor model generates behavior such that:
  - an accumulator-like model fits well in the small-N regime
  - an ANS-like model fits well in the large-N regime
  - the attractor model fits the whole range better than either

Four experiments:
  A — Fit accumulator-effective on N=1..4, evaluate on 1..9
  B — Fit ANS-effective on N=5..9, evaluate on 1..9
  C — Fit all three on full N=1..9
  D — Cross-regime failure analysis

Usage
-----
    python scripts/run_regime_recovery.py
    python scripts/run_regime_recovery.py --n-samples 20000 --output-dir output/recovery
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.synth_benchmarks import NeutralEvidenceChannel, generate_task_a
from src.models.attractor_dynamics import DynamicAttractorCore
from src.models.rt_proxy import RTProxy
from src.models.effective_baselines import AccumulatorEffective, ANSEffective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime recovery analysis: attractor vs effective baselines"
    )
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=9)
    p.add_argument("--small-n-max", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default="output/recovery")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


@torch.no_grad()
def generate_attractor_data(args):
    """Generate synthetic data from the attractor model."""
    channel = NeutralEvidenceChannel()
    model = DynamicAttractorCore(
        support_min=args.n_min, support_max=args.n_max,
        cleanup_mode="gradient",
    ).to(args.device).eval()
    rt_proxy = RTProxy(mode="steps_plus_margin").to(args.device)

    data = generate_task_a(
        args.n_min, args.n_max, args.n_samples, seed=args.seed, channel=channel,
    )
    n_true = data["n_true"].to(args.device)
    y = data["y_noisy"].to(args.device)
    out = model(y)

    return {
        "n_true": n_true.cpu().numpy(),
        "y_noisy": y.cpu().numpy(),
        "n_hat": out["n_hat"].cpu().numpy().astype(int),
        "x_star": out["x_star"].cpu().numpy(),
        "n_steps": out["n_steps"].cpu().numpy(),
        "basin_margin": out["basin_margin"].cpu().numpy(),
        "probs_clean": out["probs_clean"].cpu().numpy(),
    }


def eval_attractor_on_range(data, n_range):
    """Evaluate the attractor model on a specific N range."""
    mask = np.isin(data["n_true"], n_range)
    n_true = data["n_true"][mask]
    n_hat = data["n_hat"][mask]
    acc = float((n_hat == n_true).mean())
    mae = float(np.abs(n_hat - n_true).mean())
    return {"accuracy": acc, "mae": mae, "n_samples": int(mask.sum())}


def run_experiment_a(data, args):
    """Fit accumulator-effective on small N, evaluate on all."""
    small_range = list(range(args.n_min, args.small_n_max + 1))
    full_range = list(range(args.n_min, args.n_max + 1))

    mask_small = np.isin(data["n_true"], small_range)
    n_small = data["n_true"][mask_small]
    nhat_small = data["n_hat"][mask_small]

    acc_model = AccumulatorEffective()
    acc_model.fit(n_small, nhat_small.astype(float))

    results = {"model": "accumulator_effective", "fit_range": f"N={small_range[0]}..{small_range[-1]}"}
    for label, n_range in [("small", small_range), ("full", full_range)]:
        mask = np.isin(data["n_true"], n_range)
        n_r = data["n_true"][mask]
        nll = -acc_model.log_likelihood(n_r, data["n_hat"][mask].astype(float)) / mask.sum()
        results[f"nll_{label}"] = nll
        results[f"mae_{label}"] = float(np.abs(data["n_hat"][mask] - n_r).mean())

    return results, acc_model


def run_experiment_b(data, args):
    """Fit ANS-effective on large N, evaluate on all."""
    large_range = list(range(args.small_n_max + 1, args.n_max + 1))
    full_range = list(range(args.n_min, args.n_max + 1))

    mask_large = np.isin(data["n_true"], large_range)
    n_large = data["n_true"][mask_large]
    y_large = data["y_noisy"][mask_large]

    ans_model = ANSEffective()
    ans_model.fit(n_large, y_large)

    results = {"model": "ans_effective", "fit_range": f"N={large_range[0]}..{large_range[-1]}"}
    for label, n_range in [("large", large_range), ("full", full_range)]:
        mask = np.isin(data["n_true"], n_range)
        n_r = data["n_true"][mask]
        nll = -ans_model.log_likelihood(n_r, data["y_noisy"][mask]) / mask.sum()
        results[f"nll_{label}"] = nll

    return results, ans_model


def run_experiment_c(data, args):
    """Fit all three on full range, compare."""
    full_range = list(range(args.n_min, args.n_max + 1))
    mask = np.isin(data["n_true"], full_range)
    n_true = data["n_true"][mask]

    acc_model = AccumulatorEffective().fit(n_true, data["n_hat"][mask].astype(float))
    ans_model = ANSEffective().fit(n_true, data["y_noisy"][mask])

    attractor_acc = float((data["n_hat"][mask] == n_true).mean())
    attractor_mae = float(np.abs(data["n_hat"][mask] - n_true).mean())

    nll_acc = -acc_model.log_likelihood(n_true, data["n_hat"][mask].astype(float)) / mask.sum()
    nll_ans = -ans_model.log_likelihood(n_true, data["y_noisy"][mask]) / mask.sum()

    return {
        "attractor_accuracy": attractor_acc,
        "attractor_mae": attractor_mae,
        "accumulator_nll_full": nll_acc,
        "accumulator_params": {"sigma_count": acc_model.sigma_count, "sigma_floor": acc_model.sigma_floor},
        "ans_nll_full": nll_ans,
        "ans_params": {"nu": ans_model.nu, "rho": ans_model.rho, "sigma": ans_model.sigma, "lam": ans_model.lam},
    }


def run_experiment_d(data, args):
    """Cross-regime failure: show degradation outside home range."""
    small_range = list(range(args.n_min, args.small_n_max + 1))
    large_range = list(range(args.small_n_max + 1, args.n_max + 1))

    mask_small = np.isin(data["n_true"], small_range)
    mask_large = np.isin(data["n_true"], large_range)

    acc_model = AccumulatorEffective().fit(
        data["n_true"][mask_small], data["n_hat"][mask_small].astype(float)
    )
    ans_model = ANSEffective().fit(
        data["n_true"][mask_large], data["y_noisy"][mask_large]
    )

    per_n: dict[int, dict] = {}
    for n in range(args.n_min, args.n_max + 1):
        mask_n = data["n_true"] == n
        if mask_n.sum() == 0:
            continue
        n_true_n = data["n_true"][mask_n]
        nhat_n = data["n_hat"][mask_n].astype(float)
        y_n = data["y_noisy"][mask_n]

        per_n[n] = {
            "attractor_acc": float((data["n_hat"][mask_n] == n).mean()),
            "attractor_mae": float(np.abs(data["n_hat"][mask_n] - n).mean()),
            "acc_nll": -acc_model.log_likelihood(n_true_n, nhat_n) / mask_n.sum(),
            "ans_nll": -ans_model.log_likelihood(n_true_n, y_n) / mask_n.sum(),
        }

    return per_n


def main():
    args = parse_args()
    args.device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "─" * 60
    print(sep)
    print("  Regime Recovery Analysis — Attractor-First")
    print(f"  N=[{args.n_min},{args.n_max}]  n_samples={args.n_samples}")
    print(sep)

    print("\nGenerating attractor data ...")
    data = generate_attractor_data(args)

    # Experiment A
    print("\n--- Experiment A: Accumulator-effective fit on small N ---")
    res_a, _ = run_experiment_a(data, args)
    for k, v in res_a.items():
        print(f"  {k}: {v}")

    # Experiment B
    print("\n--- Experiment B: ANS-effective fit on large N ---")
    res_b, _ = run_experiment_b(data, args)
    for k, v in res_b.items():
        print(f"  {k}: {v}")

    # Experiment C
    print("\n--- Experiment C: Full-range comparison ---")
    res_c = run_experiment_c(data, args)
    for k, v in res_c.items():
        print(f"  {k}: {v}")

    # Experiment D
    print("\n--- Experiment D: Cross-regime failure (per-N) ---")
    per_n = run_experiment_d(data, args)
    print(f"  {'N':>4}  {'Attr Acc':>10}  {'Attr MAE':>10}  {'Acc NLL':>10}  {'ANS NLL':>10}")
    for n in sorted(per_n.keys()):
        d = per_n[n]
        print(
            f"  {n:>4}  {d['attractor_acc']:>10.4f}  {d['attractor_mae']:>10.4f}"
            f"  {d['acc_nll']:>10.4f}  {d['ans_nll']:>10.4f}"
        )

    import pandas as pd
    rows = []
    for n, d in sorted(per_n.items()):
        rows.append({"N": n, **d})
    df = pd.DataFrame(rows)
    csv_path = out_dir / "regime_recovery_per_n.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-N results → {csv_path}")

    print()
    print(sep)


if __name__ == "__main__":
    main()
