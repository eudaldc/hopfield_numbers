"""Synthetic benchmark evaluation for attractor-first cleanup dynamics.

Runs Tasks A (estimation), B (comparison), and C (verification) using the
neutral evidence channel and attractor cleanup, then prints per-task metrics
and an RT summary broken down by numerosity.

Usage examples
--------------
    # Dynamic cleanup (default)
    python scripts/run_attractor_synth.py

    # Static readout (ablation, no cleanup)
    python scripts/run_attractor_synth.py --mode static
"""

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.synth_benchmarks import (
    NeutralEvidenceChannel,
    generate_task_a,
    generate_task_b,
    generate_task_c,
)
from src.models.attractor_dynamics import DynamicAttractorCore
from src.models.rt_proxy import RTProxy


# ── argument parsing ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic numerosity benchmarks with dynamic attractor cleanup"
    )
    p.add_argument(
        "--mode", default="dynamic",
        choices=["static", "dynamic"],
        help=(
            "Evaluation mode: "
            "'static' uses attractor posterior without cleanup; "
            "'dynamic' runs iterative gradient cleanup (default)"
        ),
    )
    p.add_argument("--n-samples", type=int, default=10_000,
                   help="Number of test samples per task (default: 10000)")
    p.add_argument("--n-min", type=int, default=1,
                   help="Minimum true numerosity (default: 1)")
    p.add_argument("--n-max", type=int, default=9,
                   help="Maximum true numerosity (default: 9)")
    p.add_argument("--seed", type=int, default=7,
                   help="Random seed (default: 7)")
    p.add_argument("--rt-mode", default="steps_plus_margin",
                   choices=["steps", "steps_plus_margin"],
                   help="RT proxy mode (default: steps_plus_margin)")
    p.add_argument("--device", default="cpu",
                   help="Torch device (default: cpu)")
    return p.parse_args()


# ── per-task evaluators ────────────────────────────────────────────────────


@torch.no_grad()
def eval_task_a(
    data: dict,
    model: DynamicAttractorCore,
    rt_proxy: RTProxy,
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
    acc = (n_hat == n_true).float().mean().item()
    mae = (n_hat.float() - n_true.float()).abs().mean().item()

    metrics: dict[str, float] = {
        "accuracy": acc,
        "mae": mae,
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
def eval_task_b(
    data: dict,
    model: DynamicAttractorCore,
    rt_proxy: RTProxy,
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

    n_left, n_right = n_left[unequal], n_right[unequal]
    y_left, y_right = y_left[unequal], y_right[unequal]
    label = label[unequal]

    out_l = model(y_left)
    out_r = model(y_right)
    pred = (out_r["expected_n_clean"] > out_l["expected_n_clean"]).long()

    rt_l = rt_proxy(out_l["n_steps"], basin_margin=out_l["basin_margin"])
    rt_r = rt_proxy(out_r["n_steps"], basin_margin=out_r["basin_margin"])
    rt_pair = 0.5 * (rt_l + rt_r)

    acc = (pred == label).float().mean().item()
    dist = (n_right - n_left).abs()
    ratio = n_right.float().clamp(min=1) / n_left.float().clamp(min=1)

    metrics: dict[str, float] = {
        "accuracy": acc,
        "n_pairs": int(unequal.sum()),
        "mean_rt": rt_pair.mean().item(),
    }
    for d in range(1, int(dist.max().item()) + 1):
        mask = dist == d
        if mask.sum() == 0:
            continue
        metrics[f"acc_dist{d}"] = (pred[mask] == label[mask]).float().mean().item()
        metrics[f"rt_dist{d}"] = rt_pair[mask].mean().item()

    for r_bin_lo, r_bin_hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10.0)]:
        mask = (ratio >= r_bin_lo) & (ratio < r_bin_hi)
        if mask.sum() == 0:
            continue
        tag = f"ratio_{r_bin_lo:.1f}_{r_bin_hi:.1f}"
        metrics[f"acc_{tag}"] = (pred[mask] == label[mask]).float().mean().item()

    return metrics


@torch.no_grad()
def eval_task_c(
    data: dict,
    model: DynamicAttractorCore,
    rt_proxy: RTProxy,
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
    probs = out["probs_clean"]
    p_same = probs[torch.arange(len(target), device=device), target_idx]
    pred = (p_same > 0.5).long()

    rt = rt_proxy(out["n_steps"], basin_margin=out["basin_margin"])

    acc = (pred == label).float().mean().item()
    tp = ((pred == 1) & (label == 1)).sum().item()
    fp = ((pred == 1) & (label == 0)).sum().item()
    fn = ((pred == 0) & (label == 1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "mean_rt": rt.mean().item(),
    }


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    channel = NeutralEvidenceChannel()

    cleanup_mode = "gradient" if args.mode == "dynamic" else "none"
    model = DynamicAttractorCore(
        support_min=args.n_min,
        support_max=args.n_max,
        cleanup_mode=cleanup_mode,
    ).to(device).eval()

    rt_proxy = RTProxy(mode=args.rt_mode).to(device)

    sep = "─" * 60

    print(sep)
    print(
        f"  mode={args.mode!r}  "
        f"N=[{args.n_min},{args.n_max}]  n_samples={args.n_samples}"
    )
    print(sep)

    # ── Task A ──────────────────────────────────────────────────────
    data_a = generate_task_a(
        args.n_min, args.n_max, args.n_samples,
        seed=args.seed, channel=channel,
    )
    m_a = eval_task_a(data_a, model, rt_proxy, args.n_min, args.n_max, device)

    print("\nTask A — Absolute estimation")
    print(f"  Accuracy : {m_a['accuracy']:.4f}")
    print(f"  MAE      : {m_a['mae']:.4f}")
    print(f"  Mean RT  : {m_a['mean_rt']:.4f}")
    print(f"  Mean steps: {m_a['mean_steps']:.2f}")
    print()
    print(f"  {'N':>4}  {'Accuracy':>10}  {'RT':>10}")
    for n in range(args.n_min, args.n_max + 1):
        acc_n = m_a.get(f"acc_n{n}", float("nan"))
        rt_n = m_a.get(f"rt_n{n}", float("nan"))
        print(f"  {n:>4}  {acc_n:>10.4f}  {rt_n:>10.4f}")

    # ── Task B ──────────────────────────────────────────────────────
    data_b = generate_task_b(
        args.n_min, args.n_max, args.n_samples,
        seed=args.seed + 1, channel=channel,
    )
    m_b = eval_task_b(data_b, model, rt_proxy, device)

    print(f"\nTask B — Pairwise comparison  (unequal pairs: {m_b.get('n_pairs', '?')})")
    print(f"  Accuracy  : {m_b['accuracy']:.4f}")
    print(f"  Mean RT   : {m_b.get('mean_rt', float('nan')):.4f}")
    print()
    dist_keys = sorted(
        (k for k in m_b if k.startswith("acc_dist")),
        key=lambda k: int(k.replace("acc_dist", "")),
    )
    if dist_keys:
        print(f"  {'dist':>6}  {'Accuracy':>10}  {'RT':>10}")
        for k in dist_keys:
            d = k.replace("acc_dist", "")
            rt_k = m_b.get(f"rt_dist{d}", float("nan"))
            print(f"  {d:>6}  {m_b[k]:>10.4f}  {rt_k:>10.4f}")

    ratio_keys = sorted(k for k in m_b if k.startswith("acc_ratio"))
    if ratio_keys:
        print()
        print(f"  {'ratio':>12}  {'Accuracy':>10}")
        for k in ratio_keys:
            tag = k.replace("acc_", "")
            print(f"  {tag:>12}  {m_b[k]:>10.4f}")

    # ── Task C ──────────────────────────────────────────────────────
    data_c = generate_task_c(
        args.n_min, args.n_max, args.n_samples,
        seed=args.seed + 2, channel=channel,
    )
    m_c = eval_task_c(data_c, model, rt_proxy, args.n_min, args.n_max, device)

    print(f"\nTask C — Same/different verification")
    print(f"  Accuracy  : {m_c['accuracy']:.4f}")
    print(f"  Precision : {m_c['precision']:.4f}")
    print(f"  Recall    : {m_c['recall']:.4f}")
    print(f"  Mean RT   : {m_c.get('mean_rt', float('nan')):.4f}")
    print()
    print(sep)


if __name__ == "__main__":
    main()
