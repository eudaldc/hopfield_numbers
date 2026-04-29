#!/usr/bin/env python
"""Phase-transition parameter scan for the dynamic attractor model.

Sweeps one or two parameters of DynamicAttractorCore over a grid,
evaluates metrics on synthetic Task-A data at every grid point, saves
a CSV, and generates diagnostic plots.

Usage examples
--------------
# 1-D scan over kappa with default range
python scripts/run_phase_scan.py --param1 kappa

# Custom range and resolution
python scripts/run_phase_scan.py --param1 tau_in --range1 0.1 5.0 --n-points 25

# 2-D scan kappa x tau_in
python scripts/run_phase_scan.py \\
    --param1 kappa --range1 0.1 5.0 \\
    --param2 tau_in --range2 0.1 3.0 \\
    --n-points 15

# Log to W&B
python scripts/run_phase_scan.py --param1 sigma0 --wandb
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.phase_scan import PhaseScanConfig, ScanAxis, run_phase_scan
from src.data.synth_benchmarks import NeutralEvidenceChannel
from src.models.rt_proxy import RTProxy
from src.viz import apply_style


_VALID_PARAMS = ["kappa", "tau_in", "sigma0", "lam", "eta", "rho"]

_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "kappa":  (0.1, 5.0),
    "tau_in": (0.1, 3.0),
    "sigma0": (0.1, 2.0),
    "lam":    (0.0, 1.0),
    "eta":    (0.05, 0.5),
    "rho":    (0.01, 1.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-transition scan over DynamicAttractorCore parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param1", default="kappa", choices=_VALID_PARAMS,
                   help="Primary scan axis parameter")
    p.add_argument("--range1", nargs=2, type=float, default=None,
                   metavar=("MIN", "MAX"),
                   help="Value range for param1. Uses a sensible default per parameter.")
    p.add_argument("--param2", default=None, choices=_VALID_PARAMS,
                   help="Second scan axis parameter (enables 2-D scan)")
    p.add_argument("--range2", nargs=2, type=float, default=None,
                   metavar=("MIN", "MAX"),
                   help="Value range for param2.")
    p.add_argument("--n-points", type=int, default=20,
                   help="Points per axis")
    p.add_argument("--n-samples", type=int, default=5_000,
                   help="Task-A samples per grid point")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=9)
    p.add_argument("--small-n-max", type=int, default=4,
                   help="Upper bound of small-number regime")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rt-mode", default="steps_plus_margin",
                   choices=["steps", "steps_plus_margin"])
    p.add_argument("--output", default=None,
                   help="CSV output path. Defaults to <plot-dir>/phase_scan_<params>.csv")
    p.add_argument("--plot-dir", default="output/phase_scan",
                   help="Directory for diagnostic plots")
    p.add_argument("--device", default="cpu")
    p.add_argument("--wandb", action="store_true", default=False)
    return p.parse_args()


def _maybe_init_wandb(use_wandb: bool, param_tag: str, n_points: int):
    if not use_wandb:
        return None
    try:
        import wandb
        return wandb.init(
            project="numeric-attractors-poc",
            job_type="phase_scan",
            group="phase_scan",
            config={"param": param_tag, "n_points": n_points},
            reinit=True,
        )
    except Exception:
        return None


def main() -> None:
    apply_style()
    args = parse_args()
    device = torch.device(args.device)

    channel = NeutralEvidenceChannel()
    rt_proxy = RTProxy(mode=args.rt_mode).to(device)

    r1 = args.range1 or list(_DEFAULT_RANGES[args.param1])
    vals1 = np.linspace(r1[0], r1[1], args.n_points).tolist()
    axes = [ScanAxis(param=args.param1, values=vals1)]

    if args.param2:
        r2 = args.range2 or list(_DEFAULT_RANGES[args.param2])
        vals2 = np.linspace(r2[0], r2[1], args.n_points).tolist()
        axes.append(ScanAxis(param=args.param2, values=vals2))

    cfg = PhaseScanConfig(
        axes=axes,
        n_samples=args.n_samples,
        n_min=args.n_min,
        n_max=args.n_max,
        small_n_max=args.small_n_max,
        seed=args.seed,
        rt_mode=args.rt_mode,
    )

    param_tag = args.param1 + (f"_x_{args.param2}" if args.param2 else "")
    n_total = args.n_points if not args.param2 else args.n_points ** 2
    print(
        f"Phase scan: {param_tag}  |  {n_total} grid points  |  "
        f"{args.n_samples} samples each"
    )

    df = run_phase_scan(cfg, rt_proxy=rt_proxy, device=device, channel=channel)

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.output or str(plot_dir / f"phase_scan_{param_tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved results → {out_csv}  ({len(df)} rows, {len(df.columns)} columns)")

    wandb_run = _maybe_init_wandb(args.wandb, param_tag, args.n_points)

    from src.viz.plots_attractor import plot_phase_1d, plot_phase_2d

    if args.param2:
        print("Generating 2-D phase heatmaps ...")
        figs = plot_phase_2d(
            df, args.param1, args.param2,
            save_dir=str(plot_dir), wandb_run=wandb_run,
            wandb_prefix="phase_scan",
        )
        for fig in figs.values():
            plt.close(fig)
        print(f"  Saved {len(figs)} heatmap(s) to {plot_dir}")
    else:
        print("Generating 1-D phase curves ...")
        fig = plot_phase_1d(
            df, args.param1,
            save_dir=str(plot_dir), wandb_run=wandb_run,
            wandb_prefix="phase_scan",
        )
        plt.close(fig)
        print(f"  Saved 1-D phase plot to {plot_dir}")

    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
