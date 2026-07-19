"""Sweep over ALL Piantadosi duration conditions via wandb.

Each exposure duration (40, 80, 160, 320, 640 ms) becomes a separate
wandb run inside a grid sweep.  Per-duration plots are logged to each
run; the cross-duration summary panel is saved locally, then
``scripts.fig2_piantadosi_fit`` is run to refresh the multi-panel figure.

Usage:
    python scripts/fit_piantadosi_durations.py
    BOOTSTRAP_N=50 BOOTSTRAP_JOBS=8 python scripts/fit_piantadosi_durations.py

Env variables:
    BOOTSTRAP_N      — bootstrap refits per duration (default 0 = skip).
    BOOTSTRAP_JOBS   — parallel workers (default 1 = serial; -1 = all cores).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    extract_empirical,
    simulate,
    compute_goodness,
    hessian_ci,
    parallel_bootstrap_ci,
    refine_cis_with_profile,
    OUTPUT_DIR,
    PARAM_NAMES,
    BOUNDS,
    N_TRUE_MIN,
    N_TRUE_MAX,
    N_SUPPORT_MAX,
    RT_MODE,
    COMPRESS,
    WELL_TYPE,
    FIXED_BETA,
    FIXED_KAPPA,
    FIXED_ETA,
    FIXED,
    VARIANCE_FLOOR,
    CONSTANT_NOISE,
)
from scripts.fit_piantadosi import (
    _boot_worker_piant,
    _make_fit_with_fixed,
    _make_nll_fn,
    _make_nll_of_theta,
    fit_parameters,
)
from scripts.plots import (
    plot_comparison, plot_energy_landscape, plot_number_line, plot_summary,
)

DURATIONS = [40, 80, 160, 320, 640]

_sweep_results: dict[int, tuple[dict, pd.DataFrame, pd.DataFrame]] = {}

_shared_config = {
    "rt_mode": RT_MODE,
    "compress_numberline": COMPRESS,
    "fixed_beta": FIXED_BETA,
    "fixed_kappa": FIXED_KAPPA,
    "fixed_eta": FIXED_ETA,
    "variance_floor": VARIANCE_FLOOR,
    "constant_noise": CONSTANT_NOISE,
    "N_TRUE_MIN": N_TRUE_MIN,
    "N_TRUE_MAX": N_TRUE_MAX,
    "N_SUPPORT_MAX": N_SUPPORT_MAX,
    **FIXED,
}


def sweep_run() -> None:
    run = wandb.init()
    duration_ms = wandb.config.duration_ms

    print(f"\n{'=' * 60}")
    print(f"  Duration: {duration_ms} ms")
    print(f"{'=' * 60}")

    empirical = extract_empirical(duration_ms)
    print(f"  {len(empirical)} numerosities, {int(empirical['count'].sum())} trials")

    theta, opt_result = fit_parameters(empirical)
    model = simulate(theta, n_samples_per_n=5000, seed=73)

    tag = "compressed" if COMPRESS else WELL_TYPE
    path_comparison = plot_comparison(
        empirical, model, theta, OUTPUT_DIR, tag=tag, duration_ms=duration_ms,
    )
    path_energy = plot_energy_landscape(theta, OUTPUT_DIR, duration_ms=duration_ms)

    images = {
        "comparison_plot": wandb.Image(str(path_comparison)),
        "energy_landscape": wandb.Image(str(path_energy)),
    }

    if COMPRESS:
        path_line = plot_number_line(theta, OUTPUT_DIR, duration_ms=duration_ms)
        images["number_line"] = wandb.Image(str(path_line))

    k = len(PARAM_NAMES)
    goodness = compute_goodness(empirical, model, k=k)

    nll_fn = _make_nll_fn(empirical, seed=73)
    theta_vec_fit = np.array([theta[p] for p in PARAM_NAMES])
    ci_df = hessian_ci(
        nll_fn,
        theta_vec=theta_vec_fit,
        param_names=PARAM_NAMES,
        bounds=BOUNDS,
    )
    needs = ci_df.apply(
        lambda r: (not np.isfinite(r["se"]))
        or r["status"] in ("non_pd", "singular", "at_bound"),
        axis=1,
    )
    if needs.any():
        print(
            f"  Refining {int(needs.sum())} param(s) via profile likelihood "
            f"(duration {duration_ms} ms)"
        )
        ci_df = refine_cis_with_profile(
            ci_df,
            nll_of_theta=_make_nll_of_theta(empirical),
            fit_with_fixed=_make_fit_with_fixed(empirical),
            theta_hat=theta,
            nll_hat=float(nll_fn(theta_vec_fit)),
            bounds=BOUNDS,
            verbose=True,
        )
    ci_csv = OUTPUT_DIR / f"fit_{duration_ms}ms_ci_hessian.csv"
    ci_df.to_csv(ci_csv, index=False)

    boot_df: pd.DataFrame | None = None
    n_boot = int(os.environ.get("BOOTSTRAP_N", "0"))
    n_jobs = int(os.environ.get("BOOTSTRAP_JOBS", "1"))
    if n_boot > 0:
        print(
            f"  Bootstrap: {n_boot} refits (n_jobs={n_jobs}) for "
            f"duration {duration_ms} ms ..."
        )
        rng = np.random.default_rng(17 + duration_ms)
        seeds = rng.integers(0, 2**31 - 1, size=n_boot).tolist()
        payloads = [
            (duration_ms, int(s), "data/data_piantadosi.csv") for s in seeds
        ]
        try:
            boot_df = parallel_bootstrap_ci(
                worker_fn=_boot_worker_piant,
                payloads=payloads,
                param_names=PARAM_NAMES,
                n_jobs=n_jobs,
                desc=f"boot {duration_ms}ms",
            )
            boot_csv = OUTPUT_DIR / f"fit_{duration_ms}ms_ci_bootstrap.csv"
            boot_df.to_csv(boot_csv, index=False)
            print(f"  Saved {boot_csv}")
        except Exception as exc:
            print(f"  Bootstrap failed: {exc}")
            boot_df = None

    summary = {
        **{f"param/{k_}": v for k_, v in theta.items()},
        **{f"goodness/{k_}": v for k_, v in goodness.items()},
        "loss": opt_result.fun,
        "n_func_evals": opt_result.nfev,
        "optimizer_success": opt_result.success,
    }
    for _, row in ci_df.iterrows():
        p = row["param"]
        summary[f"ci/{p}/se"] = float(row["se"])
        summary[f"ci/{p}/lo"] = float(row["ci_lo"])
        summary[f"ci/{p}/hi"] = float(row["ci_hi"])
        summary[f"ci/{p}/status"] = row["status"]
        if "method" in ci_df.columns:
            summary[f"ci/{p}/method"] = row["method"]
    if boot_df is not None:
        for _, row in boot_df.iterrows():
            p = row["param"]
            summary[f"ci_bootstrap/{p}/lo"] = float(row["ci_lo"])
            summary[f"ci_bootstrap/{p}/hi"] = float(row["ci_hi"])
            summary[f"ci_bootstrap/{p}/median"] = float(row["median"])
    run.summary.update(summary)
    run.log(images)

    _sweep_results[duration_ms] = (theta, model, empirical)
    wandb.finish()


if __name__ == "__main__":
    fixed_params = {
        k: {"value": v}
        for k, v in _shared_config.items()
        if v is not None
    }
    sweep_config = {
        "method": "grid",
        "name": "piantadosi-duration-sweep",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "duration_ms": {"values": DURATIONS},
            **fixed_params,
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="hopfield-piantadosi")
    wandb.agent(sweep_id, function=sweep_run, count=len(DURATIONS))

    if _sweep_results:
        plot_summary(_sweep_results, OUTPUT_DIR)
        subprocess.run(
            [sys.executable, "-m", "scripts.fig2_piantadosi_fit"],
            cwd=_ROOT,
            check=True,
        )
