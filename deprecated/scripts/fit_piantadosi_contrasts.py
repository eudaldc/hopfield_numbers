"""Sweep over ALL Piantadosi contrast (opacity) conditions via wandb.

Each opacity level (0.04, 0.08, 0.16, 0.32, 0.64) becomes a separate
wandb run inside a grid sweep.  Per-opacity plots are logged to each
run; the cross-opacity summary panel is saved locally afterwards.

Usage:
    python scripts/fit_piantadosi_contrasts.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    extract_empirical_contrast,
    simulate,
    compute_goodness,
    hessian_ci,
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
    _make_fit_with_fixed,
    _make_nll_fn,
    _make_nll_of_theta,
    fit_parameters,
)
from scripts.plots import (
    plot_comparison, plot_energy_landscape, plot_number_line,
    plot_summary_contrast,
)

OPACITIES = [0.04, 0.08, 0.16, 0.32, 0.64]

_sweep_results: dict[float, tuple[dict, pd.DataFrame, pd.DataFrame]] = {}

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
    opacity = wandb.config.opacity

    print(f"\n{'=' * 60}")
    print(f"  Opacity: {opacity}")
    print(f"{'=' * 60}")

    empirical = extract_empirical_contrast(opacity)
    print(f"  {len(empirical)} numerosities, {int(empirical['count'].sum())} trials")

    theta, opt_result = fit_parameters(empirical)
    model = simulate(theta, n_samples_per_n=5000, seed=73)

    tag = "compressed" if COMPRESS else WELL_TYPE
    path_comparison = plot_comparison(
        empirical, model, theta, OUTPUT_DIR,
        tag=f"{tag}_opacity{opacity}", duration_ms=int(opacity * 1000),
    )
    path_energy = plot_energy_landscape(
        theta, OUTPUT_DIR, duration_ms=int(opacity * 1000),
    )

    images = {
        "comparison_plot": wandb.Image(str(path_comparison)),
        "energy_landscape": wandb.Image(str(path_energy)),
    }

    if COMPRESS:
        path_line = plot_number_line(
            theta, OUTPUT_DIR, duration_ms=int(opacity * 1000),
        )
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
            f"(opacity {opacity})"
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
    ci_csv = OUTPUT_DIR / f"fit_opacity{opacity}_ci_hessian.csv"
    ci_df.to_csv(ci_csv, index=False)

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
    run.summary.update(summary)
    run.log(images)

    _sweep_results[opacity] = (theta, model, empirical)
    wandb.finish()


if __name__ == "__main__":
    fixed_params = {
        k: {"value": v}
        for k, v in _shared_config.items()
        if v is not None
    }
    sweep_config = {
        "method": "grid",
        "name": "piantadosi-contrast-sweep",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "opacity": {"values": OPACITIES},
            **fixed_params,
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="hopfield-piantadosi")
    wandb.agent(sweep_id, function=sweep_run, count=len(OPACITIES))

    if _sweep_results:
        plot_summary_contrast(_sweep_results, OUTPUT_DIR)
