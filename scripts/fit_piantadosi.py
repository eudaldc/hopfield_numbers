"""Fit Piantadosi data with a simple Benford-attractor model.

Model
-----
  - Equispaced wells at integers:  mu_n = n
  - Width from Benford frequency:  sigma_n = (c / log10(1 + 1/n))^beta
  - Energy:  E(x|y0) = (x - y0)^2 / (2 tau^2)
                      - kappa * logsumexp_n( -log(sigma_n) - (x - n)^2 / (2 sigma_n^2) )
    (Gaussians are normalized so wide wells contribute less)
  - Evidence channel:  Var(y0 | n) = s00^2 + s0 * n^lambda0 (manuscript second
    argument to Normal is variance; sampling uses std = sqrt(Var)), or Var = s00^2
    when ``constant_noise: true`` (s0 fixed at 0).
  - RT proxy (configurable via RT_MODE):
      "steps"   :  t0 + c_steps * n_steps
      "entropy" :  t0 + kappa * H(posterior)
      "both"    :  t0 + c_steps * n_steps + kappa * H(posterior)
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm
import wandb

from scripts.fit_helpers import (
    OUTPUT_DIR, PARAM_NAMES, BOUNDS, FIXED, N_TRUE_MIN, N_TRUE_MAX,
    N_SUPPORT_MAX, DURATION_MS, COMPRESS, DEVICE, RT_MODE,
    VARIANCE_FLOOR, WELL_TYPE, FIXED_BETA, FIXED_KAPPA, FIXED_ETA,
    CONSTANT_NOISE, complete_piantadosi_theta,
    extract_empirical, _extract_empirical_arrays,
    simulate, loss_fn,
    compute_goodness, compute_nll, hessian_ci,
    parallel_bootstrap_ci, refine_cis_with_profile,
)

NLL_N_SAMPLES = 10000
from scripts.plots import (
    plot_comparison, plot_energy_landscape, plot_number_line,
)


# ── Optimization ─────────────────────────────────────────────────────────


def fit_parameters(
    empirical: pd.DataFrame,
    maxiter: int = 200,
    patience: int = 5,
    min_delta: float = 1e-4,
    es_start: int = 30,
) -> tuple[dict, object]:
    emp_arrays = _extract_empirical_arrays(empirical)
    sample_state = [100]  # progressive: 100 → 200 → 300
    best_loss = [float("inf")]
    es_counter = [0]
    es_best = [float("inf")]

    def tracked_loss_fn(theta_input, *args):
        result = loss_fn(theta_input, *args)
        val = float(result) if np.ndim(result) == 0 else float(np.min(result))
        if val < best_loss[0]:
            best_loss[0] = val
        return result

    pbar = tqdm(total=maxiter, desc=f"DE ({len(PARAM_NAMES)} params)", unit="gen")

    def callback(xk, convergence=0.0):
        pbar.update(1)
        gen = pbar.n
        if gen >= 20:
            sample_state[0] = 300
        elif gen >= 10:
            sample_state[0] = 200

        # Early stopping (only after es_start generations at full samples)
        stop = False
        if gen >= es_start:
            if es_best[0] - best_loss[0] > min_delta:
                es_best[0] = best_loss[0]
                es_counter[0] = 0
            else:
                es_counter[0] += 1
            if es_counter[0] >= patience:
                stop = True

        pbar.set_postfix(
            loss=f"{best_loss[0]:.6f}",
            conv=f"{convergence:.5f}",
            n_samp=sample_state[0],
            es=f"{es_counter[0]}/{patience}" if gen >= es_start else "-",
        )
        current_theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, xk)))
        wandb.log({
            "train/gen": gen,
            "train/convergence": convergence,
            "train/best_loss": best_loss[0],
            "train/n_samples": sample_state[0],
            "train/es_counter": es_counter[0],
            "train/elapsed_s": pbar.format_dict["elapsed"],
            **{f"train/param/{k}": v for k, v in current_theta.items()},
        })
        return stop

    result = differential_evolution(
        tracked_loss_fn,
        bounds=BOUNDS,
        args=(emp_arrays, sample_state, 73),
        maxiter=maxiter,
        popsize=8,
        seed=17,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.85,
        callback=callback,
        disp=False,
        polish=True,
        init="latinhypercube",
        vectorized=True,
        updating="deferred",
    )
    pbar.close()

    theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, result.x)))
    tqdm.write(f"Done. Best loss = {result.fun:.6f}")
    for k, v in theta.items():
        tqdm.write(f"  {k:>10s} = {v:.4f}")
    return theta, result


# ── Diagnostics: NLL closure, Hessian CIs, bootstrap CIs ─────────────────


def _make_nll_fn(
    empirical: pd.DataFrame,
    n_samples: int = NLL_N_SAMPLES,
    seed: int = 73,
):
    """Return a callable NLL(theta_vec) with fixed MC seed for Hessian CIs.

    A fixed ``seed`` keeps the Monte-Carlo noise deterministic across
    finite-difference Hessian evaluations so the Hessian is not dominated
    by stochasticity.
    """

    def nll_fn(theta_vec: np.ndarray) -> float:
        theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, theta_vec)))
        return compute_nll(theta, empirical, n_samples=n_samples, seed=seed)

    return nll_fn


def _make_nll_of_theta(
    empirical: pd.DataFrame,
    n_samples: int = NLL_N_SAMPLES,
    seed: int = 73,
):
    """Return a callable NLL(theta_dict) used by profile-likelihood refits."""

    def nll_of_theta(theta_dict: dict) -> float:
        theta_full = complete_piantadosi_theta(theta_dict)
        return compute_nll(
            theta_full, empirical, n_samples=n_samples, seed=seed
        )

    return nll_of_theta


def _make_fit_with_fixed(
    empirical: pd.DataFrame, include_mean_guess: bool = True
):
    """Return a callable that refits the remaining params given fixed ones."""
    from scripts.parameter_recovery import fit_silent

    emp_arr = _extract_empirical_arrays(empirical)
    if not include_mean_guess:
        emp_arr = dict(emp_arr)
        emp_arr["mean_guess"] = np.full_like(
            emp_arr["pct_exact"], np.nan, dtype=float
        )

    def fit_with_fixed(fixed: dict, seed: int) -> dict:
        theta, _ = fit_silent(emp_arr, fixed_params=fixed, seed=seed)
        return theta

    return fit_with_fixed


def _resample_piantadosi(
    duration_ms: int,
    seed: int,
    csv_path: str = "data/data_piantadosi.csv",
) -> pd.DataFrame:
    """Trial-level bootstrap stratified by n_shown."""
    df = pd.read_csv(csv_path)
    sub = df[(df["cond"] == "DURATION") & (df["exp_dur"] == duration_ms)]
    trial = (
        sub.groupby(["uniqueid", "trial_id"])
        .agg(
            n_shown=("n_shown", "first"),
            n_guess=("n_guess", "first"),
            rt=("rt", "first"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in trial.groupby("n_shown"):
        idx = rng.integers(0, len(grp), size=len(grp))
        parts.append(grp.iloc[idx])
    boot = pd.concat(parts, ignore_index=True)
    boot["exact"] = (boot["n_guess"] == boot["n_shown"]).astype(int)
    agg = (
        boot.groupby("n_shown")
        .agg(
            mean_guess=("n_guess", "mean"),
            std_guess=("n_guess", "std"),
            pct_exact=("exact", "mean"),
            median_rt=("rt", "median"),
            count=("n_guess", "count"),
        )
        .reset_index()
    )
    agg["pct_exact"] *= 100
    agg["median_rt_s"] = agg["median_rt"] / 1000.0
    agg = agg[(agg["n_shown"] >= N_TRUE_MIN) & (agg["n_shown"] <= N_TRUE_MAX)]
    return agg.reset_index(drop=True)


def _boot_worker_piant(payload: tuple) -> dict | None:
    """Module-level bootstrap worker (pickle-able for multiprocessing).

    ``payload = (duration_ms, bseed, csv_path)``. Returns a theta dict with
    an additional ``_loss`` entry on success, or ``None`` on failure. Sets
    torch to single-threaded to avoid OMP oversubscription under pool.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    from scripts.parameter_recovery import fit_silent

    duration_ms, bseed, csv_path = payload
    try:
        emp_b = _resample_piantadosi(duration_ms, bseed, csv_path=csv_path)
        emp_arr = _extract_empirical_arrays(emp_b)
        theta_b, loss_b = fit_silent(emp_arr, seed=bseed)
    except Exception:
        return None

    out = dict(theta_b)
    out["_loss"] = float(loss_b)
    return out


def _print_param_table(ci_df: pd.DataFrame) -> None:
    has_method = "method" in ci_df.columns
    header = (
        f"\n{'param':>12s}  {'value':>10s}  {'SE':>10s}  "
        f"{'95% CI':>24s}  {'status':<18s}"
    )
    if has_method:
        header += "  method"
    tqdm.write(header)
    tqdm.write("-" * (90 if has_method else 70))
    for _, row in ci_df.iterrows():
        se = row["se"]
        se_str = f"{se:10.4f}" if np.isfinite(se) else f"{'--':>10s}"
        ci_str = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
        line = (
            f"{row['param']:>12s}  {row['value']:10.4f}  "
            f"{se_str}  {ci_str:>24s}  {str(row['status']):<18s}"
        )
        if has_method:
            line += f"  {row['method']}"
        tqdm.write(line)


# ── main ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit Piantadosi attractor model")
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help=(
            "Number of bootstrap refits for percentile CIs (default: 0 = skip). "
            "Each refit uses the reduced budget from parameter_recovery.RECOVERY_BUDGET."
        ),
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel workers for bootstrap (default 1 = serial). "
            "Use -1 for all cores. Each worker consumes ~0.6-0.8 GB."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    duration_ms = DURATION_MS

    run = wandb.init(
        project="hopfield-piantadosi",
        config={
            "duration_ms": duration_ms,
            "rt_mode": RT_MODE,
            "well_type": WELL_TYPE,
            "compress_numberline": COMPRESS,
            "fixed_beta": FIXED_BETA,
            "fixed_kappa": FIXED_KAPPA,
            "fixed_eta": FIXED_ETA,
            "constant_noise": CONSTANT_NOISE,
            "variance_floor": VARIANCE_FLOOR,
            "N_TRUE_MIN": N_TRUE_MIN,
            "N_TRUE_MAX": N_TRUE_MAX,
            "N_SUPPORT_MAX": N_SUPPORT_MAX,
            "device": str(DEVICE),
            **FIXED,
        },
    )

    print(f"Extracting empirical {duration_ms} ms targets ...", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    empirical = extract_empirical(duration_ms)
    print(
        f"  {len(empirical)} numerosities, "
        f"{int(empirical['count'].sum())} trials",
        flush=True,
    )
    print(empirical.to_string(index=False), flush=True)

    theta, result = fit_parameters(empirical)

    if COMPRESS:
        rho = theta["rho"]
        print(f"\n  rho = {rho:.4f}  (1.0 = linear, <1 = compressed)")
        print(f"  Interpretation: {'minimal compression' if rho > 0.85 else 'significant compression'}")

    print("\nFinal simulation (5000 samples/n) ...", flush=True)
    model = simulate(theta, n_samples_per_n=5000, seed=73)
    print(model.to_string(index=False), flush=True)

    tag = "compressed" if COMPRESS else WELL_TYPE
    path_comparison = plot_comparison(
        empirical, model, theta, OUTPUT_DIR,
        tag=tag, duration_ms=duration_ms,
    )
    path_energy = plot_energy_landscape(theta, OUTPUT_DIR, duration_ms=duration_ms)

    images = {
        "comparison_plot": wandb.Image(str(path_comparison)),
        "energy_landscape": wandb.Image(str(path_energy)),
    }

    if COMPRESS:
        path_line = plot_number_line(theta, OUTPUT_DIR, duration_ms=duration_ms)
        images["number_line"] = wandb.Image(str(path_line))

    # ── goodness, AIC, BIC ───────────────────────────────────────────────
    k = len(PARAM_NAMES)
    goodness = compute_goodness(empirical, model, k=k)

    print("\nGoodness of fit:")
    for key in ("logL", "AIC", "BIC", "n_data", "k_params"):
        if key in goodness:
            print(f"  {key:>10s} = {goodness[key]}")

    # ── Confidence intervals: log-Hessian + profile-likelihood fallback ──
    print("\nComputing log-space Hessian 95% CIs ...", flush=True)
    nll_fn = _make_nll_fn(empirical, n_samples=NLL_N_SAMPLES, seed=73)
    theta_vec_fit = np.array([theta[p] for p in PARAM_NAMES])
    ci_df = hessian_ci(
        nll_fn,
        theta_vec=theta_vec_fit,
        param_names=PARAM_NAMES,
        bounds=BOUNDS,
    )

    needs_profile = ci_df.apply(
        lambda r: (not np.isfinite(r["se"]))
        or r["status"] in ("non_pd", "singular", "at_bound"),
        axis=1,
    )
    if needs_profile.any():
        n_bad = int(needs_profile.sum())
        print(
            f"\n{n_bad}/{len(ci_df)} params need profile-likelihood refinement "
            f"({', '.join(ci_df.loc[needs_profile, 'param'])})"
        )
        nll_of_theta = _make_nll_of_theta(empirical)
        fit_with_fixed = _make_fit_with_fixed(empirical)
        nll_hat = float(nll_fn(theta_vec_fit))
        ci_df = refine_cis_with_profile(
            ci_df,
            nll_of_theta=nll_of_theta,
            fit_with_fixed=fit_with_fixed,
            theta_hat=theta,
            nll_hat=nll_hat,
            bounds=BOUNDS,
            verbose=True,
        )

    _print_param_table(ci_df)
    ci_csv = OUTPUT_DIR / f"fit_{duration_ms}ms_ci_hessian.csv"
    ci_df.to_csv(ci_csv, index=False)
    print(f"Saved {ci_csv}")

    # ── Optional: bootstrap CIs ──────────────────────────────────────────
    boot_df: pd.DataFrame | None = None
    if args.bootstrap > 0:
        csv_path = "data/data_piantadosi.csv"
        n_jobs = args.n_jobs
        print(
            f"\nRunning {args.bootstrap} bootstrap refits "
            f"(reduced DE budget, n_jobs={n_jobs}) ..."
        )
        rng = np.random.default_rng(17)
        seeds = rng.integers(0, 2**31 - 1, size=args.bootstrap).tolist()
        payloads = [(duration_ms, int(s), csv_path) for s in seeds]

        boot_df = parallel_bootstrap_ci(
            worker_fn=_boot_worker_piant,
            payloads=payloads,
            param_names=PARAM_NAMES,
            n_jobs=n_jobs,
            desc=f"bootstrap {duration_ms}ms",
        )
        print("\nBootstrap 95% percentile CIs:")
        print(boot_df.to_string(index=False))
        boot_csv = OUTPUT_DIR / f"fit_{duration_ms}ms_ci_bootstrap.csv"
        boot_df.to_csv(boot_csv, index=False)
        print(f"Saved {boot_csv}")

    # ── wandb logging ────────────────────────────────────────────────────
    wandb_summary = {
        **{f"param/{k_}": v for k_, v in theta.items()},
        **{f"goodness/{k_}": v for k_, v in goodness.items()},
        "loss": result.fun,
        "n_func_evals": result.nfev,
        "optimizer_success": result.success,
    }
    for _, row in ci_df.iterrows():
        p = row["param"]
        wandb_summary[f"ci/{p}/se"] = float(row["se"])
        wandb_summary[f"ci/{p}/lo"] = float(row["ci_lo"])
        wandb_summary[f"ci/{p}/hi"] = float(row["ci_hi"])
        wandb_summary[f"ci/{p}/status"] = row["status"]
        if "method" in ci_df.columns:
            wandb_summary[f"ci/{p}/method"] = row["method"]
    if boot_df is not None:
        for _, row in boot_df.iterrows():
            p = row["param"]
            wandb_summary[f"ci_bootstrap/{p}/lo"] = float(row["ci_lo"])
            wandb_summary[f"ci_bootstrap/{p}/hi"] = float(row["ci_hi"])
            wandb_summary[f"ci_bootstrap/{p}/median"] = float(row["median"])
    run.summary.update(wandb_summary)

    run.log(images)
    wandb.finish()

    # ── local CSV ────────────────────────────────────────────────────────
    ci_flat: dict = {}
    for _, row in ci_df.iterrows():
        p = row["param"]
        ci_flat[f"{p}_se"] = row["se"]
        ci_flat[f"{p}_ci_lo"] = row["ci_lo"]
        ci_flat[f"{p}_ci_hi"] = row["ci_hi"]
        if "method" in ci_df.columns:
            ci_flat[f"{p}_ci_method"] = row["method"]
    csv_path = OUTPUT_DIR / f"fit_{duration_ms}ms_results.csv"
    pd.DataFrame(
        [{"duration_ms": duration_ms, **theta, **goodness, **ci_flat}]
    ).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
