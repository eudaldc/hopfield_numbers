#!/usr/bin/env python
"""Fit the attractor model separately to DD and control groups from DeCarli et al.

Data: decarli_exact_enumeration.csv (from Exact_enumeration_task.xlsx) —
per-subject accuracy and RT by numerosity (1--8), split by DD (n=32) and
controls (n=32). Fits use numerosities 1--7 only (n=8 excluded: guessing
end-effect).

The same Poisson-well attractor model from the Piantadosi fit is used, with
6 free parameters:
  sigma_c   : well width scale (representation precision)
  tau_in    : anchor strength (evidence weighting)
  s0        : coefficient in encoding variance s0 * n^lambda0
  lambda0   : exponent in that variance (SD = sqrt(s00^2 + s0 * n^lambda0))
  t0        : base reaction time
  kappa     : entropy-to-RT conversion

Unlike the Piantadosi fit, mean_guess is unavailable (no trial-level
responses), so the loss uses only accuracy and RT.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    _simulate_core,
    simulate,
    PARAM_NAMES,
    BOUNDS,
    N_TRUE_MIN,
    N_TRUE_MAX,
    N_SUPPORT_MAX,
    WELL_TYPE,
    RT_MODE,
    FIXED_BETA,
    FIXED_KAPPA,
    FIXED_ETA,
    complete_piantadosi_theta,
    compute_goodness,
    compute_nll,
    hessian_ci,
    parallel_bootstrap_ci,
    refine_cis_with_profile,
)

DECARLI_CHANNELS = ("pct_exact", "rt")
NLL_N_SAMPLES = 10000

# ── Settings ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output/decarli_fit")
ENUM_PATH = Path("data/decarli_exact_enumeration.csv")
PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations_poisson.csv")

# Trial-level files include n=8; model fits follow the paper and omit it.
DECARLI_FIT_NUMEROSITY_MAX = 7

_N_SHOWN = np.arange(N_TRUE_MIN, N_TRUE_MAX + 1)


# ── Data loading ──────────────────────────────────────────────────────────


def load_group_targets(group: str) -> pd.DataFrame:
    """Load DeCarli data for one group and return in Piantadosi-like format.

    Returns DataFrame with columns: n_shown, pct_exact, median_rt_s, count.
    """
    df = pd.read_csv(ENUM_PATH)
    sub = df[
        (df["Group"] == group)
        & (df["Numerosity"] <= DECARLI_FIT_NUMEROSITY_MAX)
    ]

    agg = (
        sub.groupby("Numerosity")
        .agg(
            pct_exact=("Accuracy", "mean"),
            median_rt_s=("RTs", "mean"),
            count=("Sub", "nunique"),
        )
        .reset_index()
        .rename(columns={"Numerosity": "n_shown"})
    )
    agg["pct_exact"] *= 100
    return agg


def _extract_arrays(targets: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pre-extract numpy arrays for the loss function hot path."""
    emp = targets.sort_values("n_shown").reset_index(drop=True)
    return {
        "n_shown": emp["n_shown"].values,
        "pct_exact": emp["pct_exact"].values,
        "median_rt_s": emp["median_rt_s"].values,
        "model_mask": np.isin(_N_SHOWN, emp["n_shown"].values),
    }


# ── Loss function (accuracy + RT only) ───────────────────────────────────


def _single_loss(
    theta_vec: np.ndarray,
    emp_arrays: dict[str, np.ndarray],
    n_samples: int,
    seed: int,
) -> float:
    theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, theta_vec)))
    try:
        model = _simulate_core(theta, n_samples_per_n=n_samples, seed=seed)
    except Exception:
        return 1e6

    mask = emp_arrays["model_mask"]

    ac_e, ac_m = emp_arrays["pct_exact"], model["pct_exact"][mask]
    loss_acc = np.mean((ac_e - ac_m) ** 2) / (np.ptp(ac_e) ** 2 + 1e-6)

    rt_e, rt_m = emp_arrays["median_rt_s"], model["median_rt"][mask]
    loss_rt = np.mean((rt_e - rt_m) ** 2) / (np.ptp(rt_e) ** 2 + 1e-6)

    total = float(loss_acc + loss_rt)
    return 1e6 if (np.isnan(total) or np.isinf(total)) else total


def _loss_fn(
    theta_input: np.ndarray,
    emp_arrays: dict[str, np.ndarray],
    sample_state: list[int],
    seed: int = 73,
) -> float | np.ndarray:
    """Vectorized-DE-compatible wrapper."""
    n_samples = sample_state[0]
    if theta_input.ndim == 1:
        return _single_loss(theta_input, emp_arrays, n_samples, seed)
    pop_size = theta_input.shape[1]
    losses = np.empty(pop_size)
    for i in range(pop_size):
        losses[i] = _single_loss(theta_input[:, i], emp_arrays, n_samples, seed)
    return losses


# ── Fitting ───────────────────────────────────────────────────────────────


def fit_decarli_from_arrays(
    emp_arrays: dict[str, np.ndarray],
    label: str = "",
    maxiter: int = 200,
    patience: int = 5,
    min_delta: float = 1e-4,
    es_start: int = 30,
    de_seed: int = 17,
    verbose: bool = True,
) -> tuple[dict, object]:
    """Differential evolution + L-BFGS polish on DeCarli-style ``emp_arrays``."""
    sample_state = [100]
    best_loss = [float("inf")]
    es_counter = [0]
    es_best = [float("inf")]

    def tracked(theta_input, *args):
        result = _loss_fn(theta_input, *args)
        val = float(result) if np.ndim(result) == 0 else float(np.min(result))
        if val < best_loss[0]:
            best_loss[0] = val
        return result

    gen_counter = [0]
    pbar = (
        tqdm(total=maxiter, desc=f"DE {label} ({len(PARAM_NAMES)}p)", unit="gen")
        if verbose
        else None
    )

    def callback(xk, convergence=0.0):
        gen_counter[0] += 1
        gen = gen_counter[0]
        if gen >= 20:
            sample_state[0] = 300
        elif gen >= 10:
            sample_state[0] = 200

        stop = False
        if gen >= es_start:
            if es_best[0] - best_loss[0] > min_delta:
                es_best[0] = best_loss[0]
                es_counter[0] = 0
            else:
                es_counter[0] += 1
            if es_counter[0] >= patience:
                stop = True

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{best_loss[0]:.6f}",
                conv=f"{convergence:.5f}",
                n_samp=sample_state[0],
                es=f"{es_counter[0]}/{patience}" if gen >= es_start else "-",
            )
        return stop

    result = differential_evolution(
        tracked,
        bounds=BOUNDS,
        args=(emp_arrays, sample_state, 73),
        maxiter=maxiter,
        popsize=8,
        seed=de_seed,
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
    if pbar is not None:
        pbar.close()

    theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, result.x)))
    if verbose:
        tqdm.write(f"\n{label} — best loss = {result.fun:.6f}")
        for k, v in theta.items():
            tqdm.write(f"  {k:>10s} = {v:.4f}")
    return theta, result


def fit_group(
    targets: pd.DataFrame,
    label: str,
    maxiter: int = 200,
    patience: int = 5,
    min_delta: float = 1e-4,
    es_start: int = 30,
) -> tuple[dict, object]:
    emp_arrays = _extract_arrays(targets)
    return fit_decarli_from_arrays(
        emp_arrays,
        label=label,
        maxiter=maxiter,
        patience=patience,
        min_delta=min_delta,
        es_start=es_start,
        de_seed=17,
        verbose=True,
    )


# ── Diagnostics: NLL closure, Hessian CIs, bootstrap CIs ─────────────────


def _make_nll_fn(
    empirical: pd.DataFrame,
    n_samples: int = NLL_N_SAMPLES,
    seed: int = 73,
):
    """Return a callable NLL(theta_vec) with fixed MC seed for Hessian CIs."""

    def nll_fn(theta_vec: np.ndarray) -> float:
        theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, theta_vec)))
        return compute_nll(
            theta, empirical, channels=DECARLI_CHANNELS,
            n_samples=n_samples, seed=seed,
        )

    return nll_fn


def _make_nll_of_theta(
    empirical: pd.DataFrame,
    n_samples: int = NLL_N_SAMPLES,
    seed: int = 73,
):
    def nll_of_theta(theta_dict: dict) -> float:
        theta_full = complete_piantadosi_theta(theta_dict)
        return compute_nll(
            theta_full, empirical, channels=DECARLI_CHANNELS,
            n_samples=n_samples, seed=seed,
        )

    return nll_of_theta


def _make_fit_with_fixed_decarli(targets: pd.DataFrame):
    """Return a callable that refits remaining params with DeCarli (no mean_guess)."""
    from scripts.parameter_recovery import fit_silent

    emp_sorted = targets.sort_values("n_shown").reset_index(drop=True)
    emp_arr = {
        "n_shown": emp_sorted["n_shown"].values,
        "mean_guess": np.full(len(emp_sorted), np.nan, dtype=float),
        "pct_exact": emp_sorted["pct_exact"].values,
        "median_rt_s": emp_sorted["median_rt_s"].values,
        "model_mask": np.isin(_N_SHOWN, emp_sorted["n_shown"].values),
    }

    def fit_with_fixed(fixed: dict, seed: int) -> dict:
        theta, _ = fit_silent(emp_arr, fixed_params=fixed, seed=seed)
        return theta

    return fit_with_fixed


def _resample_decarli_subjects(group: str, seed: int) -> pd.DataFrame:
    """Subject-level bootstrap: sample subjects within a group with replacement."""
    df = pd.read_csv(ENUM_PATH)
    sub = df[
        (df["Group"] == group)
        & (df["Numerosity"] <= DECARLI_FIT_NUMEROSITY_MAX)
    ]
    subs = sub["Sub"].unique()
    rng = np.random.default_rng(seed)
    picked = rng.choice(subs, size=len(subs), replace=True)
    parts = [sub[sub["Sub"] == s] for s in picked]
    boot = pd.concat(parts, ignore_index=True)
    agg = (
        boot.groupby("Numerosity")
        .agg(
            pct_exact=("Accuracy", "mean"),
            median_rt_s=("RTs", "mean"),
            count=("Sub", "count"),
        )
        .reset_index()
        .rename(columns={"Numerosity": "n_shown"})
    )
    agg["pct_exact"] *= 100
    return agg


def _boot_worker_decarli(payload: tuple) -> dict | None:
    """Module-level bootstrap worker (pickle-able for multiprocessing).

    Payload formats accepted:
      - ``(group, bseed)``                 : refit all PARAM_NAMES.
      - ``(group, bseed, fixed_params)``   : refit only the free parameters;
        ``fixed_params`` is a dict of {param_name: value} passed verbatim to
        ``fit_silent``. Use this for a conditional bootstrap that fixes all
        but one (or a few) parameter(s) at their point estimate — much
        cheaper per iteration.

    Returns a theta dict with an extra ``_loss`` entry on success, or
    ``None`` on failure. Forces torch to single-thread to keep pool workers
    from oversubscribing CPU.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    from scripts.parameter_recovery import fit_silent

    if len(payload) == 3:
        group, bseed, fixed_params = payload
    else:
        group, bseed = payload
        fixed_params = None

    try:
        emp_b = _resample_decarli_subjects(group, bseed)
        emp_sorted = emp_b.sort_values("n_shown").reset_index(drop=True)
        emp_arr = {
            "n_shown": emp_sorted["n_shown"].values,
            "mean_guess": np.full(len(emp_sorted), np.nan, dtype=float),
            "pct_exact": emp_sorted["pct_exact"].values,
            "median_rt_s": emp_sorted["median_rt_s"].values,
            "model_mask": np.isin(_N_SHOWN, emp_sorted["n_shown"].values),
        }
        theta_b, loss_b = fit_silent(
            emp_arr, fixed_params=fixed_params, seed=bseed
        )
    except Exception:
        return None

    out = dict(theta_b)
    out["_loss"] = float(loss_b)
    return out


def _print_param_table(ci_df: pd.DataFrame, label: str) -> None:
    tqdm.write(
        f"\n[{label}] {'param':>12s}  {'value':>10s}  {'SE':>10s}  "
        f"{'95% CI':>24s}  status"
    )
    tqdm.write("-" * 80)
    for _, row in ci_df.iterrows():
        se = row["se"]
        se_str = f"{se:10.4f}" if np.isfinite(se) else f"{'--':>10s}"
        ci_str = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
        tqdm.write(
            f"{'':>7s} {row['param']:>12s}  {row['value']:10.4f}  "
            f"{se_str}  {ci_str:>24s}  {row['status']}"
        )


# ── main ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fit DeCarli DD vs Controls attractor model"
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Per-group bootstrap refits for percentile CIs (default: 0 = skip).",
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
    ap.add_argument(
        "--only-tau",
        action="store_true",
        help=(
            "Conditional bootstrap: fix all params except tau_in at the "
            "point estimate and refit only tau_in per resample. Much faster "
            "but CIs assume the other params are known (no joint uncertainty)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Fitting DeCarli Exact Enumeration: DD vs Controls")
    print("=" * 60)
    print(f"  Well type: {WELL_TYPE}")
    print(f"  RT mode:   {RT_MODE}")
    print(f"  Params:    {PARAM_NAMES}")
    print(f"  Fixed:     beta={FIXED_BETA}, kappa_lse={FIXED_KAPPA}, eta={FIXED_ETA}")
    print(
        f"  Range:     n_true {N_TRUE_MIN}-{N_TRUE_MAX}, support up to {N_SUPPORT_MAX}"
    )

    # ── Load targets ──────────────────────────────────────────────────────
    targets_dd = load_group_targets("DD")
    targets_ctrl = load_group_targets("contr")

    print("\nDD targets:")
    print(targets_dd.to_string(index=False))
    print("\nControl targets:")
    print(targets_ctrl.to_string(index=False))

    # ── Fit ────────────────────────────────────────────────────────────────
    theta_dd, res_dd = fit_group(targets_dd, "DD")
    theta_ctrl, res_ctrl = fit_group(targets_ctrl, "Controls")

    # ── Reference: Piantadosi adult fits ──────────────────────────────────
    piant = pd.read_csv(PARAMS_CSV)
    ref_160 = piant[piant["duration_ms"] == 160].iloc[0].to_dict()
    ref_640 = piant[piant["duration_ms"] == 640].iloc[0].to_dict()

    # ── Parameter comparison table ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Parameter comparison")
    print("=" * 60)
    header = (
        f"{'Parameter':>12s}  {'DD':>10s}  {'Controls':>10s}  "
        f"{'Δ (DD−Ctrl)':>12s}  {'Ratio DD/C':>10s}  "
        f"{'Adult 160ms':>12s}  {'Adult 640ms':>12s}"
    )
    print(header)
    print("-" * len(header))

    comparison_rows: list[dict] = []
    for p in PARAM_NAMES:
        dd_val = theta_dd[p]
        ct_val = theta_ctrl[p]
        diff = dd_val - ct_val
        ratio = dd_val / ct_val if ct_val != 0 else float("inf")
        adult_160 = ref_160.get(p, float("nan"))
        adult_640 = ref_640.get(p, float("nan"))
        print(
            f"{p:>12s}  {dd_val:10.4f}  {ct_val:10.4f}  {diff:+12.4f}  "
            f"{ratio:10.3f}  {adult_160:12.4f}  {adult_640:12.4f}"
        )
        comparison_rows.append(
            {
                "parameter": p,
                "DD": dd_val,
                "Controls": ct_val,
                "diff": diff,
                "ratio": ratio,
                "adult_160ms": adult_160,
                "adult_640ms": adult_640,
            }
        )

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = OUTPUT_DIR / "parameter_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved {comp_csv}")

    # ── Simulate with fitted params ───────────────────────────────────────
    print("\nFinal simulation (5000 samples/n) ...", flush=True)
    model_dd = simulate(theta_dd, n_samples_per_n=5000, seed=73)
    model_ctrl = simulate(theta_ctrl, n_samples_per_n=5000, seed=73)

    # ── Goodness of fit (AIC/BIC, per-channel) ────────────────────────────
    k = len(PARAM_NAMES)
    good_dd = compute_goodness(
        targets_dd, model_dd, k=k, channels=DECARLI_CHANNELS
    )
    good_ctrl = compute_goodness(
        targets_ctrl, model_ctrl, k=k, channels=DECARLI_CHANNELS
    )
    for lab, g in [("DD", good_dd), ("Controls", good_ctrl)]:
        print(f"\n[{lab}] goodness:")
        for key in ("logL", "AIC", "BIC", "n_data", "k_params"):
            if key in g:
                print(f"  {key:>10s} = {g[key]}")

    # ── CIs: log-Hessian + profile-likelihood fallback ────────────────────
    print("\nComputing log-space Hessian 95% CIs ...", flush=True)

    def _ci_for_group(
        targets: pd.DataFrame, theta: dict, label: str
    ) -> pd.DataFrame:
        nll_fn = _make_nll_fn(targets)
        theta_vec = np.array([theta[p] for p in PARAM_NAMES])
        ci = hessian_ci(nll_fn, theta_vec, PARAM_NAMES, BOUNDS)
        needs = ci.apply(
            lambda r: (not np.isfinite(r["se"]))
            or r["status"] in ("non_pd", "singular", "at_bound"),
            axis=1,
        )
        if needs.any():
            print(
                f"  [{label}] refining {int(needs.sum())}/{len(ci)} params "
                f"via profile likelihood"
            )
            nll_of_theta = _make_nll_of_theta(targets)
            fit_with_fixed = _make_fit_with_fixed_decarli(targets)
            nll_hat = float(nll_fn(theta_vec))
            ci = refine_cis_with_profile(
                ci,
                nll_of_theta=nll_of_theta,
                fit_with_fixed=fit_with_fixed,
                theta_hat=theta,
                nll_hat=nll_hat,
                bounds=BOUNDS,
                verbose=True,
            )
        return ci

    ci_dd = _ci_for_group(targets_dd, theta_dd, "DD")
    ci_ctrl = _ci_for_group(targets_ctrl, theta_ctrl, "Controls")

    _print_param_table(ci_dd, "DD")
    _print_param_table(ci_ctrl, "Controls")

    ci_dd.to_csv(OUTPUT_DIR / "ci_hessian_DD.csv", index=False)
    ci_ctrl.to_csv(OUTPUT_DIR / "ci_hessian_controls.csv", index=False)

    # ── Optional bootstrap ────────────────────────────────────────────────
    boot_dd: pd.DataFrame | None = None
    boot_ctrl: pd.DataFrame | None = None
    if args.bootstrap > 0:
        n_jobs = args.n_jobs
        mode = "tau_in only (conditional)" if args.only_tau else "all params"
        print(
            f"\nRunning {args.bootstrap} bootstrap refits per group "
            f"(subject-level resampling, n_jobs={n_jobs}, {mode}) ..."
        )
        rng_dd = np.random.default_rng(17)
        rng_ct = np.random.default_rng(23)
        seeds_dd = rng_dd.integers(0, 2**31 - 1, size=args.bootstrap).tolist()
        seeds_ct = rng_ct.integers(0, 2**31 - 1, size=args.bootstrap).tolist()

        if args.only_tau:
            fixed_dd = {p: theta_dd[p] for p in PARAM_NAMES if p != "tau_in"}
            fixed_ct = {p: theta_ctrl[p] for p in PARAM_NAMES if p != "tau_in"}
            payloads_dd = [("DD", int(s), fixed_dd) for s in seeds_dd]
            payloads_ct = [("contr", int(s), fixed_ct) for s in seeds_ct]
            report_names = ["tau_in"]
            suffix = "_tau_only"
        else:
            payloads_dd = [("DD", int(s)) for s in seeds_dd]
            payloads_ct = [("contr", int(s)) for s in seeds_ct]
            report_names = list(PARAM_NAMES)
            suffix = ""

        want_draws = not args.only_tau
        boot_dd = parallel_bootstrap_ci(
            worker_fn=_boot_worker_decarli,
            payloads=payloads_dd,
            param_names=report_names,
            n_jobs=n_jobs,
            desc="bootstrap DD",
            return_draws=want_draws,
        )
        boot_ctrl = parallel_bootstrap_ci(
            worker_fn=_boot_worker_decarli,
            payloads=payloads_ct,
            param_names=report_names,
            n_jobs=n_jobs,
            desc="bootstrap Controls",
            return_draws=want_draws,
        )
        if want_draws:
            boot_dd, draws_dd = boot_dd
            boot_ctrl, draws_ct = boot_ctrl
            draws_dd.to_csv(
                OUTPUT_DIR / "bootstrap_draws_DD.csv", index=False
            )
            draws_ct.to_csv(
                OUTPUT_DIR / "bootstrap_draws_controls.csv", index=False
            )
            print(f"\nSaved {OUTPUT_DIR / 'bootstrap_draws_DD.csv'}")
            print(f"Saved {OUTPUT_DIR / 'bootstrap_draws_controls.csv'}")
        print("\n[DD] bootstrap 95% percentile CIs:")
        print(boot_dd.to_string(index=False))
        print("\n[Controls] bootstrap 95% percentile CIs:")
        print(boot_ctrl.to_string(index=False))
        boot_dd.to_csv(
            OUTPUT_DIR / f"ci_bootstrap_DD{suffix}.csv", index=False
        )
        boot_ctrl.to_csv(
            OUTPUT_DIR / f"ci_bootstrap_controls{suffix}.csv", index=False
        )

    # ── Empirical SE for plotting ─────────────────────────────────────────
    raw = pd.read_csv(ENUM_PATH)
    emp_with_se: dict[str, pd.DataFrame] = {}
    for group in ("DD", "contr"):
        sub = raw[raw["Group"] == group]
        agg = (
            sub.groupby("Numerosity")
            .agg(
                mean_acc=("Accuracy", "mean"),
                se_acc=("Accuracy", "sem"),
                mean_rt=("RTs", "mean"),
                se_rt=("RTs", "sem"),
            )
            .reset_index()
        )
        agg["mean_acc_pct"] = agg["mean_acc"] * 100
        agg["se_acc_pct"] = agg["se_acc"] * 100
        emp_with_se[group] = agg

    # ── Plot ──────────────────────────────────────────────────────────────
    from scripts.plots import plot_decarli_fit, plot_decarli_param_boxfacets

    plot_decarli_fit(
        emp_with_se,
        model_dd,
        model_ctrl,
        OUTPUT_DIR,
        good_dd=good_dd,
        good_ctrl=good_ctrl,
    )

    # ── Save fitted params ────────────────────────────────────────────────
    rows = []
    for label, theta, res, good, ci in [
        ("DD", theta_dd, res_dd, good_dd, ci_dd),
        ("Controls", theta_ctrl, res_ctrl, good_ctrl, ci_ctrl),
    ]:
        ci_flat: dict = {}
        for _, row in ci.iterrows():
            p = row["param"]
            ci_flat[f"{p}_se"] = row["se"]
            ci_flat[f"{p}_ci_lo"] = row["ci_lo"]
            ci_flat[f"{p}_ci_hi"] = row["ci_hi"]
        rows.append(
            {"group": label, "loss": res.fun, **theta, **good, **ci_flat}
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fitted_params.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'fitted_params.csv'}")

    plot_decarli_param_boxfacets(OUTPUT_DIR)


if __name__ == "__main__":
    main()
