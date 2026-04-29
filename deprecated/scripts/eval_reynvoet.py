#!/usr/bin/env python
"""Compare model Weber predictions against Reynvoet et al. (2021) Panamath data.

The model was fitted on Piantadosi estimation data (numbers 1-15).
Here we test whether its numerosity-comparison predictions generalise
to the Reynvoet Panamath task (numbers 10-40, 6 ratios, 3 area-
congruency conditions).  The comparison introduces zero free
parameters: accuracy is a pure prediction of the fitted attractor
model.

Evaluation modes
----------------
``simulation`` (default): trial-by-trial 2AFC. Each trial draws independent
noisy evidence for both arrays (same ``evidence_noise_std`` as Piantadosi),
runs the usual cleanup, reads off the posterior mean over integer wells for
each array, and scores the trial by comparing those two estimates (ties =
0.5 correct). Accuracies are Monte Carlo frequencies; they are then passed
through the same ``psychometric_weber`` fit as the empirical curve.

``deterministic`` (legacy / debug): for each Monte Carlo draw of the pair of
evidence values, the exact Bayesian 2AFC probability
``P(k_large > k_small | x) + 0.5·P(k_large = k_small | x)`` is computed from
the two categorical posteriors and averaged. This integrates decision
uncertainty given the posteriors but does not use a point-estimate decision
rule on each trial.

Key assumptions
---------------
1. Extrapolation: parameters fitted on n in [1, 15] are applied to
   n in [10, 40].  The encoding variance s0·n^lambda0 (combined with baseline
   s00 in variance space; see ``encoding_variance`` / ``evidence_noise_std``)
   is the component most sensitive to this.
2. Pair enumeration: we average over all integer pairs whose ratio
   falls within tolerance of each Panamath target ratio.
3. No area modelling: the model predicts purely numerical comparison.
   Mismatch with area-correlated/anticorrelated conditions is expected.
4. Duration: the Reynvoet task uses 500 ms exposure; we show
   predictions from fitted parameters at each duration in
   ``DURATIONS_TO_TEST`` (must match the Piantadosi duration sweep).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    benford_sigma, cleanup, FIXED, WELL_TYPE, BENFORD_PRIOR,
    FIXED_BETA, FIXED_KAPPA, FIXED_ETA,
    NO_ANCHOR,
    evidence_noise_std,
    psychometric_weber,
)

# ── Settings ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output/reynvoet_comparison")
DATA_PATH = Path("data/Reynvoet_Libertus.csv")
PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations.csv")

N_SAMPLES = 5000
SEED = 73
N_MIN = 1

PANAMATH_N_MIN = 10
PANAMATH_N_MAX = 40
TARGET_RATIOS = [1.11, 1.14, 1.20, 1.25, 1.50, 2.00]
RATIO_TOLERANCE = 0.03

# Extra integer wells beyond the largest Panamath stimulus. Extrapolation to
# n=40 must not be truncated to the Piantadosi fitting range (n<=15), and the
# margin keeps posterior mass from piling up against the edge of the support.
LANDSCAPE_MARGIN = 8
N_SUPPORT_MAX = PANAMATH_N_MAX + LANDSCAPE_MARGIN

DURATIONS_TO_TEST: list[int] = [40, 80, 160, 320, 640]

# Linestyles for model accuracy-vs-ratio curves (single colour, distinct dash).
# Keys must cover every duration used in the sweep; values are valid matplotlib ``linestyle`` args.
REYNVOET_MODEL_LINESTYLE: dict[int, str | tuple] = {
    40: "-",
    80: "--",
    160: "-.",
    320: ":",
    640: (0, (5, 2, 1, 2)),
}

CONGRUENCY_PREFIXES = {
    "correlated": "PAN_Areacorrelated",
    "equal": "PAN_areaequal",
    "anticorrelated": "PAN_anticorrelated",
}
RATIO_SUFFIXES = ["1.11", "1.14", "1.20", "1.25", "1.50", "2"]


# ── Data loading ──────────────────────────────────────────────────────────


def load_reynvoet_panamath() -> pd.DataFrame:
    """Per-ratio x congruency group means from the Reynvoet Panamath task.

    Returns columns: congruency, ratio, mean_acc (%), se_acc (%), n_subjects.
    """
    df = pd.read_csv(DATA_PATH, decimal=",", na_values=["#NULL!"])

    records: list[dict] = []
    for cong_label, prefix in CONGRUENCY_PREFIXES.items():
        for ratio_str in RATIO_SUFFIXES:
            col = f"{prefix}{ratio_str}"
            if col not in df.columns:
                print(f"  Warning: column {col} not found, skipping")
                continue
            values = df[col].dropna()
            records.append({
                "congruency": cong_label,
                "ratio": float(ratio_str),
                "mean_acc": values.mean() * 100,
                "se_acc": values.sem() * 100,
                "n_subjects": len(values),
            })

    for ratio_str in RATIO_SUFFIXES:
        cols = [
            f"{prefix}{ratio_str}"
            for prefix in CONGRUENCY_PREFIXES.values()
            if f"{prefix}{ratio_str}" in df.columns
        ]
        if cols:
            per_subject = df[cols].mean(axis=1).dropna()
            records.append({
                "congruency": "average",
                "ratio": float(ratio_str),
                "mean_acc": per_subject.mean() * 100,
                "se_acc": per_subject.sem() * 100,
                "n_subjects": len(per_subject),
            })

    return pd.DataFrame(records)


# ── Pair enumeration ──────────────────────────────────────────────────────


def enumerate_pairs(
    target_ratio: float,
    n_min: int = PANAMATH_N_MIN,
    n_max: int = PANAMATH_N_MAX,
    tol: float = RATIO_TOLERANCE,
) -> list[tuple[int, int]]:
    """All integer pairs (n_small, n_large) with ratio within *tol* of target."""
    pairs = []
    for n_s in range(n_min, n_max):
        for n_l in range(n_s + 1, n_max + 1):
            if abs(n_l / n_s - target_ratio) <= tol:
                pairs.append((n_s, n_l))
    return pairs


def infer_n_support_max(
    n_stim_max: int = PANAMATH_N_MAX,
    margin: int = LANDSCAPE_MARGIN,
) -> int:
    """Support upper bound for the attractor landscape.

    The landscape must cover every presented stimulus with a safety margin so
    that posterior mass does not saturate against the edge. Kept as a helper
    so tests and callers can pass a different ``n_stim_max`` (e.g. derived
    from a custom set of pairs).
    """
    if n_stim_max < PANAMATH_N_MIN:
        raise ValueError(
            f"n_stim_max={n_stim_max} below PANAMATH_N_MIN={PANAMATH_N_MIN}"
        )
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    return int(n_stim_max) + int(margin)


# ── Model prediction ──────────────────────────────────────────────────────


def _load_fitted_params(duration_ms: int) -> dict:
    """Load fitted parameters for a given duration from the sweep CSV."""
    df = pd.read_csv(PARAMS_CSV)
    row = df[df["duration_ms"] == duration_ms].iloc[0]
    s0_col = "s0" if "s0" in row.index else "s01"
    theta = {
        "sigma_c": row["sigma_c"],
        "s0": row[s0_col],
        "lambda0": row["lambda0"],
    }
    if not NO_ANCHOR:
        theta["tau_in"] = float(row["tau_in"])
    theta["beta"] = FIXED_BETA or 1.0
    theta["kappa_lse"] = FIXED_KAPPA or 1.0
    theta["eta"] = FIXED_ETA or 1.0
    return theta


def _setup_wells(
    theta: dict,
    n_min: int = N_MIN,
    n_max: int = N_SUPPORT_MAX,
) -> dict:
    """Build the attractor landscape with integer wells on ``[n_min, n_max]``.

    The support must cover every numerosity the model is asked to process
    (Reynvoet goes up to ``PANAMATH_N_MAX=40``), so ``n_max`` defaults to the
    module-level ``N_SUPPORT_MAX`` which already includes ``LANDSCAPE_MARGIN``
    slack beyond the largest stimulus.
    """
    if n_max < PANAMATH_N_MAX:
        raise ValueError(
            f"landscape n_max={n_max} is below the largest Panamath stimulus "
            f"({PANAMATH_N_MAX}); extrapolation would be truncated"
        )
    integers = torch.arange(n_min, n_max + 1, dtype=torch.float32)
    freq = torch.log10(1.0 + 1.0 / integers)

    sigma_sq, log_sigma = None, None
    if WELL_TYPE == "gaussian":
        sigma = benford_sigma(integers, theta["sigma_c"], theta["beta"])
        sigma_sq = sigma.square()
        log_sigma = sigma.log()

    inv_tau_sq = 0.0 if NO_ANCHOR else float(1.0 / theta["tau_in"] ** 2)
    return {
        "integers": integers,
        "mu": integers,
        "sigma_sq": sigma_sq,
        "log_sigma": log_sigma,
        "log_freq": freq.log() if BENFORD_PRIOR else torch.zeros_like(integers),
        "inv_tau_sq": inv_tau_sq,
    }


def _sample_evidence(n_true, theta, n_samples, generator):
    noise_std = evidence_noise_std(n_true, theta)
    return float(n_true) + noise_std * torch.randn(n_samples, generator=generator)


@torch.no_grad()
def _process(n_true, theta, wells, n_samples, generator):
    y0 = _sample_evidence(n_true, theta, n_samples, generator)
    out = cleanup(
        y0, wells["mu"], wells["log_freq"],
        theta["kappa_lse"], wells["inv_tau_sq"], theta["eta"],
        FIXED["max_steps"], FIXED["tol"],
        sigma_c=theta["sigma_c"],
        sigma_sq=wells["sigma_sq"], log_sigma=wells["log_sigma"],
    )
    return {"probs": out["probs"]}


def _p_correct_2afc(probs_l: torch.Tensor, probs_s: torch.Tensor) -> torch.Tensor:
    """Exact 2AFC probability P(k_l > k_s) + 0.5 * P(k_l = k_s).

    ``probs_*`` have shape ``[N, K]`` (per-trial posteriors over the shared
    integer support). Ties are split 50/50, matching the convention that a
    subject with identical posteriors guesses. Evaluated via cumsum to avoid
    an O(K^2) outer product per trial.
    """
    cum_s = probs_s.cumsum(dim=-1)  # P(k_s <= i | x_s)
    p_gt = (probs_l[..., 1:] * cum_s[..., :-1]).sum(dim=-1)  # sum over k_l > k_s
    p_eq = (probs_l * probs_s).sum(dim=-1)
    return p_gt + 0.5 * p_eq


def _posterior_mean(probs: torch.Tensor, integers: torch.Tensor) -> torch.Tensor:
    """Posterior expected numerosity ``sum_k k * p(k|x)`` per row of ``probs``."""
    return (probs * integers.to(probs.device)).sum(dim=-1)


def _trial_scores_point_estimate(
    e_small: torch.Tensor,
    e_large: torch.Tensor,
    n_small: int,
    n_large: int,
    *,
    tie_atol: float = 1e-4,
) -> torch.Tensor:
    """Per-trial score in ``{0, 0.5, 1}`` for choosing the larger array.

    Arrays are labelled so that ``n_small`` and ``n_large`` are the true
    counts (``n_large >= n_small``). When both true counts are equal, the
    task is at chance and the score is always ``0.5``.
    """
    if n_large < n_small:
        raise ValueError("expected n_large >= n_small")
    if n_small == n_large:
        return torch.full_like(e_small, 0.5)
    tie = torch.isclose(e_large, e_small, atol=tie_atol, rtol=0.0)
    pick_large = e_large > e_small
    return torch.where(tie, torch.full_like(e_small, 0.5), pick_large.float())


@torch.no_grad()
def trialwise_accuracy_one_pair(
    theta: dict,
    n_small: int,
    n_large: int,
    *,
    wells: dict,
    n_trials: int,
    generator: torch.Generator,
) -> float:
    """Monte Carlo accuracy (percent) for one numerosity pair after cleanup."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    integers = wells["integers"]
    out_s = _process(n_small, theta, wells, n_trials, generator)
    out_l = _process(n_large, theta, wells, n_trials, generator)
    e_s = _posterior_mean(out_s["probs"], integers)
    e_l = _posterior_mean(out_l["probs"], integers)
    scores = _trial_scores_point_estimate(e_s, e_l, n_small, n_large)
    return float(scores.mean().item() * 100.0)


@torch.no_grad()
def simulate_reynvoet_2afc(
    theta: dict,
    *,
    target_ratios: list[float] | None = None,
    n_trials: int = N_SAMPLES,
    seed: int = SEED,
    n_min_support: int = N_MIN,
    n_max_landscape: int | None = None,
) -> pd.DataFrame:
    """Trial-by-trial 2AFC simulation averaged over Panamath-like ratio bins.

    For each target ratio, all integer pairs
    ``(n_small, n_large)`` in ``[PANAMATH_N_MIN, PANAMATH_N_MAX]`` whose
    ratio matches within ``RATIO_TOLERANCE`` are evaluated; the reported
    accuracy is the mean of per-pair accuracies (same aggregation as
    :func:`predict_reynvoet`).

    Parameters
    ----------
    theta
        Fitted attractor / encoding parameters (``sigma_c``, ``s0``, ``lambda0``,
        etc.). No Reynvoet-specific parameters are added.
    target_ratios
        Defaults to ``TARGET_RATIOS``.
    n_trials
        Independent 2AFC trials per pair (each trial resamples both arrays).
    seed
        RNG seed for reproducibility only.
    n_max_landscape
        Upper integer well bound; defaults to ``infer_n_support_max(PANAMATH_N_MAX)``.
    """
    ratios = list(TARGET_RATIOS if target_ratios is None else target_ratios)
    n_max = (
        int(n_max_landscape)
        if n_max_landscape is not None
        else infer_n_support_max(PANAMATH_N_MAX, LANDSCAPE_MARGIN)
    )
    wells = _setup_wells(theta, n_min=n_min_support, n_max=n_max)
    g = torch.Generator().manual_seed(int(seed))

    records: list[dict] = []
    for target_ratio in ratios:
        pairs = enumerate_pairs(target_ratio)
        if not pairs:
            print(f"  No valid pairs for ratio {target_ratio}, skipping")
            continue

        pair_accs = []
        for n_s, n_l in pairs:
            acc = trialwise_accuracy_one_pair(
                theta, n_s, n_l, wells=wells, n_trials=n_trials, generator=g
            )
            pair_accs.append(acc)

        records.append({
            "ratio": target_ratio,
            "accuracy": float(np.mean(pair_accs)),
            "accuracy_std": float(np.std(pair_accs)) if len(pair_accs) > 1 else 0.0,
            "n_pairs": len(pairs),
        })

    return pd.DataFrame(records)


def predict_reynvoet(
    theta: dict,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Legacy curve: mean posterior 2AFC probability per ratio (not point decisions).

    For each draw of noisy evidence for both arrays, this uses the exact
    ``P(k_large > k_small | x) + 0.5 P(= | x)`` from the two categorical
    posteriors, then averages over draws. For trialwise point-estimate
    decisions, use :func:`simulate_reynvoet_2afc`.
    """
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(seed)

    records: list[dict] = []
    for target_ratio in TARGET_RATIOS:
        pairs = enumerate_pairs(target_ratio)
        if not pairs:
            print(f"  No valid pairs for ratio {target_ratio}, skipping")
            continue

        pair_accs = []
        for n_s, n_l in pairs:
            out_s = _process(n_s, theta, wells, n_samples, g)
            out_l = _process(n_l, theta, wells, n_samples, g)
            p_correct = _p_correct_2afc(out_l["probs"], out_s["probs"])
            acc = p_correct.mean().item() * 100
            pair_accs.append(acc)

        records.append({
            "ratio": target_ratio,
            "accuracy": np.mean(pair_accs),
            "accuracy_std": np.std(pair_accs) if len(pair_accs) > 1 else 0.0,
            "n_pairs": len(pairs),
        })

    return pd.DataFrame(records)


# ── Weber fraction fitting ───────────────────────────────────────────────


def fit_weber_fraction(ratios: np.ndarray, accuracies: np.ndarray) -> float:
    try:
        popt, _ = curve_fit(
            psychometric_weber, ratios, accuracies,
            p0=[0.20], bounds=(0.01, 3.0),
        )
        return float(popt[0])
    except RuntimeError:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=("simulation", "deterministic"),
        default="simulation",
        help="simulation: trialwise posterior-mean decisions; deterministic: "
        "mean exact posterior 2AFC probability per evidence draw (legacy).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=N_SAMPLES,
        metavar="N",
        help="Trials per numerosity pair (simulation) or evidence samples per "
        "pair side (deterministic legacy).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="RNG seed (reproducibility only).",
    )
    p.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also run the posterior-2AFC curve and add Weber rows for comparison.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Reynvoet Panamath vs. Model Weber Predictions")
    print("=" * 60)
    print(f"  Evaluation method: {args.method}")
    print(
        f"  Stimulus range: n in [{PANAMATH_N_MIN}, {PANAMATH_N_MAX}]"
        f"   |   Landscape support: n in [{N_MIN}, {N_SUPPORT_MAX}]"
        f"  (margin = {LANDSCAPE_MARGIN})"
    )

    # ── 1. Empirical data ────────────────────────────────────────────────
    print("\nLoading Reynvoet Panamath data ...", flush=True)
    empirical = load_reynvoet_panamath()
    print(empirical.to_string(index=False), flush=True)

    # ── 2. Pair counts ───────────────────────────────────────────────────
    print("\nPair counts per ratio:")
    for r in TARGET_RATIOS:
        pairs = enumerate_pairs(r)
        tag = f"  [{pairs[0]}, ..., {pairs[-1]}]" if pairs else ""
        print(f"  ratio {r:.2f}: {len(pairs)} pairs{tag}")

    # ── 3. Model predictions per duration ────────────────────────────────
    model_preds: dict[int, pd.DataFrame] = {}
    model_preds_legacy: dict[int, pd.DataFrame] = {}
    for dur in tqdm(
        DURATIONS_TO_TEST,
        desc="Reynvoet model curves",
        unit="duration",
    ):
        print(f"\nPredicting with {dur} ms parameters ...", flush=True)
        theta = _load_fitted_params(dur)
        for k, v in theta.items():
            print(f"  {k:>10s} = {v:.4f}")
        if args.method == "simulation":
            pred = simulate_reynvoet_2afc(
                theta, n_trials=args.n_trials, seed=args.seed
            )
        else:
            pred = predict_reynvoet(
                theta, n_samples=args.n_trials, seed=args.seed
            )
        model_preds[dur] = pred
        print(pred.to_string(index=False), flush=True)

        if args.compare_legacy and args.method == "simulation":
            leg = predict_reynvoet(
                theta, n_samples=args.n_trials, seed=args.seed
            )
            model_preds_legacy[dur] = leg
            print("  Legacy posterior-2AFC curve:", flush=True)
            print(leg.to_string(index=False), flush=True)

    curve_rows: list[dict] = []
    for dur, pred in sorted(model_preds.items()):
        tag = "simulation" if args.method == "simulation" else "posterior_2afc"
        for _, row in pred.iterrows():
            curve_rows.append({
                "duration_ms": dur,
                "eval_method": tag,
                "ratio": row["ratio"],
                "accuracy_pct": row["accuracy"],
                "accuracy_std_pair": row["accuracy_std"],
                "n_pairs": row["n_pairs"],
            })
    if args.compare_legacy and args.method == "simulation":
        for dur, pred in sorted(model_preds_legacy.items()):
            for _, row in pred.iterrows():
                curve_rows.append({
                    "duration_ms": dur,
                    "eval_method": "posterior_2afc",
                    "ratio": row["ratio"],
                    "accuracy_pct": row["accuracy"],
                    "accuracy_std_pair": row["accuracy_std"],
                    "n_pairs": row["n_pairs"],
                })
    curve_path = OUTPUT_DIR / "reynvoet_model_curve.csv"
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    print(f"\nSaved {curve_path}")

    # ── 4. Weber fractions ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Weber fractions")
    print("=" * 60)

    weber_rows: list[dict] = []
    for cong in ["correlated", "equal", "anticorrelated", "average"]:
        sub = empirical[empirical["congruency"] == cong].sort_values("ratio")
        w = fit_weber_fraction(sub["ratio"].values, sub["mean_acc"].values)
        weber_rows.append({"source": f"Empirical ({cong})", "weber_fraction": w})
        print(f"  Empirical {cong:15s}: w = {w:.4f}")

    mod_label = "sim" if args.method == "simulation" else "post2afc"
    for dur, pred in sorted(model_preds.items()):
        w = fit_weber_fraction(pred["ratio"].values, pred["accuracy"].values)
        weber_rows.append({
            "source": f"Model {mod_label} ({dur} ms)",
            "weber_fraction": w,
        })
        print(f"  Model {mod_label:8s} {dur:>4d} ms : w = {w:.4f}")

    if args.compare_legacy and args.method == "simulation":
        for dur, pred in sorted(model_preds_legacy.items()):
            w = fit_weber_fraction(pred["ratio"].values, pred["accuracy"].values)
            weber_rows.append({
                "source": f"Model post2afc-legacy ({dur} ms)",
                "weber_fraction": w,
            })
            print(f"  Model post2afc leg {dur:>4d} ms : w = {w:.4f}")

    weber_df = pd.DataFrame(weber_rows)

    # ── 5. Plot ──────────────────────────────────────────────────────────
    from scripts.plots import plot_reynvoet_comparison

    plot_label = (
        "Model (sim, {dur} ms)"
        if args.method == "simulation"
        else "Model (post2afc, {dur} ms)"
    )
    plot_reynvoet_comparison(
        empirical,
        model_preds,
        weber_df,
        OUTPUT_DIR,
        model_curve_label_fmt=plot_label,
    )

    csv_path = OUTPUT_DIR / "reynvoet_comparison.csv"
    weber_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
