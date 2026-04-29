#!/usr/bin/env python
"""Compare model predictions against symbolic number comparison data.

Children see two Arabic digits (0-9) and choose which is larger,
under speed pressure (40 s for as many items as possible).

Both digits are processed through independent attractor cleanup
channels, and the model responds based on the posterior expected
values: P(correct) = P(E[n|left] > E[n|right]) when left > right,
and vice versa.  Zero free parameters.

The model's support starts at 1 (Poisson wells and Benford prior
are undefined for 0), so pairs containing 0 are shown empirically
but excluded from the model comparison.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    benford_sigma, cleanup, FIXED, WELL_TYPE, BENFORD_PRIOR,
    FIXED_BETA, FIXED_KAPPA, FIXED_ETA,
    evidence_noise_std,
)

# ── Settings ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output/symbolic_comparison")
DATA_PATH = Path("data/processed/symbolic_comparison.parquet")
PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations_poisson.csv")

N_SAMPLES = 5000
SEED = 73
N_MIN = 1
N_SUPPORT_MAX = 18

DURATIONS_TO_TEST = [160, 320, 640]


# ── Empirical data ───────────────────────────────────────────────────────


def load_symbolic_empirical(
    grade: int | None = None,
    exclude_zero: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load symbolic comparison data and summarise.

    Returns
    -------
    by_pair : per-(smaller, larger) accuracy, RT
    by_distance : accuracy and RT by |left - right|
    by_ratio : accuracy by max/min ratio (excluding pairs with 0)
    """
    df = pd.read_parquet(DATA_PATH)
    if grade is not None:
        df = df[df["grade"] == grade]
    if exclude_zero:
        df = df[(df["left_digit"] > 0) & (df["right_digit"] > 0)]

    df["larger"] = df[["left_digit", "right_digit"]].max(axis=1)
    df["smaller"] = df[["left_digit", "right_digit"]].min(axis=1)
    df["distance"] = df["larger"] - df["smaller"]

    # ── per pair (canonicalized as smaller, larger) ───────────────────────
    by_pair = (
        df.groupby(["smaller", "larger"])
        .agg(
            n_trials=("correct", "count"),
            accuracy=("correct", "mean"),
            se=("correct", "sem"),
            mean_rt=("rt_ms", "mean"),
            se_rt=("rt_ms", "sem"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    by_pair["distance"] = by_pair["larger"] - by_pair["smaller"]
    by_pair["accuracy_pct"] = by_pair["accuracy"] * 100
    by_pair["se_pct"] = by_pair["se"] * 100
    by_pair["ratio"] = by_pair["larger"] / by_pair["smaller"].replace(0, np.nan)

    # ── by distance ───────────────────────────────────────────────────────
    by_distance = (
        df.groupby("distance")
        .agg(
            mean_acc=("correct", "mean"),
            se_acc=("correct", "sem"),
            mean_rt=("rt_ms", "mean"),
            se_rt=("rt_ms", "sem"),
            n_trials=("correct", "count"),
        )
        .reset_index()
    )
    by_distance["mean_acc"] *= 100
    by_distance["se_acc"] *= 100

    # ── by ratio (only pairs without 0) ───────────────────────────────────
    no_zero = df[(df["left_digit"] > 0) & (df["right_digit"] > 0)].copy()
    no_zero["ratio"] = (
        no_zero[["left_digit", "right_digit"]].max(axis=1)
        / no_zero[["left_digit", "right_digit"]].min(axis=1)
    )
    by_ratio = (
        no_zero.groupby("ratio")
        .agg(
            mean_acc=("correct", "mean"),
            se_acc=("correct", "sem"),
            n_trials=("correct", "count"),
        )
        .reset_index()
    )
    by_ratio["mean_acc"] *= 100
    by_ratio["se_acc"] *= 100

    return by_pair, by_distance, by_ratio


# ── Model prediction ──────────────────────────────────────────────────────


def _load_fitted_params(duration_ms: int) -> dict:
    df = pd.read_csv(PARAMS_CSV)
    row = df[df["duration_ms"] == duration_ms].iloc[0]
    theta = {
        "sigma_c": row["sigma_c"],
        "tau_in": row["tau_in"],
        "s0": row["s0"],
        "lambda0": row["lambda0"],
    }
    theta["beta"] = FIXED_BETA or 1.0
    theta["kappa"] = FIXED_KAPPA or 1.0
    theta["eta"] = FIXED_ETA or 1.0
    return theta


def _setup_wells(theta: dict) -> dict:
    integers = torch.arange(N_MIN, N_SUPPORT_MAX + 1, dtype=torch.float32)
    freq = torch.log10(1.0 + 1.0 / integers)

    sigma_sq, log_sigma = None, None
    if WELL_TYPE == "gaussian":
        sigma = benford_sigma(integers, theta["sigma_c"], theta["beta"])
        sigma_sq = sigma.square()
        log_sigma = sigma.log()

    return {
        "integers": integers,
        "mu": integers,
        "sigma_sq": sigma_sq,
        "log_sigma": log_sigma,
        "log_freq": freq.log() if BENFORD_PRIOR else torch.zeros_like(integers),
        "inv_tau_sq": 1.0 / theta["tau_in"] ** 2,
    }


def _sample_evidence(n_true, theta, n_samples, generator):
    noise_std = evidence_noise_std(n_true, theta)
    return float(n_true) + noise_std * torch.randn(n_samples, generator=generator)


@torch.no_grad()
def _process(n_true, theta, wells, n_samples, generator):
    y0 = _sample_evidence(n_true, theta, n_samples, generator)
    out = cleanup(
        y0, wells["mu"], wells["log_freq"],
        theta["kappa"], wells["inv_tau_sq"], theta["eta"],
        FIXED["max_steps"], FIXED["tol"],
        sigma_c=theta["sigma_c"],
        sigma_sq=wells["sigma_sq"], log_sigma=wells["log_sigma"],
    )
    probs = out["probs"]
    expected_n = (probs * wells["integers"]).sum(dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    return {"expected_n": expected_n, "entropy": entropy}


def predict_comparison(
    theta: dict, pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    """Predict accuracy for each (smaller, larger) pair.

    Decision rule: compare posterior expected values from two
    independent channels.
    """
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)

    cache: dict[int, dict] = {}
    for n in sorted({n for p in pairs for n in p}):
        cache[n] = _process(n, theta, wells, N_SAMPLES, g)

    records: list[dict] = []
    for n_small, n_large in pairs:
        acc = (
            (cache[n_large]["expected_n"] > cache[n_small]["expected_n"])
            .float().mean().item() * 100
        )
        se = np.sqrt(acc / 100 * (1 - acc / 100) / N_SAMPLES) * 100
        mean_entropy = (
            cache[n_small]["entropy"].mean().item()
            + cache[n_large]["entropy"].mean().item()
        ) / 2

        records.append({
            "smaller": n_small,
            "larger": n_large,
            "distance": n_large - n_small,
            "ratio": n_large / n_small,
            "accuracy": acc,
            "se": se,
            "mean_entropy": mean_entropy,
        })

    return pd.DataFrame(records)


def aggregate_model_by_distance(model_df: pd.DataFrame) -> pd.DataFrame:
    return (
        model_df.groupby("distance")
        .agg(
            mean_acc=("accuracy", "mean"),
            se_acc=("se", "mean"),
            mean_entropy=("mean_entropy", "mean"),
        )
        .reset_index()
    )


def aggregate_model_by_ratio(model_df: pd.DataFrame) -> pd.DataFrame:
    return (
        model_df.groupby("ratio")
        .agg(mean_acc=("accuracy", "mean"), se_acc=("se", "mean"))
        .reset_index()
    )


# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Symbolic Comparison: Model vs. Empirical")
    print("=" * 60)

    # ── 1. Pair list (model: digits >= 1 only) ───────────────────────────
    by_pair_nz, _, _ = load_symbolic_empirical(grade=None, exclude_zero=True)
    pairs = [
        (int(r.smaller), int(r.larger))
        for _, r in by_pair_nz.iterrows()
    ]
    print(f"\nPairs for model (no zero): {len(pairs)}")

    # ── 2. Model predictions ──────────────────────────────────────────────
    model_preds: dict[int, pd.DataFrame] = {}
    model_by_dist: dict[int, pd.DataFrame] = {}
    model_by_ratio: dict[int, pd.DataFrame] = {}

    for dur in DURATIONS_TO_TEST:
        print(f"\nPredicting with {dur} ms parameters ...", flush=True)
        theta = _load_fitted_params(dur)
        for k, v in theta.items():
            print(f"  {k:>10s} = {v:.4f}")
        pred = predict_comparison(theta, pairs)
        model_preds[dur] = pred
        model_by_dist[dur] = aggregate_model_by_distance(pred)
        model_by_ratio[dur] = aggregate_model_by_ratio(pred)
        print(pred.to_string(index=False), flush=True)

    # ── 3. Per-grade analysis ─────────────────────────────────────────────
    from scripts.plots import plot_symbolic_comparison

    for grade_label, grade_val in [("all", None), ("grade1", 1), ("grade2", 2)]:
        print(f"\n{'=' * 60}")
        print(f"  Grade: {grade_label}")
        print(f"{'=' * 60}")

        by_pair_full, by_dist_full, _ = load_symbolic_empirical(
            grade=grade_val, exclude_zero=False,
        )
        by_pair_nz, by_dist_nz, by_ratio_nz = load_symbolic_empirical(
            grade=grade_val, exclude_zero=True,
        )

        print(f"\n  Accuracy by distance (no zero):")
        print(by_dist_nz.to_string(index=False), flush=True)

        print(f"\n  Accuracy by distance (all pairs incl. zero):")
        print(by_dist_full.to_string(index=False), flush=True)

        plot_symbolic_comparison(
            by_pair_nz, by_dist_nz, by_dist_full, by_ratio_nz,
            model_preds, model_by_dist, model_by_ratio,
            OUTPUT_DIR, tag=grade_label,
        )

    # ── 4. Save summary ──────────────────────────────────────────────────
    _, by_dist_nz_all, _ = load_symbolic_empirical(
        grade=None, exclude_zero=True,
    )
    summary_rows: list[dict] = []
    for _, row in by_dist_nz_all.iterrows():
        d = int(row["distance"])
        entry = {"distance": d, "empirical": row["mean_acc"]}
        for dur in DURATIONS_TO_TEST:
            mbd = model_by_dist[dur]
            match = mbd[mbd["distance"] == d]
            entry[f"model_{dur}ms"] = (
                match["mean_acc"].values[0] if len(match) else np.nan
            )
        summary_rows.append(entry)

    csv_path = OUTPUT_DIR / "symbolic_comparison.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
