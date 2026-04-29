"""Model comparison for the Piantadosi/Cheyette duration data.

This script evaluates the main Hopfield-Poisson model against lightweight
baselines on the same summary targets used in the manuscript: mean estimate,
exact accuracy, and median response time.

Run from the project root:

    .venv/bin/python -m scripts.model_comparison
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.special import gammaln, logsumexp
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.fit_helpers as fh


DURATIONS: tuple[int, ...] = (40, 80, 160, 320, 640)
OUT_DIR = Path("output/model_comparison")
SUMMARY_CSV = Path("output/piantadosi_fit/summary_all_durations.csv")
T0_FIXED = 1.30
N_SUPPORT_MAX = fh.N_SUPPORT_MAX
SUPPORT = np.arange(fh.N_TRUE_MIN, N_SUPPORT_MAX + 1, dtype=float)
CHANNELS: tuple[str, ...] = ("mean_guess", "pct_exact", "rt")
SEED = 73
AIC_PLOT_STYLE: dict[str, object] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 14,
    "axes.titlesize": 25.2,
    "axes.labelsize": 21,
    "xtick.labelsize": 18.2,
    "ytick.labelsize": 18.2,
    "legend.fontsize": 18.2,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    k_params: int
    bounds: tuple[tuple[float, float], ...]
    predict: Callable[[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]


def empirical_for_duration(duration_ms: int) -> pd.DataFrame:
    return (
        fh.extract_empirical(duration_ms).sort_values("n_shown").reset_index(drop=True)
    )


def make_eps(n_samples: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_count = fh.N_TRUE_MAX - fh.N_TRUE_MIN + 1
    return rng.standard_normal((n_count, n_samples))


def poisson_logits(y: np.ndarray, c: float) -> np.ndarray:
    y_scaled = np.clip(y[..., None] / c, 0.1, None)
    n_scaled = SUPPORT / c
    return y_scaled * np.log(n_scaled) - n_scaled - gammaln(y_scaled + 1.0)


def benford_sigma(c_g: float, beta: float = 0.5) -> np.ndarray:
    freq = np.log10(1.0 + 1.0 / SUPPORT)
    return np.power(c_g / freq, beta)


def gaussian_logits(y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (y[..., None] - SUPPORT) / sigma
    return -0.5 * z * z - np.log(sigma)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logsumexp(logits, axis=-1, keepdims=True)
    return np.exp(shifted)


def summarize_trials(
    n_values: np.ndarray,
    probs: np.ndarray,
    kappa: float,
    t0: float = T0_FIXED,
) -> pd.DataFrame:
    expected = (probs * SUPPORT).sum(axis=-1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=-1)
    modes = SUPPORT[np.argmax(probs, axis=-1)]
    true = n_values[:, None]
    exact = modes == true
    rt = t0 + kappa * entropy
    return pd.DataFrame(
        {
            "n_shown": n_values.astype(int),
            "mean_guess": expected.mean(axis=1),
            "pct_exact": exact.mean(axis=1) * 100.0,
            "median_rt": np.median(rt, axis=1),
        }
    )


def predict_no_cleanup_poisson(
    theta: np.ndarray,
    n_values: np.ndarray,
    eps: np.ndarray,
) -> pd.DataFrame:
    c, s0, lambda0, kappa = theta
    sd = np.sqrt(0.05**2 + s0 * np.power(n_values, lambda0))
    y = n_values[:, None] + sd[:, None] * eps[: len(n_values)]
    return summarize_trials(n_values, softmax(poisson_logits(y, c)), kappa)


def predict_gaussian_benford(
    theta: np.ndarray,
    n_values: np.ndarray,
    eps: np.ndarray,
) -> pd.DataFrame:
    c_g, s0, lambda0, kappa = theta
    sd_in = np.sqrt(0.05**2 + s0 * np.power(n_values, lambda0))
    y = n_values[:, None] + sd_in[:, None] * eps[: len(n_values)]
    sigma = benford_sigma(c_g, beta=0.5)
    return summarize_trials(n_values, softmax(gaussian_logits(y, sigma)), kappa)


def predict_weber_ans(
    theta: np.ndarray,
    n_values: np.ndarray,
    eps: np.ndarray,
) -> pd.DataFrame:
    sigma0, exponent, decision_width, kappa = theta
    sd = np.maximum(1e-4, sigma0 * np.power(n_values, exponent))
    y = n_values[:, None] + sd[:, None] * eps[: len(n_values)]
    sigma = np.maximum(1e-4, decision_width * SUPPORT)
    return summarize_trials(n_values, softmax(gaussian_logits(y, sigma)), kappa)


def predict_cp_log_prior_proxy(
    theta: np.ndarray,
    n_values: np.ndarray,
    eps: np.ndarray,
) -> pd.DataFrame:
    sigma_log, alpha, readout_temp, kappa = theta
    y = np.log(n_values)[:, None] + sigma_log * eps[: len(n_values)]
    log_support = np.log(SUPPORT)
    logits = -0.5 * ((y[..., None] - log_support) / sigma_log) ** 2
    logits = logits - alpha * log_support
    logits = logits / readout_temp
    return summarize_trials(n_values, softmax(logits), kappa)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="no_cleanup_poisson",
        label="No-cleanup Poisson",
        k_params=4,
        bounds=((0.01, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 3.0)),
        predict=predict_no_cleanup_poisson,
    ),
    ModelSpec(
        key="gaussian_benford",
        label="Gaussian/Benford readout",
        k_params=4,
        bounds=((0.005, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 3.0)),
        predict=predict_gaussian_benford,
    ),
    ModelSpec(
        key="weber_ans",
        label="Weber/ANS baseline",
        k_params=4,
        bounds=((0.01, 2.0), (0.0, 1.5), (0.01, 1.0), (0.1, 3.0)),
        predict=predict_weber_ans,
    ),
    ModelSpec(
        key="cp_log_prior_proxy",
        label="C&P log-prior proxy",
        k_params=4,
        bounds=((0.02, 2.0), (-2.5, 2.5), (0.25, 4.0), (0.1, 3.0)),
        predict=predict_cp_log_prior_proxy,
    ),
)


def align_emp_model(
    emp: pd.DataFrame, model: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ns = np.intersect1d(emp["n_shown"].values, model["n_shown"].values)
    emp_a = emp[emp["n_shown"].isin(ns)].sort_values("n_shown").reset_index(drop=True)
    mod_a = (
        model[model["n_shown"].isin(ns)].sort_values("n_shown").reset_index(drop=True)
    )
    return emp_a, mod_a


def channel_arrays(
    emp: pd.DataFrame, model: pd.DataFrame, channel: str
) -> tuple[np.ndarray, np.ndarray]:
    emp_col = "median_rt_s" if channel == "rt" else channel
    mod_col = "median_rt" if channel == "rt" else channel
    return emp[emp_col].to_numpy(float), model[mod_col].to_numpy(float)


def normalized_loss(emp: pd.DataFrame, model: pd.DataFrame) -> float:
    emp_a, mod_a = align_emp_model(emp, model)
    loss = 0.0
    used = 0
    for channel in CHANNELS:
        y, yhat = channel_arrays(emp_a, mod_a, channel)
        scale = np.ptp(y) ** 2 + 1e-6
        loss += float(np.mean((y - yhat) ** 2) / scale)
        used += 1
    return loss / max(used, 1)


def prediction_loss_with_reference_scale(
    emp: pd.DataFrame,
    model: pd.DataFrame,
    reference_emp: pd.DataFrame,
) -> float:
    """Prediction loss for small held-out sets using training-set scale."""
    emp_a, mod_a = align_emp_model(emp, model)
    loss = 0.0
    used = 0
    for channel in CHANNELS:
        y, yhat = channel_arrays(emp_a, mod_a, channel)
        ref_col = "median_rt_s" if channel == "rt" else channel
        ref_y = reference_emp[ref_col].to_numpy(float)
        scale = np.ptp(ref_y) ** 2 + 1e-6
        loss += float(np.mean((y - yhat) ** 2) / scale)
        used += 1
    return loss / max(used, 1)


def profile_metrics(
    emp: pd.DataFrame, model: pd.DataFrame, k_params: int
) -> dict[str, float | int]:
    emp_a, mod_a = align_emp_model(emp, model)
    metrics: dict[str, float | int] = {}
    nll_total = 0.0
    n_total = 0
    for channel in CHANNELS:
        y, yhat = channel_arrays(emp_a, mod_a, channel)
        rss = float(np.sum((y - yhat) ** 2))
        n = int(y.size)
        sigma2 = rss / n if n else np.nan
        metrics[f"RMSE_{channel}"] = float(np.sqrt(sigma2))
        metrics[f"MAE_{channel}"] = float(np.mean(np.abs(y - yhat)))
        metrics[f"R2_{channel}"] = float(
            1.0 - rss / (np.sum((y - y.mean()) ** 2) + 1e-12)
        )
        if sigma2 > 0 and np.isfinite(sigma2):
            nll_total += 0.5 * n * (np.log(2.0 * np.pi) + np.log(sigma2) + 1.0)
            n_total += n
    metrics["logL"] = -float(nll_total)
    metrics["AIC"] = float(2 * k_params + 2 * nll_total)
    metrics["BIC"] = float(k_params * np.log(n_total) + 2 * nll_total)
    metrics["n_data"] = int(n_total)
    metrics["k_params"] = int(k_params)
    return metrics


def fit_spec(
    spec: ModelSpec,
    emp: pd.DataFrame,
    eps: np.ndarray,
    *,
    maxiter: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    n_values = emp["n_shown"].to_numpy(float)

    def objective(theta: np.ndarray) -> float:
        if any(
            theta[i] < lo or theta[i] > hi for i, (lo, hi) in enumerate(spec.bounds)
        ):
            return 1e9
        pred = spec.predict(theta, n_values, eps)
        value = normalized_loss(emp, pred)
        return 1e9 if not np.isfinite(value) else value

    result_de = differential_evolution(
        objective,
        bounds=spec.bounds,
        maxiter=maxiter,
        popsize=6,
        tol=1e-4,
        polish=False,
        seed=seed,
        updating="immediate",
        workers=1,
    )
    result = minimize(
        objective,
        result_de.x,
        method="Nelder-Mead",
        options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-5},
    )
    theta = result.x if result.fun <= result_de.fun else result_de.x
    loss = float(min(result.fun, result_de.fun))
    return theta, loss


def refit_spec_local(
    spec: ModelSpec,
    emp: pd.DataFrame,
    eps: np.ndarray,
    start: np.ndarray,
    *,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    n_values = emp["n_shown"].to_numpy(float)
    lo = np.asarray([b[0] for b in spec.bounds], dtype=float)
    hi = np.asarray([b[1] for b in spec.bounds], dtype=float)

    def objective(z: np.ndarray) -> float:
        theta = np.clip(z, lo, hi)
        pred = spec.predict(theta, n_values, eps)
        value = normalized_loss(emp, pred)
        return 1e9 if not np.isfinite(value) else value

    result = minimize(
        objective,
        np.clip(start, lo, hi),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-5},
    )
    theta = np.clip(result.x, lo, hi)
    return theta, float(objective(theta))


def load_hopfield_theta(duration_ms: int, summary_csv: Path = SUMMARY_CSV) -> dict:
    row = pd.read_csv(summary_csv).query("duration_ms == @duration_ms").iloc[0]
    return {
        "sigma_c": float(row["sigma_c"]),
        "s0": float(row["s0"]),
        "lambda0": float(row["lambda0"]),
        "kappa": float(row["kappa"]),
        "t0": float(row["t0"]),
        "kappa_lse": fh.FIXED_KAPPA or 1.0,
        "eta": fh.FIXED_ETA or 1.0,
        "tau_in": float("inf"),
    }


def predict_hopfield_from_summary(
    duration_ms: int, n_samples: int, seed: int = SEED
) -> pd.DataFrame:
    theta = load_hopfield_theta(duration_ms)
    return fh.simulate(theta, n_samples_per_n=n_samples, seed=seed)


def constrained_duration_prediction(
    holdout_duration: int,
    n_samples: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Predict a held-out duration with shared c/kappa and exposure trends."""
    df = pd.read_csv(SUMMARY_CSV)
    train = df[df["duration_ms"] != holdout_duration].copy()
    stable = train[train["duration_ms"] >= 160]
    if stable.empty:
        stable = train
    log_d = np.log(train["duration_ms"].to_numpy(float))
    theta = {
        "sigma_c": float(stable["sigma_c"].mean()),
        "kappa": float(stable["kappa"].mean()),
        "t0": T0_FIXED,
        "kappa_lse": fh.FIXED_KAPPA or 1.0,
        "eta": fh.FIXED_ETA or 1.0,
        "tau_in": float("inf"),
    }
    theta["s0"] = float(
        np.exp(
            np.polyval(
                np.polyfit(log_d, np.log(train["s0"]), deg=1), np.log(holdout_duration)
            )
        )
    )
    theta["lambda0"] = float(
        np.polyval(np.polyfit(log_d, train["lambda0"], deg=1), np.log(holdout_duration))
    )
    return fh.simulate(theta, n_samples_per_n=n_samples, seed=seed)


def leave_one_numerosity_loss(
    spec: ModelSpec,
    duration_ms: int,
    eps: np.ndarray,
    start_theta: np.ndarray,
    *,
    maxiter: int,
) -> float:
    emp = empirical_for_duration(duration_ms)
    losses = []
    for n_holdout in emp["n_shown"].astype(int):
        train = emp[emp["n_shown"] != n_holdout].reset_index(drop=True)
        held = emp[emp["n_shown"] == n_holdout].reset_index(drop=True)
        theta, _ = refit_spec_local(spec, train, eps, start_theta, maxiter=maxiter)
        pred_all = spec.predict(theta, emp["n_shown"].to_numpy(float), eps)
        pred = pred_all[pred_all["n_shown"] == n_holdout]
        losses.append(prediction_loss_with_reference_scale(held, pred, emp))
    return float(np.mean(losses))


def leave_one_duration_loss(
    spec: ModelSpec,
    holdout_duration: int,
    fitted_by_duration: dict[int, np.ndarray],
    eps: np.ndarray,
) -> float:
    train_durs = np.asarray(
        [d for d in DURATIONS if d != holdout_duration], dtype=float
    )
    theta_mat = np.vstack([fitted_by_duration[int(d)] for d in train_durs])
    x = np.log(train_durs)
    pred_theta = []
    for col, bounds in enumerate(spec.bounds):
        vals = theta_mat[:, col]
        if bounds[0] > 0 and np.all(vals > 0):
            y = np.log(vals)
            value = float(
                np.exp(np.polyval(np.polyfit(x, y, deg=1), np.log(holdout_duration)))
            )
        else:
            value = float(
                np.polyval(np.polyfit(x, vals, deg=1), np.log(holdout_duration))
            )
        pred_theta.append(float(np.clip(value, bounds[0], bounds[1])))
    emp = empirical_for_duration(holdout_duration)
    pred = spec.predict(np.asarray(pred_theta), emp["n_shown"].to_numpy(float), eps)
    return normalized_loss(emp, pred)


def evaluate_models(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eps = make_eps(args.samples, seed=args.seed)
    fit_rows: list[dict] = []
    cv_rows: list[dict] = []
    fitted: dict[str, dict[int, np.ndarray]] = {spec.key: {} for spec in MODEL_SPECS}

    hopfield_rows = []
    for duration_ms in tqdm(DURATIONS, desc="Hopfield in-sample", unit="duration"):
        emp = empirical_for_duration(duration_ms)
        model = predict_hopfield_from_summary(
            duration_ms, n_samples=args.samples, seed=args.seed
        )
        metrics = profile_metrics(emp, model, k_params=4)
        fit_rows.append(
            {
                "model": "Hopfield-Poisson",
                "model_key": "hopfield_poisson",
                "duration_ms": duration_ms,
                **metrics,
            }
        )
        hopfield_rows.append(
            {"duration_ms": duration_ms, "loss": normalized_loss(emp, model)}
        )

    for spec in MODEL_SPECS:
        for duration_ms in tqdm(DURATIONS, desc=f"Fit {spec.label}", unit="duration"):
            emp = empirical_for_duration(duration_ms)
            theta, loss = fit_spec(
                spec,
                emp,
                eps,
                maxiter=args.maxiter,
                seed=args.seed + duration_ms,
            )
            fitted[spec.key][duration_ms] = theta
            pred = spec.predict(theta, emp["n_shown"].to_numpy(float), eps)
            metrics = profile_metrics(emp, pred, k_params=spec.k_params)
            fit_rows.append(
                {
                    "model": spec.label,
                    "model_key": spec.key,
                    "duration_ms": duration_ms,
                    "fit_loss": loss,
                    **{f"theta_{i + 1}": v for i, v in enumerate(theta)},
                    **metrics,
                }
            )

    for spec in MODEL_SPECS:
        lono_losses = []
        lod_losses = []
        for duration_ms in tqdm(DURATIONS, desc=f"CV {spec.label}", unit="duration"):
            lono_losses.append(
                leave_one_numerosity_loss(
                    spec,
                    duration_ms,
                    eps,
                    fitted[spec.key][duration_ms],
                    maxiter=max(6, args.cv_maxiter),
                )
            )
            lod_losses.append(
                leave_one_duration_loss(spec, duration_ms, fitted[spec.key], eps)
            )
        cv_rows.append(
            {
                "model": spec.label,
                "model_key": spec.key,
                "lono_loss": float(np.mean(lono_losses)),
                "lodo_loss": float(np.mean(lod_losses)),
                "cv_note": "refit summaries for LONO; log-duration extrapolation for LODO",
            }
        )

    hop_lono = float(np.mean([r["loss"] for r in hopfield_rows]))
    hop_lodo = []
    for duration_ms in tqdm(DURATIONS, desc="CV constrained Hopfield", unit="duration"):
        emp = empirical_for_duration(duration_ms)
        pred = constrained_duration_prediction(
            duration_ms, n_samples=args.samples, seed=args.seed
        )
        hop_lodo.append(normalized_loss(emp, pred))
    cv_rows.insert(
        0,
        {
            "model": "Hopfield-Poisson",
            "model_key": "hopfield_poisson",
            "lono_loss": hop_lono,
            "lodo_loss": float(np.mean(hop_lodo)),
            "cv_note": "published per-duration fit for LONO; shared c/kappa trend for LODO",
        },
    )

    fit_df = pd.DataFrame(fit_rows)
    cv_df = pd.DataFrame(cv_rows)
    summary = summarize_model_table(fit_df, cv_df)
    fit_df.to_csv(OUT_DIR / "per_duration_fit_metrics.csv", index=False)
    cv_df.to_csv(OUT_DIR / "cross_validation_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)
    plot_aic_by_duration(fit_df, OUT_DIR)
    return fit_df, cv_df, summary


def summarize_model_table(fit_df: pd.DataFrame, cv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, group in fit_df.groupby("model_key", sort=False):
        cv = cv_df[cv_df["model_key"] == model_key].iloc[0]
        rows.append(
            {
                "model": group["model"].iloc[0],
                "model_key": model_key,
                "k_params_per_duration": int(group["k_params"].iloc[0]),
                "AIC_mean": float(group["AIC"].mean()),
                "BIC_mean": float(group["BIC"].mean()),
                "RMSE_mean_guess_mean": float(group["RMSE_mean_guess"].mean()),
                "RMSE_pct_exact_mean": float(group["RMSE_pct_exact"].mean()),
                "RMSE_rt_mean": float(group["RMSE_rt"].mean()),
                "lono_loss": float(cv["lono_loss"]),
                "lodo_loss": float(cv["lodo_loss"]),
                "cv_note": cv["cv_note"],
            }
        )
    return pd.DataFrame(rows)


def latex_table_rows(summary: pd.DataFrame) -> str:
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row['model']} & {int(row['k_params_per_duration'])} & "
            f"{row['AIC_mean']:.1f} & {row['BIC_mean']:.1f} & "
            f"{row['lono_loss']:.3f} & {row['lodo_loss']:.3f} \\\\"
        )
    return "\n".join(lines)


def plot_aic_by_duration(
    fit_df: pd.DataFrame, out_dir: Path = OUT_DIR
) -> tuple[Path, Path]:
    """Plot per-duration AIC values without implying inferential error bars."""
    plt.rcParams.update(AIC_PLOT_STYLE)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = fit_df[["model_key", "model"]].drop_duplicates()
    durations = sorted(fit_df["duration_ms"].unique())
    plasma = plt.colormaps["plasma"]
    color_by_model = {
        "hopfield_poisson": plasma(0.91),
        "gaussian_benford": plasma(0.76),
        "no_cleanup_poisson": plasma(0.60),
        "weber_ans": plasma(0.34),
        "cp_log_prior_proxy": plasma(0.16),
    }

    fig, ax = plt.subplots(figsize=(14.0, 7.8))
    group_labels = [f"{d} ms" for d in durations] + ["Global"]
    x = np.arange(len(group_labels))
    n_models = len(models)
    width = min(0.14, 0.8 / n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    for idx, (_, row) in enumerate(models.iterrows()):
        color = color_by_model.get(row["model_key"], plasma(0.5))
        sub = (
            fit_df[fit_df["model_key"] == row["model_key"]]
            .sort_values("duration_ms")
            .reset_index(drop=True)
        )
        y_duration = sub["AIC"].to_numpy(float)
        y = np.concatenate([y_duration, [y_duration.mean()]])
        ax.bar(
            x + offsets[idx],
            y,
            width=width,
            color=color,
            label=row["model"],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("Exposure duration")
    ax.set_ylabel("AIC (summary-curve profile likelihood)")
    ax.set_title("Model comparison across exposure durations", fontweight="bold")
    ax.axvline(len(durations) - 0.5, color="0.7", lw=1.0, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    png_path = out_dir / "aic_by_duration.png"
    pdf_path = out_dir / "aic_by_duration.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--cv-maxiter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    fit_df, cv_df, summary = evaluate_models(args)
    print("\nPer-model summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nLaTeX rows:")
    print(latex_table_rows(summary))
    print(f"\nSaved comparison outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
