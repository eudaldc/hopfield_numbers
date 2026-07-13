"""Transition-to-density analysis for the Poisson attractor model.

For each fitted exposure condition (40-640 ms) we:

  1. Simulate M Monte-Carlo trials of the Hopfield cleanup at every
     integer n in [1, N_MAX_ANALYSIS] with the fitted parameters.
  2. Compute per-n metrics: mean posterior entropy, effective number
     of states, probability of the mode, empirical Weber fraction
     w_eff(n) = std(n_hat)/n, and an analytical well d'(n).
  3. Locate four critical numerosities at which discrete counting is
     predicted to break down:
         (a) d'(n) < 1             (closed form  n = 1/c)
         (b) H_mean(n) > log(3)
         (c) p_max(n) < 0.5
         (d) BC(n, n+1) > 0.5
     where BC is the Bhattacharyya coefficient between the
     marginal decision distributions at true n and n+1.
  4. Validate the 1/sqrt(c*n) prediction on Reynvoet et al.'s
     10-40 numerosity range (no refit) via AIC comparison of a
     constant-Weber model and a w = a/sqrt(n) model.
  5. Produce a 4-panel publication figure and persist all metrics.

Run from repo root:  .venv/bin/python -m scripts.density_transition
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.fit_helpers import (
    BENFORD_PRIOR,
    DEVICE,
    FIXED,
    FIXED_KAPPA,
    FIXED_ETA,
    NO_ANCHOR,
    WELL_TYPE,
    benford_sigma,
    cleanup,
    evidence_noise_std,
)

# ── Configuration ───────────────────────────────────────────────────────

SUMMARY_CSV = Path("output/piantadosi_fit/summary_all_durations.csv")
OUT_DIR = Path("output/density_transition")

N_MIN_ANALYSIS = 1
N_MAX_ANALYSIS = 50
SUPPORT_PADDING = 5
N_SUPPORT_MAX = N_MAX_ANALYSIS + SUPPORT_PADDING
M_TRIALS = 10_000
REFERENCE_DURATION = 640
SEED = 73

ENTROPY_THRESHOLD = np.log(3.0)
PMAX_THRESHOLD = 0.5
BC_THRESHOLD = 0.5
DPRIME_THRESHOLD = 1.0

# Empirical density-transition ranges reported in the literature.
# Anobile, Cicchini & Burr (2014, Psychological Science) isolate a density
# regime at ~0.25 dots/deg^2 in central vision with 9 deg-radius stimuli
# (area ~ pi*9^2 = 254 deg^2), giving n_width ~ 60. Earlier reports (Ross
# & Burr, 2012; Anobile et al. 2015) place the breakpoint around 20-40
# depending on stimulus size.  Durgin (1995, 2008) argues for ~20.
ANOBILE_N_RANGE = (30.0, 60.0)
DURGIN_N = 20.0

DURATIONS = (40, 80, 160, 320, 640)


# ── Parameter loading ───────────────────────────────────────────────────


def load_fitted_params(duration_ms: int) -> dict:
    df = pd.read_csv(SUMMARY_CSV)
    row = df[df["duration_ms"] == duration_ms].iloc[0]
    s0_col = "s0" if "s0" in row.index else "s01"
    theta = {
        "sigma_c": float(row["sigma_c"]),
        "s0": float(row[s0_col]),
        "lambda0": float(row["lambda0"]),
        "t0": float(row["t0"]),
    }
    if "tau_in" in row.index and pd.notna(row["tau_in"]):
        theta["tau_in"] = float(row["tau_in"])
    elif NO_ANCHOR:
        theta["tau_in"] = float("inf")
    else:
        raise KeyError("fitted summary row missing tau_in (required when no_anchor is false)")
    if "kappa" in row.index:
        theta["kappa"] = float(row["kappa"])
    if "c_steps" in row.index:
        theta["c_steps"] = float(row["c_steps"])
    theta["kappa_lse"] = FIXED_KAPPA if FIXED_KAPPA is not None else 1.0
    theta["eta"] = FIXED_ETA if FIXED_ETA is not None else 1.0
    return theta


# ── Simulation with extended support ────────────────────────────────────


def _build_wells(theta: dict) -> dict[str, torch.Tensor]:
    integers = torch.arange(
        N_MIN_ANALYSIS,
        N_SUPPORT_MAX + 1,
        dtype=torch.float32,
        device=DEVICE,
    )
    freq = torch.log10(1.0 + 1.0 / integers)
    log_freq = freq.log() if BENFORD_PRIOR else torch.zeros_like(integers)
    sigma_sq, log_sigma = None, None
    if WELL_TYPE == "gaussian":
        sigma = benford_sigma(integers, theta["sigma_c"], theta.get("beta", 1.0))
        sigma_sq = sigma.square()
        log_sigma = sigma.log()
    return {
        "integers": integers,
        "mu": integers,
        "log_freq": log_freq,
        "sigma_sq": sigma_sq,
        "log_sigma": log_sigma,
    }


@torch.no_grad()
def simulate_extended(
    theta: dict,
    n_values: Iterable[int],
    m_trials: int = M_TRIALS,
    seed: int = SEED,
) -> dict:
    """Run the cleanup model M trials per n, returning posteriors and decodings.

    Shapes:
      posteriors : (len(n_values), m_trials, support_size)
      n_hat      : (len(n_values), m_trials)   MAP integer estimates
      expected_n : (len(n_values), m_trials)   posterior mean (real-valued)
      n_steps    : (len(n_values), m_trials)
    """
    n_values = list(n_values)
    wells = _build_wells(theta)
    integers = wells["integers"]
    inv_tau_sq = 0.0 if NO_ANCHOR else 1.0 / theta["tau_in"] ** 2
    kappa = theta["kappa_lse"]
    eta = theta["eta"]

    g = torch.Generator().manual_seed(seed)
    y0_parts = []
    for n in n_values:
        std = evidence_noise_std(n, theta)
        y0_n = float(n) + std * torch.randn(m_trials, generator=g)
        y0_parts.append(y0_n)
    y0_all = torch.cat(y0_parts).to(DEVICE)

    out = cleanup(
        y0_all,
        wells["mu"],
        wells["log_freq"],
        kappa,
        inv_tau_sq,
        eta,
        FIXED["max_steps"],
        FIXED["tol"],
        sigma_c=theta["sigma_c"],
        v_min=theta.get("v_min", 0.0),
        sigma_sq=wells["sigma_sq"],
        log_sigma=wells["log_sigma"],
    )

    K = integers.numel()
    probs = out["probs"].view(len(n_values), m_trials, K)
    n_hat_idx = out["n_hat_idx"].view(len(n_values), m_trials)
    n_hat = integers[n_hat_idx]
    expected_n = (probs * integers).sum(dim=-1)
    n_steps = out["n_steps"].float().view(len(n_values), m_trials)

    return {
        "n_values": np.asarray(n_values, dtype=int),
        "integers": integers.cpu().numpy(),
        "posteriors": probs.cpu().numpy(),
        "n_hat": n_hat.cpu().numpy(),
        "expected_n": expected_n.cpu().numpy(),
        "n_steps": n_steps.cpu().numpy(),
    }


# ── Metrics ─────────────────────────────────────────────────────────────


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)))


def compute_metrics(sim: dict, c: float) -> pd.DataFrame:
    """Per-n summary metrics, averaged across trials."""
    probs = sim["posteriors"]       # (N, M, K)
    n_hat = sim["n_hat"]            # (N, M)
    exp_n = sim["expected_n"]       # (N, M)
    n_vals = sim["n_values"]
    integers = sim["integers"]

    H = -np.sum(probs * np.log(probs + 1e-12), axis=-1)  # (N, M)
    H_mean = H.mean(axis=1)
    n_eff = np.exp(H).mean(axis=1)
    p_max = probs.max(axis=-1).mean(axis=1)

    std_nhat = n_hat.std(axis=1)
    std_exp = exp_n.std(axis=1)
    bias_nhat = n_hat.mean(axis=1) - n_vals
    bias_exp = exp_n.mean(axis=1) - n_vals

    w_eff_map = std_nhat / np.maximum(n_vals, 1e-12)
    w_eff_mean = std_exp / np.maximum(n_vals, 1e-12)
    w_theory = 1.0 / np.sqrt(c * np.maximum(n_vals, 1e-12))
    d_prime = 1.0 / np.sqrt(c * np.maximum(n_vals, 1e-12))

    # Marginal p(n_hat | true_n) = empirical distribution of MAP estimates.
    # Then BC(n, n+1) uses those marginals on the shared integer support.
    K = integers.size
    N = n_vals.size
    marginals = np.zeros((N, K))
    for i in range(N):
        counts = np.bincount(
            (n_hat[i] - int(integers[0])).astype(int),
            minlength=K,
        )[:K]
        marginals[i] = counts / counts.sum()

    bc_adjacent = np.full(N, np.nan)
    for i in range(N - 1):
        bc_adjacent[i] = _bhattacharyya(marginals[i], marginals[i + 1])

    # Analytical BC between two Poisson wells with means n, n+1 (on x-axis)
    # after the c-scaling (X/c ~ Poisson(n/c)).  Closed form:
    #     BC = exp(-(sqrt(lam1) - sqrt(lam2))**2 / 2)
    # with lam_i = n_i / c.
    lam = n_vals / c
    lam_next = (n_vals + 1) / c
    bc_theory = np.exp(-0.5 * (np.sqrt(lam_next) - np.sqrt(lam)) ** 2)

    return pd.DataFrame({
        "n": n_vals,
        "H_mean": H_mean,
        "N_eff": n_eff,
        "p_max": p_max,
        "std_nhat": std_nhat,
        "std_expected": std_exp,
        "bias_map": bias_nhat,
        "bias_mean": bias_exp,
        "w_eff_map": w_eff_map,
        "w_eff_mean": w_eff_mean,
        "w_theory": w_theory,
        "d_prime": d_prime,
        "BC_adjacent": bc_adjacent,
        "BC_theory": bc_theory,
    })


def first_n_where(values: np.ndarray, n_vals: np.ndarray, cond: np.ndarray) -> float:
    """Smallest n in n_vals such that cond holds; NaN if never."""
    idx = np.where(cond)[0]
    if idx.size == 0:
        return float("nan")
    return float(n_vals[idx[0]])


def critical_n_from_metrics(metrics: pd.DataFrame, c: float) -> dict:
    n = metrics["n"].values
    return {
        "c": c,
        "n_width_analytical": 1.0 / c,
        "n_width_dprime": first_n_where(
            metrics["d_prime"].values, n, metrics["d_prime"].values < DPRIME_THRESHOLD
        ),
        "n_threshold_entropy": first_n_where(
            metrics["H_mean"].values, n, metrics["H_mean"].values > ENTROPY_THRESHOLD
        ),
        "n_threshold_pmax": first_n_where(
            metrics["p_max"].values, n, metrics["p_max"].values < PMAX_THRESHOLD
        ),
        "n_threshold_bc_behav": first_n_where(
            metrics["BC_adjacent"].values,
            n,
            metrics["BC_adjacent"].values > BC_THRESHOLD,
        ),
        "n_threshold_bc_theory": first_n_where(
            metrics["BC_theory"].values,
            n,
            metrics["BC_theory"].values > BC_THRESHOLD,
        ),
    }


# ── Reynvoet validation (10 - 40) ───────────────────────────────────────


def reynvoet_weber_validation(theta: dict, c: float) -> dict:
    """Simulate n in [10, 40] and fit w_eff with two models."""
    sim = simulate_extended(
        theta, n_values=range(10, 41), m_trials=M_TRIALS, seed=SEED + 1
    )
    metrics = compute_metrics(sim, c=c)
    n = metrics["n"].values.astype(float)
    w = metrics["w_eff_map"].values

    # Model 1: constant Weber, w(n) = w0
    w0_const = np.mean(w)
    rss_const = float(np.sum((w - w0_const) ** 2))

    # Model 2: w(n) = a / sqrt(n).  LS in a gives a = sum(w/sqrt(n)) / sum(1/n).
    inv_sqrt_n = 1.0 / np.sqrt(n)
    a_hat = float(np.sum(w * inv_sqrt_n) / np.sum(inv_sqrt_n ** 2))
    rss_sqrt = float(np.sum((w - a_hat * inv_sqrt_n) ** 2))

    k_obs = len(n)
    # Gaussian profile-likelihood AIC: AIC = k_obs * log(RSS/k) + 2*p + const
    def _aic(rss: float, p: int) -> float:
        return k_obs * np.log(rss / k_obs + 1e-30) + 2 * p

    aic_const = _aic(rss_const, p=1)
    aic_sqrt = _aic(rss_sqrt, p=1)

    theory_a = 1.0 / np.sqrt(c)
    w_theory_vals = theory_a / np.sqrt(n)
    rss_theory = float(np.sum((w - w_theory_vals) ** 2))

    return {
        "n": n,
        "w_eff": w,
        "w0_const": w0_const,
        "a_sqrt_fit": a_hat,
        "a_sqrt_theory": theory_a,
        "rss_const": rss_const,
        "rss_sqrt": rss_sqrt,
        "rss_theory": rss_theory,
        "aic_const": aic_const,
        "aic_sqrt": aic_sqrt,
        "delta_aic": aic_const - aic_sqrt,
    }


# ── Anobile digitised range ─────────────────────────────────────────────


def anobile_weber_reference(n: np.ndarray) -> np.ndarray:
    """Approximate Weber fractions digitised from Anobile et al. (2014, Fig. 2).

    Piecewise: ~0.25 in the numerosity regime (n <= ~30), then dropping by
    half per doubling in the density regime.  Values are order-of-magnitude
    and meant for visual comparison only; numeric values are explicitly
    flagged as such in the figure legend.
    """
    w = np.where(n <= 32, 0.25, 0.25 * (32.0 / n) ** 0.5)
    return w


# ── Figure ──────────────────────────────────────────────────────────────


STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

PALETTE_DURATION = {
    40: "#4C72B0",
    80: "#55A868",
    160: "#DD8452",
    320: "#C44E52",
    640: "#8172B3",
}


def _vline_threshold(ax, x, color, ls, label):
    if np.isfinite(x):
        ax.axvline(x, color=color, ls=ls, lw=1.6, alpha=0.9, label=label)


def make_figure(
    metrics_ref: pd.DataFrame,
    crit_ref: dict,
    sim_ref: dict,
    crit_per_dur: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    mpl.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    n = metrics_ref["n"].values
    c_ref = crit_ref["c"]

    # ── Panel A: diagnostics vs n ────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(n, metrics_ref["H_mean"], "-", color="#1f77b4", lw=2.2, label="H (nats)")
    ax.plot(n, metrics_ref["d_prime"], "-", color="#d62728", lw=2.2, label="d'(n)")
    ax.plot(n, metrics_ref["p_max"], "-", color="#2ca02c", lw=2.2, label=r"$p_{\max}$")
    ax.plot(
        n, np.log(metrics_ref["N_eff"]), ":", color="#1f77b4", lw=1.5, alpha=0.7,
        label=r"$\log N_{\mathrm{eff}}$",
    )
    ax.axhline(ENTROPY_THRESHOLD, color="#1f77b4", ls=":", lw=1, alpha=0.4)
    ax.axhline(DPRIME_THRESHOLD, color="#d62728", ls=":", lw=1, alpha=0.4)
    ax.axhline(PMAX_THRESHOLD, color="#2ca02c", ls=":", lw=1, alpha=0.4)

    _vline_threshold(ax, crit_ref["n_width_analytical"], "0.2", "-",
                     rf"$1/c = {crit_ref['n_width_analytical']:.1f}$")
    _vline_threshold(ax, crit_ref["n_width_dprime"], "#d62728", "--",
                     rf"$d'<1: n={crit_ref['n_width_dprime']:.0f}$")
    _vline_threshold(ax, crit_ref["n_threshold_entropy"], "#1f77b4", "--",
                     rf"$H>\log 3: n={crit_ref['n_threshold_entropy']:.0f}$")
    _vline_threshold(ax, crit_ref["n_threshold_pmax"], "#2ca02c", "--",
                     rf"$p_{{\max}}<0.5: n={crit_ref['n_threshold_pmax']:.0f}$")

    ax.set_xlabel("true numerosity n")
    ax.set_ylabel("metric value")
    ax.set_title(f"A. Diagnostics vs n ({REFERENCE_DURATION} ms)",
                 fontweight="bold", loc="left")
    ax.set_xlim(N_MIN_ANALYSIS, N_MAX_ANALYSIS)
    ax.set_ylim(0, max(3.5, float(np.max(metrics_ref["H_mean"].values)) + 0.3))
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)

    # ── Panel B: Weber fractions vs n ────────────────────────────────
    # The analytical "raw-decoder" scaling 1 / sqrt(c n) is plotted for
    # reference; the cleanup readout (MAP / posterior mean) compresses
    # behaviour onto a nearly-constant Weber fraction over the numerosity
    # regime, with a mild increase that signals the approach to the
    # density regime.
    ax = axes[0, 1]
    mask_plot = n >= 2  # drop n=1 where w_eff is zero (discrete readout)
    ax.plot(n[mask_plot], metrics_ref["w_eff_map"].values[mask_plot],
            "-", color="#1f77b4", lw=2.4, label=r"model $w_\mathrm{eff}$ (MAP)")
    ax.plot(n[mask_plot], metrics_ref["w_eff_mean"].values[mask_plot],
            "--", color="#6baed6", lw=1.6, alpha=0.95,
            label=r"model $w_\mathrm{eff}$ (posterior mean)")
    ax.plot(n, metrics_ref["w_theory"], ":", color="k", lw=1.8,
            label=r"raw-decoder  $1/\sqrt{c\,n}$")

    w_bar = float(np.mean(metrics_ref["w_eff_map"].values[mask_plot]))
    ax.axhline(w_bar, color="#8172B3", ls="-.", lw=1.4,
               label=rf"constant Weber (mean) $= {w_bar:.3f}$")

    n_fine = np.linspace(2, n.max(), 200)
    ax.plot(n_fine, anobile_weber_reference(n_fine), "-", color="#DD8452",
            lw=1.6, alpha=0.85,
            label="Anobile 2014 (approx.)")

    _vline_threshold(ax, crit_ref["n_width_analytical"], "0.3", "-",
                     rf"$1/c$")

    ax.set_xlabel("true numerosity n")
    ax.set_ylabel(r"Weber fraction  $w(n)$")
    ax.set_title("B. Effective Weber fraction", fontweight="bold", loc="left")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(2, n.max())
    ax.set_ylim(0.02, 10)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

    # ── Panel C: threshold indices vs duration ───────────────────────
    ax = axes[1, 0]
    durs = crit_per_dur["duration_ms"].values
    offsets = {
        "n_width_analytical":   (-0.18, "o", "1/c  (analytical)"),
        "n_width_dprime":       (-0.06, "s", "d' < 1"),
        "n_threshold_entropy":  ( 0.06, "D", "H > log 3"),
        "n_threshold_pmax":     ( 0.18, "^", "p_max < 0.5"),
    }
    for key, (dx, marker, label) in offsets.items():
        y = crit_per_dur[key].values
        ax.plot(
            np.arange(len(durs)) + dx, y,
            marker=marker, ls="-", ms=8, lw=1.4,
            color={
                "n_width_analytical": "k",
                "n_width_dprime": "#d62728",
                "n_threshold_entropy": "#1f77b4",
                "n_threshold_pmax": "#2ca02c",
            }[key],
            label=label,
            markerfacecolor="white" if key == "n_width_analytical" else None,
        )

    ax.axhspan(ANOBILE_N_RANGE[0], ANOBILE_N_RANGE[1],
               color="#DD8452", alpha=0.15,
               label=f"Anobile 2014 range ({ANOBILE_N_RANGE[0]:.0f}–"
                     f"{ANOBILE_N_RANGE[1]:.0f})")
    ax.axhline(DURGIN_N, color="#8172B3", ls="-.", lw=1.3,
               label=f"Durgin ≈ {DURGIN_N:.0f}")

    ax.set_xticks(range(len(durs)))
    ax.set_xticklabels([f"{d} ms" for d in durs])
    ax.set_xlabel("exposure duration")
    ax.set_ylabel("Threshold numerosity")
    ax.set_title("C. Transition across durations", fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right", ncol=1)

    # ── Panel D: example posteriors ──────────────────────────────────
    ax = axes[1, 1]
    integers = sim_ref["integers"]
    n_targets = [5, 20, 40]
    n_to_row = {int(n_): i for i, n_ in enumerate(sim_ref["n_values"])}
    styles = [("-", "#1f77b4"), ("--", "#d62728"), (":", "#2ca02c")]
    for (ls, color), nt in zip(styles, n_targets):
        if nt in n_to_row:
            p_avg = sim_ref["posteriors"][n_to_row[nt]].mean(axis=0)
            ax.plot(integers, p_avg, ls=ls, color=color, lw=2.2,
                    label=f"true n = {nt}")
    ax.set_xlabel("integer k")
    ax.set_ylabel(r"$\mathbb{E}_{x^*}[\,p(k\mid x^*)\,]$")
    ax.set_title("D. Example posteriors", fontweight="bold", loc="left")
    ax.set_xlim(0, max(n_targets) + 10)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Transition to density-based perception "
        f"(Poisson attractor, c = {c_ref:.4f}, "
        f"{REFERENCE_DURATION} ms)",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "density_transition.png"
    pdf_path = out_dir / "density_transition.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


# ── Main ────────────────────────────────────────────────────────────────


def _print_critical_table(crit_df: pd.DataFrame) -> None:
    print("\nCritical n per duration:")
    print(crit_df.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-duration critical n values (criterion a uses closed form; b/c/d
    # need a simulation sweep per duration so we run them at reduced M).
    print("Computing width and response thresholds for each exposure duration ...", flush=True)
    crit_rows: list[dict] = []
    all_metrics: dict[int, pd.DataFrame] = {}
    sim_ref: dict | None = None
    metrics_ref: pd.DataFrame | None = None
    crit_ref: dict | None = None

    for dur in DURATIONS:
        theta = load_fitted_params(dur)
        c = theta["sigma_c"]
        m = M_TRIALS if dur == REFERENCE_DURATION else max(M_TRIALS // 4, 2000)
        sim = simulate_extended(
            theta,
            n_values=range(N_MIN_ANALYSIS, N_MAX_ANALYSIS + 1),
            m_trials=m,
            seed=SEED + dur,
        )
        metrics = compute_metrics(sim, c=c)
        crit = critical_n_from_metrics(metrics, c=c)
        crit["duration_ms"] = dur
        crit_rows.append(crit)
        all_metrics[dur] = metrics
        print(
            f"  {dur:>4d} ms:  c={c:.4f}  1/c={1.0/c:5.1f}  "
            f"d'<1 @ n={crit['n_width_dprime']:5.1f}  "
            f"H>log3 @ n={crit['n_threshold_entropy']:5.1f}  "
            f"p_max<.5 @ n={crit['n_threshold_pmax']:5.1f}  "
            f"BC_behav>.5 @ n={crit['n_threshold_bc_behav']:5.1f}",
            flush=True,
        )
        if dur == REFERENCE_DURATION:
            sim_ref = sim
            metrics_ref = metrics
            crit_ref = crit

    crit_df = pd.DataFrame(crit_rows)[[
        "duration_ms", "c", "n_width_analytical", "n_width_dprime",
        "n_threshold_entropy", "n_threshold_pmax", "n_threshold_bc_behav",
        "n_threshold_bc_theory",
    ]]
    _print_critical_table(crit_df)
    crit_csv = OUT_DIR / "critical_n_summary.csv"
    crit_df.to_csv(crit_csv, index=False)
    print(f"Saved {crit_csv}")

    # Reference-duration metrics (full M_TRIALS)
    assert sim_ref is not None and metrics_ref is not None and crit_ref is not None
    metrics_csv = OUT_DIR / f"metrics_{REFERENCE_DURATION}ms.csv"
    metrics_ref.to_csv(metrics_csv, index=False)
    print(f"Saved {metrics_csv}")

    # Convergence check
    keys = ["n_width_dprime", "n_threshold_entropy", "n_threshold_pmax"]
    valid_ref = [crit_ref[k] for k in keys if np.isfinite(crit_ref[k])]
    if valid_ref:
        lo, hi = min(valid_ref), max(valid_ref)
        ratio = hi / lo if lo > 0 else float("inf")
        print(
            f"\nConvergence (d', H, p_max) at {REFERENCE_DURATION} ms: "
            f"[{lo:.1f}, {hi:.1f}]  max/min = {ratio:.2f}"
        )
        print(
            f"  analytical 1/c = {crit_ref['n_width_analytical']:.2f};  "
            f"BC_behav>.5 at n={crit_ref['n_threshold_bc_behav']:.1f};  "
            f"BC_theory>.5 at n={crit_ref['n_threshold_bc_theory']:.1f}"
        )

    # Reynvoet 10-40 validation
    theta_ref = load_fitted_params(REFERENCE_DURATION)
    c_ref = theta_ref["sigma_c"]
    print(
        f"\nReynvoet (10-40) Weber-law validation with "
        f"{REFERENCE_DURATION} ms parameters ..."
    )
    rv = reynvoet_weber_validation(theta_ref, c=c_ref)
    print(
        f"  constant-Weber mean w0 = {rv['w0_const']:.4f}  "
        f"(RSS={rv['rss_const']:.3e})"
    )
    print(
        f"  w = a / sqrt(n):  a_hat = {rv['a_sqrt_fit']:.4f}  "
        f"(RSS={rv['rss_sqrt']:.3e},  theory a=1/sqrt(c)={rv['a_sqrt_theory']:.4f})"
    )
    print(
        f"  dAIC (const - sqrt) = {rv['delta_aic']:+.2f}  "
        f"(positive favours the 1/sqrt(n) model)"
    )
    rv_df = pd.DataFrame({
        "n": rv["n"].astype(int),
        "w_eff": rv["w_eff"],
        "w_const": rv["w0_const"],
        "w_sqrt_fit": rv["a_sqrt_fit"] / np.sqrt(rv["n"]),
        "w_sqrt_theory": rv["a_sqrt_theory"] / np.sqrt(rv["n"]),
    })
    rv_csv = OUT_DIR / "reynvoet_weber_validation.csv"
    rv_df.to_csv(rv_csv, index=False)
    rv_summary = OUT_DIR / "reynvoet_weber_aic.csv"
    pd.DataFrame([{
        "w0_const": rv["w0_const"],
        "a_sqrt_fit": rv["a_sqrt_fit"],
        "a_sqrt_theory": rv["a_sqrt_theory"],
        "rss_const": rv["rss_const"],
        "rss_sqrt": rv["rss_sqrt"],
        "rss_theory": rv["rss_theory"],
        "aic_const": rv["aic_const"],
        "aic_sqrt": rv["aic_sqrt"],
        "delta_aic": rv["delta_aic"],
    }]).to_csv(rv_summary, index=False)
    print(f"Saved {rv_csv}  and  {rv_summary}")

    # Figure
    png, pdf = make_figure(
        metrics_ref=metrics_ref,
        crit_ref=crit_ref,
        sim_ref=sim_ref,
        crit_per_dur=crit_df,
        out_dir=OUT_DIR,
    )
    print(f"\nSaved {png}\nSaved {pdf}")


if __name__ == "__main__":
    main()
