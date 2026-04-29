"""Generate Figure 2: model fit to Cheyette & Piantadosi (2020) data
across all five exposure durations (40, 80, 160, 320, 640 ms).

Usage:
    python -m figures.fig2_piantadosi_fit
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.fit_helpers as fh
from scripts.fit_helpers import (
    extract_empirical, simulate,
    N_TRUE_MIN, N_TRUE_MAX,
    FIXED_KAPPA, FIXED_ETA,
)

# Empirical layer for estimation and RT columns: "boxplot" (per-trial distributions)
# or "points" (per-n summary: mean guess / median RT, same spirit as Exact accuracy).
EMPIRICAL_ESTIMATION = "points"  # "boxplot" | "points"
EMPIRICAL_RT = "points"  # "boxplot" | "points"
ACC_ERROR_BARS = False  # toggle ±1 SD error bars on accuracy panel


DURATIONS = (40, 80, 160, 320, 640)
PARAM_CSV = Path("output/piantadosi_fit/summary_all_durations.csv")
RAW_CSV = Path("data/data_piantadosi.csv")
OUT_DIR = Path("figures")
N_SAMPLES = 5000
SEED = 73

# Fitted parameters come from a Poisson-well run
fh.WELL_TYPE = "poisson"

SCALE = 2.1
STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10 * SCALE,
    "axes.titlesize": 13 * SCALE,
    "axes.labelsize": 11 * SCALE,
    "xtick.labelsize": 9 * SCALE,
    "ytick.labelsize": 9 * SCALE,
    "legend.fontsize": 9 * SCALE,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_fitted_params() -> tuple[dict[int, dict], dict[int, dict]]:
    df = pd.read_csv(PARAM_CSV)
    params, r2 = {}, {}
    for _, row in df.iterrows():
        dur = int(row["duration_ms"])
        if "tau_in" in row.index and pd.notna(row["tau_in"]):
            tau_in = float(row["tau_in"])
        else:
            tau_in = float("inf")
        params[dur] = {
            "sigma_c":   row["sigma_c"],
            "tau_in":    tau_in,
            "s0":        row["s0"],
            "lambda0":   row["lambda0"],
            "t0":        row["t0"],
            "kappa":     row["kappa"],
            "kappa_lse": FIXED_KAPPA or 1.0,
            "eta":       FIXED_ETA or 1.0,
        }
        r2[dur] = {
            "mg": row["R2_mean_guess"],
            "ea": row["R2_pct_exact"],
            "rt": row["R2_rt"],
        }
    return params, r2


RT_UPPER = 15.0   # seconds; excludes <1% of trials




def extract_trial_data(duration_ms: int) -> pd.DataFrame:
    """Per-trial data for boxplots (guess and RT)."""
    df = pd.read_csv(RAW_CSV)
    subset = df[(df["cond"] == "DURATION") & (df["exp_dur"] == duration_ms)]
    trial = (
        subset.groupby(["uniqueid", "trial_id"])
        .agg(n_shown=("n_shown", "first"),
             n_guess=("n_guess", "first"),
             rt=("rt", "first"))
        .reset_index()
    )
    trial["rt_s"] = trial["rt"] / 1000.0
    trial["exact_pct"] = (trial["n_guess"] == trial["n_shown"]).astype(float) * 100
    return trial[
        (trial["n_shown"] >= N_TRUE_MIN) & (trial["n_shown"] <= N_TRUE_MAX)
    ].reset_index(drop=True)


def main() -> None:
    plt.rcParams.update(STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params, r2 = load_fitted_params()

    print("Extracting empirical data and running simulations ...")
    empiricals, models, trials = {}, {}, {}
    for dur in DURATIONS:
        print(f"  {dur} ms", flush=True)
        empiricals[dur] = extract_empirical(dur)
        models[dur] = simulate(params[dur], n_samples_per_n=N_SAMPLES, seed=SEED)
        trials[dur] = extract_trial_data(dur)

    n_rows = len(DURATIONS)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(15, 3.2 * n_rows),
        sharex=True,
    )

    emp_color = "#2171b5"
    mod_color = "#cb181d"

    bp_props = dict(
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops=dict(color=emp_color, lw=1.5),
        boxprops=dict(facecolor=emp_color, alpha=0.25,
                      edgecolor=emp_color, lw=0.8),
        whiskerprops=dict(color=emp_color, lw=0.8),
        capprops=dict(color=emp_color, lw=0.8),
    )
    pt_kw = dict(ms=5, alpha=0.8, color=emp_color, zorder=3)

    for row, dur in enumerate(DURATIONS):
        emp, mod = empiricals[dur], models[dur]
        trial = trials[dur]
        ns_e = emp["n_shown"].values
        ns_m = mod["n_shown"].values
        positions = sorted(trial["n_shown"].unique())

        # ── Col 0: Guess (empirical: boxplots or per-n means) ─────────────
        ax = axes[row, 0]
        if EMPIRICAL_ESTIMATION == "points":
            ax.plot(ns_e, emp["mean_guess"], "o", **pt_kw)
        else:
            bp_data = [
                trial.loc[trial["n_shown"] == n, "n_guess"].values
                for n in positions
            ]
            ax.boxplot(bp_data, positions=positions, **bp_props)
        sem_guess = mod["std_guess"] / np.sqrt(N_SAMPLES)
        ax.fill_between(
            ns_m,
            np.maximum(mod["mean_guess"] - sem_guess, 0),
            mod["mean_guess"] + sem_guess,
            color=mod_color, alpha=0.15, zorder=4,
        )
        ax.plot(ns_m, mod["mean_guess"], "-",
                color=mod_color, lw=2.5, label="Model ± SEM", zorder=5)
        ax.plot([N_TRUE_MIN, N_TRUE_MAX], [N_TRUE_MIN, N_TRUE_MAX],
                "k--", lw=0.8, alpha=0.35, zorder=1)
        ax.set_ylim(bottom=0)
        if row == 0:
            ax.set_title("Estimation", fontweight="bold")

        # ── Col 1: Mean exact accuracy ────────────────────────────────────
        ax = axes[row, 1]
        if ACC_ERROR_BARS:
            std_acc = trial.groupby("n_shown")["exact_pct"].std()
            std_acc = std_acc.reindex(ns_e).values
            ax.errorbar(ns_e, emp["pct_exact"], yerr=std_acc,
                        fmt="o", capsize=3, ms=5, lw=1, alpha=0.8,
                        color=emp_color, zorder=3)
        else:
            ax.plot(ns_e, emp["pct_exact"], "o", **pt_kw)
        sem_acc = mod["std_exact"] / np.sqrt(N_SAMPLES)
        ax.fill_between(
            ns_m,
            np.maximum(mod["pct_exact"] - sem_acc, 0),
            np.minimum(mod["pct_exact"] + sem_acc, 100),
            color=mod_color, alpha=0.15, zorder=4,
        )
        ax.plot(ns_m, mod["pct_exact"], "-",
                color=mod_color, lw=2.5, zorder=5)
        ax.set_ylim(bottom=0)
        if row == 0:
            ax.set_title("Exact accuracy", fontweight="bold")
            ax.legend(
                [plt.Line2D([], [], color=mod_color, lw=2.5),
                 plt.Line2D([], [], marker="o", ls="", color=emp_color, ms=5, alpha=0.8)],
                ["Model ± SEM", "Empirical"],
                loc="upper center",
            )

        # ── Col 2: RT (empirical: boxplots trimmed, or per-n medians) ────
        ax = axes[row, 2]
        if EMPIRICAL_RT == "points":
            ax.plot(ns_e, emp["median_rt_s"], "o", **pt_kw)
        else:
            trial_rt = trial[trial["rt_s"] <= RT_UPPER]
            rt_data = [
                trial_rt.loc[trial_rt["n_shown"] == n, "rt_s"].values
                for n in positions
            ]
            ax.boxplot(rt_data, positions=positions, **bp_props)
        sem_rt = mod["std_rt"] / np.sqrt(N_SAMPLES)
        ax.fill_between(
            ns_m,
            mod["median_rt"] - sem_rt,
            mod["median_rt"] + sem_rt,
            color=mod_color, alpha=0.15, zorder=4,
        )
        ax.plot(ns_m, mod["median_rt"], "-",
                color=mod_color, lw=2.5, zorder=5)
        if row == 0:
            ax.set_title("Response time", fontweight="bold")

        # ── R² annotations ─────────────────────────────────────────────────
        r2_kw = dict(fontsize=9 * SCALE, fontweight="bold", zorder=10)
        r2_dur = r2[dur]
        axes[row, 0].text(
            0.22, 0.92, f"$R^2 = {r2_dur['mg']:.2f}$",
            transform=axes[row, 0].transAxes, ha="left", va="top",
            **r2_kw,
        )
        r2_ea_y = 0.50 if row == 0 else 0.65
        axes[row, 1].text(
            0.7, r2_ea_y, f"$R^2 = {r2_dur['ea']:.2f}$",
            transform=axes[row, 1].transAxes, ha="center", va="center",
            **r2_kw,
        )
        axes[row, 2].text(
            0.97, 0.04, f"$R^2 = {r2_dur['rt']:.2f}$",
            transform=axes[row, 2].transAxes, ha="right", va="bottom",
            **r2_kw,
        )

        # ── Panel labels ──────────────────────────────────────────────────
        for col in range(3):
            label = chr(ord("a") + row * 3 + col)
            if col == 1:
                x_lbl, ha_lbl = 0.98, "right"
            else:
                x_lbl, ha_lbl = 0.02, "left"
            axes[row, col].text(
                x_lbl, 0.97, label,
                transform=axes[row, col].transAxes,
                ha=ha_lbl, va="top",
                fontsize=14 * SCALE, fontweight="bold",
            )

        # ── Duration label on right y-axis ─────────────────────────────────
        ax_r = axes[row, 2].twinx()
        ax_r.set_yticks([])
        ax_r.set_ylabel(
            f"{dur} ms", fontweight="bold",
            rotation=270, labelpad=20, fontsize=14 * SCALE,
        )

    mid = n_rows // 2
    col_ylabels = ["Guess", "% exact correct", "RT (s)"]
    xt = list(range(N_TRUE_MIN, N_TRUE_MAX + 1))
    xtick_lbls = [str(n) if n % 2 == 1 else f"\n{n}" for n in xt]
    for col in range(3):
        axes[mid, col].set_ylabel(col_ylabels[col])
        axes[-1, col].set_xticks(xt)
        axes[-1, col].set_xticklabels(xtick_lbls)
        axes[-1, col].tick_params(axis="x", pad=2)
    axes[-1, 1].set_xlabel("$n$ shown")

    for row in range(n_rows - 1):
        for col in range(3):
            axes[row, col].tick_params(labelbottom=False)

    fig.tight_layout(h_pad=0.4)

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig2_piantadosi_fit.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
