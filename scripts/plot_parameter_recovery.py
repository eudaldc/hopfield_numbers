"""Plots for the identifiability / parameter-recovery analyses.

Reads the CSVs produced by ``scripts.run_parameter_recovery`` and emits:

  si_parameter_recovery.{pdf,png} — recovered vs true, per-param scatter + tradeoff heatmap
  si_profile_likelihood.pdf   — loss profile for c (sigma_c) and tau at 160 ms
  si_dd_bootstrap.pdf         — distribution of recovered tau_DD / tau_ctrl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import PARAM_NAMES

OUTPUT_DIR = Path("output/parameter_recovery")


PARAM_DISPLAY = {
    "sigma_c": r"$c$ (well scale)",
    "tau_in": r"$\tau$ (anchoring)",
    "s0": r"$s_0$ (noise scale)",
    "lambda0": r"$\lambda_0$ (noise exponent)",
    "t0": r"$t_0$ (basal RT, s)",
    "kappa": r"$\kappa$ (entropy→RT)",
}

LOG_PARAMS = {"sigma_c", "tau_in", "s0", "kappa"}


def _param_title(p: str) -> str:
    return PARAM_DISPLAY.get(p, p)


def _label_ms(lab: str) -> int:
    tail = lab.rsplit("_", 1)[-1]
    return int(tail.replace("ms", "")) if tail.endswith("ms") else 0


def _text_on_cmap(cmap, z: float, vmin: float, vmax: float) -> str:
    t = (z - vmin) / (vmax - vmin + 1e-12)
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b, _ = cmap(t)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.58 else "black"


# ── Helpers ──────────────────────────────────────────────────────────────


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


# ── Recovery scatter + trade-off matrix ──────────────────────────────────


def plot_recovery(csv_path: Path, out_pdf: Path) -> None:
    df = pd.read_csv(csv_path)
    _style()
    present = set(df["param"].unique())
    params = [p for p in PARAM_NAMES if p in present]
    if not params:
        raise ValueError(f"No recognized params in {csv_path}: {present}")

    n_p = len(params)
    if n_p == 4:
        scatter_ncols, scatter_nrows = 2, 2
    else:
        scatter_ncols = min(3, n_p) if n_p > 1 else 1
        scatter_nrows = int(np.ceil(n_p / scatter_ncols))
    total_rows = scatter_nrows + 1
    fig_w = 4.2 * scatter_ncols
    fig_h = 3.2 * scatter_nrows + 3.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = [1] * scatter_nrows + [0.85]
    gs = fig.add_gridspec(
        total_rows,
        scatter_ncols,
        height_ratios=height_ratios,
        wspace=0.35,
        hspace=0.45,
    )

    summary_rows: list[dict] = []
    for i, p in enumerate(params):
        ax = fig.add_subplot(gs[i // scatter_ncols, i % scatter_ncols])
        sub = df[df["param"] == p]
        x = sub["true"].values
        y = sub["recovered"].values

        use_log = p in LOG_PARAMS and np.all(x > 0) and np.all(y > 0)
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            xref = np.geomspace(x.min() * 0.9, x.max() * 1.1, 100)
        else:
            xref = np.linspace(x.min(), x.max(), 100)
        ax.plot(xref, xref, "--", color="#888", lw=1, alpha=0.7, label="identity")

        by_label = {lab: sub[sub["label"] == lab] for lab in sub["label"].unique()}

        cmap = plt.colormaps["plasma"]
        labs_sorted = sorted(by_label.keys(), key=_label_ms)
        for k, lab in enumerate(labs_sorted):
            s = by_label[lab]
            ax.scatter(
                s["true"],
                s["recovered"],
                s=18,
                alpha=0.65,
                color=cmap(k / max(len(labs_sorted) - 1, 1)),
                label=lab.replace("piant_", "").replace("ms", " ms"),
                edgecolors="none",
            )

        r2 = _r2(x, y)
        rho, _ = spearmanr(x, y)
        ax.set_title(f"{_param_title(p)}\n$R^2={r2:.2f}$  $\\rho={rho:.2f}$")
        ax.set_xlabel("true")
        ax.set_ylabel("recovered")
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        summary_rows.append({"param": p, "R2": r2, "spearman_rho": rho})

    ax = fig.add_subplot(gs[scatter_nrows, :])
    err_wide = (
        df.assign(err=df["recovered"] - df["true"])
        .pivot_table(
            index=["label", "run_id"], columns="param", values="err", aggfunc="mean"
        )
        .reindex(columns=params)
    )
    corr = err_wide.corr(method="spearman").values
    cmap_hm = plt.colormaps["plasma"]
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap_hm, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_yticks(range(len(params)))
    ax.set_xticklabels(params)
    ax.set_yticklabels(params)
    for ii in range(len(params)):
        for jj in range(len(params)):
            v = corr[ii, jj]
            ax.text(
                jj,
                ii,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=_text_on_cmap(cmap_hm, v, -1.0, 1.0),
            )
    ax.set_title("Trade-off matrix: Spearman correlation of recovery errors")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\rho$(err$_i$, err$_j$)")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"))
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(
        out_pdf.with_suffix(".summary.csv"), index=False
    )
    print(f"Saved {out_pdf} and {out_pdf.with_suffix('.png')}")


# ── Profile likelihood ───────────────────────────────────────────────────


def plot_profile_likelihood(
    csv_path: Path, out_pdf: Path, fitted_values: dict[str, float] | None = None
) -> None:
    df = pd.read_csv(csv_path)
    _style()
    params = df["param_fixed"].unique().tolist()
    fig, axes = plt.subplots(
        1, len(params), figsize=(5.5 * len(params), 4.2), squeeze=False
    )
    for ax, p in zip(axes[0], params):
        sub = df[df["param_fixed"] == p].sort_values("value")
        ax.plot(sub["value"], sub["loss"], "o-", color="#4C72B0", lw=2, ms=6)
        if p in LOG_PARAMS:
            ax.set_xscale("log")
        if fitted_values and p in fitted_values:
            ax.axvline(
                fitted_values[p], color="#cb181d", ls="--", lw=1.5, label="MLE"
            )
            ax.legend(frameon=False)
        ax.set_xlabel(PARAM_DISPLAY.get(p, p))
        ax.set_ylabel("loss (refit)")
        ax.set_title(f"Profile likelihood: {PARAM_DISPLAY.get(p, p)}")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_pdf}")


# ── DD bootstrap ─────────────────────────────────────────────────────────


def plot_dd_bootstrap(csv_path: Path, out_pdf: Path) -> None:
    df = pd.read_csv(csv_path)
    _style()
    tau = df[df["param"] == "tau_in"].copy()
    tau_dd = tau[tau["group"] == "DD"][["run_id", "recovered"]].rename(
        columns={"recovered": "tau_dd"}
    )
    tau_ct = tau[tau["group"] == "Controls"][["run_id", "recovered"]].rename(
        columns={"recovered": "tau_ctrl"}
    )
    merged = tau_dd.merge(tau_ct, on="run_id")
    merged["ratio"] = merged["tau_dd"] / merged["tau_ctrl"]
    true_ratio = (
        tau[tau["group"] == "DD"]["true"].iloc[0]
        / tau[tau["group"] == "Controls"]["true"].iloc[0]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.hist(merged["tau_dd"], bins=20, color="#cb181d", alpha=0.7, label="DD")
    ax.axvline(
        tau[tau["group"] == "DD"]["true"].iloc[0],
        color="k",
        ls="--",
        lw=1.5,
        label="true",
    )
    ax.set_xlabel(r"recovered $\tau_{DD}$")
    ax.set_ylabel("count")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    ax.set_title("DD group")

    ax = axes[1]
    ax.hist(merged["tau_ctrl"], bins=20, color="#2171b5", alpha=0.7, label="Controls")
    ax.axvline(
        tau[tau["group"] == "Controls"]["true"].iloc[0],
        color="k",
        ls="--",
        lw=1.5,
        label="true",
    )
    ax.set_xlabel(r"recovered $\tau_{ctrl}$")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    ax.set_title("Controls group")

    ax = axes[2]
    ax.hist(merged["ratio"], bins=20, color="#55A868", alpha=0.75)
    ax.axvline(true_ratio, color="k", ls="--", lw=1.5, label=f"true = {true_ratio:.1f}×")
    ax.axvline(1.0, color="#999", ls=":", lw=1.2, label="1× (null)")
    ax.set_xlabel(r"$\hat\tau_{DD} / \hat\tau_{ctrl}$")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    ax.set_title(
        "Ratio distribution\n"
        f"median = {merged['ratio'].median():.1f}×  "
        f"IQR [{merged['ratio'].quantile(0.25):.1f}, "
        f"{merged['ratio'].quantile(0.75):.1f}]"
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    merged.to_csv(out_pdf.with_suffix(".ratios.csv"), index=False)
    print(f"Saved {out_pdf}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--which", choices=["recovery", "profile", "dd", "all"],
                        default="all")
    args = parser.parse_args()

    tag = "_smoke" if args.smoke_test else ""

    if args.which in ("recovery", "all"):
        plot_recovery(
            OUTPUT_DIR / f"recovery_all_durations{tag}.csv",
            OUTPUT_DIR / f"si_parameter_recovery{tag}.pdf",
        )
    if args.which in ("profile", "all"):
        piant = pd.read_csv("output/piantadosi_fit/summary_all_durations_poisson.csv")
        ref = piant[piant["duration_ms"] == 160].iloc[0]
        fitted = {"sigma_c": float(ref["sigma_c"]), "tau_in": float(ref["tau_in"])}
        plot_profile_likelihood(
            OUTPUT_DIR / f"profile_likelihood{tag}.csv",
            OUTPUT_DIR / f"si_profile_likelihood{tag}.pdf",
            fitted_values=fitted,
        )
    if args.which in ("dd", "all"):
        plot_dd_bootstrap(
            OUTPUT_DIR / f"dd_bootstrap{tag}.csv",
            OUTPUT_DIR / f"si_dd_bootstrap{tag}.pdf",
        )


if __name__ == "__main__":
    main()
