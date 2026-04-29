"""Plotting functions for the Piantadosi attractor model and evaluation tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import false_discovery_control, mannwhitneyu, norm

from scripts.fit_helpers import (
    N_TRUE_MIN,
    N_TRUE_MAX,
    N_SUPPORT_MAX,
    WELL_TYPE,
    FIXED_BETA,
    FIXED_KAPPA,
    FIXED_ETA,
    PARAM_NAMES,
    benford_sigma,
    compute_goodness,
    psychometric_weber,
    _gaussian_f_n,
    _poisson_f_n,
)
REYNVOET_MODEL_LINESTYLE = {
    40: "-",
    80: "--",
    160: "-.",
    320: ":",
    640: (0, (5, 2, 1, 2)),
}

# ── Shared style ─────────────────────────────────────────────────────────

STYLE: dict = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]

EMP_COLOR = "#2171b5"
MOD_COLOR = "#cb181d"


# ── Piantadosi fit plots ────────────────────────────────────────────────


def plot_comparison(
    empirical: pd.DataFrame,
    model: pd.DataFrame,
    theta: dict,
    output_dir: Path,
    tag: str = "fitted",
    duration_ms: int = 160,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ns_e = empirical["n_shown"].values
    ns_m = model["n_shown"].values
    dur_label = f"Empirical {duration_ms} ms"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.errorbar(
        ns_e,
        empirical["mean_guess"],
        yerr=empirical["std_guess"],
        fmt="o-",
        capsize=3,
        color="#2171b5",
        label=dur_label,
        ms=5,
    )
    ax.plot(ns_m, model["mean_guess"], "s--", color="#cb181d", label="Model", ms=5)
    ax.plot(
        [N_TRUE_MIN, N_TRUE_MAX],
        [N_TRUE_MIN, N_TRUE_MAX],
        "k:",
        lw=1,
        alpha=0.4,
        label="Perfect",
    )
    ax.set_xlabel("n shown")
    ax.set_ylabel("Mean guess")
    ax.set_title("Mean estimation")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        ns_e,
        empirical["pct_exact"],
        "o-",
        color="#2171b5",
        label=dur_label,
        ms=5,
    )
    ax.plot(ns_m, model["pct_exact"], "s--", color="#cb181d", label="Model", ms=5)
    ax.set_xlabel("n shown")
    ax.set_ylabel("% exact correct")
    ax.set_title("Exact accuracy")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(
        ns_e,
        empirical["median_rt_s"],
        "o-",
        color="#2171b5",
        label=dur_label,
        ms=5,
    )
    ax.plot(ns_m, model["median_rt"], "s--", color="#cb181d", label="Model", ms=5)
    ax.set_xlabel("n shown")
    ax.set_ylabel("Median RT (s)")
    ax.set_title("Response time")
    ax.legend(fontsize=8)

    param_str = "  ".join(f"{k}={v:.3f}" for k, v in theta.items())
    fig.suptitle(f"Piantadosi {duration_ms} ms — {tag}\n{param_str}", fontsize=9)
    fig.tight_layout()
    path = output_dir / f"fit_{duration_ms}ms_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_energy_landscape(
    theta: dict,
    output_dir: Path,
    duration_ms: int = 160,
) -> Path:
    """Two-panel figure: well potential and per-n width profile."""
    output_dir.mkdir(parents=True, exist_ok=True)

    integers = torch.arange(N_TRUE_MIN, N_SUPPORT_MAX + 1, dtype=torch.float32)
    rho = theta.get("rho", 1.0)
    mu = integers.pow(rho)
    kappa = theta.get("kappa_lse", FIXED_KAPPA or 1.0)

    # ── Panel A: well potential V(x) = -kappa_lse * logsumexp(f_n) ───────
    x = torch.linspace(0.1, mu[-1].item() + 1.0, 2000)
    x_col = x.unsqueeze(-1)
    if WELL_TYPE == "poisson":
        f_n = _poisson_f_n(x_col, mu, theta["sigma_c"])
    else:
        sigma = benford_sigma(
            integers, theta["sigma_c"], theta.get("beta", FIXED_BETA or 1.0)
        )
        f_n = _gaussian_f_n(x_col, mu, sigma.square(), sigma.log())
    V = -kappa * torch.logsumexp(f_n, dim=-1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].plot(x.numpy(), V.numpy(), "k-", lw=1.5)
    n_range = N_TRUE_MAX - N_TRUE_MIN + 1
    for i in range(n_range):
        mu_i = mu[i].item()
        axes[0].axvline(mu_i, color="gray", ls=":", lw=0.4, alpha=0.6)
        axes[0].text(
            mu_i,
            axes[0].get_ylim()[0] if i > 0 else V.min().item(),
            str(int(integers[i].item())),
            ha="center",
            fontsize=7,
            color="gray",
        )
    axes[0].set_ylabel(
        "$V(x) = -\\kappa_{\\mathrm{LSE}} \\cdot \\mathrm{LSE}(\\ldots)$"
    )
    line_label = f"compressed, $\\rho$={rho:.3f}" if rho < 1.0 else "equispaced"
    if WELL_TYPE == "poisson":
        well_width_phrase = "Poisson dispersion"
    else:
        well_width_phrase = "Benford-scaled Gaussian widths"
    axes[0].set_title(f"A.  Well potential ({line_label} centres, {well_width_phrase})")

    # ── Panel B: effective well width ──────────────────────────────────────
    n_range = N_TRUE_MAX - N_TRUE_MIN + 1
    n_vals = integers[:n_range].numpy()
    c = theta["sigma_c"]

    if WELL_TYPE == "poisson":
        sigma_eff = np.sqrt(c * n_vals)
        width_label = f"$\\sigma_{{eff}} = \\sqrt{{c \\cdot n}}$,  c={c:.4f}"
    else:
        beta = theta.get("beta", FIXED_BETA or 1.0)
        freq_t = torch.log10(1.0 + 1.0 / integers[:n_range])
        sigma_eff = (c / freq_t).pow(beta).numpy()
        width_label = (
            f"$\\sigma_n = (c / \\mathrm{{freq}}_n)^\\beta$,  "
            f"c={c:.4f}, $\\beta$={beta:.3f}"
        )

    color_s, color_f = "#2171b5", "#cb181d"
    ax_s = axes[1]
    ax_s.bar(n_vals, sigma_eff, color=color_s, alpha=0.7, width=0.6)
    ax_s.set_xlabel("n")
    ax_s.set_ylabel("Effective $\\sigma_n$", color=color_s, fontsize=12)
    ax_s.tick_params(axis="y", labelcolor=color_s)
    ax_s.set_xticks(range(N_TRUE_MIN, N_TRUE_MAX + 1))

    if WELL_TYPE != "poisson":
        freq = torch.log10(1.0 + 1.0 / integers[:n_range]).numpy()
        ax_f = ax_s.twinx()
        ax_f.plot(n_vals, freq, "o-", color=color_f, ms=5, lw=1.5, zorder=5)
        ax_f.set_ylabel(
            "$\\mathrm{freq}_n = \\log_{10}(1 + 1/n)$",
            color=color_f,
            fontsize=11,
        )
        ax_f.tick_params(axis="y", labelcolor=color_f)

    ax_s.set_title(f"B.  Well width ({WELL_TYPE}):  {width_label}")

    fig.tight_layout()
    path = output_dir / f"fit_{duration_ms}ms_energy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_number_line(
    theta: dict,
    output_dir: Path,
    duration_ms: int = 160,
) -> Path:
    """Show the compressed vs linear number line and well structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    integers = torch.arange(N_TRUE_MIN, N_SUPPORT_MAX + 1, dtype=torch.float32)
    rho = theta.get("rho", 1.0)
    mu = integers.pow(rho)
    sigma = benford_sigma(
        integers, theta["sigma_c"], theta.get("beta", FIXED_BETA or 1.0)
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    n_range = N_TRUE_MAX - N_TRUE_MIN + 1
    n_vals = integers[:n_range].numpy()
    mu_vals = mu[:n_range].numpy()
    axes[0].plot(n_vals, n_vals, "k--", lw=1, alpha=0.4, label="linear ($\\rho$=1)")
    axes[0].plot(
        n_vals,
        mu_vals,
        "o-",
        color="#2171b5",
        ms=6,
        label=f"compressed ($\\rho$={rho:.3f})",
    )
    axes[0].set_xlabel("n (integer)")
    axes[0].set_ylabel("$\\mu_n = n^\\rho$")
    axes[0].set_title("A.  Number line compression")
    axes[0].legend()

    sigma_sq = sigma.square()
    log_sigma = sigma.log()
    kappa = theta.get("kappa_lse", FIXED_KAPPA or 1.0)

    x = torch.linspace(0.0, mu[-1].item() + 1.0, 2000)
    diff = x.unsqueeze(-1) - mu
    f_n = -log_sigma - (diff**2) / (2.0 * sigma_sq)
    V = -kappa * torch.logsumexp(f_n, dim=-1)

    axes[1].plot(x.numpy(), V.numpy(), "k-", lw=1.5)
    for i in range(n_range):
        axes[1].axvline(mu[i].item(), color="gray", ls=":", lw=0.4, alpha=0.6)
        axes[1].text(
            mu[i].item(),
            V.min().item() * 0.95,
            str(i + N_TRUE_MIN),
            ha="center",
            fontsize=7,
            color="gray",
        )
    axes[1].set_xlabel("x (latent position)")
    axes[1].set_ylabel("V(x)")
    axes[1].set_title("B.  Well potential on compressed number line")

    fig.tight_layout()
    path = output_dir / f"fit_{duration_ms}ms_compressed_line.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ── Duration sweep summary ──────────────────────────────────────────────


def _load_ci_flat(path: Path) -> dict:
    """Flatten a per-run Hessian-CI CSV into ``{param}_se``/``_ci_lo``/``_ci_hi``/``_status``.

    Returns an empty dict if the file is missing so callers stay robust to
    runs that predate the CI output.
    """
    if not path.exists():
        return {}
    try:
        ci = pd.read_csv(path)
    except Exception:
        return {}
    flat: dict = {}
    for _, row in ci.iterrows():
        p = row["param"]
        flat[f"{p}_se"] = row["se"]
        flat[f"{p}_ci_lo"] = row["ci_lo"]
        flat[f"{p}_ci_hi"] = row["ci_hi"]
        if "status" in row:
            flat[f"{p}_ci_status"] = row["status"]
        if "method" in row:
            flat[f"{p}_ci_method"] = row["method"]
    return flat


def _sweep_table_param_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are fitted constants (not GOF metrics, not CI flatten keys)."""
    blocklist = {
        "duration_ms",
        "opacity",
        "n_data",
        "k_params",
        "channels_used",
        "logL",
        "AIC",
        "BIC",
    }

    def _is_goodness(c: str) -> bool:
        return c.startswith(("R2_", "RMSE_", "MAE_", "sigma2_"))

    derived_suffixes = (
        "_se",
        "_ci_lo",
        "_ci_hi",
        "_ci_status",
        "_ci_method",
        "_boot_ci_lo",
        "_boot_ci_hi",
        "_boot_median",
        "_boot_n",
    )

    def _is_derived(c: str) -> bool:
        return any(c.endswith(s) for s in derived_suffixes)

    candidates = [
        c
        for c in df.columns
        if c not in blocklist and not _is_goodness(c) and not _is_derived(c)
    ]
    ordered = [p for p in PARAM_NAMES if p in candidates]
    for c in sorted(x for x in candidates if x not in ordered):
        ordered.append(c)
    return ordered


def print_sweep_params_ci_table(
    df: pd.DataFrame,
    group_col: str,
    *,
    title: str,
) -> None:
    """Print parameter point estimates and CIs (Hessian; bootstrap if present) to stdout."""
    param_cols = _sweep_table_param_columns(df)
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    boot_keys = [
        c
        for c in df.columns
        if c.endswith("_boot_ci_lo") and c.removesuffix("_boot_ci_lo") in param_cols
    ]
    has_boot = bool(boot_keys) and any(df[k].notna().any() for k in boot_keys)

    for _, row in df.iterrows():
        label = row[group_col]
        if group_col == "duration_ms":
            print(f"\n--- {int(label)} ms ---")
        else:
            print(f"\n--- {group_col} = {label} ---")
        hdr = (
            f"{'param':>14}  {'estimate':>14}  {'95% CI':>30}  "
            f"{'SE':>12}  {'status':<14}  {'method':<14}"
        )
        if has_boot:
            hdr += "  boot 95% CI"
        print(hdr)
        print("-" * (92 + (22 if has_boot else 0)))
        for p in param_cols:
            if p not in row.index:
                continue
            est = row[p]
            if pd.isna(est):
                continue
            lo_c, hi_c = f"{p}_ci_lo", f"{p}_ci_hi"
            se_c, st_c = f"{p}_se", f"{p}_ci_status"
            meth_c = f"{p}_ci_method"
            b_lo, b_hi = f"{p}_boot_ci_lo", f"{p}_boot_ci_hi"
            est_str = f"{float(est):.6g}"
            if lo_c in row.index and hi_c in row.index:
                lo_v, hi_v = row[lo_c], row[hi_c]
                if pd.notna(lo_v) and pd.notna(hi_v):
                    ci_str = f"[{float(lo_v):.6g}, {float(hi_v):.6g}]"
                else:
                    ci_str = ""
            else:
                ci_str = ""
            if se_c in row.index and pd.notna(row[se_c]):
                se_v = float(row[se_c])
                se_str = f"{se_v:.6g}" if np.isfinite(se_v) else "--"
            else:
                se_str = "--"
            st_str = str(row[st_c]) if st_c in row.index and pd.notna(row[st_c]) else ""
            meth_str = (
                str(row[meth_c])
                if meth_c in row.index and pd.notna(row[meth_c])
                else ""
            )
            line = (
                f"{p:>14}  {est_str:>14}  {ci_str:>30}  "
                f"{se_str:>12}  {st_str:<14}  {meth_str:<14}"
            )
            if has_boot:
                if b_lo in row.index and b_hi in row.index:
                    lb, hb = row[b_lo], row[b_hi]
                    if pd.notna(lb) and pd.notna(hb):
                        line += f"  [{float(lb):.6g}, {float(hb):.6g}]"
                    else:
                        line += "  " + " " * 18
                else:
                    line += "  " + " " * 18
            print(line)
    print()


def _load_bootstrap_flat(path: Path) -> dict:
    """Flatten a bootstrap-CI CSV into ``{param}_boot_ci_lo``/``_boot_ci_hi``/``_boot_median``."""
    if not path.exists():
        return {}
    try:
        boot = pd.read_csv(path)
    except Exception:
        return {}
    flat: dict = {}
    for _, row in boot.iterrows():
        p = row["param"]
        flat[f"{p}_boot_ci_lo"] = row["ci_lo"]
        flat[f"{p}_boot_ci_hi"] = row["ci_hi"]
        if "median" in row:
            flat[f"{p}_boot_median"] = row["median"]
        if "n_boot" in row:
            flat[f"{p}_boot_n"] = row["n_boot"]
    return flat


def plot_summary(
    results: dict[int, tuple[dict, pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Summary panel across durations + CSV with params and goodness metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    durations = sorted(results.keys())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(durations)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for color, dur in zip(cmap, durations):
        theta, model, emp = results[dur]
        ns = model["n_shown"].values
        lbl = f"{dur} ms"
        axes[0].plot(ns, model["mean_guess"], "s--", color=color, label=lbl, ms=4)
        axes[1].plot(ns, model["pct_exact"], "s--", color=color, label=lbl, ms=4)
        axes[2].plot(ns, model["median_rt"], "s--", color=color, label=lbl, ms=4)

    axes[0].plot(
        [N_TRUE_MIN, N_TRUE_MAX], [N_TRUE_MIN, N_TRUE_MAX], "k:", lw=1, alpha=0.4
    )
    axes[0].set_xlabel("n shown")
    axes[0].set_ylabel("Mean guess")
    axes[0].set_title("Mean estimation")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("n shown")
    axes[1].set_ylabel("% exact correct")
    axes[1].set_title("Exact accuracy")
    axes[1].legend(fontsize=7)

    axes[2].set_xlabel("n shown")
    axes[2].set_ylabel("Median RT (s)")
    axes[2].set_title("Response time")
    axes[2].legend(fontsize=7)

    fig.suptitle("Model fit across exposure durations", fontsize=13)
    fig.tight_layout()
    path = output_dir / "summary_all_durations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")

    rows = []
    for dur in durations:
        theta, model, emp = results[dur]
        goodness = compute_goodness(emp, model, k=len(PARAM_NAMES))
        ci_flat = _load_ci_flat(output_dir / f"fit_{dur}ms_ci_hessian.csv")
        boot_flat = _load_bootstrap_flat(output_dir / f"fit_{dur}ms_ci_bootstrap.csv")
        rows.append({"duration_ms": dur, **theta, **goodness, **ci_flat, **boot_flat})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "summary_all_durations.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print_sweep_params_ci_table(
        df,
        "duration_ms",
        title=(
            "Fitted parameters and 95% CIs by exposure duration "
            "(same columns as summary_all_durations.csv)"
        ),
    )


# ── Contrast sweep summary ───────────────────────────────────────────


def plot_summary_contrast(
    results: dict[float, tuple[dict, pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Summary panel across opacity levels + CSV with params and goodness metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    opacities = sorted(results.keys())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(opacities)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for color, opa in zip(cmap, opacities):
        theta, model, emp = results[opa]
        ns = model["n_shown"].values
        lbl = f"opacity {opa}"
        axes[0].plot(ns, model["mean_guess"], "s--", color=color, label=lbl, ms=4)
        axes[1].plot(ns, model["pct_exact"], "s--", color=color, label=lbl, ms=4)
        axes[2].plot(ns, model["median_rt"], "s--", color=color, label=lbl, ms=4)

    axes[0].plot(
        [N_TRUE_MIN, N_TRUE_MAX], [N_TRUE_MIN, N_TRUE_MAX], "k:", lw=1, alpha=0.4
    )
    axes[0].set_xlabel("n shown")
    axes[0].set_ylabel("Mean guess")
    axes[0].set_title("Mean estimation")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("n shown")
    axes[1].set_ylabel("% exact correct")
    axes[1].set_title("Exact accuracy")
    axes[1].legend(fontsize=7)

    axes[2].set_xlabel("n shown")
    axes[2].set_ylabel("Median RT (s)")
    axes[2].set_title("Response time")
    axes[2].legend(fontsize=7)

    fig.suptitle("Model fit across contrast (opacity) levels", fontsize=13)
    fig.tight_layout()
    path = output_dir / "summary_all_contrasts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")

    rows = []
    for opa in opacities:
        theta, model, emp = results[opa]
        goodness = compute_goodness(emp, model, k=len(PARAM_NAMES))
        ci_flat = _load_ci_flat(output_dir / f"fit_opacity{opa}_ci_hessian.csv")
        boot_flat = _load_bootstrap_flat(
            output_dir / f"fit_opacity{opa}_ci_bootstrap.csv"
        )
        rows.append({"opacity": opa, **theta, **goodness, **ci_flat, **boot_flat})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "summary_all_contrasts.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print_sweep_params_ci_table(
        df,
        "opacity",
        title=(
            "Fitted parameters and 95% CIs by opacity "
            "(same columns as summary_all_contrasts.csv)"
        ),
    )


# ── Distance & size / SNARC plots ──────────────────────────────────────


def plot_distance_size(df: pd.DataFrame, output_dir: Path, n_max: int) -> None:
    mpl.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # A — Accuracy vs distance
    ax = axes[0]
    by_d = df.groupby("distance")["accuracy"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        by_d["distance"],
        by_d["mean"],
        yerr=by_d["sem"] * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
    )
    ax.set_xlabel("Numerical distance  $|n_1 - n_2|$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A.  Distance effect (accuracy)", fontweight="bold", loc="left")
    ax.set_xticks(range(1, n_max))

    # B — RT vs distance, split by pair magnitude
    ax = axes[1]
    bins = [
        (1.5, 4.0, "Small pairs (2–4)"),
        (4.0, 6.5, "Medium pairs (4–6)"),
        (6.5, 9.5, "Large pairs (7–9)"),
    ]
    for i, (lo, hi, lbl) in enumerate(bins):
        sub = df[(df["size"] >= lo) & (df["size"] < hi)]
        by_d = sub.groupby("distance")["mean_rt_ms"].mean().reset_index()
        if len(by_d) > 1:
            ax.plot(
                by_d["distance"],
                by_d["mean_rt_ms"],
                "o-",
                color=PALETTE[i],
                label=lbl,
                ms=6,
                lw=2,
            )
    ax.set_xlabel("Numerical distance")
    ax.set_ylabel("RT (ms)")
    ax.set_title("B.  Distance × size (RT)", fontweight="bold", loc="left")
    ax.legend()
    ax.set_xticks(range(1, n_max))

    # C — Overall RT vs distance
    ax = axes[2]
    by_d = df.groupby("distance")["mean_rt_ms"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        by_d["distance"],
        by_d["mean"],
        yerr=by_d["sem"] * 1.96,
        fmt="s-",
        color=PALETTE[3],
        capsize=4,
        ms=8,
        lw=2,
    )
    ax.set_xlabel("Numerical distance")
    ax.set_ylabel("RT (ms)")
    ax.set_title("C.  Distance effect (RT)", fontweight="bold", loc="left")
    ax.set_xticks(range(1, n_max))

    fig.tight_layout()
    path = output_dir / "distance_size_effects.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_snarc(df: pd.DataFrame, output_dir: Path) -> None:
    mpl.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # A — dRT vs number (SNARC slope)
    ax = axes[0]
    ax.errorbar(
        df["number"],
        df["mean_drt_ms"],
        yerr=df["se_drt_ms"] * 1.96,
        fmt="o",
        color=PALETTE[0],
        capsize=4,
        ms=9,
        zorder=5,
    )

    coeffs = np.polyfit(df["number"], df["mean_drt_ms"], 1)
    r = np.corrcoef(df["number"], df["mean_drt_ms"])[0, 1]
    x_fit = np.linspace(0.5, 9.5, 100)
    ax.plot(
        x_fit,
        np.polyval(coeffs, x_fit),
        "--",
        color=PALETTE[3],
        lw=2,
        zorder=3,
        label=f"slope = {coeffs[0]:.1f} ms/digit,  $R^2$ = {r**2:.3f}",
    )

    ax.axhline(0, color="0.7", ls=":", lw=1)
    ax.set_xlabel("Number")
    ax.set_ylabel("dRT  (right − left,  ms)")
    ax.set_title("A.  SNARC effect", fontweight="bold", loc="left")
    ax.legend(loc="upper right")
    ax.set_xticks(range(1, 10))

    # B — Left vs Right hand RT
    ax = axes[1]
    ax.plot(
        df["number"],
        df["mean_rt_left_ms"],
        "o-",
        color=PALETTE[0],
        ms=6,
        lw=2,
        label="Left hand",
    )
    ax.plot(
        df["number"],
        df["mean_rt_right_ms"],
        "s-",
        color=PALETTE[3],
        ms=6,
        lw=2,
        label="Right hand",
    )
    ax.set_xlabel("Number")
    ax.set_ylabel("RT (ms)")
    ax.set_title("B.  RT by response hand", fontweight="bold", loc="left")
    ax.legend()
    ax.set_xticks(range(1, 10))

    fig.tight_layout()
    path = output_dir / "snarc_effect.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Weber law plots ─────────────────────────────────────────────────────


def plot_weber(
    df: pd.DataFrame,
    wf: pd.DataFrame,
    output_dir: Path,
    max_test_ratio: float,
) -> None:
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # A — Accuracy vs ratio
    ax = axes[0]
    for i, (n_ref, sub) in enumerate(df.groupby("n_ref")):
        sub = sub.sort_values("ratio")
        ax.plot(
            sub["ratio"],
            sub["accuracy"],
            "o-",
            color=PALETTE[i],
            ms=4,
            lw=1.8,
            label=f"$n_{{ref}}$ = {int(n_ref)}",
        )
        row = wf[wf["n_ref"] == n_ref]
        if len(row) and not np.isnan(row["weber_fraction"].values[0]):
            r_fit = np.linspace(1.0, sub["ratio"].max() + 0.05, 200)
            ax.plot(
                r_fit,
                psychometric_weber(r_fit, row["weber_fraction"].values[0]),
                "--",
                color=PALETTE[i],
                lw=1,
                alpha=0.5,
            )
    ax.axhline(75, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.text(max_test_ratio - 0.02, 76, "75 %", ha="right", fontsize=8, color="0.5")
    ax.set_xlabel("Ratio  $n_{test}\\,/\\,n_{ref}$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A.  Accuracy vs ratio", fontweight="bold", loc="left")
    ax.legend()

    # B — Accuracy vs absolute difference
    ax = axes[1]
    for i, (n_ref, sub) in enumerate(df.groupby("n_ref")):
        sub = sub.sort_values("distance")
        ax.plot(
            sub["distance"],
            sub["accuracy"],
            "o-",
            color=PALETTE[i],
            ms=4,
            lw=1.8,
            label=f"$n_{{ref}}$ = {int(n_ref)}",
        )
    ax.axhline(75, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Absolute difference  $n_{test} - n_{ref}$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B.  Accuracy vs difference", fontweight="bold", loc="left")
    ax.legend()

    # C — Weber fraction by reference magnitude
    ax = axes[2]
    valid = wf.dropna()
    bars = ax.bar(
        range(len(valid)),
        valid["weber_fraction"],
        color=[PALETTE[i] for i in range(len(valid))],
        width=0.65,
        edgecolor="white",
    )
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([str(int(v)) for v in valid["n_ref"]])
    ax.set_xlabel("Reference numerosity  $n_{ref}$")
    ax.set_ylabel("Weber fraction  $w$")
    ax.set_title("C.  Weber fraction", fontweight="bold", loc="left")
    for bar, w in zip(bars, valid["weber_fraction"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{w:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    path = output_dir / "weber_law.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Reynvoet comparison ─────────────────────────────────────────────────


def plot_reynvoet_comparison(
    empirical: pd.DataFrame,
    model_preds: dict[int, pd.DataFrame],
    weber_df: pd.DataFrame,
    output_dir: Path,
    *,
    model_curve_label_fmt: str | None = None,
) -> Path:
    """Two-panel figure: accuracy-vs-ratio and Weber fraction comparison.

    model_curve_label_fmt
        ``str.format`` pattern with ``{dur}`` for duration ms. When ``None``,
        labels are ``Model ({dur} ms params)``.
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Panel A: accuracy vs ratio ───────────────────────────────────────
    ax = axes[0]

    cong_style = {
        "correlated": {"marker": "^", "label": "Area-correlated"},
        "equal": {"marker": "o", "label": "Area-equal"},
        "anticorrelated": {"marker": "v", "label": "Area-anticorrelated"},
    }
    for cong, sty in cong_style.items():
        sub = empirical[empirical["congruency"] == cong].sort_values("ratio")
        ax.errorbar(
            sub["ratio"],
            sub["mean_acc"],
            yerr=sub["se_acc"] * 1.96,
            fmt=f"{sty['marker']}-",
            color=EMP_COLOR,
            capsize=4,
            ms=8,
            lw=2,
            label=sty["label"],
        )

    for dur in sorted(model_preds):
        pred = model_preds[dur].sort_values("ratio")
        label = (
            model_curve_label_fmt.format(dur=dur)
            if model_curve_label_fmt is not None
            else f"Model ({dur} ms params)"
        )
        ax.plot(
            pred["ratio"],
            pred["accuracy"],
            color=MOD_COLOR,
            lw=2.5,
            ms=6,
            marker="s",
            ls=REYNVOET_MODEL_LINESTYLE.get(dur, "-"),
            label=label,
        )

    ax.axhline(50, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Ratio  $n_{large}\\,/\\,n_{small}$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A.  Accuracy vs ratio", fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(40, 105)

    # ── Panel B: Weber fractions ─────────────────────────────────────────
    ax = axes[1]
    emp_w = weber_df[weber_df["source"].str.startswith("Empirical")].reset_index(
        drop=True
    )
    mod_w = weber_df[weber_df["source"].str.startswith("Model")].reset_index(drop=True)

    emp_colors = [EMP_COLOR for _ in emp_w["source"]]

    positions_emp = np.arange(len(emp_w))
    gap = 0.6
    positions_mod = np.arange(len(mod_w)) + len(emp_w) + gap

    bars_emp = ax.bar(
        positions_emp,
        emp_w["weber_fraction"],
        color=emp_colors,
        width=0.65,
        edgecolor="white",
    )
    bars_mod = ax.bar(
        positions_mod,
        mod_w["weber_fraction"],
        color=[MOD_COLOR for _ in mod_w["source"]],
        width=0.65,
        edgecolor="white",
    )

    all_positions = np.concatenate([positions_emp, positions_mod])
    all_labels = [s.replace("Empirical ", "") for s in emp_w["source"]] + [
        s.replace("Model ", "") for s in mod_w["source"]
    ]
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Weber fraction  $w$")
    ax.set_title("B.  Weber fractions", fontweight="bold", loc="left")

    for bars in (bars_emp, bars_mod):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.tight_layout()
    path = output_dir / "reynvoet_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ── Match-to-sample comparison ───────────────────────────────────────────


def plot_mts_comparison(
    by_pair: pd.DataFrame,
    by_ratio: pd.DataFrame,
    by_sample: pd.DataFrame,
    model_preds: dict[int, pd.DataFrame],
    model_by_ratio: dict[int, pd.DataFrame],
    model_sample_eff: dict[int, pd.DataFrame],
    output_dir: Path,
    tag: str = "all",
) -> Path:
    """Four-panel figure for match-to-sample model comparison.

    A: "Different" trial accuracy vs ratio (with error bars)
    B: "Same" trial accuracy by numerosity (with error bars)
    C: Overall accuracy by sample numerosity
    D: Per-pair accuracy scatter (empirical vs model)
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    dur_ls = {160: "--", 320: "-.", 640: ":"}
    dur_gray = {160: "#222222", 320: "#555555", 640: "#888888"}
    title_suffix = "" if tag == "all" else f" ({tag})"

    # ── Panel A: different-trial accuracy vs ratio ────────────────────────
    ax = axes[0, 0]
    br = by_ratio.sort_values("ratio")
    ax.errorbar(
        br["ratio"],
        br["mean_acc"],
        yerr=br["se_acc"] * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
        label="Empirical",
    )
    for dur in sorted(model_by_ratio):
        mr = model_by_ratio[dur].sort_values("ratio")
        ax.errorbar(
            mr["ratio"],
            mr["mean_acc"],
            yerr=mr["se_acc"] * 1.96,
            fmt="s" + dur_ls[dur][0],
            color=dur_gray[dur],
            capsize=3,
            ms=6,
            lw=2.5,
            label=f"Model ({dur} ms)",
        )
    ax.axhline(50, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Ratio  $\\max(n_s, n_t)\\,/\\,\\min(n_s, n_t)$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"A.  'Different' trials{title_suffix}", fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_ylim(40, 105)

    # ── Panel B: same-trial accuracy by numerosity ────────────────────────
    ax = axes[0, 1]
    same_emp = by_pair[by_pair["same"]].sort_values("sample_true_n")
    x_n = same_emp["sample_true_n"].values
    n_dur = len(model_preds)
    bar_w = 0.7 / (1 + n_dur)
    offsets = np.linspace(-0.35 + bar_w / 2, 0.35 - bar_w / 2, 1 + n_dur)

    ax.bar(
        x_n + offsets[0],
        same_emp["accuracy_pct"],
        width=bar_w,
        color=PALETTE[0],
        label="Empirical",
        edgecolor="white",
        yerr=same_emp["se_pct"] * 1.96,
        capsize=3,
        error_kw={"lw": 1},
    )
    for i, dur in enumerate(sorted(model_preds)):
        mp = model_preds[dur]
        same_mod = mp[mp["same"]].sort_values("sample_n")
        ax.bar(
            same_mod["sample_n"].values + offsets[i + 1],
            same_mod["accuracy"],
            width=bar_w,
            color=dur_gray[dur],
            alpha=0.7,
            label=f"Model ({dur} ms)",
            edgecolor="white",
            yerr=same_mod["se"].values * 1.96,
            capsize=3,
            error_kw={"lw": 1},
        )

    ax.axhline(50, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Numerosity  $n$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"B.  'Same' trials{title_suffix}", fontweight="bold", loc="left")
    ax.set_xticks(sorted(x_n))
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # ── Panel C: accuracy by sample numerosity (overall) ──────────────────
    ax = axes[1, 0]
    bs = by_sample.sort_values("sample_n")
    ax.errorbar(
        bs["sample_n"],
        bs["mean_acc"],
        yerr=bs["se_acc"] * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
        label="Empirical",
    )
    for dur in sorted(model_sample_eff):
        ms = model_sample_eff[dur].sort_values("sample_n")
        ax.plot(
            ms["sample_n"],
            ms["mean_acc"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2.5,
            ms=6,
            marker="s",
            label=f"Model ({dur} ms)",
        )
    ax.set_xlabel("Sample numerosity")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(
        f"C.  Accuracy by sample size{title_suffix}", fontweight="bold", loc="left"
    )
    ax.set_xticks(sorted(bs["sample_n"].values))
    ax.legend(fontsize=8)

    # ── Panel D: per-pair scatter (empirical vs model) ────────────────────
    ax = axes[1, 1]
    ref_dur = max(model_preds.keys())
    mp = model_preds[ref_dur]

    merged = by_pair.merge(
        mp.rename(
            columns={
                "accuracy": "model_acc",
                "se": "model_se",
                "sample_n": "sample_true_n",
                "target_n": "target_true_n",
            }
        ),
        on=["sample_true_n", "target_true_n"],
        suffixes=("", "_mod"),
    )

    same_m = merged[merged["same"]]
    diff_m = merged[~merged["same"]]

    ax.errorbar(
        same_m["accuracy_pct"],
        same_m["model_acc"],
        xerr=same_m["se_pct"] * 1.96,
        yerr=same_m["model_se"] * 1.96,
        fmt="o",
        color=PALETTE[0],
        ms=8,
        capsize=3,
        zorder=5,
        label="Same pairs",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax.errorbar(
        diff_m["accuracy_pct"],
        diff_m["model_acc"],
        xerr=diff_m["se_pct"] * 1.96,
        yerr=diff_m["model_se"] * 1.96,
        fmt="^",
        color=PALETTE[3],
        ms=8,
        capsize=3,
        zorder=5,
        label="Different pairs",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    for _, r in merged.iterrows():
        lbl = f"{int(r.sample_true_n)},{int(r.target_true_n)}"
        ax.annotate(
            lbl,
            (r.accuracy_pct, r.model_acc),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )

    lims = [20, 105]
    ax.plot(lims, lims, "k:", lw=1, alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Empirical accuracy (%)")
    ax.set_ylabel(f"Model accuracy (%, {ref_dur} ms)")
    ax.set_title(
        f"D.  Per-pair comparison{title_suffix}", fontweight="bold", loc="left"
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = output_dir / f"mts_comparison_{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Dot enumeration comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_dot_enum_comparison(
    by_pair: pd.DataFrame,
    by_distance: pd.DataFrame,
    by_same_n: pd.DataFrame,
    rt_by_n: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    model_preds: dict[int, pd.DataFrame],
    model_by_dist: dict[int, pd.DataFrame],
    model_same: dict[int, pd.DataFrame],
    model_difficulty: dict[int, pd.DataFrame],
    output_dir: Path,
    tag: str = "all",
) -> Path:
    """Four-panel figure for dot-enumeration model comparison.

    A: "Same" trial accuracy by numerosity
    B: Empirical RT vs model cleanup steps (dual axes)
    C: Per-pair scatter (empirical vs model)
    D: Model confusion matrix heatmap
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    dur_ls = {160: "--", 320: "-.", 640: ":"}
    dur_gray = {160: "#222222", 320: "#555555", 640: "#888888"}
    title_suffix = "" if tag == "all" else f" ({tag})"

    # ── Panel A: "same" accuracy by numerosity ────────────────────────────
    ax = axes[0, 0]
    sn = by_same_n.sort_values("n")
    ax.errorbar(
        sn["n"],
        sn["mean_acc"],
        yerr=sn["se_acc"] * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
        label="Empirical",
    )
    for dur in sorted(model_same):
        ms = model_same[dur].sort_values("n")
        ax.errorbar(
            ms["n"],
            ms["mean_acc"],
            yerr=ms["se_acc"] * 1.96,
            fmt="s" + dur_ls[dur][0],
            color=dur_gray[dur],
            capsize=3,
            ms=6,
            lw=2,
            label=f"Model ({dur} ms)",
        )
    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"A.  'Same' trials{title_suffix}", fontweight="bold", loc="left")
    ax.set_xticks(range(1, 10))
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # ── Panel B: empirical RT vs model entropy (dual y-axis) ───────────────
    ax = axes[0, 1]
    rn = rt_by_n.sort_values("n")
    ax.errorbar(
        rn["n"],
        rn["mean_rt"] / 1000,
        yerr=rn["se_rt"] / 1000 * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
        label="Empirical RT",
    )
    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Mean RT (s)", color=PALETTE[0])
    ax.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax.twinx()
    for dur in sorted(model_difficulty):
        md = model_difficulty[dur]
        ax2.plot(
            md["n"],
            md["mean_entropy"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2,
            ms=6,
            marker="s",
            label=f"Model entropy ({dur} ms)",
        )
    ax2.set_ylabel("Mean posterior entropy (nats)", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.set_xticks(range(1, 10))
    ax.set_title(
        f"B.  RT vs model entropy{title_suffix}", fontweight="bold", loc="left"
    )

    # ── Panel C: per-pair scatter (empirical vs model 640 ms) ─────────────
    ax = axes[1, 0]
    ref_dur = max(model_preds.keys())
    mp = model_preds[ref_dur]

    merged = by_pair.merge(
        mp.rename(
            columns={
                "accuracy": "model_acc",
                "se": "model_se",
            }
        ),
        on=["true_n", "shown_digit"],
        suffixes=("", "_mod"),
    )

    same_m = merged[merged["same"]]
    diff_m = merged[~merged["same"]]

    ax.errorbar(
        same_m["accuracy_pct"],
        same_m["model_acc"],
        xerr=same_m["se_pct"] * 1.96,
        yerr=same_m["model_se"] * 1.96,
        fmt="o",
        color=PALETTE[0],
        ms=7,
        capsize=3,
        zorder=5,
        label="Same",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax.errorbar(
        diff_m["accuracy_pct"],
        diff_m["model_acc"],
        xerr=diff_m["se_pct"] * 1.96,
        yerr=diff_m["model_se"] * 1.96,
        fmt="^",
        color=PALETTE[3],
        ms=7,
        capsize=3,
        zorder=5,
        label="Different",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    for _, r in merged.iterrows():
        lbl = f"{int(r.true_n)},{int(r.shown_digit)}"
        ax.annotate(
            lbl,
            (r.accuracy_pct, r.model_acc),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6,
        )

    lims = [0, 105]
    ax.plot(lims, lims, "k:", lw=1, alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Empirical accuracy (%)")
    ax.set_ylabel(f"Model accuracy (%, {ref_dur} ms)")
    ax.set_title(
        f"C.  Per-pair comparison{title_suffix}", fontweight="bold", loc="left"
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # ── Panel D: model confusion matrix heatmap ───────────────────────────
    ax = axes[1, 1]
    pivot = confusion_matrix.pivot(
        index="true_n",
        columns="map_n",
        values="prob",
    )
    im = ax.imshow(
        pivot.values,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="equal",
        origin="lower",
    )
    ticks = list(range(len(pivot.index)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(ticks)
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_xlabel("MAP estimate")
    ax.set_ylabel("True n")
    ax.set_title("D.  Model confusion matrix (640 ms)", fontweight="bold", loc="left")

    for i, row_n in enumerate(pivot.index):
        for j, col_n in enumerate(pivot.columns):
            v = pivot.values[i, j]
            if v >= 0.01:
                color = "white" if v > 0.5 else "black"
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color
                )

    fig.colorbar(im, ax=ax, shrink=0.8, label="P(MAP = k | true n)")

    fig.tight_layout()
    path = output_dir / f"dot_enum_{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Symbolic comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_symbolic_comparison(
    by_pair: pd.DataFrame,
    by_distance_nz: pd.DataFrame,
    by_distance_full: pd.DataFrame,
    by_ratio: pd.DataFrame,
    model_preds: dict[int, pd.DataFrame],
    model_by_dist: dict[int, pd.DataFrame],
    model_by_ratio: dict[int, pd.DataFrame],
    output_dir: Path,
    tag: str = "all",
) -> Path:
    """Five-panel figure for symbolic comparison.

    A: Accuracy vs distance (empirical with/without 0 + model)
    B: RT vs model entropy by distance
    C: Accuracy vs ratio (empirical + model)
    D: Per-pair accuracy scatter
    E: Per-pair RT vs model entropy scatter
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    dur_ls = {160: "--", 320: "-.", 640: ":"}
    dur_gray = {160: "#222222", 320: "#555555", 640: "#888888"}
    title_suffix = "" if tag == "all" else f" ({tag})"

    # ── Panel A: accuracy vs distance ─────────────────────────────────────
    ax = axes[0, 0]

    bf = by_distance_full.sort_values("distance")
    ax.errorbar(
        bf["distance"],
        bf["mean_acc"],
        yerr=bf["se_acc"] * 1.96,
        fmt="o-",
        color=PALETTE[2],
        capsize=4,
        ms=6,
        lw=1.5,
        alpha=0.5,
        label="Empirical (incl. 0)",
    )

    bn = by_distance_nz.sort_values("distance")
    ax.errorbar(
        bn["distance"],
        bn["mean_acc"],
        yerr=bn["se_acc"] * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
        label="Empirical (no 0)",
    )

    for dur in sorted(model_by_dist):
        md = model_by_dist[dur].sort_values("distance")
        ax.errorbar(
            md["distance"],
            md["mean_acc"],
            yerr=md["se_acc"] * 1.96,
            fmt="s" + dur_ls[dur][0],
            color=dur_gray[dur],
            capsize=3,
            ms=6,
            lw=2,
            label=f"Model ({dur} ms)",
        )
    ax.axhline(50, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Distance  |larger − smaller|")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"A.  Distance effect{title_suffix}", fontweight="bold", loc="left")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(50, 105)

    # ── Panel B: RT vs model entropy by distance ──────────────────────────
    ax = axes[0, 1]
    bn_rt = by_distance_nz.sort_values("distance")
    ax.errorbar(
        bn_rt["distance"],
        bn_rt["mean_rt"] / 1000,
        yerr=bn_rt["se_rt"] / 1000 * 1.96,
        fmt="o-",
        color=PALETTE[0],
        capsize=4,
        ms=8,
        lw=2,
        label="Empirical RT (no 0)",
    )
    ax.set_xlabel("Distance  |larger − smaller|")
    ax.set_ylabel("Mean RT (s)", color=PALETTE[0])
    ax.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax.twinx()
    for dur in sorted(model_by_dist):
        md = model_by_dist[dur].sort_values("distance")
        ax2.plot(
            md["distance"],
            md["mean_entropy"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2,
            ms=6,
            marker="s",
            label=f"Entropy ({dur} ms)",
        )
    ax2.set_ylabel("Mean posterior entropy (nats)", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    ax.set_title(
        f"B.  RT vs model entropy{title_suffix}", fontweight="bold", loc="left"
    )

    # ── Panel C: accuracy vs ratio ────────────────────────────────────────
    ax = axes[0, 2]
    br = by_ratio.sort_values("ratio")
    ax.errorbar(
        br["ratio"],
        br["mean_acc"],
        yerr=br["se_acc"] * 1.96,
        fmt="o",
        color=PALETTE[0],
        capsize=3,
        ms=5,
        lw=0,
        alpha=0.6,
        label="Empirical",
    )
    for dur in sorted(model_by_ratio):
        mr = model_by_ratio[dur].sort_values("ratio")
        ax.plot(
            mr["ratio"],
            mr["mean_acc"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2,
            ms=5,
            marker="s",
            label=f"Model ({dur} ms)",
        )
    ax.axhline(50, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Ratio  larger / smaller")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"C.  Ratio effect{title_suffix}", fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_ylim(50, 105)

    # ── Panel D: per-pair accuracy scatter ──────────────────────────────────
    ax = axes[1, 0]
    ref_dur = max(model_preds.keys())
    mp = model_preds[ref_dur]

    merged = by_pair.merge(
        mp.rename(
            columns={
                "accuracy": "model_acc",
                "se": "model_se",
            }
        ),
        on=["smaller", "larger"],
        suffixes=("", "_mod"),
    )

    sc = ax.scatter(
        merged["accuracy_pct"],
        merged["model_acc"],
        c=merged["distance"],
        cmap="viridis",
        s=50,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        vmin=1,
        vmax=8,
    )
    ax.errorbar(
        merged["accuracy_pct"],
        merged["model_acc"],
        xerr=merged["se_pct"] * 1.96,
        yerr=merged["model_se"] * 1.96,
        fmt="none",
        ecolor="0.7",
        elinewidth=0.5,
        capsize=0,
        zorder=3,
    )
    for _, r in merged.iterrows():
        lbl = f"{int(r.smaller)},{int(r.larger)}"
        ax.annotate(
            lbl,
            (r.accuracy_pct, r.model_acc),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=5.5,
        )

    lims = [50, 105]
    ax.plot(lims, lims, "k:", lw=1, alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Empirical accuracy (%)")
    ax.set_ylabel(f"Model accuracy (%, {ref_dur} ms)")
    ax.set_title(
        f"D.  Per-pair comparison{title_suffix}", fontweight="bold", loc="left"
    )
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Distance")
    ax.set_aspect("equal")

    # ── Panel E: per-pair RT vs model entropy scatter ─────────────────────
    ax = axes[1, 1]
    ref_dur = max(model_preds.keys())
    mp = model_preds[ref_dur]

    merged_rt = by_pair.merge(
        mp.rename(
            columns={
                "accuracy": "model_acc",
                "se": "model_se",
            }
        ),
        on=["smaller", "larger"],
        suffixes=("", "_mod"),
    )

    sc2 = ax.scatter(
        merged_rt["mean_rt"] / 1000,
        merged_rt["mean_entropy"],
        c=merged_rt["distance"],
        cmap="viridis",
        s=50,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        vmin=1,
        vmax=8,
    )
    ax.errorbar(
        merged_rt["mean_rt"] / 1000,
        merged_rt["mean_entropy"],
        xerr=merged_rt["se_rt"] / 1000 * 1.96,
        fmt="none",
        ecolor="0.7",
        elinewidth=0.5,
        capsize=0,
        zorder=3,
    )
    for _, r in merged_rt.iterrows():
        lbl = f"{int(r.smaller)},{int(r.larger)}"
        ax.annotate(
            lbl,
            (r.mean_rt / 1000, r.mean_entropy),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=5.5,
        )

    ax.set_xlabel("Empirical mean RT (s)")
    ax.set_ylabel(f"Model entropy (nats, {ref_dur} ms)")
    ax.set_title(
        f"E.  Per-pair RT comparison{title_suffix}", fontweight="bold", loc="left"
    )
    fig.colorbar(sc2, ax=ax, shrink=0.8, label="Distance")

    # ── Hide unused panel ─────────────────────────────────────────────────
    axes[1, 2].set_visible(False)

    fig.tight_layout()
    path = output_dir / f"symbolic_comparison_{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# DeCarli et al. comparisons
# ═══════════════════════════════════════════════════════════════════════════

_DC_DD_COLOR = "#d62728"
_DC_CTRL_COLOR = "#2ca02c"

# Short labels for DeCarli parameter facet panels (fit figure + boxfacet export)
_DECARLI_FACET_PARAM_LABELS: dict[str, str] = {
    "sigma_c": r"$c$",
    "lambda0": r"$\lambda_0$",
    "kappa": r"$\kappa$",
    "tau_in": r"$\tau_{\mathrm{in}}$",
    "s00": r"$s_{00}$",
    "s0": r"$s_0$",
    "t0": r"$t_0$",
    "c_steps": r"$c_{\mathrm{steps}}$",
    "beta": r"$\beta$",
}


def _decarli_facet_param_label(p: str) -> str:
    return _DECARLI_FACET_PARAM_LABELS.get(p, p)


def plot_decarli_enumeration(
    empirical: pd.DataFrame,
    model_preds: dict[int, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Two-panel figure: exact enumeration accuracy and RT by numerosity.

    Empirical: DD and control groups (from DeCarli et al.).
    Model: predictions for multiple durations.
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_min = int(empirical["Numerosity"].min())
    n_max = int(empirical["Numerosity"].max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dur_ls = {160: "--", 320: "-.", 640: ":"}
    dur_gray = {160: "#222222", 320: "#555555", 640: "#888888"}

    # ── Panel A: accuracy by numerosity ───────────────────────────────────
    ax = axes[0]
    for group, color, label in [
        ("DD", _DC_DD_COLOR, "DD (dyscalculia)"),
        ("contr", _DC_CTRL_COLOR, "Controls"),
    ]:
        sub = empirical[empirical["Group"] == group].sort_values("Numerosity")
        ax.errorbar(
            sub["Numerosity"],
            sub["mean_acc_pct"],
            yerr=sub["se_acc_pct"] * 1.96,
            fmt="o-",
            color=color,
            capsize=4,
            ms=8,
            lw=2,
            label=label,
        )

    for dur in sorted(model_preds):
        mp = model_preds[dur].sort_values("Numerosity")
        ax.plot(
            mp["Numerosity"],
            mp["exact_acc"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2.5,
            ms=6,
            marker="s",
            label=f"Model ({dur} ms)",
        )

    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_title("A.  Exact enumeration accuracy", fontweight="bold", loc="left")
    ax.set_xticks(range(n_min, n_max + 1))
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # ── Panel B: RT vs model entropy ──────────────────────────────────────
    ax = axes[1]
    for group, color, label in [
        ("DD", _DC_DD_COLOR, "DD RT"),
        ("contr", _DC_CTRL_COLOR, "Control RT"),
    ]:
        sub = empirical[empirical["Group"] == group].sort_values("Numerosity")
        ax.errorbar(
            sub["Numerosity"],
            sub["mean_rt"],
            yerr=sub["se_rt"] * 1.96,
            fmt="o-",
            color=color,
            capsize=4,
            ms=8,
            lw=2,
            label=label,
        )

    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Mean RT (s)")
    ax.tick_params(axis="y")

    ax2 = ax.twinx()
    for dur in sorted(model_preds):
        mp = model_preds[dur].sort_values("Numerosity")
        ax2.plot(
            mp["Numerosity"],
            mp["mean_entropy"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2,
            ms=6,
            marker="s",
            label=f"Entropy ({dur} ms)",
        )
    ax2.set_ylabel("Model posterior entropy (nats)", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.set_xticks(range(n_min, n_max + 1))
    ax.set_title("B.  RT vs model entropy", fontweight="bold", loc="left")

    fig.suptitle(
        "DeCarli et al. — Exact enumeration: DD vs Controls vs Model",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path = output_dir / "decarli_enumeration.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_decarli_comparison(
    empirical: pd.DataFrame,
    model_preds: dict[int, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Two-panel figure: symbolic comparison accuracy and RT by distance.

    Empirical: DD and control groups (from DeCarli et al.).
    Model: predictions for multiple durations.
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dur_ls = {160: "--", 320: "-.", 640: ":"}
    dur_gray = {160: "#222222", 320: "#555555", 640: "#888888"}

    # ── Panel A: accuracy by distance ─────────────────────────────────────
    ax = axes[0]
    for group, color, label in [
        ("DD", _DC_DD_COLOR, "DD (dyscalculia)"),
        ("contr", _DC_CTRL_COLOR, "Controls"),
    ]:
        sub = empirical[empirical["Group"] == group].sort_values("Distance")
        ax.errorbar(
            sub["Distance"],
            sub["mean_acc_pct"],
            yerr=sub["se_acc_pct"] * 1.96,
            fmt="o-",
            color=color,
            capsize=4,
            ms=8,
            lw=2,
            label=label,
        )

    for dur in sorted(model_preds):
        mp = model_preds[dur].sort_values("Distance")
        ax.plot(
            mp["Distance"],
            mp["mean_acc"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2.5,
            ms=6,
            marker="s",
            label=f"Model ({dur} ms)",
        )

    ax.axhline(50, color="0.7", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Distance  |larger − smaller|")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A.  Symbolic comparison accuracy", fontweight="bold", loc="left")
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(50, 105)

    # ── Panel B: RT vs model entropy ──────────────────────────────────────
    ax = axes[1]
    for group, color, label in [
        ("DD", _DC_DD_COLOR, "DD RT"),
        ("contr", _DC_CTRL_COLOR, "Control RT"),
    ]:
        sub = empirical[empirical["Group"] == group].sort_values("Distance")
        ax.errorbar(
            sub["Distance"],
            sub["mean_rt"],
            yerr=sub["se_rt"] * 1.96,
            fmt="o-",
            color=color,
            capsize=4,
            ms=8,
            lw=2,
            label=label,
        )

    ax.set_xlabel("Distance  |larger − smaller|")
    ax.set_ylabel("Mean RT (ms)")

    ax2 = ax.twinx()
    for dur in sorted(model_preds):
        mp = model_preds[dur].sort_values("Distance")
        ax2.plot(
            mp["Distance"],
            mp["mean_entropy"],
            dur_ls[dur],
            color=dur_gray[dur],
            lw=2,
            ms=6,
            marker="s",
            label=f"Entropy ({dur} ms)",
        )
    ax2.set_ylabel("Model posterior entropy (nats)", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    ax.set_xticks(range(1, 9))
    ax.set_title("B.  RT vs model entropy", fontweight="bold", loc="left")

    fig.suptitle(
        "DeCarli et al. — Symbolic comparison: DD vs Controls vs Model",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path = output_dir / "decarli_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# DeCarli et al. — model FIT (DD vs controls)
# ═══════════════════════════════════════════════════════════════════════════


def _decarli_fit_gof_annotation(good_dd: dict, good_ctrl: dict) -> str:
    lines = []
    for label, g in ("DD", good_dd), ("Controls", good_ctrl):
        r2a = float(g.get("R2_pct_exact", float("nan")))
        r2r = float(g.get("R2_rt", float("nan")))
        aic = float(g.get("AIC", float("nan")))
        lines.append(
            rf"{label}: $R^2_{{\mathrm{{acc}}}}$={r2a:.3f}, "
            rf"$R^2_{{\mathrm{{RT}}}}$={r2r:.3f}, AIC={aic:.1f}"
        )
    return "\n".join(lines)


def plot_decarli_fit(
    emp_with_se: dict[str, pd.DataFrame],
    model_dd: pd.DataFrame,
    model_ctrl: pd.DataFrame,
    output_dir: Path,
    *,
    good_dd: dict | None = None,
    good_ctrl: dict | None = None,
) -> Path:
    """Three-panel figure comparing model fits for DD vs controls.

    Panel A: accuracy by numerosity (data + fits)
    Panel B: RT by numerosity (data + fits)
    Panel C: parameter boxplots in one row (DD vs controls per parameter;
    same summaries as ``plot_decarli_param_boxfacets``).
    """
    mpl.rcParams.update(STYLE)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 6.4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    all_emp = pd.concat(emp_with_se.values())
    n_min = int(all_emp["Numerosity"].min())
    n_max = int(all_emp["Numerosity"].max())
    mdd = model_dd[model_dd["n_shown"].between(n_min, n_max)].sort_values("n_shown")
    mct = model_ctrl[model_ctrl["n_shown"].between(n_min, n_max)].sort_values("n_shown")

    # ── Panel A: accuracy ─────────────────────────────────────────────────
    ax = ax_a
    for group, color, label in [
        ("DD", _DC_DD_COLOR, "DD (data)"),
        ("contr", _DC_CTRL_COLOR, "Controls (data)"),
    ]:
        d = emp_with_se[group].sort_values("Numerosity")
        ax.errorbar(
            d["Numerosity"],
            d["mean_acc_pct"],
            yerr=d["se_acc_pct"] * 1.96,
            fmt="o",
            color=color,
            capsize=4,
            ms=8,
            lw=2,
            label=label,
        )

    ax.plot(
        mdd["n_shown"],
        mdd["pct_exact"],
        "s--",
        color=_DC_DD_COLOR,
        ms=6,
        lw=2,
        alpha=0.7,
        label="DD (fit)",
    )
    ax.plot(
        mct["n_shown"],
        mct["pct_exact"],
        "s--",
        color=_DC_CTRL_COLOR,
        ms=6,
        lw=2,
        alpha=0.7,
        label="Controls (fit)",
    )

    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_title("A.  Accuracy", fontweight="bold", loc="left")
    ax.set_xticks(range(n_min, n_max + 1))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    if good_dd is not None and good_ctrl is not None:
        ax.text(
            0.02,
            0.02,
            _decarli_fit_gof_annotation(good_dd, good_ctrl),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.92,
            ),
        )

    # ── Panel B: RT ───────────────────────────────────────────────────────
    ax = ax_b
    for group, color, label in [
        ("DD", _DC_DD_COLOR, "DD (data)"),
        ("contr", _DC_CTRL_COLOR, "Controls (data)"),
    ]:
        d = emp_with_se[group].sort_values("Numerosity")
        ax.errorbar(
            d["Numerosity"],
            d["mean_rt"],
            yerr=d["se_rt"] * 1.96,
            fmt="o",
            color=color,
            capsize=4,
            ms=8,
            lw=2,
            label=label,
        )

    ax.plot(
        mdd["n_shown"],
        mdd["median_rt"],
        "s--",
        color=_DC_DD_COLOR,
        ms=6,
        lw=2,
        alpha=0.7,
        label="DD (fit)",
    )
    ax.plot(
        mct["n_shown"],
        mct["median_rt"],
        "s--",
        color=_DC_CTRL_COLOR,
        ms=6,
        lw=2,
        alpha=0.7,
        label="Controls (fit)",
    )

    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Mean RT (s)")
    ax.set_title("B.  Reaction time", fontweight="bold", loc="left")
    ax.set_xticks(range(n_min, n_max + 1))
    ax.legend(fontsize=8)
    if good_dd is not None and good_ctrl is not None:
        ax.text(
            0.02,
            0.02,
            _decarli_fit_gof_annotation(good_dd, good_ctrl),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.92,
            ),
        )

    # ── Panel C: parameters in one row (bootstrap or CI box summaries) ───────
    p_params, use_boot, row_dd, row_ct, dd_draws, ct_draws = (
        _decarli_load_param_facet_state(output_dir)
    )
    n_p = len(p_params)
    nrows_c, ncols_c = 1, n_p
    wspace_c = 0.1 if ncols_c > 1 else 0.2
    sub = fig.add_subfigure(gs[0, 2])
    sub.suptitle(
        "C.  Parameter comparison",
        fontweight="bold",
        fontsize=STYLE["axes.titlesize"],
        y=1.0,
        ha="center",
        x=0.5,
    )
    inner = sub.add_gridspec(nrows_c, ncols_c, wspace=wspace_c)
    raw_ps = np.array(
        [
            _decarli_raw_pvalue(
                p_params[k],
                use_boot=use_boot,
                row_dd=row_dd,
                row_ct=row_ct,
                dd_draws=dd_draws,
                ct_draws=ct_draws,
            )
            for k in range(n_p)
        ],
        dtype=float,
    )
    p_adj = benjamini_hochberg_adjusted_p(raw_ps)
    for k in range(n_p):
        ax_c = sub.add_subplot(inner[0, k])
        _decarli_draw_param_facet_axis(
            ax_c,
            p_params[k],
            p_value=float(p_adj[k]),
            use_boot=use_boot,
            row_dd=row_dd,
            row_ct=row_ct,
            dd_draws=dd_draws,
            ct_draws=ct_draws,
        )

    fig.suptitle(
        "DeCarli et al. — Model fit: DD vs Controls",
        fontsize=14,
        fontweight="bold",
    )
    path = output_dir / "decarli_fit.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def _five_point_from_ci(lo: float, mid: float, hi: float) -> list[float]:
    """Symmetric ladder inside [lo, hi] for a readable single-summary boxplot."""
    return [
        float(lo),
        float(0.5 * (lo + mid)),
        float(mid),
        float(0.5 * (mid + hi)),
        float(hi),
    ]


def _decarli_wald_z_pvalue(
    m_dd: float, m_ct: float, se_dd: float, se_ct: float
) -> float:
    var = float(se_dd) ** 2 + float(se_ct) ** 2
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    z = (m_dd - m_ct) / np.sqrt(var)
    return float(2 * (1 - norm.cdf(abs(z))))


def benjamini_hochberg_adjusted_p(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR (same family as R ``p.adjust(..., "BH")``; SciPy)."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    fin = np.isfinite(p) & (p >= 0) & (p <= 1)
    if not fin.any():
        o = p.copy()
        o[~fin] = float("nan")
        return o
    p_fill = np.where(fin, p, 1.0)
    p_fill = np.clip(p_fill, 0.0, 1.0)
    out = false_discovery_control(p_fill, method="bh")
    if not fin.all():
        out = out.copy()
        out[~fin] = float("nan")
    return out


def _decarli_build_pair_data(
    p: str,
    *,
    use_boot: bool,
    row_dd: pd.Series | None,
    row_ct: pd.Series | None,
    dd_draws: pd.DataFrame | None,
    ct_draws: pd.DataFrame | None,
) -> list:
    if use_boot:
        assert dd_draws is not None and ct_draws is not None
        return [
            dd_draws[p].dropna().to_numpy(dtype=float),
            ct_draws[p].dropna().to_numpy(dtype=float),
        ]
    assert row_dd is not None and row_ct is not None
    return [
        _five_point_from_ci(
            float(row_dd[f"{p}_ci_lo"]),
            float(row_dd[p]),
            float(row_dd[f"{p}_ci_hi"]),
        ),
        _five_point_from_ci(
            float(row_ct[f"{p}_ci_lo"]),
            float(row_ct[p]),
            float(row_ct[f"{p}_ci_hi"]),
        ),
    ]


def _decarli_raw_pvalue(
    p: str,
    *,
    use_boot: bool,
    row_dd: pd.Series | None,
    row_ct: pd.Series | None,
    dd_draws: pd.DataFrame | None,
    ct_draws: pd.DataFrame | None,
) -> float:
    if use_boot:
        data = _decarli_build_pair_data(
            p,
            use_boot=True,
            row_dd=row_dd,
            row_ct=row_ct,
            dd_draws=dd_draws,
            ct_draws=ct_draws,
        )
        return _decarli_mannwhitney_pvalue(data[0], data[1])
    assert row_dd is not None and row_ct is not None
    return _decarli_wald_z_pvalue(
        float(row_dd[p]),
        float(row_ct[p]),
        float(row_dd[f"{p}_se"]),
        float(row_ct[f"{p}_se"]),
    )


def _decarli_mannwhitney_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    res = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    return float(res.pvalue)


def _format_decarli_hypothesis_line(p: float) -> str:
    if not np.isfinite(p):
        return "p=—"
    if p < 0.001:
        base = "p<0.001"
    else:
        base = f"p={p:.3f}"
    if p >= 0.05:
        return f"{base} (n.s.)"
    if p < 0.001:
        return f"{base} ***"
    if p < 0.01:
        return f"{base} **"
    return f"{base} *"


def _decarli_significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p >= 0.05:
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _decarli_boxplot_data_ylim_range(data, bp: dict) -> tuple[float, float]:
    y_min = min(float(np.min(d)) for d in data)
    y_max = max(float(np.max(d)) for d in data)
    for w in bp.get("whiskers", []):
        yd = w.get_ydata()
        if len(yd) > 0:
            y_min = min(y_min, float(np.min(yd)))
            y_max = max(y_max, float(np.max(yd)))
    for fl in bp.get("fliers", []):
        yd = fl.get_ydata()
        if len(yd) > 0:
            y_min = min(y_min, float(np.min(yd)))
            y_max = max(y_max, float(np.max(yd)))
    return y_min, y_max


def _decarli_load_param_facet_state(
    output_dir: Path,
    *,
    fitted_path: Path | None = None,
    boot_dd_path: Path | None = None,
    boot_ctrl_path: Path | None = None,
) -> tuple[
    list[str],
    bool,
    pd.Series | None,
    pd.Series | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    output_dir = Path(output_dir)
    fitted_path = fitted_path or output_dir / "fitted_params.csv"
    boot_dd_path = boot_dd_path or output_dir / "bootstrap_draws_DD.csv"
    boot_ctrl_path = boot_ctrl_path or output_dir / "bootstrap_draws_controls.csv"

    use_boot = boot_dd_path.is_file() and boot_ctrl_path.is_file()
    dd_draws: pd.DataFrame | None = None
    ct_draws: pd.DataFrame | None = None
    row_dd: pd.Series | None = None
    row_ct: pd.Series | None = None

    if use_boot:
        dd_draws = pd.read_csv(boot_dd_path)
        ct_draws = pd.read_csv(boot_ctrl_path)
        if len(dd_draws) < 2 or len(ct_draws) < 2:
            use_boot = False
            dd_draws = None
            ct_draws = None

    if not use_boot:
        fp = pd.read_csv(fitted_path)
        row_dd = fp[fp["group"] == "DD"].iloc[0]
        row_ct = fp[fp["group"] == "Controls"].iloc[0]

    params: list[str] = []
    for p in PARAM_NAMES:
        if use_boot:
            assert dd_draws is not None and ct_draws is not None
            if p in dd_draws.columns and p in ct_draws.columns:
                params.append(p)
        else:
            assert row_dd is not None and row_ct is not None
            if (
                p in row_dd.index
                and f"{p}_ci_lo" in row_dd.index
                and f"{p}_ci_hi" in row_dd.index
                and f"{p}_se" in row_dd.index
                and f"{p}_se" in row_ct.index
            ):
                params.append(p)

    if not params:
        raise ValueError("No overlapping parameters to plot for DeCarli comparison")

    return params, use_boot, row_dd, row_ct, dd_draws, ct_draws


def _decarli_draw_param_facet_axis(
    ax,
    p: str,
    *,
    p_value: float | None,
    use_boot: bool,
    row_dd: pd.Series | None,
    row_ct: pd.Series | None,
    dd_draws: pd.DataFrame | None,
    ct_draws: pd.DataFrame | None,
    dd_color: str | tuple = _DC_DD_COLOR,
    ctrl_color: str | tuple = _DC_CTRL_COLOR,
) -> None:
    data = _decarli_build_pair_data(
        p,
        use_boot=use_boot,
        row_dd=row_dd,
        row_ct=row_ct,
        dd_draws=dd_draws,
        ct_draws=ct_draws,
    )
    box_width = 0.65
    pos_l, pos_r = 1.0, 2.0
    bp = ax.boxplot(
        data,
        tick_labels=["DD", "Control"],
        patch_artist=True,
        widths=box_width,
        positions=[pos_l, pos_r],
        medianprops={"color": "0.2", "linewidth": 1.2},
    )
    for patch, color in zip(bp["boxes"], (dd_color, ctrl_color)):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    tsize = float(mpl.rcParams["axes.labelsize"])
    tick_size = float(
        min(mpl.rcParams["xtick.labelsize"], mpl.rcParams["ytick.labelsize"])
    )
    ax.set_title(
        _decarli_facet_param_label(p),
        fontweight="bold",
        fontsize=tsize,
        pad=16,
    )
    ax.tick_params(axis="x", labelsize=tick_size, pad=2)
    ax.tick_params(axis="y", labelsize=tick_size)

    y_lo, y_hi = _decarli_boxplot_data_ylim_range(data, bp)
    y_span = y_hi - y_lo if y_hi > y_lo else max(abs(y_hi), 1.0) * 0.05
    y0, _ = ax.get_ylim()
    if p_value is not None:
        y_hline = y_hi + 0.1 * y_span
        ax.plot(
            [pos_l, pos_r],
            [y_hline, y_hline],
            color="0.25",
            lw=0.9,
            clip_on=False,
            zorder=5,
        )
        y_stars = y_hline + 0.05 * y_span
        ax.text(
            0.5 * (pos_l + pos_r),
            y_stars,
            _decarli_significance_stars(p_value),
            ha="center",
            va="bottom",
            fontsize=tsize * 0.88,
            color="0.2",
            zorder=6,
        )
        y_top_plot = y_stars + 0.12 * y_span
    else:
        y_top_plot = y_hi + 0.08 * y_span
    y_bottom = min(y0, y_lo) - 0.02 * y_span
    ax.set_ylim(y_bottom, y_top_plot)


def plot_decarli_param_boxfacets(
    output_dir: Path,
    *,
    fitted_path: Path | None = None,
    boot_dd_path: Path | None = None,
    boot_ctrl_path: Path | None = None,
    out_name: str = "decarli_param_boxfacets.png",
) -> Path:
    """One row of boxplots: each free parameter, DD vs Controls.

    If ``bootstrap_draws_DD.csv`` and ``bootstrap_draws_controls.csv`` exist
    (from ``fit_decarli.py --bootstrap`` without ``--only-tau``), uses one
    box per group from bootstrap refits; the DD–control p-value is the
    two-sided Mann--Whitney U with an asymptotic (normal) null distribution
    (``method='asymptotic'`` in SciPy, tie and continuity corrections).
    Otherwise builds each box from the 95% CI in ``fitted_params.csv``
    (five-point summary along the interval) and uses a two-sample Wald test.
    """
    mpl.rcParams.update(STYLE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params, use_boot, row_dd, row_ct, dd_draws, ct_draws = (
        _decarli_load_param_facet_state(
            output_dir,
            fitted_path=fitted_path,
            boot_dd_path=boot_dd_path,
            boot_ctrl_path=boot_ctrl_path,
        )
    )

    if use_boot:
        assert dd_draws is not None and ct_draws is not None
        note = (
            f"Subject-level bootstrap refits "
            f"(n={len(dd_draws)} DD, {len(ct_draws)} controls)"
        )
        h_test_note = (
            "DD vs controls, two-sided Mann–Whitney U, asymptotic normal p-value "
            "(tie-corrected; continuity correction) on bootstrap refit draws per group."
        )
    else:
        note = "95% CI from fit (Hessian/profile); boxes are a five-point summary"
        h_test_note = (
            "DD vs controls, two-sided Wald z-test (pooled SEs from Hessian / profile)."
        )

    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n + 1.2, 3.5), squeeze=False)

    raw_ps = np.array(
        [
            _decarli_raw_pvalue(
                p,
                use_boot=use_boot,
                row_dd=row_dd,
                row_ct=row_ct,
                dd_draws=dd_draws,
                ct_draws=ct_draws,
            )
            for p in params
        ],
        dtype=float,
    )
    p_adj = benjamini_hochberg_adjusted_p(raw_ps)
    fdr_note = "Multiple testing: FDR (Benjamini–Hochberg) across parameters."
    for idx, p in enumerate(params):
        _decarli_draw_param_facet_axis(
            axes[0][idx],
            p,
            p_value=float(p_adj[idx]),
            use_boot=use_boot,
            row_dd=row_dd,
            row_ct=row_ct,
            dd_draws=dd_draws,
            ct_draws=ct_draws,
        )

    fig.suptitle(
        "DeCarli fit — parameters (DD vs controls)\n"
        + note
        + "\n"
        + fdr_note
        + "\n"
        + h_test_note,
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.12,
        top=0.79,
        wspace=0.14,
    )
    path = output_dir / out_name
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path
