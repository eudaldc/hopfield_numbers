#!/usr/bin/env python
"""Generate Figure 4: DeCarli exact enumeration and fitted parameters.

* (a) DeCarli accuracy, (b) DeCarli RT, (c) parameter boxplots
  (same summaries as ``output/decarli_fit/decarli_param_boxfacets``, plasma DD/control
  colours to match the DeCarli line panels).

Saves PDF and PNG to ``figures/fig3_decarli`` (manuscript path)."""

import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_decarli import DECARLI_FIT_NUMEROSITY_MAX, load_group_targets
from scripts.fit_helpers import compute_goodness, simulate
from scripts.plots import (
    STYLE,
    _decarli_draw_param_facet_axis,
    _decarli_load_param_facet_state,
)

OUTPUT_DIR = Path("figures")
DECARLI_FIT_DIR = _ROOT / "output" / "decarli_fit"
FONT_SCALE = 1.4
SHOW_FIT_AIC = False
FITTED_PARAMS_CSV = DECARLI_FIT_DIR / "fitted_params.csv"
ENUM_PATH = _ROOT / "data" / "decarli_exact_enumeration.csv"

_plasma = mpl.colormaps["plasma"]
_DD_DATA = _plasma(0.16)
_DD_FIT = _plasma(0.34)
_CTRL_DATA = _plasma(0.76)
_CTRL_FIT = _plasma(0.91)

_GROUP_STYLES = [
    ("DD", "DD", _DD_DATA, _DD_FIT, "DD (data)", "DD (fit)"),
    ("contr", "Controls", _CTRL_DATA, _CTRL_FIT, "Controls (data)", "Controls (fit)"),
]


def _scaled_style() -> dict:
    """Matplotlib rc equivalent to ``sns.set_theme(font_scale=FONT_SCALE)``."""
    p = dict(STYLE)
    keys = (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    )
    for key in keys:
        base = p.get(key)
        if base is None:
            base = mpl.rcParamsDefault[key]
        if isinstance(base, (int, float)):
            p[key] = base * FONT_SCALE
    return p


def _load_decarli_data() -> tuple[dict, dict, int, int, int]:
    raw = pd.read_csv(ENUM_PATH)
    raw = raw[raw["Numerosity"] <= DECARLI_FIT_NUMEROSITY_MAX]
    emp: dict[str, pd.DataFrame] = {}
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
        emp[group] = agg

    all_emp = pd.concat(emp.values())
    n_min = int(all_emp["Numerosity"].min())
    n_max = int(all_emp["Numerosity"].max())

    fitted = pd.read_csv(FITTED_PARAMS_CSV)
    k_params = int(fitted["k_params"].dropna().iloc[0])
    models: dict[str, pd.DataFrame] = {}
    for _, row in fitted.iterrows():
        theta = row.to_dict()
        label = theta.pop("group")
        del theta["loss"]
        model = simulate(theta, n_samples_per_n=5000, seed=73)
        models[label] = model[model["n_shown"].between(n_min, n_max)].sort_values("n_shown")

    return emp, models, n_min, n_max, k_params


def _fit_stats_line(
    label: str, r2: float, aic: float, *, show_aic: bool
) -> str:
    s = rf"{label}: $R^2$ = {r2:.3f}"
    if show_aic:
        s += rf", AIC = {aic:.1f}"
    return s


def _add_fit_stats_text(
    ax: plt.Axes,
    g_dd: dict,
    g_ct: dict,
    *,
    r2_key: str,
    corner: str,
    show_aic: bool = False,
) -> None:
    r2d = float(g_dd.get(r2_key, float("nan")))
    r2c = float(g_ct.get(r2_key, float("nan")))
    aicd = float(g_dd.get("AIC", float("nan")))
    aicc = float(g_ct.get("AIC", float("nan")))
    line_ct = _fit_stats_line("Controls", r2c, aicc, show_aic=show_aic)
    line_dd = _fit_stats_line("DD", r2d, aicd, show_aic=show_aic)
    fs = float(mpl.rcParams["legend.fontsize"]) * 0.92
    line_gap = 0.068
    kw = dict(
        transform=ax.transAxes,
        fontsize=fs,
        zorder=5,
        color="black",
    )
    if corner == "bl_above_legend":
        x, ha = 0.02, "left"
        y0 = 0.30
        ax.text(
            x,
            y0 + line_gap,
            line_ct,
            ha=ha,
            va="bottom",
            **kw,
        )
        ax.text(
            x,
            y0,
            line_dd,
            ha=ha,
            va="bottom",
            **kw,
        )
    elif corner == "tl":
        x, ha = 0.02, "left"
        y_top = 0.97
        ax.text(
            x,
            y_top,
            line_ct,
            ha=ha,
            va="top",
            **kw,
        )
        ax.text(
            x,
            y_top - line_gap,
            line_dd,
            ha=ha,
            va="top",
            **kw,
        )
    else:
        raise ValueError(f"unknown corner: {corner!r}")


def _draw_panel_acc(
    ax: plt.Axes,
    emp: dict,
    models: dict,
    n_min: int,
    n_max: int,
    *,
    g_dd: dict,
    g_ct: dict,
    panel_letter: str = "a",
    show_aic: bool = False,
) -> None:
    for grp_key, fit_key, c_data, c_fit, lbl_data, lbl_fit in _GROUP_STYLES:
        d = emp[grp_key].sort_values("Numerosity")
        ns = d["Numerosity"].values
        m = models[fit_key]

        ax.fill_between(
            ns,
            d["mean_acc_pct"] - d["se_acc_pct"],
            d["mean_acc_pct"] + d["se_acc_pct"],
            color=c_data, alpha=0.12,
        )
        ax.plot(
            ns, d["mean_acc_pct"], "o-",
            color=c_data, ms=5, lw=1.5, label=lbl_data,
            markeredgecolor="white", markeredgewidth=0.3,
        )
        ax.plot(
            m["n_shown"], m["pct_exact"], "s--",
            color=c_fit, ms=4, lw=1.5, label=lbl_fit,
        )

    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_title(panel_letter, fontweight="bold", loc="left", x=0.075)
    ax.set_xticks(range(n_min, n_max + 1))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", ncol=2)
    _add_fit_stats_text(
        ax,
        g_dd,
        g_ct,
        r2_key="R2_pct_exact",
        corner="bl_above_legend",
        show_aic=show_aic,
    )


def _draw_panel_rt(
    ax: plt.Axes,
    emp: dict,
    models: dict,
    n_min: int,
    n_max: int,
    *,
    g_dd: dict,
    g_ct: dict,
    panel_letter: str = "b",
    show_aic: bool = False,
) -> None:
    for grp_key, fit_key, c_data, c_fit, lbl_data, lbl_fit in _GROUP_STYLES:
        d = emp[grp_key].sort_values("Numerosity")
        ns = d["Numerosity"].values
        m = models[fit_key]

        ax.fill_between(
            ns,
            d["mean_rt"] - d["se_rt"],
            d["mean_rt"] + d["se_rt"],
            color=c_data, alpha=0.12,
        )
        ax.plot(
            ns, d["mean_rt"], "o-",
            color=c_data, ms=5, lw=1.5, label=lbl_data,
            markeredgecolor="white", markeredgewidth=0.3,
        )
        ax.plot(
            m["n_shown"], m["median_rt"], "s--",
            color=c_fit, ms=4, lw=1.5, label=lbl_fit,
        )

    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Mean RT (s)")
    ax.set_title(panel_letter, fontweight="bold", loc="left", x=0.075)
    ax.set_xticks(range(n_min, n_max + 1))
    _add_fit_stats_text(
        ax,
        g_dd,
        g_ct,
        r2_key="R2_rt",
        corner="tl",
        show_aic=show_aic,
    )


def _save_fig(fig: plt.Figure, base_name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{base_name}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(_scaled_style())

    print("Loading DeCarli data and parameter posteriors ...", flush=True)
    emp, models, n_min, n_max, k_params = _load_decarli_data()
    targets_dd = load_group_targets("DD")
    targets_contr = load_group_targets("contr")
    g_dd_acc = compute_goodness(
        targets_dd, models["DD"], k=k_params, channels=("pct_exact",)
    )
    g_ct_acc = compute_goodness(
        targets_contr, models["Controls"], k=k_params, channels=("pct_exact",)
    )
    g_dd_rt = compute_goodness(
        targets_dd, models["DD"], k=k_params, channels=("rt",)
    )
    g_ct_rt = compute_goodness(
        targets_contr, models["Controls"], k=k_params, channels=("rt",)
    )
    p_params, use_boot, row_dd, row_ct, dd_draws, ct_draws = (
        _decarli_load_param_facet_state(DECARLI_FIT_DIR)
    )
    n_p = len(p_params)
    wspace = 0.08 if n_p > 1 else 0.12
    print("Drawing fig4_decarli ...", flush=True)
    fig = plt.figure(figsize=(14, 9.0), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.5], hspace=0.14)
    sub_top = fig.add_subfigure(gs[0, 0])
    gs_top = sub_top.add_gridspec(1, 2, wspace=0.1)
    ax_a = sub_top.add_subplot(gs_top[0, 0])
    ax_b = sub_top.add_subplot(gs_top[0, 1])
    _draw_panel_acc(
        ax_a, emp, models, n_min, n_max, g_dd=g_dd_acc, g_ct=g_ct_acc,
        panel_letter="a", show_aic=SHOW_FIT_AIC,
    )
    _draw_panel_rt(
        ax_b, emp, models, n_min, n_max, g_dd=g_dd_rt, g_ct=g_ct_rt,
        panel_letter="b", show_aic=SHOW_FIT_AIC,
    )
    sub_bot = fig.add_subfigure(gs[1, 0])
    sub_bot.text(
        0.01,
        0.98,
        "c",
        fontweight="bold",
        transform=sub_bot.transSubfigure,
        ha="left",
        va="top",
        fontsize=mpl.rcParams["axes.titlesize"],
    )
    gsb = sub_bot.add_gridspec(1, n_p, wspace=wspace)
    for j in range(n_p):
        ax_c = sub_bot.add_subplot(gsb[0, j])
        _decarli_draw_param_facet_axis(
            ax_c,
            p_params[j],
            p_value=None,
            use_boot=use_boot,
            row_dd=row_dd,
            row_ct=row_ct,
            dd_draws=dd_draws,
            ct_draws=ct_draws,
            dd_color=_DD_DATA,
            ctrl_color=_CTRL_DATA,
        )
    _save_fig(fig, "fig4_decarli")
    plt.close(fig)


if __name__ == "__main__":
    main()
