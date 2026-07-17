"""Per-duration joint (response + RT) AIC — curated model set, grouped-bar format.

Same grouped-bar layout as the old response-only figure (ΔAIC within each
duration), with a companion panel showing the mean over all durations, and two
changes requested for the revision:

  * **joint** (response + RT), not response-only — so the attractor landscapes
    appear twice, once with the dynamics (hybrid = steps + entropy) RT channel and
    once with the plain entropy RT channel;
  * a **curated set** that drops the Poisson family and keeps the eight models
    that carry the argument:

      1. Super-linear Gaussian attractor + dynamics RT   [winner ★]
      2. Super-linear Gaussian attractor + entropy RT
      3. Power-law attractor + dynamics RT
      4. Power-law attractor + entropy RT
      5. Super-linear Gaussian readout (no clean-up) + entropy RT
      6. Power-law readout (no clean-up) + entropy RT
      7. Weber/ANS + entropy RT
      8. C&P log-prior proxy + entropy RT

Joint AIC per (model, duration) uses the coupled p(r, rt | n) (``T.fit_joint``),
the same definition as ``winner_joint.csv`` and ``per_duration_joint_likelihood``.
Cached joints are reused where available (winner: ``winner_joint.csv``; Weber and
C&P: ``per_duration_joint_likelihood.csv``, entropy). The three free-width models
have no cached RT channel, so they are fit here: the response landscape is refit
with the follow-up settings (deterministic, reproduces the cached response AIC),
then the coupled RT channel is fit on top. Static read-outs have no clean-up
dynamics, so only the entropy RT channel is defined for them.

    .venv/bin/python -m scripts.trial_level_joint_per_duration            # fit + plot
    .venv/bin/python -m scripts.trial_level_joint_per_duration --replot   # figure only
"""

from __future__ import annotations

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

import scripts.trial_level_model_comparison as T

QUAD_FIT, QUAD_EVAL, QUAD_JOINT = 201, 1601, 101
MAXITER, POPSIZE = 35, 10

OUT_CSV = T.LEGACY_OUT_DIR / "joint_per_duration_aic.csv"
OUT_FIG = T.LEGACY_OUT_DIR / "joint_per_duration_aic.png"

# Free-width specs, copied verbatim from the follow-up scripts so a refit here
# reproduces the cached response AIC (same family / bounds / settings / seeds).
SPEC_POW_ATTR = T.ModelSpec(
    key="gaussian_pow_attractor", label="Power-law wells + clean-up",
    family="gaussian_pow", cleanup=True,
    param_names=("sigma_c", "s0", "lambda0", "pow"),
    bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0)),
)
SPEC_GAUSS_STATIC = T.ModelSpec(
    key="gaussian_benford_free_beta", label="Gaussian-Benford readout, free beta",
    family="gaussian_freebeta", cleanup=False,
    param_names=("sigma_c", "s0", "lambda0", "beta"),
    bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
)
SPEC_POW_STATIC = T.ModelSpec(
    key="gaussian_pow_readout", label="Gaussian power-law readout",
    family="gaussian_pow", cleanup=False,
    param_names=("sigma_c", "s0", "lambda0", "pow"),
    bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0)),
)

# The eight bars, in plotting order. `source` says where the joint AIC comes from:
#   winner        -> winner_joint.csv (by metric)
#   cached_joint  -> per_duration_joint_likelihood.csv (entropy)
#   fit           -> refit response + coupled fit_joint here (needs `spec`)
BARS = [
    {"id": "gauss_attr_hybrid",  "label": "Gaussian attractor + dynamics RT ★",
     "model_key": "gaussian_freebeta_attractor", "metric": "hybrid",
     "source": "winner", "color": "#b2182b"},
    {"id": "gauss_attr_entropy", "label": "Gaussian attractor + entropy RT",
     "model_key": "gaussian_freebeta_attractor", "metric": "entropy",
     "source": "winner", "color": "#ef8a62"},
    {"id": "pow_attr_hybrid",    "label": "Power-law attractor + dynamics RT",
     "model_key": "gaussian_pow_attractor", "metric": "hybrid",
     "source": "fit", "spec": SPEC_POW_ATTR, "color": "#e6550d"},
    {"id": "pow_attr_entropy",   "label": "Power-law attractor + entropy RT",
     "model_key": "gaussian_pow_attractor", "metric": "entropy",
     "source": "fit", "spec": SPEC_POW_ATTR, "color": "#fdae6b"},
    {"id": "gauss_static",       "label": "Gaussian readout (no clean-up) + entropy RT",
     "model_key": "gaussian_benford_free_beta", "metric": "entropy",
     "source": "fit", "spec": SPEC_GAUSS_STATIC, "color": "#2166ac"},
    {"id": "pow_static",         "label": "Power-law readout (no clean-up) + entropy RT",
     "model_key": "gaussian_pow_readout", "metric": "entropy",
     "source": "fit", "spec": SPEC_POW_STATIC, "color": "#67a9cf"},
    {"id": "weber",              "label": "Weber/ANS + entropy RT",
     "model_key": "weber_ans", "metric": "entropy",
     "source": "cached_joint", "color": "#7f7f7f"},
    {"id": "cp",                 "label": "C&P log-prior proxy + entropy RT",
     "model_key": "cp_log_prior_proxy", "metric": "entropy",
     "source": "cached_joint", "color": "#252525"},
]


def winner_joint_aic() -> dict:
    df = pd.read_csv(T.LEGACY_OUT_DIR / "winner_joint.csv")
    return {(int(r.duration_ms), r.metric): (float(r.AIC), int(r.k_params))
            for r in df.itertuples()}


def cached_joint_aic(model_key: str, metric: str = "entropy") -> dict:
    df = pd.read_csv(T.LEGACY_OUT_DIR / "per_duration_joint_likelihood.csv")
    df = df[(df.model_key == model_key) & (df.metric == metric)]
    return {int(r.duration_ms): (float(r.AIC), int(r.k_params)) for r in df.itertuples()}


def _fit_cache_path(model_key: str) -> Path:
    return T.LEGACY_OUT_DIR / f"joint_fit_cache_{model_key}.csv"


def fit_model_joint(spec: T.ModelSpec, metrics: tuple[str, ...],
                    data_by_dur: dict, r_max: int) -> dict:
    """Refit the response landscape, then the coupled RT channel per metric.

    Cached per model to ``joint_fit_cache_<key>.csv`` so a re-run (or a timeout
    mid-way through the slow attractor refit) does not repeat completed models.
    Returns {(duration, metric): (joint_AIC, k_params)}."""
    cache = _fit_cache_path(spec.key)
    if cache.exists():
        cdf = pd.read_csv(cache)
        print(f"  [{spec.key}] reusing cached joint fit ({cache.name})", flush=True)
        return {(int(r.duration_ms), r.metric): (float(r.joint_AIC), int(r.k_params))
                for r in cdf.itertuples()}
    quad_fit = T.make_quadrature(QUAD_FIT, kind="grid")
    quad_eval = T.make_quadrature(QUAD_EVAL, kind="grid")
    quad_joint = T.make_quadrature(QUAD_JOINT, kind="grid")
    out: dict = {}
    rows = []
    for d in T.DURATIONS:
        rf = T.fit_response(spec, data_by_dur[d], quad_fit, r_max,
                            maxiter=MAXITER, popsize=POPSIZE, seed=T.SEED + d,
                            quad_eval=quad_eval)
        print(f"  [{spec.key} @ {d}ms] resp AIC={rf['AIC']:.1f}", flush=True)
        for metric in metrics:
            jt = T.fit_joint(spec, data_by_dur[d], rf, quad_joint, r_max,
                             metric=metric, seed=T.SEED + d)
            out[(int(d), metric)] = (float(jt["AIC"]), int(jt["k_params"]))
            rows.append({"duration_ms": int(d), "metric": metric,
                         "joint_AIC": float(jt["AIC"]), "k_params": int(jt["k_params"])})
            print(f"      {metric:7s} joint AIC={jt['AIC']:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(cache, index=False)
    return out


def assemble() -> pd.DataFrame:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    win = winner_joint_aic()

    # Pre-fit the free-width models, cheapest first (static read-outs are seconds;
    # the power-law *attractor* refit is ~7 min/duration) so the fast caches land
    # early and a re-run only ever repeats an unfinished attractor.
    fit_specs = {b["model_key"]: b["spec"] for b in BARS if b["source"] == "fit"}
    fit_order = sorted(fit_specs, key=lambda k: fit_specs[k].cleanup)   # static (False) first
    fit_cache: dict[str, dict] = {}          # model_key -> {(dur, metric): (AIC, k)}
    for mk in fit_order:
        metrics = tuple(b["metric"] for b in BARS
                        if b["source"] == "fit" and b["model_key"] == mk)
        fit_cache[mk] = fit_model_joint(fit_specs[mk], metrics, data_by_dur, r_max)

    rows = []
    for bar in BARS:
        mk, metric = bar["model_key"], bar["metric"]
        if bar["source"] == "winner":
            per_dur = {d: win[(d, metric)] for d in T.DURATIONS}
        elif bar["source"] == "cached_joint":
            per_dur = cached_joint_aic(mk, metric)
        else:  # fit
            per_dur = {d: fit_cache[mk][(d, metric)] for d in T.DURATIONS}
        for d in T.DURATIONS:
            aic, k = per_dur[int(d)]
            rows.append({"bar_id": bar["id"], "label": bar["label"],
                         "model_key": mk, "metric": metric,
                         "duration_ms": int(d), "joint_AIC": aic, "k_params": k})
    return pd.DataFrame(rows)


def add_within_duration_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_AIC"] = df.groupby("duration_ms")["joint_AIC"].transform(lambda s: s - s.min())
    return df


def plot_grouped(df: pd.DataFrame) -> None:
    durations = list(T.DURATIONS)
    order = [b["id"] for b in BARS]
    colors = {b["id"]: b["color"] for b in BARS}
    labels = {b["id"]: b["label"] for b in BARS}
    x = np.arange(len(durations))
    width = 0.8 / len(order)
    offs = (np.arange(len(order)) - (len(order) - 1) / 2) * width

    fig = plt.figure(figsize=(1.9 * len(durations) + 5.6, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(durations), 1.5], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1], sharey=ax)

    handles = []
    for i, bid in enumerate(order):
        sub = df[df.bar_id == bid].set_index("duration_ms").reindex(durations)
        h = ax.bar(x + offs[i], sub["delta_AIC"].to_numpy(), width=width,
                   color=colors[bid], label=labels[bid], alpha=0.95)
        handles.append(h[0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} ms" for d in durations])
    ax.set_ylabel(r"$\Delta$AIC within duration (joint: response + RT)")
    ax.set_title("By exposure duration", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Companion panel: mean over durations of each model's within-duration ΔAIC
    # (i.e. the average height of that model's bars across the five groups).
    means = df.groupby("bar_id")["delta_AIC"].mean().reindex(order)
    for i, bid in enumerate(order):
        axm.bar(i, means[bid], width=0.85, color=colors[bid], alpha=0.95)
        axm.text(i, means[bid] + 20, f"{means[bid]:.0f}", ha="center", va="bottom",
                 fontsize=7, rotation=90)
    axm.set_xticks([])
    axm.set_xlabel("mean over\nall durations")
    axm.set_title("Mean", fontweight="bold")
    axm.spines["top"].set_visible(False)
    axm.spines["right"].set_visible(False)
    plt.setp(axm.get_yticklabels(), visible=False)

    fig.suptitle("Trial-level joint model comparison", fontweight="bold", y=1.0)
    fig.legend(handles, [labels[b] for b in order], frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    T.LEGACY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    replot = "--replot" in sys.argv[1:]
    if replot and OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        print(f"Replotting from cached {OUT_CSV.name}.")
    else:
        df = add_within_duration_delta(assemble())
        df.to_csv(OUT_CSV, index=False)
        print("\n=== Joint AIC per duration (ΔAIC within duration) ===")
        piv = df.pivot_table(index="bar_id", columns="duration_ms", values="delta_AIC")
        piv = piv.reindex([b["id"] for b in BARS])
        print(piv.round(1).to_string())
        print("\n=== Summed joint AIC ladder (over durations) ===")
        lad = (df.groupby(["bar_id", "label"])["joint_AIC"].sum()
               .reset_index().sort_values("joint_AIC"))
        lad["dAIC"] = lad["joint_AIC"] - lad["joint_AIC"].min()
        print(lad[["label", "joint_AIC", "dAIC"]].round(1).to_string(index=False))

    plot_grouped(df)
    print(f"\nSaved {OUT_CSV.name}, {OUT_FIG.name}")


if __name__ == "__main__":
    main()
