"""Per-duration RESPONSE-only AIC — curated model set, grouped-bar + mean panel.

Response-only counterpart of ``trial_level_joint_per_duration.py``. The response
distribution is the primary channel for model selection (RT is a secondary
cross-prediction, and the shortest exposure — 40 ms — is too fast to process
reliably, as C&P themselves note), so this figure drops the RT channel entirely.
Without RT the hybrid/entropy split disappears and the set collapses to the six
unique response models (Poisson dropped, as before):

  1. Super-linear Gaussian attractor (free beta)   [winner ★]
  2. Power-law attractor
  3. Super-linear Gaussian readout (no clean-up)
  4. Power-law readout (no clean-up)
  5. Weber/ANS
  6. C&P log-prior proxy

Everything is read from cached response fits (no fitting):
``freewidth_attractor.csv`` (the two attractors), ``well_width_followup.csv`` (the
two static read-outs) and ``per_duration_response_likelihood.csv`` (Weber, C&P).
Same layout as the joint figure: grouped bars of ΔAIC within each duration, plus a
companion panel with the mean over durations.

    .venv/bin/python -m scripts.trial_level_response_per_duration
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

OUT_CSV = T.OUT_DIR / "response_per_duration_aic.csv"
OUT_FIG = T.OUT_DIR / "response_per_duration_aic.png"

# model_key -> which cached CSV holds its per-duration response AIC
SOURCE = {
    "gaussian_freebeta_attractor": "freewidth_attractor.csv",
    "gaussian_pow_attractor": "freewidth_attractor.csv",
    "gaussian_benford_free_beta": "well_width_followup.csv",
    "gaussian_pow_readout": "well_width_followup.csv",
    "weber_ans": "per_duration_response_likelihood.csv",
    "cp_log_prior_proxy": "per_duration_response_likelihood.csv",
}

# bars in plotting order (best -> worst on the summed ladder)
BARS = [
    {"id": "gauss_attr",  "key": "gaussian_freebeta_attractor",
     "label": "Gaussian attractor ★",              "color": "#b2182b"},
    {"id": "pow_attr",    "key": "gaussian_pow_attractor",
     "label": "Power-law attractor",               "color": "#e6550d"},
    {"id": "gauss_static", "key": "gaussian_benford_free_beta",
     "label": "Gaussian readout (no clean-up)",    "color": "#2166ac"},
    {"id": "pow_static",  "key": "gaussian_pow_readout",
     "label": "Power-law readout (no clean-up)",   "color": "#67a9cf"},
    {"id": "weber",       "key": "weber_ans",
     "label": "Weber/ANS",                         "color": "#7f7f7f"},
    {"id": "cp",          "key": "cp_log_prior_proxy",
     "label": "C&P log-prior proxy",               "color": "#252525"},
]


def load_response_aic() -> pd.DataFrame:
    """{model_key -> {duration_ms -> response AIC}} from the cached CSVs."""
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for bar in BARS:
        mk, fname = bar["key"], SOURCE[bar["key"]]
        if fname not in cache:
            cache[fname] = pd.read_csv(T.OUT_DIR / fname)
        sub = cache[fname]
        sub = sub[sub.model_key == mk]
        for _, r in sub.iterrows():
            rows.append({"bar_id": bar["id"], "label": bar["label"],
                         "model_key": mk, "duration_ms": int(r["duration_ms"]),
                         "resp_AIC": float(r["AIC"])})
    df = pd.DataFrame(rows)
    df["delta_AIC"] = df.groupby("duration_ms")["resp_AIC"].transform(lambda s: s - s.min())
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
    ax.set_ylabel(r"$\Delta$AIC within duration (response only)")
    ax.set_title("By exposure duration", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    means = df.groupby("bar_id")["delta_AIC"].mean().reindex(order)
    for i, bid in enumerate(order):
        axm.bar(i, means[bid], width=0.85, color=colors[bid], alpha=0.95)
        axm.text(i, means[bid] + 15, f"{means[bid]:.0f}", ha="center", va="bottom",
                 fontsize=7, rotation=90)
    axm.set_xticks([])
    axm.set_xlabel("mean over\nall durations")
    axm.set_title("Mean", fontweight="bold")
    axm.spines["top"].set_visible(False)
    axm.spines["right"].set_visible(False)
    plt.setp(axm.get_yticklabels(), visible=False)

    fig.suptitle("Trial-level response model comparison", fontweight="bold", y=1.0)
    fig.legend(handles, [labels[b] for b in order], frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    T.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_response_aic()
    df.to_csv(OUT_CSV, index=False)

    print("=== Response-only ΔAIC within duration ===")
    piv = df.pivot_table(index="bar_id", columns="duration_ms", values="delta_AIC")
    print(piv.reindex([b["id"] for b in BARS]).round(1).to_string())
    print("\n=== Mean over durations (of within-duration ΔAIC) ===")
    print(df.groupby("bar_id")["delta_AIC"].mean().reindex([b["id"] for b in BARS])
          .round(1).to_string())

    plot_grouped(df)
    print(f"\nSaved {OUT_CSV.name}, {OUT_FIG.name}")


if __name__ == "__main__":
    main()
