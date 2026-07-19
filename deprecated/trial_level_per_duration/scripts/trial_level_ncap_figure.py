"""Small-range robustness figure (Poisson-free), replotted from cached CSVs.

Reads the legacy file ``ncap10_ncrit.csv`` (written by
``trial_level_ncap_fit.py``) and shows that
refitting the winning model on n <= 10 alone leaves the width exponent beta and the
continuous width index n_width essentially unchanged versus the full-range n <= 15 fit
(only the noise-dominated 40 ms flash shifts). Overwrites ``ncap10_comparison.png``
in place, so the supplementary reference is unchanged. No re-fitting.

    .venv/bin/python -m scripts.trial_level_ncap_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_style import C_ATTR, apply_style, panel_label

CSV = _ROOT / "output" / "trial_level_model_comparison" / "ncap10_ncrit.csv"
OUT = _ROOT / "output" / "trial_level_model_comparison" / "ncap10_comparison.png"


def main() -> None:
    apply_style(2.2)   # rendered at \columnwidth, 13 in canvas
    df = pd.read_csv(CSV)
    durs = sorted(df["duration_ms"].unique())
    cap = df[df.regime == "cap"].set_index("duration_ms")
    full = df[df.regime == "full"].set_index("duration_ms")
    x = np.arange(len(durs))
    w = 0.38

    fig, (ax_b, ax_n) = plt.subplots(1, 2, figsize=(13, 5))

    # same model, two fitting ranges -> one hue, filled (full) vs open (capped),
    # matching the caption and the paper-wide role colors (C_ATTR = the winner)
    ax_b.bar(x - w / 2, [full.loc[d, "pow"] for d in durs], w,
             color=C_ATTR, label=r"full range ($n \leq 15$)")
    ax_b.bar(x + w / 2, [cap.loc[d, "pow"] for d in durs], w,
             facecolor="none", edgecolor=C_ATTR, linewidth=1.6,
             label=r"small range ($n \leq 10$)")
    ax_b.set_xticks(x, [f"{d}" for d in durs])
    ax_b.set_xlabel("Exposure (ms)")
    ax_b.set_ylabel(r"Width exponent $p$")
    panel_label(ax_b, "a")

    ax_n.bar(x - w / 2, [full.loc[d, "n_width"] for d in durs], w, color=C_ATTR)
    ax_n.bar(x + w / 2, [cap.loc[d, "n_width"] for d in durs], w,
             facecolor="none", edgecolor=C_ATTR, linewidth=1.6)
    ax_n.set_xticks(x, [f"{d}" for d in durs])
    ax_n.set_xlabel("Exposure (ms)")
    ax_n.set_ylabel(r"Width index $n_{\rm width}$")
    panel_label(ax_n, "b")

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {OUT} and .pdf")


if __name__ == "__main__":
    main()
