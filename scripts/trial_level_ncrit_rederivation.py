"""Re-derive width-profile crossings for the super-linear-width attractor.

The continuous width index is defined by sigma_n = 1, where the fitted component
width equals unit integer spacing. For Poisson wells sigma_n = sqrt(c n), this
gives n_width = 1/c. For the super-linear wells we solve sigma_n = 1 for n:

  * power-law      sigma_n = c n^p          -> n* = (1/c)^(1/p)
  * Benford free-b sigma_n = (c/freq_n)^b   -> solve (c/freq_n)^b = 1 numerically

We also report where sigma_n reaches 2 and 3 as additional descriptions of the
fitted width law. None of these crossings identifies a topological merger.

    .venv/bin/python -m scripts.trial_level_ncrit_rederivation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT = T.OUT_DIR / "ncrit_rederivation.csv"


def sigma_n(form: str, theta: dict, n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, float)
    if form == "power_law":
        return theta["sigma_c"] * np.power(n, theta["pow"])
    if form == "benford_freebeta":
        freq = np.log10(1.0 + 1.0 / n)
        return np.power(theta["sigma_c"] / freq, theta["beta"])
    raise ValueError(form)


def crossing(form: str, theta: dict, level: float) -> float:
    """Smallest n>=1 (continuous) where sigma_n == level, via fine search."""
    ns = np.linspace(1.0, 60.0, 60000)
    s = sigma_n(form, theta, ns)
    above = np.where(s >= level)[0]
    if above.size == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return 1.0
    # linear interpolation between ns[i-1], ns[i]
    x0, x1, y0, y1 = ns[i - 1], ns[i], s[i - 1], s[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def main() -> None:
    sources = {
        "power_law": (T.OUT_DIR / "pow_theta.csv", ("sigma_c", "pow")),
        "benford_freebeta": (T.OUT_DIR / "winner_theta.csv", ("sigma_c", "beta")),
    }
    rows = []
    for form, (path, pcols) in sources.items():
        if not path.exists():
            print(f"  [skip {form}] {path} not found")
            continue
        df = pd.read_csv(path).set_index("duration_ms")
        for d in T.DURATIONS:
            theta = {"sigma_c": float(df.loc[d, "sigma_c"])}
            theta[pcols[1]] = float(df.loc[d, pcols[1]])
            n_width = crossing(form, theta, 1.0)
            n_sig2 = crossing(form, theta, 2.0)
            n_sig3 = crossing(form, theta, 3.0)
            rows.append({
                "form": form, "duration_ms": d,
                "sigma_c": theta["sigma_c"], "exponent": theta[pcols[1]],
                "n_width(sigma=1)": n_width,
                "n_sigma=2": n_sig2, "n_sigma=3": n_sig3,
                "poisson_1_over_c_ref": 1.0 / theta["sigma_c"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    for form in out.form.unique():
        print(f"\n=== {form} ===")
        sub = out[out.form == form][
            ["duration_ms", "sigma_c", "exponent", "n_width(sigma=1)",
             "n_sigma=2", "n_sigma=3", "poisson_1_over_c_ref"]]
        print(sub.round(2).to_string(index=False))
    print(f"\nSaved {OUT}")
    print("\nReading: sigma=1 gives n_width; sigma=2 and sigma=3 are additional "
          "width-profile crossings; poisson_1_over_c_ref is the corresponding "
          "Poisson-width reference.")


if __name__ == "__main__":
    main()
