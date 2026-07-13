"""Cache the pooled De Carli landscape fit (the `common` mechanism).

`decarli_gaussian_freebeta_subjects.pooled_fit` is a 7-parameter differential-evolution
fit that costs ~50 min. Both `decarli_locus.py` and `decarli_locus_beta.py` need exactly
the same pooled values, and it is deterministic (seed=3), so run it once and cache.

Writes output/decarli_fit/pooled_landscape.csv.

    .venv/bin/python -m scripts.decarli_pooled_cache
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
from scripts.decarli_gaussian_freebeta import DATA, OUT_DIR
from scripts.decarli_gaussian_freebeta_subjects import pooled_fit

CACHE = OUT_DIR / "pooled_landscape.csv"
PNAMES = ("sigma_c", "s0", "lambda0", "pow", "lapse", "t0", "rt_scale")


def load_pooled() -> np.ndarray:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} missing; run `.venv/bin/python -m scripts.decarli_pooled_cache` first")
    row = pd.read_csv(CACHE).iloc[0]
    return np.array([float(row[p]) for p in PNAMES])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)
    pooled = pooled_fit(df, quad)
    pd.DataFrame([dict(zip(PNAMES, pooled))]).to_csv(CACHE, index=False)
    print("pooled:", {k: round(float(v), 6) for k, v in zip(PNAMES, pooled)})
    print(f"saved {CACHE}")


if __name__ == "__main__":
    main()
