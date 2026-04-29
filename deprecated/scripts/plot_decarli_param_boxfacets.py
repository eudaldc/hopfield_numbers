#!/usr/bin/env python
"""Regenerate parameter boxplots (one row) from saved DeCarli fit outputs."""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plots import plot_decarli_param_boxfacets


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot DD vs control parameters using bootstrap_draws_*.csv if "
            "present, else fitted_params.csv CIs."
        )
    )
    ap.add_argument(
        "output_dir",
        nargs="?",
        default="output/decarli_fit",
        type=Path,
        help="Directory with fitted_params.csv (default: output/decarli_fit)",
    )
    args = ap.parse_args()
    plot_decarli_param_boxfacets(args.output_dir)


if __name__ == "__main__":
    main()
