#!/usr/bin/env bash
# Full-data fits of the driven family (Stage 5-6). Run from the repo root:
#   bash driven_attractor/run_all_fits.sh
# Each fit uses 12 DE workers; fits run sequentially. Seeds fixed inside.
set -uo pipefail
PY=.venv/bin/python
LOG=driven_attractor/output/fits
mkdir -p "$LOG"

run () {
  echo "=== $* ($(date +%H:%M:%S)) ==="
  $PY driven_attractor/fit_driven.py "$@" 2>&1 | tail -5
}

# primary candidates: soft tether, serial and parallel registration
run --model M2_tether --gamma 0
run --model M2_tether --gamma 1
# accumulation-at-offset control (M1 == M4 for the response channel)
run --model M1_offset --gamma 0
run --model M1_offset --gamma 1
# driven static control (no attractor force)
run --model M3_static --gamma 0
run --model M3_static --gamma 1
# pulse robustness variant
run --model M2_pulse --gamma 0
run --model M2_pulse --gamma 1

echo "ALL FITS DONE ($(date +%H:%M:%S))"
