#!/usr/bin/env bash
# Round-2 batch-variant fits (deterministic quadrature; single-process DE).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for fam in ddm kalman pulse; do
  for var in batch_attr batch_static; do
    echo "=== $fam $var ($(date +%H:%M:%S)) ==="
    $PY -u driven_attractor/fit_family.py --family "$fam" --variant "$var" \
      --workers 1 --maxiter 25 --popsize 8 2>&1 | grep -E "^\[done\]|theta:|lapses:"
  done
done
echo BATCH_FITS_DONE
