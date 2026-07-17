#!/usr/bin/env bash
# DDM online variants, warm-started from the matched batch fits at the nesting
# limit (stiff tether => online == batch), so any improvement is real.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for var in online_attr online_static; do
  echo "=== ddm $var ($(date +%H:%M:%S)) ==="
  $PY -u driven_attractor/fit_family.py --family ddm --variant "$var" \
    --warm output/fits2/ddm_batch_attr.json --workers 1 \
    --m-fit 1024 --m-final 8192 2>&1 | grep -E "^\[done\]|theta:|lapses:"
done
echo ONLINE_DDM_DONE
