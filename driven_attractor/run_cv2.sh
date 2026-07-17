#!/usr/bin/env bash
# Participant-blocked 5-fold CV for the decisive round-2 models + M0 refs.
# Same folds as round 1 (seed 20260711). Runs the 5 folds in parallel.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODELS="ddm_batch_attr,ddm_batch_static,kalman_batch_attr,kalman_batch_static"
for f in 0 1 2 3 4; do
  $PY -u driven_attractor/cv_family.py --fold $f --with-m0 --models "$MODELS" \
    --budget 150 > driven_attractor/output/cv2_fold${f}.log 2>&1 &
done
wait
echo CV2_ALL_DONE
