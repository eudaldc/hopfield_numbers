#!/usr/bin/env bash
set -euo pipefail

PRESETS=(default weber)
MODES=(dynamic static)

for preset in "${PRESETS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "============================================"
        echo "  Training preset=$preset  mode=$mode"
        echo "============================================"
        python scripts/train_unified.py --preset "$preset" --attractor-mode "$mode"
    done
done

echo "All presets done."
