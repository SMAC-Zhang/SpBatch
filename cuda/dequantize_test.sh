#!/bin/bash

set -euo pipefail

CSV="${1:-dequantize.csv}"
ITER="${2:-5}"
GPU=7

cd "$(dirname "$0")"

make dequantize_test
rm -f "$CSV"

for M in 8 16 32 64; do
    echo "===== GPU ${GPU}, M=${M}, iter=${ITER} ====="
    CUDA_VISIBLE_DEVICES="$GPU" ./dequantize_test "$CSV" "$M" "$ITER"
    echo ""
done

echo "Results written to ${CSV}"
