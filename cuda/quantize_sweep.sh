#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT="${OUT:-quantize.csv}"
N="${N:-11008}"
K="${K:-4096}"
ITER="${ITER:-10}"
SLEEP="${SLEEP:-1}"
SINGLE_SPARSITY=90

make quantize_test

printf "M,N,K,single_sparsity,merged_sparsity,up_fp16_ms,up_int4_ms,up_int8_ms,down_fp16_ms,down_int4_ms,down_int8_ms,cublas_up_fp16_ms,cublas_up_deq_int4_ms,cublas_up_deq_int8_ms,cublas_down_fp16_ms,cublas_down_deq_int4_ms,cublas_down_deq_int8_ms,up_int4_max_abs,up_int4_max_rel,up_int4_mismatches,up_int8_max_abs,up_int8_max_rel,up_int8_mismatches,down_int4_max_abs,down_int4_max_rel,down_int4_mismatches,down_int8_max_abs,down_int8_max_rel,down_int8_mismatches\n" > "$OUT"

for ((M = 8; M <= 64; M *= 2)); do
    for ((MERGED_SPARSITY = 60; MERGED_SPARSITY <= 80; MERGED_SPARSITY += 5)); do
        echo "Running quantize_test: M=$M N=$N K=$K single_sparsity=$SINGLE_SPARSITY merged_sparsity=$MERGED_SPARSITY iter=$ITER"
        ./quantize_test "$OUT" "$M" "$N" "$K" "$SINGLE_SPARSITY" "$MERGED_SPARSITY" "$ITER"
        sleep "$SLEEP"
    done
done

echo "Wrote $OUT"
