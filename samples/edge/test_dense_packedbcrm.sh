#!/usr/bin/env bash
set -eo pipefail

# Arguments M N K beta reps
# l_r is fixed to 16, when we FP32
# l_r is fixed to 8, when we FP64

ITERS=10

echo "scatter, element, f32"
./dense_packedbcrm_f32 9 729 35 0.0 ${ITERS}
./dense_packedbcrm_f32 9 729 35 1.0 ${ITERS}

echo "scatter, surface, f32"
./dense_packedbcrm_f32 9 81 35 0.0 ${ITERS}
./dense_packedbcrm_f32 9 81 35 1.0 ${ITERS}

echo "gather, element, f32"
./dense_packedbcrm_f32 9 35 729 0.0 ${ITERS}
./dense_packedbcrm_f32 9 35 729 1.0 ${ITERS}

echo "gather, surface, f32"
./dense_packedbcrm_f32 9 35 81 0.0 ${ITERS}
./dense_packedbcrm_f32 9 35 81 1.0 ${ITERS}

echo "scatter, element, f64"
./dense_packedbcrm_f64 9 729 35 0.0 ${ITERS}
./dense_packedbcrm_f64 9 729 35 1.0 ${ITERS}

echo "scatter, surface, f64"
./dense_packedbcrm_f64 9 81 35 0.0 ${ITERS}
./dense_packedbcrm_f64 9 81 35 1.0 ${ITERS}

echo "gather, element, f64"
./dense_packedbcrm_f64 9 35 729 0.0 ${ITERS}
./dense_packedbcrm_f64 9 35 729 1.0 ${ITERS}

echo "gather, surface, f64"
./dense_packedbcrm_f64 9 35 81 0.0 ${ITERS}
./dense_packedbcrm_f64 9 35 81 1.0 ${ITERS}
