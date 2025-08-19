#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

# Arguments M N K beta reps
# l_r is fixed to 16, when we FP32
# l_r is fixed to 8, when we FP64

ITERS=10

cd ${HERE} && cat <<EOM | ${EXEC} "$@"
# scatter, element, f32
./dense_packedbcrm_f32 9 729 35 0.0 ${ITERS}
./dense_packedbcrm_f32 9 729 35 1.0 ${ITERS}
# scatter, surface, f32
./dense_packedbcrm_f32 9 81 35 0.0 ${ITERS}
./dense_packedbcrm_f32 9 81 35 1.0 ${ITERS}
# gather, element, f32
./dense_packedbcrm_f32 9 35 729 0.0 ${ITERS}
./dense_packedbcrm_f32 9 35 729 1.0 ${ITERS}
# gather, surface, f32
./dense_packedbcrm_f32 9 35 81 0.0 ${ITERS}
./dense_packedbcrm_f32 9 35 81 1.0 ${ITERS}
# scatter, element, f64
./dense_packedbcrm_f64 9 729 35 0.0 ${ITERS}
./dense_packedbcrm_f64 9 729 35 1.0 ${ITERS}
# scatter, surface, f64
./dense_packedbcrm_f64 9 81 35 0.0 ${ITERS}
./dense_packedbcrm_f64 9 81 35 1.0 ${ITERS}
# gather, element, f64
./dense_packedbcrm_f64 9 35 729 0.0 ${ITERS}
./dense_packedbcrm_f64 9 35 729 1.0 ${ITERS}
# gather, surface, f64
./dense_packedbcrm_f64 9 35 81 0.0 ${ITERS}
./dense_packedbcrm_f64 9 35 81 1.0 ${ITERS}
EOM
