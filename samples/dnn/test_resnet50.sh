#!/bin/bash

set -x

export OMP_NUM_THREADS=70
export KMP_AFFINITY=granularity=fine,compact,1,2
ITERS=1000

srun ./run_resnet50.sh 70 ${ITERS} 1 f32 F L 1 | grep PERFDUMP
srun ./run_resnet50.sh 70 ${ITERS} 1 f32 B L 1 | grep PERFDUMP
srun ./run_resnet50.sh 70 ${ITERS} 1 f32 U L 1 | grep PERFDUMP
srun ./run_resnet50.sh 70 ${ITERS} 1 f32 F L 0 | grep PERFDUMP
srun ./run_resnet50.sh 70 ${ITERS} 1 f32 B L 0 | grep PERFDUMP
srun ./run_resnet50.sh 70 ${ITERS} 1 f32 U L 0 | grep PERFDUMP
srun ./run_resnet50_mock.sh 70 ${ITERS} 1 f32 F L 1 | grep PERFDUMP
srun ./run_resnet50_mock.sh 70 ${ITERS} 1 f32 B L 1 | grep PERFDUMP
srun ./run_resnet50_mock.sh 70 ${ITERS} 1 f32 U L 1 | grep PERFDUMP
srun ./run_resnet50_mock.sh 70 ${ITERS} 1 f32 F L 0 | grep PERFDUMP
srun ./run_resnet50_mock.sh 70 ${ITERS} 1 f32 B L 0 | grep PERFDUMP
srun ./run_resnet50_mock.sh 70 ${ITERS} 1 f32 U L 0 | grep PERFDUMP

set +x
