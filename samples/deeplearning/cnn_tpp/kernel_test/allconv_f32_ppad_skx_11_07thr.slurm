#!/usr/bin/env bash

#SBATCH -J allconv_f32_ppad_skx_11_07thr
#SBATCH --get-user-env
#SBATCH --partition=skx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=7
export KMP_AFFINITY=granularity=fine,compact,1,28
export ITERS=1
#export LIBXSMM_VERBOSE=-1

date
srun ./run_convs.sh 11 ${ITERS} -1 f32 F 1 0 64 64 0
srun ./run_convs.sh 11 ${ITERS} -1 f32 B 1 0 64 64 0
srun ./run_convs.sh 11 ${ITERS} -1 f32 U 1 0 64 64 0
date
