#!/bin/bash

export KMP_AFFINITY=compact,1,0,granularity=fine
# export LD_PRELOAD=~/anaconda3/lib/libiomp5.so                   # needed for KMP_AFFINITY
export OMP_NUM_THREADS=28

./equation_batchnorm 28 1 12544 64 256                  # params order = NP, CP, HW, CB, num_HW_blocks, datatype, num_iters
./equation_batchnorm 28 1 3136 64 64
./equation_batchnorm 28 4 3136 64 64
./equation_batchnorm 28 2 784 64 16
./equation_batchnorm 28 8 784 64 16
./equation_batchnorm 28 4 196 64 4

echo "\n\n\n GroupNorm starting now... \n\n\n"
./equation_groupnorm 28 1 1 12544 64 1 256              # params order = NP, CP, NB, HW, CB, G, num_HW_blocks, datatype, num_iters
./equation_groupnorm 28 1 1 3136 64 1 64
./equation_groupnorm 28 4 1 3136 64 1 64
./equation_groupnorm 28 2 1 784 64 1 16
./equation_groupnorm 28 8 1 784 64 1 16
./equation_groupnorm 28 4 1 196 64 1 4

./equation_groupnorm 28 1 1 12544 64 16 256
./equation_groupnorm 28 1 1 3136 64 16 64
./equation_groupnorm 28 4 1 3136 64 16 64
./equation_groupnorm 28 2 1 784 64 16 16
./equation_groupnorm 28 8 1 784 64 16 16
./equation_groupnorm 28 4 1 196 64 16 4

./equation_groupnorm 28 1 1 12544 64 64 256
./equation_groupnorm 28 1 1 3136 64 64 64
./equation_groupnorm 28 4 1 3136 64 64 64
./equation_groupnorm 28 2 1 784 64 64 16
./equation_groupnorm 28 8 1 784 64 64 16
./equation_groupnorm 28 4 1 196 64 64 4

