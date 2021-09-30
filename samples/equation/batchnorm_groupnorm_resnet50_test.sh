#!/bin/bash

export KMP_AFFINITY=compact,1,0,granularity=fine
export LD_PRELOAD=~/anaconda3/lib/libiomp5.so                   # needed for KMP_AFFINITY
export OMP_NUM_THREADS=28

# ./equation_batchnorm 28 1 12544 64 256 0 10000                 # params order = NP, CP, HW, CB, num_HW_blocks, datatype, num_iters
# ./equation_batchnorm 28 1 3136 64 64 0 10000
# ./equation_batchnorm 28 4 3136 64 64 0 10000
# ./equation_batchnorm 28 2 784 64 16 0 10000
# ./equation_batchnorm 28 8 784 64 16 0 10000
# ./equation_batchnorm 28 4 196 64 4 0 10000
# ./equation_batchnorm 28 16 196 64 4 0 10000
# ./equation_batchnorm 28 8 49 64 1 0 10000
# ./equation_batchnorm 28 32 49 64 1 0 10000

echo "\n\n\n GroupNorm starting now... \n\n\n"
# (Group_size=CP*CB)
# ./equation_groupnorm 28 1 12544 64 1 256 0 1000               # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
# ./equation_groupnorm 28 1 3136 64 1 64 0 1000
# ./equation_groupnorm 28 4 3136 64 1 64 0 1000
# ./equation_groupnorm 28 2 784 64 1 16 0 1000
# ./equation_groupnorm 28 8 784 64 1 16 0 1000
# ./equation_groupnorm 28 4 196 64 1 4 0 1000
# ./equation_groupnorm 28 16 196 64 1 4 0 1000
# ./equation_groupnorm 28 8 49 64 1 1 0 1000
# ./equation_groupnorm 28 32 49 64 1 1 0 1000

# (Group_size=16)
./equation_groupnorm 28 1 12544 64 4 256 0 1000             # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
./equation_groupnorm 28 1 3136 64 4 64 0 1000
./equation_groupnorm 28 4 3136 64 16 64 0 1000
./equation_groupnorm 28 2 784 64 8 16 0 1000
./equation_groupnorm 28 8 784 64 32 16 0 1000
./equation_groupnorm 28 4 196 64 16 4 0 1000
./equation_groupnorm 28 16 196 64 64 4 0 1000
./equation_groupnorm 28 8 49 64 32 1 0 1000
./equation_groupnorm 28 32 49 64 128 1 0 1000

# (Group_size=4)
# ./equation_groupnorm 28 1 12544 64 16 256 0 1000             # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
# ./equation_groupnorm 28 1 3136 64 16 64 0 1000
# ./equation_groupnorm 28 4 3136 64 64 64 0 1000
# ./equation_groupnorm 28 2 784 64 32 16 0 1000
# ./equation_groupnorm 28 8 784 64 128 16 0 1000
# ./equation_groupnorm 28 4 196 64 64 4 0 1000
# ./equation_groupnorm 28 16 196 64 256 4 0 1000
# ./equation_groupnorm 28 8 49 64 128 1 0 1000
# ./equation_groupnorm 28 32 49 64 512 1 0 1000

# (Group_size=1)
# ./equation_groupnorm 28 1 12544 64 64 256 0 1000             # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
# ./equation_groupnorm 28 1 3136 64 64 64 0 1000
# ./equation_groupnorm 28 4 3136 64 256 64 0 1000
# ./equation_groupnorm 28 2 784 64 128 16 0 1000
# ./equation_groupnorm 28 8 784 64 512 16 0 1000
# ./equation_groupnorm 28 4 196 64 256 4 0 1000
# ./equation_groupnorm 28 16 196 64 1024 4 0 1000
# ./equation_groupnorm 28 8 49 64 512 1 0 1000
# ./equation_groupnorm 28 32 49 64 2048 1 0 1000
