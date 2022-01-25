#!/usr/bin/env bash

export KMP_AFFINITY=compact,1,0,granularity=fine
# export LD_PRELOAD=~/anaconda3/lib/libiomp5.so                   # needed for KMP_AFFINITY
export LD_PRELOAD=/swtools/intel/compilers_and_libraries/linux/lib/intel64/libiomp5.so
export OMP_NUM_THREADS=28

./equation_batchnorm 28 2 12544 32 256 1 1000                 # params order = NP, CP, HW, CB, num_HW_blocks, datatype, num_iters
./equation_batchnorm 28 2 3136 32 64 0 1000
./equation_batchnorm 28 8 3136 32 64 0 1000
./equation_batchnorm 28 4 784 32 16 0 1000
./equation_batchnorm 28 16 784 32 16 0 1000
./equation_batchnorm 28 8 196 32 4 0 1000
./equation_batchnorm 28 32 196 32 4 0 1000
./equation_batchnorm 28 16 49 32 1 0 1000
./equation_batchnorm 28 64 49 32 1 0 1000

echo "\n\n\n GroupNorm starting now... \n\n\n"
# (Group_size=CP*CB)
./equation_groupnorm 28 2 12544 32 1 256 1 1000               # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
./equation_groupnorm 28 2 3136 32 1 64 0 1000
./equation_groupnorm 28 8 3136 32 1 64 0 1000
./equation_groupnorm 28 4 784 32 1 16 0 1000
./equation_groupnorm 28 16 784 32 1 16 0 1000
./equation_groupnorm 28 8 196 32 1 4 0 1000
./equation_groupnorm 28 32 196 32 1 4 0 1000
./equation_groupnorm 28 16 49 32 1 1 0 1000
./equation_groupnorm 28 64 49 32 1 1 0 1000

# (Group_size=16)
./equation_groupnorm 28 2 12544 32 4 256 1 1000             # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
./equation_groupnorm 28 2 3136 32 4 64 0 1000
./equation_groupnorm 28 8 3136 32 16 64 0 1000
./equation_groupnorm 28 4 784 32 8 16 0 1000
./equation_groupnorm 28 16 784 32 32 16 0 1000
./equation_groupnorm 28 8 196 32 16 4 0 1000
./equation_groupnorm 28 32 196 32 64 4 0 1000
./equation_groupnorm 28 16 49 32 32 1 0 1000
./equation_groupnorm 28 64 49 32 128 1 0 1000

# (Group_size=4)
./equation_groupnorm 28 2 12544 32 16 256 1 1000             # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
./equation_groupnorm 28 2 3136 32 16 64 0 1000
./equation_groupnorm 28 8 3136 32 64 64 0 1000
./equation_groupnorm 28 4 784 32 32 16 0 1000
./equation_groupnorm 28 16 784 32 128 16 0 1000
./equation_groupnorm 28 8 196 32 64 4 0 1000
./equation_groupnorm 28 32 196 32 256 4 0 1000
./equation_groupnorm 28 16 49 32 128 1 0 1000
./equation_groupnorm 28 64 49 32 512 1 0 1000

# (Group_size=1)
./equation_groupnorm 28 2 12544 32 64 256 1 1000             # params order = NP, CP, HW, CB, G, num_HW_blocks, datatype, num_iters
./equation_groupnorm 28 2 3136 32 64 64 0 1000
./equation_groupnorm 28 8 3136 32 256 64 0 1000
./equation_groupnorm 28 4 784 32 128 16 0 1000
./equation_groupnorm 28 16 784 32 512 16 0 1000
./equation_groupnorm 28 8 196 32 256 4 0 1000
./equation_groupnorm 28 32 196 32 1024 4 0 1000
./equation_groupnorm 28 16 49 32 512 1 0 1000
./equation_groupnorm 28 64 49 32 2048 1 0 1000