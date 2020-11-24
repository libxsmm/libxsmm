#!/bin/bash

export KMP_AFFINITY=compact,1,0,granularity=fine              # Set KMP affinity
# export KMP_BLOCKTIME=1

export OMP_NUM_THREADS=28                                     # Set number of threads
export LD_LIBRARY_PATH=/nfs_home/nchaudh1/libxsmm/lib/        # Set LD_LIBRARY_PATH

python torch_example.py                                       # Run the pytorch example