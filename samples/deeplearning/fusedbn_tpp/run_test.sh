#!/usr/bin/env bash
set -eo pipefail

# Source the compiler

#export OMP_NUM_THREADS=1
export CHECK=1

iters=10
N=16
C=64
H=112
W=112
CB=64
pad_w_in=0
pad_h_in=0
pad_w_out=0
pad_h_out=0
stride=1
norm_type=0 # 0: full batchnorm 1: scale only (not supported)

for fuse_type in 0 1 2 3 4 5; do
  echo "--- Testing with fuse_type = $fuse_type (fuse_type 1 and 3 are fine to bail out with an unsupported op) ---"
  #LIBXSMM_VERBOSE=-1 gdb --args  ./layer_example_f32 $iters $N $C $H $W $CB $pad_w_in $pad_h_in $pad_w_out $pad_h_out $stride $norm_type $fuse_type
  LIBXSMM_VERBOSE=-1              ./layer_example_f32 $iters $N $C $H $W $CB $pad_w_in $pad_h_in $pad_w_out $pad_h_out $stride $norm_type $fuse_type
  echo "--- End of a test instance ---"
  echo ""
done
