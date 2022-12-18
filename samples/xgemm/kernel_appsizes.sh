#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

REPS=10000
TEST_EDGE="4_9_4 4_9_9 10_9_10 10_9_9 20_9_20 20_9_9 35_9_35 35_9_9 56_9_56 56_9_9"
TEST_EDGE_PAD="4_9_4 4_9_9 12_9_12 12_9_9 20_9_20 20_9_9 36_9_36 36_9_9 56_9_56 56_9_9"
TEST_SU2="1008_5_75 75_5_756 147_5_75 48_5_35 184_5_35 35_5_138 75_5_147 35_5_48 48_5_75 108_5_75 75_5_48 16_5_15 15_5_16 49_5_25 25_5_49"
TEST_SU2_2F="1008_10_75 75_10_756 147_10_75 48_10_35 184_10_35 35_10_138 75_10_147 35_10_48 48_10_75 108_10_75 75_10_48 16_10_15 15_10_16 49_10_25 25_10_49"
TEST_SU2_3F="1008_15_75 75_15_756 147_15_75 48_15_35 184_15_35 35_15_138 75_15_147 35_15_48 48_15_75 108_15_75 75_15_48 16_15_15 15_15_16 49_15_25 25_15_49"
TEST=${TEST_EDGE}$

# select precision
PREC=F64
if [ $# -eq 1 ]
then
  PREC=$1
fi

for t in ${TEST}
do
  M=`echo ${t} | awk -F"_" '{print $1}'`
  N=`echo ${t} | awk -F"_" '{print $2}'`
  K=`echo ${t} | awk -F"_" '{print $3}'`
  lda=$M
  ldb=$K
  ldc=$M
  ./gemm_kernel $M $N $K $lda $ldb $ldc 1 1 0 0 0 0 nopf ${PREC} nobr 1 0 ${REPS}
done
