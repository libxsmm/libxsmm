#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Alexander Heinecke (Intel Corp.)
###############################################################################

echo "Please use sufficient affinities when running this benchmark"
echo "e.g.:"
echo "export OMP_NUM_THREADS=X"
echo "export KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=67
export KMP_AFFINITY=granularity=fine,compact,1,0

numactl --preferred=1 ./pyfr_gemm_rm 150 2048 125 1000
numactl --preferred=1 ./pyfr_gemm_rm 150 48000 125 1000
numactl --preferred=1 ./pyfr_gemm_rm 150 96000 125 1000

numactl --preferred=1 ./pyfr_gemm_cm 150 2048 125 1000
numactl --preferred=1 ./pyfr_gemm_cm 150 48000 125 1000
numactl --preferred=1 ./pyfr_gemm_cm 150 96000 125 1000

numactl --preferred=1 ./pyfr_gemm_rm 105 2048 75 1000
numactl --preferred=1 ./pyfr_gemm_rm 105 48000 75 1000
numactl --preferred=1 ./pyfr_gemm_rm 105 96000 75 1000

numactl --preferred=1 ./pyfr_gemm_cm 105 2048 75 1000
numactl --preferred=1 ./pyfr_gemm_cm 105 48000 75 1000
numactl --preferred=1 ./pyfr_gemm_cm 105 96000 75 1000

numactl --preferred=1 ./pyfr_driver_asp_reg ./mats/p3/hex/m6-sp.mtx 48000 10000
