#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Alexander Heinecke (Intel Corp.)
###############################################################################

echo "Please use sufficient affinities when running this benchmark"
echo "e.g.:"
echo "export OMP_NUM_THREADS=X"
echo "export KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=8
#export KMP_AFFINITY=granularity=fine,compact,1,0

./pyfr_driver_asp_reg ./mats/p3/hex/m6-sp.mtx 48000 10000
