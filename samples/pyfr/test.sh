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

export OMP_NUM_THREADS=2
#export KMP_AFFINITY=granularity=fine,compact,1,0

bold=$(tput bold)
normal=$(tput sgr0)

for m in ./mats/p*/*/*-sp.mtx
do
  ./pyfr_driver_asp_reg $m 48000 10 > /dev/null
  if [ $? -eq 0 ]
  then
    echo "$m passed!"
  else
    echo "${bold}$m failed!${normal}"
  fi
done
