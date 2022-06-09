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

HERE=$(cd "$(dirname "$0")" && pwd -P)
BOLD=$(tput bold)
NORM=$(tput sgr0)

#export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=2

EXIT=0
for M in "${HERE}"/mats/p*/*/*-sp.mtx; do
  "${HERE}/pyfr_driver_asp_reg" "${M}" 48000 10 >/dev/null
  RESULT=$?
  if [ "0" = "${RESULT}" ]; then
    echo "${M} passed!"
  else
    if [ "0" = "${EXIT}" ]; then EXIT=${RESULT}; fi
    echo "${BOLD}${M} failed!${NORM}"
  fi
done

exit ${EXIT}
