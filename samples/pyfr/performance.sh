#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
#
# Build PyFR sample code with "make OMP=0".
# Consider fixing CPU clock frequency, and
# disabling all kinds of "turbo boost".
#
export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}
export PERF_R=${PERF_R:-200000}
export PERF_N=${PERF_N:-40}
export FSSPMDM_NBLOCK=40

for MTX in "${HERE}"/mats/p*/{pri,hex}/m{3,6}-sp.mtx; do
  "${HERE}/pyfr_driver_asp_reg" "${MTX}" "${PERF_N}" "${PERF_R}" 1
done

export FSSPMDM_NTS=0
for MTX in "${HERE}"/mats/p*/{pri,hex}/m{0,132,460}-sp.mtx; do
  "${HERE}/pyfr_driver_asp_reg" "${MTX}" "${PERF_N}" "${PERF_R}" 0
done
