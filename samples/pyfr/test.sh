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
EXEC=${HERE}/../../scripts/tool_pexec.sh

export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TEST_N=${TEST_N:-48000}

for MTX in "${HERE}"/mats/p*/*/*-sp.mtx; do
  echo "${HERE}/pyfr_driver_asp_reg ${MTX} ${TEST_N} 10"
done | ${EXEC} -c 6-8 "$@"
