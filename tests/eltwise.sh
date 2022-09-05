#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
# shellcheck disable=SC2143
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
GREP=$(command -v grep)
CUT=$(command -v cut)
TR=$(command -v tr)

if [ "${HERE}" ] && [ "${GREP}" ] && [ "${CUT}" ] && [ "${TR}" ]; then
  UNAME=$(if [ "$(command -v uname)" ]; then uname; fi)
  ARCH=$(uname -m)

  # disable log files
  export PEXEC_LG=/dev/null

  if [ "x86_64" = "${ARCH}" ]; then
    if [ -e /proc/cpuinfo ]; then
      CPUFLAGS=$(${GREP} -m1 flags /proc/cpuinfo \
      | ${CUT} -d: -f2-)
    elif [ "Darwin" = "${UNAME}" ]; then
      CPUFLAGS=$(sysctl -a machdep.cpu.features \
        machdep.cpu.extfeatures \
        machdep.cpu.leaf7_features \
      | ${CUT} -d: -f2- | ${TR} -s "\n" " " \
      | ${TR} "[:upper:]." "[:lower:]_")
    fi
    if [ "$(echo "${CPUFLAGS}" | ${GREP} -w avx512f | ${GREP} -w avx512vl)" ]; then
      "${HERE}/../samples/eltwise/run_test.sh" -n 50
    elif [ "$(echo "${CPUFLAGS}" | ${GREP} -w avx2)" ]; then
      "${HERE}/../samples/eltwise/run_test_avx2.sh" -n 50
    elif [ "$(echo "${CPUFLAGS}" | ${GREP} -w sse4_2)" ]; then
      "${HERE}/../samples/eltwise/run_test_sse42.sh" -n 15
    fi
  elif [ "arm64" = "${ARCH}" ]; then
    #export LIBXSMM_TARGET="arm_v82"
    "${HERE}/../samples/eltwise/run_test_aarch64.sh" -n 30
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
